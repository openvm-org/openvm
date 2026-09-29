// Lean compiler output
// Module: Aesop.Script.SpecificTactics
// Imports: public import Init public meta import Init public import Lean.Meta.Tactic.Cases public import Lean.Meta.Tactic.Simp.Types public import Aesop.Util.Tactic.Ext public import Aesop.Script.CtorNames public import Aesop.Script.ScriptM import Batteries.Lean.Meta.Inaccessible import Aesop.Util.Tactic import Aesop.Util.Tactic.Unfold import Aesop.Util.Unfold import Batteries.Lean.Meta.Basic import Lean.Elab.Tactic.RCases import Lean.Elab.Tactic.Simp import Lean.Meta.Tactic.Split
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
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_FVarId_getUserName___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_withAllTransparencySyntax(uint8_t, lean_object*);
lean_object* lp_aesop_Aesop_Script_Tactic_unstructured(lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Meta_getSimpTheorems___redArg(lean_object*);
lean_object* l_Lean_Meta_Simp_getSimprocs___redArg(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_Meta_Origin_key(lean_object*);
uint64_t lean_uint64_mix_hash(uint64_t, uint64_t);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_mkSimpOnly___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_FVarId_getDecl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_userName(lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_MVarId_assertHypotheses(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_MVarId_tryClear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_splitTarget_x3f(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* l_Lean_PrettyPrinter_delab(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_LocalDecl_isImplementationDetail(lean_object*);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
lean_object* lp_aesop_Aesop_unfoldManyAt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_FVarId_getUserName___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_splitLocalDecl_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_introsUnfolding___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_tactic_hygienic;
lean_object* l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Name_isPrefixOf(lean_object*, lean_object*);
extern lean_object* l_Lean_diagnostics;
extern lean_object* l_Lean_maxRecDepth;
lean_object* l_Lean_Kernel_enableDiag(lean_object*, uint8_t);
uint8_t l_Lean_Kernel_isDiagnosticsEnabled(lean_object*);
lean_object* lp_aesop_Aesop_withScriptStep___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Lean_MVarId_renameInaccessibleFVars___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_CtorNames_toAltVarNames(lean_object*);
lean_object* lp_aesop_Aesop_unfoldManyTarget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_delab___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_clear___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_zip___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ctorNamesToInductionAlts(lean_object*);
lean_object* l_Lean_MVarId_apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_CtorNames_mkFreshArgNames(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_replaceFVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* lp_aesop_Aesop_ctorNamesToRCasesPats(lean_object*);
lean_object* lp_aesop_Aesop_Script_Tactic_structured(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_CtorNames_toRCasesPat(lean_object*);
lean_object* lp_aesop_Aesop_withOptScriptStep___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_Syntax_mkNumLit(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_cases___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_intros___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_toExpr(lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_type(lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_tryClearMany_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_straightLineExt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_Script_Tactic_skip___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_Tactic_skip___closed__0;
static const lean_string_object lp_aesop_Aesop_Script_Tactic_skip___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_aesop_Aesop_Script_Tactic_skip___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Script_Tactic_skip___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_aesop_Aesop_Script_Tactic_skip___closed__2 = (const lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value;
static const lean_string_object lp_aesop_Aesop_Script_Tactic_skip___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_aesop_Aesop_Script_Tactic_skip___closed__3 = (const lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value;
static const lean_string_object lp_aesop_Aesop_Script_Tactic_skip___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "skip"};
static const lean_object* lp_aesop_Aesop_Script_Tactic_skip___closed__4 = (const lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_Script_Tactic_skip___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_Tactic_skip___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__5_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_Tactic_skip___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__5_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_Tactic_skip___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__5_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__4_value),LEAN_SCALAR_PTR_LITERAL(244, 42, 145, 170, 145, 147, 228, 105)}};
static const lean_object* lp_aesop_Aesop_Script_Tactic_skip___closed__5 = (const lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__5_value;
static lean_once_cell_t lp_aesop_Aesop_Script_Tactic_skip___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_Tactic_skip___closed__6;
static lean_once_cell_t lp_aesop_Aesop_Script_Tactic_skip___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_Tactic_skip___closed__7;
static lean_once_cell_t lp_aesop_Aesop_Script_Tactic_skip___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_Tactic_skip___closed__8;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_Tactic_skip;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_applyStx___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_applyStx___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_applyStx___redArg___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_applyStx___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_applyStx___redArg___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_applyStx___redArg___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_applyStx___redArg___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_applyStx___redArg___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_applyStx___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_applyStx___redArg___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_applyStx___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(202, 125, 237, 78, 179, 140, 218, 80)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_applyStx___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_applyStx___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_applyStx___redArg(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_applyStx___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_applyStx(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_applyStx___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_apply(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_apply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_exactFVar___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "exact"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_exactFVar___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_exactFVar___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_exactFVar___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_exactFVar___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_exactFVar___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_exactFVar___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_exactFVar___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_exactFVar___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_exactFVar___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_exactFVar___closed__0_value),LEAN_SCALAR_PTR_LITERAL(108, 106, 111, 83, 219, 207, 32, 208)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_exactFVar___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_exactFVar___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_exactFVar(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_exactFVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "replace"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_replace___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__0_value),LEAN_SCALAR_PTR_LITERAL(209, 191, 89, 106, 186, 34, 132, 63)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_replace___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_replace___closed__2 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__2_value;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "letDecl"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_replace___closed__3 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__4_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__4_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__4_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__3_value),LEAN_SCALAR_PTR_LITERAL(61, 47, 121, 206, 37, 68, 134, 111)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_replace___closed__4 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__4_value;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "letIdDecl"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_replace___closed__5 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__6_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__6_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__6_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__5_value),LEAN_SCALAR_PTR_LITERAL(82, 96, 243, 36, 251, 209, 136, 237)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_replace___closed__6 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__6_value;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "letId"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_replace___closed__7 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__7_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__8_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__8_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__8_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__7_value),LEAN_SCALAR_PTR_LITERAL(67, 92, 92, 51, 38, 250, 60, 190)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_replace___closed__8 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__8_value;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_replace___closed__9 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__9_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__9_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10_value;
static lean_once_cell_t lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "typeSpec"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_replace___closed__12 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__12_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__13_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__13_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__13_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__12_value),LEAN_SCALAR_PTR_LITERAL(77, 126, 241, 117, 174, 189, 108, 62)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_replace___closed__13 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__13_value;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_replace___closed__14 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__14_value;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_replace___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":="};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_replace___closed__15 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__15_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_replace(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_replace___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "tacticHave__"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(57, 244, 114, 225, 1, 158, 79, 25)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "have"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__2 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__2_value;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "letConfig"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__3 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__4_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__4_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__4_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(5, 186, 227, 151, 19, 40, 136, 241)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__4 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__4_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_clear_spec__0___redArg(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_clear_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_clear_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_clear_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_clear___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "clear"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_clear___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_clear___lam__0___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_clear___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_clear___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_clear___lam__0___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_clear___lam__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_clear___lam__0___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_clear___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_clear___lam__0___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_clear___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(190, 197, 160, 206, 26, 199, 189, 206)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_clear___lam__0___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_clear___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_clear___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_clear___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_clear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_clear___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_clear_spec__0(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_clear_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cases"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(197, 49, 98, 208, 150, 151, 163, 74)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__2;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "elimTarget"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__3 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__4_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__4_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__4_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(136, 63, 46, 91, 99, 29, 205, 171)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__4 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "rcases"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(140, 76, 101, 33, 30, 11, 121, 59)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "with"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__2 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__2_value;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "rcasesPatLo"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__3 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__4_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__4_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__4_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(133, 222, 245, 138, 122, 92, 170, 214)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__4 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__4_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_cases(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_cases___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "obtain"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(11, 177, 143, 165, 56, 37, 104, 113)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "rcasesPatMed"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__2 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__3_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__3_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__3_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(253, 13, 65, 195, 228, 27, 47, 149)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__3 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__3_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_obtain(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_obtain___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_casesOrObtain(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_casesOrObtain___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_renameInaccessibleFVars_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "binderIdent"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_renameInaccessibleFVars_spec__0___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_renameInaccessibleFVars_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_renameInaccessibleFVars_spec__0___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_renameInaccessibleFVars_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_renameInaccessibleFVars_spec__0___redArg___closed__1_value_aux_0),((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_renameInaccessibleFVars_spec__0___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(37, 194, 68, 106, 254, 181, 31, 191)}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_renameInaccessibleFVars_spec__0___redArg___closed__1 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_renameInaccessibleFVars_spec__0___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_renameInaccessibleFVars_spec__0___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_renameInaccessibleFVars_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "renameI"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___lam__0___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___lam__0___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___lam__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___lam__0___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___lam__0___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(20, 41, 101, 89, 107, 117, 242, 244)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___lam__0___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___lam__0___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "rename_i"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___lam__0___closed__2 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___lam__0___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_renameInaccessibleFVars_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_renameInaccessibleFVars_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_unfold_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_unfold_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unfold"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(40, 165, 59, 41, 65, 58, 102, 253)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Aesop"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__2_value;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "tacticAesop_unfold_"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__3 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__4_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(233, 125, 94, 48, 81, 53, 160, 125)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__4 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__4_value;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "aesop_unfold"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__5 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__5_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfold(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfold___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "location"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(124, 82, 43, 228, 241, 102, 135, 24)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "at"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__2 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__2_value;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "locationHyp"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__3 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__4_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__4_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__4_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(229, 146, 67, 234, 45, 36, 143, 176)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__4 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__4_value;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "tacticAesop_unfold_At_"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__5 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__6_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(29, 223, 0, 176, 223, 131, 246, 4)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__6 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__6_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfoldAt(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "rintroPat"};
static const lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__0_value;
static const lean_string_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "one"};
static const lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__1_value;
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__2_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__2_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__2_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__2_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(120, 93, 179, 129, 121, 199, 215, 253)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__2_value_aux_3),((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(40, 214, 202, 122, 59, 249, 35, 61)}};
static const lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__2_value;
static const lean_string_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "rcasesPat"};
static const lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__3_value;
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__4_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__4_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__4_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__4_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(162, 181, 165, 225, 136, 177, 169, 19)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__4_value_aux_3),((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(186, 152, 172, 228, 11, 240, 156, 168)}};
static const lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__4_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_extN_spec__0___redArg(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_extN_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_extN_spec__0(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_extN_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticBuilder_extN_spec__1___boxed__const__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + sizeof(size_t)*1, .m_other = 0, .m_tag = 0}, .m_objs = {(lean_object*)(size_t)(0ULL)}};
LEAN_EXPORT const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticBuilder_extN_spec__1___boxed__const__1 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticBuilder_extN_spec__1___boxed__const__1_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticBuilder_extN_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticBuilder_extN_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_extN___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_extN___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_extN___closed__0_value;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_extN___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Ext"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_extN___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_extN___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_extN___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "ext"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_extN___closed__2 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_extN___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_extN___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_extN___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_extN___closed__3_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_extN___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_extN___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_extN___closed__3_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(161, 230, 229, 85, 182, 144, 182, 176)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_extN___closed__3_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_extN___closed__3_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_extN___closed__1_value),LEAN_SCALAR_PTR_LITERAL(49, 70, 231, 255, 233, 213, 189, 46)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_extN___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_extN___closed__3_value_aux_3),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_extN___closed__2_value),LEAN_SCALAR_PTR_LITERAL(243, 8, 190, 90, 148, 21, 56, 73)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_extN___closed__3 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_extN___closed__3_value;
static const lean_array_object lp_aesop_Aesop_Script_TacticBuilder_extN___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_extN___closed__4 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_extN___closed__4_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_extN(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_extN___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__0_value;
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__1_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(50, 13, 241, 145, 67, 153, 105, 177)}};
static const lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__1_value;
static const lean_string_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__2_value;
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__3_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__3_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__3_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__3_value;
static const lean_string_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "locationWildcard"};
static const lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__4_value;
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__5_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__5_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__5_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(134, 218, 71, 35, 220, 118, 132, 17)}};
static const lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__5 = (const lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__5_value;
static const lean_string_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "*"};
static const lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__6 = (const lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__6_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__3(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "configItem"};
static const lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__0_value;
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__1_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__0_value),LEAN_SCALAR_PTR_LITERAL(205, 9, 236, 192, 59, 252, 178, 140)}};
static const lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__1_value;
static const lean_string_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "valConfigItem"};
static const lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__2_value;
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__3_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__3_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__3_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__2_value),LEAN_SCALAR_PTR_LITERAL(135, 67, 19, 169, 17, 95, 109, 188)}};
static const lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__3_value;
static const lean_string_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__4_value;
static const lean_string_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "config"};
static const lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__5 = (const lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__5_value;
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__5_value),LEAN_SCALAR_PTR_LITERAL(207, 146, 87, 28, 198, 178, 209, 199)}};
static const lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__6 = (const lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__6_value;
static lean_once_cell_t lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__7;
static const lean_string_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__8 = (const lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__8_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "simpAll"};
static const lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__0_value;
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__1_value_aux_2),((lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__0_value),LEAN_SCALAR_PTR_LITERAL(5, 49, 55, 92, 153, 191, 153, 249)}};
static const lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__1_value;
static const lean_string_object lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "simp_all"};
static const lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__5___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__9(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg(lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnlyStx_spec__0___redArg(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnlyStx_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnlyStx_spec__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnlyStx_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnlyStx(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnlyStx___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnly(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnly___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0_spec__0___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1_spec__2_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1_spec__2___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0_spec__0(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1_spec__2(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0_spec__1_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0_spec__2___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2_spec__5___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__0;
static lean_once_cell_t lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__1;
static lean_once_cell_t lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__2;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 40, .m_capacity = 40, .m_length = 39, .m_data = "simp tactic builder: unexpected syntax:"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__3 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__3_value;
static lean_once_cell_t lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__5 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__5_value;
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__6 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__6_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0_spec__2(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0_spec__1_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_intros___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "intro"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_intros___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_intros___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_intros___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_intros___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_intros___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_intros___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_intros___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_intros___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_intros___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_intros___closed__0_value),LEAN_SCALAR_PTR_LITERAL(41, 145, 9, 18, 75, 146, 159, 78)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_intros___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_intros___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_intros(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_intros___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "split"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(104, 58, 38, 157, 113, 69, 9, 24)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_splitTarget(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_splitTarget___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_splitAt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_splitAt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Script_TacticBuilder_substFVars___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "subst"};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_substFVars___closed__0 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_substFVars___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_substFVars___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_substFVars___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_substFVars___closed__1_value_aux_0),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__2_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_substFVars___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_substFVars___closed__1_value_aux_1),((lean_object*)&lp_aesop_Aesop_Script_Tactic_skip___closed__3_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_Script_TacticBuilder_substFVars___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_substFVars___closed__1_value_aux_2),((lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_substFVars___closed__0_value),LEAN_SCALAR_PTR_LITERAL(72, 253, 96, 40, 175, 171, 7, 145)}};
static const lean_object* lp_aesop_Aesop_Script_TacticBuilder_substFVars___closed__1 = (const lean_object*)&lp_aesop_Aesop_Script_TacticBuilder_substFVars___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_substFVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_substFVars___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_substFVars_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_substFVars_x27___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_substFVars_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_substFVars_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_assertHypothesisS_tacticBuilder___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_assertHypothesisS_tacticBuilder___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_assertHypothesisS_tacticBuilder(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_assertHypothesisS_tacticBuilder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_assertHypothesisS___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_assertHypothesisS___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_assertHypothesisS___lam__1(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_assertHypothesisS___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_assertHypothesisS___lam__2___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_assertHypothesisS___lam__3(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_assertHypothesisS___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_assertHypothesisS___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_assertHypothesisS___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_assertHypothesisS___closed__0 = (const lean_object*)&lp_aesop_Aesop_assertHypothesisS___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_assertHypothesisS___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_assertHypothesisS___lam__2___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_assertHypothesisS___closed__1 = (const lean_object*)&lp_aesop_Aesop_assertHypothesisS___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_assertHypothesisS(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_assertHypothesisS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_applyS_tacticBuilder___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_applyS_tacticBuilder___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_applyS_tacticBuilder(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_applyS_tacticBuilder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyS___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyS___lam__0___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_applyS___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyS___lam__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyS___lam__2(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyS___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_applyS___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_applyS___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_applyS___closed__0 = (const lean_object*)&lp_aesop_Aesop_applyS___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_applyS___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_applyS___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_applyS___closed__1 = (const lean_object*)&lp_aesop_Aesop_applyS___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_applyS___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 8, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(0, 1, 0, 1, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_applyS___closed__2 = (const lean_object*)&lp_aesop_Aesop_applyS___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyS(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_replaceFVarS_tacticBuilder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_replaceFVarS_tacticBuilder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_replaceFVarS___lam__0(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_replaceFVarS___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_replaceFVarS___lam__1___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_replaceFVarS___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_replaceFVarS___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_replaceFVarS___closed__0 = (const lean_object*)&lp_aesop_Aesop_replaceFVarS___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_replaceFVarS___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_replaceFVarS___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_replaceFVarS___closed__1 = (const lean_object*)&lp_aesop_Aesop_replaceFVarS___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_replaceFVarS(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_replaceFVarS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_clearS_tacticBuilder___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_clearS_tacticBuilder___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_clearS_tacticBuilder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_clearS_tacticBuilder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_clearS___lam__0(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_clearS___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_clearS___lam__1___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_clearS___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_clearS___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_clearS___closed__0 = (const lean_object*)&lp_aesop_Aesop_clearS___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_clearS___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_clearS___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_clearS___closed__1 = (const lean_object*)&lp_aesop_Aesop_clearS___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_clearS(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_clearS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_tryClearS_tacticBuilder___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_tryClearS_tacticBuilder___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_tryClearS_tacticBuilder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_tryClearS_tacticBuilder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryClearS___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryClearS___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryClearS___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryClearS___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryClearS(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryClearS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_tryClearManyS_tacticBuilder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_tryClearManyS_tacticBuilder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_tryClearManyS___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryClearManyS___lam__1___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_tryClearManyS___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_tryClearManyS___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_tryClearManyS___closed__0 = (const lean_object*)&lp_aesop_Aesop_tryClearManyS___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryClearManyS(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryClearManyS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_tryCasesS_getUnusedCtorNames_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_tryCasesS_getUnusedCtorNames_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_tryCasesS_getUnusedCtorNames(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_tryCasesS_getUnusedCtorNames___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_observing_x3f___at___00Aesop_tryCasesS_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_observing_x3f___at___00Aesop_tryCasesS_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_observing_x3f___at___00Aesop_tryCasesS_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_observing_x3f___at___00Aesop_tryCasesS_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_tryCasesS_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_tryCasesS_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryCasesS___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryCasesS___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryCasesS___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_tryCasesS_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_tryCasesS_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_tryCasesS___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_tryCasesS___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_tryCasesS___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_tryCasesS___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryCasesS___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryCasesS___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryCasesS(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryCasesS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_renameInaccessibleFVarsS_tacticBuilder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_renameInaccessibleFVarsS_tacticBuilder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_renameInaccessibleFVarsS___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_renameInaccessibleFVarsS_tacticBuilder___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_renameInaccessibleFVarsS___closed__0 = (const lean_object*)&lp_aesop_Aesop_renameInaccessibleFVarsS___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_renameInaccessibleFVarsS(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_renameInaccessibleFVarsS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00Aesop_unfoldManyTargetS_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00Aesop_unfoldManyTargetS_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00Aesop_unfoldManyTargetS_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00Aesop_unfoldManyTargetS_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyTargetS(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyTargetS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyAtS___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyAtS___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyAtS(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyAtS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyStarS_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyStarS_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyStarS_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyStarS_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyStarS_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyStarS_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0_spec__3_spec__4___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0_spec__2(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__1_spec__5___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__1_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyStarS___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyStarS___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyStarS(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyStarS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__1_spec__5(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0_spec__3_spec__4(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_introsS_tacticBuilder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_introsS_tacticBuilder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Lean_Options_set___at___00Lean_Option_set___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__0_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_aesop_Lean_Options_set___at___00Lean_Option_set___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__0_spec__1___closed__0 = (const lean_object*)&lp_aesop_Lean_Options_set___at___00Lean_Option_set___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__0_spec__1___closed__0_value;
static const lean_ctor_object lp_aesop_Lean_Options_set___at___00Lean_Option_set___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__0_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Lean_Options_set___at___00Lean_Option_set___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__0_spec__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_aesop_Lean_Options_set___at___00Lean_Option_set___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__0_spec__1___closed__1 = (const lean_object*)&lp_aesop_Lean_Options_set___at___00Lean_Option_set___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__0_spec__1___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Lean_Options_set___at___00Lean_Option_set___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__0_spec__1(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Lean_Options_set___at___00Lean_Option_set___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_set___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__0(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_set___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__1___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg___closed__0;
static lean_once_cell_t lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg___closed__1;
static lean_once_cell_t lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_introsS___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_introsS___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_introsS___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_introsS_tacticBuilder___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_introsS___closed__0 = (const lean_object*)&lp_aesop_Aesop_introsS___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_introsS(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_introsS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_introsUnfoldingS_tacticBuilder(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_introsUnfoldingS_tacticBuilder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_introsUnfoldingS___lam__2(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_introsUnfoldingS___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_introsUnfoldingS(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_introsUnfoldingS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_straightLineExtS_tacticBuilder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_straightLineExtS_tacticBuilder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_straightLineExtS_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_straightLineExtS_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_straightLineExtS___lam__0(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_straightLineExtS___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_straightLineExtS___lam__1___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_straightLineExtS___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_straightLineExtS___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_straightLineExtS___closed__0 = (const lean_object*)&lp_aesop_Aesop_straightLineExtS___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_straightLineExtS___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_straightLineExtS___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_straightLineExtS___closed__1 = (const lean_object*)&lp_aesop_Aesop_straightLineExtS___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_straightLineExtS___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Script_TacticBuilder_extN___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_straightLineExtS___closed__2 = (const lean_object*)&lp_aesop_Aesop_straightLineExtS___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_straightLineExtS(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_straightLineExtS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1_spec__3___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_tryExactFVarS___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_tryExactFVarS___closed__0 = (const lean_object*)&lp_aesop_Aesop_tryExactFVarS___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryExactFVarS(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryExactFVarS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1_spec__3(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_renameInaccessibleFVarsS_x27_spec__0(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_renameInaccessibleFVarsS_x27_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_renameInaccessibleFVarsS_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_renameInaccessibleFVarsS_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_splitTargetS_x3f_tacticBuilder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_splitTargetS_x3f_tacticBuilder___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_splitTargetS_x3f_tacticBuilder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_splitTargetS_x3f_tacticBuilder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_splitTargetS_x3f___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_splitTargetS_x3f___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_splitTargetS_x3f___lam__2(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_splitTargetS_x3f___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_splitTargetS_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_splitTargetS_x3f___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_splitTargetS_x3f___closed__0 = (const lean_object*)&lp_aesop_Aesop_splitTargetS_x3f___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_splitTargetS_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_splitTargetS_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_splitFirstHypothesisS_x3f_tacticBuilder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_splitFirstHypothesisS_x3f_tacticBuilder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_splitFirstHypothesisS_x3f___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_splitFirstHypothesisS_x3f___lam__0___boxed(lean_object*);
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0_spec__2_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0_spec__2_spec__3___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0_spec__2_spec__3___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0_spec__2_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__1_spec__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__1_spec__4___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__1_spec__4___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__1_spec__4(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop_Aesop_splitFirstHypothesisS_x3f___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_splitFirstHypothesisS_x3f___lam__1___closed__0 = (const lean_object*)&lp_aesop_Aesop_splitFirstHypothesisS_x3f___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_splitFirstHypothesisS_x3f___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_splitFirstHypothesisS_x3f___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_splitFirstHypothesisS_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_splitFirstHypothesisS_x3f___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_splitFirstHypothesisS_x3f___closed__0 = (const lean_object*)&lp_aesop_Aesop_splitFirstHypothesisS_x3f___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_splitFirstHypothesisS_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_splitFirstHypothesisS_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_aesop_Aesop_Script_Tactic_skip___closed__0(void){
_start:
{
uint8_t v___x_1_; lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_1_ = 0;
v___x_2_ = lean_box(0);
v___x_3_ = l_Lean_SourceInfo_fromRef(v___x_2_, v___x_1_);
return v___x_3_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_Tactic_skip___closed__6(void){
_start:
{
lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_13_ = ((lean_object*)(lp_aesop_Aesop_Script_Tactic_skip___closed__4));
v___x_14_ = lean_obj_once(&lp_aesop_Aesop_Script_Tactic_skip___closed__0, &lp_aesop_Aesop_Script_Tactic_skip___closed__0_once, _init_lp_aesop_Aesop_Script_Tactic_skip___closed__0);
v___x_15_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_15_, 0, v___x_14_);
lean_ctor_set(v___x_15_, 1, v___x_13_);
return v___x_15_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_Tactic_skip___closed__7(void){
_start:
{
lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; 
v___x_16_ = lean_obj_once(&lp_aesop_Aesop_Script_Tactic_skip___closed__6, &lp_aesop_Aesop_Script_Tactic_skip___closed__6_once, _init_lp_aesop_Aesop_Script_Tactic_skip___closed__6);
v___x_17_ = ((lean_object*)(lp_aesop_Aesop_Script_Tactic_skip___closed__5));
v___x_18_ = lean_obj_once(&lp_aesop_Aesop_Script_Tactic_skip___closed__0, &lp_aesop_Aesop_Script_Tactic_skip___closed__0_once, _init_lp_aesop_Aesop_Script_Tactic_skip___closed__0);
v___x_19_ = l_Lean_Syntax_node1(v___x_18_, v___x_17_, v___x_16_);
return v___x_19_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_Tactic_skip___closed__8(void){
_start:
{
lean_object* v___x_20_; lean_object* v___x_21_; 
v___x_20_ = lean_obj_once(&lp_aesop_Aesop_Script_Tactic_skip___closed__7, &lp_aesop_Aesop_Script_Tactic_skip___closed__7_once, _init_lp_aesop_Aesop_Script_Tactic_skip___closed__7);
v___x_21_ = lp_aesop_Aesop_Script_Tactic_unstructured(v___x_20_);
return v___x_21_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_Tactic_skip(void){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lean_obj_once(&lp_aesop_Aesop_Script_Tactic_skip___closed__8, &lp_aesop_Aesop_Script_Tactic_skip___closed__8_once, _init_lp_aesop_Aesop_Script_Tactic_skip___closed__8);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_applyStx___redArg(lean_object* v_e_29_, uint8_t v_md_30_, lean_object* v_a_31_){
_start:
{
lean_object* v_ref_33_; uint8_t v___x_34_; lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; 
v_ref_33_ = lean_ctor_get(v_a_31_, 5);
v___x_34_ = 0;
v___x_35_ = l_Lean_SourceInfo_fromRef(v_ref_33_, v___x_34_);
v___x_36_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_applyStx___redArg___closed__0));
v___x_37_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_applyStx___redArg___closed__1));
lean_inc(v___x_35_);
v___x_38_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_38_, 0, v___x_35_);
lean_ctor_set(v___x_38_, 1, v___x_36_);
v___x_39_ = l_Lean_Syntax_node2(v___x_35_, v___x_37_, v___x_38_, v_e_29_);
v___x_40_ = lp_aesop_Aesop_withAllTransparencySyntax(v_md_30_, v___x_39_);
v___x_41_ = lp_aesop_Aesop_Script_Tactic_unstructured(v___x_40_);
v___x_42_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_42_, 0, v___x_41_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_applyStx___redArg___boxed(lean_object* v_e_43_, lean_object* v_md_44_, lean_object* v_a_45_, lean_object* v_a_46_){
_start:
{
uint8_t v_md_boxed_47_; lean_object* v_res_48_; 
v_md_boxed_47_ = lean_unbox(v_md_44_);
v_res_48_ = lp_aesop_Aesop_Script_TacticBuilder_applyStx___redArg(v_e_43_, v_md_boxed_47_, v_a_45_);
lean_dec_ref(v_a_45_);
return v_res_48_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_applyStx(lean_object* v_e_49_, uint8_t v_md_50_, lean_object* v_a_51_, lean_object* v_a_52_, lean_object* v_a_53_, lean_object* v_a_54_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lp_aesop_Aesop_Script_TacticBuilder_applyStx___redArg(v_e_49_, v_md_50_, v_a_53_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_applyStx___boxed(lean_object* v_e_57_, lean_object* v_md_58_, lean_object* v_a_59_, lean_object* v_a_60_, lean_object* v_a_61_, lean_object* v_a_62_, lean_object* v_a_63_){
_start:
{
uint8_t v_md_boxed_64_; lean_object* v_res_65_; 
v_md_boxed_64_ = lean_unbox(v_md_58_);
v_res_65_ = lp_aesop_Aesop_Script_TacticBuilder_applyStx(v_e_57_, v_md_boxed_64_, v_a_59_, v_a_60_, v_a_61_, v_a_62_);
lean_dec(v_a_62_);
lean_dec_ref(v_a_61_);
lean_dec(v_a_60_);
lean_dec_ref(v_a_59_);
return v_res_65_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___redArg(lean_object* v_mvarId_66_, lean_object* v_x_67_, lean_object* v___y_68_, lean_object* v___y_69_, lean_object* v___y_70_, lean_object* v___y_71_){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_66_, v_x_67_, v___y_68_, v___y_69_, v___y_70_, v___y_71_);
if (lean_obj_tag(v___x_73_) == 0)
{
lean_object* v_a_74_; lean_object* v___x_76_; uint8_t v_isShared_77_; uint8_t v_isSharedCheck_81_; 
v_a_74_ = lean_ctor_get(v___x_73_, 0);
v_isSharedCheck_81_ = !lean_is_exclusive(v___x_73_);
if (v_isSharedCheck_81_ == 0)
{
v___x_76_ = v___x_73_;
v_isShared_77_ = v_isSharedCheck_81_;
goto v_resetjp_75_;
}
else
{
lean_inc(v_a_74_);
lean_dec(v___x_73_);
v___x_76_ = lean_box(0);
v_isShared_77_ = v_isSharedCheck_81_;
goto v_resetjp_75_;
}
v_resetjp_75_:
{
lean_object* v___x_79_; 
if (v_isShared_77_ == 0)
{
v___x_79_ = v___x_76_;
goto v_reusejp_78_;
}
else
{
lean_object* v_reuseFailAlloc_80_; 
v_reuseFailAlloc_80_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_80_, 0, v_a_74_);
v___x_79_ = v_reuseFailAlloc_80_;
goto v_reusejp_78_;
}
v_reusejp_78_:
{
return v___x_79_;
}
}
}
else
{
lean_object* v_a_82_; lean_object* v___x_84_; uint8_t v_isShared_85_; uint8_t v_isSharedCheck_89_; 
v_a_82_ = lean_ctor_get(v___x_73_, 0);
v_isSharedCheck_89_ = !lean_is_exclusive(v___x_73_);
if (v_isSharedCheck_89_ == 0)
{
v___x_84_ = v___x_73_;
v_isShared_85_ = v_isSharedCheck_89_;
goto v_resetjp_83_;
}
else
{
lean_inc(v_a_82_);
lean_dec(v___x_73_);
v___x_84_ = lean_box(0);
v_isShared_85_ = v_isSharedCheck_89_;
goto v_resetjp_83_;
}
v_resetjp_83_:
{
lean_object* v___x_87_; 
if (v_isShared_85_ == 0)
{
v___x_87_ = v___x_84_;
goto v_reusejp_86_;
}
else
{
lean_object* v_reuseFailAlloc_88_; 
v_reuseFailAlloc_88_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_88_, 0, v_a_82_);
v___x_87_ = v_reuseFailAlloc_88_;
goto v_reusejp_86_;
}
v_reusejp_86_:
{
return v___x_87_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___redArg___boxed(lean_object* v_mvarId_90_, lean_object* v_x_91_, lean_object* v___y_92_, lean_object* v___y_93_, lean_object* v___y_94_, lean_object* v___y_95_, lean_object* v___y_96_){
_start:
{
lean_object* v_res_97_; 
v_res_97_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___redArg(v_mvarId_90_, v_x_91_, v___y_92_, v___y_93_, v___y_94_, v___y_95_);
lean_dec(v___y_95_);
lean_dec_ref(v___y_94_);
lean_dec(v___y_93_);
lean_dec_ref(v___y_92_);
return v_res_97_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0(lean_object* v_00_u03b1_98_, lean_object* v_mvarId_99_, lean_object* v_x_100_, lean_object* v___y_101_, lean_object* v___y_102_, lean_object* v___y_103_, lean_object* v___y_104_){
_start:
{
lean_object* v___x_106_; 
v___x_106_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___redArg(v_mvarId_99_, v_x_100_, v___y_101_, v___y_102_, v___y_103_, v___y_104_);
return v___x_106_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___boxed(lean_object* v_00_u03b1_107_, lean_object* v_mvarId_108_, lean_object* v_x_109_, lean_object* v___y_110_, lean_object* v___y_111_, lean_object* v___y_112_, lean_object* v___y_113_, lean_object* v___y_114_){
_start:
{
lean_object* v_res_115_; 
v_res_115_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0(v_00_u03b1_107_, v_mvarId_108_, v_x_109_, v___y_110_, v___y_111_, v___y_112_, v___y_113_);
lean_dec(v___y_113_);
lean_dec_ref(v___y_112_);
lean_dec(v___y_111_);
lean_dec_ref(v___y_110_);
return v_res_115_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_apply(lean_object* v_mvarId_116_, lean_object* v_e_117_, uint8_t v_md_118_, lean_object* v_a_119_, lean_object* v_a_120_, lean_object* v_a_121_, lean_object* v_a_122_){
_start:
{
lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; 
v___x_124_ = lean_box(1);
v___x_125_ = lean_alloc_closure((void*)(l_Lean_PrettyPrinter_delab___boxed), 7, 2);
lean_closure_set(v___x_125_, 0, v_e_117_);
lean_closure_set(v___x_125_, 1, v___x_124_);
v___x_126_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___redArg(v_mvarId_116_, v___x_125_, v_a_119_, v_a_120_, v_a_121_, v_a_122_);
if (lean_obj_tag(v___x_126_) == 0)
{
lean_object* v_a_127_; lean_object* v___x_129_; uint8_t v_isShared_130_; uint8_t v_isSharedCheck_143_; 
v_a_127_ = lean_ctor_get(v___x_126_, 0);
v_isSharedCheck_143_ = !lean_is_exclusive(v___x_126_);
if (v_isSharedCheck_143_ == 0)
{
v___x_129_ = v___x_126_;
v_isShared_130_ = v_isSharedCheck_143_;
goto v_resetjp_128_;
}
else
{
lean_inc(v_a_127_);
lean_dec(v___x_126_);
v___x_129_ = lean_box(0);
v_isShared_130_ = v_isSharedCheck_143_;
goto v_resetjp_128_;
}
v_resetjp_128_:
{
lean_object* v_ref_131_; uint8_t v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_141_; 
v_ref_131_ = lean_ctor_get(v_a_121_, 5);
v___x_132_ = 0;
v___x_133_ = l_Lean_SourceInfo_fromRef(v_ref_131_, v___x_132_);
v___x_134_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_applyStx___redArg___closed__0));
v___x_135_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_applyStx___redArg___closed__1));
lean_inc(v___x_133_);
v___x_136_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_136_, 0, v___x_133_);
lean_ctor_set(v___x_136_, 1, v___x_134_);
v___x_137_ = l_Lean_Syntax_node2(v___x_133_, v___x_135_, v___x_136_, v_a_127_);
v___x_138_ = lp_aesop_Aesop_withAllTransparencySyntax(v_md_118_, v___x_137_);
v___x_139_ = lp_aesop_Aesop_Script_Tactic_unstructured(v___x_138_);
if (v_isShared_130_ == 0)
{
lean_ctor_set(v___x_129_, 0, v___x_139_);
v___x_141_ = v___x_129_;
goto v_reusejp_140_;
}
else
{
lean_object* v_reuseFailAlloc_142_; 
v_reuseFailAlloc_142_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_142_, 0, v___x_139_);
v___x_141_ = v_reuseFailAlloc_142_;
goto v_reusejp_140_;
}
v_reusejp_140_:
{
return v___x_141_;
}
}
}
else
{
lean_object* v_a_144_; lean_object* v___x_146_; uint8_t v_isShared_147_; uint8_t v_isSharedCheck_151_; 
v_a_144_ = lean_ctor_get(v___x_126_, 0);
v_isSharedCheck_151_ = !lean_is_exclusive(v___x_126_);
if (v_isSharedCheck_151_ == 0)
{
v___x_146_ = v___x_126_;
v_isShared_147_ = v_isSharedCheck_151_;
goto v_resetjp_145_;
}
else
{
lean_inc(v_a_144_);
lean_dec(v___x_126_);
v___x_146_ = lean_box(0);
v_isShared_147_ = v_isSharedCheck_151_;
goto v_resetjp_145_;
}
v_resetjp_145_:
{
lean_object* v___x_149_; 
if (v_isShared_147_ == 0)
{
v___x_149_ = v___x_146_;
goto v_reusejp_148_;
}
else
{
lean_object* v_reuseFailAlloc_150_; 
v_reuseFailAlloc_150_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_150_, 0, v_a_144_);
v___x_149_ = v_reuseFailAlloc_150_;
goto v_reusejp_148_;
}
v_reusejp_148_:
{
return v___x_149_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_apply___boxed(lean_object* v_mvarId_152_, lean_object* v_e_153_, lean_object* v_md_154_, lean_object* v_a_155_, lean_object* v_a_156_, lean_object* v_a_157_, lean_object* v_a_158_, lean_object* v_a_159_){
_start:
{
uint8_t v_md_boxed_160_; lean_object* v_res_161_; 
v_md_boxed_160_ = lean_unbox(v_md_154_);
v_res_161_ = lp_aesop_Aesop_Script_TacticBuilder_apply(v_mvarId_152_, v_e_153_, v_md_boxed_160_, v_a_155_, v_a_156_, v_a_157_, v_a_158_);
lean_dec(v_a_158_);
lean_dec_ref(v_a_157_);
lean_dec(v_a_156_);
lean_dec_ref(v_a_155_);
return v_res_161_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_exactFVar(lean_object* v_goal_168_, lean_object* v_fvarId_169_, uint8_t v_md_170_, lean_object* v_a_171_, lean_object* v_a_172_, lean_object* v_a_173_, lean_object* v_a_174_){
_start:
{
lean_object* v___x_176_; lean_object* v___x_177_; 
v___x_176_ = lean_alloc_closure((void*)(l_Lean_FVarId_getUserName___boxed), 6, 1);
lean_closure_set(v___x_176_, 0, v_fvarId_169_);
v___x_177_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___redArg(v_goal_168_, v___x_176_, v_a_171_, v_a_172_, v_a_173_, v_a_174_);
if (lean_obj_tag(v___x_177_) == 0)
{
lean_object* v_a_178_; lean_object* v___x_180_; uint8_t v_isShared_181_; uint8_t v_isSharedCheck_195_; 
v_a_178_ = lean_ctor_get(v___x_177_, 0);
v_isSharedCheck_195_ = !lean_is_exclusive(v___x_177_);
if (v_isSharedCheck_195_ == 0)
{
v___x_180_ = v___x_177_;
v_isShared_181_ = v_isSharedCheck_195_;
goto v_resetjp_179_;
}
else
{
lean_inc(v_a_178_);
lean_dec(v___x_177_);
v___x_180_ = lean_box(0);
v_isShared_181_ = v_isSharedCheck_195_;
goto v_resetjp_179_;
}
v_resetjp_179_:
{
lean_object* v_ref_182_; lean_object* v___x_183_; uint8_t v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_193_; 
v_ref_182_ = lean_ctor_get(v_a_173_, 5);
v___x_183_ = l_Lean_mkIdent(v_a_178_);
v___x_184_ = 0;
v___x_185_ = l_Lean_SourceInfo_fromRef(v_ref_182_, v___x_184_);
v___x_186_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_exactFVar___closed__0));
v___x_187_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_exactFVar___closed__1));
lean_inc(v___x_185_);
v___x_188_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_188_, 0, v___x_185_);
lean_ctor_set(v___x_188_, 1, v___x_186_);
v___x_189_ = l_Lean_Syntax_node2(v___x_185_, v___x_187_, v___x_188_, v___x_183_);
v___x_190_ = lp_aesop_Aesop_withAllTransparencySyntax(v_md_170_, v___x_189_);
v___x_191_ = lp_aesop_Aesop_Script_Tactic_unstructured(v___x_190_);
if (v_isShared_181_ == 0)
{
lean_ctor_set(v___x_180_, 0, v___x_191_);
v___x_193_ = v___x_180_;
goto v_reusejp_192_;
}
else
{
lean_object* v_reuseFailAlloc_194_; 
v_reuseFailAlloc_194_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_194_, 0, v___x_191_);
v___x_193_ = v_reuseFailAlloc_194_;
goto v_reusejp_192_;
}
v_reusejp_192_:
{
return v___x_193_;
}
}
}
else
{
lean_object* v_a_196_; lean_object* v___x_198_; uint8_t v_isShared_199_; uint8_t v_isSharedCheck_203_; 
v_a_196_ = lean_ctor_get(v___x_177_, 0);
v_isSharedCheck_203_ = !lean_is_exclusive(v___x_177_);
if (v_isSharedCheck_203_ == 0)
{
v___x_198_ = v___x_177_;
v_isShared_199_ = v_isSharedCheck_203_;
goto v_resetjp_197_;
}
else
{
lean_inc(v_a_196_);
lean_dec(v___x_177_);
v___x_198_ = lean_box(0);
v_isShared_199_ = v_isSharedCheck_203_;
goto v_resetjp_197_;
}
v_resetjp_197_:
{
lean_object* v___x_201_; 
if (v_isShared_199_ == 0)
{
v___x_201_ = v___x_198_;
goto v_reusejp_200_;
}
else
{
lean_object* v_reuseFailAlloc_202_; 
v_reuseFailAlloc_202_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_202_, 0, v_a_196_);
v___x_201_ = v_reuseFailAlloc_202_;
goto v_reusejp_200_;
}
v_reusejp_200_:
{
return v___x_201_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_exactFVar___boxed(lean_object* v_goal_204_, lean_object* v_fvarId_205_, lean_object* v_md_206_, lean_object* v_a_207_, lean_object* v_a_208_, lean_object* v_a_209_, lean_object* v_a_210_, lean_object* v_a_211_){
_start:
{
uint8_t v_md_boxed_212_; lean_object* v_res_213_; 
v_md_boxed_212_ = lean_unbox(v_md_206_);
v_res_213_ = lp_aesop_Aesop_Script_TacticBuilder_exactFVar(v_goal_204_, v_fvarId_205_, v_md_boxed_212_, v_a_207_, v_a_208_, v_a_209_, v_a_210_);
lean_dec(v_a_210_);
lean_dec_ref(v_a_209_);
lean_dec(v_a_208_);
lean_dec_ref(v_a_207_);
return v_res_213_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11(void){
_start:
{
lean_object* v___x_242_; 
v___x_242_ = l_Array_mkArray0(lean_box(0));
return v___x_242_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_replace(lean_object* v_preGoal_251_, lean_object* v_postGoal_252_, lean_object* v_fvarId_253_, lean_object* v_type_254_, lean_object* v_proof_255_, lean_object* v_a_256_, lean_object* v_a_257_, lean_object* v_a_258_, lean_object* v_a_259_){
_start:
{
lean_object* v___x_261_; lean_object* v___x_262_; 
v___x_261_ = lean_alloc_closure((void*)(l_Lean_FVarId_getUserName___boxed), 6, 1);
lean_closure_set(v___x_261_, 0, v_fvarId_253_);
lean_inc(v_preGoal_251_);
v___x_262_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___redArg(v_preGoal_251_, v___x_261_, v_a_256_, v_a_257_, v_a_258_, v_a_259_);
if (lean_obj_tag(v___x_262_) == 0)
{
lean_object* v_a_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; 
v_a_263_ = lean_ctor_get(v___x_262_, 0);
lean_inc(v_a_263_);
lean_dec_ref_known(v___x_262_, 1);
v___x_264_ = lean_box(1);
v___x_265_ = lean_alloc_closure((void*)(l_Lean_PrettyPrinter_delab___boxed), 7, 2);
lean_closure_set(v___x_265_, 0, v_proof_255_);
lean_closure_set(v___x_265_, 1, v___x_264_);
v___x_266_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___redArg(v_preGoal_251_, v___x_265_, v_a_256_, v_a_257_, v_a_258_, v_a_259_);
if (lean_obj_tag(v___x_266_) == 0)
{
lean_object* v_a_267_; lean_object* v___x_268_; lean_object* v___x_269_; 
v_a_267_ = lean_ctor_get(v___x_266_, 0);
lean_inc(v_a_267_);
lean_dec_ref_known(v___x_266_, 1);
v___x_268_ = lean_alloc_closure((void*)(l_Lean_PrettyPrinter_delab___boxed), 7, 2);
lean_closure_set(v___x_268_, 0, v_type_254_);
lean_closure_set(v___x_268_, 1, v___x_264_);
v___x_269_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___redArg(v_postGoal_252_, v___x_268_, v_a_256_, v_a_257_, v_a_258_, v_a_259_);
if (lean_obj_tag(v___x_269_) == 0)
{
lean_object* v_a_270_; lean_object* v___x_272_; uint8_t v_isShared_273_; uint8_t v_isSharedCheck_302_; 
v_a_270_ = lean_ctor_get(v___x_269_, 0);
v_isSharedCheck_302_ = !lean_is_exclusive(v___x_269_);
if (v_isSharedCheck_302_ == 0)
{
v___x_272_ = v___x_269_;
v_isShared_273_ = v_isSharedCheck_302_;
goto v_resetjp_271_;
}
else
{
lean_inc(v_a_270_);
lean_dec(v___x_269_);
v___x_272_ = lean_box(0);
v_isShared_273_ = v_isSharedCheck_302_;
goto v_resetjp_271_;
}
v_resetjp_271_:
{
lean_object* v_ref_274_; uint8_t v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_300_; 
v_ref_274_ = lean_ctor_get(v_a_258_, 5);
v___x_275_ = 0;
v___x_276_ = l_Lean_SourceInfo_fromRef(v_ref_274_, v___x_275_);
v___x_277_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__0));
v___x_278_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__1));
lean_inc_n(v___x_276_, 9);
v___x_279_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_279_, 0, v___x_276_);
lean_ctor_set(v___x_279_, 1, v___x_277_);
v___x_280_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__4));
v___x_281_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__6));
v___x_282_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__8));
v___x_283_ = l_Lean_mkIdent(v_a_263_);
v___x_284_ = l_Lean_Syntax_node1(v___x_276_, v___x_282_, v___x_283_);
v___x_285_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_286_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
v___x_287_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_287_, 0, v___x_276_);
lean_ctor_set(v___x_287_, 1, v___x_285_);
lean_ctor_set(v___x_287_, 2, v___x_286_);
v___x_288_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__13));
v___x_289_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__14));
v___x_290_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_290_, 0, v___x_276_);
lean_ctor_set(v___x_290_, 1, v___x_289_);
v___x_291_ = l_Lean_Syntax_node2(v___x_276_, v___x_288_, v___x_290_, v_a_270_);
v___x_292_ = l_Lean_Syntax_node1(v___x_276_, v___x_285_, v___x_291_);
v___x_293_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__15));
v___x_294_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_294_, 0, v___x_276_);
lean_ctor_set(v___x_294_, 1, v___x_293_);
v___x_295_ = l_Lean_Syntax_node5(v___x_276_, v___x_281_, v___x_284_, v___x_287_, v___x_292_, v___x_294_, v_a_267_);
v___x_296_ = l_Lean_Syntax_node1(v___x_276_, v___x_280_, v___x_295_);
v___x_297_ = l_Lean_Syntax_node2(v___x_276_, v___x_278_, v___x_279_, v___x_296_);
v___x_298_ = lp_aesop_Aesop_Script_Tactic_unstructured(v___x_297_);
if (v_isShared_273_ == 0)
{
lean_ctor_set(v___x_272_, 0, v___x_298_);
v___x_300_ = v___x_272_;
goto v_reusejp_299_;
}
else
{
lean_object* v_reuseFailAlloc_301_; 
v_reuseFailAlloc_301_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_301_, 0, v___x_298_);
v___x_300_ = v_reuseFailAlloc_301_;
goto v_reusejp_299_;
}
v_reusejp_299_:
{
return v___x_300_;
}
}
}
else
{
lean_object* v_a_303_; lean_object* v___x_305_; uint8_t v_isShared_306_; uint8_t v_isSharedCheck_310_; 
lean_dec(v_a_267_);
lean_dec(v_a_263_);
v_a_303_ = lean_ctor_get(v___x_269_, 0);
v_isSharedCheck_310_ = !lean_is_exclusive(v___x_269_);
if (v_isSharedCheck_310_ == 0)
{
v___x_305_ = v___x_269_;
v_isShared_306_ = v_isSharedCheck_310_;
goto v_resetjp_304_;
}
else
{
lean_inc(v_a_303_);
lean_dec(v___x_269_);
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
lean_dec(v_a_263_);
lean_dec_ref(v_type_254_);
lean_dec(v_postGoal_252_);
v_a_311_ = lean_ctor_get(v___x_266_, 0);
v_isSharedCheck_318_ = !lean_is_exclusive(v___x_266_);
if (v_isSharedCheck_318_ == 0)
{
v___x_313_ = v___x_266_;
v_isShared_314_ = v_isSharedCheck_318_;
goto v_resetjp_312_;
}
else
{
lean_inc(v_a_311_);
lean_dec(v___x_266_);
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
lean_dec_ref(v_proof_255_);
lean_dec_ref(v_type_254_);
lean_dec(v_postGoal_252_);
lean_dec(v_preGoal_251_);
v_a_319_ = lean_ctor_get(v___x_262_, 0);
v_isSharedCheck_326_ = !lean_is_exclusive(v___x_262_);
if (v_isSharedCheck_326_ == 0)
{
v___x_321_ = v___x_262_;
v_isShared_322_ = v_isSharedCheck_326_;
goto v_resetjp_320_;
}
else
{
lean_inc(v_a_319_);
lean_dec(v___x_262_);
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_replace___boxed(lean_object* v_preGoal_327_, lean_object* v_postGoal_328_, lean_object* v_fvarId_329_, lean_object* v_type_330_, lean_object* v_proof_331_, lean_object* v_a_332_, lean_object* v_a_333_, lean_object* v_a_334_, lean_object* v_a_335_, lean_object* v_a_336_){
_start:
{
lean_object* v_res_337_; 
v_res_337_ = lp_aesop_Aesop_Script_TacticBuilder_replace(v_preGoal_327_, v_postGoal_328_, v_fvarId_329_, v_type_330_, v_proof_331_, v_a_332_, v_a_333_, v_a_334_, v_a_335_);
lean_dec(v_a_335_);
lean_dec_ref(v_a_334_);
lean_dec(v_a_333_);
lean_dec_ref(v_a_332_);
return v_res_337_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0(lean_object* v_type_351_, lean_object* v___x_352_, lean_object* v_value_353_, lean_object* v_userName_354_, uint8_t v_md_355_, lean_object* v___y_356_, lean_object* v___y_357_, lean_object* v___y_358_, lean_object* v___y_359_){
_start:
{
lean_object* v___x_361_; 
lean_inc(v___x_352_);
v___x_361_ = l_Lean_PrettyPrinter_delab(v_type_351_, v___x_352_, v___y_356_, v___y_357_, v___y_358_, v___y_359_);
if (lean_obj_tag(v___x_361_) == 0)
{
lean_object* v_a_362_; lean_object* v___x_363_; 
v_a_362_ = lean_ctor_get(v___x_361_, 0);
lean_inc(v_a_362_);
lean_dec_ref_known(v___x_361_, 1);
v___x_363_ = l_Lean_PrettyPrinter_delab(v_value_353_, v___x_352_, v___y_356_, v___y_357_, v___y_358_, v___y_359_);
if (lean_obj_tag(v___x_363_) == 0)
{
lean_object* v_a_364_; lean_object* v___x_366_; uint8_t v_isShared_367_; uint8_t v_isSharedCheck_399_; 
v_a_364_ = lean_ctor_get(v___x_363_, 0);
v_isSharedCheck_399_ = !lean_is_exclusive(v___x_363_);
if (v_isSharedCheck_399_ == 0)
{
v___x_366_ = v___x_363_;
v_isShared_367_ = v_isSharedCheck_399_;
goto v_resetjp_365_;
}
else
{
lean_inc(v_a_364_);
lean_dec(v___x_363_);
v___x_366_ = lean_box(0);
v_isShared_367_ = v_isSharedCheck_399_;
goto v_resetjp_365_;
}
v_resetjp_365_:
{
lean_object* v_ref_368_; uint8_t v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_397_; 
v_ref_368_ = lean_ctor_get(v___y_358_, 5);
v___x_369_ = 0;
v___x_370_ = l_Lean_SourceInfo_fromRef(v_ref_368_, v___x_369_);
v___x_371_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__1));
v___x_372_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__2));
lean_inc_n(v___x_370_, 10);
v___x_373_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_373_, 0, v___x_370_);
lean_ctor_set(v___x_373_, 1, v___x_372_);
v___x_374_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___closed__4));
v___x_375_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_376_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
v___x_377_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_377_, 0, v___x_370_);
lean_ctor_set(v___x_377_, 1, v___x_375_);
lean_ctor_set(v___x_377_, 2, v___x_376_);
lean_inc_ref(v___x_377_);
v___x_378_ = l_Lean_Syntax_node1(v___x_370_, v___x_374_, v___x_377_);
v___x_379_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__4));
v___x_380_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__6));
v___x_381_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__8));
v___x_382_ = l_Lean_mkIdent(v_userName_354_);
v___x_383_ = l_Lean_Syntax_node1(v___x_370_, v___x_381_, v___x_382_);
v___x_384_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__13));
v___x_385_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__14));
v___x_386_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_386_, 0, v___x_370_);
lean_ctor_set(v___x_386_, 1, v___x_385_);
v___x_387_ = l_Lean_Syntax_node2(v___x_370_, v___x_384_, v___x_386_, v_a_362_);
v___x_388_ = l_Lean_Syntax_node1(v___x_370_, v___x_375_, v___x_387_);
v___x_389_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__15));
v___x_390_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_390_, 0, v___x_370_);
lean_ctor_set(v___x_390_, 1, v___x_389_);
v___x_391_ = l_Lean_Syntax_node5(v___x_370_, v___x_380_, v___x_383_, v___x_377_, v___x_388_, v___x_390_, v_a_364_);
v___x_392_ = l_Lean_Syntax_node1(v___x_370_, v___x_379_, v___x_391_);
v___x_393_ = l_Lean_Syntax_node3(v___x_370_, v___x_371_, v___x_373_, v___x_378_, v___x_392_);
v___x_394_ = lp_aesop_Aesop_withAllTransparencySyntax(v_md_355_, v___x_393_);
v___x_395_ = lp_aesop_Aesop_Script_Tactic_unstructured(v___x_394_);
if (v_isShared_367_ == 0)
{
lean_ctor_set(v___x_366_, 0, v___x_395_);
v___x_397_ = v___x_366_;
goto v_reusejp_396_;
}
else
{
lean_object* v_reuseFailAlloc_398_; 
v_reuseFailAlloc_398_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_398_, 0, v___x_395_);
v___x_397_ = v_reuseFailAlloc_398_;
goto v_reusejp_396_;
}
v_reusejp_396_:
{
return v___x_397_;
}
}
}
else
{
lean_object* v_a_400_; lean_object* v___x_402_; uint8_t v_isShared_403_; uint8_t v_isSharedCheck_407_; 
lean_dec(v_a_362_);
lean_dec(v_userName_354_);
v_a_400_ = lean_ctor_get(v___x_363_, 0);
v_isSharedCheck_407_ = !lean_is_exclusive(v___x_363_);
if (v_isSharedCheck_407_ == 0)
{
v___x_402_ = v___x_363_;
v_isShared_403_ = v_isSharedCheck_407_;
goto v_resetjp_401_;
}
else
{
lean_inc(v_a_400_);
lean_dec(v___x_363_);
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
else
{
lean_object* v_a_408_; lean_object* v___x_410_; uint8_t v_isShared_411_; uint8_t v_isSharedCheck_415_; 
lean_dec(v_userName_354_);
lean_dec_ref(v_value_353_);
lean_dec(v___x_352_);
v_a_408_ = lean_ctor_get(v___x_361_, 0);
v_isSharedCheck_415_ = !lean_is_exclusive(v___x_361_);
if (v_isSharedCheck_415_ == 0)
{
v___x_410_ = v___x_361_;
v_isShared_411_ = v_isSharedCheck_415_;
goto v_resetjp_409_;
}
else
{
lean_inc(v_a_408_);
lean_dec(v___x_361_);
v___x_410_ = lean_box(0);
v_isShared_411_ = v_isSharedCheck_415_;
goto v_resetjp_409_;
}
v_resetjp_409_:
{
lean_object* v___x_413_; 
if (v_isShared_411_ == 0)
{
v___x_413_ = v___x_410_;
goto v_reusejp_412_;
}
else
{
lean_object* v_reuseFailAlloc_414_; 
v_reuseFailAlloc_414_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_414_, 0, v_a_408_);
v___x_413_ = v_reuseFailAlloc_414_;
goto v_reusejp_412_;
}
v_reusejp_412_:
{
return v___x_413_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___boxed(lean_object* v_type_416_, lean_object* v___x_417_, lean_object* v_value_418_, lean_object* v_userName_419_, lean_object* v_md_420_, lean_object* v___y_421_, lean_object* v___y_422_, lean_object* v___y_423_, lean_object* v___y_424_, lean_object* v___y_425_){
_start:
{
uint8_t v_md_boxed_426_; lean_object* v_res_427_; 
v_md_boxed_426_ = lean_unbox(v_md_420_);
v_res_427_ = lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0(v_type_416_, v___x_417_, v_value_418_, v_userName_419_, v_md_boxed_426_, v___y_421_, v___y_422_, v___y_423_, v___y_424_);
lean_dec(v___y_424_);
lean_dec_ref(v___y_423_);
lean_dec(v___y_422_);
lean_dec_ref(v___y_421_);
return v_res_427_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis(lean_object* v_goal_428_, lean_object* v_h_429_, uint8_t v_md_430_, lean_object* v_a_431_, lean_object* v_a_432_, lean_object* v_a_433_, lean_object* v_a_434_){
_start:
{
lean_object* v_userName_436_; lean_object* v_type_437_; lean_object* v_value_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___f_441_; lean_object* v___x_442_; 
v_userName_436_ = lean_ctor_get(v_h_429_, 0);
lean_inc(v_userName_436_);
v_type_437_ = lean_ctor_get(v_h_429_, 1);
lean_inc_ref(v_type_437_);
v_value_438_ = lean_ctor_get(v_h_429_, 2);
lean_inc_ref(v_value_438_);
lean_dec_ref(v_h_429_);
v___x_439_ = lean_box(1);
v___x_440_ = lean_box(v_md_430_);
v___f_441_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___lam__0___boxed), 10, 5);
lean_closure_set(v___f_441_, 0, v_type_437_);
lean_closure_set(v___f_441_, 1, v___x_439_);
lean_closure_set(v___f_441_, 2, v_value_438_);
lean_closure_set(v___f_441_, 3, v_userName_436_);
lean_closure_set(v___f_441_, 4, v___x_440_);
v___x_442_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___redArg(v_goal_428_, v___f_441_, v_a_431_, v_a_432_, v_a_433_, v_a_434_);
return v___x_442_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis___boxed(lean_object* v_goal_443_, lean_object* v_h_444_, lean_object* v_md_445_, lean_object* v_a_446_, lean_object* v_a_447_, lean_object* v_a_448_, lean_object* v_a_449_, lean_object* v_a_450_){
_start:
{
uint8_t v_md_boxed_451_; lean_object* v_res_452_; 
v_md_boxed_451_ = lean_unbox(v_md_445_);
v_res_452_ = lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis(v_goal_443_, v_h_444_, v_md_boxed_451_, v_a_446_, v_a_447_, v_a_448_, v_a_449_);
lean_dec(v_a_449_);
lean_dec_ref(v_a_448_);
lean_dec(v_a_447_);
lean_dec_ref(v_a_446_);
return v_res_452_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_clear_spec__0___redArg(size_t v_sz_453_, size_t v_i_454_, lean_object* v_bs_455_, lean_object* v___y_456_, lean_object* v___y_457_, lean_object* v___y_458_){
_start:
{
uint8_t v___x_460_; 
v___x_460_ = lean_usize_dec_lt(v_i_454_, v_sz_453_);
if (v___x_460_ == 0)
{
lean_object* v___x_461_; 
v___x_461_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_461_, 0, v_bs_455_);
return v___x_461_;
}
else
{
lean_object* v_v_462_; lean_object* v___x_463_; 
v_v_462_ = lean_array_uget_borrowed(v_bs_455_, v_i_454_);
lean_inc(v_v_462_);
v___x_463_ = l_Lean_FVarId_getUserName___redArg(v_v_462_, v___y_456_, v___y_457_, v___y_458_);
if (lean_obj_tag(v___x_463_) == 0)
{
lean_object* v_a_464_; lean_object* v___x_465_; lean_object* v_bs_x27_466_; lean_object* v___x_467_; size_t v___x_468_; size_t v___x_469_; lean_object* v___x_470_; 
v_a_464_ = lean_ctor_get(v___x_463_, 0);
lean_inc(v_a_464_);
lean_dec_ref_known(v___x_463_, 1);
v___x_465_ = lean_unsigned_to_nat(0u);
v_bs_x27_466_ = lean_array_uset(v_bs_455_, v_i_454_, v___x_465_);
v___x_467_ = l_Lean_mkIdent(v_a_464_);
v___x_468_ = ((size_t)1ULL);
v___x_469_ = lean_usize_add(v_i_454_, v___x_468_);
v___x_470_ = lean_array_uset(v_bs_x27_466_, v_i_454_, v___x_467_);
v_i_454_ = v___x_469_;
v_bs_455_ = v___x_470_;
goto _start;
}
else
{
lean_object* v_a_472_; lean_object* v___x_474_; uint8_t v_isShared_475_; uint8_t v_isSharedCheck_479_; 
lean_dec_ref(v_bs_455_);
v_a_472_ = lean_ctor_get(v___x_463_, 0);
v_isSharedCheck_479_ = !lean_is_exclusive(v___x_463_);
if (v_isSharedCheck_479_ == 0)
{
v___x_474_ = v___x_463_;
v_isShared_475_ = v_isSharedCheck_479_;
goto v_resetjp_473_;
}
else
{
lean_inc(v_a_472_);
lean_dec(v___x_463_);
v___x_474_ = lean_box(0);
v_isShared_475_ = v_isSharedCheck_479_;
goto v_resetjp_473_;
}
v_resetjp_473_:
{
lean_object* v___x_477_; 
if (v_isShared_475_ == 0)
{
v___x_477_ = v___x_474_;
goto v_reusejp_476_;
}
else
{
lean_object* v_reuseFailAlloc_478_; 
v_reuseFailAlloc_478_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_478_, 0, v_a_472_);
v___x_477_ = v_reuseFailAlloc_478_;
goto v_reusejp_476_;
}
v_reusejp_476_:
{
return v___x_477_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_clear_spec__0___redArg___boxed(lean_object* v_sz_480_, lean_object* v_i_481_, lean_object* v_bs_482_, lean_object* v___y_483_, lean_object* v___y_484_, lean_object* v___y_485_, lean_object* v___y_486_){
_start:
{
size_t v_sz_boxed_487_; size_t v_i_boxed_488_; lean_object* v_res_489_; 
v_sz_boxed_487_ = lean_unbox_usize(v_sz_480_);
lean_dec(v_sz_480_);
v_i_boxed_488_ = lean_unbox_usize(v_i_481_);
lean_dec(v_i_481_);
v_res_489_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_clear_spec__0___redArg(v_sz_boxed_487_, v_i_boxed_488_, v_bs_482_, v___y_483_, v___y_484_, v___y_485_);
lean_dec(v___y_485_);
lean_dec_ref(v___y_484_);
lean_dec_ref(v___y_483_);
return v_res_489_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_clear_spec__1(size_t v_sz_490_, size_t v_i_491_, lean_object* v_bs_492_){
_start:
{
uint8_t v___x_493_; 
v___x_493_ = lean_usize_dec_lt(v_i_491_, v_sz_490_);
if (v___x_493_ == 0)
{
return v_bs_492_;
}
else
{
lean_object* v_v_494_; lean_object* v___x_495_; lean_object* v_bs_x27_496_; size_t v___x_497_; size_t v___x_498_; lean_object* v___x_499_; 
v_v_494_ = lean_array_uget(v_bs_492_, v_i_491_);
v___x_495_ = lean_unsigned_to_nat(0u);
v_bs_x27_496_ = lean_array_uset(v_bs_492_, v_i_491_, v___x_495_);
v___x_497_ = ((size_t)1ULL);
v___x_498_ = lean_usize_add(v_i_491_, v___x_497_);
v___x_499_ = lean_array_uset(v_bs_x27_496_, v_i_491_, v_v_494_);
v_i_491_ = v___x_498_;
v_bs_492_ = v___x_499_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_clear_spec__1___boxed(lean_object* v_sz_501_, lean_object* v_i_502_, lean_object* v_bs_503_){
_start:
{
size_t v_sz_boxed_504_; size_t v_i_boxed_505_; lean_object* v_res_506_; 
v_sz_boxed_504_ = lean_unbox_usize(v_sz_501_);
lean_dec(v_sz_501_);
v_i_boxed_505_ = lean_unbox_usize(v_i_502_);
lean_dec(v_i_502_);
v_res_506_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_clear_spec__1(v_sz_boxed_504_, v_i_boxed_505_, v_bs_503_);
return v_res_506_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_clear___lam__0(lean_object* v_fvarIds_513_, lean_object* v___y_514_, lean_object* v___y_515_, lean_object* v___y_516_, lean_object* v___y_517_){
_start:
{
size_t v_sz_519_; size_t v___x_520_; lean_object* v___x_521_; 
v_sz_519_ = lean_array_size(v_fvarIds_513_);
v___x_520_ = ((size_t)0ULL);
v___x_521_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_clear_spec__0___redArg(v_sz_519_, v___x_520_, v_fvarIds_513_, v___y_514_, v___y_516_, v___y_517_);
if (lean_obj_tag(v___x_521_) == 0)
{
lean_object* v_a_522_; lean_object* v___x_524_; uint8_t v_isShared_525_; uint8_t v_isSharedCheck_543_; 
v_a_522_ = lean_ctor_get(v___x_521_, 0);
v_isSharedCheck_543_ = !lean_is_exclusive(v___x_521_);
if (v_isSharedCheck_543_ == 0)
{
v___x_524_ = v___x_521_;
v_isShared_525_ = v_isSharedCheck_543_;
goto v_resetjp_523_;
}
else
{
lean_inc(v_a_522_);
lean_dec(v___x_521_);
v___x_524_ = lean_box(0);
v_isShared_525_ = v_isSharedCheck_543_;
goto v_resetjp_523_;
}
v_resetjp_523_:
{
lean_object* v_ref_526_; uint8_t v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; size_t v_sz_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_541_; 
v_ref_526_ = lean_ctor_get(v___y_516_, 5);
v___x_527_ = 0;
v___x_528_ = l_Lean_SourceInfo_fromRef(v_ref_526_, v___x_527_);
v___x_529_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_clear___lam__0___closed__0));
v___x_530_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_clear___lam__0___closed__1));
lean_inc_n(v___x_528_, 2);
v___x_531_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_531_, 0, v___x_528_);
lean_ctor_set(v___x_531_, 1, v___x_529_);
v___x_532_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_533_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
v_sz_534_ = lean_array_size(v_a_522_);
v___x_535_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_clear_spec__1(v_sz_534_, v___x_520_, v_a_522_);
v___x_536_ = l_Array_append___redArg(v___x_533_, v___x_535_);
lean_dec_ref(v___x_535_);
v___x_537_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_537_, 0, v___x_528_);
lean_ctor_set(v___x_537_, 1, v___x_532_);
lean_ctor_set(v___x_537_, 2, v___x_536_);
v___x_538_ = l_Lean_Syntax_node2(v___x_528_, v___x_530_, v___x_531_, v___x_537_);
v___x_539_ = lp_aesop_Aesop_Script_Tactic_unstructured(v___x_538_);
if (v_isShared_525_ == 0)
{
lean_ctor_set(v___x_524_, 0, v___x_539_);
v___x_541_ = v___x_524_;
goto v_reusejp_540_;
}
else
{
lean_object* v_reuseFailAlloc_542_; 
v_reuseFailAlloc_542_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_542_, 0, v___x_539_);
v___x_541_ = v_reuseFailAlloc_542_;
goto v_reusejp_540_;
}
v_reusejp_540_:
{
return v___x_541_;
}
}
}
else
{
lean_object* v_a_544_; lean_object* v___x_546_; uint8_t v_isShared_547_; uint8_t v_isSharedCheck_551_; 
v_a_544_ = lean_ctor_get(v___x_521_, 0);
v_isSharedCheck_551_ = !lean_is_exclusive(v___x_521_);
if (v_isSharedCheck_551_ == 0)
{
v___x_546_ = v___x_521_;
v_isShared_547_ = v_isSharedCheck_551_;
goto v_resetjp_545_;
}
else
{
lean_inc(v_a_544_);
lean_dec(v___x_521_);
v___x_546_ = lean_box(0);
v_isShared_547_ = v_isSharedCheck_551_;
goto v_resetjp_545_;
}
v_resetjp_545_:
{
lean_object* v___x_549_; 
if (v_isShared_547_ == 0)
{
v___x_549_ = v___x_546_;
goto v_reusejp_548_;
}
else
{
lean_object* v_reuseFailAlloc_550_; 
v_reuseFailAlloc_550_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_550_, 0, v_a_544_);
v___x_549_ = v_reuseFailAlloc_550_;
goto v_reusejp_548_;
}
v_reusejp_548_:
{
return v___x_549_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_clear___lam__0___boxed(lean_object* v_fvarIds_552_, lean_object* v___y_553_, lean_object* v___y_554_, lean_object* v___y_555_, lean_object* v___y_556_, lean_object* v___y_557_){
_start:
{
lean_object* v_res_558_; 
v_res_558_ = lp_aesop_Aesop_Script_TacticBuilder_clear___lam__0(v_fvarIds_552_, v___y_553_, v___y_554_, v___y_555_, v___y_556_);
lean_dec(v___y_556_);
lean_dec_ref(v___y_555_);
lean_dec(v___y_554_);
lean_dec_ref(v___y_553_);
return v_res_558_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_clear(lean_object* v_goal_559_, lean_object* v_fvarIds_560_, lean_object* v_a_561_, lean_object* v_a_562_, lean_object* v_a_563_, lean_object* v_a_564_){
_start:
{
lean_object* v___f_566_; lean_object* v___x_567_; 
v___f_566_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticBuilder_clear___lam__0___boxed), 6, 1);
lean_closure_set(v___f_566_, 0, v_fvarIds_560_);
v___x_567_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___redArg(v_goal_559_, v___f_566_, v_a_561_, v_a_562_, v_a_563_, v_a_564_);
return v___x_567_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_clear___boxed(lean_object* v_goal_568_, lean_object* v_fvarIds_569_, lean_object* v_a_570_, lean_object* v_a_571_, lean_object* v_a_572_, lean_object* v_a_573_, lean_object* v_a_574_){
_start:
{
lean_object* v_res_575_; 
v_res_575_ = lp_aesop_Aesop_Script_TacticBuilder_clear(v_goal_568_, v_fvarIds_569_, v_a_570_, v_a_571_, v_a_572_, v_a_573_);
lean_dec(v_a_573_);
lean_dec_ref(v_a_572_);
lean_dec(v_a_571_);
lean_dec_ref(v_a_570_);
return v_res_575_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_clear_spec__0(size_t v_sz_576_, size_t v_i_577_, lean_object* v_bs_578_, lean_object* v___y_579_, lean_object* v___y_580_, lean_object* v___y_581_, lean_object* v___y_582_){
_start:
{
lean_object* v___x_584_; 
v___x_584_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_clear_spec__0___redArg(v_sz_576_, v_i_577_, v_bs_578_, v___y_579_, v___y_581_, v___y_582_);
return v___x_584_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_clear_spec__0___boxed(lean_object* v_sz_585_, lean_object* v_i_586_, lean_object* v_bs_587_, lean_object* v___y_588_, lean_object* v___y_589_, lean_object* v___y_590_, lean_object* v___y_591_, lean_object* v___y_592_){
_start:
{
size_t v_sz_boxed_593_; size_t v_i_boxed_594_; lean_object* v_res_595_; 
v_sz_boxed_593_ = lean_unbox_usize(v_sz_585_);
lean_dec(v_sz_585_);
v_i_boxed_594_ = lean_unbox_usize(v_i_586_);
lean_dec(v_i_586_);
v_res_595_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_clear_spec__0(v_sz_boxed_593_, v_i_boxed_594_, v_bs_587_, v___y_588_, v___y_589_, v___y_590_, v___y_591_);
lean_dec(v___y_591_);
lean_dec_ref(v___y_590_);
lean_dec(v___y_589_);
lean_dec_ref(v___y_588_);
return v_res_595_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__2(void){
_start:
{
lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; 
v___x_602_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__0));
v___x_603_ = lean_obj_once(&lp_aesop_Aesop_Script_Tactic_skip___closed__0, &lp_aesop_Aesop_Script_Tactic_skip___closed__0_once, _init_lp_aesop_Aesop_Script_Tactic_skip___closed__0);
v___x_604_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_604_, 0, v___x_603_);
lean_ctor_set(v___x_604_, 1, v___x_602_);
return v___x_604_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__5(void){
_start:
{
lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; 
v___x_611_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
v___x_612_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_613_ = lean_obj_once(&lp_aesop_Aesop_Script_Tactic_skip___closed__0, &lp_aesop_Aesop_Script_Tactic_skip___closed__0_once, _init_lp_aesop_Aesop_Script_Tactic_skip___closed__0);
v___x_614_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_614_, 0, v___x_613_);
lean_ctor_set(v___x_614_, 1, v___x_612_);
lean_ctor_set(v___x_614_, 2, v___x_611_);
return v___x_614_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0(lean_object* v_ctorNames_615_, lean_object* v_a_616_, lean_object* v_conts_617_){
_start:
{
lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; 
v___x_618_ = l_Array_zip___redArg(v_ctorNames_615_, v_conts_617_);
v___x_619_ = lp_aesop_Aesop_ctorNamesToInductionAlts(v___x_618_);
v___x_620_ = lean_obj_once(&lp_aesop_Aesop_Script_Tactic_skip___closed__0, &lp_aesop_Aesop_Script_Tactic_skip___closed__0_once, _init_lp_aesop_Aesop_Script_Tactic_skip___closed__0);
v___x_621_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__1));
v___x_622_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__2, &lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__2_once, _init_lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__2);
v___x_623_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_624_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__4));
v___x_625_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__5, &lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__5_once, _init_lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__5);
v___x_626_ = l_Lean_Syntax_node2(v___x_620_, v___x_624_, v___x_625_, v_a_616_);
v___x_627_ = l_Lean_Syntax_node1(v___x_620_, v___x_623_, v___x_626_);
v___x_628_ = l_Lean_Syntax_node1(v___x_620_, v___x_623_, v___x_619_);
v___x_629_ = l_Lean_Syntax_node4(v___x_620_, v___x_621_, v___x_622_, v___x_627_, v___x_625_, v___x_628_);
return v___x_629_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___boxed(lean_object* v_ctorNames_630_, lean_object* v_a_631_, lean_object* v_conts_632_){
_start:
{
lean_object* v_res_633_; 
v_res_633_ = lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0(v_ctorNames_630_, v_a_631_, v_conts_632_);
lean_dec_ref(v_conts_632_);
lean_dec_ref(v_ctorNames_630_);
return v_res_633_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1(lean_object* v_e_647_, lean_object* v___x_648_, lean_object* v_ctorNames_649_, lean_object* v_rcasesPat_650_, lean_object* v___y_651_, lean_object* v___y_652_, lean_object* v___y_653_, lean_object* v___y_654_){
_start:
{
lean_object* v___x_656_; 
v___x_656_ = l_Lean_PrettyPrinter_delab(v_e_647_, v___x_648_, v___y_651_, v___y_652_, v___y_653_, v___y_654_);
if (lean_obj_tag(v___x_656_) == 0)
{
lean_object* v_a_657_; lean_object* v___x_659_; uint8_t v_isShared_660_; uint8_t v_isSharedCheck_688_; 
v_a_657_ = lean_ctor_get(v___x_656_, 0);
v_isSharedCheck_688_ = !lean_is_exclusive(v___x_656_);
if (v_isSharedCheck_688_ == 0)
{
v___x_659_ = v___x_656_;
v_isShared_660_ = v_isSharedCheck_688_;
goto v_resetjp_658_;
}
else
{
lean_inc(v_a_657_);
lean_dec(v___x_656_);
v___x_659_ = lean_box(0);
v_isShared_660_ = v_isSharedCheck_688_;
goto v_resetjp_658_;
}
v_resetjp_658_:
{
lean_object* v_ref_661_; lean_object* v___f_662_; uint8_t v___x_663_; lean_object* v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_686_; 
v_ref_661_ = lean_ctor_get(v___y_653_, 5);
lean_inc(v_a_657_);
lean_inc_ref(v_ctorNames_649_);
v___f_662_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___boxed), 3, 2);
lean_closure_set(v___f_662_, 0, v_ctorNames_649_);
lean_closure_set(v___f_662_, 1, v_a_657_);
v___x_663_ = 0;
v___x_664_ = l_Lean_SourceInfo_fromRef(v_ref_661_, v___x_663_);
v___x_665_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__0));
v___x_666_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__1));
lean_inc_n(v___x_664_, 6);
v___x_667_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_667_, 0, v___x_664_);
lean_ctor_set(v___x_667_, 1, v___x_665_);
v___x_668_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_669_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__4));
v___x_670_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
v___x_671_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_671_, 0, v___x_664_);
lean_ctor_set(v___x_671_, 1, v___x_668_);
lean_ctor_set(v___x_671_, 2, v___x_670_);
v___x_672_ = l_Lean_Syntax_node2(v___x_664_, v___x_669_, v___x_671_, v_a_657_);
v___x_673_ = l_Lean_Syntax_node1(v___x_664_, v___x_668_, v___x_672_);
v___x_674_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__2));
v___x_675_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_675_, 0, v___x_664_);
lean_ctor_set(v___x_675_, 1, v___x_674_);
v___x_676_ = lean_obj_once(&lp_aesop_Aesop_Script_Tactic_skip___closed__0, &lp_aesop_Aesop_Script_Tactic_skip___closed__0_once, _init_lp_aesop_Aesop_Script_Tactic_skip___closed__0);
v___x_677_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___closed__4));
v___x_678_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__5, &lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__5_once, _init_lp_aesop_Aesop_Script_TacticBuilder_cases___lam__0___closed__5);
v___x_679_ = l_Lean_Syntax_node2(v___x_676_, v___x_677_, v_rcasesPat_650_, v___x_678_);
v___x_680_ = l_Lean_Syntax_node2(v___x_664_, v___x_668_, v___x_675_, v___x_679_);
v___x_681_ = l_Lean_Syntax_node3(v___x_664_, v___x_666_, v___x_667_, v___x_673_, v___x_680_);
v___x_682_ = lean_array_get_size(v_ctorNames_649_);
lean_dec_ref(v_ctorNames_649_);
v___x_683_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_683_, 0, v___x_682_);
lean_ctor_set(v___x_683_, 1, v___f_662_);
v___x_684_ = lp_aesop_Aesop_Script_Tactic_structured(v___x_681_, v___x_683_);
if (v_isShared_660_ == 0)
{
lean_ctor_set(v___x_659_, 0, v___x_684_);
v___x_686_ = v___x_659_;
goto v_reusejp_685_;
}
else
{
lean_object* v_reuseFailAlloc_687_; 
v_reuseFailAlloc_687_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_687_, 0, v___x_684_);
v___x_686_ = v_reuseFailAlloc_687_;
goto v_reusejp_685_;
}
v_reusejp_685_:
{
return v___x_686_;
}
}
}
else
{
lean_object* v_a_689_; lean_object* v___x_691_; uint8_t v_isShared_692_; uint8_t v_isSharedCheck_696_; 
lean_dec(v_rcasesPat_650_);
lean_dec_ref(v_ctorNames_649_);
v_a_689_ = lean_ctor_get(v___x_656_, 0);
v_isSharedCheck_696_ = !lean_is_exclusive(v___x_656_);
if (v_isSharedCheck_696_ == 0)
{
v___x_691_ = v___x_656_;
v_isShared_692_ = v_isSharedCheck_696_;
goto v_resetjp_690_;
}
else
{
lean_inc(v_a_689_);
lean_dec(v___x_656_);
v___x_691_ = lean_box(0);
v_isShared_692_ = v_isSharedCheck_696_;
goto v_resetjp_690_;
}
v_resetjp_690_:
{
lean_object* v___x_694_; 
if (v_isShared_692_ == 0)
{
v___x_694_ = v___x_691_;
goto v_reusejp_693_;
}
else
{
lean_object* v_reuseFailAlloc_695_; 
v_reuseFailAlloc_695_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_695_, 0, v_a_689_);
v___x_694_ = v_reuseFailAlloc_695_;
goto v_reusejp_693_;
}
v_reusejp_693_:
{
return v___x_694_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___boxed(lean_object* v_e_697_, lean_object* v___x_698_, lean_object* v_ctorNames_699_, lean_object* v_rcasesPat_700_, lean_object* v___y_701_, lean_object* v___y_702_, lean_object* v___y_703_, lean_object* v___y_704_, lean_object* v___y_705_){
_start:
{
lean_object* v_res_706_; 
v_res_706_ = lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1(v_e_697_, v___x_698_, v_ctorNames_699_, v_rcasesPat_700_, v___y_701_, v___y_702_, v___y_703_, v___y_704_);
lean_dec(v___y_704_);
lean_dec_ref(v___y_703_);
lean_dec(v___y_702_);
lean_dec_ref(v___y_701_);
return v_res_706_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_cases(lean_object* v_goal_707_, lean_object* v_e_708_, lean_object* v_ctorNames_709_, lean_object* v_a_710_, lean_object* v_a_711_, lean_object* v_a_712_, lean_object* v_a_713_){
_start:
{
lean_object* v_rcasesPat_715_; lean_object* v___x_716_; lean_object* v___f_717_; lean_object* v___x_718_; 
lean_inc_ref(v_ctorNames_709_);
v_rcasesPat_715_ = lp_aesop_Aesop_ctorNamesToRCasesPats(v_ctorNames_709_);
v___x_716_ = lean_box(1);
v___f_717_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticBuilder_cases___lam__1___boxed), 9, 4);
lean_closure_set(v___f_717_, 0, v_e_708_);
lean_closure_set(v___f_717_, 1, v___x_716_);
lean_closure_set(v___f_717_, 2, v_ctorNames_709_);
lean_closure_set(v___f_717_, 3, v_rcasesPat_715_);
v___x_718_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___redArg(v_goal_707_, v___f_717_, v_a_710_, v_a_711_, v_a_712_, v_a_713_);
return v___x_718_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_cases___boxed(lean_object* v_goal_719_, lean_object* v_e_720_, lean_object* v_ctorNames_721_, lean_object* v_a_722_, lean_object* v_a_723_, lean_object* v_a_724_, lean_object* v_a_725_, lean_object* v_a_726_){
_start:
{
lean_object* v_res_727_; 
v_res_727_ = lp_aesop_Aesop_Script_TacticBuilder_cases(v_goal_719_, v_e_720_, v_ctorNames_721_, v_a_722_, v_a_723_, v_a_724_, v_a_725_);
lean_dec(v_a_725_);
lean_dec_ref(v_a_724_);
lean_dec(v_a_723_);
lean_dec_ref(v_a_722_);
return v_res_727_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0(lean_object* v_e_740_, lean_object* v___x_741_, lean_object* v_ctorNames_742_, lean_object* v___y_743_, lean_object* v___y_744_, lean_object* v___y_745_, lean_object* v___y_746_){
_start:
{
lean_object* v___x_748_; 
v___x_748_ = l_Lean_PrettyPrinter_delab(v_e_740_, v___x_741_, v___y_743_, v___y_744_, v___y_745_, v___y_746_);
if (lean_obj_tag(v___x_748_) == 0)
{
lean_object* v_a_749_; lean_object* v___x_751_; uint8_t v_isShared_752_; uint8_t v_isSharedCheck_777_; 
v_a_749_ = lean_ctor_get(v___x_748_, 0);
v_isSharedCheck_777_ = !lean_is_exclusive(v___x_748_);
if (v_isSharedCheck_777_ == 0)
{
v___x_751_ = v___x_748_;
v_isShared_752_ = v_isSharedCheck_777_;
goto v_resetjp_750_;
}
else
{
lean_inc(v_a_749_);
lean_dec(v___x_748_);
v___x_751_ = lean_box(0);
v_isShared_752_ = v_isSharedCheck_777_;
goto v_resetjp_750_;
}
v_resetjp_750_:
{
lean_object* v_ref_753_; uint8_t v___x_754_; lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_771_; lean_object* v___x_772_; lean_object* v___x_773_; lean_object* v___x_775_; 
v_ref_753_ = lean_ctor_get(v___y_745_, 5);
v___x_754_ = 0;
v___x_755_ = l_Lean_SourceInfo_fromRef(v_ref_753_, v___x_754_);
v___x_756_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__0));
v___x_757_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__1));
lean_inc_n(v___x_755_, 6);
v___x_758_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_758_, 0, v___x_755_);
lean_ctor_set(v___x_758_, 1, v___x_756_);
v___x_759_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_760_ = lean_obj_once(&lp_aesop_Aesop_Script_Tactic_skip___closed__0, &lp_aesop_Aesop_Script_Tactic_skip___closed__0_once, _init_lp_aesop_Aesop_Script_Tactic_skip___closed__0);
v___x_761_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___closed__3));
v___x_762_ = lp_aesop_Aesop_CtorNames_toRCasesPat(v_ctorNames_742_);
v___x_763_ = l_Lean_Syntax_node1(v___x_760_, v___x_759_, v___x_762_);
v___x_764_ = l_Lean_Syntax_node1(v___x_760_, v___x_761_, v___x_763_);
v___x_765_ = l_Lean_Syntax_node1(v___x_755_, v___x_759_, v___x_764_);
v___x_766_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
v___x_767_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_767_, 0, v___x_755_);
lean_ctor_set(v___x_767_, 1, v___x_759_);
lean_ctor_set(v___x_767_, 2, v___x_766_);
v___x_768_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__15));
v___x_769_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_769_, 0, v___x_755_);
lean_ctor_set(v___x_769_, 1, v___x_768_);
v___x_770_ = l_Lean_Syntax_node1(v___x_755_, v___x_759_, v_a_749_);
v___x_771_ = l_Lean_Syntax_node2(v___x_755_, v___x_759_, v___x_769_, v___x_770_);
v___x_772_ = l_Lean_Syntax_node4(v___x_755_, v___x_757_, v___x_758_, v___x_765_, v___x_767_, v___x_771_);
v___x_773_ = lp_aesop_Aesop_Script_Tactic_unstructured(v___x_772_);
if (v_isShared_752_ == 0)
{
lean_ctor_set(v___x_751_, 0, v___x_773_);
v___x_775_ = v___x_751_;
goto v_reusejp_774_;
}
else
{
lean_object* v_reuseFailAlloc_776_; 
v_reuseFailAlloc_776_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_776_, 0, v___x_773_);
v___x_775_ = v_reuseFailAlloc_776_;
goto v_reusejp_774_;
}
v_reusejp_774_:
{
return v___x_775_;
}
}
}
else
{
lean_object* v_a_778_; lean_object* v___x_780_; uint8_t v_isShared_781_; uint8_t v_isSharedCheck_785_; 
lean_dec_ref(v_ctorNames_742_);
v_a_778_ = lean_ctor_get(v___x_748_, 0);
v_isSharedCheck_785_ = !lean_is_exclusive(v___x_748_);
if (v_isSharedCheck_785_ == 0)
{
v___x_780_ = v___x_748_;
v_isShared_781_ = v_isSharedCheck_785_;
goto v_resetjp_779_;
}
else
{
lean_inc(v_a_778_);
lean_dec(v___x_748_);
v___x_780_ = lean_box(0);
v_isShared_781_ = v_isSharedCheck_785_;
goto v_resetjp_779_;
}
v_resetjp_779_:
{
lean_object* v___x_783_; 
if (v_isShared_781_ == 0)
{
v___x_783_ = v___x_780_;
goto v_reusejp_782_;
}
else
{
lean_object* v_reuseFailAlloc_784_; 
v_reuseFailAlloc_784_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_784_, 0, v_a_778_);
v___x_783_ = v_reuseFailAlloc_784_;
goto v_reusejp_782_;
}
v_reusejp_782_:
{
return v___x_783_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___boxed(lean_object* v_e_786_, lean_object* v___x_787_, lean_object* v_ctorNames_788_, lean_object* v___y_789_, lean_object* v___y_790_, lean_object* v___y_791_, lean_object* v___y_792_, lean_object* v___y_793_){
_start:
{
lean_object* v_res_794_; 
v_res_794_ = lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0(v_e_786_, v___x_787_, v_ctorNames_788_, v___y_789_, v___y_790_, v___y_791_, v___y_792_);
lean_dec(v___y_792_);
lean_dec_ref(v___y_791_);
lean_dec(v___y_790_);
lean_dec_ref(v___y_789_);
return v_res_794_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_obtain(lean_object* v_goal_795_, lean_object* v_e_796_, lean_object* v_ctorNames_797_, lean_object* v_a_798_, lean_object* v_a_799_, lean_object* v_a_800_, lean_object* v_a_801_){
_start:
{
lean_object* v___x_803_; lean_object* v___f_804_; lean_object* v___x_805_; 
v___x_803_ = lean_box(1);
v___f_804_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticBuilder_obtain___lam__0___boxed), 8, 3);
lean_closure_set(v___f_804_, 0, v_e_796_);
lean_closure_set(v___f_804_, 1, v___x_803_);
lean_closure_set(v___f_804_, 2, v_ctorNames_797_);
v___x_805_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___redArg(v_goal_795_, v___f_804_, v_a_798_, v_a_799_, v_a_800_, v_a_801_);
return v___x_805_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_obtain___boxed(lean_object* v_goal_806_, lean_object* v_e_807_, lean_object* v_ctorNames_808_, lean_object* v_a_809_, lean_object* v_a_810_, lean_object* v_a_811_, lean_object* v_a_812_, lean_object* v_a_813_){
_start:
{
lean_object* v_res_814_; 
v_res_814_ = lp_aesop_Aesop_Script_TacticBuilder_obtain(v_goal_806_, v_e_807_, v_ctorNames_808_, v_a_809_, v_a_810_, v_a_811_, v_a_812_);
lean_dec(v_a_812_);
lean_dec_ref(v_a_811_);
lean_dec(v_a_810_);
lean_dec_ref(v_a_809_);
return v_res_814_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_casesOrObtain(lean_object* v_goal_815_, lean_object* v_e_816_, lean_object* v_ctorNames_817_, lean_object* v_a_818_, lean_object* v_a_819_, lean_object* v_a_820_, lean_object* v_a_821_){
_start:
{
lean_object* v___x_823_; lean_object* v___x_824_; uint8_t v___x_825_; 
v___x_823_ = lean_array_get_size(v_ctorNames_817_);
v___x_824_ = lean_unsigned_to_nat(1u);
v___x_825_ = lean_nat_dec_eq(v___x_823_, v___x_824_);
if (v___x_825_ == 0)
{
lean_object* v___x_826_; 
v___x_826_ = lp_aesop_Aesop_Script_TacticBuilder_cases(v_goal_815_, v_e_816_, v_ctorNames_817_, v_a_818_, v_a_819_, v_a_820_, v_a_821_);
return v___x_826_;
}
else
{
lean_object* v___x_827_; lean_object* v___x_828_; lean_object* v___x_829_; 
v___x_827_ = lean_unsigned_to_nat(0u);
v___x_828_ = lean_array_fget(v_ctorNames_817_, v___x_827_);
lean_dec_ref(v_ctorNames_817_);
v___x_829_ = lp_aesop_Aesop_Script_TacticBuilder_obtain(v_goal_815_, v_e_816_, v___x_828_, v_a_818_, v_a_819_, v_a_820_, v_a_821_);
return v___x_829_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_casesOrObtain___boxed(lean_object* v_goal_830_, lean_object* v_e_831_, lean_object* v_ctorNames_832_, lean_object* v_a_833_, lean_object* v_a_834_, lean_object* v_a_835_, lean_object* v_a_836_, lean_object* v_a_837_){
_start:
{
lean_object* v_res_838_; 
v_res_838_ = lp_aesop_Aesop_Script_TacticBuilder_casesOrObtain(v_goal_830_, v_e_831_, v_ctorNames_832_, v_a_833_, v_a_834_, v_a_835_, v_a_836_);
lean_dec(v_a_836_);
lean_dec_ref(v_a_835_);
lean_dec(v_a_834_);
lean_dec_ref(v_a_833_);
return v_res_838_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_renameInaccessibleFVars_spec__0___redArg(lean_object* v___x_843_, size_t v_sz_844_, size_t v_i_845_, lean_object* v_bs_846_, lean_object* v___y_847_, lean_object* v___y_848_, lean_object* v___y_849_){
_start:
{
uint8_t v___x_851_; 
v___x_851_ = lean_usize_dec_lt(v_i_845_, v_sz_844_);
if (v___x_851_ == 0)
{
lean_object* v___x_852_; 
v___x_852_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_852_, 0, v_bs_846_);
return v___x_852_;
}
else
{
lean_object* v_v_853_; lean_object* v___x_854_; 
v_v_853_ = lean_array_uget_borrowed(v_bs_846_, v_i_845_);
lean_inc(v_v_853_);
v___x_854_ = l_Lean_FVarId_getDecl___redArg(v_v_853_, v___y_847_, v___y_848_, v___y_849_);
if (lean_obj_tag(v___x_854_) == 0)
{
lean_object* v_a_855_; lean_object* v_ref_856_; lean_object* v___x_857_; uint8_t v___x_858_; lean_object* v_bs_x27_859_; lean_object* v___x_860_; lean_object* v___x_861_; lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_864_; size_t v___x_865_; size_t v___x_866_; lean_object* v___x_867_; 
v_a_855_ = lean_ctor_get(v___x_854_, 0);
lean_inc(v_a_855_);
lean_dec_ref_known(v___x_854_, 1);
v_ref_856_ = lean_ctor_get(v___y_848_, 5);
v___x_857_ = lean_unsigned_to_nat(0u);
v___x_858_ = lean_nat_dec_eq(v___x_843_, v___x_857_);
v_bs_x27_859_ = lean_array_uset(v_bs_846_, v_i_845_, v___x_857_);
v___x_860_ = l_Lean_LocalDecl_userName(v_a_855_);
lean_dec(v_a_855_);
v___x_861_ = l_Lean_mkIdent(v___x_860_);
v___x_862_ = l_Lean_SourceInfo_fromRef(v_ref_856_, v___x_858_);
v___x_863_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_renameInaccessibleFVars_spec__0___redArg___closed__1));
v___x_864_ = l_Lean_Syntax_node1(v___x_862_, v___x_863_, v___x_861_);
v___x_865_ = ((size_t)1ULL);
v___x_866_ = lean_usize_add(v_i_845_, v___x_865_);
v___x_867_ = lean_array_uset(v_bs_x27_859_, v_i_845_, v___x_864_);
v_i_845_ = v___x_866_;
v_bs_846_ = v___x_867_;
goto _start;
}
else
{
lean_object* v_a_869_; lean_object* v___x_871_; uint8_t v_isShared_872_; uint8_t v_isSharedCheck_876_; 
lean_dec_ref(v_bs_846_);
v_a_869_ = lean_ctor_get(v___x_854_, 0);
v_isSharedCheck_876_ = !lean_is_exclusive(v___x_854_);
if (v_isSharedCheck_876_ == 0)
{
v___x_871_ = v___x_854_;
v_isShared_872_ = v_isSharedCheck_876_;
goto v_resetjp_870_;
}
else
{
lean_inc(v_a_869_);
lean_dec(v___x_854_);
v___x_871_ = lean_box(0);
v_isShared_872_ = v_isSharedCheck_876_;
goto v_resetjp_870_;
}
v_resetjp_870_:
{
lean_object* v___x_874_; 
if (v_isShared_872_ == 0)
{
v___x_874_ = v___x_871_;
goto v_reusejp_873_;
}
else
{
lean_object* v_reuseFailAlloc_875_; 
v_reuseFailAlloc_875_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_875_, 0, v_a_869_);
v___x_874_ = v_reuseFailAlloc_875_;
goto v_reusejp_873_;
}
v_reusejp_873_:
{
return v___x_874_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_renameInaccessibleFVars_spec__0___redArg___boxed(lean_object* v___x_877_, lean_object* v_sz_878_, lean_object* v_i_879_, lean_object* v_bs_880_, lean_object* v___y_881_, lean_object* v___y_882_, lean_object* v___y_883_, lean_object* v___y_884_){
_start:
{
size_t v_sz_boxed_885_; size_t v_i_boxed_886_; lean_object* v_res_887_; 
v_sz_boxed_885_ = lean_unbox_usize(v_sz_878_);
lean_dec(v_sz_878_);
v_i_boxed_886_ = lean_unbox_usize(v_i_879_);
lean_dec(v_i_879_);
v_res_887_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_renameInaccessibleFVars_spec__0___redArg(v___x_877_, v_sz_boxed_885_, v_i_boxed_886_, v_bs_880_, v___y_881_, v___y_882_, v___y_883_);
lean_dec(v___y_883_);
lean_dec_ref(v___y_882_);
lean_dec_ref(v___y_881_);
lean_dec(v___x_877_);
return v_res_887_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___lam__0(lean_object* v_renamedFVars_895_, lean_object* v___x_896_, uint8_t v___x_897_, lean_object* v___y_898_, lean_object* v___y_899_, lean_object* v___y_900_, lean_object* v___y_901_){
_start:
{
size_t v_sz_903_; size_t v___x_904_; lean_object* v___x_905_; 
v_sz_903_ = lean_array_size(v_renamedFVars_895_);
v___x_904_ = ((size_t)0ULL);
v___x_905_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_renameInaccessibleFVars_spec__0___redArg(v___x_896_, v_sz_903_, v___x_904_, v_renamedFVars_895_, v___y_898_, v___y_900_, v___y_901_);
if (lean_obj_tag(v___x_905_) == 0)
{
lean_object* v_a_906_; lean_object* v___x_908_; uint8_t v_isShared_909_; uint8_t v_isSharedCheck_924_; 
v_a_906_ = lean_ctor_get(v___x_905_, 0);
v_isSharedCheck_924_ = !lean_is_exclusive(v___x_905_);
if (v_isSharedCheck_924_ == 0)
{
v___x_908_ = v___x_905_;
v_isShared_909_ = v_isSharedCheck_924_;
goto v_resetjp_907_;
}
else
{
lean_inc(v_a_906_);
lean_dec(v___x_905_);
v___x_908_ = lean_box(0);
v_isShared_909_ = v_isSharedCheck_924_;
goto v_resetjp_907_;
}
v_resetjp_907_:
{
lean_object* v_ref_910_; lean_object* v___x_911_; lean_object* v___x_912_; lean_object* v___x_913_; lean_object* v___x_914_; lean_object* v___x_915_; lean_object* v___x_916_; lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v___x_922_; 
v_ref_910_ = lean_ctor_get(v___y_900_, 5);
v___x_911_ = l_Lean_SourceInfo_fromRef(v_ref_910_, v___x_897_);
v___x_912_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___lam__0___closed__1));
v___x_913_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___lam__0___closed__2));
lean_inc_n(v___x_911_, 2);
v___x_914_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_914_, 0, v___x_911_);
lean_ctor_set(v___x_914_, 1, v___x_913_);
v___x_915_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_916_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
v___x_917_ = l_Array_append___redArg(v___x_916_, v_a_906_);
lean_dec(v_a_906_);
v___x_918_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_918_, 0, v___x_911_);
lean_ctor_set(v___x_918_, 1, v___x_915_);
lean_ctor_set(v___x_918_, 2, v___x_917_);
v___x_919_ = l_Lean_Syntax_node2(v___x_911_, v___x_912_, v___x_914_, v___x_918_);
v___x_920_ = lp_aesop_Aesop_Script_Tactic_unstructured(v___x_919_);
if (v_isShared_909_ == 0)
{
lean_ctor_set(v___x_908_, 0, v___x_920_);
v___x_922_ = v___x_908_;
goto v_reusejp_921_;
}
else
{
lean_object* v_reuseFailAlloc_923_; 
v_reuseFailAlloc_923_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_923_, 0, v___x_920_);
v___x_922_ = v_reuseFailAlloc_923_;
goto v_reusejp_921_;
}
v_reusejp_921_:
{
return v___x_922_;
}
}
}
else
{
lean_object* v_a_925_; lean_object* v___x_927_; uint8_t v_isShared_928_; uint8_t v_isSharedCheck_932_; 
v_a_925_ = lean_ctor_get(v___x_905_, 0);
v_isSharedCheck_932_ = !lean_is_exclusive(v___x_905_);
if (v_isSharedCheck_932_ == 0)
{
v___x_927_ = v___x_905_;
v_isShared_928_ = v_isSharedCheck_932_;
goto v_resetjp_926_;
}
else
{
lean_inc(v_a_925_);
lean_dec(v___x_905_);
v___x_927_ = lean_box(0);
v_isShared_928_ = v_isSharedCheck_932_;
goto v_resetjp_926_;
}
v_resetjp_926_:
{
lean_object* v___x_930_; 
if (v_isShared_928_ == 0)
{
v___x_930_ = v___x_927_;
goto v_reusejp_929_;
}
else
{
lean_object* v_reuseFailAlloc_931_; 
v_reuseFailAlloc_931_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_931_, 0, v_a_925_);
v___x_930_ = v_reuseFailAlloc_931_;
goto v_reusejp_929_;
}
v_reusejp_929_:
{
return v___x_930_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___lam__0___boxed(lean_object* v_renamedFVars_933_, lean_object* v___x_934_, lean_object* v___x_935_, lean_object* v___y_936_, lean_object* v___y_937_, lean_object* v___y_938_, lean_object* v___y_939_, lean_object* v___y_940_){
_start:
{
uint8_t v___x_3008__boxed_941_; lean_object* v_res_942_; 
v___x_3008__boxed_941_ = lean_unbox(v___x_935_);
v_res_942_ = lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___lam__0(v_renamedFVars_933_, v___x_934_, v___x_3008__boxed_941_, v___y_936_, v___y_937_, v___y_938_, v___y_939_);
lean_dec(v___y_939_);
lean_dec_ref(v___y_938_);
lean_dec(v___y_937_);
lean_dec_ref(v___y_936_);
lean_dec(v___x_934_);
return v_res_942_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars(lean_object* v_postGoal_943_, lean_object* v_renamedFVars_944_, lean_object* v_a_945_, lean_object* v_a_946_, lean_object* v_a_947_, lean_object* v_a_948_){
_start:
{
lean_object* v___x_950_; lean_object* v___x_951_; uint8_t v___x_952_; 
v___x_950_ = lean_array_get_size(v_renamedFVars_944_);
v___x_951_ = lean_unsigned_to_nat(0u);
v___x_952_ = lean_nat_dec_eq(v___x_950_, v___x_951_);
if (v___x_952_ == 0)
{
lean_object* v___x_953_; lean_object* v___f_954_; lean_object* v___x_955_; 
v___x_953_ = lean_box(v___x_952_);
v___f_954_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___lam__0___boxed), 8, 3);
lean_closure_set(v___f_954_, 0, v_renamedFVars_944_);
lean_closure_set(v___f_954_, 1, v___x_950_);
lean_closure_set(v___f_954_, 2, v___x_953_);
v___x_955_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___redArg(v_postGoal_943_, v___f_954_, v_a_945_, v_a_946_, v_a_947_, v_a_948_);
return v___x_955_;
}
else
{
lean_object* v___x_956_; lean_object* v___x_957_; 
lean_dec_ref(v_renamedFVars_944_);
lean_dec(v_postGoal_943_);
v___x_956_ = lp_aesop_Aesop_Script_Tactic_skip;
v___x_957_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_957_, 0, v___x_956_);
return v___x_957_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars___boxed(lean_object* v_postGoal_958_, lean_object* v_renamedFVars_959_, lean_object* v_a_960_, lean_object* v_a_961_, lean_object* v_a_962_, lean_object* v_a_963_, lean_object* v_a_964_){
_start:
{
lean_object* v_res_965_; 
v_res_965_ = lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars(v_postGoal_958_, v_renamedFVars_959_, v_a_960_, v_a_961_, v_a_962_, v_a_963_);
lean_dec(v_a_963_);
lean_dec_ref(v_a_962_);
lean_dec(v_a_961_);
lean_dec_ref(v_a_960_);
return v_res_965_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_renameInaccessibleFVars_spec__0(lean_object* v___x_966_, size_t v_sz_967_, size_t v_i_968_, lean_object* v_bs_969_, lean_object* v___y_970_, lean_object* v___y_971_, lean_object* v___y_972_, lean_object* v___y_973_){
_start:
{
lean_object* v___x_975_; 
v___x_975_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_renameInaccessibleFVars_spec__0___redArg(v___x_966_, v_sz_967_, v_i_968_, v_bs_969_, v___y_970_, v___y_972_, v___y_973_);
return v___x_975_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_renameInaccessibleFVars_spec__0___boxed(lean_object* v___x_976_, lean_object* v_sz_977_, lean_object* v_i_978_, lean_object* v_bs_979_, lean_object* v___y_980_, lean_object* v___y_981_, lean_object* v___y_982_, lean_object* v___y_983_, lean_object* v___y_984_){
_start:
{
size_t v_sz_boxed_985_; size_t v_i_boxed_986_; lean_object* v_res_987_; 
v_sz_boxed_985_ = lean_unbox_usize(v_sz_977_);
lean_dec(v_sz_977_);
v_i_boxed_986_ = lean_unbox_usize(v_i_978_);
lean_dec(v_i_978_);
v_res_987_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_renameInaccessibleFVars_spec__0(v___x_976_, v_sz_boxed_985_, v_i_boxed_986_, v_bs_979_, v___y_980_, v___y_981_, v___y_982_, v___y_983_);
lean_dec(v___y_983_);
lean_dec_ref(v___y_982_);
lean_dec(v___y_981_);
lean_dec_ref(v___y_980_);
lean_dec(v___x_976_);
return v_res_987_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_unfold_spec__0(size_t v_sz_988_, size_t v_i_989_, lean_object* v_bs_990_){
_start:
{
uint8_t v___x_991_; 
v___x_991_ = lean_usize_dec_lt(v_i_989_, v_sz_988_);
if (v___x_991_ == 0)
{
return v_bs_990_;
}
else
{
lean_object* v_v_992_; lean_object* v___x_993_; lean_object* v_bs_x27_994_; lean_object* v___x_995_; size_t v___x_996_; size_t v___x_997_; lean_object* v___x_998_; 
v_v_992_ = lean_array_uget(v_bs_990_, v_i_989_);
v___x_993_ = lean_unsigned_to_nat(0u);
v_bs_x27_994_ = lean_array_uset(v_bs_990_, v_i_989_, v___x_993_);
v___x_995_ = l_Lean_mkIdent(v_v_992_);
v___x_996_ = ((size_t)1ULL);
v___x_997_ = lean_usize_add(v_i_989_, v___x_996_);
v___x_998_ = lean_array_uset(v_bs_x27_994_, v_i_989_, v___x_995_);
v_i_989_ = v___x_997_;
v_bs_990_ = v___x_998_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_unfold_spec__0___boxed(lean_object* v_sz_1000_, lean_object* v_i_1001_, lean_object* v_bs_1002_){
_start:
{
size_t v_sz_boxed_1003_; size_t v_i_boxed_1004_; lean_object* v_res_1005_; 
v_sz_boxed_1003_ = lean_unbox_usize(v_sz_1000_);
lean_dec(v_sz_1000_);
v_i_boxed_1004_ = lean_unbox_usize(v_i_1001_);
lean_dec(v_i_1001_);
v_res_1005_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_unfold_spec__0(v_sz_boxed_1003_, v_i_boxed_1004_, v_bs_1002_);
return v_res_1005_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg(lean_object* v_usedDecls_1018_, uint8_t v_aesopUnfold_1019_, lean_object* v_a_1020_){
_start:
{
lean_object* v_tac_1023_; size_t v_sz_1026_; size_t v___x_1027_; lean_object* v_usedDecls_1028_; 
v_sz_1026_ = lean_array_size(v_usedDecls_1018_);
v___x_1027_ = ((size_t)0ULL);
v_usedDecls_1028_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_unfold_spec__0(v_sz_1026_, v___x_1027_, v_usedDecls_1018_);
if (v_aesopUnfold_1019_ == 0)
{
lean_object* v_ref_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; lean_object* v___x_1032_; lean_object* v___x_1033_; lean_object* v___x_1034_; lean_object* v___x_1035_; lean_object* v___x_1036_; lean_object* v___x_1037_; lean_object* v___x_1038_; lean_object* v___x_1039_; 
v_ref_1029_ = lean_ctor_get(v_a_1020_, 5);
v___x_1030_ = l_Lean_SourceInfo_fromRef(v_ref_1029_, v_aesopUnfold_1019_);
v___x_1031_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__0));
v___x_1032_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__1));
lean_inc_n(v___x_1030_, 3);
v___x_1033_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1033_, 0, v___x_1030_);
lean_ctor_set(v___x_1033_, 1, v___x_1031_);
v___x_1034_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_1035_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
v___x_1036_ = l_Array_append___redArg(v___x_1035_, v_usedDecls_1028_);
lean_dec_ref(v_usedDecls_1028_);
v___x_1037_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1037_, 0, v___x_1030_);
lean_ctor_set(v___x_1037_, 1, v___x_1034_);
lean_ctor_set(v___x_1037_, 2, v___x_1036_);
v___x_1038_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1038_, 0, v___x_1030_);
lean_ctor_set(v___x_1038_, 1, v___x_1034_);
lean_ctor_set(v___x_1038_, 2, v___x_1035_);
v___x_1039_ = l_Lean_Syntax_node3(v___x_1030_, v___x_1032_, v___x_1033_, v___x_1037_, v___x_1038_);
v_tac_1023_ = v___x_1039_;
goto v___jp_1022_;
}
else
{
lean_object* v_ref_1040_; uint8_t v___x_1041_; lean_object* v___x_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; lean_object* v___x_1047_; lean_object* v___x_1048_; lean_object* v___x_1049_; lean_object* v___x_1050_; 
v_ref_1040_ = lean_ctor_get(v_a_1020_, 5);
v___x_1041_ = 0;
v___x_1042_ = l_Lean_SourceInfo_fromRef(v_ref_1040_, v___x_1041_);
v___x_1043_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__4));
v___x_1044_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__5));
lean_inc_n(v___x_1042_, 2);
v___x_1045_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1045_, 0, v___x_1042_);
lean_ctor_set(v___x_1045_, 1, v___x_1044_);
v___x_1046_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_1047_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
v___x_1048_ = l_Array_append___redArg(v___x_1047_, v_usedDecls_1028_);
lean_dec_ref(v_usedDecls_1028_);
v___x_1049_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1049_, 0, v___x_1042_);
lean_ctor_set(v___x_1049_, 1, v___x_1046_);
lean_ctor_set(v___x_1049_, 2, v___x_1048_);
v___x_1050_ = l_Lean_Syntax_node2(v___x_1042_, v___x_1043_, v___x_1045_, v___x_1049_);
v_tac_1023_ = v___x_1050_;
goto v___jp_1022_;
}
v___jp_1022_:
{
lean_object* v___x_1024_; lean_object* v___x_1025_; 
v___x_1024_ = lp_aesop_Aesop_Script_Tactic_unstructured(v_tac_1023_);
v___x_1025_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1025_, 0, v___x_1024_);
return v___x_1025_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___boxed(lean_object* v_usedDecls_1051_, lean_object* v_aesopUnfold_1052_, lean_object* v_a_1053_, lean_object* v_a_1054_){
_start:
{
uint8_t v_aesopUnfold_boxed_1055_; lean_object* v_res_1056_; 
v_aesopUnfold_boxed_1055_ = lean_unbox(v_aesopUnfold_1052_);
v_res_1056_ = lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg(v_usedDecls_1051_, v_aesopUnfold_boxed_1055_, v_a_1053_);
lean_dec_ref(v_a_1053_);
return v_res_1056_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfold(lean_object* v_usedDecls_1057_, uint8_t v_aesopUnfold_1058_, lean_object* v_a_1059_, lean_object* v_a_1060_, lean_object* v_a_1061_, lean_object* v_a_1062_){
_start:
{
lean_object* v___x_1064_; 
v___x_1064_ = lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg(v_usedDecls_1057_, v_aesopUnfold_1058_, v_a_1061_);
return v___x_1064_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfold___boxed(lean_object* v_usedDecls_1065_, lean_object* v_aesopUnfold_1066_, lean_object* v_a_1067_, lean_object* v_a_1068_, lean_object* v_a_1069_, lean_object* v_a_1070_, lean_object* v_a_1071_){
_start:
{
uint8_t v_aesopUnfold_boxed_1072_; lean_object* v_res_1073_; 
v_aesopUnfold_boxed_1072_ = lean_unbox(v_aesopUnfold_1066_);
v_res_1073_ = lp_aesop_Aesop_Script_TacticBuilder_unfold(v_usedDecls_1065_, v_aesopUnfold_boxed_1072_, v_a_1067_, v_a_1068_, v_a_1069_, v_a_1070_);
lean_dec(v_a_1070_);
lean_dec_ref(v_a_1069_);
lean_dec(v_a_1068_);
lean_dec_ref(v_a_1067_);
return v_res_1073_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0(lean_object* v_goal_1091_, lean_object* v___x_1092_, lean_object* v_usedDecls_1093_, uint8_t v_aesopUnfold_1094_, lean_object* v___y_1095_, lean_object* v___y_1096_, lean_object* v___y_1097_, lean_object* v___y_1098_){
_start:
{
lean_object* v___x_1100_; 
v___x_1100_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___redArg(v_goal_1091_, v___x_1092_, v___y_1095_, v___y_1096_, v___y_1097_, v___y_1098_);
if (lean_obj_tag(v___x_1100_) == 0)
{
lean_object* v_a_1101_; lean_object* v___x_1103_; uint8_t v_isShared_1104_; uint8_t v_isSharedCheck_1147_; 
v_a_1101_ = lean_ctor_get(v___x_1100_, 0);
v_isSharedCheck_1147_ = !lean_is_exclusive(v___x_1100_);
if (v_isSharedCheck_1147_ == 0)
{
v___x_1103_ = v___x_1100_;
v_isShared_1104_ = v_isSharedCheck_1147_;
goto v_resetjp_1102_;
}
else
{
lean_inc(v_a_1101_);
lean_dec(v___x_1100_);
v___x_1103_ = lean_box(0);
v_isShared_1104_ = v_isSharedCheck_1147_;
goto v_resetjp_1102_;
}
v_resetjp_1102_:
{
lean_object* v_tac_1106_; lean_object* v___x_1111_; size_t v_sz_1112_; size_t v___x_1113_; lean_object* v___x_1114_; 
v___x_1111_ = l_Lean_mkIdent(v_a_1101_);
v_sz_1112_ = lean_array_size(v_usedDecls_1093_);
v___x_1113_ = ((size_t)0ULL);
v___x_1114_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_unfold_spec__0(v_sz_1112_, v___x_1113_, v_usedDecls_1093_);
if (v_aesopUnfold_1094_ == 0)
{
lean_object* v_ref_1115_; lean_object* v___x_1116_; lean_object* v___x_1117_; lean_object* v___x_1118_; lean_object* v___x_1119_; lean_object* v___x_1120_; lean_object* v___x_1121_; lean_object* v___x_1122_; lean_object* v___x_1123_; lean_object* v___x_1124_; lean_object* v___x_1125_; lean_object* v___x_1126_; lean_object* v___x_1127_; lean_object* v___x_1128_; lean_object* v___x_1129_; lean_object* v___x_1130_; lean_object* v___x_1131_; lean_object* v___x_1132_; 
v_ref_1115_ = lean_ctor_get(v___y_1097_, 5);
v___x_1116_ = l_Lean_SourceInfo_fromRef(v_ref_1115_, v_aesopUnfold_1094_);
v___x_1117_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__0));
v___x_1118_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__1));
lean_inc_n(v___x_1116_, 7);
v___x_1119_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1119_, 0, v___x_1116_);
lean_ctor_set(v___x_1119_, 1, v___x_1117_);
v___x_1120_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_1121_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
v___x_1122_ = l_Array_append___redArg(v___x_1121_, v___x_1114_);
lean_dec_ref(v___x_1114_);
v___x_1123_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1123_, 0, v___x_1116_);
lean_ctor_set(v___x_1123_, 1, v___x_1120_);
lean_ctor_set(v___x_1123_, 2, v___x_1122_);
v___x_1124_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__1));
v___x_1125_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__2));
v___x_1126_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1126_, 0, v___x_1116_);
lean_ctor_set(v___x_1126_, 1, v___x_1125_);
v___x_1127_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__4));
v___x_1128_ = l_Lean_Syntax_node1(v___x_1116_, v___x_1120_, v___x_1111_);
v___x_1129_ = l_Lean_Syntax_node1(v___x_1116_, v___x_1127_, v___x_1128_);
v___x_1130_ = l_Lean_Syntax_node2(v___x_1116_, v___x_1124_, v___x_1126_, v___x_1129_);
v___x_1131_ = l_Lean_Syntax_node1(v___x_1116_, v___x_1120_, v___x_1130_);
v___x_1132_ = l_Lean_Syntax_node3(v___x_1116_, v___x_1118_, v___x_1119_, v___x_1123_, v___x_1131_);
v_tac_1106_ = v___x_1132_;
goto v___jp_1105_;
}
else
{
lean_object* v_ref_1133_; uint8_t v___x_1134_; lean_object* v___x_1135_; lean_object* v___x_1136_; lean_object* v___x_1137_; lean_object* v___x_1138_; lean_object* v___x_1139_; lean_object* v___x_1140_; lean_object* v___x_1141_; lean_object* v___x_1142_; lean_object* v___x_1143_; lean_object* v___x_1144_; lean_object* v___x_1145_; lean_object* v___x_1146_; 
v_ref_1133_ = lean_ctor_get(v___y_1097_, 5);
v___x_1134_ = 0;
v___x_1135_ = l_Lean_SourceInfo_fromRef(v_ref_1133_, v___x_1134_);
v___x_1136_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__6));
v___x_1137_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfold___redArg___closed__5));
lean_inc_n(v___x_1135_, 4);
v___x_1138_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1138_, 0, v___x_1135_);
lean_ctor_set(v___x_1138_, 1, v___x_1137_);
v___x_1139_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_1140_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
v___x_1141_ = l_Array_append___redArg(v___x_1140_, v___x_1114_);
lean_dec_ref(v___x_1114_);
v___x_1142_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1142_, 0, v___x_1135_);
lean_ctor_set(v___x_1142_, 1, v___x_1139_);
lean_ctor_set(v___x_1142_, 2, v___x_1141_);
v___x_1143_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__2));
v___x_1144_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1144_, 0, v___x_1135_);
lean_ctor_set(v___x_1144_, 1, v___x_1143_);
v___x_1145_ = l_Lean_Syntax_node1(v___x_1135_, v___x_1139_, v___x_1111_);
v___x_1146_ = l_Lean_Syntax_node4(v___x_1135_, v___x_1136_, v___x_1138_, v___x_1142_, v___x_1144_, v___x_1145_);
v_tac_1106_ = v___x_1146_;
goto v___jp_1105_;
}
v___jp_1105_:
{
lean_object* v___x_1107_; lean_object* v___x_1109_; 
v___x_1107_ = lp_aesop_Aesop_Script_Tactic_unstructured(v_tac_1106_);
if (v_isShared_1104_ == 0)
{
lean_ctor_set(v___x_1103_, 0, v___x_1107_);
v___x_1109_ = v___x_1103_;
goto v_reusejp_1108_;
}
else
{
lean_object* v_reuseFailAlloc_1110_; 
v_reuseFailAlloc_1110_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1110_, 0, v___x_1107_);
v___x_1109_ = v_reuseFailAlloc_1110_;
goto v_reusejp_1108_;
}
v_reusejp_1108_:
{
return v___x_1109_;
}
}
}
}
else
{
lean_object* v_a_1148_; lean_object* v___x_1150_; uint8_t v_isShared_1151_; uint8_t v_isSharedCheck_1155_; 
lean_dec_ref(v_usedDecls_1093_);
v_a_1148_ = lean_ctor_get(v___x_1100_, 0);
v_isSharedCheck_1155_ = !lean_is_exclusive(v___x_1100_);
if (v_isSharedCheck_1155_ == 0)
{
v___x_1150_ = v___x_1100_;
v_isShared_1151_ = v_isSharedCheck_1155_;
goto v_resetjp_1149_;
}
else
{
lean_inc(v_a_1148_);
lean_dec(v___x_1100_);
v___x_1150_ = lean_box(0);
v_isShared_1151_ = v_isSharedCheck_1155_;
goto v_resetjp_1149_;
}
v_resetjp_1149_:
{
lean_object* v___x_1153_; 
if (v_isShared_1151_ == 0)
{
v___x_1153_ = v___x_1150_;
goto v_reusejp_1152_;
}
else
{
lean_object* v_reuseFailAlloc_1154_; 
v_reuseFailAlloc_1154_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1154_, 0, v_a_1148_);
v___x_1153_ = v_reuseFailAlloc_1154_;
goto v_reusejp_1152_;
}
v_reusejp_1152_:
{
return v___x_1153_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___boxed(lean_object* v_goal_1156_, lean_object* v___x_1157_, lean_object* v_usedDecls_1158_, lean_object* v_aesopUnfold_1159_, lean_object* v___y_1160_, lean_object* v___y_1161_, lean_object* v___y_1162_, lean_object* v___y_1163_, lean_object* v___y_1164_){
_start:
{
uint8_t v_aesopUnfold_boxed_1165_; lean_object* v_res_1166_; 
v_aesopUnfold_boxed_1165_ = lean_unbox(v_aesopUnfold_1159_);
v_res_1166_ = lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0(v_goal_1156_, v___x_1157_, v_usedDecls_1158_, v_aesopUnfold_boxed_1165_, v___y_1160_, v___y_1161_, v___y_1162_, v___y_1163_);
lean_dec(v___y_1163_);
lean_dec_ref(v___y_1162_);
lean_dec(v___y_1161_);
lean_dec_ref(v___y_1160_);
return v_res_1166_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfoldAt(lean_object* v_goal_1167_, lean_object* v_fvarId_1168_, lean_object* v_usedDecls_1169_, uint8_t v_aesopUnfold_1170_, lean_object* v_a_1171_, lean_object* v_a_1172_, lean_object* v_a_1173_, lean_object* v_a_1174_){
_start:
{
lean_object* v___x_1176_; lean_object* v___x_1177_; lean_object* v___f_1178_; lean_object* v___x_1179_; 
v___x_1176_ = lean_alloc_closure((void*)(l_Lean_FVarId_getUserName___boxed), 6, 1);
lean_closure_set(v___x_1176_, 0, v_fvarId_1168_);
v___x_1177_ = lean_box(v_aesopUnfold_1170_);
lean_inc(v_goal_1167_);
v___f_1178_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___boxed), 9, 4);
lean_closure_set(v___f_1178_, 0, v_goal_1167_);
lean_closure_set(v___f_1178_, 1, v___x_1176_);
lean_closure_set(v___f_1178_, 2, v_usedDecls_1169_);
lean_closure_set(v___f_1178_, 3, v___x_1177_);
v___x_1179_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___redArg(v_goal_1167_, v___f_1178_, v_a_1171_, v_a_1172_, v_a_1173_, v_a_1174_);
return v___x_1179_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___boxed(lean_object* v_goal_1180_, lean_object* v_fvarId_1181_, lean_object* v_usedDecls_1182_, lean_object* v_aesopUnfold_1183_, lean_object* v_a_1184_, lean_object* v_a_1185_, lean_object* v_a_1186_, lean_object* v_a_1187_, lean_object* v_a_1188_){
_start:
{
uint8_t v_aesopUnfold_boxed_1189_; lean_object* v_res_1190_; 
v_aesopUnfold_boxed_1189_ = lean_unbox(v_aesopUnfold_1183_);
v_res_1190_ = lp_aesop_Aesop_Script_TacticBuilder_unfoldAt(v_goal_1180_, v_fvarId_1181_, v_usedDecls_1182_, v_aesopUnfold_boxed_1189_, v_a_1184_, v_a_1185_, v_a_1186_, v_a_1187_);
lean_dec(v_a_1187_);
lean_dec_ref(v_a_1186_);
lean_dec(v_a_1185_);
lean_dec_ref(v_a_1184_);
return v_res_1190_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg(lean_object* v_fvarId_1206_, lean_object* v_a_1207_, lean_object* v_a_1208_, lean_object* v_a_1209_){
_start:
{
lean_object* v___x_1211_; 
v___x_1211_ = l_Lean_FVarId_getUserName___redArg(v_fvarId_1206_, v_a_1207_, v_a_1208_, v_a_1209_);
if (lean_obj_tag(v___x_1211_) == 0)
{
lean_object* v_a_1212_; lean_object* v___x_1214_; uint8_t v_isShared_1215_; uint8_t v_isSharedCheck_1227_; 
v_a_1212_ = lean_ctor_get(v___x_1211_, 0);
v_isSharedCheck_1227_ = !lean_is_exclusive(v___x_1211_);
if (v_isSharedCheck_1227_ == 0)
{
v___x_1214_ = v___x_1211_;
v_isShared_1215_ = v_isSharedCheck_1227_;
goto v_resetjp_1213_;
}
else
{
lean_inc(v_a_1212_);
lean_dec(v___x_1211_);
v___x_1214_ = lean_box(0);
v_isShared_1215_ = v_isSharedCheck_1227_;
goto v_resetjp_1213_;
}
v_resetjp_1213_:
{
lean_object* v_ref_1216_; uint8_t v___x_1217_; lean_object* v___x_1218_; lean_object* v___x_1219_; lean_object* v___x_1220_; lean_object* v___x_1221_; lean_object* v___x_1222_; lean_object* v___x_1223_; lean_object* v___x_1225_; 
v_ref_1216_ = lean_ctor_get(v_a_1208_, 5);
v___x_1217_ = 0;
v___x_1218_ = l_Lean_SourceInfo_fromRef(v_ref_1216_, v___x_1217_);
v___x_1219_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__2));
v___x_1220_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___closed__4));
v___x_1221_ = l_Lean_mkIdent(v_a_1212_);
lean_inc(v___x_1218_);
v___x_1222_ = l_Lean_Syntax_node1(v___x_1218_, v___x_1220_, v___x_1221_);
v___x_1223_ = l_Lean_Syntax_node1(v___x_1218_, v___x_1219_, v___x_1222_);
if (v_isShared_1215_ == 0)
{
lean_ctor_set(v___x_1214_, 0, v___x_1223_);
v___x_1225_ = v___x_1214_;
goto v_reusejp_1224_;
}
else
{
lean_object* v_reuseFailAlloc_1226_; 
v_reuseFailAlloc_1226_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1226_, 0, v___x_1223_);
v___x_1225_ = v_reuseFailAlloc_1226_;
goto v_reusejp_1224_;
}
v_reusejp_1224_:
{
return v___x_1225_;
}
}
}
else
{
lean_object* v_a_1228_; lean_object* v___x_1230_; uint8_t v_isShared_1231_; uint8_t v_isSharedCheck_1235_; 
v_a_1228_ = lean_ctor_get(v___x_1211_, 0);
v_isSharedCheck_1235_ = !lean_is_exclusive(v___x_1211_);
if (v_isSharedCheck_1235_ == 0)
{
v___x_1230_ = v___x_1211_;
v_isShared_1231_ = v_isSharedCheck_1235_;
goto v_resetjp_1229_;
}
else
{
lean_inc(v_a_1228_);
lean_dec(v___x_1211_);
v___x_1230_ = lean_box(0);
v_isShared_1231_ = v_isSharedCheck_1235_;
goto v_resetjp_1229_;
}
v_resetjp_1229_:
{
lean_object* v___x_1233_; 
if (v_isShared_1231_ == 0)
{
v___x_1233_ = v___x_1230_;
goto v_reusejp_1232_;
}
else
{
lean_object* v_reuseFailAlloc_1234_; 
v_reuseFailAlloc_1234_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1234_, 0, v_a_1228_);
v___x_1233_ = v_reuseFailAlloc_1234_;
goto v_reusejp_1232_;
}
v_reusejp_1232_:
{
return v___x_1233_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg___boxed(lean_object* v_fvarId_1236_, lean_object* v_a_1237_, lean_object* v_a_1238_, lean_object* v_a_1239_, lean_object* v_a_1240_){
_start:
{
lean_object* v_res_1241_; 
v_res_1241_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg(v_fvarId_1236_, v_a_1237_, v_a_1238_, v_a_1239_);
lean_dec(v_a_1239_);
lean_dec_ref(v_a_1238_);
lean_dec_ref(v_a_1237_);
return v_res_1241_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat(lean_object* v_fvarId_1242_, lean_object* v_a_1243_, lean_object* v_a_1244_, lean_object* v_a_1245_, lean_object* v_a_1246_){
_start:
{
lean_object* v___x_1248_; 
v___x_1248_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg(v_fvarId_1242_, v_a_1243_, v_a_1245_, v_a_1246_);
return v___x_1248_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___boxed(lean_object* v_fvarId_1249_, lean_object* v_a_1250_, lean_object* v_a_1251_, lean_object* v_a_1252_, lean_object* v_a_1253_, lean_object* v_a_1254_){
_start:
{
lean_object* v_res_1255_; 
v_res_1255_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat(v_fvarId_1249_, v_a_1250_, v_a_1251_, v_a_1252_, v_a_1253_);
lean_dec(v_a_1253_);
lean_dec_ref(v_a_1252_);
lean_dec(v_a_1251_);
lean_dec_ref(v_a_1250_);
return v_res_1255_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_extN_spec__0___redArg(size_t v_sz_1256_, size_t v_i_1257_, lean_object* v_bs_1258_, lean_object* v___y_1259_, lean_object* v___y_1260_, lean_object* v___y_1261_){
_start:
{
uint8_t v___x_1263_; 
v___x_1263_ = lean_usize_dec_lt(v_i_1257_, v_sz_1256_);
if (v___x_1263_ == 0)
{
lean_object* v___x_1264_; 
v___x_1264_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1264_, 0, v_bs_1258_);
return v___x_1264_;
}
else
{
lean_object* v_v_1265_; lean_object* v___x_1266_; 
v_v_1265_ = lean_array_uget_borrowed(v_bs_1258_, v_i_1257_);
lean_inc(v_v_1265_);
v___x_1266_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_extN_mkPat___redArg(v_v_1265_, v___y_1259_, v___y_1260_, v___y_1261_);
if (lean_obj_tag(v___x_1266_) == 0)
{
lean_object* v_a_1267_; lean_object* v___x_1268_; lean_object* v_bs_x27_1269_; size_t v___x_1270_; size_t v___x_1271_; lean_object* v___x_1272_; 
v_a_1267_ = lean_ctor_get(v___x_1266_, 0);
lean_inc(v_a_1267_);
lean_dec_ref_known(v___x_1266_, 1);
v___x_1268_ = lean_unsigned_to_nat(0u);
v_bs_x27_1269_ = lean_array_uset(v_bs_1258_, v_i_1257_, v___x_1268_);
v___x_1270_ = ((size_t)1ULL);
v___x_1271_ = lean_usize_add(v_i_1257_, v___x_1270_);
v___x_1272_ = lean_array_uset(v_bs_x27_1269_, v_i_1257_, v_a_1267_);
v_i_1257_ = v___x_1271_;
v_bs_1258_ = v___x_1272_;
goto _start;
}
else
{
lean_object* v_a_1274_; lean_object* v___x_1276_; uint8_t v_isShared_1277_; uint8_t v_isSharedCheck_1281_; 
lean_dec_ref(v_bs_1258_);
v_a_1274_ = lean_ctor_get(v___x_1266_, 0);
v_isSharedCheck_1281_ = !lean_is_exclusive(v___x_1266_);
if (v_isSharedCheck_1281_ == 0)
{
v___x_1276_ = v___x_1266_;
v_isShared_1277_ = v_isSharedCheck_1281_;
goto v_resetjp_1275_;
}
else
{
lean_inc(v_a_1274_);
lean_dec(v___x_1266_);
v___x_1276_ = lean_box(0);
v_isShared_1277_ = v_isSharedCheck_1281_;
goto v_resetjp_1275_;
}
v_resetjp_1275_:
{
lean_object* v___x_1279_; 
if (v_isShared_1277_ == 0)
{
v___x_1279_ = v___x_1276_;
goto v_reusejp_1278_;
}
else
{
lean_object* v_reuseFailAlloc_1280_; 
v_reuseFailAlloc_1280_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1280_, 0, v_a_1274_);
v___x_1279_ = v_reuseFailAlloc_1280_;
goto v_reusejp_1278_;
}
v_reusejp_1278_:
{
return v___x_1279_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_extN_spec__0___redArg___boxed(lean_object* v_sz_1282_, lean_object* v_i_1283_, lean_object* v_bs_1284_, lean_object* v___y_1285_, lean_object* v___y_1286_, lean_object* v___y_1287_, lean_object* v___y_1288_){
_start:
{
size_t v_sz_boxed_1289_; size_t v_i_boxed_1290_; lean_object* v_res_1291_; 
v_sz_boxed_1289_ = lean_unbox_usize(v_sz_1282_);
lean_dec(v_sz_1282_);
v_i_boxed_1290_ = lean_unbox_usize(v_i_1283_);
lean_dec(v_i_1283_);
v_res_1291_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_extN_spec__0___redArg(v_sz_boxed_1289_, v_i_boxed_1290_, v_bs_1284_, v___y_1285_, v___y_1286_, v___y_1287_);
lean_dec(v___y_1287_);
lean_dec_ref(v___y_1286_);
lean_dec_ref(v___y_1285_);
return v_res_1291_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_extN_spec__0(size_t v_sz_1292_, size_t v_i_1293_, lean_object* v_bs_1294_, lean_object* v___y_1295_, lean_object* v___y_1296_, lean_object* v___y_1297_, lean_object* v___y_1298_){
_start:
{
lean_object* v___x_1300_; 
v___x_1300_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_extN_spec__0___redArg(v_sz_1292_, v_i_1293_, v_bs_1294_, v___y_1295_, v___y_1297_, v___y_1298_);
return v___x_1300_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_extN_spec__0___boxed(lean_object* v_sz_1301_, lean_object* v_i_1302_, lean_object* v_bs_1303_, lean_object* v___y_1304_, lean_object* v___y_1305_, lean_object* v___y_1306_, lean_object* v___y_1307_, lean_object* v___y_1308_){
_start:
{
size_t v_sz_boxed_1309_; size_t v_i_boxed_1310_; lean_object* v_res_1311_; 
v_sz_boxed_1309_ = lean_unbox_usize(v_sz_1301_);
lean_dec(v_sz_1301_);
v_i_boxed_1310_ = lean_unbox_usize(v_i_1302_);
lean_dec(v_i_1302_);
v_res_1311_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_extN_spec__0(v_sz_boxed_1309_, v_i_boxed_1310_, v_bs_1303_, v___y_1304_, v___y_1305_, v___y_1306_, v___y_1307_);
lean_dec(v___y_1307_);
lean_dec_ref(v___y_1306_);
lean_dec(v___y_1305_);
lean_dec_ref(v___y_1304_);
return v_res_1311_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticBuilder_extN_spec__1(lean_object* v_as_1314_, size_t v_sz_1315_, size_t v_i_1316_, lean_object* v_b_1317_, lean_object* v___y_1318_, lean_object* v___y_1319_, lean_object* v___y_1320_, lean_object* v___y_1321_){
_start:
{
uint8_t v___x_1323_; 
v___x_1323_ = lean_usize_dec_lt(v_i_1316_, v_sz_1315_);
if (v___x_1323_ == 0)
{
lean_object* v___x_1324_; 
v___x_1324_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1324_, 0, v_b_1317_);
return v___x_1324_;
}
else
{
lean_object* v_a_1325_; lean_object* v_fst_1326_; lean_object* v_snd_1327_; size_t v_sz_1328_; lean_object* v___x_1329_; lean_object* v___x_1330_; lean_object* v___x_1331_; lean_object* v___x_1332_; 
v_a_1325_ = lean_array_uget_borrowed(v_as_1314_, v_i_1316_);
v_fst_1326_ = lean_ctor_get(v_a_1325_, 0);
v_snd_1327_ = lean_ctor_get(v_a_1325_, 1);
v_sz_1328_ = lean_array_size(v_snd_1327_);
v___x_1329_ = lean_box_usize(v_sz_1328_);
v___x_1330_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticBuilder_extN_spec__1___boxed__const__1));
lean_inc(v_snd_1327_);
v___x_1331_ = lean_alloc_closure((void*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_extN_spec__0___boxed), 8, 3);
lean_closure_set(v___x_1331_, 0, v___x_1329_);
lean_closure_set(v___x_1331_, 1, v___x_1330_);
lean_closure_set(v___x_1331_, 2, v_snd_1327_);
lean_inc(v_fst_1326_);
v___x_1332_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___redArg(v_fst_1326_, v___x_1331_, v___y_1318_, v___y_1319_, v___y_1320_, v___y_1321_);
if (lean_obj_tag(v___x_1332_) == 0)
{
lean_object* v_a_1333_; lean_object* v___x_1334_; size_t v___x_1335_; size_t v___x_1336_; 
v_a_1333_ = lean_ctor_get(v___x_1332_, 0);
lean_inc(v_a_1333_);
lean_dec_ref_known(v___x_1332_, 1);
v___x_1334_ = l_Array_append___redArg(v_b_1317_, v_a_1333_);
lean_dec(v_a_1333_);
v___x_1335_ = ((size_t)1ULL);
v___x_1336_ = lean_usize_add(v_i_1316_, v___x_1335_);
v_i_1316_ = v___x_1336_;
v_b_1317_ = v___x_1334_;
goto _start;
}
else
{
lean_dec_ref(v_b_1317_);
return v___x_1332_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticBuilder_extN_spec__1___boxed(lean_object* v_as_1338_, lean_object* v_sz_1339_, lean_object* v_i_1340_, lean_object* v_b_1341_, lean_object* v___y_1342_, lean_object* v___y_1343_, lean_object* v___y_1344_, lean_object* v___y_1345_, lean_object* v___y_1346_){
_start:
{
size_t v_sz_boxed_1347_; size_t v_i_boxed_1348_; lean_object* v_res_1349_; 
v_sz_boxed_1347_ = lean_unbox_usize(v_sz_1339_);
lean_dec(v_sz_1339_);
v_i_boxed_1348_ = lean_unbox_usize(v_i_1340_);
lean_dec(v_i_1340_);
v_res_1349_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticBuilder_extN_spec__1(v_as_1338_, v_sz_boxed_1347_, v_i_boxed_1348_, v_b_1341_, v___y_1342_, v___y_1343_, v___y_1344_, v___y_1345_);
lean_dec(v___y_1345_);
lean_dec_ref(v___y_1344_);
lean_dec(v___y_1343_);
lean_dec_ref(v___y_1342_);
lean_dec_ref(v_as_1338_);
return v_res_1349_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_extN(lean_object* v_r_1361_, lean_object* v_a_1362_, lean_object* v_a_1363_, lean_object* v_a_1364_, lean_object* v_a_1365_){
_start:
{
lean_object* v_depth_1367_; lean_object* v_commonFVarIds_1368_; lean_object* v_goals_1369_; lean_object* v___x_1371_; uint8_t v_isShared_1372_; uint8_t v_isSharedCheck_1432_; 
v_depth_1367_ = lean_ctor_get(v_r_1361_, 0);
v_commonFVarIds_1368_ = lean_ctor_get(v_r_1361_, 1);
v_goals_1369_ = lean_ctor_get(v_r_1361_, 2);
v_isSharedCheck_1432_ = !lean_is_exclusive(v_r_1361_);
if (v_isSharedCheck_1432_ == 0)
{
v___x_1371_ = v_r_1361_;
v_isShared_1372_ = v_isSharedCheck_1432_;
goto v_resetjp_1370_;
}
else
{
lean_inc(v_goals_1369_);
lean_inc(v_commonFVarIds_1368_);
lean_inc(v_depth_1367_);
lean_dec(v_r_1361_);
v___x_1371_ = lean_box(0);
v_isShared_1372_ = v_isSharedCheck_1432_;
goto v_resetjp_1370_;
}
v_resetjp_1370_:
{
lean_object* v___x_1373_; uint8_t v___x_1374_; lean_object* v_pats_1376_; lean_object* v___y_1377_; 
v___x_1373_ = lean_unsigned_to_nat(0u);
v___x_1374_ = lean_nat_dec_eq(v_depth_1367_, v___x_1373_);
if (v___x_1374_ == 0)
{
lean_object* v_pats_1398_; lean_object* v___x_1399_; uint8_t v___x_1400_; 
v_pats_1398_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_extN___closed__4));
v___x_1399_ = lean_array_get_size(v_goals_1369_);
v___x_1400_ = lean_nat_dec_lt(v___x_1373_, v___x_1399_);
if (v___x_1400_ == 0)
{
lean_dec_ref(v_goals_1369_);
lean_dec_ref(v_commonFVarIds_1368_);
v_pats_1376_ = v_pats_1398_;
v___y_1377_ = v_a_1364_;
goto v___jp_1375_;
}
else
{
lean_object* v___x_1401_; lean_object* v_fst_1402_; size_t v_sz_1403_; size_t v___x_1404_; lean_object* v___x_1405_; lean_object* v___x_1406_; lean_object* v___x_1407_; lean_object* v___x_1408_; 
v___x_1401_ = lean_array_fget_borrowed(v_goals_1369_, v___x_1373_);
v_fst_1402_ = lean_ctor_get(v___x_1401_, 0);
v_sz_1403_ = lean_array_size(v_commonFVarIds_1368_);
v___x_1404_ = ((size_t)0ULL);
v___x_1405_ = lean_box_usize(v_sz_1403_);
v___x_1406_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticBuilder_extN_spec__1___boxed__const__1));
v___x_1407_ = lean_alloc_closure((void*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_extN_spec__0___boxed), 8, 3);
lean_closure_set(v___x_1407_, 0, v___x_1405_);
lean_closure_set(v___x_1407_, 1, v___x_1406_);
lean_closure_set(v___x_1407_, 2, v_commonFVarIds_1368_);
lean_inc(v_fst_1402_);
v___x_1408_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___redArg(v_fst_1402_, v___x_1407_, v_a_1362_, v_a_1363_, v_a_1364_, v_a_1365_);
if (lean_obj_tag(v___x_1408_) == 0)
{
lean_object* v_a_1409_; lean_object* v___x_1410_; size_t v_sz_1411_; lean_object* v___x_1412_; 
v_a_1409_ = lean_ctor_get(v___x_1408_, 0);
lean_inc(v_a_1409_);
lean_dec_ref_known(v___x_1408_, 1);
v___x_1410_ = l_Array_append___redArg(v_pats_1398_, v_a_1409_);
lean_dec(v_a_1409_);
v_sz_1411_ = lean_array_size(v_goals_1369_);
v___x_1412_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticBuilder_extN_spec__1(v_goals_1369_, v_sz_1411_, v___x_1404_, v___x_1410_, v_a_1362_, v_a_1363_, v_a_1364_, v_a_1365_);
lean_dec_ref(v_goals_1369_);
if (lean_obj_tag(v___x_1412_) == 0)
{
lean_object* v_a_1413_; 
v_a_1413_ = lean_ctor_get(v___x_1412_, 0);
lean_inc(v_a_1413_);
lean_dec_ref_known(v___x_1412_, 1);
v_pats_1376_ = v_a_1413_;
v___y_1377_ = v_a_1364_;
goto v___jp_1375_;
}
else
{
lean_object* v_a_1414_; lean_object* v___x_1416_; uint8_t v_isShared_1417_; uint8_t v_isSharedCheck_1421_; 
lean_del_object(v___x_1371_);
lean_dec(v_depth_1367_);
v_a_1414_ = lean_ctor_get(v___x_1412_, 0);
v_isSharedCheck_1421_ = !lean_is_exclusive(v___x_1412_);
if (v_isSharedCheck_1421_ == 0)
{
v___x_1416_ = v___x_1412_;
v_isShared_1417_ = v_isSharedCheck_1421_;
goto v_resetjp_1415_;
}
else
{
lean_inc(v_a_1414_);
lean_dec(v___x_1412_);
v___x_1416_ = lean_box(0);
v_isShared_1417_ = v_isSharedCheck_1421_;
goto v_resetjp_1415_;
}
v_resetjp_1415_:
{
lean_object* v___x_1419_; 
if (v_isShared_1417_ == 0)
{
v___x_1419_ = v___x_1416_;
goto v_reusejp_1418_;
}
else
{
lean_object* v_reuseFailAlloc_1420_; 
v_reuseFailAlloc_1420_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1420_, 0, v_a_1414_);
v___x_1419_ = v_reuseFailAlloc_1420_;
goto v_reusejp_1418_;
}
v_reusejp_1418_:
{
return v___x_1419_;
}
}
}
}
else
{
lean_object* v_a_1422_; lean_object* v___x_1424_; uint8_t v_isShared_1425_; uint8_t v_isSharedCheck_1429_; 
lean_del_object(v___x_1371_);
lean_dec_ref(v_goals_1369_);
lean_dec(v_depth_1367_);
v_a_1422_ = lean_ctor_get(v___x_1408_, 0);
v_isSharedCheck_1429_ = !lean_is_exclusive(v___x_1408_);
if (v_isSharedCheck_1429_ == 0)
{
v___x_1424_ = v___x_1408_;
v_isShared_1425_ = v_isSharedCheck_1429_;
goto v_resetjp_1423_;
}
else
{
lean_inc(v_a_1422_);
lean_dec(v___x_1408_);
v___x_1424_ = lean_box(0);
v_isShared_1425_ = v_isSharedCheck_1429_;
goto v_resetjp_1423_;
}
v_resetjp_1423_:
{
lean_object* v___x_1427_; 
if (v_isShared_1425_ == 0)
{
v___x_1427_ = v___x_1424_;
goto v_reusejp_1426_;
}
else
{
lean_object* v_reuseFailAlloc_1428_; 
v_reuseFailAlloc_1428_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1428_, 0, v_a_1422_);
v___x_1427_ = v_reuseFailAlloc_1428_;
goto v_reusejp_1426_;
}
v_reusejp_1426_:
{
return v___x_1427_;
}
}
}
}
}
else
{
lean_object* v___x_1430_; lean_object* v___x_1431_; 
lean_del_object(v___x_1371_);
lean_dec_ref(v_goals_1369_);
lean_dec_ref(v_commonFVarIds_1368_);
lean_dec(v_depth_1367_);
v___x_1430_ = lp_aesop_Aesop_Script_Tactic_skip;
v___x_1431_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1431_, 0, v___x_1430_);
return v___x_1431_;
}
v___jp_1375_:
{
lean_object* v_ref_1378_; lean_object* v___x_1379_; lean_object* v___x_1380_; lean_object* v_depthStx_1381_; lean_object* v___x_1382_; lean_object* v___x_1383_; lean_object* v___x_1384_; lean_object* v___x_1385_; lean_object* v___x_1386_; lean_object* v___x_1387_; lean_object* v___x_1388_; lean_object* v___x_1390_; 
v_ref_1378_ = lean_ctor_get(v___y_1377_, 5);
v___x_1379_ = l_Nat_reprFast(v_depth_1367_);
v___x_1380_ = lean_box(2);
v_depthStx_1381_ = l_Lean_Syntax_mkNumLit(v___x_1379_, v___x_1380_);
v___x_1382_ = l_Lean_SourceInfo_fromRef(v_ref_1378_, v___x_1374_);
v___x_1383_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_extN___closed__2));
v___x_1384_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_extN___closed__3));
lean_inc_n(v___x_1382_, 2);
v___x_1385_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1385_, 0, v___x_1382_);
lean_ctor_set(v___x_1385_, 1, v___x_1383_);
v___x_1386_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_1387_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
v___x_1388_ = l_Array_append___redArg(v___x_1387_, v_pats_1376_);
lean_dec_ref(v_pats_1376_);
if (v_isShared_1372_ == 0)
{
lean_ctor_set_tag(v___x_1371_, 1);
lean_ctor_set(v___x_1371_, 2, v___x_1388_);
lean_ctor_set(v___x_1371_, 1, v___x_1386_);
lean_ctor_set(v___x_1371_, 0, v___x_1382_);
v___x_1390_ = v___x_1371_;
goto v_reusejp_1389_;
}
else
{
lean_object* v_reuseFailAlloc_1397_; 
v_reuseFailAlloc_1397_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1397_, 0, v___x_1382_);
lean_ctor_set(v_reuseFailAlloc_1397_, 1, v___x_1386_);
lean_ctor_set(v_reuseFailAlloc_1397_, 2, v___x_1388_);
v___x_1390_ = v_reuseFailAlloc_1397_;
goto v_reusejp_1389_;
}
v_reusejp_1389_:
{
lean_object* v___x_1391_; lean_object* v___x_1392_; lean_object* v___x_1393_; lean_object* v___x_1394_; lean_object* v___x_1395_; lean_object* v___x_1396_; 
v___x_1391_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__14));
lean_inc_n(v___x_1382_, 2);
v___x_1392_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1392_, 0, v___x_1382_);
lean_ctor_set(v___x_1392_, 1, v___x_1391_);
v___x_1393_ = l_Lean_Syntax_node2(v___x_1382_, v___x_1386_, v___x_1392_, v_depthStx_1381_);
v___x_1394_ = l_Lean_Syntax_node3(v___x_1382_, v___x_1384_, v___x_1385_, v___x_1390_, v___x_1393_);
v___x_1395_ = lp_aesop_Aesop_Script_Tactic_unstructured(v___x_1394_);
v___x_1396_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1396_, 0, v___x_1395_);
return v___x_1396_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_extN___boxed(lean_object* v_r_1433_, lean_object* v_a_1434_, lean_object* v_a_1435_, lean_object* v_a_1436_, lean_object* v_a_1437_, lean_object* v_a_1438_){
_start:
{
lean_object* v_res_1439_; 
v_res_1439_ = lp_aesop_Aesop_Script_TacticBuilder_extN(v_r_1433_, v_a_1434_, v_a_1435_, v_a_1436_, v_a_1437_);
lean_dec(v_a_1437_);
lean_dec_ref(v_a_1436_);
lean_dec(v_a_1435_);
lean_dec_ref(v_a_1434_);
return v_res_1439_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0(lean_object* v_info_1459_, lean_object* v_toPure_1460_, lean_object* v_quotCtx_1461_){
_start:
{
lean_object* v___x_1462_; lean_object* v___x_1463_; lean_object* v___x_1464_; lean_object* v___x_1465_; lean_object* v___x_1466_; lean_object* v___x_1467_; lean_object* v___x_1468_; lean_object* v___x_1469_; lean_object* v___x_1470_; lean_object* v___x_1471_; lean_object* v___x_1472_; lean_object* v___x_1473_; lean_object* v___x_1474_; lean_object* v___x_1475_; lean_object* v___x_1476_; lean_object* v___x_1477_; lean_object* v___x_1478_; lean_object* v___x_1479_; lean_object* v___x_1480_; 
v___x_1462_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__0));
v___x_1463_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__1));
lean_inc_n(v_info_1459_, 8);
v___x_1464_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1464_, 0, v_info_1459_);
lean_ctor_set(v___x_1464_, 1, v___x_1462_);
v___x_1465_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__3));
v___x_1466_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_1467_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
v___x_1468_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1468_, 0, v_info_1459_);
lean_ctor_set(v___x_1468_, 1, v___x_1466_);
lean_ctor_set(v___x_1468_, 2, v___x_1467_);
lean_inc_ref_n(v___x_1468_, 3);
v___x_1469_ = l_Lean_Syntax_node1(v_info_1459_, v___x_1465_, v___x_1468_);
v___x_1470_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__1));
v___x_1471_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__2));
v___x_1472_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1472_, 0, v_info_1459_);
lean_ctor_set(v___x_1472_, 1, v___x_1471_);
v___x_1473_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__5));
v___x_1474_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__6));
v___x_1475_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1475_, 0, v_info_1459_);
lean_ctor_set(v___x_1475_, 1, v___x_1474_);
v___x_1476_ = l_Lean_Syntax_node1(v_info_1459_, v___x_1473_, v___x_1475_);
v___x_1477_ = l_Lean_Syntax_node2(v_info_1459_, v___x_1470_, v___x_1472_, v___x_1476_);
v___x_1478_ = l_Lean_Syntax_node1(v_info_1459_, v___x_1466_, v___x_1477_);
v___x_1479_ = l_Lean_Syntax_node6(v_info_1459_, v___x_1463_, v___x_1464_, v___x_1469_, v___x_1468_, v___x_1468_, v___x_1468_, v___x_1478_);
v___x_1480_ = lean_apply_2(v_toPure_1460_, lean_box(0), v___x_1479_);
return v___x_1480_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___boxed(lean_object* v_info_1481_, lean_object* v_toPure_1482_, lean_object* v_quotCtx_1483_){
_start:
{
lean_object* v_res_1484_; 
v_res_1484_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0(v_info_1481_, v_toPure_1482_, v_quotCtx_1483_);
lean_dec(v_quotCtx_1483_);
return v_res_1484_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__1(lean_object* v_toBind_1485_, lean_object* v_getContext_1486_, lean_object* v___f_1487_, lean_object* v_scp_1488_){
_start:
{
lean_object* v___x_1489_; 
v___x_1489_ = lean_apply_4(v_toBind_1485_, lean_box(0), lean_box(0), v_getContext_1486_, v___f_1487_);
return v___x_1489_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__1___boxed(lean_object* v_toBind_1490_, lean_object* v_getContext_1491_, lean_object* v___f_1492_, lean_object* v_scp_1493_){
_start:
{
lean_object* v_res_1494_; 
v_res_1494_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__1(v_toBind_1490_, v_getContext_1491_, v___f_1492_, v_scp_1493_);
lean_dec(v_scp_1493_);
return v_res_1494_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__2(lean_object* v_toPure_1495_, lean_object* v_toBind_1496_, lean_object* v_getContext_1497_, lean_object* v_getCurrMacroScope_1498_, lean_object* v_info_1499_){
_start:
{
lean_object* v___f_1500_; lean_object* v___f_1501_; lean_object* v___x_1502_; 
v___f_1500_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1500_, 0, v_info_1499_);
lean_closure_set(v___f_1500_, 1, v_toPure_1495_);
lean_inc(v_toBind_1496_);
v___f_1501_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_1501_, 0, v_toBind_1496_);
lean_closure_set(v___f_1501_, 1, v_getContext_1497_);
lean_closure_set(v___f_1501_, 2, v___f_1500_);
v___x_1502_ = lean_apply_4(v_toBind_1496_, lean_box(0), lean_box(0), v_getCurrMacroScope_1498_, v___f_1501_);
return v___x_1502_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__3(uint8_t v_simpAll_1503_, lean_object* v_toPure_1504_, lean_object* v_____do__lift_1505_){
_start:
{
lean_object* v___x_1506_; lean_object* v___x_1507_; 
v___x_1506_ = l_Lean_SourceInfo_fromRef(v_____do__lift_1505_, v_simpAll_1503_);
v___x_1507_ = lean_apply_2(v_toPure_1504_, lean_box(0), v___x_1506_);
return v___x_1507_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__3___boxed(lean_object* v_simpAll_1508_, lean_object* v_toPure_1509_, lean_object* v_____do__lift_1510_){
_start:
{
uint8_t v_simpAll_boxed_1511_; lean_object* v_res_1512_; 
v_simpAll_boxed_1511_ = lean_unbox(v_simpAll_1508_);
v_res_1512_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__3(v_simpAll_boxed_1511_, v_toPure_1509_, v_____do__lift_1510_);
lean_dec(v_____do__lift_1510_);
return v_res_1512_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__7(void){
_start:
{
lean_object* v___x_1529_; lean_object* v___x_1530_; 
v___x_1529_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__6));
v___x_1530_ = l_Lean_mkIdent(v___x_1529_);
return v___x_1530_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4(lean_object* v_info_1532_, lean_object* v_val_1533_, lean_object* v_toPure_1534_, lean_object* v_quotCtx_1535_){
_start:
{
lean_object* v___x_1536_; lean_object* v___x_1537_; lean_object* v___x_1538_; lean_object* v___x_1539_; lean_object* v___x_1540_; lean_object* v___x_1541_; lean_object* v___x_1542_; lean_object* v___x_1543_; lean_object* v___x_1544_; lean_object* v___x_1545_; lean_object* v___x_1546_; lean_object* v___x_1547_; lean_object* v___x_1548_; lean_object* v___x_1549_; lean_object* v___x_1550_; lean_object* v___x_1551_; lean_object* v___x_1552_; lean_object* v___x_1553_; lean_object* v___x_1554_; lean_object* v___x_1555_; lean_object* v___x_1556_; lean_object* v___x_1557_; lean_object* v___x_1558_; lean_object* v___x_1559_; lean_object* v___x_1560_; lean_object* v___x_1561_; lean_object* v___x_1562_; lean_object* v___x_1563_; lean_object* v___x_1564_; lean_object* v___x_1565_; lean_object* v___x_1566_; 
v___x_1536_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__0));
v___x_1537_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__1));
lean_inc_n(v_info_1532_, 14);
v___x_1538_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1538_, 0, v_info_1532_);
lean_ctor_set(v___x_1538_, 1, v___x_1536_);
v___x_1539_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__3));
v___x_1540_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_1541_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__1));
v___x_1542_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__3));
v___x_1543_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__4));
v___x_1544_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1544_, 0, v_info_1532_);
lean_ctor_set(v___x_1544_, 1, v___x_1543_);
v___x_1545_ = lean_obj_once(&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__7, &lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__7_once, _init_lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__7);
v___x_1546_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__15));
v___x_1547_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1547_, 0, v_info_1532_);
lean_ctor_set(v___x_1547_, 1, v___x_1546_);
v___x_1548_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__8));
v___x_1549_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1549_, 0, v_info_1532_);
lean_ctor_set(v___x_1549_, 1, v___x_1548_);
v___x_1550_ = l_Lean_Syntax_node5(v_info_1532_, v___x_1542_, v___x_1544_, v___x_1545_, v___x_1547_, v_val_1533_, v___x_1549_);
v___x_1551_ = l_Lean_Syntax_node1(v_info_1532_, v___x_1541_, v___x_1550_);
v___x_1552_ = l_Lean_Syntax_node1(v_info_1532_, v___x_1540_, v___x_1551_);
v___x_1553_ = l_Lean_Syntax_node1(v_info_1532_, v___x_1539_, v___x_1552_);
v___x_1554_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
v___x_1555_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1555_, 0, v_info_1532_);
lean_ctor_set(v___x_1555_, 1, v___x_1540_);
lean_ctor_set(v___x_1555_, 2, v___x_1554_);
v___x_1556_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__1));
v___x_1557_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__2));
v___x_1558_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1558_, 0, v_info_1532_);
lean_ctor_set(v___x_1558_, 1, v___x_1557_);
v___x_1559_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__5));
v___x_1560_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__6));
v___x_1561_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1561_, 0, v_info_1532_);
lean_ctor_set(v___x_1561_, 1, v___x_1560_);
v___x_1562_ = l_Lean_Syntax_node1(v_info_1532_, v___x_1559_, v___x_1561_);
v___x_1563_ = l_Lean_Syntax_node2(v_info_1532_, v___x_1556_, v___x_1558_, v___x_1562_);
v___x_1564_ = l_Lean_Syntax_node1(v_info_1532_, v___x_1540_, v___x_1563_);
lean_inc_ref_n(v___x_1555_, 2);
v___x_1565_ = l_Lean_Syntax_node6(v_info_1532_, v___x_1537_, v___x_1538_, v___x_1553_, v___x_1555_, v___x_1555_, v___x_1555_, v___x_1564_);
v___x_1566_ = lean_apply_2(v_toPure_1534_, lean_box(0), v___x_1565_);
return v___x_1566_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___boxed(lean_object* v_info_1567_, lean_object* v_val_1568_, lean_object* v_toPure_1569_, lean_object* v_quotCtx_1570_){
_start:
{
lean_object* v_res_1571_; 
v_res_1571_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4(v_info_1567_, v_val_1568_, v_toPure_1569_, v_quotCtx_1570_);
lean_dec(v_quotCtx_1570_);
return v_res_1571_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__6(lean_object* v_val_1572_, lean_object* v_toPure_1573_, lean_object* v_toBind_1574_, lean_object* v_getContext_1575_, lean_object* v_getCurrMacroScope_1576_, lean_object* v_info_1577_){
_start:
{
lean_object* v___f_1578_; lean_object* v___f_1579_; lean_object* v___x_1580_; 
v___f_1578_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___boxed), 4, 3);
lean_closure_set(v___f_1578_, 0, v_info_1577_);
lean_closure_set(v___f_1578_, 1, v_val_1572_);
lean_closure_set(v___f_1578_, 2, v_toPure_1573_);
lean_inc(v_toBind_1574_);
v___f_1579_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_1579_, 0, v_toBind_1574_);
lean_closure_set(v___f_1579_, 1, v_getContext_1575_);
lean_closure_set(v___f_1579_, 2, v___f_1578_);
v___x_1580_ = lean_apply_4(v_toBind_1574_, lean_box(0), lean_box(0), v_getCurrMacroScope_1576_, v___f_1579_);
return v___x_1580_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7(lean_object* v_info_1588_, lean_object* v_toPure_1589_, lean_object* v_quotCtx_1590_){
_start:
{
lean_object* v___x_1591_; lean_object* v___x_1592_; lean_object* v___x_1593_; lean_object* v___x_1594_; lean_object* v___x_1595_; lean_object* v___x_1596_; lean_object* v___x_1597_; lean_object* v___x_1598_; lean_object* v___x_1599_; lean_object* v___x_1600_; 
v___x_1591_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__1));
v___x_1592_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__2));
lean_inc_n(v_info_1588_, 3);
v___x_1593_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1593_, 0, v_info_1588_);
lean_ctor_set(v___x_1593_, 1, v___x_1592_);
v___x_1594_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__3));
v___x_1595_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_1596_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
v___x_1597_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1597_, 0, v_info_1588_);
lean_ctor_set(v___x_1597_, 1, v___x_1595_);
lean_ctor_set(v___x_1597_, 2, v___x_1596_);
lean_inc_ref_n(v___x_1597_, 3);
v___x_1598_ = l_Lean_Syntax_node1(v_info_1588_, v___x_1594_, v___x_1597_);
v___x_1599_ = l_Lean_Syntax_node5(v_info_1588_, v___x_1591_, v___x_1593_, v___x_1598_, v___x_1597_, v___x_1597_, v___x_1597_);
v___x_1600_ = lean_apply_2(v_toPure_1589_, lean_box(0), v___x_1599_);
return v___x_1600_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___boxed(lean_object* v_info_1601_, lean_object* v_toPure_1602_, lean_object* v_quotCtx_1603_){
_start:
{
lean_object* v_res_1604_; 
v_res_1604_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7(v_info_1601_, v_toPure_1602_, v_quotCtx_1603_);
lean_dec(v_quotCtx_1603_);
return v_res_1604_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__8(lean_object* v_toPure_1605_, lean_object* v_toBind_1606_, lean_object* v_getContext_1607_, lean_object* v_getCurrMacroScope_1608_, lean_object* v_info_1609_){
_start:
{
lean_object* v___f_1610_; lean_object* v___f_1611_; lean_object* v___x_1612_; 
v___f_1610_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___boxed), 3, 2);
lean_closure_set(v___f_1610_, 0, v_info_1609_);
lean_closure_set(v___f_1610_, 1, v_toPure_1605_);
lean_inc(v_toBind_1606_);
v___f_1611_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_1611_, 0, v_toBind_1606_);
lean_closure_set(v___f_1611_, 1, v_getContext_1607_);
lean_closure_set(v___f_1611_, 2, v___f_1610_);
v___x_1612_ = lean_apply_4(v_toBind_1606_, lean_box(0), lean_box(0), v_getCurrMacroScope_1608_, v___f_1611_);
return v___x_1612_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__5(lean_object* v_toPure_1613_, lean_object* v_____do__lift_1614_){
_start:
{
uint8_t v___x_1615_; lean_object* v___x_1616_; lean_object* v___x_1617_; 
v___x_1615_ = 0;
v___x_1616_ = l_Lean_SourceInfo_fromRef(v_____do__lift_1614_, v___x_1615_);
v___x_1617_ = lean_apply_2(v_toPure_1613_, lean_box(0), v___x_1616_);
return v___x_1617_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__5___boxed(lean_object* v_toPure_1618_, lean_object* v_____do__lift_1619_){
_start:
{
lean_object* v_res_1620_; 
v_res_1620_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__5(v_toPure_1618_, v_____do__lift_1619_);
lean_dec(v_____do__lift_1619_);
return v_res_1620_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__9(lean_object* v_info_1621_, lean_object* v_val_1622_, lean_object* v_toPure_1623_, lean_object* v_quotCtx_1624_){
_start:
{
lean_object* v___x_1625_; lean_object* v___x_1626_; lean_object* v___x_1627_; lean_object* v___x_1628_; lean_object* v___x_1629_; lean_object* v___x_1630_; lean_object* v___x_1631_; lean_object* v___x_1632_; lean_object* v___x_1633_; lean_object* v___x_1634_; lean_object* v___x_1635_; lean_object* v___x_1636_; lean_object* v___x_1637_; lean_object* v___x_1638_; lean_object* v___x_1639_; lean_object* v___x_1640_; lean_object* v___x_1641_; lean_object* v___x_1642_; lean_object* v___x_1643_; lean_object* v___x_1644_; lean_object* v___x_1645_; lean_object* v___x_1646_; 
v___x_1625_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__1));
v___x_1626_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__2));
lean_inc_n(v_info_1621_, 9);
v___x_1627_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1627_, 0, v_info_1621_);
lean_ctor_set(v___x_1627_, 1, v___x_1626_);
v___x_1628_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__3));
v___x_1629_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_1630_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__1));
v___x_1631_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__3));
v___x_1632_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__4));
v___x_1633_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1633_, 0, v_info_1621_);
lean_ctor_set(v___x_1633_, 1, v___x_1632_);
v___x_1634_ = lean_obj_once(&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__7, &lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__7_once, _init_lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__7);
v___x_1635_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__15));
v___x_1636_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1636_, 0, v_info_1621_);
lean_ctor_set(v___x_1636_, 1, v___x_1635_);
v___x_1637_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__8));
v___x_1638_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1638_, 0, v_info_1621_);
lean_ctor_set(v___x_1638_, 1, v___x_1637_);
v___x_1639_ = l_Lean_Syntax_node5(v_info_1621_, v___x_1631_, v___x_1633_, v___x_1634_, v___x_1636_, v_val_1622_, v___x_1638_);
v___x_1640_ = l_Lean_Syntax_node1(v_info_1621_, v___x_1630_, v___x_1639_);
v___x_1641_ = l_Lean_Syntax_node1(v_info_1621_, v___x_1629_, v___x_1640_);
v___x_1642_ = l_Lean_Syntax_node1(v_info_1621_, v___x_1628_, v___x_1641_);
v___x_1643_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
v___x_1644_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1644_, 0, v_info_1621_);
lean_ctor_set(v___x_1644_, 1, v___x_1629_);
lean_ctor_set(v___x_1644_, 2, v___x_1643_);
lean_inc_ref_n(v___x_1644_, 2);
v___x_1645_ = l_Lean_Syntax_node5(v_info_1621_, v___x_1625_, v___x_1627_, v___x_1642_, v___x_1644_, v___x_1644_, v___x_1644_);
v___x_1646_ = lean_apply_2(v_toPure_1623_, lean_box(0), v___x_1645_);
return v___x_1646_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__9___boxed(lean_object* v_info_1647_, lean_object* v_val_1648_, lean_object* v_toPure_1649_, lean_object* v_quotCtx_1650_){
_start:
{
lean_object* v_res_1651_; 
v_res_1651_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__9(v_info_1647_, v_val_1648_, v_toPure_1649_, v_quotCtx_1650_);
lean_dec(v_quotCtx_1650_);
return v_res_1651_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__11(lean_object* v_val_1652_, lean_object* v_toPure_1653_, lean_object* v_toBind_1654_, lean_object* v_getContext_1655_, lean_object* v_getCurrMacroScope_1656_, lean_object* v_info_1657_){
_start:
{
lean_object* v___f_1658_; lean_object* v___f_1659_; lean_object* v___x_1660_; 
v___f_1658_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__9___boxed), 4, 3);
lean_closure_set(v___f_1658_, 0, v_info_1657_);
lean_closure_set(v___f_1658_, 1, v_val_1652_);
lean_closure_set(v___f_1658_, 2, v_toPure_1653_);
lean_inc(v_toBind_1654_);
v___f_1659_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_1659_, 0, v_toBind_1654_);
lean_closure_set(v___f_1659_, 1, v_getContext_1655_);
lean_closure_set(v___f_1659_, 2, v___f_1658_);
v___x_1660_ = lean_apply_4(v_toBind_1654_, lean_box(0), lean_box(0), v_getCurrMacroScope_1656_, v___f_1659_);
return v___x_1660_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg(lean_object* v_inst_1661_, lean_object* v_inst_1662_, uint8_t v_simpAll_1663_, lean_object* v_configStx_x3f_1664_){
_start:
{
if (v_simpAll_1663_ == 0)
{
if (lean_obj_tag(v_configStx_x3f_1664_) == 0)
{
lean_object* v_toMonadRef_1665_; lean_object* v_toApplicative_1666_; lean_object* v_toBind_1667_; lean_object* v_getCurrMacroScope_1668_; lean_object* v_getContext_1669_; lean_object* v_getRef_1670_; lean_object* v_toPure_1671_; lean_object* v___f_1672_; lean_object* v___x_1673_; lean_object* v___f_1674_; lean_object* v___x_1675_; lean_object* v___x_1676_; 
v_toMonadRef_1665_ = lean_ctor_get(v_inst_1662_, 0);
lean_inc_ref(v_toMonadRef_1665_);
v_toApplicative_1666_ = lean_ctor_get(v_inst_1661_, 0);
lean_inc_ref(v_toApplicative_1666_);
v_toBind_1667_ = lean_ctor_get(v_inst_1661_, 1);
lean_inc_n(v_toBind_1667_, 3);
lean_dec_ref(v_inst_1661_);
v_getCurrMacroScope_1668_ = lean_ctor_get(v_inst_1662_, 1);
lean_inc(v_getCurrMacroScope_1668_);
v_getContext_1669_ = lean_ctor_get(v_inst_1662_, 2);
lean_inc(v_getContext_1669_);
lean_dec_ref(v_inst_1662_);
v_getRef_1670_ = lean_ctor_get(v_toMonadRef_1665_, 0);
lean_inc(v_getRef_1670_);
lean_dec_ref(v_toMonadRef_1665_);
v_toPure_1671_ = lean_ctor_get(v_toApplicative_1666_, 1);
lean_inc_n(v_toPure_1671_, 2);
lean_dec_ref(v_toApplicative_1666_);
v___f_1672_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__2), 5, 4);
lean_closure_set(v___f_1672_, 0, v_toPure_1671_);
lean_closure_set(v___f_1672_, 1, v_toBind_1667_);
lean_closure_set(v___f_1672_, 2, v_getContext_1669_);
lean_closure_set(v___f_1672_, 3, v_getCurrMacroScope_1668_);
v___x_1673_ = lean_box(v_simpAll_1663_);
v___f_1674_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__3___boxed), 3, 2);
lean_closure_set(v___f_1674_, 0, v___x_1673_);
lean_closure_set(v___f_1674_, 1, v_toPure_1671_);
v___x_1675_ = lean_apply_4(v_toBind_1667_, lean_box(0), lean_box(0), v_getRef_1670_, v___f_1674_);
v___x_1676_ = lean_apply_4(v_toBind_1667_, lean_box(0), lean_box(0), v___x_1675_, v___f_1672_);
return v___x_1676_;
}
else
{
lean_object* v_toMonadRef_1677_; lean_object* v_toApplicative_1678_; lean_object* v_val_1679_; lean_object* v_toBind_1680_; lean_object* v_getCurrMacroScope_1681_; lean_object* v_getContext_1682_; lean_object* v_getRef_1683_; lean_object* v_toPure_1684_; lean_object* v___f_1685_; lean_object* v___x_1686_; lean_object* v___f_1687_; lean_object* v___x_1688_; lean_object* v___x_1689_; 
v_toMonadRef_1677_ = lean_ctor_get(v_inst_1662_, 0);
lean_inc_ref(v_toMonadRef_1677_);
v_toApplicative_1678_ = lean_ctor_get(v_inst_1661_, 0);
lean_inc_ref(v_toApplicative_1678_);
v_val_1679_ = lean_ctor_get(v_configStx_x3f_1664_, 0);
lean_inc(v_val_1679_);
lean_dec_ref_known(v_configStx_x3f_1664_, 1);
v_toBind_1680_ = lean_ctor_get(v_inst_1661_, 1);
lean_inc_n(v_toBind_1680_, 3);
lean_dec_ref(v_inst_1661_);
v_getCurrMacroScope_1681_ = lean_ctor_get(v_inst_1662_, 1);
lean_inc(v_getCurrMacroScope_1681_);
v_getContext_1682_ = lean_ctor_get(v_inst_1662_, 2);
lean_inc(v_getContext_1682_);
lean_dec_ref(v_inst_1662_);
v_getRef_1683_ = lean_ctor_get(v_toMonadRef_1677_, 0);
lean_inc(v_getRef_1683_);
lean_dec_ref(v_toMonadRef_1677_);
v_toPure_1684_ = lean_ctor_get(v_toApplicative_1678_, 1);
lean_inc_n(v_toPure_1684_, 2);
lean_dec_ref(v_toApplicative_1678_);
v___f_1685_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__6), 6, 5);
lean_closure_set(v___f_1685_, 0, v_val_1679_);
lean_closure_set(v___f_1685_, 1, v_toPure_1684_);
lean_closure_set(v___f_1685_, 2, v_toBind_1680_);
lean_closure_set(v___f_1685_, 3, v_getContext_1682_);
lean_closure_set(v___f_1685_, 4, v_getCurrMacroScope_1681_);
v___x_1686_ = lean_box(v_simpAll_1663_);
v___f_1687_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__3___boxed), 3, 2);
lean_closure_set(v___f_1687_, 0, v___x_1686_);
lean_closure_set(v___f_1687_, 1, v_toPure_1684_);
v___x_1688_ = lean_apply_4(v_toBind_1680_, lean_box(0), lean_box(0), v_getRef_1683_, v___f_1687_);
v___x_1689_ = lean_apply_4(v_toBind_1680_, lean_box(0), lean_box(0), v___x_1688_, v___f_1685_);
return v___x_1689_;
}
}
else
{
if (lean_obj_tag(v_configStx_x3f_1664_) == 0)
{
lean_object* v_toMonadRef_1690_; lean_object* v_toApplicative_1691_; lean_object* v_toBind_1692_; lean_object* v_getCurrMacroScope_1693_; lean_object* v_getContext_1694_; lean_object* v_getRef_1695_; lean_object* v_toPure_1696_; lean_object* v___f_1697_; lean_object* v___f_1698_; lean_object* v___x_1699_; lean_object* v___x_1700_; 
v_toMonadRef_1690_ = lean_ctor_get(v_inst_1662_, 0);
lean_inc_ref(v_toMonadRef_1690_);
v_toApplicative_1691_ = lean_ctor_get(v_inst_1661_, 0);
lean_inc_ref(v_toApplicative_1691_);
v_toBind_1692_ = lean_ctor_get(v_inst_1661_, 1);
lean_inc_n(v_toBind_1692_, 3);
lean_dec_ref(v_inst_1661_);
v_getCurrMacroScope_1693_ = lean_ctor_get(v_inst_1662_, 1);
lean_inc(v_getCurrMacroScope_1693_);
v_getContext_1694_ = lean_ctor_get(v_inst_1662_, 2);
lean_inc(v_getContext_1694_);
lean_dec_ref(v_inst_1662_);
v_getRef_1695_ = lean_ctor_get(v_toMonadRef_1690_, 0);
lean_inc(v_getRef_1695_);
lean_dec_ref(v_toMonadRef_1690_);
v_toPure_1696_ = lean_ctor_get(v_toApplicative_1691_, 1);
lean_inc_n(v_toPure_1696_, 2);
lean_dec_ref(v_toApplicative_1691_);
v___f_1697_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__8), 5, 4);
lean_closure_set(v___f_1697_, 0, v_toPure_1696_);
lean_closure_set(v___f_1697_, 1, v_toBind_1692_);
lean_closure_set(v___f_1697_, 2, v_getContext_1694_);
lean_closure_set(v___f_1697_, 3, v_getCurrMacroScope_1693_);
v___f_1698_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__5___boxed), 2, 1);
lean_closure_set(v___f_1698_, 0, v_toPure_1696_);
v___x_1699_ = lean_apply_4(v_toBind_1692_, lean_box(0), lean_box(0), v_getRef_1695_, v___f_1698_);
v___x_1700_ = lean_apply_4(v_toBind_1692_, lean_box(0), lean_box(0), v___x_1699_, v___f_1697_);
return v___x_1700_;
}
else
{
lean_object* v_toMonadRef_1701_; lean_object* v_toApplicative_1702_; lean_object* v_val_1703_; lean_object* v_toBind_1704_; lean_object* v_getCurrMacroScope_1705_; lean_object* v_getContext_1706_; lean_object* v_getRef_1707_; lean_object* v_toPure_1708_; lean_object* v___f_1709_; lean_object* v___f_1710_; lean_object* v___x_1711_; lean_object* v___x_1712_; 
v_toMonadRef_1701_ = lean_ctor_get(v_inst_1662_, 0);
lean_inc_ref(v_toMonadRef_1701_);
v_toApplicative_1702_ = lean_ctor_get(v_inst_1661_, 0);
lean_inc_ref(v_toApplicative_1702_);
v_val_1703_ = lean_ctor_get(v_configStx_x3f_1664_, 0);
lean_inc(v_val_1703_);
lean_dec_ref_known(v_configStx_x3f_1664_, 1);
v_toBind_1704_ = lean_ctor_get(v_inst_1661_, 1);
lean_inc_n(v_toBind_1704_, 3);
lean_dec_ref(v_inst_1661_);
v_getCurrMacroScope_1705_ = lean_ctor_get(v_inst_1662_, 1);
lean_inc(v_getCurrMacroScope_1705_);
v_getContext_1706_ = lean_ctor_get(v_inst_1662_, 2);
lean_inc(v_getContext_1706_);
lean_dec_ref(v_inst_1662_);
v_getRef_1707_ = lean_ctor_get(v_toMonadRef_1701_, 0);
lean_inc(v_getRef_1707_);
lean_dec_ref(v_toMonadRef_1701_);
v_toPure_1708_ = lean_ctor_get(v_toApplicative_1702_, 1);
lean_inc_n(v_toPure_1708_, 2);
lean_dec_ref(v_toApplicative_1702_);
v___f_1709_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__11), 6, 5);
lean_closure_set(v___f_1709_, 0, v_val_1703_);
lean_closure_set(v___f_1709_, 1, v_toPure_1708_);
lean_closure_set(v___f_1709_, 2, v_toBind_1704_);
lean_closure_set(v___f_1709_, 3, v_getContext_1706_);
lean_closure_set(v___f_1709_, 4, v_getCurrMacroScope_1705_);
v___f_1710_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__5___boxed), 2, 1);
lean_closure_set(v___f_1710_, 0, v_toPure_1708_);
v___x_1711_ = lean_apply_4(v_toBind_1704_, lean_box(0), lean_box(0), v_getRef_1707_, v___f_1710_);
v___x_1712_ = lean_apply_4(v_toBind_1704_, lean_box(0), lean_box(0), v___x_1711_, v___f_1709_);
return v___x_1712_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___boxed(lean_object* v_inst_1713_, lean_object* v_inst_1714_, lean_object* v_simpAll_1715_, lean_object* v_configStx_x3f_1716_){
_start:
{
uint8_t v_simpAll_boxed_1717_; lean_object* v_res_1718_; 
v_simpAll_boxed_1717_ = lean_unbox(v_simpAll_1715_);
v_res_1718_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg(v_inst_1713_, v_inst_1714_, v_simpAll_boxed_1717_, v_configStx_x3f_1716_);
return v_res_1718_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx(lean_object* v_m_1719_, lean_object* v_inst_1720_, lean_object* v_inst_1721_, uint8_t v_simpAll_1722_, lean_object* v_configStx_x3f_1723_){
_start:
{
lean_object* v___x_1724_; 
v___x_1724_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg(v_inst_1720_, v_inst_1721_, v_simpAll_1722_, v_configStx_x3f_1723_);
return v___x_1724_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___boxed(lean_object* v_m_1725_, lean_object* v_inst_1726_, lean_object* v_inst_1727_, lean_object* v_simpAll_1728_, lean_object* v_configStx_x3f_1729_){
_start:
{
uint8_t v_simpAll_boxed_1730_; lean_object* v_res_1731_; 
v_simpAll_boxed_1730_ = lean_unbox(v_simpAll_1728_);
v_res_1731_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx(v_m_1725_, v_inst_1726_, v_inst_1727_, v_simpAll_boxed_1730_, v_configStx_x3f_1729_);
return v_res_1731_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnlyStx_spec__0___redArg(uint8_t v_simpAll_1732_, lean_object* v_configStx_x3f_1733_, lean_object* v___y_1734_){
_start:
{
if (v_simpAll_1732_ == 0)
{
if (lean_obj_tag(v_configStx_x3f_1733_) == 0)
{
lean_object* v_ref_1736_; lean_object* v___x_1737_; lean_object* v___x_1738_; lean_object* v___x_1739_; lean_object* v___x_1740_; lean_object* v___x_1741_; lean_object* v___x_1742_; lean_object* v___x_1743_; lean_object* v___x_1744_; lean_object* v___x_1745_; lean_object* v___x_1746_; lean_object* v___x_1747_; lean_object* v___x_1748_; lean_object* v___x_1749_; lean_object* v___x_1750_; lean_object* v___x_1751_; lean_object* v___x_1752_; lean_object* v___x_1753_; lean_object* v___x_1754_; lean_object* v___x_1755_; lean_object* v___x_1756_; 
v_ref_1736_ = lean_ctor_get(v___y_1734_, 5);
v___x_1737_ = l_Lean_SourceInfo_fromRef(v_ref_1736_, v_simpAll_1732_);
v___x_1738_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__0));
v___x_1739_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__1));
lean_inc_n(v___x_1737_, 8);
v___x_1740_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1740_, 0, v___x_1737_);
lean_ctor_set(v___x_1740_, 1, v___x_1738_);
v___x_1741_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__3));
v___x_1742_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_1743_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
v___x_1744_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1744_, 0, v___x_1737_);
lean_ctor_set(v___x_1744_, 1, v___x_1742_);
lean_ctor_set(v___x_1744_, 2, v___x_1743_);
lean_inc_ref_n(v___x_1744_, 3);
v___x_1745_ = l_Lean_Syntax_node1(v___x_1737_, v___x_1741_, v___x_1744_);
v___x_1746_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__1));
v___x_1747_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__2));
v___x_1748_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1748_, 0, v___x_1737_);
lean_ctor_set(v___x_1748_, 1, v___x_1747_);
v___x_1749_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__5));
v___x_1750_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__6));
v___x_1751_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1751_, 0, v___x_1737_);
lean_ctor_set(v___x_1751_, 1, v___x_1750_);
v___x_1752_ = l_Lean_Syntax_node1(v___x_1737_, v___x_1749_, v___x_1751_);
v___x_1753_ = l_Lean_Syntax_node2(v___x_1737_, v___x_1746_, v___x_1748_, v___x_1752_);
v___x_1754_ = l_Lean_Syntax_node1(v___x_1737_, v___x_1742_, v___x_1753_);
v___x_1755_ = l_Lean_Syntax_node6(v___x_1737_, v___x_1739_, v___x_1740_, v___x_1745_, v___x_1744_, v___x_1744_, v___x_1744_, v___x_1754_);
v___x_1756_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1756_, 0, v___x_1755_);
return v___x_1756_;
}
else
{
lean_object* v_val_1757_; lean_object* v___x_1759_; uint8_t v_isShared_1760_; uint8_t v_isSharedCheck_1796_; 
v_val_1757_ = lean_ctor_get(v_configStx_x3f_1733_, 0);
v_isSharedCheck_1796_ = !lean_is_exclusive(v_configStx_x3f_1733_);
if (v_isSharedCheck_1796_ == 0)
{
v___x_1759_ = v_configStx_x3f_1733_;
v_isShared_1760_ = v_isSharedCheck_1796_;
goto v_resetjp_1758_;
}
else
{
lean_inc(v_val_1757_);
lean_dec(v_configStx_x3f_1733_);
v___x_1759_ = lean_box(0);
v_isShared_1760_ = v_isSharedCheck_1796_;
goto v_resetjp_1758_;
}
v_resetjp_1758_:
{
lean_object* v_ref_1761_; lean_object* v___x_1762_; lean_object* v___x_1763_; lean_object* v___x_1764_; lean_object* v___x_1765_; lean_object* v___x_1766_; lean_object* v___x_1767_; lean_object* v___x_1768_; lean_object* v___x_1769_; lean_object* v___x_1770_; lean_object* v___x_1771_; lean_object* v___x_1772_; lean_object* v___x_1773_; lean_object* v___x_1774_; lean_object* v___x_1775_; lean_object* v___x_1776_; lean_object* v___x_1777_; lean_object* v___x_1778_; lean_object* v___x_1779_; lean_object* v___x_1780_; lean_object* v___x_1781_; lean_object* v___x_1782_; lean_object* v___x_1783_; lean_object* v___x_1784_; lean_object* v___x_1785_; lean_object* v___x_1786_; lean_object* v___x_1787_; lean_object* v___x_1788_; lean_object* v___x_1789_; lean_object* v___x_1790_; lean_object* v___x_1791_; lean_object* v___x_1792_; lean_object* v___x_1794_; 
v_ref_1761_ = lean_ctor_get(v___y_1734_, 5);
v___x_1762_ = l_Lean_SourceInfo_fromRef(v_ref_1761_, v_simpAll_1732_);
v___x_1763_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__0));
v___x_1764_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__1));
lean_inc_n(v___x_1762_, 14);
v___x_1765_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1765_, 0, v___x_1762_);
lean_ctor_set(v___x_1765_, 1, v___x_1763_);
v___x_1766_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__3));
v___x_1767_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_1768_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__1));
v___x_1769_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__3));
v___x_1770_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__4));
v___x_1771_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1771_, 0, v___x_1762_);
lean_ctor_set(v___x_1771_, 1, v___x_1770_);
v___x_1772_ = lean_obj_once(&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__7, &lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__7_once, _init_lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__7);
v___x_1773_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__15));
v___x_1774_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1774_, 0, v___x_1762_);
lean_ctor_set(v___x_1774_, 1, v___x_1773_);
v___x_1775_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__8));
v___x_1776_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1776_, 0, v___x_1762_);
lean_ctor_set(v___x_1776_, 1, v___x_1775_);
v___x_1777_ = l_Lean_Syntax_node5(v___x_1762_, v___x_1769_, v___x_1771_, v___x_1772_, v___x_1774_, v_val_1757_, v___x_1776_);
v___x_1778_ = l_Lean_Syntax_node1(v___x_1762_, v___x_1768_, v___x_1777_);
v___x_1779_ = l_Lean_Syntax_node1(v___x_1762_, v___x_1767_, v___x_1778_);
v___x_1780_ = l_Lean_Syntax_node1(v___x_1762_, v___x_1766_, v___x_1779_);
v___x_1781_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
v___x_1782_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1782_, 0, v___x_1762_);
lean_ctor_set(v___x_1782_, 1, v___x_1767_);
lean_ctor_set(v___x_1782_, 2, v___x_1781_);
v___x_1783_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__1));
v___x_1784_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__2));
v___x_1785_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1785_, 0, v___x_1762_);
lean_ctor_set(v___x_1785_, 1, v___x_1784_);
v___x_1786_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__5));
v___x_1787_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__6));
v___x_1788_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1788_, 0, v___x_1762_);
lean_ctor_set(v___x_1788_, 1, v___x_1787_);
v___x_1789_ = l_Lean_Syntax_node1(v___x_1762_, v___x_1786_, v___x_1788_);
v___x_1790_ = l_Lean_Syntax_node2(v___x_1762_, v___x_1783_, v___x_1785_, v___x_1789_);
v___x_1791_ = l_Lean_Syntax_node1(v___x_1762_, v___x_1767_, v___x_1790_);
lean_inc_ref_n(v___x_1782_, 2);
v___x_1792_ = l_Lean_Syntax_node6(v___x_1762_, v___x_1764_, v___x_1765_, v___x_1780_, v___x_1782_, v___x_1782_, v___x_1782_, v___x_1791_);
if (v_isShared_1760_ == 0)
{
lean_ctor_set_tag(v___x_1759_, 0);
lean_ctor_set(v___x_1759_, 0, v___x_1792_);
v___x_1794_ = v___x_1759_;
goto v_reusejp_1793_;
}
else
{
lean_object* v_reuseFailAlloc_1795_; 
v_reuseFailAlloc_1795_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1795_, 0, v___x_1792_);
v___x_1794_ = v_reuseFailAlloc_1795_;
goto v_reusejp_1793_;
}
v_reusejp_1793_:
{
return v___x_1794_;
}
}
}
}
else
{
if (lean_obj_tag(v_configStx_x3f_1733_) == 0)
{
lean_object* v_ref_1797_; uint8_t v___x_1798_; lean_object* v___x_1799_; lean_object* v___x_1800_; lean_object* v___x_1801_; lean_object* v___x_1802_; lean_object* v___x_1803_; lean_object* v___x_1804_; lean_object* v___x_1805_; lean_object* v___x_1806_; lean_object* v___x_1807_; lean_object* v___x_1808_; lean_object* v___x_1809_; 
v_ref_1797_ = lean_ctor_get(v___y_1734_, 5);
v___x_1798_ = 0;
v___x_1799_ = l_Lean_SourceInfo_fromRef(v_ref_1797_, v___x_1798_);
v___x_1800_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__1));
v___x_1801_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__2));
lean_inc_n(v___x_1799_, 3);
v___x_1802_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1802_, 0, v___x_1799_);
lean_ctor_set(v___x_1802_, 1, v___x_1801_);
v___x_1803_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__3));
v___x_1804_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_1805_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
v___x_1806_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1806_, 0, v___x_1799_);
lean_ctor_set(v___x_1806_, 1, v___x_1804_);
lean_ctor_set(v___x_1806_, 2, v___x_1805_);
lean_inc_ref_n(v___x_1806_, 3);
v___x_1807_ = l_Lean_Syntax_node1(v___x_1799_, v___x_1803_, v___x_1806_);
v___x_1808_ = l_Lean_Syntax_node5(v___x_1799_, v___x_1800_, v___x_1802_, v___x_1807_, v___x_1806_, v___x_1806_, v___x_1806_);
v___x_1809_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1809_, 0, v___x_1808_);
return v___x_1809_;
}
else
{
lean_object* v_val_1810_; lean_object* v___x_1812_; uint8_t v_isShared_1813_; uint8_t v_isSharedCheck_1841_; 
v_val_1810_ = lean_ctor_get(v_configStx_x3f_1733_, 0);
v_isSharedCheck_1841_ = !lean_is_exclusive(v_configStx_x3f_1733_);
if (v_isSharedCheck_1841_ == 0)
{
v___x_1812_ = v_configStx_x3f_1733_;
v_isShared_1813_ = v_isSharedCheck_1841_;
goto v_resetjp_1811_;
}
else
{
lean_inc(v_val_1810_);
lean_dec(v_configStx_x3f_1733_);
v___x_1812_ = lean_box(0);
v_isShared_1813_ = v_isSharedCheck_1841_;
goto v_resetjp_1811_;
}
v_resetjp_1811_:
{
lean_object* v_ref_1814_; uint8_t v___x_1815_; lean_object* v___x_1816_; lean_object* v___x_1817_; lean_object* v___x_1818_; lean_object* v___x_1819_; lean_object* v___x_1820_; lean_object* v___x_1821_; lean_object* v___x_1822_; lean_object* v___x_1823_; lean_object* v___x_1824_; lean_object* v___x_1825_; lean_object* v___x_1826_; lean_object* v___x_1827_; lean_object* v___x_1828_; lean_object* v___x_1829_; lean_object* v___x_1830_; lean_object* v___x_1831_; lean_object* v___x_1832_; lean_object* v___x_1833_; lean_object* v___x_1834_; lean_object* v___x_1835_; lean_object* v___x_1836_; lean_object* v___x_1837_; lean_object* v___x_1839_; 
v_ref_1814_ = lean_ctor_get(v___y_1734_, 5);
v___x_1815_ = 0;
v___x_1816_ = l_Lean_SourceInfo_fromRef(v_ref_1814_, v___x_1815_);
v___x_1817_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__1));
v___x_1818_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__2));
lean_inc_n(v___x_1816_, 9);
v___x_1819_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1819_, 0, v___x_1816_);
lean_ctor_set(v___x_1819_, 1, v___x_1818_);
v___x_1820_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__3));
v___x_1821_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_1822_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__1));
v___x_1823_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__3));
v___x_1824_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__4));
v___x_1825_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1825_, 0, v___x_1816_);
lean_ctor_set(v___x_1825_, 1, v___x_1824_);
v___x_1826_ = lean_obj_once(&lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__7, &lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__7_once, _init_lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__7);
v___x_1827_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__15));
v___x_1828_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1828_, 0, v___x_1816_);
lean_ctor_set(v___x_1828_, 1, v___x_1827_);
v___x_1829_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__4___closed__8));
v___x_1830_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1830_, 0, v___x_1816_);
lean_ctor_set(v___x_1830_, 1, v___x_1829_);
v___x_1831_ = l_Lean_Syntax_node5(v___x_1816_, v___x_1823_, v___x_1825_, v___x_1826_, v___x_1828_, v_val_1810_, v___x_1830_);
v___x_1832_ = l_Lean_Syntax_node1(v___x_1816_, v___x_1822_, v___x_1831_);
v___x_1833_ = l_Lean_Syntax_node1(v___x_1816_, v___x_1821_, v___x_1832_);
v___x_1834_ = l_Lean_Syntax_node1(v___x_1816_, v___x_1820_, v___x_1833_);
v___x_1835_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
v___x_1836_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1836_, 0, v___x_1816_);
lean_ctor_set(v___x_1836_, 1, v___x_1821_);
lean_ctor_set(v___x_1836_, 2, v___x_1835_);
lean_inc_ref_n(v___x_1836_, 2);
v___x_1837_ = l_Lean_Syntax_node5(v___x_1816_, v___x_1817_, v___x_1819_, v___x_1834_, v___x_1836_, v___x_1836_, v___x_1836_);
if (v_isShared_1813_ == 0)
{
lean_ctor_set_tag(v___x_1812_, 0);
lean_ctor_set(v___x_1812_, 0, v___x_1837_);
v___x_1839_ = v___x_1812_;
goto v_reusejp_1838_;
}
else
{
lean_object* v_reuseFailAlloc_1840_; 
v_reuseFailAlloc_1840_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1840_, 0, v___x_1837_);
v___x_1839_ = v_reuseFailAlloc_1840_;
goto v_reusejp_1838_;
}
v_reusejp_1838_:
{
return v___x_1839_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnlyStx_spec__0___redArg___boxed(lean_object* v_simpAll_1842_, lean_object* v_configStx_x3f_1843_, lean_object* v___y_1844_, lean_object* v___y_1845_){
_start:
{
uint8_t v_simpAll_boxed_1846_; lean_object* v_res_1847_; 
v_simpAll_boxed_1846_ = lean_unbox(v_simpAll_1842_);
v_res_1847_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnlyStx_spec__0___redArg(v_simpAll_boxed_1846_, v_configStx_x3f_1843_, v___y_1844_);
lean_dec_ref(v___y_1844_);
return v_res_1847_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnlyStx_spec__0(uint8_t v_simpAll_1848_, lean_object* v_configStx_x3f_1849_, lean_object* v___y_1850_, lean_object* v___y_1851_, lean_object* v___y_1852_, lean_object* v___y_1853_){
_start:
{
lean_object* v___x_1855_; 
v___x_1855_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnlyStx_spec__0___redArg(v_simpAll_1848_, v_configStx_x3f_1849_, v___y_1852_);
return v___x_1855_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnlyStx_spec__0___boxed(lean_object* v_simpAll_1856_, lean_object* v_configStx_x3f_1857_, lean_object* v___y_1858_, lean_object* v___y_1859_, lean_object* v___y_1860_, lean_object* v___y_1861_, lean_object* v___y_1862_){
_start:
{
uint8_t v_simpAll_boxed_1863_; lean_object* v_res_1864_; 
v_simpAll_boxed_1863_ = lean_unbox(v_simpAll_1856_);
v_res_1864_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnlyStx_spec__0(v_simpAll_boxed_1863_, v_configStx_x3f_1857_, v___y_1858_, v___y_1859_, v___y_1860_, v___y_1861_);
lean_dec(v___y_1861_);
lean_dec_ref(v___y_1860_);
lean_dec(v___y_1859_);
lean_dec_ref(v___y_1858_);
return v_res_1864_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnlyStx(uint8_t v_simpAll_1865_, lean_object* v_inGoal_1866_, lean_object* v_configStx_x3f_1867_, lean_object* v_usedTheorems_1868_, lean_object* v_a_1869_, lean_object* v_a_1870_, lean_object* v_a_1871_, lean_object* v_a_1872_){
_start:
{
lean_object* v___x_1874_; lean_object* v_a_1875_; lean_object* v___x_1876_; lean_object* v___x_1877_; 
v___x_1874_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnlyStx_spec__0___redArg(v_simpAll_1865_, v_configStx_x3f_1867_, v_a_1871_);
v_a_1875_ = lean_ctor_get(v___x_1874_, 0);
lean_inc(v_a_1875_);
lean_dec_ref(v___x_1874_);
v___x_1876_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_mkSimpOnly___boxed), 7, 2);
lean_closure_set(v___x_1876_, 0, v_a_1875_);
lean_closure_set(v___x_1876_, 1, v_usedTheorems_1868_);
v___x_1877_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___redArg(v_inGoal_1866_, v___x_1876_, v_a_1869_, v_a_1870_, v_a_1871_, v_a_1872_);
if (lean_obj_tag(v___x_1877_) == 0)
{
lean_object* v_a_1878_; lean_object* v___x_1880_; uint8_t v_isShared_1881_; uint8_t v_isSharedCheck_1885_; 
v_a_1878_ = lean_ctor_get(v___x_1877_, 0);
v_isSharedCheck_1885_ = !lean_is_exclusive(v___x_1877_);
if (v_isSharedCheck_1885_ == 0)
{
v___x_1880_ = v___x_1877_;
v_isShared_1881_ = v_isSharedCheck_1885_;
goto v_resetjp_1879_;
}
else
{
lean_inc(v_a_1878_);
lean_dec(v___x_1877_);
v___x_1880_ = lean_box(0);
v_isShared_1881_ = v_isSharedCheck_1885_;
goto v_resetjp_1879_;
}
v_resetjp_1879_:
{
lean_object* v___x_1883_; 
if (v_isShared_1881_ == 0)
{
v___x_1883_ = v___x_1880_;
goto v_reusejp_1882_;
}
else
{
lean_object* v_reuseFailAlloc_1884_; 
v_reuseFailAlloc_1884_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1884_, 0, v_a_1878_);
v___x_1883_ = v_reuseFailAlloc_1884_;
goto v_reusejp_1882_;
}
v_reusejp_1882_:
{
return v___x_1883_;
}
}
}
else
{
lean_object* v_a_1886_; lean_object* v___x_1888_; uint8_t v_isShared_1889_; uint8_t v_isSharedCheck_1893_; 
v_a_1886_ = lean_ctor_get(v___x_1877_, 0);
v_isSharedCheck_1893_ = !lean_is_exclusive(v___x_1877_);
if (v_isSharedCheck_1893_ == 0)
{
v___x_1888_ = v___x_1877_;
v_isShared_1889_ = v_isSharedCheck_1893_;
goto v_resetjp_1887_;
}
else
{
lean_inc(v_a_1886_);
lean_dec(v___x_1877_);
v___x_1888_ = lean_box(0);
v_isShared_1889_ = v_isSharedCheck_1893_;
goto v_resetjp_1887_;
}
v_resetjp_1887_:
{
lean_object* v___x_1891_; 
if (v_isShared_1889_ == 0)
{
v___x_1891_ = v___x_1888_;
goto v_reusejp_1890_;
}
else
{
lean_object* v_reuseFailAlloc_1892_; 
v_reuseFailAlloc_1892_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1892_, 0, v_a_1886_);
v___x_1891_ = v_reuseFailAlloc_1892_;
goto v_reusejp_1890_;
}
v_reusejp_1890_:
{
return v___x_1891_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnlyStx___boxed(lean_object* v_simpAll_1894_, lean_object* v_inGoal_1895_, lean_object* v_configStx_x3f_1896_, lean_object* v_usedTheorems_1897_, lean_object* v_a_1898_, lean_object* v_a_1899_, lean_object* v_a_1900_, lean_object* v_a_1901_, lean_object* v_a_1902_){
_start:
{
uint8_t v_simpAll_boxed_1903_; lean_object* v_res_1904_; 
v_simpAll_boxed_1903_ = lean_unbox(v_simpAll_1894_);
v_res_1904_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnlyStx(v_simpAll_boxed_1903_, v_inGoal_1895_, v_configStx_x3f_1896_, v_usedTheorems_1897_, v_a_1898_, v_a_1899_, v_a_1900_, v_a_1901_);
lean_dec(v_a_1901_);
lean_dec_ref(v_a_1900_);
lean_dec(v_a_1899_);
lean_dec_ref(v_a_1898_);
return v_res_1904_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnly(uint8_t v_simpAll_1905_, lean_object* v_inGoal_1906_, lean_object* v_configStx_x3f_1907_, lean_object* v_usedTheorems_1908_, lean_object* v_a_1909_, lean_object* v_a_1910_, lean_object* v_a_1911_, lean_object* v_a_1912_){
_start:
{
lean_object* v___x_1914_; 
v___x_1914_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnlyStx(v_simpAll_1905_, v_inGoal_1906_, v_configStx_x3f_1907_, v_usedTheorems_1908_, v_a_1909_, v_a_1910_, v_a_1911_, v_a_1912_);
if (lean_obj_tag(v___x_1914_) == 0)
{
lean_object* v_a_1915_; lean_object* v___x_1917_; uint8_t v_isShared_1918_; uint8_t v_isSharedCheck_1923_; 
v_a_1915_ = lean_ctor_get(v___x_1914_, 0);
v_isSharedCheck_1923_ = !lean_is_exclusive(v___x_1914_);
if (v_isSharedCheck_1923_ == 0)
{
v___x_1917_ = v___x_1914_;
v_isShared_1918_ = v_isSharedCheck_1923_;
goto v_resetjp_1916_;
}
else
{
lean_inc(v_a_1915_);
lean_dec(v___x_1914_);
v___x_1917_ = lean_box(0);
v_isShared_1918_ = v_isSharedCheck_1923_;
goto v_resetjp_1916_;
}
v_resetjp_1916_:
{
lean_object* v___x_1919_; lean_object* v___x_1921_; 
v___x_1919_ = lp_aesop_Aesop_Script_Tactic_unstructured(v_a_1915_);
if (v_isShared_1918_ == 0)
{
lean_ctor_set(v___x_1917_, 0, v___x_1919_);
v___x_1921_ = v___x_1917_;
goto v_reusejp_1920_;
}
else
{
lean_object* v_reuseFailAlloc_1922_; 
v_reuseFailAlloc_1922_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1922_, 0, v___x_1919_);
v___x_1921_ = v_reuseFailAlloc_1922_;
goto v_reusejp_1920_;
}
v_reusejp_1920_:
{
return v___x_1921_;
}
}
}
else
{
lean_object* v_a_1924_; lean_object* v___x_1926_; uint8_t v_isShared_1927_; uint8_t v_isSharedCheck_1931_; 
v_a_1924_ = lean_ctor_get(v___x_1914_, 0);
v_isSharedCheck_1931_ = !lean_is_exclusive(v___x_1914_);
if (v_isSharedCheck_1931_ == 0)
{
v___x_1926_ = v___x_1914_;
v_isShared_1927_ = v_isSharedCheck_1931_;
goto v_resetjp_1925_;
}
else
{
lean_inc(v_a_1924_);
lean_dec(v___x_1914_);
v___x_1926_ = lean_box(0);
v_isShared_1927_ = v_isSharedCheck_1931_;
goto v_resetjp_1925_;
}
v_resetjp_1925_:
{
lean_object* v___x_1929_; 
if (v_isShared_1927_ == 0)
{
v___x_1929_ = v___x_1926_;
goto v_reusejp_1928_;
}
else
{
lean_object* v_reuseFailAlloc_1930_; 
v_reuseFailAlloc_1930_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1930_, 0, v_a_1924_);
v___x_1929_ = v_reuseFailAlloc_1930_;
goto v_reusejp_1928_;
}
v_reusejp_1928_:
{
return v___x_1929_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnly___boxed(lean_object* v_simpAll_1932_, lean_object* v_inGoal_1933_, lean_object* v_configStx_x3f_1934_, lean_object* v_usedTheorems_1935_, lean_object* v_a_1936_, lean_object* v_a_1937_, lean_object* v_a_1938_, lean_object* v_a_1939_, lean_object* v_a_1940_){
_start:
{
uint8_t v_simpAll_boxed_1941_; lean_object* v_res_1942_; 
v_simpAll_boxed_1941_ = lean_unbox(v_simpAll_1932_);
v_res_1942_ = lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnly(v_simpAll_boxed_1941_, v_inGoal_1933_, v_configStx_x3f_1934_, v_usedTheorems_1935_, v_a_1936_, v_a_1937_, v_a_1938_, v_a_1939_);
lean_dec(v_a_1939_);
lean_dec_ref(v_a_1938_);
lean_dec(v_a_1937_);
lean_dec_ref(v_a_1936_);
return v_res_1942_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0_spec__0_spec__1___redArg(lean_object* v_keys_1943_, lean_object* v_i_1944_, lean_object* v_k_1945_){
_start:
{
lean_object* v___x_1946_; uint8_t v___x_1947_; 
v___x_1946_ = lean_array_get_size(v_keys_1943_);
v___x_1947_ = lean_nat_dec_lt(v_i_1944_, v___x_1946_);
if (v___x_1947_ == 0)
{
lean_dec(v_i_1944_);
return v___x_1947_;
}
else
{
lean_object* v_k_x27_1948_; uint8_t v___x_1949_; 
v_k_x27_1948_ = lean_array_fget_borrowed(v_keys_1943_, v_i_1944_);
v___x_1949_ = lean_name_eq(v_k_1945_, v_k_x27_1948_);
if (v___x_1949_ == 0)
{
lean_object* v___x_1950_; lean_object* v___x_1951_; 
v___x_1950_ = lean_unsigned_to_nat(1u);
v___x_1951_ = lean_nat_add(v_i_1944_, v___x_1950_);
lean_dec(v_i_1944_);
v_i_1944_ = v___x_1951_;
goto _start;
}
else
{
lean_dec(v_i_1944_);
return v___x_1949_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_keys_1953_, lean_object* v_i_1954_, lean_object* v_k_1955_){
_start:
{
uint8_t v_res_1956_; lean_object* v_r_1957_; 
v_res_1956_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0_spec__0_spec__1___redArg(v_keys_1953_, v_i_1954_, v_k_1955_);
lean_dec(v_k_1955_);
lean_dec_ref(v_keys_1953_);
v_r_1957_ = lean_box(v_res_1956_);
return v_r_1957_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0_spec__0___redArg(lean_object* v_x_1958_, size_t v_x_1959_, lean_object* v_x_1960_){
_start:
{
if (lean_obj_tag(v_x_1958_) == 0)
{
lean_object* v_es_1961_; lean_object* v___x_1962_; size_t v___x_1963_; size_t v___x_1964_; lean_object* v_j_1965_; lean_object* v___x_1966_; 
v_es_1961_ = lean_ctor_get(v_x_1958_, 0);
v___x_1962_ = lean_box(2);
v___x_1963_ = ((size_t)31ULL);
v___x_1964_ = lean_usize_land(v_x_1959_, v___x_1963_);
v_j_1965_ = lean_usize_to_nat(v___x_1964_);
v___x_1966_ = lean_array_get_borrowed(v___x_1962_, v_es_1961_, v_j_1965_);
lean_dec(v_j_1965_);
switch(lean_obj_tag(v___x_1966_))
{
case 0:
{
lean_object* v_key_1967_; uint8_t v___x_1968_; 
v_key_1967_ = lean_ctor_get(v___x_1966_, 0);
v___x_1968_ = lean_name_eq(v_x_1960_, v_key_1967_);
return v___x_1968_;
}
case 1:
{
lean_object* v_node_1969_; size_t v___x_1970_; size_t v___x_1971_; 
v_node_1969_ = lean_ctor_get(v___x_1966_, 0);
v___x_1970_ = ((size_t)5ULL);
v___x_1971_ = lean_usize_shift_right(v_x_1959_, v___x_1970_);
v_x_1958_ = v_node_1969_;
v_x_1959_ = v___x_1971_;
goto _start;
}
default: 
{
uint8_t v___x_1973_; 
v___x_1973_ = 0;
return v___x_1973_;
}
}
}
else
{
lean_object* v_ks_1974_; lean_object* v___x_1975_; uint8_t v___x_1976_; 
v_ks_1974_ = lean_ctor_get(v_x_1958_, 0);
v___x_1975_ = lean_unsigned_to_nat(0u);
v___x_1976_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0_spec__0_spec__1___redArg(v_ks_1974_, v___x_1975_, v_x_1960_);
return v___x_1976_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0_spec__0___redArg___boxed(lean_object* v_x_1977_, lean_object* v_x_1978_, lean_object* v_x_1979_){
_start:
{
size_t v_x_478__boxed_1980_; uint8_t v_res_1981_; lean_object* v_r_1982_; 
v_x_478__boxed_1980_ = lean_unbox_usize(v_x_1978_);
lean_dec(v_x_1978_);
v_res_1981_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0_spec__0___redArg(v_x_1977_, v_x_478__boxed_1980_, v_x_1979_);
lean_dec(v_x_1979_);
lean_dec_ref(v_x_1977_);
v_r_1982_ = lean_box(v_res_1981_);
return v_r_1982_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0___redArg(lean_object* v_x_1983_, lean_object* v_x_1984_){
_start:
{
uint64_t v___y_1986_; 
if (lean_obj_tag(v_x_1984_) == 0)
{
uint64_t v___x_1989_; 
v___x_1989_ = 1723ULL;
v___y_1986_ = v___x_1989_;
goto v___jp_1985_;
}
else
{
uint64_t v_hash_1990_; 
v_hash_1990_ = lean_ctor_get_uint64(v_x_1984_, sizeof(void*)*2);
v___y_1986_ = v_hash_1990_;
goto v___jp_1985_;
}
v___jp_1985_:
{
size_t v___x_1987_; uint8_t v___x_1988_; 
v___x_1987_ = lean_uint64_to_usize(v___y_1986_);
v___x_1988_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0_spec__0___redArg(v_x_1983_, v___x_1987_, v_x_1984_);
return v___x_1988_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0___redArg___boxed(lean_object* v_x_1991_, lean_object* v_x_1992_){
_start:
{
uint8_t v_res_1993_; lean_object* v_r_1994_; 
v_res_1993_ = lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0___redArg(v_x_1991_, v_x_1992_);
lean_dec(v_x_1992_);
lean_dec_ref(v_x_1991_);
v_r_1994_ = lean_box(v_res_1993_);
return v_r_1994_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1_spec__2_spec__4___redArg(lean_object* v_keys_1995_, lean_object* v_i_1996_, lean_object* v_k_1997_){
_start:
{
uint8_t v___y_2003_; lean_object* v___x_2004_; uint8_t v___x_2005_; 
v___x_2004_ = lean_array_get_size(v_keys_1995_);
v___x_2005_ = lean_nat_dec_lt(v_i_1996_, v___x_2004_);
if (v___x_2005_ == 0)
{
lean_dec(v_i_1996_);
return v___x_2005_;
}
else
{
lean_object* v_k_x27_2006_; 
v_k_x27_2006_ = lean_array_fget_borrowed(v_keys_1995_, v_i_1996_);
if (lean_obj_tag(v_k_1997_) == 0)
{
if (lean_obj_tag(v_k_x27_2006_) == 0)
{
lean_object* v_declName_2007_; uint8_t v_inv_2008_; lean_object* v_declName_2009_; uint8_t v_inv_2010_; uint8_t v___x_2011_; 
v_declName_2007_ = lean_ctor_get(v_k_1997_, 0);
v_inv_2008_ = lean_ctor_get_uint8(v_k_1997_, sizeof(void*)*1 + 1);
v_declName_2009_ = lean_ctor_get(v_k_x27_2006_, 0);
v_inv_2010_ = lean_ctor_get_uint8(v_k_x27_2006_, sizeof(void*)*1 + 1);
v___x_2011_ = lean_name_eq(v_declName_2007_, v_declName_2009_);
if (v___x_2011_ == 0)
{
v___y_2003_ = v___x_2011_;
goto v___jp_2002_;
}
else
{
if (v_inv_2008_ == 0)
{
if (v_inv_2010_ == 0)
{
v___y_2003_ = v___x_2011_;
goto v___jp_2002_;
}
else
{
goto v___jp_1998_;
}
}
else
{
v___y_2003_ = v_inv_2010_;
goto v___jp_2002_;
}
}
}
else
{
goto v___jp_1998_;
}
}
else
{
if (lean_obj_tag(v_k_x27_2006_) == 0)
{
goto v___jp_1998_;
}
else
{
lean_object* v___x_2012_; lean_object* v___x_2013_; uint8_t v___x_2014_; 
v___x_2012_ = l_Lean_Meta_Origin_key(v_k_1997_);
v___x_2013_ = l_Lean_Meta_Origin_key(v_k_x27_2006_);
v___x_2014_ = lean_name_eq(v___x_2012_, v___x_2013_);
lean_dec(v___x_2013_);
lean_dec(v___x_2012_);
v___y_2003_ = v___x_2014_;
goto v___jp_2002_;
}
}
}
v___jp_1998_:
{
lean_object* v___x_1999_; lean_object* v___x_2000_; 
v___x_1999_ = lean_unsigned_to_nat(1u);
v___x_2000_ = lean_nat_add(v_i_1996_, v___x_1999_);
lean_dec(v_i_1996_);
v_i_1996_ = v___x_2000_;
goto _start;
}
v___jp_2002_:
{
if (v___y_2003_ == 0)
{
goto v___jp_1998_;
}
else
{
lean_dec(v_i_1996_);
return v___y_2003_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1_spec__2_spec__4___redArg___boxed(lean_object* v_keys_2015_, lean_object* v_i_2016_, lean_object* v_k_2017_){
_start:
{
uint8_t v_res_2018_; lean_object* v_r_2019_; 
v_res_2018_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1_spec__2_spec__4___redArg(v_keys_2015_, v_i_2016_, v_k_2017_);
lean_dec_ref(v_k_2017_);
lean_dec_ref(v_keys_2015_);
v_r_2019_ = lean_box(v_res_2018_);
return v_r_2019_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1_spec__2___redArg(lean_object* v_x_2020_, size_t v_x_2021_, lean_object* v_x_2022_){
_start:
{
if (lean_obj_tag(v_x_2020_) == 0)
{
lean_object* v_es_2023_; lean_object* v___x_2024_; size_t v___x_2025_; size_t v___x_2026_; lean_object* v_j_2027_; lean_object* v___x_2028_; 
v_es_2023_ = lean_ctor_get(v_x_2020_, 0);
v___x_2024_ = lean_box(2);
v___x_2025_ = ((size_t)31ULL);
v___x_2026_ = lean_usize_land(v_x_2021_, v___x_2025_);
v_j_2027_ = lean_usize_to_nat(v___x_2026_);
v___x_2028_ = lean_array_get_borrowed(v___x_2024_, v_es_2023_, v_j_2027_);
lean_dec(v_j_2027_);
switch(lean_obj_tag(v___x_2028_))
{
case 0:
{
if (lean_obj_tag(v_x_2022_) == 0)
{
lean_object* v_key_2029_; 
v_key_2029_ = lean_ctor_get(v___x_2028_, 0);
if (lean_obj_tag(v_key_2029_) == 0)
{
lean_object* v_declName_2030_; uint8_t v_inv_2031_; lean_object* v_declName_2032_; uint8_t v_inv_2033_; uint8_t v___x_2034_; 
v_declName_2030_ = lean_ctor_get(v_x_2022_, 0);
v_inv_2031_ = lean_ctor_get_uint8(v_x_2022_, sizeof(void*)*1 + 1);
v_declName_2032_ = lean_ctor_get(v_key_2029_, 0);
v_inv_2033_ = lean_ctor_get_uint8(v_key_2029_, sizeof(void*)*1 + 1);
v___x_2034_ = lean_name_eq(v_declName_2030_, v_declName_2032_);
if (v___x_2034_ == 0)
{
return v___x_2034_;
}
else
{
if (v_inv_2031_ == 0)
{
if (v_inv_2033_ == 0)
{
return v___x_2034_;
}
else
{
return v_inv_2031_;
}
}
else
{
return v_inv_2033_;
}
}
}
else
{
uint8_t v___x_2035_; 
v___x_2035_ = 0;
return v___x_2035_;
}
}
else
{
lean_object* v_key_2036_; 
v_key_2036_ = lean_ctor_get(v___x_2028_, 0);
if (lean_obj_tag(v_key_2036_) == 0)
{
uint8_t v___x_2037_; 
v___x_2037_ = 0;
return v___x_2037_;
}
else
{
lean_object* v___x_2038_; lean_object* v___x_2039_; uint8_t v___x_2040_; 
v___x_2038_ = l_Lean_Meta_Origin_key(v_x_2022_);
v___x_2039_ = l_Lean_Meta_Origin_key(v_key_2036_);
v___x_2040_ = lean_name_eq(v___x_2038_, v___x_2039_);
lean_dec(v___x_2039_);
lean_dec(v___x_2038_);
return v___x_2040_;
}
}
}
case 1:
{
lean_object* v_node_2041_; size_t v___x_2042_; size_t v___x_2043_; 
v_node_2041_ = lean_ctor_get(v___x_2028_, 0);
v___x_2042_ = ((size_t)5ULL);
v___x_2043_ = lean_usize_shift_right(v_x_2021_, v___x_2042_);
v_x_2020_ = v_node_2041_;
v_x_2021_ = v___x_2043_;
goto _start;
}
default: 
{
uint8_t v___x_2045_; 
v___x_2045_ = 0;
return v___x_2045_;
}
}
}
else
{
lean_object* v_ks_2046_; lean_object* v___x_2047_; uint8_t v___x_2048_; 
v_ks_2046_ = lean_ctor_get(v_x_2020_, 0);
v___x_2047_ = lean_unsigned_to_nat(0u);
v___x_2048_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1_spec__2_spec__4___redArg(v_ks_2046_, v___x_2047_, v_x_2022_);
return v___x_2048_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1_spec__2___redArg___boxed(lean_object* v_x_2049_, lean_object* v_x_2050_, lean_object* v_x_2051_){
_start:
{
size_t v_x_571__boxed_2052_; uint8_t v_res_2053_; lean_object* v_r_2054_; 
v_x_571__boxed_2052_ = lean_unbox_usize(v_x_2050_);
lean_dec(v_x_2050_);
v_res_2053_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1_spec__2___redArg(v_x_2049_, v_x_571__boxed_2052_, v_x_2051_);
lean_dec_ref(v_x_2051_);
lean_dec_ref(v_x_2049_);
v_r_2054_ = lean_box(v_res_2053_);
return v_r_2054_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1___redArg(lean_object* v_x_2055_, lean_object* v_x_2056_){
_start:
{
uint64_t v___y_2058_; uint64_t v___y_2062_; uint64_t v___y_2066_; 
if (lean_obj_tag(v_x_2056_) == 0)
{
uint8_t v_inv_2069_; 
v_inv_2069_ = lean_ctor_get_uint8(v_x_2056_, sizeof(void*)*1 + 1);
if (v_inv_2069_ == 0)
{
lean_object* v_declName_2070_; 
v_declName_2070_ = lean_ctor_get(v_x_2056_, 0);
if (lean_obj_tag(v_declName_2070_) == 0)
{
uint64_t v___x_2071_; 
v___x_2071_ = 1723ULL;
v___y_2062_ = v___x_2071_;
goto v___jp_2061_;
}
else
{
uint64_t v_hash_2072_; 
v_hash_2072_ = lean_ctor_get_uint64(v_declName_2070_, sizeof(void*)*2);
v___y_2062_ = v_hash_2072_;
goto v___jp_2061_;
}
}
else
{
lean_object* v_declName_2073_; 
v_declName_2073_ = lean_ctor_get(v_x_2056_, 0);
if (lean_obj_tag(v_declName_2073_) == 0)
{
uint64_t v___x_2074_; 
v___x_2074_ = 1723ULL;
v___y_2066_ = v___x_2074_;
goto v___jp_2065_;
}
else
{
uint64_t v_hash_2075_; 
v_hash_2075_ = lean_ctor_get_uint64(v_declName_2073_, sizeof(void*)*2);
v___y_2066_ = v_hash_2075_;
goto v___jp_2065_;
}
}
}
else
{
lean_object* v___x_2076_; 
v___x_2076_ = l_Lean_Meta_Origin_key(v_x_2056_);
if (lean_obj_tag(v___x_2076_) == 0)
{
uint64_t v___x_2077_; 
v___x_2077_ = 1723ULL;
v___y_2058_ = v___x_2077_;
goto v___jp_2057_;
}
else
{
uint64_t v_hash_2078_; 
v_hash_2078_ = lean_ctor_get_uint64(v___x_2076_, sizeof(void*)*2);
lean_dec(v___x_2076_);
v___y_2058_ = v_hash_2078_;
goto v___jp_2057_;
}
}
v___jp_2057_:
{
size_t v___x_2059_; uint8_t v___x_2060_; 
v___x_2059_ = lean_uint64_to_usize(v___y_2058_);
v___x_2060_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1_spec__2___redArg(v_x_2055_, v___x_2059_, v_x_2056_);
return v___x_2060_;
}
v___jp_2061_:
{
uint64_t v___x_2063_; uint64_t v___x_2064_; 
v___x_2063_ = 13ULL;
v___x_2064_ = lean_uint64_mix_hash(v___y_2062_, v___x_2063_);
v___y_2058_ = v___x_2064_;
goto v___jp_2057_;
}
v___jp_2065_:
{
uint64_t v___x_2067_; uint64_t v___x_2068_; 
v___x_2067_ = 11ULL;
v___x_2068_ = lean_uint64_mix_hash(v___y_2066_, v___x_2067_);
v___y_2058_ = v___x_2068_;
goto v___jp_2057_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1___redArg___boxed(lean_object* v_x_2079_, lean_object* v_x_2080_){
_start:
{
uint8_t v_res_2081_; lean_object* v_r_2082_; 
v_res_2081_ = lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1___redArg(v_x_2079_, v_x_2080_);
lean_dec_ref(v_x_2080_);
lean_dec_ref(v_x_2079_);
v_r_2082_ = lean_box(v_res_2081_);
return v_r_2082_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem(lean_object* v_thms_2083_, lean_object* v_simprocs_2084_, lean_object* v_x_2085_){
_start:
{
if (lean_obj_tag(v_x_2085_) == 0)
{
lean_object* v_declName_2086_; uint8_t v_post_2087_; uint8_t v_inv_2088_; uint8_t v___y_2090_; 
v_declName_2086_ = lean_ctor_get(v_x_2085_, 0);
v_post_2087_ = lean_ctor_get_uint8(v_x_2085_, sizeof(void*)*1);
v_inv_2088_ = lean_ctor_get_uint8(v_x_2085_, sizeof(void*)*1 + 1);
if (v_post_2087_ == 1)
{
if (v_inv_2088_ == 0)
{
lean_object* v_simprocNames_2095_; uint8_t v___x_2096_; 
v_simprocNames_2095_ = lean_ctor_get(v_simprocs_2084_, 2);
v___x_2096_ = lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0___redArg(v_simprocNames_2095_, v_declName_2086_);
if (v___x_2096_ == 0)
{
lean_object* v_lemmaNames_2097_; uint8_t v___x_2098_; 
v_lemmaNames_2097_ = lean_ctor_get(v_thms_2083_, 2);
v___x_2098_ = lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1___redArg(v_lemmaNames_2097_, v_x_2085_);
v___y_2090_ = v___x_2098_;
goto v___jp_2089_;
}
else
{
v___y_2090_ = v___x_2096_;
goto v___jp_2089_;
}
}
else
{
uint8_t v___x_2099_; 
v___x_2099_ = 0;
return v___x_2099_;
}
}
else
{
uint8_t v___x_2100_; 
v___x_2100_ = 0;
return v___x_2100_;
}
v___jp_2089_:
{
if (v___y_2090_ == 0)
{
lean_object* v_toUnfold_2091_; lean_object* v_toUnfoldThms_2092_; uint8_t v___x_2093_; 
v_toUnfold_2091_ = lean_ctor_get(v_thms_2083_, 3);
v_toUnfoldThms_2092_ = lean_ctor_get(v_thms_2083_, 5);
v___x_2093_ = lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0___redArg(v_toUnfold_2091_, v_declName_2086_);
if (v___x_2093_ == 0)
{
uint8_t v___x_2094_; 
v___x_2094_ = lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0___redArg(v_toUnfoldThms_2092_, v_declName_2086_);
return v___x_2094_;
}
else
{
return v___x_2093_;
}
}
else
{
return v___y_2090_;
}
}
}
else
{
uint8_t v___x_2101_; 
v___x_2101_ = 0;
return v___x_2101_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem___boxed(lean_object* v_thms_2102_, lean_object* v_simprocs_2103_, lean_object* v_x_2104_){
_start:
{
uint8_t v_res_2105_; lean_object* v_r_2106_; 
v_res_2105_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem(v_thms_2102_, v_simprocs_2103_, v_x_2104_);
lean_dec_ref(v_x_2104_);
lean_dec_ref(v_simprocs_2103_);
lean_dec_ref(v_thms_2102_);
v_r_2106_ = lean_box(v_res_2105_);
return v_r_2106_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0(lean_object* v_00_u03b2_2107_, lean_object* v_x_2108_, lean_object* v_x_2109_){
_start:
{
uint8_t v___x_2110_; 
v___x_2110_ = lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0___redArg(v_x_2108_, v_x_2109_);
return v___x_2110_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0___boxed(lean_object* v_00_u03b2_2111_, lean_object* v_x_2112_, lean_object* v_x_2113_){
_start:
{
uint8_t v_res_2114_; lean_object* v_r_2115_; 
v_res_2114_ = lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0(v_00_u03b2_2111_, v_x_2112_, v_x_2113_);
lean_dec(v_x_2113_);
lean_dec_ref(v_x_2112_);
v_r_2115_ = lean_box(v_res_2114_);
return v_r_2115_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1(lean_object* v_00_u03b2_2116_, lean_object* v_x_2117_, lean_object* v_x_2118_){
_start:
{
uint8_t v___x_2119_; 
v___x_2119_ = lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1___redArg(v_x_2117_, v_x_2118_);
return v___x_2119_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1___boxed(lean_object* v_00_u03b2_2120_, lean_object* v_x_2121_, lean_object* v_x_2122_){
_start:
{
uint8_t v_res_2123_; lean_object* v_r_2124_; 
v_res_2123_ = lp_aesop_Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1(v_00_u03b2_2120_, v_x_2121_, v_x_2122_);
lean_dec_ref(v_x_2122_);
lean_dec_ref(v_x_2121_);
v_r_2124_ = lean_box(v_res_2123_);
return v_r_2124_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0_spec__0(lean_object* v_00_u03b2_2125_, lean_object* v_x_2126_, size_t v_x_2127_, lean_object* v_x_2128_){
_start:
{
uint8_t v___x_2129_; 
v___x_2129_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0_spec__0___redArg(v_x_2126_, v_x_2127_, v_x_2128_);
return v___x_2129_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0_spec__0___boxed(lean_object* v_00_u03b2_2130_, lean_object* v_x_2131_, lean_object* v_x_2132_, lean_object* v_x_2133_){
_start:
{
size_t v_x_722__boxed_2134_; uint8_t v_res_2135_; lean_object* v_r_2136_; 
v_x_722__boxed_2134_ = lean_unbox_usize(v_x_2132_);
lean_dec(v_x_2132_);
v_res_2135_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0_spec__0(v_00_u03b2_2130_, v_x_2131_, v_x_722__boxed_2134_, v_x_2133_);
lean_dec(v_x_2133_);
lean_dec_ref(v_x_2131_);
v_r_2136_ = lean_box(v_res_2135_);
return v_r_2136_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1_spec__2(lean_object* v_00_u03b2_2137_, lean_object* v_x_2138_, size_t v_x_2139_, lean_object* v_x_2140_){
_start:
{
uint8_t v___x_2141_; 
v___x_2141_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1_spec__2___redArg(v_x_2138_, v_x_2139_, v_x_2140_);
return v___x_2141_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1_spec__2___boxed(lean_object* v_00_u03b2_2142_, lean_object* v_x_2143_, lean_object* v_x_2144_, lean_object* v_x_2145_){
_start:
{
size_t v_x_733__boxed_2146_; uint8_t v_res_2147_; lean_object* v_r_2148_; 
v_x_733__boxed_2146_ = lean_unbox_usize(v_x_2144_);
lean_dec(v_x_2144_);
v_res_2147_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1_spec__2(v_00_u03b2_2142_, v_x_2143_, v_x_733__boxed_2146_, v_x_2145_);
lean_dec_ref(v_x_2145_);
lean_dec_ref(v_x_2143_);
v_r_2148_ = lean_box(v_res_2147_);
return v_r_2148_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0_spec__0_spec__1(lean_object* v_00_u03b2_2149_, lean_object* v_keys_2150_, lean_object* v_vals_2151_, lean_object* v_heq_2152_, lean_object* v_i_2153_, lean_object* v_k_2154_){
_start:
{
uint8_t v___x_2155_; 
v___x_2155_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0_spec__0_spec__1___redArg(v_keys_2150_, v_i_2153_, v_k_2154_);
return v___x_2155_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b2_2156_, lean_object* v_keys_2157_, lean_object* v_vals_2158_, lean_object* v_heq_2159_, lean_object* v_i_2160_, lean_object* v_k_2161_){
_start:
{
uint8_t v_res_2162_; lean_object* v_r_2163_; 
v_res_2162_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__0_spec__0_spec__1(v_00_u03b2_2156_, v_keys_2157_, v_vals_2158_, v_heq_2159_, v_i_2160_, v_k_2161_);
lean_dec(v_k_2161_);
lean_dec_ref(v_vals_2158_);
lean_dec_ref(v_keys_2157_);
v_r_2163_ = lean_box(v_res_2162_);
return v_r_2163_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1_spec__2_spec__4(lean_object* v_00_u03b2_2164_, lean_object* v_keys_2165_, lean_object* v_vals_2166_, lean_object* v_heq_2167_, lean_object* v_i_2168_, lean_object* v_k_2169_){
_start:
{
uint8_t v___x_2170_; 
v___x_2170_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1_spec__2_spec__4___redArg(v_keys_2165_, v_i_2168_, v_k_2169_);
return v___x_2170_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1_spec__2_spec__4___boxed(lean_object* v_00_u03b2_2171_, lean_object* v_keys_2172_, lean_object* v_vals_2173_, lean_object* v_heq_2174_, lean_object* v_i_2175_, lean_object* v_k_2176_){
_start:
{
uint8_t v_res_2177_; lean_object* v_r_2178_; 
v_res_2177_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem_spec__1_spec__2_spec__4(v_00_u03b2_2171_, v_keys_2172_, v_vals_2173_, v_heq_2174_, v_i_2175_, v_k_2176_);
lean_dec_ref(v_k_2176_);
lean_dec_ref(v_vals_2173_);
lean_dec_ref(v_keys_2172_);
v_r_2178_ = lean_box(v_res_2177_);
return v_r_2178_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0_spec__1_spec__5___redArg(lean_object* v_x_2179_, lean_object* v_x_2180_, lean_object* v_x_2181_, lean_object* v_x_2182_){
_start:
{
lean_object* v_ks_2183_; lean_object* v_vs_2184_; lean_object* v___x_2186_; uint8_t v_isShared_2187_; uint8_t v_isSharedCheck_2214_; 
v_ks_2183_ = lean_ctor_get(v_x_2179_, 0);
v_vs_2184_ = lean_ctor_get(v_x_2179_, 1);
v_isSharedCheck_2214_ = !lean_is_exclusive(v_x_2179_);
if (v_isSharedCheck_2214_ == 0)
{
v___x_2186_ = v_x_2179_;
v_isShared_2187_ = v_isSharedCheck_2214_;
goto v_resetjp_2185_;
}
else
{
lean_inc(v_vs_2184_);
lean_inc(v_ks_2183_);
lean_dec(v_x_2179_);
v___x_2186_ = lean_box(0);
v_isShared_2187_ = v_isSharedCheck_2214_;
goto v_resetjp_2185_;
}
v_resetjp_2185_:
{
uint8_t v___y_2196_; lean_object* v___x_2200_; uint8_t v___x_2201_; 
v___x_2200_ = lean_array_get_size(v_ks_2183_);
v___x_2201_ = lean_nat_dec_lt(v_x_2180_, v___x_2200_);
if (v___x_2201_ == 0)
{
lean_object* v___x_2202_; lean_object* v___x_2203_; lean_object* v___x_2204_; 
lean_del_object(v___x_2186_);
lean_dec(v_x_2180_);
v___x_2202_ = lean_array_push(v_ks_2183_, v_x_2181_);
v___x_2203_ = lean_array_push(v_vs_2184_, v_x_2182_);
v___x_2204_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2204_, 0, v___x_2202_);
lean_ctor_set(v___x_2204_, 1, v___x_2203_);
return v___x_2204_;
}
else
{
lean_object* v_k_x27_2205_; 
v_k_x27_2205_ = lean_array_fget_borrowed(v_ks_2183_, v_x_2180_);
if (lean_obj_tag(v_x_2181_) == 0)
{
if (lean_obj_tag(v_k_x27_2205_) == 0)
{
lean_object* v_declName_2206_; uint8_t v_inv_2207_; lean_object* v_declName_2208_; uint8_t v_inv_2209_; uint8_t v___x_2210_; 
v_declName_2206_ = lean_ctor_get(v_x_2181_, 0);
v_inv_2207_ = lean_ctor_get_uint8(v_x_2181_, sizeof(void*)*1 + 1);
v_declName_2208_ = lean_ctor_get(v_k_x27_2205_, 0);
v_inv_2209_ = lean_ctor_get_uint8(v_k_x27_2205_, sizeof(void*)*1 + 1);
v___x_2210_ = lean_name_eq(v_declName_2206_, v_declName_2208_);
if (v___x_2210_ == 0)
{
v___y_2196_ = v___x_2210_;
goto v___jp_2195_;
}
else
{
if (v_inv_2207_ == 0)
{
if (v_inv_2209_ == 0)
{
v___y_2196_ = v___x_2210_;
goto v___jp_2195_;
}
else
{
goto v___jp_2188_;
}
}
else
{
v___y_2196_ = v_inv_2209_;
goto v___jp_2195_;
}
}
}
else
{
goto v___jp_2188_;
}
}
else
{
if (lean_obj_tag(v_k_x27_2205_) == 0)
{
goto v___jp_2188_;
}
else
{
lean_object* v___x_2211_; lean_object* v___x_2212_; uint8_t v___x_2213_; 
v___x_2211_ = l_Lean_Meta_Origin_key(v_x_2181_);
v___x_2212_ = l_Lean_Meta_Origin_key(v_k_x27_2205_);
v___x_2213_ = lean_name_eq(v___x_2211_, v___x_2212_);
lean_dec(v___x_2212_);
lean_dec(v___x_2211_);
v___y_2196_ = v___x_2213_;
goto v___jp_2195_;
}
}
}
v___jp_2188_:
{
lean_object* v___x_2190_; 
if (v_isShared_2187_ == 0)
{
v___x_2190_ = v___x_2186_;
goto v_reusejp_2189_;
}
else
{
lean_object* v_reuseFailAlloc_2194_; 
v_reuseFailAlloc_2194_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2194_, 0, v_ks_2183_);
lean_ctor_set(v_reuseFailAlloc_2194_, 1, v_vs_2184_);
v___x_2190_ = v_reuseFailAlloc_2194_;
goto v_reusejp_2189_;
}
v_reusejp_2189_:
{
lean_object* v___x_2191_; lean_object* v___x_2192_; 
v___x_2191_ = lean_unsigned_to_nat(1u);
v___x_2192_ = lean_nat_add(v_x_2180_, v___x_2191_);
lean_dec(v_x_2180_);
v_x_2179_ = v___x_2190_;
v_x_2180_ = v___x_2192_;
goto _start;
}
}
v___jp_2195_:
{
if (v___y_2196_ == 0)
{
goto v___jp_2188_;
}
else
{
lean_object* v___x_2197_; lean_object* v___x_2198_; lean_object* v___x_2199_; 
lean_del_object(v___x_2186_);
v___x_2197_ = lean_array_fset(v_ks_2183_, v_x_2180_, v_x_2181_);
v___x_2198_ = lean_array_fset(v_vs_2184_, v_x_2180_, v_x_2182_);
lean_dec(v_x_2180_);
v___x_2199_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2199_, 0, v___x_2197_);
lean_ctor_set(v___x_2199_, 1, v___x_2198_);
return v___x_2199_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0_spec__1___redArg(lean_object* v_n_2215_, lean_object* v_k_2216_, lean_object* v_v_2217_){
_start:
{
lean_object* v___x_2218_; lean_object* v___x_2219_; 
v___x_2218_ = lean_unsigned_to_nat(0u);
v___x_2219_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0_spec__1_spec__5___redArg(v_n_2215_, v___x_2218_, v_k_2216_, v_v_2217_);
return v___x_2219_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_2220_; 
v___x_2220_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_2220_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0___redArg(lean_object* v_x_2221_, size_t v_x_2222_, size_t v_x_2223_, lean_object* v_x_2224_, lean_object* v_x_2225_){
_start:
{
if (lean_obj_tag(v_x_2221_) == 0)
{
lean_object* v_es_2226_; size_t v___x_2227_; size_t v___x_2228_; lean_object* v_j_2229_; lean_object* v___x_2230_; uint8_t v___x_2231_; 
v_es_2226_ = lean_ctor_get(v_x_2221_, 0);
v___x_2227_ = ((size_t)31ULL);
v___x_2228_ = lean_usize_land(v_x_2222_, v___x_2227_);
v_j_2229_ = lean_usize_to_nat(v___x_2228_);
v___x_2230_ = lean_array_get_size(v_es_2226_);
v___x_2231_ = lean_nat_dec_lt(v_j_2229_, v___x_2230_);
if (v___x_2231_ == 0)
{
lean_dec(v_j_2229_);
lean_dec(v_x_2225_);
lean_dec_ref(v_x_2224_);
return v_x_2221_;
}
else
{
lean_object* v___x_2233_; uint8_t v_isShared_2234_; uint8_t v_isSharedCheck_2280_; 
lean_inc_ref(v_es_2226_);
v_isSharedCheck_2280_ = !lean_is_exclusive(v_x_2221_);
if (v_isSharedCheck_2280_ == 0)
{
lean_object* v_unused_2281_; 
v_unused_2281_ = lean_ctor_get(v_x_2221_, 0);
lean_dec(v_unused_2281_);
v___x_2233_ = v_x_2221_;
v_isShared_2234_ = v_isSharedCheck_2280_;
goto v_resetjp_2232_;
}
else
{
lean_dec(v_x_2221_);
v___x_2233_ = lean_box(0);
v_isShared_2234_ = v_isSharedCheck_2280_;
goto v_resetjp_2232_;
}
v_resetjp_2232_:
{
lean_object* v_v_2235_; lean_object* v___x_2236_; lean_object* v_xs_x27_2237_; lean_object* v___y_2239_; 
v_v_2235_ = lean_array_fget(v_es_2226_, v_j_2229_);
v___x_2236_ = lean_box(0);
v_xs_x27_2237_ = lean_array_fset(v_es_2226_, v_j_2229_, v___x_2236_);
switch(lean_obj_tag(v_v_2235_))
{
case 0:
{
lean_object* v_key_2244_; lean_object* v_val_2245_; lean_object* v___x_2247_; uint8_t v_isShared_2248_; uint8_t v_isSharedCheck_2265_; 
v_key_2244_ = lean_ctor_get(v_v_2235_, 0);
v_val_2245_ = lean_ctor_get(v_v_2235_, 1);
v_isSharedCheck_2265_ = !lean_is_exclusive(v_v_2235_);
if (v_isSharedCheck_2265_ == 0)
{
v___x_2247_ = v_v_2235_;
v_isShared_2248_ = v_isSharedCheck_2265_;
goto v_resetjp_2246_;
}
else
{
lean_inc(v_val_2245_);
lean_inc(v_key_2244_);
lean_dec(v_v_2235_);
v___x_2247_ = lean_box(0);
v_isShared_2248_ = v_isSharedCheck_2265_;
goto v_resetjp_2246_;
}
v_resetjp_2246_:
{
uint8_t v___y_2253_; 
if (lean_obj_tag(v_x_2224_) == 0)
{
if (lean_obj_tag(v_key_2244_) == 0)
{
lean_object* v_declName_2257_; uint8_t v_inv_2258_; lean_object* v_declName_2259_; uint8_t v_inv_2260_; uint8_t v___x_2261_; 
v_declName_2257_ = lean_ctor_get(v_x_2224_, 0);
v_inv_2258_ = lean_ctor_get_uint8(v_x_2224_, sizeof(void*)*1 + 1);
v_declName_2259_ = lean_ctor_get(v_key_2244_, 0);
v_inv_2260_ = lean_ctor_get_uint8(v_key_2244_, sizeof(void*)*1 + 1);
v___x_2261_ = lean_name_eq(v_declName_2257_, v_declName_2259_);
if (v___x_2261_ == 0)
{
v___y_2253_ = v___x_2261_;
goto v___jp_2252_;
}
else
{
if (v_inv_2258_ == 0)
{
if (v_inv_2260_ == 0)
{
v___y_2253_ = v___x_2261_;
goto v___jp_2252_;
}
else
{
lean_del_object(v___x_2247_);
goto v___jp_2249_;
}
}
else
{
v___y_2253_ = v_inv_2260_;
goto v___jp_2252_;
}
}
}
else
{
lean_del_object(v___x_2247_);
goto v___jp_2249_;
}
}
else
{
if (lean_obj_tag(v_key_2244_) == 0)
{
lean_del_object(v___x_2247_);
goto v___jp_2249_;
}
else
{
lean_object* v___x_2262_; lean_object* v___x_2263_; uint8_t v___x_2264_; 
v___x_2262_ = l_Lean_Meta_Origin_key(v_x_2224_);
v___x_2263_ = l_Lean_Meta_Origin_key(v_key_2244_);
v___x_2264_ = lean_name_eq(v___x_2262_, v___x_2263_);
lean_dec(v___x_2263_);
lean_dec(v___x_2262_);
v___y_2253_ = v___x_2264_;
goto v___jp_2252_;
}
}
v___jp_2249_:
{
lean_object* v___x_2250_; lean_object* v___x_2251_; 
v___x_2250_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_2244_, v_val_2245_, v_x_2224_, v_x_2225_);
v___x_2251_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2251_, 0, v___x_2250_);
v___y_2239_ = v___x_2251_;
goto v___jp_2238_;
}
v___jp_2252_:
{
if (v___y_2253_ == 0)
{
lean_del_object(v___x_2247_);
goto v___jp_2249_;
}
else
{
lean_object* v___x_2255_; 
lean_dec(v_val_2245_);
lean_dec(v_key_2244_);
if (v_isShared_2248_ == 0)
{
lean_ctor_set(v___x_2247_, 1, v_x_2225_);
lean_ctor_set(v___x_2247_, 0, v_x_2224_);
v___x_2255_ = v___x_2247_;
goto v_reusejp_2254_;
}
else
{
lean_object* v_reuseFailAlloc_2256_; 
v_reuseFailAlloc_2256_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2256_, 0, v_x_2224_);
lean_ctor_set(v_reuseFailAlloc_2256_, 1, v_x_2225_);
v___x_2255_ = v_reuseFailAlloc_2256_;
goto v_reusejp_2254_;
}
v_reusejp_2254_:
{
v___y_2239_ = v___x_2255_;
goto v___jp_2238_;
}
}
}
}
}
case 1:
{
lean_object* v_node_2266_; lean_object* v___x_2268_; uint8_t v_isShared_2269_; uint8_t v_isSharedCheck_2278_; 
v_node_2266_ = lean_ctor_get(v_v_2235_, 0);
v_isSharedCheck_2278_ = !lean_is_exclusive(v_v_2235_);
if (v_isSharedCheck_2278_ == 0)
{
v___x_2268_ = v_v_2235_;
v_isShared_2269_ = v_isSharedCheck_2278_;
goto v_resetjp_2267_;
}
else
{
lean_inc(v_node_2266_);
lean_dec(v_v_2235_);
v___x_2268_ = lean_box(0);
v_isShared_2269_ = v_isSharedCheck_2278_;
goto v_resetjp_2267_;
}
v_resetjp_2267_:
{
size_t v___x_2270_; size_t v___x_2271_; size_t v___x_2272_; size_t v___x_2273_; lean_object* v___x_2274_; lean_object* v___x_2276_; 
v___x_2270_ = ((size_t)5ULL);
v___x_2271_ = lean_usize_shift_right(v_x_2222_, v___x_2270_);
v___x_2272_ = ((size_t)1ULL);
v___x_2273_ = lean_usize_add(v_x_2223_, v___x_2272_);
v___x_2274_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0___redArg(v_node_2266_, v___x_2271_, v___x_2273_, v_x_2224_, v_x_2225_);
if (v_isShared_2269_ == 0)
{
lean_ctor_set(v___x_2268_, 0, v___x_2274_);
v___x_2276_ = v___x_2268_;
goto v_reusejp_2275_;
}
else
{
lean_object* v_reuseFailAlloc_2277_; 
v_reuseFailAlloc_2277_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2277_, 0, v___x_2274_);
v___x_2276_ = v_reuseFailAlloc_2277_;
goto v_reusejp_2275_;
}
v_reusejp_2275_:
{
v___y_2239_ = v___x_2276_;
goto v___jp_2238_;
}
}
}
default: 
{
lean_object* v___x_2279_; 
v___x_2279_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2279_, 0, v_x_2224_);
lean_ctor_set(v___x_2279_, 1, v_x_2225_);
v___y_2239_ = v___x_2279_;
goto v___jp_2238_;
}
}
v___jp_2238_:
{
lean_object* v___x_2240_; lean_object* v___x_2242_; 
v___x_2240_ = lean_array_fset(v_xs_x27_2237_, v_j_2229_, v___y_2239_);
lean_dec(v_j_2229_);
if (v_isShared_2234_ == 0)
{
lean_ctor_set(v___x_2233_, 0, v___x_2240_);
v___x_2242_ = v___x_2233_;
goto v_reusejp_2241_;
}
else
{
lean_object* v_reuseFailAlloc_2243_; 
v_reuseFailAlloc_2243_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2243_, 0, v___x_2240_);
v___x_2242_ = v_reuseFailAlloc_2243_;
goto v_reusejp_2241_;
}
v_reusejp_2241_:
{
return v___x_2242_;
}
}
}
}
}
else
{
lean_object* v_ks_2282_; lean_object* v_vs_2283_; lean_object* v___x_2285_; uint8_t v_isShared_2286_; uint8_t v_isSharedCheck_2303_; 
v_ks_2282_ = lean_ctor_get(v_x_2221_, 0);
v_vs_2283_ = lean_ctor_get(v_x_2221_, 1);
v_isSharedCheck_2303_ = !lean_is_exclusive(v_x_2221_);
if (v_isSharedCheck_2303_ == 0)
{
v___x_2285_ = v_x_2221_;
v_isShared_2286_ = v_isSharedCheck_2303_;
goto v_resetjp_2284_;
}
else
{
lean_inc(v_vs_2283_);
lean_inc(v_ks_2282_);
lean_dec(v_x_2221_);
v___x_2285_ = lean_box(0);
v_isShared_2286_ = v_isSharedCheck_2303_;
goto v_resetjp_2284_;
}
v_resetjp_2284_:
{
lean_object* v___x_2288_; 
if (v_isShared_2286_ == 0)
{
v___x_2288_ = v___x_2285_;
goto v_reusejp_2287_;
}
else
{
lean_object* v_reuseFailAlloc_2302_; 
v_reuseFailAlloc_2302_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2302_, 0, v_ks_2282_);
lean_ctor_set(v_reuseFailAlloc_2302_, 1, v_vs_2283_);
v___x_2288_ = v_reuseFailAlloc_2302_;
goto v_reusejp_2287_;
}
v_reusejp_2287_:
{
lean_object* v_newNode_2289_; uint8_t v___y_2291_; size_t v___x_2297_; uint8_t v___x_2298_; 
v_newNode_2289_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0_spec__1___redArg(v___x_2288_, v_x_2224_, v_x_2225_);
v___x_2297_ = ((size_t)7ULL);
v___x_2298_ = lean_usize_dec_le(v___x_2297_, v_x_2223_);
if (v___x_2298_ == 0)
{
lean_object* v___x_2299_; lean_object* v___x_2300_; uint8_t v___x_2301_; 
v___x_2299_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_2289_);
v___x_2300_ = lean_unsigned_to_nat(4u);
v___x_2301_ = lean_nat_dec_lt(v___x_2299_, v___x_2300_);
lean_dec(v___x_2299_);
v___y_2291_ = v___x_2301_;
goto v___jp_2290_;
}
else
{
v___y_2291_ = v___x_2298_;
goto v___jp_2290_;
}
v___jp_2290_:
{
if (v___y_2291_ == 0)
{
lean_object* v_ks_2292_; lean_object* v_vs_2293_; lean_object* v___x_2294_; lean_object* v___x_2295_; lean_object* v___x_2296_; 
v_ks_2292_ = lean_ctor_get(v_newNode_2289_, 0);
lean_inc_ref(v_ks_2292_);
v_vs_2293_ = lean_ctor_get(v_newNode_2289_, 1);
lean_inc_ref(v_vs_2293_);
lean_dec_ref(v_newNode_2289_);
v___x_2294_ = lean_unsigned_to_nat(0u);
v___x_2295_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0___redArg___closed__0, &lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0___redArg___closed__0_once, _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0___redArg___closed__0);
v___x_2296_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0_spec__2___redArg(v_x_2223_, v_ks_2292_, v_vs_2293_, v___x_2294_, v___x_2295_);
lean_dec_ref(v_vs_2293_);
lean_dec_ref(v_ks_2292_);
return v___x_2296_;
}
else
{
return v_newNode_2289_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0_spec__2___redArg(size_t v_depth_2304_, lean_object* v_keys_2305_, lean_object* v_vals_2306_, lean_object* v_i_2307_, lean_object* v_entries_2308_){
_start:
{
lean_object* v___x_2309_; uint8_t v___x_2310_; 
v___x_2309_ = lean_array_get_size(v_keys_2305_);
v___x_2310_ = lean_nat_dec_lt(v_i_2307_, v___x_2309_);
if (v___x_2310_ == 0)
{
lean_dec(v_i_2307_);
return v_entries_2308_;
}
else
{
lean_object* v_k_2311_; lean_object* v_v_2312_; uint64_t v___y_2314_; uint64_t v___y_2326_; uint64_t v___y_2330_; 
v_k_2311_ = lean_array_fget_borrowed(v_keys_2305_, v_i_2307_);
v_v_2312_ = lean_array_fget_borrowed(v_vals_2306_, v_i_2307_);
if (lean_obj_tag(v_k_2311_) == 0)
{
uint8_t v_inv_2333_; 
v_inv_2333_ = lean_ctor_get_uint8(v_k_2311_, sizeof(void*)*1 + 1);
if (v_inv_2333_ == 0)
{
lean_object* v_declName_2334_; 
v_declName_2334_ = lean_ctor_get(v_k_2311_, 0);
if (lean_obj_tag(v_declName_2334_) == 0)
{
uint64_t v___x_2335_; 
v___x_2335_ = 1723ULL;
v___y_2326_ = v___x_2335_;
goto v___jp_2325_;
}
else
{
uint64_t v_hash_2336_; 
v_hash_2336_ = lean_ctor_get_uint64(v_declName_2334_, sizeof(void*)*2);
v___y_2326_ = v_hash_2336_;
goto v___jp_2325_;
}
}
else
{
lean_object* v_declName_2337_; 
v_declName_2337_ = lean_ctor_get(v_k_2311_, 0);
if (lean_obj_tag(v_declName_2337_) == 0)
{
uint64_t v___x_2338_; 
v___x_2338_ = 1723ULL;
v___y_2330_ = v___x_2338_;
goto v___jp_2329_;
}
else
{
uint64_t v_hash_2339_; 
v_hash_2339_ = lean_ctor_get_uint64(v_declName_2337_, sizeof(void*)*2);
v___y_2330_ = v_hash_2339_;
goto v___jp_2329_;
}
}
}
else
{
lean_object* v___x_2340_; 
v___x_2340_ = l_Lean_Meta_Origin_key(v_k_2311_);
if (lean_obj_tag(v___x_2340_) == 0)
{
uint64_t v___x_2341_; 
v___x_2341_ = 1723ULL;
v___y_2314_ = v___x_2341_;
goto v___jp_2313_;
}
else
{
uint64_t v_hash_2342_; 
v_hash_2342_ = lean_ctor_get_uint64(v___x_2340_, sizeof(void*)*2);
lean_dec(v___x_2340_);
v___y_2314_ = v_hash_2342_;
goto v___jp_2313_;
}
}
v___jp_2313_:
{
size_t v_h_2315_; size_t v___x_2316_; lean_object* v___x_2317_; size_t v___x_2318_; size_t v___x_2319_; size_t v___x_2320_; size_t v_h_2321_; lean_object* v___x_2322_; lean_object* v___x_2323_; 
v_h_2315_ = lean_uint64_to_usize(v___y_2314_);
v___x_2316_ = ((size_t)5ULL);
v___x_2317_ = lean_unsigned_to_nat(1u);
v___x_2318_ = ((size_t)1ULL);
v___x_2319_ = lean_usize_sub(v_depth_2304_, v___x_2318_);
v___x_2320_ = lean_usize_mul(v___x_2316_, v___x_2319_);
v_h_2321_ = lean_usize_shift_right(v_h_2315_, v___x_2320_);
v___x_2322_ = lean_nat_add(v_i_2307_, v___x_2317_);
lean_dec(v_i_2307_);
lean_inc(v_v_2312_);
lean_inc(v_k_2311_);
v___x_2323_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0___redArg(v_entries_2308_, v_h_2321_, v_depth_2304_, v_k_2311_, v_v_2312_);
v_i_2307_ = v___x_2322_;
v_entries_2308_ = v___x_2323_;
goto _start;
}
v___jp_2325_:
{
uint64_t v___x_2327_; uint64_t v___x_2328_; 
v___x_2327_ = 13ULL;
v___x_2328_ = lean_uint64_mix_hash(v___y_2326_, v___x_2327_);
v___y_2314_ = v___x_2328_;
goto v___jp_2313_;
}
v___jp_2329_:
{
uint64_t v___x_2331_; uint64_t v___x_2332_; 
v___x_2331_ = 11ULL;
v___x_2332_ = lean_uint64_mix_hash(v___y_2330_, v___x_2331_);
v___y_2314_ = v___x_2332_;
goto v___jp_2313_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0_spec__2___redArg___boxed(lean_object* v_depth_2343_, lean_object* v_keys_2344_, lean_object* v_vals_2345_, lean_object* v_i_2346_, lean_object* v_entries_2347_){
_start:
{
size_t v_depth_boxed_2348_; lean_object* v_res_2349_; 
v_depth_boxed_2348_ = lean_unbox_usize(v_depth_2343_);
lean_dec(v_depth_2343_);
v_res_2349_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0_spec__2___redArg(v_depth_boxed_2348_, v_keys_2344_, v_vals_2345_, v_i_2346_, v_entries_2347_);
lean_dec_ref(v_vals_2345_);
lean_dec_ref(v_keys_2344_);
return v_res_2349_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0___redArg___boxed(lean_object* v_x_2350_, lean_object* v_x_2351_, lean_object* v_x_2352_, lean_object* v_x_2353_, lean_object* v_x_2354_){
_start:
{
size_t v_x_22712__boxed_2355_; size_t v_x_22713__boxed_2356_; lean_object* v_res_2357_; 
v_x_22712__boxed_2355_ = lean_unbox_usize(v_x_2351_);
lean_dec(v_x_2351_);
v_x_22713__boxed_2356_ = lean_unbox_usize(v_x_2352_);
lean_dec(v_x_2352_);
v_res_2357_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0___redArg(v_x_2350_, v_x_22712__boxed_2355_, v_x_22713__boxed_2356_, v_x_2353_, v_x_2354_);
return v_res_2357_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0___redArg(lean_object* v_x_2358_, lean_object* v_x_2359_, lean_object* v_x_2360_){
_start:
{
uint64_t v___y_2362_; uint64_t v___y_2367_; uint64_t v___y_2371_; 
if (lean_obj_tag(v_x_2359_) == 0)
{
uint8_t v_inv_2374_; 
v_inv_2374_ = lean_ctor_get_uint8(v_x_2359_, sizeof(void*)*1 + 1);
if (v_inv_2374_ == 0)
{
lean_object* v_declName_2375_; 
v_declName_2375_ = lean_ctor_get(v_x_2359_, 0);
if (lean_obj_tag(v_declName_2375_) == 0)
{
uint64_t v___x_2376_; 
v___x_2376_ = 1723ULL;
v___y_2367_ = v___x_2376_;
goto v___jp_2366_;
}
else
{
uint64_t v_hash_2377_; 
v_hash_2377_ = lean_ctor_get_uint64(v_declName_2375_, sizeof(void*)*2);
v___y_2367_ = v_hash_2377_;
goto v___jp_2366_;
}
}
else
{
lean_object* v_declName_2378_; 
v_declName_2378_ = lean_ctor_get(v_x_2359_, 0);
if (lean_obj_tag(v_declName_2378_) == 0)
{
uint64_t v___x_2379_; 
v___x_2379_ = 1723ULL;
v___y_2371_ = v___x_2379_;
goto v___jp_2370_;
}
else
{
uint64_t v_hash_2380_; 
v_hash_2380_ = lean_ctor_get_uint64(v_declName_2378_, sizeof(void*)*2);
v___y_2371_ = v_hash_2380_;
goto v___jp_2370_;
}
}
}
else
{
lean_object* v___x_2381_; 
v___x_2381_ = l_Lean_Meta_Origin_key(v_x_2359_);
if (lean_obj_tag(v___x_2381_) == 0)
{
uint64_t v___x_2382_; 
v___x_2382_ = 1723ULL;
v___y_2362_ = v___x_2382_;
goto v___jp_2361_;
}
else
{
uint64_t v_hash_2383_; 
v_hash_2383_ = lean_ctor_get_uint64(v___x_2381_, sizeof(void*)*2);
lean_dec(v___x_2381_);
v___y_2362_ = v_hash_2383_;
goto v___jp_2361_;
}
}
v___jp_2361_:
{
size_t v___x_2363_; size_t v___x_2364_; lean_object* v___x_2365_; 
v___x_2363_ = lean_uint64_to_usize(v___y_2362_);
v___x_2364_ = ((size_t)1ULL);
v___x_2365_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0___redArg(v_x_2358_, v___x_2363_, v___x_2364_, v_x_2359_, v_x_2360_);
return v___x_2365_;
}
v___jp_2366_:
{
uint64_t v___x_2368_; uint64_t v___x_2369_; 
v___x_2368_ = 13ULL;
v___x_2369_ = lean_uint64_mix_hash(v___y_2367_, v___x_2368_);
v___y_2362_ = v___x_2369_;
goto v___jp_2361_;
}
v___jp_2370_:
{
uint64_t v___x_2372_; uint64_t v___x_2373_; 
v___x_2372_ = 11ULL;
v___x_2373_ = lean_uint64_mix_hash(v___y_2371_, v___x_2372_);
v___y_2362_ = v___x_2373_;
goto v___jp_2361_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___lam__0(lean_object* v_a_2384_, lean_object* v_a_2385_, lean_object* v_x_2386_, lean_object* v_origin_2387_, lean_object* v_thm_2388_, lean_object* v___y_2389_, lean_object* v___y_2390_, lean_object* v___y_2391_, lean_object* v___y_2392_){
_start:
{
lean_object* v_fst_2394_; lean_object* v_snd_2395_; uint8_t v___x_2396_; 
v_fst_2394_ = lean_ctor_get(v_x_2386_, 0);
v_snd_2395_ = lean_ctor_get(v_x_2386_, 1);
v___x_2396_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_isGlobalSimpTheorem(v_a_2384_, v_a_2385_, v_origin_2387_);
if (v___x_2396_ == 0)
{
lean_object* v___x_2398_; uint8_t v_isShared_2399_; uint8_t v_isSharedCheck_2407_; 
lean_inc(v_snd_2395_);
lean_inc(v_fst_2394_);
v_isSharedCheck_2407_ = !lean_is_exclusive(v_x_2386_);
if (v_isSharedCheck_2407_ == 0)
{
lean_object* v_unused_2408_; lean_object* v_unused_2409_; 
v_unused_2408_ = lean_ctor_get(v_x_2386_, 1);
lean_dec(v_unused_2408_);
v_unused_2409_ = lean_ctor_get(v_x_2386_, 0);
lean_dec(v_unused_2409_);
v___x_2398_ = v_x_2386_;
v_isShared_2399_ = v_isSharedCheck_2407_;
goto v_resetjp_2397_;
}
else
{
lean_dec(v_x_2386_);
v___x_2398_ = lean_box(0);
v_isShared_2399_ = v_isSharedCheck_2407_;
goto v_resetjp_2397_;
}
v_resetjp_2397_:
{
lean_object* v___x_2400_; lean_object* v___x_2401_; lean_object* v___x_2402_; lean_object* v___x_2404_; 
v___x_2400_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0___redArg(v_fst_2394_, v_origin_2387_, v_thm_2388_);
v___x_2401_ = lean_unsigned_to_nat(1u);
v___x_2402_ = lean_nat_add(v_snd_2395_, v___x_2401_);
lean_dec(v_snd_2395_);
if (v_isShared_2399_ == 0)
{
lean_ctor_set(v___x_2398_, 1, v___x_2402_);
lean_ctor_set(v___x_2398_, 0, v___x_2400_);
v___x_2404_ = v___x_2398_;
goto v_reusejp_2403_;
}
else
{
lean_object* v_reuseFailAlloc_2406_; 
v_reuseFailAlloc_2406_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2406_, 0, v___x_2400_);
lean_ctor_set(v_reuseFailAlloc_2406_, 1, v___x_2402_);
v___x_2404_ = v_reuseFailAlloc_2406_;
goto v_reusejp_2403_;
}
v_reusejp_2403_:
{
lean_object* v___x_2405_; 
v___x_2405_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2405_, 0, v___x_2404_);
return v___x_2405_;
}
}
}
else
{
lean_object* v___x_2410_; 
lean_dec(v_thm_2388_);
lean_dec_ref(v_origin_2387_);
v___x_2410_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2410_, 0, v_x_2386_);
return v___x_2410_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___lam__0___boxed(lean_object* v_a_2411_, lean_object* v_a_2412_, lean_object* v_x_2413_, lean_object* v_origin_2414_, lean_object* v_thm_2415_, lean_object* v___y_2416_, lean_object* v___y_2417_, lean_object* v___y_2418_, lean_object* v___y_2419_, lean_object* v___y_2420_){
_start:
{
lean_object* v_res_2421_; 
v_res_2421_ = lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___lam__0(v_a_2411_, v_a_2412_, v_x_2413_, v_origin_2414_, v_thm_2415_, v___y_2416_, v___y_2417_, v___y_2418_, v___y_2419_);
lean_dec(v___y_2419_);
lean_dec_ref(v___y_2418_);
lean_dec(v___y_2417_);
lean_dec_ref(v___y_2416_);
lean_dec_ref(v_a_2412_);
lean_dec_ref(v_a_2411_);
return v_res_2421_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2_spec__4(lean_object* v_msgData_2422_, lean_object* v___y_2423_, lean_object* v___y_2424_, lean_object* v___y_2425_, lean_object* v___y_2426_){
_start:
{
lean_object* v___x_2428_; lean_object* v_env_2429_; lean_object* v___x_2430_; lean_object* v_mctx_2431_; lean_object* v_lctx_2432_; lean_object* v_options_2433_; lean_object* v___x_2434_; lean_object* v___x_2435_; lean_object* v___x_2436_; 
v___x_2428_ = lean_st_ref_get(v___y_2426_);
v_env_2429_ = lean_ctor_get(v___x_2428_, 0);
lean_inc_ref(v_env_2429_);
lean_dec(v___x_2428_);
v___x_2430_ = lean_st_ref_get(v___y_2424_);
v_mctx_2431_ = lean_ctor_get(v___x_2430_, 0);
lean_inc_ref(v_mctx_2431_);
lean_dec(v___x_2430_);
v_lctx_2432_ = lean_ctor_get(v___y_2423_, 2);
v_options_2433_ = lean_ctor_get(v___y_2425_, 2);
lean_inc_ref(v_options_2433_);
lean_inc_ref(v_lctx_2432_);
v___x_2434_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2434_, 0, v_env_2429_);
lean_ctor_set(v___x_2434_, 1, v_mctx_2431_);
lean_ctor_set(v___x_2434_, 2, v_lctx_2432_);
lean_ctor_set(v___x_2434_, 3, v_options_2433_);
v___x_2435_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_2435_, 0, v___x_2434_);
lean_ctor_set(v___x_2435_, 1, v_msgData_2422_);
v___x_2436_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2436_, 0, v___x_2435_);
return v___x_2436_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2_spec__4___boxed(lean_object* v_msgData_2437_, lean_object* v___y_2438_, lean_object* v___y_2439_, lean_object* v___y_2440_, lean_object* v___y_2441_, lean_object* v___y_2442_){
_start:
{
lean_object* v_res_2443_; 
v_res_2443_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2_spec__4(v_msgData_2437_, v___y_2438_, v___y_2439_, v___y_2440_, v___y_2441_);
lean_dec(v___y_2441_);
lean_dec_ref(v___y_2440_);
lean_dec(v___y_2439_);
lean_dec_ref(v___y_2438_);
return v_res_2443_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2___redArg(lean_object* v_msg_2444_, lean_object* v___y_2445_, lean_object* v___y_2446_, lean_object* v___y_2447_, lean_object* v___y_2448_){
_start:
{
lean_object* v_ref_2450_; lean_object* v___x_2451_; lean_object* v_a_2452_; lean_object* v___x_2454_; uint8_t v_isShared_2455_; uint8_t v_isSharedCheck_2460_; 
v_ref_2450_ = lean_ctor_get(v___y_2447_, 5);
v___x_2451_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2_spec__4(v_msg_2444_, v___y_2445_, v___y_2446_, v___y_2447_, v___y_2448_);
v_a_2452_ = lean_ctor_get(v___x_2451_, 0);
v_isSharedCheck_2460_ = !lean_is_exclusive(v___x_2451_);
if (v_isSharedCheck_2460_ == 0)
{
v___x_2454_ = v___x_2451_;
v_isShared_2455_ = v_isSharedCheck_2460_;
goto v_resetjp_2453_;
}
else
{
lean_inc(v_a_2452_);
lean_dec(v___x_2451_);
v___x_2454_ = lean_box(0);
v_isShared_2455_ = v_isSharedCheck_2460_;
goto v_resetjp_2453_;
}
v_resetjp_2453_:
{
lean_object* v___x_2456_; lean_object* v___x_2458_; 
lean_inc(v_ref_2450_);
v___x_2456_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2456_, 0, v_ref_2450_);
lean_ctor_set(v___x_2456_, 1, v_a_2452_);
if (v_isShared_2455_ == 0)
{
lean_ctor_set_tag(v___x_2454_, 1);
lean_ctor_set(v___x_2454_, 0, v___x_2456_);
v___x_2458_ = v___x_2454_;
goto v_reusejp_2457_;
}
else
{
lean_object* v_reuseFailAlloc_2459_; 
v_reuseFailAlloc_2459_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2459_, 0, v___x_2456_);
v___x_2458_ = v_reuseFailAlloc_2459_;
goto v_reusejp_2457_;
}
v_reusejp_2457_:
{
return v___x_2458_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2___redArg___boxed(lean_object* v_msg_2461_, lean_object* v___y_2462_, lean_object* v___y_2463_, lean_object* v___y_2464_, lean_object* v___y_2465_, lean_object* v___y_2466_){
_start:
{
lean_object* v_res_2467_; 
v_res_2467_ = lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2___redArg(v_msg_2461_, v___y_2462_, v___y_2463_, v___y_2464_, v___y_2465_);
lean_dec(v___y_2465_);
lean_dec_ref(v___y_2464_);
lean_dec(v___y_2463_);
lean_dec_ref(v___y_2462_);
return v_res_2467_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2_spec__6___redArg(lean_object* v_f_2468_, lean_object* v_keys_2469_, lean_object* v_vals_2470_, lean_object* v_i_2471_, lean_object* v_acc_2472_, lean_object* v___y_2473_, lean_object* v___y_2474_, lean_object* v___y_2475_, lean_object* v___y_2476_){
_start:
{
lean_object* v___x_2478_; uint8_t v___x_2479_; 
v___x_2478_ = lean_array_get_size(v_keys_2469_);
v___x_2479_ = lean_nat_dec_lt(v_i_2471_, v___x_2478_);
if (v___x_2479_ == 0)
{
lean_object* v___x_2480_; 
lean_dec(v_i_2471_);
lean_dec_ref(v_f_2468_);
v___x_2480_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2480_, 0, v_acc_2472_);
return v___x_2480_;
}
else
{
lean_object* v_k_2481_; lean_object* v_v_2482_; lean_object* v___x_2483_; 
v_k_2481_ = lean_array_fget_borrowed(v_keys_2469_, v_i_2471_);
v_v_2482_ = lean_array_fget_borrowed(v_vals_2470_, v_i_2471_);
lean_inc_ref(v_f_2468_);
lean_inc(v___y_2476_);
lean_inc_ref(v___y_2475_);
lean_inc(v___y_2474_);
lean_inc_ref(v___y_2473_);
lean_inc(v_v_2482_);
lean_inc(v_k_2481_);
v___x_2483_ = lean_apply_8(v_f_2468_, v_acc_2472_, v_k_2481_, v_v_2482_, v___y_2473_, v___y_2474_, v___y_2475_, v___y_2476_, lean_box(0));
if (lean_obj_tag(v___x_2483_) == 0)
{
lean_object* v_a_2484_; lean_object* v___x_2485_; lean_object* v___x_2486_; 
v_a_2484_ = lean_ctor_get(v___x_2483_, 0);
lean_inc(v_a_2484_);
lean_dec_ref_known(v___x_2483_, 1);
v___x_2485_ = lean_unsigned_to_nat(1u);
v___x_2486_ = lean_nat_add(v_i_2471_, v___x_2485_);
lean_dec(v_i_2471_);
v_i_2471_ = v___x_2486_;
v_acc_2472_ = v_a_2484_;
goto _start;
}
else
{
lean_dec(v_i_2471_);
lean_dec_ref(v_f_2468_);
return v___x_2483_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2_spec__6___redArg___boxed(lean_object* v_f_2488_, lean_object* v_keys_2489_, lean_object* v_vals_2490_, lean_object* v_i_2491_, lean_object* v_acc_2492_, lean_object* v___y_2493_, lean_object* v___y_2494_, lean_object* v___y_2495_, lean_object* v___y_2496_, lean_object* v___y_2497_){
_start:
{
lean_object* v_res_2498_; 
v_res_2498_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2_spec__6___redArg(v_f_2488_, v_keys_2489_, v_vals_2490_, v_i_2491_, v_acc_2492_, v___y_2493_, v___y_2494_, v___y_2495_, v___y_2496_);
lean_dec(v___y_2496_);
lean_dec_ref(v___y_2495_);
lean_dec(v___y_2494_);
lean_dec_ref(v___y_2493_);
lean_dec_ref(v_vals_2490_);
lean_dec_ref(v_keys_2489_);
return v_res_2498_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2___redArg(lean_object* v_f_2499_, lean_object* v_x_2500_, lean_object* v_x_2501_, lean_object* v___y_2502_, lean_object* v___y_2503_, lean_object* v___y_2504_, lean_object* v___y_2505_){
_start:
{
if (lean_obj_tag(v_x_2500_) == 0)
{
lean_object* v_es_2507_; lean_object* v___x_2509_; uint8_t v_isShared_2510_; uint8_t v_isSharedCheck_2527_; 
v_es_2507_ = lean_ctor_get(v_x_2500_, 0);
v_isSharedCheck_2527_ = !lean_is_exclusive(v_x_2500_);
if (v_isSharedCheck_2527_ == 0)
{
v___x_2509_ = v_x_2500_;
v_isShared_2510_ = v_isSharedCheck_2527_;
goto v_resetjp_2508_;
}
else
{
lean_inc(v_es_2507_);
lean_dec(v_x_2500_);
v___x_2509_ = lean_box(0);
v_isShared_2510_ = v_isSharedCheck_2527_;
goto v_resetjp_2508_;
}
v_resetjp_2508_:
{
lean_object* v___x_2511_; lean_object* v___x_2512_; uint8_t v___x_2513_; 
v___x_2511_ = lean_unsigned_to_nat(0u);
v___x_2512_ = lean_array_get_size(v_es_2507_);
v___x_2513_ = lean_nat_dec_lt(v___x_2511_, v___x_2512_);
if (v___x_2513_ == 0)
{
lean_object* v___x_2515_; 
lean_dec_ref(v_es_2507_);
lean_dec_ref(v_f_2499_);
if (v_isShared_2510_ == 0)
{
lean_ctor_set(v___x_2509_, 0, v_x_2501_);
v___x_2515_ = v___x_2509_;
goto v_reusejp_2514_;
}
else
{
lean_object* v_reuseFailAlloc_2516_; 
v_reuseFailAlloc_2516_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2516_, 0, v_x_2501_);
v___x_2515_ = v_reuseFailAlloc_2516_;
goto v_reusejp_2514_;
}
v_reusejp_2514_:
{
return v___x_2515_;
}
}
else
{
uint8_t v___x_2517_; 
v___x_2517_ = lean_nat_dec_le(v___x_2512_, v___x_2512_);
if (v___x_2517_ == 0)
{
if (v___x_2513_ == 0)
{
lean_object* v___x_2519_; 
lean_dec_ref(v_es_2507_);
lean_dec_ref(v_f_2499_);
if (v_isShared_2510_ == 0)
{
lean_ctor_set(v___x_2509_, 0, v_x_2501_);
v___x_2519_ = v___x_2509_;
goto v_reusejp_2518_;
}
else
{
lean_object* v_reuseFailAlloc_2520_; 
v_reuseFailAlloc_2520_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2520_, 0, v_x_2501_);
v___x_2519_ = v_reuseFailAlloc_2520_;
goto v_reusejp_2518_;
}
v_reusejp_2518_:
{
return v___x_2519_;
}
}
else
{
size_t v___x_2521_; size_t v___x_2522_; lean_object* v___x_2523_; 
lean_del_object(v___x_2509_);
v___x_2521_ = ((size_t)0ULL);
v___x_2522_ = lean_usize_of_nat(v___x_2512_);
v___x_2523_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2_spec__5___redArg(v_f_2499_, v_es_2507_, v___x_2521_, v___x_2522_, v_x_2501_, v___y_2502_, v___y_2503_, v___y_2504_, v___y_2505_);
lean_dec_ref(v_es_2507_);
return v___x_2523_;
}
}
else
{
size_t v___x_2524_; size_t v___x_2525_; lean_object* v___x_2526_; 
lean_del_object(v___x_2509_);
v___x_2524_ = ((size_t)0ULL);
v___x_2525_ = lean_usize_of_nat(v___x_2512_);
v___x_2526_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2_spec__5___redArg(v_f_2499_, v_es_2507_, v___x_2524_, v___x_2525_, v_x_2501_, v___y_2502_, v___y_2503_, v___y_2504_, v___y_2505_);
lean_dec_ref(v_es_2507_);
return v___x_2526_;
}
}
}
}
else
{
lean_object* v_ks_2528_; lean_object* v_vs_2529_; lean_object* v___x_2530_; lean_object* v___x_2531_; 
v_ks_2528_ = lean_ctor_get(v_x_2500_, 0);
lean_inc_ref(v_ks_2528_);
v_vs_2529_ = lean_ctor_get(v_x_2500_, 1);
lean_inc_ref(v_vs_2529_);
lean_dec_ref_known(v_x_2500_, 2);
v___x_2530_ = lean_unsigned_to_nat(0u);
v___x_2531_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2_spec__6___redArg(v_f_2499_, v_ks_2528_, v_vs_2529_, v___x_2530_, v_x_2501_, v___y_2502_, v___y_2503_, v___y_2504_, v___y_2505_);
lean_dec_ref(v_vs_2529_);
lean_dec_ref(v_ks_2528_);
return v___x_2531_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2_spec__5___redArg(lean_object* v_f_2532_, lean_object* v_as_2533_, size_t v_i_2534_, size_t v_stop_2535_, lean_object* v_b_2536_, lean_object* v___y_2537_, lean_object* v___y_2538_, lean_object* v___y_2539_, lean_object* v___y_2540_){
_start:
{
lean_object* v_a_2543_; lean_object* v___y_2548_; uint8_t v___x_2550_; 
v___x_2550_ = lean_usize_dec_eq(v_i_2534_, v_stop_2535_);
if (v___x_2550_ == 0)
{
lean_object* v___x_2551_; 
v___x_2551_ = lean_array_uget_borrowed(v_as_2533_, v_i_2534_);
switch(lean_obj_tag(v___x_2551_))
{
case 0:
{
lean_object* v_key_2552_; lean_object* v_val_2553_; lean_object* v___x_2554_; 
v_key_2552_ = lean_ctor_get(v___x_2551_, 0);
v_val_2553_ = lean_ctor_get(v___x_2551_, 1);
lean_inc_ref(v_f_2532_);
lean_inc(v___y_2540_);
lean_inc_ref(v___y_2539_);
lean_inc(v___y_2538_);
lean_inc_ref(v___y_2537_);
lean_inc(v_val_2553_);
lean_inc(v_key_2552_);
v___x_2554_ = lean_apply_8(v_f_2532_, v_b_2536_, v_key_2552_, v_val_2553_, v___y_2537_, v___y_2538_, v___y_2539_, v___y_2540_, lean_box(0));
v___y_2548_ = v___x_2554_;
goto v___jp_2547_;
}
case 1:
{
lean_object* v_node_2555_; lean_object* v___x_2556_; 
v_node_2555_ = lean_ctor_get(v___x_2551_, 0);
lean_inc(v_node_2555_);
lean_inc_ref(v_f_2532_);
v___x_2556_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2___redArg(v_f_2532_, v_node_2555_, v_b_2536_, v___y_2537_, v___y_2538_, v___y_2539_, v___y_2540_);
v___y_2548_ = v___x_2556_;
goto v___jp_2547_;
}
default: 
{
v_a_2543_ = v_b_2536_;
goto v___jp_2542_;
}
}
}
else
{
lean_object* v___x_2557_; 
lean_dec_ref(v_f_2532_);
v___x_2557_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2557_, 0, v_b_2536_);
return v___x_2557_;
}
v___jp_2542_:
{
size_t v___x_2544_; size_t v___x_2545_; 
v___x_2544_ = ((size_t)1ULL);
v___x_2545_ = lean_usize_add(v_i_2534_, v___x_2544_);
v_i_2534_ = v___x_2545_;
v_b_2536_ = v_a_2543_;
goto _start;
}
v___jp_2547_:
{
if (lean_obj_tag(v___y_2548_) == 0)
{
lean_object* v_a_2549_; 
v_a_2549_ = lean_ctor_get(v___y_2548_, 0);
lean_inc(v_a_2549_);
lean_dec_ref_known(v___y_2548_, 1);
v_a_2543_ = v_a_2549_;
goto v___jp_2542_;
}
else
{
lean_dec_ref(v_f_2532_);
return v___y_2548_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2_spec__5___redArg___boxed(lean_object* v_f_2558_, lean_object* v_as_2559_, lean_object* v_i_2560_, lean_object* v_stop_2561_, lean_object* v_b_2562_, lean_object* v___y_2563_, lean_object* v___y_2564_, lean_object* v___y_2565_, lean_object* v___y_2566_, lean_object* v___y_2567_){
_start:
{
size_t v_i_boxed_2568_; size_t v_stop_boxed_2569_; lean_object* v_res_2570_; 
v_i_boxed_2568_ = lean_unbox_usize(v_i_2560_);
lean_dec(v_i_2560_);
v_stop_boxed_2569_ = lean_unbox_usize(v_stop_2561_);
lean_dec(v_stop_2561_);
v_res_2570_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2_spec__5___redArg(v_f_2558_, v_as_2559_, v_i_boxed_2568_, v_stop_boxed_2569_, v_b_2562_, v___y_2563_, v___y_2564_, v___y_2565_, v___y_2566_);
lean_dec(v___y_2566_);
lean_dec_ref(v___y_2565_);
lean_dec(v___y_2564_);
lean_dec_ref(v___y_2563_);
lean_dec_ref(v_as_2559_);
return v_res_2570_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2___redArg___boxed(lean_object* v_f_2571_, lean_object* v_x_2572_, lean_object* v_x_2573_, lean_object* v___y_2574_, lean_object* v___y_2575_, lean_object* v___y_2576_, lean_object* v___y_2577_, lean_object* v___y_2578_){
_start:
{
lean_object* v_res_2579_; 
v_res_2579_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2___redArg(v_f_2571_, v_x_2572_, v_x_2573_, v___y_2574_, v___y_2575_, v___y_2576_, v___y_2577_);
lean_dec(v___y_2577_);
lean_dec_ref(v___y_2576_);
lean_dec(v___y_2575_);
lean_dec_ref(v___y_2574_);
return v_res_2579_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__0(void){
_start:
{
lean_object* v___x_2580_; 
v___x_2580_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_2580_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__1(void){
_start:
{
lean_object* v___x_2581_; lean_object* v___x_2582_; 
v___x_2581_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__0, &lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__0_once, _init_lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__0);
v___x_2582_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2582_, 0, v___x_2581_);
return v___x_2582_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__2(void){
_start:
{
lean_object* v___x_2583_; lean_object* v___x_2584_; lean_object* v___x_2585_; 
v___x_2583_ = lean_unsigned_to_nat(0u);
v___x_2584_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__1, &lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__1_once, _init_lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__1);
v___x_2585_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2585_, 0, v___x_2584_);
lean_ctor_set(v___x_2585_, 1, v___x_2583_);
return v___x_2585_;
}
}
static lean_object* _init_lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4(void){
_start:
{
lean_object* v___x_2587_; lean_object* v___x_2588_; 
v___x_2587_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__3));
v___x_2588_ = l_Lean_stringToMessageData(v___x_2587_);
return v___x_2588_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar(uint8_t v_simpAll_2591_, lean_object* v_inGoal_2592_, lean_object* v_configStx_x3f_2593_, lean_object* v_usedTheorems_2594_, lean_object* v_a_2595_, lean_object* v_a_2596_, lean_object* v_a_2597_, lean_object* v_a_2598_){
_start:
{
lean_object* v_stx_2601_; lean_object* v___x_2604_; 
v___x_2604_ = l_Lean_Meta_getSimpTheorems___redArg(v_a_2598_);
if (lean_obj_tag(v___x_2604_) == 0)
{
lean_object* v_a_2605_; lean_object* v___x_2606_; 
v_a_2605_ = lean_ctor_get(v___x_2604_, 0);
lean_inc(v_a_2605_);
lean_dec_ref_known(v___x_2604_, 1);
v___x_2606_ = l_Lean_Meta_Simp_getSimprocs___redArg(v_a_2598_);
if (lean_obj_tag(v___x_2606_) == 0)
{
lean_object* v_a_2607_; lean_object* v_map_2608_; lean_object* v___x_2610_; uint8_t v_isShared_2611_; uint8_t v_isSharedCheck_2991_; 
v_a_2607_ = lean_ctor_get(v___x_2606_, 0);
lean_inc(v_a_2607_);
lean_dec_ref_known(v___x_2606_, 1);
v_map_2608_ = lean_ctor_get(v_usedTheorems_2594_, 0);
v_isSharedCheck_2991_ = !lean_is_exclusive(v_usedTheorems_2594_);
if (v_isSharedCheck_2991_ == 0)
{
lean_object* v_unused_2992_; 
v_unused_2992_ = lean_ctor_get(v_usedTheorems_2594_, 1);
lean_dec(v_unused_2992_);
v___x_2610_ = v_usedTheorems_2594_;
v_isShared_2611_ = v_isSharedCheck_2991_;
goto v_resetjp_2609_;
}
else
{
lean_inc(v_map_2608_);
lean_dec(v_usedTheorems_2594_);
v___x_2610_ = lean_box(0);
v_isShared_2611_ = v_isSharedCheck_2991_;
goto v_resetjp_2609_;
}
v_resetjp_2609_:
{
lean_object* v___f_2612_; lean_object* v___x_2613_; lean_object* v___x_2614_; lean_object* v___x_2615_; 
v___f_2612_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___lam__0___boxed), 10, 2);
lean_closure_set(v___f_2612_, 0, v_a_2605_);
lean_closure_set(v___f_2612_, 1, v_a_2607_);
v___x_2613_ = lean_unsigned_to_nat(0u);
v___x_2614_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__2, &lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__2_once, _init_lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__2);
v___x_2615_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2___redArg(v___f_2612_, v_map_2608_, v___x_2614_, v_a_2595_, v_a_2596_, v_a_2597_, v_a_2598_);
if (lean_obj_tag(v___x_2615_) == 0)
{
lean_object* v_a_2616_; lean_object* v_fst_2617_; lean_object* v_snd_2618_; lean_object* v___x_2620_; uint8_t v_isShared_2621_; uint8_t v_isSharedCheck_2982_; 
v_a_2616_ = lean_ctor_get(v___x_2615_, 0);
lean_inc(v_a_2616_);
lean_dec_ref_known(v___x_2615_, 1);
v_fst_2617_ = lean_ctor_get(v_a_2616_, 0);
v_snd_2618_ = lean_ctor_get(v_a_2616_, 1);
v_isSharedCheck_2982_ = !lean_is_exclusive(v_a_2616_);
if (v_isSharedCheck_2982_ == 0)
{
v___x_2620_ = v_a_2616_;
v_isShared_2621_ = v_isSharedCheck_2982_;
goto v_resetjp_2619_;
}
else
{
lean_inc(v_snd_2618_);
lean_inc(v_fst_2617_);
lean_dec(v_a_2616_);
v___x_2620_ = lean_box(0);
v_isShared_2621_ = v_isSharedCheck_2982_;
goto v_resetjp_2619_;
}
v_resetjp_2619_:
{
lean_object* v___x_2623_; 
if (v_isShared_2611_ == 0)
{
lean_ctor_set(v___x_2610_, 1, v_snd_2618_);
lean_ctor_set(v___x_2610_, 0, v_fst_2617_);
v___x_2623_ = v___x_2610_;
goto v_reusejp_2622_;
}
else
{
lean_object* v_reuseFailAlloc_2981_; 
v_reuseFailAlloc_2981_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2981_, 0, v_fst_2617_);
lean_ctor_set(v_reuseFailAlloc_2981_, 1, v_snd_2618_);
v___x_2623_ = v_reuseFailAlloc_2981_;
goto v_reusejp_2622_;
}
v_reusejp_2622_:
{
lean_object* v___x_2624_; 
v___x_2624_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarOnlyStx(v_simpAll_2591_, v_inGoal_2592_, v_configStx_x3f_2593_, v___x_2623_, v_a_2595_, v_a_2596_, v_a_2597_, v_a_2598_);
if (lean_obj_tag(v___x_2624_) == 0)
{
lean_object* v_a_2625_; lean_object* v___x_2626_; uint8_t v___x_2627_; 
v_a_2625_ = lean_ctor_get(v___x_2624_, 0);
lean_inc_n(v_a_2625_, 2);
lean_dec_ref_known(v___x_2624_, 1);
v___x_2626_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__1));
v___x_2627_ = l_Lean_Syntax_isOfKind(v_a_2625_, v___x_2626_);
if (v___x_2627_ == 0)
{
lean_object* v___x_2628_; lean_object* v___x_2629_; uint8_t v___x_2630_; 
v___x_2628_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__0));
v___x_2629_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__1));
lean_inc(v_a_2625_);
v___x_2630_ = l_Lean_Syntax_isOfKind(v_a_2625_, v___x_2629_);
if (v___x_2630_ == 0)
{
lean_object* v___x_2631_; lean_object* v___x_2632_; lean_object* v___x_2633_; lean_object* v___x_2635_; 
v___x_2631_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4, &lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4_once, _init_lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4);
v___x_2632_ = l_Lean_MessageData_ofSyntax(v_a_2625_);
v___x_2633_ = l_Lean_indentD(v___x_2632_);
if (v_isShared_2621_ == 0)
{
lean_ctor_set_tag(v___x_2620_, 7);
lean_ctor_set(v___x_2620_, 1, v___x_2633_);
lean_ctor_set(v___x_2620_, 0, v___x_2631_);
v___x_2635_ = v___x_2620_;
goto v_reusejp_2634_;
}
else
{
lean_object* v_reuseFailAlloc_2645_; 
v_reuseFailAlloc_2645_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2645_, 0, v___x_2631_);
lean_ctor_set(v_reuseFailAlloc_2645_, 1, v___x_2633_);
v___x_2635_ = v_reuseFailAlloc_2645_;
goto v_reusejp_2634_;
}
v_reusejp_2634_:
{
lean_object* v___x_2636_; lean_object* v_a_2637_; lean_object* v___x_2639_; uint8_t v_isShared_2640_; uint8_t v_isSharedCheck_2644_; 
v___x_2636_ = lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2___redArg(v___x_2635_, v_a_2595_, v_a_2596_, v_a_2597_, v_a_2598_);
v_a_2637_ = lean_ctor_get(v___x_2636_, 0);
v_isSharedCheck_2644_ = !lean_is_exclusive(v___x_2636_);
if (v_isSharedCheck_2644_ == 0)
{
v___x_2639_ = v___x_2636_;
v_isShared_2640_ = v_isSharedCheck_2644_;
goto v_resetjp_2638_;
}
else
{
lean_inc(v_a_2637_);
lean_dec(v___x_2636_);
v___x_2639_ = lean_box(0);
v_isShared_2640_ = v_isSharedCheck_2644_;
goto v_resetjp_2638_;
}
v_resetjp_2638_:
{
lean_object* v___x_2642_; 
if (v_isShared_2640_ == 0)
{
v___x_2642_ = v___x_2639_;
goto v_reusejp_2641_;
}
else
{
lean_object* v_reuseFailAlloc_2643_; 
v_reuseFailAlloc_2643_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2643_, 0, v_a_2637_);
v___x_2642_ = v_reuseFailAlloc_2643_;
goto v_reusejp_2641_;
}
v_reusejp_2641_:
{
return v___x_2642_;
}
}
}
}
else
{
lean_object* v___x_2646_; lean_object* v___x_2647_; lean_object* v___x_2648_; uint8_t v___x_2649_; 
v___x_2646_ = lean_unsigned_to_nat(1u);
v___x_2647_ = l_Lean_Syntax_getArg(v_a_2625_, v___x_2646_);
v___x_2648_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__3));
lean_inc(v___x_2647_);
v___x_2649_ = l_Lean_Syntax_isOfKind(v___x_2647_, v___x_2648_);
if (v___x_2649_ == 0)
{
lean_object* v___x_2650_; lean_object* v___x_2651_; lean_object* v___x_2652_; lean_object* v___x_2654_; 
lean_dec(v___x_2647_);
v___x_2650_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4, &lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4_once, _init_lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4);
v___x_2651_ = l_Lean_MessageData_ofSyntax(v_a_2625_);
v___x_2652_ = l_Lean_indentD(v___x_2651_);
if (v_isShared_2621_ == 0)
{
lean_ctor_set_tag(v___x_2620_, 7);
lean_ctor_set(v___x_2620_, 1, v___x_2652_);
lean_ctor_set(v___x_2620_, 0, v___x_2650_);
v___x_2654_ = v___x_2620_;
goto v_reusejp_2653_;
}
else
{
lean_object* v_reuseFailAlloc_2664_; 
v_reuseFailAlloc_2664_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2664_, 0, v___x_2650_);
lean_ctor_set(v_reuseFailAlloc_2664_, 1, v___x_2652_);
v___x_2654_ = v_reuseFailAlloc_2664_;
goto v_reusejp_2653_;
}
v_reusejp_2653_:
{
lean_object* v___x_2655_; lean_object* v_a_2656_; lean_object* v___x_2658_; uint8_t v_isShared_2659_; uint8_t v_isSharedCheck_2663_; 
v___x_2655_ = lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2___redArg(v___x_2654_, v_a_2595_, v_a_2596_, v_a_2597_, v_a_2598_);
v_a_2656_ = lean_ctor_get(v___x_2655_, 0);
v_isSharedCheck_2663_ = !lean_is_exclusive(v___x_2655_);
if (v_isSharedCheck_2663_ == 0)
{
v___x_2658_ = v___x_2655_;
v_isShared_2659_ = v_isSharedCheck_2663_;
goto v_resetjp_2657_;
}
else
{
lean_inc(v_a_2656_);
lean_dec(v___x_2655_);
v___x_2658_ = lean_box(0);
v_isShared_2659_ = v_isSharedCheck_2663_;
goto v_resetjp_2657_;
}
v_resetjp_2657_:
{
lean_object* v___x_2661_; 
if (v_isShared_2659_ == 0)
{
v___x_2661_ = v___x_2658_;
goto v_reusejp_2660_;
}
else
{
lean_object* v_reuseFailAlloc_2662_; 
v_reuseFailAlloc_2662_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2662_, 0, v_a_2656_);
v___x_2661_ = v_reuseFailAlloc_2662_;
goto v_reusejp_2660_;
}
v_reusejp_2660_:
{
return v___x_2661_;
}
}
}
}
else
{
lean_object* v___x_2665_; lean_object* v___x_2666_; uint8_t v___x_2667_; 
v___x_2665_ = lean_unsigned_to_nat(2u);
v___x_2666_ = l_Lean_Syntax_getArg(v_a_2625_, v___x_2665_);
v___x_2667_ = l_Lean_Syntax_matchesNull(v___x_2666_, v___x_2613_);
if (v___x_2667_ == 0)
{
lean_object* v___x_2668_; lean_object* v___x_2669_; lean_object* v___x_2670_; lean_object* v___x_2672_; 
lean_dec(v___x_2647_);
v___x_2668_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4, &lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4_once, _init_lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4);
v___x_2669_ = l_Lean_MessageData_ofSyntax(v_a_2625_);
v___x_2670_ = l_Lean_indentD(v___x_2669_);
if (v_isShared_2621_ == 0)
{
lean_ctor_set_tag(v___x_2620_, 7);
lean_ctor_set(v___x_2620_, 1, v___x_2670_);
lean_ctor_set(v___x_2620_, 0, v___x_2668_);
v___x_2672_ = v___x_2620_;
goto v_reusejp_2671_;
}
else
{
lean_object* v_reuseFailAlloc_2682_; 
v_reuseFailAlloc_2682_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2682_, 0, v___x_2668_);
lean_ctor_set(v_reuseFailAlloc_2682_, 1, v___x_2670_);
v___x_2672_ = v_reuseFailAlloc_2682_;
goto v_reusejp_2671_;
}
v_reusejp_2671_:
{
lean_object* v___x_2673_; lean_object* v_a_2674_; lean_object* v___x_2676_; uint8_t v_isShared_2677_; uint8_t v_isSharedCheck_2681_; 
v___x_2673_ = lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2___redArg(v___x_2672_, v_a_2595_, v_a_2596_, v_a_2597_, v_a_2598_);
v_a_2674_ = lean_ctor_get(v___x_2673_, 0);
v_isSharedCheck_2681_ = !lean_is_exclusive(v___x_2673_);
if (v_isSharedCheck_2681_ == 0)
{
v___x_2676_ = v___x_2673_;
v_isShared_2677_ = v_isSharedCheck_2681_;
goto v_resetjp_2675_;
}
else
{
lean_inc(v_a_2674_);
lean_dec(v___x_2673_);
v___x_2676_ = lean_box(0);
v_isShared_2677_ = v_isSharedCheck_2681_;
goto v_resetjp_2675_;
}
v_resetjp_2675_:
{
lean_object* v___x_2679_; 
if (v_isShared_2677_ == 0)
{
v___x_2679_ = v___x_2676_;
goto v_reusejp_2678_;
}
else
{
lean_object* v_reuseFailAlloc_2680_; 
v_reuseFailAlloc_2680_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2680_, 0, v_a_2674_);
v___x_2679_ = v_reuseFailAlloc_2680_;
goto v_reusejp_2678_;
}
v_reusejp_2678_:
{
return v___x_2679_;
}
}
}
}
else
{
lean_object* v___x_2683_; lean_object* v___x_2684_; uint8_t v___x_2685_; 
v___x_2683_ = lean_unsigned_to_nat(3u);
v___x_2684_ = l_Lean_Syntax_getArg(v_a_2625_, v___x_2683_);
v___x_2685_ = l_Lean_Syntax_matchesNull(v___x_2684_, v___x_2646_);
if (v___x_2685_ == 0)
{
lean_object* v___x_2686_; lean_object* v___x_2687_; lean_object* v___x_2688_; lean_object* v___x_2690_; 
lean_dec(v___x_2647_);
v___x_2686_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4, &lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4_once, _init_lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4);
v___x_2687_ = l_Lean_MessageData_ofSyntax(v_a_2625_);
v___x_2688_ = l_Lean_indentD(v___x_2687_);
if (v_isShared_2621_ == 0)
{
lean_ctor_set_tag(v___x_2620_, 7);
lean_ctor_set(v___x_2620_, 1, v___x_2688_);
lean_ctor_set(v___x_2620_, 0, v___x_2686_);
v___x_2690_ = v___x_2620_;
goto v_reusejp_2689_;
}
else
{
lean_object* v_reuseFailAlloc_2700_; 
v_reuseFailAlloc_2700_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2700_, 0, v___x_2686_);
lean_ctor_set(v_reuseFailAlloc_2700_, 1, v___x_2688_);
v___x_2690_ = v_reuseFailAlloc_2700_;
goto v_reusejp_2689_;
}
v_reusejp_2689_:
{
lean_object* v___x_2691_; lean_object* v_a_2692_; lean_object* v___x_2694_; uint8_t v_isShared_2695_; uint8_t v_isSharedCheck_2699_; 
v___x_2691_ = lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2___redArg(v___x_2690_, v_a_2595_, v_a_2596_, v_a_2597_, v_a_2598_);
v_a_2692_ = lean_ctor_get(v___x_2691_, 0);
v_isSharedCheck_2699_ = !lean_is_exclusive(v___x_2691_);
if (v_isSharedCheck_2699_ == 0)
{
v___x_2694_ = v___x_2691_;
v_isShared_2695_ = v_isSharedCheck_2699_;
goto v_resetjp_2693_;
}
else
{
lean_inc(v_a_2692_);
lean_dec(v___x_2691_);
v___x_2694_ = lean_box(0);
v_isShared_2695_ = v_isSharedCheck_2699_;
goto v_resetjp_2693_;
}
v_resetjp_2693_:
{
lean_object* v___x_2697_; 
if (v_isShared_2695_ == 0)
{
v___x_2697_ = v___x_2694_;
goto v_reusejp_2696_;
}
else
{
lean_object* v_reuseFailAlloc_2698_; 
v_reuseFailAlloc_2698_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2698_, 0, v_a_2692_);
v___x_2697_ = v_reuseFailAlloc_2698_;
goto v_reusejp_2696_;
}
v_reusejp_2696_:
{
return v___x_2697_;
}
}
}
}
else
{
lean_object* v___x_2701_; lean_object* v___x_2702_; uint8_t v___x_2703_; 
v___x_2701_ = lean_unsigned_to_nat(4u);
v___x_2702_ = l_Lean_Syntax_getArg(v_a_2625_, v___x_2701_);
lean_inc(v___x_2702_);
v___x_2703_ = l_Lean_Syntax_matchesNull(v___x_2702_, v___x_2683_);
if (v___x_2703_ == 0)
{
uint8_t v___x_2704_; 
v___x_2704_ = l_Lean_Syntax_matchesNull(v___x_2702_, v___x_2613_);
if (v___x_2704_ == 0)
{
lean_object* v___x_2705_; lean_object* v___x_2706_; lean_object* v___x_2707_; lean_object* v___x_2709_; 
lean_dec(v___x_2647_);
v___x_2705_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4, &lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4_once, _init_lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4);
v___x_2706_ = l_Lean_MessageData_ofSyntax(v_a_2625_);
v___x_2707_ = l_Lean_indentD(v___x_2706_);
if (v_isShared_2621_ == 0)
{
lean_ctor_set_tag(v___x_2620_, 7);
lean_ctor_set(v___x_2620_, 1, v___x_2707_);
lean_ctor_set(v___x_2620_, 0, v___x_2705_);
v___x_2709_ = v___x_2620_;
goto v_reusejp_2708_;
}
else
{
lean_object* v_reuseFailAlloc_2719_; 
v_reuseFailAlloc_2719_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2719_, 0, v___x_2705_);
lean_ctor_set(v_reuseFailAlloc_2719_, 1, v___x_2707_);
v___x_2709_ = v_reuseFailAlloc_2719_;
goto v_reusejp_2708_;
}
v_reusejp_2708_:
{
lean_object* v___x_2710_; lean_object* v_a_2711_; lean_object* v___x_2713_; uint8_t v_isShared_2714_; uint8_t v_isSharedCheck_2718_; 
v___x_2710_ = lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2___redArg(v___x_2709_, v_a_2595_, v_a_2596_, v_a_2597_, v_a_2598_);
v_a_2711_ = lean_ctor_get(v___x_2710_, 0);
v_isSharedCheck_2718_ = !lean_is_exclusive(v___x_2710_);
if (v_isSharedCheck_2718_ == 0)
{
v___x_2713_ = v___x_2710_;
v_isShared_2714_ = v_isSharedCheck_2718_;
goto v_resetjp_2712_;
}
else
{
lean_inc(v_a_2711_);
lean_dec(v___x_2710_);
v___x_2713_ = lean_box(0);
v_isShared_2714_ = v_isSharedCheck_2718_;
goto v_resetjp_2712_;
}
v_resetjp_2712_:
{
lean_object* v___x_2716_; 
if (v_isShared_2714_ == 0)
{
v___x_2716_ = v___x_2713_;
goto v_reusejp_2715_;
}
else
{
lean_object* v_reuseFailAlloc_2717_; 
v_reuseFailAlloc_2717_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2717_, 0, v_a_2711_);
v___x_2716_ = v_reuseFailAlloc_2717_;
goto v_reusejp_2715_;
}
v_reusejp_2715_:
{
return v___x_2716_;
}
}
}
}
else
{
lean_object* v___x_2720_; lean_object* v___x_2721_; uint8_t v___x_2722_; 
v___x_2720_ = lean_unsigned_to_nat(5u);
v___x_2721_ = l_Lean_Syntax_getArg(v_a_2625_, v___x_2720_);
lean_inc(v___x_2721_);
v___x_2722_ = l_Lean_Syntax_matchesNull(v___x_2721_, v___x_2646_);
if (v___x_2722_ == 0)
{
lean_object* v___x_2723_; lean_object* v___x_2724_; lean_object* v___x_2725_; lean_object* v___x_2727_; 
lean_dec(v___x_2721_);
lean_dec(v___x_2647_);
v___x_2723_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4, &lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4_once, _init_lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4);
v___x_2724_ = l_Lean_MessageData_ofSyntax(v_a_2625_);
v___x_2725_ = l_Lean_indentD(v___x_2724_);
if (v_isShared_2621_ == 0)
{
lean_ctor_set_tag(v___x_2620_, 7);
lean_ctor_set(v___x_2620_, 1, v___x_2725_);
lean_ctor_set(v___x_2620_, 0, v___x_2723_);
v___x_2727_ = v___x_2620_;
goto v_reusejp_2726_;
}
else
{
lean_object* v_reuseFailAlloc_2737_; 
v_reuseFailAlloc_2737_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2737_, 0, v___x_2723_);
lean_ctor_set(v_reuseFailAlloc_2737_, 1, v___x_2725_);
v___x_2727_ = v_reuseFailAlloc_2737_;
goto v_reusejp_2726_;
}
v_reusejp_2726_:
{
lean_object* v___x_2728_; lean_object* v_a_2729_; lean_object* v___x_2731_; uint8_t v_isShared_2732_; uint8_t v_isSharedCheck_2736_; 
v___x_2728_ = lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2___redArg(v___x_2727_, v_a_2595_, v_a_2596_, v_a_2597_, v_a_2598_);
v_a_2729_ = lean_ctor_get(v___x_2728_, 0);
v_isSharedCheck_2736_ = !lean_is_exclusive(v___x_2728_);
if (v_isSharedCheck_2736_ == 0)
{
v___x_2731_ = v___x_2728_;
v_isShared_2732_ = v_isSharedCheck_2736_;
goto v_resetjp_2730_;
}
else
{
lean_inc(v_a_2729_);
lean_dec(v___x_2728_);
v___x_2731_ = lean_box(0);
v_isShared_2732_ = v_isSharedCheck_2736_;
goto v_resetjp_2730_;
}
v_resetjp_2730_:
{
lean_object* v___x_2734_; 
if (v_isShared_2732_ == 0)
{
v___x_2734_ = v___x_2731_;
goto v_reusejp_2733_;
}
else
{
lean_object* v_reuseFailAlloc_2735_; 
v_reuseFailAlloc_2735_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2735_, 0, v_a_2729_);
v___x_2734_ = v_reuseFailAlloc_2735_;
goto v_reusejp_2733_;
}
v_reusejp_2733_:
{
return v___x_2734_;
}
}
}
}
else
{
lean_object* v___x_2738_; lean_object* v___x_2739_; uint8_t v___x_2740_; 
v___x_2738_ = l_Lean_Syntax_getArg(v___x_2721_, v___x_2613_);
lean_dec(v___x_2721_);
v___x_2739_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__1));
lean_inc(v___x_2738_);
v___x_2740_ = l_Lean_Syntax_isOfKind(v___x_2738_, v___x_2739_);
if (v___x_2740_ == 0)
{
lean_object* v___x_2741_; lean_object* v___x_2742_; lean_object* v___x_2743_; lean_object* v___x_2745_; 
lean_dec(v___x_2738_);
lean_dec(v___x_2647_);
v___x_2741_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4, &lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4_once, _init_lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4);
v___x_2742_ = l_Lean_MessageData_ofSyntax(v_a_2625_);
v___x_2743_ = l_Lean_indentD(v___x_2742_);
if (v_isShared_2621_ == 0)
{
lean_ctor_set_tag(v___x_2620_, 7);
lean_ctor_set(v___x_2620_, 1, v___x_2743_);
lean_ctor_set(v___x_2620_, 0, v___x_2741_);
v___x_2745_ = v___x_2620_;
goto v_reusejp_2744_;
}
else
{
lean_object* v_reuseFailAlloc_2755_; 
v_reuseFailAlloc_2755_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2755_, 0, v___x_2741_);
lean_ctor_set(v_reuseFailAlloc_2755_, 1, v___x_2743_);
v___x_2745_ = v_reuseFailAlloc_2755_;
goto v_reusejp_2744_;
}
v_reusejp_2744_:
{
lean_object* v___x_2746_; lean_object* v_a_2747_; lean_object* v___x_2749_; uint8_t v_isShared_2750_; uint8_t v_isSharedCheck_2754_; 
v___x_2746_ = lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2___redArg(v___x_2745_, v_a_2595_, v_a_2596_, v_a_2597_, v_a_2598_);
v_a_2747_ = lean_ctor_get(v___x_2746_, 0);
v_isSharedCheck_2754_ = !lean_is_exclusive(v___x_2746_);
if (v_isSharedCheck_2754_ == 0)
{
v___x_2749_ = v___x_2746_;
v_isShared_2750_ = v_isSharedCheck_2754_;
goto v_resetjp_2748_;
}
else
{
lean_inc(v_a_2747_);
lean_dec(v___x_2746_);
v___x_2749_ = lean_box(0);
v_isShared_2750_ = v_isSharedCheck_2754_;
goto v_resetjp_2748_;
}
v_resetjp_2748_:
{
lean_object* v___x_2752_; 
if (v_isShared_2750_ == 0)
{
v___x_2752_ = v___x_2749_;
goto v_reusejp_2751_;
}
else
{
lean_object* v_reuseFailAlloc_2753_; 
v_reuseFailAlloc_2753_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2753_, 0, v_a_2747_);
v___x_2752_ = v_reuseFailAlloc_2753_;
goto v_reusejp_2751_;
}
v_reusejp_2751_:
{
return v___x_2752_;
}
}
}
}
else
{
lean_object* v___x_2756_; lean_object* v___x_2757_; uint8_t v___x_2758_; 
v___x_2756_ = l_Lean_Syntax_getArg(v___x_2738_, v___x_2646_);
lean_dec(v___x_2738_);
v___x_2757_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__5));
v___x_2758_ = l_Lean_Syntax_isOfKind(v___x_2756_, v___x_2757_);
if (v___x_2758_ == 0)
{
lean_object* v___x_2759_; lean_object* v___x_2760_; lean_object* v___x_2761_; lean_object* v___x_2763_; 
lean_dec(v___x_2647_);
v___x_2759_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4, &lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4_once, _init_lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4);
v___x_2760_ = l_Lean_MessageData_ofSyntax(v_a_2625_);
v___x_2761_ = l_Lean_indentD(v___x_2760_);
if (v_isShared_2621_ == 0)
{
lean_ctor_set_tag(v___x_2620_, 7);
lean_ctor_set(v___x_2620_, 1, v___x_2761_);
lean_ctor_set(v___x_2620_, 0, v___x_2759_);
v___x_2763_ = v___x_2620_;
goto v_reusejp_2762_;
}
else
{
lean_object* v_reuseFailAlloc_2773_; 
v_reuseFailAlloc_2773_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2773_, 0, v___x_2759_);
lean_ctor_set(v_reuseFailAlloc_2773_, 1, v___x_2761_);
v___x_2763_ = v_reuseFailAlloc_2773_;
goto v_reusejp_2762_;
}
v_reusejp_2762_:
{
lean_object* v___x_2764_; lean_object* v_a_2765_; lean_object* v___x_2767_; uint8_t v_isShared_2768_; uint8_t v_isSharedCheck_2772_; 
v___x_2764_ = lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2___redArg(v___x_2763_, v_a_2595_, v_a_2596_, v_a_2597_, v_a_2598_);
v_a_2765_ = lean_ctor_get(v___x_2764_, 0);
v_isSharedCheck_2772_ = !lean_is_exclusive(v___x_2764_);
if (v_isSharedCheck_2772_ == 0)
{
v___x_2767_ = v___x_2764_;
v_isShared_2768_ = v_isSharedCheck_2772_;
goto v_resetjp_2766_;
}
else
{
lean_inc(v_a_2765_);
lean_dec(v___x_2764_);
v___x_2767_ = lean_box(0);
v_isShared_2768_ = v_isSharedCheck_2772_;
goto v_resetjp_2766_;
}
v_resetjp_2766_:
{
lean_object* v___x_2770_; 
if (v_isShared_2768_ == 0)
{
v___x_2770_ = v___x_2767_;
goto v_reusejp_2769_;
}
else
{
lean_object* v_reuseFailAlloc_2771_; 
v_reuseFailAlloc_2771_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2771_, 0, v_a_2765_);
v___x_2770_ = v_reuseFailAlloc_2771_;
goto v_reusejp_2769_;
}
v_reusejp_2769_:
{
return v___x_2770_;
}
}
}
}
else
{
lean_object* v_ref_2774_; lean_object* v___x_2775_; lean_object* v___x_2777_; 
lean_dec(v_a_2625_);
v_ref_2774_ = lean_ctor_get(v_a_2597_, 5);
v___x_2775_ = l_Lean_SourceInfo_fromRef(v_ref_2774_, v___x_2703_);
lean_inc(v___x_2775_);
if (v_isShared_2621_ == 0)
{
lean_ctor_set_tag(v___x_2620_, 2);
lean_ctor_set(v___x_2620_, 1, v___x_2628_);
lean_ctor_set(v___x_2620_, 0, v___x_2775_);
v___x_2777_ = v___x_2620_;
goto v_reusejp_2776_;
}
else
{
lean_object* v_reuseFailAlloc_2789_; 
v_reuseFailAlloc_2789_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2789_, 0, v___x_2775_);
lean_ctor_set(v_reuseFailAlloc_2789_, 1, v___x_2628_);
v___x_2777_ = v_reuseFailAlloc_2789_;
goto v_reusejp_2776_;
}
v_reusejp_2776_:
{
lean_object* v___x_2778_; lean_object* v___x_2779_; lean_object* v___x_2780_; lean_object* v___x_2781_; lean_object* v___x_2782_; lean_object* v___x_2783_; lean_object* v___x_2784_; lean_object* v___x_2785_; lean_object* v___x_2786_; lean_object* v___x_2787_; lean_object* v___x_2788_; 
v___x_2778_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_2779_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
lean_inc_n(v___x_2775_, 6);
v___x_2780_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2780_, 0, v___x_2775_);
lean_ctor_set(v___x_2780_, 1, v___x_2778_);
lean_ctor_set(v___x_2780_, 2, v___x_2779_);
v___x_2781_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__2));
v___x_2782_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2782_, 0, v___x_2775_);
lean_ctor_set(v___x_2782_, 1, v___x_2781_);
v___x_2783_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__6));
v___x_2784_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2784_, 0, v___x_2775_);
lean_ctor_set(v___x_2784_, 1, v___x_2783_);
v___x_2785_ = l_Lean_Syntax_node1(v___x_2775_, v___x_2757_, v___x_2784_);
v___x_2786_ = l_Lean_Syntax_node2(v___x_2775_, v___x_2739_, v___x_2782_, v___x_2785_);
v___x_2787_ = l_Lean_Syntax_node1(v___x_2775_, v___x_2778_, v___x_2786_);
lean_inc_ref_n(v___x_2780_, 2);
v___x_2788_ = l_Lean_Syntax_node6(v___x_2775_, v___x_2629_, v___x_2777_, v___x_2647_, v___x_2780_, v___x_2780_, v___x_2780_, v___x_2787_);
v_stx_2601_ = v___x_2788_;
goto v___jp_2600_;
}
}
}
}
}
}
else
{
lean_object* v___x_2790_; lean_object* v___x_2791_; uint8_t v___x_2792_; 
v___x_2790_ = lean_unsigned_to_nat(5u);
v___x_2791_ = l_Lean_Syntax_getArg(v_a_2625_, v___x_2790_);
lean_inc(v___x_2791_);
v___x_2792_ = l_Lean_Syntax_matchesNull(v___x_2791_, v___x_2646_);
if (v___x_2792_ == 0)
{
lean_object* v___x_2793_; lean_object* v___x_2794_; lean_object* v___x_2795_; lean_object* v___x_2797_; 
lean_dec(v___x_2791_);
lean_dec(v___x_2702_);
lean_dec(v___x_2647_);
v___x_2793_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4, &lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4_once, _init_lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4);
v___x_2794_ = l_Lean_MessageData_ofSyntax(v_a_2625_);
v___x_2795_ = l_Lean_indentD(v___x_2794_);
if (v_isShared_2621_ == 0)
{
lean_ctor_set_tag(v___x_2620_, 7);
lean_ctor_set(v___x_2620_, 1, v___x_2795_);
lean_ctor_set(v___x_2620_, 0, v___x_2793_);
v___x_2797_ = v___x_2620_;
goto v_reusejp_2796_;
}
else
{
lean_object* v_reuseFailAlloc_2807_; 
v_reuseFailAlloc_2807_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2807_, 0, v___x_2793_);
lean_ctor_set(v_reuseFailAlloc_2807_, 1, v___x_2795_);
v___x_2797_ = v_reuseFailAlloc_2807_;
goto v_reusejp_2796_;
}
v_reusejp_2796_:
{
lean_object* v___x_2798_; lean_object* v_a_2799_; lean_object* v___x_2801_; uint8_t v_isShared_2802_; uint8_t v_isSharedCheck_2806_; 
v___x_2798_ = lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2___redArg(v___x_2797_, v_a_2595_, v_a_2596_, v_a_2597_, v_a_2598_);
v_a_2799_ = lean_ctor_get(v___x_2798_, 0);
v_isSharedCheck_2806_ = !lean_is_exclusive(v___x_2798_);
if (v_isSharedCheck_2806_ == 0)
{
v___x_2801_ = v___x_2798_;
v_isShared_2802_ = v_isSharedCheck_2806_;
goto v_resetjp_2800_;
}
else
{
lean_inc(v_a_2799_);
lean_dec(v___x_2798_);
v___x_2801_ = lean_box(0);
v_isShared_2802_ = v_isSharedCheck_2806_;
goto v_resetjp_2800_;
}
v_resetjp_2800_:
{
lean_object* v___x_2804_; 
if (v_isShared_2802_ == 0)
{
v___x_2804_ = v___x_2801_;
goto v_reusejp_2803_;
}
else
{
lean_object* v_reuseFailAlloc_2805_; 
v_reuseFailAlloc_2805_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2805_, 0, v_a_2799_);
v___x_2804_ = v_reuseFailAlloc_2805_;
goto v_reusejp_2803_;
}
v_reusejp_2803_:
{
return v___x_2804_;
}
}
}
}
else
{
lean_object* v___x_2808_; lean_object* v___x_2809_; uint8_t v___x_2810_; 
v___x_2808_ = l_Lean_Syntax_getArg(v___x_2791_, v___x_2613_);
lean_dec(v___x_2791_);
v___x_2809_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__1));
lean_inc(v___x_2808_);
v___x_2810_ = l_Lean_Syntax_isOfKind(v___x_2808_, v___x_2809_);
if (v___x_2810_ == 0)
{
lean_object* v___x_2811_; lean_object* v___x_2812_; lean_object* v___x_2813_; lean_object* v___x_2815_; 
lean_dec(v___x_2808_);
lean_dec(v___x_2702_);
lean_dec(v___x_2647_);
v___x_2811_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4, &lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4_once, _init_lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4);
v___x_2812_ = l_Lean_MessageData_ofSyntax(v_a_2625_);
v___x_2813_ = l_Lean_indentD(v___x_2812_);
if (v_isShared_2621_ == 0)
{
lean_ctor_set_tag(v___x_2620_, 7);
lean_ctor_set(v___x_2620_, 1, v___x_2813_);
lean_ctor_set(v___x_2620_, 0, v___x_2811_);
v___x_2815_ = v___x_2620_;
goto v_reusejp_2814_;
}
else
{
lean_object* v_reuseFailAlloc_2825_; 
v_reuseFailAlloc_2825_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2825_, 0, v___x_2811_);
lean_ctor_set(v_reuseFailAlloc_2825_, 1, v___x_2813_);
v___x_2815_ = v_reuseFailAlloc_2825_;
goto v_reusejp_2814_;
}
v_reusejp_2814_:
{
lean_object* v___x_2816_; lean_object* v_a_2817_; lean_object* v___x_2819_; uint8_t v_isShared_2820_; uint8_t v_isSharedCheck_2824_; 
v___x_2816_ = lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2___redArg(v___x_2815_, v_a_2595_, v_a_2596_, v_a_2597_, v_a_2598_);
v_a_2817_ = lean_ctor_get(v___x_2816_, 0);
v_isSharedCheck_2824_ = !lean_is_exclusive(v___x_2816_);
if (v_isSharedCheck_2824_ == 0)
{
v___x_2819_ = v___x_2816_;
v_isShared_2820_ = v_isSharedCheck_2824_;
goto v_resetjp_2818_;
}
else
{
lean_inc(v_a_2817_);
lean_dec(v___x_2816_);
v___x_2819_ = lean_box(0);
v_isShared_2820_ = v_isSharedCheck_2824_;
goto v_resetjp_2818_;
}
v_resetjp_2818_:
{
lean_object* v___x_2822_; 
if (v_isShared_2820_ == 0)
{
v___x_2822_ = v___x_2819_;
goto v_reusejp_2821_;
}
else
{
lean_object* v_reuseFailAlloc_2823_; 
v_reuseFailAlloc_2823_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2823_, 0, v_a_2817_);
v___x_2822_ = v_reuseFailAlloc_2823_;
goto v_reusejp_2821_;
}
v_reusejp_2821_:
{
return v___x_2822_;
}
}
}
}
else
{
lean_object* v___x_2826_; lean_object* v___x_2827_; uint8_t v___x_2828_; 
v___x_2826_ = l_Lean_Syntax_getArg(v___x_2808_, v___x_2646_);
lean_dec(v___x_2808_);
v___x_2827_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__5));
v___x_2828_ = l_Lean_Syntax_isOfKind(v___x_2826_, v___x_2827_);
if (v___x_2828_ == 0)
{
lean_object* v___x_2829_; lean_object* v___x_2830_; lean_object* v___x_2831_; lean_object* v___x_2833_; 
lean_dec(v___x_2702_);
lean_dec(v___x_2647_);
v___x_2829_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4, &lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4_once, _init_lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4);
v___x_2830_ = l_Lean_MessageData_ofSyntax(v_a_2625_);
v___x_2831_ = l_Lean_indentD(v___x_2830_);
if (v_isShared_2621_ == 0)
{
lean_ctor_set_tag(v___x_2620_, 7);
lean_ctor_set(v___x_2620_, 1, v___x_2831_);
lean_ctor_set(v___x_2620_, 0, v___x_2829_);
v___x_2833_ = v___x_2620_;
goto v_reusejp_2832_;
}
else
{
lean_object* v_reuseFailAlloc_2843_; 
v_reuseFailAlloc_2843_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2843_, 0, v___x_2829_);
lean_ctor_set(v_reuseFailAlloc_2843_, 1, v___x_2831_);
v___x_2833_ = v_reuseFailAlloc_2843_;
goto v_reusejp_2832_;
}
v_reusejp_2832_:
{
lean_object* v___x_2834_; lean_object* v_a_2835_; lean_object* v___x_2837_; uint8_t v_isShared_2838_; uint8_t v_isSharedCheck_2842_; 
v___x_2834_ = lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2___redArg(v___x_2833_, v_a_2595_, v_a_2596_, v_a_2597_, v_a_2598_);
v_a_2835_ = lean_ctor_get(v___x_2834_, 0);
v_isSharedCheck_2842_ = !lean_is_exclusive(v___x_2834_);
if (v_isSharedCheck_2842_ == 0)
{
v___x_2837_ = v___x_2834_;
v_isShared_2838_ = v_isSharedCheck_2842_;
goto v_resetjp_2836_;
}
else
{
lean_inc(v_a_2835_);
lean_dec(v___x_2834_);
v___x_2837_ = lean_box(0);
v_isShared_2838_ = v_isSharedCheck_2842_;
goto v_resetjp_2836_;
}
v_resetjp_2836_:
{
lean_object* v___x_2840_; 
if (v_isShared_2838_ == 0)
{
v___x_2840_ = v___x_2837_;
goto v_reusejp_2839_;
}
else
{
lean_object* v_reuseFailAlloc_2841_; 
v_reuseFailAlloc_2841_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2841_, 0, v_a_2835_);
v___x_2840_ = v_reuseFailAlloc_2841_;
goto v_reusejp_2839_;
}
v_reusejp_2839_:
{
return v___x_2840_;
}
}
}
}
else
{
lean_object* v_ref_2844_; lean_object* v___x_2845_; lean_object* v___x_2846_; lean_object* v___x_2847_; lean_object* v___x_2849_; 
lean_dec(v_a_2625_);
v_ref_2844_ = lean_ctor_get(v_a_2597_, 5);
v___x_2845_ = l_Lean_Syntax_getArg(v___x_2702_, v___x_2646_);
lean_dec(v___x_2702_);
v___x_2846_ = l_Lean_Syntax_getArgs(v___x_2845_);
lean_dec(v___x_2845_);
v___x_2847_ = l_Lean_SourceInfo_fromRef(v_ref_2844_, v___x_2627_);
lean_inc(v___x_2847_);
if (v_isShared_2621_ == 0)
{
lean_ctor_set_tag(v___x_2620_, 2);
lean_ctor_set(v___x_2620_, 1, v___x_2628_);
lean_ctor_set(v___x_2620_, 0, v___x_2847_);
v___x_2849_ = v___x_2620_;
goto v_reusejp_2848_;
}
else
{
lean_object* v_reuseFailAlloc_2868_; 
v_reuseFailAlloc_2868_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2868_, 0, v___x_2847_);
lean_ctor_set(v_reuseFailAlloc_2868_, 1, v___x_2628_);
v___x_2849_ = v_reuseFailAlloc_2868_;
goto v_reusejp_2848_;
}
v_reusejp_2848_:
{
lean_object* v___x_2850_; lean_object* v___x_2851_; lean_object* v___x_2852_; lean_object* v___x_2853_; lean_object* v___x_2854_; lean_object* v___x_2855_; lean_object* v___x_2856_; lean_object* v___x_2857_; lean_object* v___x_2858_; lean_object* v___x_2859_; lean_object* v___x_2860_; lean_object* v___x_2861_; lean_object* v___x_2862_; lean_object* v___x_2863_; lean_object* v___x_2864_; lean_object* v___x_2865_; lean_object* v___x_2866_; lean_object* v___x_2867_; 
v___x_2850_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_2851_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
lean_inc_n(v___x_2847_, 10);
v___x_2852_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2852_, 0, v___x_2847_);
lean_ctor_set(v___x_2852_, 1, v___x_2850_);
lean_ctor_set(v___x_2852_, 2, v___x_2851_);
v___x_2853_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__5));
v___x_2854_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2854_, 0, v___x_2847_);
lean_ctor_set(v___x_2854_, 1, v___x_2853_);
v___x_2855_ = l_Array_append___redArg(v___x_2851_, v___x_2846_);
lean_dec_ref(v___x_2846_);
v___x_2856_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2856_, 0, v___x_2847_);
lean_ctor_set(v___x_2856_, 1, v___x_2850_);
lean_ctor_set(v___x_2856_, 2, v___x_2855_);
v___x_2857_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__6));
v___x_2858_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2858_, 0, v___x_2847_);
lean_ctor_set(v___x_2858_, 1, v___x_2857_);
v___x_2859_ = l_Lean_Syntax_node3(v___x_2847_, v___x_2850_, v___x_2854_, v___x_2856_, v___x_2858_);
v___x_2860_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__2));
v___x_2861_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2861_, 0, v___x_2847_);
lean_ctor_set(v___x_2861_, 1, v___x_2860_);
v___x_2862_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__6));
v___x_2863_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2863_, 0, v___x_2847_);
lean_ctor_set(v___x_2863_, 1, v___x_2862_);
v___x_2864_ = l_Lean_Syntax_node1(v___x_2847_, v___x_2827_, v___x_2863_);
v___x_2865_ = l_Lean_Syntax_node2(v___x_2847_, v___x_2809_, v___x_2861_, v___x_2864_);
v___x_2866_ = l_Lean_Syntax_node1(v___x_2847_, v___x_2850_, v___x_2865_);
lean_inc_ref(v___x_2852_);
v___x_2867_ = l_Lean_Syntax_node6(v___x_2847_, v___x_2629_, v___x_2849_, v___x_2647_, v___x_2852_, v___x_2852_, v___x_2859_, v___x_2866_);
v_stx_2601_ = v___x_2867_;
goto v___jp_2600_;
}
}
}
}
}
}
}
}
}
}
else
{
lean_object* v___x_2869_; lean_object* v___x_2870_; lean_object* v___x_2871_; uint8_t v___x_2872_; 
v___x_2869_ = lean_unsigned_to_nat(1u);
v___x_2870_ = l_Lean_Syntax_getArg(v_a_2625_, v___x_2869_);
v___x_2871_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__0___closed__3));
lean_inc(v___x_2870_);
v___x_2872_ = l_Lean_Syntax_isOfKind(v___x_2870_, v___x_2871_);
if (v___x_2872_ == 0)
{
lean_object* v___x_2873_; lean_object* v___x_2874_; lean_object* v___x_2875_; lean_object* v___x_2877_; 
lean_dec(v___x_2870_);
v___x_2873_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4, &lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4_once, _init_lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4);
v___x_2874_ = l_Lean_MessageData_ofSyntax(v_a_2625_);
v___x_2875_ = l_Lean_indentD(v___x_2874_);
if (v_isShared_2621_ == 0)
{
lean_ctor_set_tag(v___x_2620_, 7);
lean_ctor_set(v___x_2620_, 1, v___x_2875_);
lean_ctor_set(v___x_2620_, 0, v___x_2873_);
v___x_2877_ = v___x_2620_;
goto v_reusejp_2876_;
}
else
{
lean_object* v_reuseFailAlloc_2887_; 
v_reuseFailAlloc_2887_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2887_, 0, v___x_2873_);
lean_ctor_set(v_reuseFailAlloc_2887_, 1, v___x_2875_);
v___x_2877_ = v_reuseFailAlloc_2887_;
goto v_reusejp_2876_;
}
v_reusejp_2876_:
{
lean_object* v___x_2878_; lean_object* v_a_2879_; lean_object* v___x_2881_; uint8_t v_isShared_2882_; uint8_t v_isSharedCheck_2886_; 
v___x_2878_ = lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2___redArg(v___x_2877_, v_a_2595_, v_a_2596_, v_a_2597_, v_a_2598_);
v_a_2879_ = lean_ctor_get(v___x_2878_, 0);
v_isSharedCheck_2886_ = !lean_is_exclusive(v___x_2878_);
if (v_isSharedCheck_2886_ == 0)
{
v___x_2881_ = v___x_2878_;
v_isShared_2882_ = v_isSharedCheck_2886_;
goto v_resetjp_2880_;
}
else
{
lean_inc(v_a_2879_);
lean_dec(v___x_2878_);
v___x_2881_ = lean_box(0);
v_isShared_2882_ = v_isSharedCheck_2886_;
goto v_resetjp_2880_;
}
v_resetjp_2880_:
{
lean_object* v___x_2884_; 
if (v_isShared_2882_ == 0)
{
v___x_2884_ = v___x_2881_;
goto v_reusejp_2883_;
}
else
{
lean_object* v_reuseFailAlloc_2885_; 
v_reuseFailAlloc_2885_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2885_, 0, v_a_2879_);
v___x_2884_ = v_reuseFailAlloc_2885_;
goto v_reusejp_2883_;
}
v_reusejp_2883_:
{
return v___x_2884_;
}
}
}
}
else
{
lean_object* v___x_2888_; lean_object* v___x_2889_; uint8_t v___x_2890_; 
v___x_2888_ = lean_unsigned_to_nat(2u);
v___x_2889_ = l_Lean_Syntax_getArg(v_a_2625_, v___x_2888_);
v___x_2890_ = l_Lean_Syntax_matchesNull(v___x_2889_, v___x_2613_);
if (v___x_2890_ == 0)
{
lean_object* v___x_2891_; lean_object* v___x_2892_; lean_object* v___x_2893_; lean_object* v___x_2895_; 
lean_dec(v___x_2870_);
v___x_2891_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4, &lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4_once, _init_lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4);
v___x_2892_ = l_Lean_MessageData_ofSyntax(v_a_2625_);
v___x_2893_ = l_Lean_indentD(v___x_2892_);
if (v_isShared_2621_ == 0)
{
lean_ctor_set_tag(v___x_2620_, 7);
lean_ctor_set(v___x_2620_, 1, v___x_2893_);
lean_ctor_set(v___x_2620_, 0, v___x_2891_);
v___x_2895_ = v___x_2620_;
goto v_reusejp_2894_;
}
else
{
lean_object* v_reuseFailAlloc_2905_; 
v_reuseFailAlloc_2905_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2905_, 0, v___x_2891_);
lean_ctor_set(v_reuseFailAlloc_2905_, 1, v___x_2893_);
v___x_2895_ = v_reuseFailAlloc_2905_;
goto v_reusejp_2894_;
}
v_reusejp_2894_:
{
lean_object* v___x_2896_; lean_object* v_a_2897_; lean_object* v___x_2899_; uint8_t v_isShared_2900_; uint8_t v_isSharedCheck_2904_; 
v___x_2896_ = lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2___redArg(v___x_2895_, v_a_2595_, v_a_2596_, v_a_2597_, v_a_2598_);
v_a_2897_ = lean_ctor_get(v___x_2896_, 0);
v_isSharedCheck_2904_ = !lean_is_exclusive(v___x_2896_);
if (v_isSharedCheck_2904_ == 0)
{
v___x_2899_ = v___x_2896_;
v_isShared_2900_ = v_isSharedCheck_2904_;
goto v_resetjp_2898_;
}
else
{
lean_inc(v_a_2897_);
lean_dec(v___x_2896_);
v___x_2899_ = lean_box(0);
v_isShared_2900_ = v_isSharedCheck_2904_;
goto v_resetjp_2898_;
}
v_resetjp_2898_:
{
lean_object* v___x_2902_; 
if (v_isShared_2900_ == 0)
{
v___x_2902_ = v___x_2899_;
goto v_reusejp_2901_;
}
else
{
lean_object* v_reuseFailAlloc_2903_; 
v_reuseFailAlloc_2903_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2903_, 0, v_a_2897_);
v___x_2902_ = v_reuseFailAlloc_2903_;
goto v_reusejp_2901_;
}
v_reusejp_2901_:
{
return v___x_2902_;
}
}
}
}
else
{
lean_object* v___x_2906_; lean_object* v___x_2907_; uint8_t v___x_2908_; 
v___x_2906_ = lean_unsigned_to_nat(3u);
v___x_2907_ = l_Lean_Syntax_getArg(v_a_2625_, v___x_2906_);
v___x_2908_ = l_Lean_Syntax_matchesNull(v___x_2907_, v___x_2869_);
if (v___x_2908_ == 0)
{
lean_object* v___x_2909_; lean_object* v___x_2910_; lean_object* v___x_2911_; lean_object* v___x_2913_; 
lean_dec(v___x_2870_);
v___x_2909_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4, &lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4_once, _init_lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4);
v___x_2910_ = l_Lean_MessageData_ofSyntax(v_a_2625_);
v___x_2911_ = l_Lean_indentD(v___x_2910_);
if (v_isShared_2621_ == 0)
{
lean_ctor_set_tag(v___x_2620_, 7);
lean_ctor_set(v___x_2620_, 1, v___x_2911_);
lean_ctor_set(v___x_2620_, 0, v___x_2909_);
v___x_2913_ = v___x_2620_;
goto v_reusejp_2912_;
}
else
{
lean_object* v_reuseFailAlloc_2923_; 
v_reuseFailAlloc_2923_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2923_, 0, v___x_2909_);
lean_ctor_set(v_reuseFailAlloc_2923_, 1, v___x_2911_);
v___x_2913_ = v_reuseFailAlloc_2923_;
goto v_reusejp_2912_;
}
v_reusejp_2912_:
{
lean_object* v___x_2914_; lean_object* v_a_2915_; lean_object* v___x_2917_; uint8_t v_isShared_2918_; uint8_t v_isSharedCheck_2922_; 
v___x_2914_ = lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2___redArg(v___x_2913_, v_a_2595_, v_a_2596_, v_a_2597_, v_a_2598_);
v_a_2915_ = lean_ctor_get(v___x_2914_, 0);
v_isSharedCheck_2922_ = !lean_is_exclusive(v___x_2914_);
if (v_isSharedCheck_2922_ == 0)
{
v___x_2917_ = v___x_2914_;
v_isShared_2918_ = v_isSharedCheck_2922_;
goto v_resetjp_2916_;
}
else
{
lean_inc(v_a_2915_);
lean_dec(v___x_2914_);
v___x_2917_ = lean_box(0);
v_isShared_2918_ = v_isSharedCheck_2922_;
goto v_resetjp_2916_;
}
v_resetjp_2916_:
{
lean_object* v___x_2920_; 
if (v_isShared_2918_ == 0)
{
v___x_2920_ = v___x_2917_;
goto v_reusejp_2919_;
}
else
{
lean_object* v_reuseFailAlloc_2921_; 
v_reuseFailAlloc_2921_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2921_, 0, v_a_2915_);
v___x_2920_ = v_reuseFailAlloc_2921_;
goto v_reusejp_2919_;
}
v_reusejp_2919_:
{
return v___x_2920_;
}
}
}
}
else
{
lean_object* v___x_2924_; lean_object* v___x_2925_; uint8_t v___x_2926_; 
v___x_2924_ = lean_unsigned_to_nat(4u);
v___x_2925_ = l_Lean_Syntax_getArg(v_a_2625_, v___x_2924_);
lean_inc(v___x_2925_);
v___x_2926_ = l_Lean_Syntax_matchesNull(v___x_2925_, v___x_2906_);
if (v___x_2926_ == 0)
{
uint8_t v___x_2927_; 
v___x_2927_ = l_Lean_Syntax_matchesNull(v___x_2925_, v___x_2613_);
if (v___x_2927_ == 0)
{
lean_object* v___x_2928_; lean_object* v___x_2929_; lean_object* v___x_2930_; lean_object* v___x_2932_; 
lean_dec(v___x_2870_);
v___x_2928_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4, &lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4_once, _init_lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__4);
v___x_2929_ = l_Lean_MessageData_ofSyntax(v_a_2625_);
v___x_2930_ = l_Lean_indentD(v___x_2929_);
if (v_isShared_2621_ == 0)
{
lean_ctor_set_tag(v___x_2620_, 7);
lean_ctor_set(v___x_2620_, 1, v___x_2930_);
lean_ctor_set(v___x_2620_, 0, v___x_2928_);
v___x_2932_ = v___x_2620_;
goto v_reusejp_2931_;
}
else
{
lean_object* v_reuseFailAlloc_2942_; 
v_reuseFailAlloc_2942_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2942_, 0, v___x_2928_);
lean_ctor_set(v_reuseFailAlloc_2942_, 1, v___x_2930_);
v___x_2932_ = v_reuseFailAlloc_2942_;
goto v_reusejp_2931_;
}
v_reusejp_2931_:
{
lean_object* v___x_2933_; lean_object* v_a_2934_; lean_object* v___x_2936_; uint8_t v_isShared_2937_; uint8_t v_isSharedCheck_2941_; 
v___x_2933_ = lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2___redArg(v___x_2932_, v_a_2595_, v_a_2596_, v_a_2597_, v_a_2598_);
v_a_2934_ = lean_ctor_get(v___x_2933_, 0);
v_isSharedCheck_2941_ = !lean_is_exclusive(v___x_2933_);
if (v_isSharedCheck_2941_ == 0)
{
v___x_2936_ = v___x_2933_;
v_isShared_2937_ = v_isSharedCheck_2941_;
goto v_resetjp_2935_;
}
else
{
lean_inc(v_a_2934_);
lean_dec(v___x_2933_);
v___x_2936_ = lean_box(0);
v_isShared_2937_ = v_isSharedCheck_2941_;
goto v_resetjp_2935_;
}
v_resetjp_2935_:
{
lean_object* v___x_2939_; 
if (v_isShared_2937_ == 0)
{
v___x_2939_ = v___x_2936_;
goto v_reusejp_2938_;
}
else
{
lean_object* v_reuseFailAlloc_2940_; 
v_reuseFailAlloc_2940_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2940_, 0, v_a_2934_);
v___x_2939_ = v_reuseFailAlloc_2940_;
goto v_reusejp_2938_;
}
v_reusejp_2938_:
{
return v___x_2939_;
}
}
}
}
else
{
lean_object* v_ref_2943_; lean_object* v___x_2944_; lean_object* v___x_2945_; lean_object* v___x_2947_; 
lean_dec(v_a_2625_);
v_ref_2943_ = lean_ctor_get(v_a_2597_, 5);
v___x_2944_ = l_Lean_SourceInfo_fromRef(v_ref_2943_, v___x_2926_);
v___x_2945_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__2));
lean_inc(v___x_2944_);
if (v_isShared_2621_ == 0)
{
lean_ctor_set_tag(v___x_2620_, 2);
lean_ctor_set(v___x_2620_, 1, v___x_2945_);
lean_ctor_set(v___x_2620_, 0, v___x_2944_);
v___x_2947_ = v___x_2620_;
goto v_reusejp_2946_;
}
else
{
lean_object* v_reuseFailAlloc_2952_; 
v_reuseFailAlloc_2952_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2952_, 0, v___x_2944_);
lean_ctor_set(v_reuseFailAlloc_2952_, 1, v___x_2945_);
v___x_2947_ = v_reuseFailAlloc_2952_;
goto v_reusejp_2946_;
}
v_reusejp_2946_:
{
lean_object* v___x_2948_; lean_object* v___x_2949_; lean_object* v___x_2950_; lean_object* v___x_2951_; 
v___x_2948_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_2949_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
lean_inc(v___x_2944_);
v___x_2950_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2950_, 0, v___x_2944_);
lean_ctor_set(v___x_2950_, 1, v___x_2948_);
lean_ctor_set(v___x_2950_, 2, v___x_2949_);
lean_inc_ref_n(v___x_2950_, 2);
v___x_2951_ = l_Lean_Syntax_node5(v___x_2944_, v___x_2626_, v___x_2947_, v___x_2870_, v___x_2950_, v___x_2950_, v___x_2950_);
v_stx_2601_ = v___x_2951_;
goto v___jp_2600_;
}
}
}
else
{
lean_object* v_ref_2953_; lean_object* v___x_2954_; lean_object* v___x_2955_; uint8_t v___x_2956_; lean_object* v___x_2957_; lean_object* v___x_2958_; lean_object* v___x_2960_; 
lean_dec(v_a_2625_);
v_ref_2953_ = lean_ctor_get(v_a_2597_, 5);
v___x_2954_ = l_Lean_Syntax_getArg(v___x_2925_, v___x_2869_);
lean_dec(v___x_2925_);
v___x_2955_ = l_Lean_Syntax_getArgs(v___x_2954_);
lean_dec(v___x_2954_);
v___x_2956_ = 0;
v___x_2957_ = l_Lean_SourceInfo_fromRef(v_ref_2953_, v___x_2956_);
v___x_2958_ = ((lean_object*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_Script_TacticBuilder_simpAllOrSimpAtStarStx___redArg___lam__7___closed__2));
lean_inc(v___x_2957_);
if (v_isShared_2621_ == 0)
{
lean_ctor_set_tag(v___x_2620_, 2);
lean_ctor_set(v___x_2620_, 1, v___x_2958_);
lean_ctor_set(v___x_2620_, 0, v___x_2957_);
v___x_2960_ = v___x_2620_;
goto v_reusejp_2959_;
}
else
{
lean_object* v_reuseFailAlloc_2972_; 
v_reuseFailAlloc_2972_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2972_, 0, v___x_2957_);
lean_ctor_set(v_reuseFailAlloc_2972_, 1, v___x_2958_);
v___x_2960_ = v_reuseFailAlloc_2972_;
goto v_reusejp_2959_;
}
v_reusejp_2959_:
{
lean_object* v___x_2961_; lean_object* v___x_2962_; lean_object* v___x_2963_; lean_object* v___x_2964_; lean_object* v___x_2965_; lean_object* v___x_2966_; lean_object* v___x_2967_; lean_object* v___x_2968_; lean_object* v___x_2969_; lean_object* v___x_2970_; lean_object* v___x_2971_; 
v___x_2961_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_2962_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
lean_inc_n(v___x_2957_, 5);
v___x_2963_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2963_, 0, v___x_2957_);
lean_ctor_set(v___x_2963_, 1, v___x_2961_);
lean_ctor_set(v___x_2963_, 2, v___x_2962_);
v___x_2964_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__5));
v___x_2965_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2965_, 0, v___x_2957_);
lean_ctor_set(v___x_2965_, 1, v___x_2964_);
v___x_2966_ = l_Array_append___redArg(v___x_2962_, v___x_2955_);
lean_dec_ref(v___x_2955_);
v___x_2967_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2967_, 0, v___x_2957_);
lean_ctor_set(v___x_2967_, 1, v___x_2961_);
lean_ctor_set(v___x_2967_, 2, v___x_2966_);
v___x_2968_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___closed__6));
v___x_2969_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2969_, 0, v___x_2957_);
lean_ctor_set(v___x_2969_, 1, v___x_2968_);
v___x_2970_ = l_Lean_Syntax_node3(v___x_2957_, v___x_2961_, v___x_2965_, v___x_2967_, v___x_2969_);
lean_inc_ref(v___x_2963_);
v___x_2971_ = l_Lean_Syntax_node5(v___x_2957_, v___x_2626_, v___x_2960_, v___x_2870_, v___x_2963_, v___x_2963_, v___x_2970_);
v_stx_2601_ = v___x_2971_;
goto v___jp_2600_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_2973_; lean_object* v___x_2975_; uint8_t v_isShared_2976_; uint8_t v_isSharedCheck_2980_; 
lean_del_object(v___x_2620_);
v_a_2973_ = lean_ctor_get(v___x_2624_, 0);
v_isSharedCheck_2980_ = !lean_is_exclusive(v___x_2624_);
if (v_isSharedCheck_2980_ == 0)
{
v___x_2975_ = v___x_2624_;
v_isShared_2976_ = v_isSharedCheck_2980_;
goto v_resetjp_2974_;
}
else
{
lean_inc(v_a_2973_);
lean_dec(v___x_2624_);
v___x_2975_ = lean_box(0);
v_isShared_2976_ = v_isSharedCheck_2980_;
goto v_resetjp_2974_;
}
v_resetjp_2974_:
{
lean_object* v___x_2978_; 
if (v_isShared_2976_ == 0)
{
v___x_2978_ = v___x_2975_;
goto v_reusejp_2977_;
}
else
{
lean_object* v_reuseFailAlloc_2979_; 
v_reuseFailAlloc_2979_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2979_, 0, v_a_2973_);
v___x_2978_ = v_reuseFailAlloc_2979_;
goto v_reusejp_2977_;
}
v_reusejp_2977_:
{
return v___x_2978_;
}
}
}
}
}
}
else
{
lean_object* v_a_2983_; lean_object* v___x_2985_; uint8_t v_isShared_2986_; uint8_t v_isSharedCheck_2990_; 
lean_del_object(v___x_2610_);
lean_dec(v_configStx_x3f_2593_);
lean_dec(v_inGoal_2592_);
v_a_2983_ = lean_ctor_get(v___x_2615_, 0);
v_isSharedCheck_2990_ = !lean_is_exclusive(v___x_2615_);
if (v_isSharedCheck_2990_ == 0)
{
v___x_2985_ = v___x_2615_;
v_isShared_2986_ = v_isSharedCheck_2990_;
goto v_resetjp_2984_;
}
else
{
lean_inc(v_a_2983_);
lean_dec(v___x_2615_);
v___x_2985_ = lean_box(0);
v_isShared_2986_ = v_isSharedCheck_2990_;
goto v_resetjp_2984_;
}
v_resetjp_2984_:
{
lean_object* v___x_2988_; 
if (v_isShared_2986_ == 0)
{
v___x_2988_ = v___x_2985_;
goto v_reusejp_2987_;
}
else
{
lean_object* v_reuseFailAlloc_2989_; 
v_reuseFailAlloc_2989_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2989_, 0, v_a_2983_);
v___x_2988_ = v_reuseFailAlloc_2989_;
goto v_reusejp_2987_;
}
v_reusejp_2987_:
{
return v___x_2988_;
}
}
}
}
}
else
{
lean_object* v_a_2993_; lean_object* v___x_2995_; uint8_t v_isShared_2996_; uint8_t v_isSharedCheck_3000_; 
lean_dec(v_a_2605_);
lean_dec_ref(v_usedTheorems_2594_);
lean_dec(v_configStx_x3f_2593_);
lean_dec(v_inGoal_2592_);
v_a_2993_ = lean_ctor_get(v___x_2606_, 0);
v_isSharedCheck_3000_ = !lean_is_exclusive(v___x_2606_);
if (v_isSharedCheck_3000_ == 0)
{
v___x_2995_ = v___x_2606_;
v_isShared_2996_ = v_isSharedCheck_3000_;
goto v_resetjp_2994_;
}
else
{
lean_inc(v_a_2993_);
lean_dec(v___x_2606_);
v___x_2995_ = lean_box(0);
v_isShared_2996_ = v_isSharedCheck_3000_;
goto v_resetjp_2994_;
}
v_resetjp_2994_:
{
lean_object* v___x_2998_; 
if (v_isShared_2996_ == 0)
{
v___x_2998_ = v___x_2995_;
goto v_reusejp_2997_;
}
else
{
lean_object* v_reuseFailAlloc_2999_; 
v_reuseFailAlloc_2999_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2999_, 0, v_a_2993_);
v___x_2998_ = v_reuseFailAlloc_2999_;
goto v_reusejp_2997_;
}
v_reusejp_2997_:
{
return v___x_2998_;
}
}
}
}
else
{
lean_object* v_a_3001_; lean_object* v___x_3003_; uint8_t v_isShared_3004_; uint8_t v_isSharedCheck_3008_; 
lean_dec_ref(v_usedTheorems_2594_);
lean_dec(v_configStx_x3f_2593_);
lean_dec(v_inGoal_2592_);
v_a_3001_ = lean_ctor_get(v___x_2604_, 0);
v_isSharedCheck_3008_ = !lean_is_exclusive(v___x_2604_);
if (v_isSharedCheck_3008_ == 0)
{
v___x_3003_ = v___x_2604_;
v_isShared_3004_ = v_isSharedCheck_3008_;
goto v_resetjp_3002_;
}
else
{
lean_inc(v_a_3001_);
lean_dec(v___x_2604_);
v___x_3003_ = lean_box(0);
v_isShared_3004_ = v_isSharedCheck_3008_;
goto v_resetjp_3002_;
}
v_resetjp_3002_:
{
lean_object* v___x_3006_; 
if (v_isShared_3004_ == 0)
{
v___x_3006_ = v___x_3003_;
goto v_reusejp_3005_;
}
else
{
lean_object* v_reuseFailAlloc_3007_; 
v_reuseFailAlloc_3007_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3007_, 0, v_a_3001_);
v___x_3006_ = v_reuseFailAlloc_3007_;
goto v_reusejp_3005_;
}
v_reusejp_3005_:
{
return v___x_3006_;
}
}
}
v___jp_2600_:
{
lean_object* v___x_2602_; lean_object* v___x_2603_; 
v___x_2602_ = lp_aesop_Aesop_Script_Tactic_unstructured(v_stx_2601_);
v___x_2603_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2603_, 0, v___x_2602_);
return v___x_2603_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar___boxed(lean_object* v_simpAll_3009_, lean_object* v_inGoal_3010_, lean_object* v_configStx_x3f_3011_, lean_object* v_usedTheorems_3012_, lean_object* v_a_3013_, lean_object* v_a_3014_, lean_object* v_a_3015_, lean_object* v_a_3016_, lean_object* v_a_3017_){
_start:
{
uint8_t v_simpAll_boxed_3018_; lean_object* v_res_3019_; 
v_simpAll_boxed_3018_ = lean_unbox(v_simpAll_3009_);
v_res_3019_ = lp_aesop_Aesop_Script_TacticBuilder_simpAllOrSimpAtStar(v_simpAll_boxed_3018_, v_inGoal_3010_, v_configStx_x3f_3011_, v_usedTheorems_3012_, v_a_3013_, v_a_3014_, v_a_3015_, v_a_3016_);
lean_dec(v_a_3016_);
lean_dec_ref(v_a_3015_);
lean_dec(v_a_3014_);
lean_dec_ref(v_a_3013_);
return v_res_3019_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0(lean_object* v_00_u03b2_3020_, lean_object* v_x_3021_, lean_object* v_x_3022_, lean_object* v_x_3023_){
_start:
{
lean_object* v___x_3024_; 
v___x_3024_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0___redArg(v_x_3021_, v_x_3022_, v_x_3023_);
return v___x_3024_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1___redArg(lean_object* v_map_3025_, lean_object* v_f_3026_, lean_object* v_init_3027_, lean_object* v___y_3028_, lean_object* v___y_3029_, lean_object* v___y_3030_, lean_object* v___y_3031_){
_start:
{
lean_object* v___x_3033_; 
v___x_3033_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2___redArg(v_f_3026_, v_map_3025_, v_init_3027_, v___y_3028_, v___y_3029_, v___y_3030_, v___y_3031_);
return v___x_3033_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1___redArg___boxed(lean_object* v_map_3034_, lean_object* v_f_3035_, lean_object* v_init_3036_, lean_object* v___y_3037_, lean_object* v___y_3038_, lean_object* v___y_3039_, lean_object* v___y_3040_, lean_object* v___y_3041_){
_start:
{
lean_object* v_res_3042_; 
v_res_3042_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1___redArg(v_map_3034_, v_f_3035_, v_init_3036_, v___y_3037_, v___y_3038_, v___y_3039_, v___y_3040_);
lean_dec(v___y_3040_);
lean_dec_ref(v___y_3039_);
lean_dec(v___y_3038_);
lean_dec_ref(v___y_3037_);
return v_res_3042_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1(lean_object* v_00_u03c3_3043_, lean_object* v_00_u03b2_3044_, lean_object* v_map_3045_, lean_object* v_f_3046_, lean_object* v_init_3047_, lean_object* v___y_3048_, lean_object* v___y_3049_, lean_object* v___y_3050_, lean_object* v___y_3051_){
_start:
{
lean_object* v___x_3053_; 
v___x_3053_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2___redArg(v_f_3046_, v_map_3045_, v_init_3047_, v___y_3048_, v___y_3049_, v___y_3050_, v___y_3051_);
return v___x_3053_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1___boxed(lean_object* v_00_u03c3_3054_, lean_object* v_00_u03b2_3055_, lean_object* v_map_3056_, lean_object* v_f_3057_, lean_object* v_init_3058_, lean_object* v___y_3059_, lean_object* v___y_3060_, lean_object* v___y_3061_, lean_object* v___y_3062_, lean_object* v___y_3063_){
_start:
{
lean_object* v_res_3064_; 
v_res_3064_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1(v_00_u03c3_3054_, v_00_u03b2_3055_, v_map_3056_, v_f_3057_, v_init_3058_, v___y_3059_, v___y_3060_, v___y_3061_, v___y_3062_);
lean_dec(v___y_3062_);
lean_dec_ref(v___y_3061_);
lean_dec(v___y_3060_);
lean_dec_ref(v___y_3059_);
return v_res_3064_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2(lean_object* v_00_u03b1_3065_, lean_object* v_msg_3066_, lean_object* v___y_3067_, lean_object* v___y_3068_, lean_object* v___y_3069_, lean_object* v___y_3070_){
_start:
{
lean_object* v___x_3072_; 
v___x_3072_ = lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2___redArg(v_msg_3066_, v___y_3067_, v___y_3068_, v___y_3069_, v___y_3070_);
return v___x_3072_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2___boxed(lean_object* v_00_u03b1_3073_, lean_object* v_msg_3074_, lean_object* v___y_3075_, lean_object* v___y_3076_, lean_object* v___y_3077_, lean_object* v___y_3078_, lean_object* v___y_3079_){
_start:
{
lean_object* v_res_3080_; 
v_res_3080_ = lp_aesop_Lean_throwError___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__2(v_00_u03b1_3073_, v_msg_3074_, v___y_3075_, v___y_3076_, v___y_3077_, v___y_3078_);
lean_dec(v___y_3078_);
lean_dec_ref(v___y_3077_);
lean_dec(v___y_3076_);
lean_dec_ref(v___y_3075_);
return v_res_3080_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0(lean_object* v_00_u03b2_3081_, lean_object* v_x_3082_, size_t v_x_3083_, size_t v_x_3084_, lean_object* v_x_3085_, lean_object* v_x_3086_){
_start:
{
lean_object* v___x_3087_; 
v___x_3087_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0___redArg(v_x_3082_, v_x_3083_, v_x_3084_, v_x_3085_, v_x_3086_);
return v___x_3087_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0___boxed(lean_object* v_00_u03b2_3088_, lean_object* v_x_3089_, lean_object* v_x_3090_, lean_object* v_x_3091_, lean_object* v_x_3092_, lean_object* v_x_3093_){
_start:
{
size_t v_x_24257__boxed_3094_; size_t v_x_24258__boxed_3095_; lean_object* v_res_3096_; 
v_x_24257__boxed_3094_ = lean_unbox_usize(v_x_3090_);
lean_dec(v_x_3090_);
v_x_24258__boxed_3095_ = lean_unbox_usize(v_x_3091_);
lean_dec(v_x_3091_);
v_res_3096_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0(v_00_u03b2_3088_, v_x_3089_, v_x_24257__boxed_3094_, v_x_24258__boxed_3095_, v_x_3092_, v_x_3093_);
return v_res_3096_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2(lean_object* v_00_u03c3_3097_, lean_object* v_00_u03b1_3098_, lean_object* v_00_u03b2_3099_, lean_object* v_f_3100_, lean_object* v_x_3101_, lean_object* v_x_3102_, lean_object* v___y_3103_, lean_object* v___y_3104_, lean_object* v___y_3105_, lean_object* v___y_3106_){
_start:
{
lean_object* v___x_3108_; 
v___x_3108_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2___redArg(v_f_3100_, v_x_3101_, v_x_3102_, v___y_3103_, v___y_3104_, v___y_3105_, v___y_3106_);
return v___x_3108_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2___boxed(lean_object* v_00_u03c3_3109_, lean_object* v_00_u03b1_3110_, lean_object* v_00_u03b2_3111_, lean_object* v_f_3112_, lean_object* v_x_3113_, lean_object* v_x_3114_, lean_object* v___y_3115_, lean_object* v___y_3116_, lean_object* v___y_3117_, lean_object* v___y_3118_, lean_object* v___y_3119_){
_start:
{
lean_object* v_res_3120_; 
v_res_3120_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2(v_00_u03c3_3109_, v_00_u03b1_3110_, v_00_u03b2_3111_, v_f_3112_, v_x_3113_, v_x_3114_, v___y_3115_, v___y_3116_, v___y_3117_, v___y_3118_);
lean_dec(v___y_3118_);
lean_dec_ref(v___y_3117_);
lean_dec(v___y_3116_);
lean_dec_ref(v___y_3115_);
return v_res_3120_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0_spec__1(lean_object* v_00_u03b2_3121_, lean_object* v_n_3122_, lean_object* v_k_3123_, lean_object* v_v_3124_){
_start:
{
lean_object* v___x_3125_; 
v___x_3125_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0_spec__1___redArg(v_n_3122_, v_k_3123_, v_v_3124_);
return v___x_3125_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0_spec__2(lean_object* v_00_u03b2_3126_, size_t v_depth_3127_, lean_object* v_keys_3128_, lean_object* v_vals_3129_, lean_object* v_heq_3130_, lean_object* v_i_3131_, lean_object* v_entries_3132_){
_start:
{
lean_object* v___x_3133_; 
v___x_3133_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0_spec__2___redArg(v_depth_3127_, v_keys_3128_, v_vals_3129_, v_i_3131_, v_entries_3132_);
return v___x_3133_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0_spec__2___boxed(lean_object* v_00_u03b2_3134_, lean_object* v_depth_3135_, lean_object* v_keys_3136_, lean_object* v_vals_3137_, lean_object* v_heq_3138_, lean_object* v_i_3139_, lean_object* v_entries_3140_){
_start:
{
size_t v_depth_boxed_3141_; lean_object* v_res_3142_; 
v_depth_boxed_3141_ = lean_unbox_usize(v_depth_3135_);
lean_dec(v_depth_3135_);
v_res_3142_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0_spec__2(v_00_u03b2_3134_, v_depth_boxed_3141_, v_keys_3136_, v_vals_3137_, v_heq_3138_, v_i_3139_, v_entries_3140_);
lean_dec_ref(v_vals_3137_);
lean_dec_ref(v_keys_3136_);
return v_res_3142_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2_spec__5(lean_object* v_00_u03b1_3143_, lean_object* v_00_u03b2_3144_, lean_object* v_00_u03c3_3145_, lean_object* v_f_3146_, lean_object* v_as_3147_, size_t v_i_3148_, size_t v_stop_3149_, lean_object* v_b_3150_, lean_object* v___y_3151_, lean_object* v___y_3152_, lean_object* v___y_3153_, lean_object* v___y_3154_){
_start:
{
lean_object* v___x_3156_; 
v___x_3156_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2_spec__5___redArg(v_f_3146_, v_as_3147_, v_i_3148_, v_stop_3149_, v_b_3150_, v___y_3151_, v___y_3152_, v___y_3153_, v___y_3154_);
return v___x_3156_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2_spec__5___boxed(lean_object* v_00_u03b1_3157_, lean_object* v_00_u03b2_3158_, lean_object* v_00_u03c3_3159_, lean_object* v_f_3160_, lean_object* v_as_3161_, lean_object* v_i_3162_, lean_object* v_stop_3163_, lean_object* v_b_3164_, lean_object* v___y_3165_, lean_object* v___y_3166_, lean_object* v___y_3167_, lean_object* v___y_3168_, lean_object* v___y_3169_){
_start:
{
size_t v_i_boxed_3170_; size_t v_stop_boxed_3171_; lean_object* v_res_3172_; 
v_i_boxed_3170_ = lean_unbox_usize(v_i_3162_);
lean_dec(v_i_3162_);
v_stop_boxed_3171_ = lean_unbox_usize(v_stop_3163_);
lean_dec(v_stop_3163_);
v_res_3172_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2_spec__5(v_00_u03b1_3157_, v_00_u03b2_3158_, v_00_u03c3_3159_, v_f_3160_, v_as_3161_, v_i_boxed_3170_, v_stop_boxed_3171_, v_b_3164_, v___y_3165_, v___y_3166_, v___y_3167_, v___y_3168_);
lean_dec(v___y_3168_);
lean_dec_ref(v___y_3167_);
lean_dec(v___y_3166_);
lean_dec_ref(v___y_3165_);
lean_dec_ref(v_as_3161_);
return v_res_3172_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2_spec__6(lean_object* v_00_u03c3_3173_, lean_object* v_00_u03b1_3174_, lean_object* v_00_u03b2_3175_, lean_object* v_f_3176_, lean_object* v_keys_3177_, lean_object* v_vals_3178_, lean_object* v_heq_3179_, lean_object* v_i_3180_, lean_object* v_acc_3181_, lean_object* v___y_3182_, lean_object* v___y_3183_, lean_object* v___y_3184_, lean_object* v___y_3185_){
_start:
{
lean_object* v___x_3187_; 
v___x_3187_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2_spec__6___redArg(v_f_3176_, v_keys_3177_, v_vals_3178_, v_i_3180_, v_acc_3181_, v___y_3182_, v___y_3183_, v___y_3184_, v___y_3185_);
return v___x_3187_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2_spec__6___boxed(lean_object* v_00_u03c3_3188_, lean_object* v_00_u03b1_3189_, lean_object* v_00_u03b2_3190_, lean_object* v_f_3191_, lean_object* v_keys_3192_, lean_object* v_vals_3193_, lean_object* v_heq_3194_, lean_object* v_i_3195_, lean_object* v_acc_3196_, lean_object* v___y_3197_, lean_object* v___y_3198_, lean_object* v___y_3199_, lean_object* v___y_3200_, lean_object* v___y_3201_){
_start:
{
lean_object* v_res_3202_; 
v_res_3202_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__1_spec__2_spec__6(v_00_u03c3_3188_, v_00_u03b1_3189_, v_00_u03b2_3190_, v_f_3191_, v_keys_3192_, v_vals_3193_, v_heq_3194_, v_i_3195_, v_acc_3196_, v___y_3197_, v___y_3198_, v___y_3199_, v___y_3200_);
lean_dec(v___y_3200_);
lean_dec_ref(v___y_3199_);
lean_dec(v___y_3198_);
lean_dec_ref(v___y_3197_);
lean_dec_ref(v_vals_3193_);
lean_dec_ref(v_keys_3192_);
return v_res_3202_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0_spec__1_spec__5(lean_object* v_00_u03b2_3203_, lean_object* v_x_3204_, lean_object* v_x_3205_, lean_object* v_x_3206_, lean_object* v_x_3207_){
_start:
{
lean_object* v___x_3208_; 
v___x_3208_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Script_TacticBuilder_simpAllOrSimpAtStar_spec__0_spec__0_spec__1_spec__5___redArg(v_x_3204_, v_x_3205_, v_x_3206_, v_x_3207_);
return v___x_3208_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_intros(lean_object* v_postGoal_3215_, lean_object* v_newFVarIds_3216_, uint8_t v_md_3217_, lean_object* v_a_3218_, lean_object* v_a_3219_, lean_object* v_a_3220_, lean_object* v_a_3221_){
_start:
{
size_t v_sz_3223_; lean_object* v___x_3224_; lean_object* v___x_3225_; lean_object* v___x_3226_; lean_object* v___x_3227_; 
v_sz_3223_ = lean_array_size(v_newFVarIds_3216_);
v___x_3224_ = lean_box_usize(v_sz_3223_);
v___x_3225_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticBuilder_extN_spec__1___boxed__const__1));
v___x_3226_ = lean_alloc_closure((void*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_clear_spec__0___boxed), 8, 3);
lean_closure_set(v___x_3226_, 0, v___x_3224_);
lean_closure_set(v___x_3226_, 1, v___x_3225_);
lean_closure_set(v___x_3226_, 2, v_newFVarIds_3216_);
v___x_3227_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___redArg(v_postGoal_3215_, v___x_3226_, v_a_3218_, v_a_3219_, v_a_3220_, v_a_3221_);
if (lean_obj_tag(v___x_3227_) == 0)
{
lean_object* v_a_3228_; lean_object* v___x_3230_; uint8_t v_isShared_3231_; uint8_t v_isSharedCheck_3248_; 
v_a_3228_ = lean_ctor_get(v___x_3227_, 0);
v_isSharedCheck_3248_ = !lean_is_exclusive(v___x_3227_);
if (v_isSharedCheck_3248_ == 0)
{
v___x_3230_ = v___x_3227_;
v_isShared_3231_ = v_isSharedCheck_3248_;
goto v_resetjp_3229_;
}
else
{
lean_inc(v_a_3228_);
lean_dec(v___x_3227_);
v___x_3230_ = lean_box(0);
v_isShared_3231_ = v_isSharedCheck_3248_;
goto v_resetjp_3229_;
}
v_resetjp_3229_:
{
lean_object* v_ref_3232_; uint8_t v___x_3233_; lean_object* v___x_3234_; lean_object* v___x_3235_; lean_object* v___x_3236_; lean_object* v___x_3237_; lean_object* v___x_3238_; lean_object* v___x_3239_; lean_object* v___x_3240_; lean_object* v___x_3241_; lean_object* v___x_3242_; lean_object* v___x_3243_; lean_object* v___x_3244_; lean_object* v___x_3246_; 
v_ref_3232_ = lean_ctor_get(v_a_3220_, 5);
v___x_3233_ = 0;
v___x_3234_ = l_Lean_SourceInfo_fromRef(v_ref_3232_, v___x_3233_);
v___x_3235_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_intros___closed__0));
v___x_3236_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_intros___closed__1));
lean_inc_n(v___x_3234_, 2);
v___x_3237_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3237_, 0, v___x_3234_);
lean_ctor_set(v___x_3237_, 1, v___x_3235_);
v___x_3238_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_3239_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
v___x_3240_ = l_Array_append___redArg(v___x_3239_, v_a_3228_);
lean_dec(v_a_3228_);
v___x_3241_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3241_, 0, v___x_3234_);
lean_ctor_set(v___x_3241_, 1, v___x_3238_);
lean_ctor_set(v___x_3241_, 2, v___x_3240_);
v___x_3242_ = l_Lean_Syntax_node2(v___x_3234_, v___x_3236_, v___x_3237_, v___x_3241_);
v___x_3243_ = lp_aesop_Aesop_withAllTransparencySyntax(v_md_3217_, v___x_3242_);
v___x_3244_ = lp_aesop_Aesop_Script_Tactic_unstructured(v___x_3243_);
if (v_isShared_3231_ == 0)
{
lean_ctor_set(v___x_3230_, 0, v___x_3244_);
v___x_3246_ = v___x_3230_;
goto v_reusejp_3245_;
}
else
{
lean_object* v_reuseFailAlloc_3247_; 
v_reuseFailAlloc_3247_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3247_, 0, v___x_3244_);
v___x_3246_ = v_reuseFailAlloc_3247_;
goto v_reusejp_3245_;
}
v_reusejp_3245_:
{
return v___x_3246_;
}
}
}
else
{
lean_object* v_a_3249_; lean_object* v___x_3251_; uint8_t v_isShared_3252_; uint8_t v_isSharedCheck_3256_; 
v_a_3249_ = lean_ctor_get(v___x_3227_, 0);
v_isSharedCheck_3256_ = !lean_is_exclusive(v___x_3227_);
if (v_isSharedCheck_3256_ == 0)
{
v___x_3251_ = v___x_3227_;
v_isShared_3252_ = v_isSharedCheck_3256_;
goto v_resetjp_3250_;
}
else
{
lean_inc(v_a_3249_);
lean_dec(v___x_3227_);
v___x_3251_ = lean_box(0);
v_isShared_3252_ = v_isSharedCheck_3256_;
goto v_resetjp_3250_;
}
v_resetjp_3250_:
{
lean_object* v___x_3254_; 
if (v_isShared_3252_ == 0)
{
v___x_3254_ = v___x_3251_;
goto v_reusejp_3253_;
}
else
{
lean_object* v_reuseFailAlloc_3255_; 
v_reuseFailAlloc_3255_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3255_, 0, v_a_3249_);
v___x_3254_ = v_reuseFailAlloc_3255_;
goto v_reusejp_3253_;
}
v_reusejp_3253_:
{
return v___x_3254_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_intros___boxed(lean_object* v_postGoal_3257_, lean_object* v_newFVarIds_3258_, lean_object* v_md_3259_, lean_object* v_a_3260_, lean_object* v_a_3261_, lean_object* v_a_3262_, lean_object* v_a_3263_, lean_object* v_a_3264_){
_start:
{
uint8_t v_md_boxed_3265_; lean_object* v_res_3266_; 
v_md_boxed_3265_ = lean_unbox(v_md_3259_);
v_res_3266_ = lp_aesop_Aesop_Script_TacticBuilder_intros(v_postGoal_3257_, v_newFVarIds_3258_, v_md_boxed_3265_, v_a_3260_, v_a_3261_, v_a_3262_, v_a_3263_);
lean_dec(v_a_3263_);
lean_dec_ref(v_a_3262_);
lean_dec(v_a_3261_);
lean_dec_ref(v_a_3260_);
return v_res_3266_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg(lean_object* v_a_3273_){
_start:
{
lean_object* v_ref_3275_; uint8_t v___x_3276_; lean_object* v___x_3277_; lean_object* v___x_3278_; lean_object* v___x_3279_; lean_object* v___x_3280_; lean_object* v___x_3281_; lean_object* v___x_3282_; lean_object* v___x_3283_; lean_object* v___x_3284_; lean_object* v___x_3285_; lean_object* v___x_3286_; 
v_ref_3275_ = lean_ctor_get(v_a_3273_, 5);
v___x_3276_ = 0;
v___x_3277_ = l_Lean_SourceInfo_fromRef(v_ref_3275_, v___x_3276_);
v___x_3278_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg___closed__0));
v___x_3279_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg___closed__1));
lean_inc_n(v___x_3277_, 2);
v___x_3280_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3280_, 0, v___x_3277_);
lean_ctor_set(v___x_3280_, 1, v___x_3278_);
v___x_3281_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_3282_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
v___x_3283_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3283_, 0, v___x_3277_);
lean_ctor_set(v___x_3283_, 1, v___x_3281_);
lean_ctor_set(v___x_3283_, 2, v___x_3282_);
lean_inc_ref(v___x_3283_);
v___x_3284_ = l_Lean_Syntax_node3(v___x_3277_, v___x_3279_, v___x_3280_, v___x_3283_, v___x_3283_);
v___x_3285_ = lp_aesop_Aesop_Script_Tactic_unstructured(v___x_3284_);
v___x_3286_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3286_, 0, v___x_3285_);
return v___x_3286_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg___boxed(lean_object* v_a_3287_, lean_object* v_a_3288_){
_start:
{
lean_object* v_res_3289_; 
v_res_3289_ = lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg(v_a_3287_);
lean_dec_ref(v_a_3287_);
return v_res_3289_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_splitTarget(lean_object* v_a_3290_, lean_object* v_a_3291_, lean_object* v_a_3292_, lean_object* v_a_3293_){
_start:
{
lean_object* v___x_3295_; 
v___x_3295_ = lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg(v_a_3292_);
return v___x_3295_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_splitTarget___boxed(lean_object* v_a_3296_, lean_object* v_a_3297_, lean_object* v_a_3298_, lean_object* v_a_3299_, lean_object* v_a_3300_){
_start:
{
lean_object* v_res_3301_; 
v_res_3301_ = lp_aesop_Aesop_Script_TacticBuilder_splitTarget(v_a_3296_, v_a_3297_, v_a_3298_, v_a_3299_);
lean_dec(v_a_3299_);
lean_dec_ref(v_a_3298_);
lean_dec(v_a_3297_);
lean_dec_ref(v_a_3296_);
return v_res_3301_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_splitAt(lean_object* v_goal_3302_, lean_object* v_fvarId_3303_, lean_object* v_a_3304_, lean_object* v_a_3305_, lean_object* v_a_3306_, lean_object* v_a_3307_){
_start:
{
lean_object* v___x_3309_; lean_object* v___x_3310_; 
v___x_3309_ = lean_alloc_closure((void*)(l_Lean_FVarId_getUserName___boxed), 6, 1);
lean_closure_set(v___x_3309_, 0, v_fvarId_3303_);
v___x_3310_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___redArg(v_goal_3302_, v___x_3309_, v_a_3304_, v_a_3305_, v_a_3306_, v_a_3307_);
if (lean_obj_tag(v___x_3310_) == 0)
{
lean_object* v_a_3311_; lean_object* v___x_3313_; uint8_t v_isShared_3314_; uint8_t v_isSharedCheck_3338_; 
v_a_3311_ = lean_ctor_get(v___x_3310_, 0);
v_isSharedCheck_3338_ = !lean_is_exclusive(v___x_3310_);
if (v_isSharedCheck_3338_ == 0)
{
v___x_3313_ = v___x_3310_;
v_isShared_3314_ = v_isSharedCheck_3338_;
goto v_resetjp_3312_;
}
else
{
lean_inc(v_a_3311_);
lean_dec(v___x_3310_);
v___x_3313_ = lean_box(0);
v_isShared_3314_ = v_isSharedCheck_3338_;
goto v_resetjp_3312_;
}
v_resetjp_3312_:
{
lean_object* v_ref_3315_; uint8_t v___x_3316_; lean_object* v___x_3317_; lean_object* v___x_3318_; lean_object* v___x_3319_; lean_object* v___x_3320_; lean_object* v___x_3321_; lean_object* v___x_3322_; lean_object* v___x_3323_; lean_object* v___x_3324_; lean_object* v___x_3325_; lean_object* v___x_3326_; lean_object* v___x_3327_; lean_object* v___x_3328_; lean_object* v___x_3329_; lean_object* v___x_3330_; lean_object* v___x_3331_; lean_object* v___x_3332_; lean_object* v___x_3333_; lean_object* v___x_3334_; lean_object* v___x_3336_; 
v_ref_3315_ = lean_ctor_get(v_a_3306_, 5);
v___x_3316_ = 0;
v___x_3317_ = l_Lean_SourceInfo_fromRef(v_ref_3315_, v___x_3316_);
v___x_3318_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg___closed__0));
v___x_3319_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg___closed__1));
lean_inc_n(v___x_3317_, 7);
v___x_3320_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3320_, 0, v___x_3317_);
lean_ctor_set(v___x_3320_, 1, v___x_3318_);
v___x_3321_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_3322_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
v___x_3323_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3323_, 0, v___x_3317_);
lean_ctor_set(v___x_3323_, 1, v___x_3321_);
lean_ctor_set(v___x_3323_, 2, v___x_3322_);
v___x_3324_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__1));
v___x_3325_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__2));
v___x_3326_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3326_, 0, v___x_3317_);
lean_ctor_set(v___x_3326_, 1, v___x_3325_);
v___x_3327_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___lam__0___closed__4));
v___x_3328_ = l_Lean_mkIdent(v_a_3311_);
v___x_3329_ = l_Lean_Syntax_node1(v___x_3317_, v___x_3321_, v___x_3328_);
v___x_3330_ = l_Lean_Syntax_node1(v___x_3317_, v___x_3327_, v___x_3329_);
v___x_3331_ = l_Lean_Syntax_node2(v___x_3317_, v___x_3324_, v___x_3326_, v___x_3330_);
v___x_3332_ = l_Lean_Syntax_node1(v___x_3317_, v___x_3321_, v___x_3331_);
v___x_3333_ = l_Lean_Syntax_node3(v___x_3317_, v___x_3319_, v___x_3320_, v___x_3323_, v___x_3332_);
v___x_3334_ = lp_aesop_Aesop_Script_Tactic_unstructured(v___x_3333_);
if (v_isShared_3314_ == 0)
{
lean_ctor_set(v___x_3313_, 0, v___x_3334_);
v___x_3336_ = v___x_3313_;
goto v_reusejp_3335_;
}
else
{
lean_object* v_reuseFailAlloc_3337_; 
v_reuseFailAlloc_3337_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3337_, 0, v___x_3334_);
v___x_3336_ = v_reuseFailAlloc_3337_;
goto v_reusejp_3335_;
}
v_reusejp_3335_:
{
return v___x_3336_;
}
}
}
else
{
lean_object* v_a_3339_; lean_object* v___x_3341_; uint8_t v_isShared_3342_; uint8_t v_isSharedCheck_3346_; 
v_a_3339_ = lean_ctor_get(v___x_3310_, 0);
v_isSharedCheck_3346_ = !lean_is_exclusive(v___x_3310_);
if (v_isSharedCheck_3346_ == 0)
{
v___x_3341_ = v___x_3310_;
v_isShared_3342_ = v_isSharedCheck_3346_;
goto v_resetjp_3340_;
}
else
{
lean_inc(v_a_3339_);
lean_dec(v___x_3310_);
v___x_3341_ = lean_box(0);
v_isShared_3342_ = v_isSharedCheck_3346_;
goto v_resetjp_3340_;
}
v_resetjp_3340_:
{
lean_object* v___x_3344_; 
if (v_isShared_3342_ == 0)
{
v___x_3344_ = v___x_3341_;
goto v_reusejp_3343_;
}
else
{
lean_object* v_reuseFailAlloc_3345_; 
v_reuseFailAlloc_3345_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3345_, 0, v_a_3339_);
v___x_3344_ = v_reuseFailAlloc_3345_;
goto v_reusejp_3343_;
}
v_reusejp_3343_:
{
return v___x_3344_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_splitAt___boxed(lean_object* v_goal_3347_, lean_object* v_fvarId_3348_, lean_object* v_a_3349_, lean_object* v_a_3350_, lean_object* v_a_3351_, lean_object* v_a_3352_, lean_object* v_a_3353_){
_start:
{
lean_object* v_res_3354_; 
v_res_3354_ = lp_aesop_Aesop_Script_TacticBuilder_splitAt(v_goal_3347_, v_fvarId_3348_, v_a_3349_, v_a_3350_, v_a_3351_, v_a_3352_);
lean_dec(v_a_3352_);
lean_dec_ref(v_a_3351_);
lean_dec(v_a_3350_);
lean_dec_ref(v_a_3349_);
return v_res_3354_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_substFVars(lean_object* v_goal_3361_, lean_object* v_fvarIds_3362_, lean_object* v_a_3363_, lean_object* v_a_3364_, lean_object* v_a_3365_, lean_object* v_a_3366_){
_start:
{
size_t v_sz_3368_; lean_object* v___x_3369_; lean_object* v___x_3370_; lean_object* v___x_3371_; lean_object* v___x_3372_; 
v_sz_3368_ = lean_array_size(v_fvarIds_3362_);
v___x_3369_ = lean_box_usize(v_sz_3368_);
v___x_3370_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Script_TacticBuilder_extN_spec__1___boxed__const__1));
v___x_3371_ = lean_alloc_closure((void*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_clear_spec__0___boxed), 8, 3);
lean_closure_set(v___x_3371_, 0, v___x_3369_);
lean_closure_set(v___x_3371_, 1, v___x_3370_);
lean_closure_set(v___x_3371_, 2, v_fvarIds_3362_);
v___x_3372_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_Script_TacticBuilder_apply_spec__0___redArg(v_goal_3361_, v___x_3371_, v_a_3363_, v_a_3364_, v_a_3365_, v_a_3366_);
if (lean_obj_tag(v___x_3372_) == 0)
{
lean_object* v_a_3373_; lean_object* v___x_3375_; uint8_t v_isShared_3376_; uint8_t v_isSharedCheck_3392_; 
v_a_3373_ = lean_ctor_get(v___x_3372_, 0);
v_isSharedCheck_3392_ = !lean_is_exclusive(v___x_3372_);
if (v_isSharedCheck_3392_ == 0)
{
v___x_3375_ = v___x_3372_;
v_isShared_3376_ = v_isSharedCheck_3392_;
goto v_resetjp_3374_;
}
else
{
lean_inc(v_a_3373_);
lean_dec(v___x_3372_);
v___x_3375_ = lean_box(0);
v_isShared_3376_ = v_isSharedCheck_3392_;
goto v_resetjp_3374_;
}
v_resetjp_3374_:
{
lean_object* v_ref_3377_; uint8_t v___x_3378_; lean_object* v___x_3379_; lean_object* v___x_3380_; lean_object* v___x_3381_; lean_object* v___x_3382_; lean_object* v___x_3383_; lean_object* v___x_3384_; lean_object* v___x_3385_; lean_object* v___x_3386_; lean_object* v___x_3387_; lean_object* v___x_3388_; lean_object* v___x_3390_; 
v_ref_3377_ = lean_ctor_get(v_a_3365_, 5);
v___x_3378_ = 0;
v___x_3379_ = l_Lean_SourceInfo_fromRef(v_ref_3377_, v___x_3378_);
v___x_3380_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_substFVars___closed__0));
v___x_3381_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_substFVars___closed__1));
lean_inc_n(v___x_3379_, 2);
v___x_3382_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3382_, 0, v___x_3379_);
lean_ctor_set(v___x_3382_, 1, v___x_3380_);
v___x_3383_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_3384_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
v___x_3385_ = l_Array_append___redArg(v___x_3384_, v_a_3373_);
lean_dec(v_a_3373_);
v___x_3386_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3386_, 0, v___x_3379_);
lean_ctor_set(v___x_3386_, 1, v___x_3383_);
lean_ctor_set(v___x_3386_, 2, v___x_3385_);
v___x_3387_ = l_Lean_Syntax_node2(v___x_3379_, v___x_3381_, v___x_3382_, v___x_3386_);
v___x_3388_ = lp_aesop_Aesop_Script_Tactic_unstructured(v___x_3387_);
if (v_isShared_3376_ == 0)
{
lean_ctor_set(v___x_3375_, 0, v___x_3388_);
v___x_3390_ = v___x_3375_;
goto v_reusejp_3389_;
}
else
{
lean_object* v_reuseFailAlloc_3391_; 
v_reuseFailAlloc_3391_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3391_, 0, v___x_3388_);
v___x_3390_ = v_reuseFailAlloc_3391_;
goto v_reusejp_3389_;
}
v_reusejp_3389_:
{
return v___x_3390_;
}
}
}
else
{
lean_object* v_a_3393_; lean_object* v___x_3395_; uint8_t v_isShared_3396_; uint8_t v_isSharedCheck_3400_; 
v_a_3393_ = lean_ctor_get(v___x_3372_, 0);
v_isSharedCheck_3400_ = !lean_is_exclusive(v___x_3372_);
if (v_isSharedCheck_3400_ == 0)
{
v___x_3395_ = v___x_3372_;
v_isShared_3396_ = v_isSharedCheck_3400_;
goto v_resetjp_3394_;
}
else
{
lean_inc(v_a_3393_);
lean_dec(v___x_3372_);
v___x_3395_ = lean_box(0);
v_isShared_3396_ = v_isSharedCheck_3400_;
goto v_resetjp_3394_;
}
v_resetjp_3394_:
{
lean_object* v___x_3398_; 
if (v_isShared_3396_ == 0)
{
v___x_3398_ = v___x_3395_;
goto v_reusejp_3397_;
}
else
{
lean_object* v_reuseFailAlloc_3399_; 
v_reuseFailAlloc_3399_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3399_, 0, v_a_3393_);
v___x_3398_ = v_reuseFailAlloc_3399_;
goto v_reusejp_3397_;
}
v_reusejp_3397_:
{
return v___x_3398_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_substFVars___boxed(lean_object* v_goal_3401_, lean_object* v_fvarIds_3402_, lean_object* v_a_3403_, lean_object* v_a_3404_, lean_object* v_a_3405_, lean_object* v_a_3406_, lean_object* v_a_3407_){
_start:
{
lean_object* v_res_3408_; 
v_res_3408_ = lp_aesop_Aesop_Script_TacticBuilder_substFVars(v_goal_3401_, v_fvarIds_3402_, v_a_3403_, v_a_3404_, v_a_3405_, v_a_3406_);
lean_dec(v_a_3406_);
lean_dec_ref(v_a_3405_);
lean_dec(v_a_3404_);
lean_dec_ref(v_a_3403_);
return v_res_3408_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_substFVars_x27___redArg(lean_object* v_fvarUserNames_3409_, lean_object* v_a_3410_){
_start:
{
lean_object* v_ref_3412_; size_t v_sz_3413_; size_t v___x_3414_; lean_object* v_fvarUserNames_3415_; uint8_t v___x_3416_; lean_object* v___x_3417_; lean_object* v___x_3418_; lean_object* v___x_3419_; lean_object* v___x_3420_; lean_object* v___x_3421_; lean_object* v___x_3422_; lean_object* v___x_3423_; lean_object* v___x_3424_; lean_object* v___x_3425_; lean_object* v___x_3426_; lean_object* v___x_3427_; 
v_ref_3412_ = lean_ctor_get(v_a_3410_, 5);
v_sz_3413_ = lean_array_size(v_fvarUserNames_3409_);
v___x_3414_ = ((size_t)0ULL);
v_fvarUserNames_3415_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_Script_TacticBuilder_unfold_spec__0(v_sz_3413_, v___x_3414_, v_fvarUserNames_3409_);
v___x_3416_ = 0;
v___x_3417_ = l_Lean_SourceInfo_fromRef(v_ref_3412_, v___x_3416_);
v___x_3418_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_substFVars___closed__0));
v___x_3419_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_substFVars___closed__1));
lean_inc_n(v___x_3417_, 2);
v___x_3420_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3420_, 0, v___x_3417_);
lean_ctor_set(v___x_3420_, 1, v___x_3418_);
v___x_3421_ = ((lean_object*)(lp_aesop_Aesop_Script_TacticBuilder_replace___closed__10));
v___x_3422_ = lean_obj_once(&lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11, &lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11_once, _init_lp_aesop_Aesop_Script_TacticBuilder_replace___closed__11);
v___x_3423_ = l_Array_append___redArg(v___x_3422_, v_fvarUserNames_3415_);
lean_dec_ref(v_fvarUserNames_3415_);
v___x_3424_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3424_, 0, v___x_3417_);
lean_ctor_set(v___x_3424_, 1, v___x_3421_);
lean_ctor_set(v___x_3424_, 2, v___x_3423_);
v___x_3425_ = l_Lean_Syntax_node2(v___x_3417_, v___x_3419_, v___x_3420_, v___x_3424_);
v___x_3426_ = lp_aesop_Aesop_Script_Tactic_unstructured(v___x_3425_);
v___x_3427_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3427_, 0, v___x_3426_);
return v___x_3427_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_substFVars_x27___redArg___boxed(lean_object* v_fvarUserNames_3428_, lean_object* v_a_3429_, lean_object* v_a_3430_){
_start:
{
lean_object* v_res_3431_; 
v_res_3431_ = lp_aesop_Aesop_Script_TacticBuilder_substFVars_x27___redArg(v_fvarUserNames_3428_, v_a_3429_);
lean_dec_ref(v_a_3429_);
return v_res_3431_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_substFVars_x27(lean_object* v_fvarUserNames_3432_, lean_object* v_a_3433_, lean_object* v_a_3434_, lean_object* v_a_3435_, lean_object* v_a_3436_){
_start:
{
lean_object* v___x_3438_; 
v___x_3438_ = lp_aesop_Aesop_Script_TacticBuilder_substFVars_x27___redArg(v_fvarUserNames_3432_, v_a_3435_);
return v___x_3438_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Script_TacticBuilder_substFVars_x27___boxed(lean_object* v_fvarUserNames_3439_, lean_object* v_a_3440_, lean_object* v_a_3441_, lean_object* v_a_3442_, lean_object* v_a_3443_, lean_object* v_a_3444_){
_start:
{
lean_object* v_res_3445_; 
v_res_3445_ = lp_aesop_Aesop_Script_TacticBuilder_substFVars_x27(v_fvarUserNames_3439_, v_a_3440_, v_a_3441_, v_a_3442_, v_a_3443_);
lean_dec(v_a_3443_);
lean_dec_ref(v_a_3442_);
lean_dec(v_a_3441_);
lean_dec_ref(v_a_3440_);
return v_res_3445_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_assertHypothesisS_tacticBuilder___redArg(lean_object* v_goal_3446_, lean_object* v_h_3447_, uint8_t v_md_3448_, lean_object* v_a_3449_, lean_object* v_a_3450_, lean_object* v_a_3451_, lean_object* v_a_3452_){
_start:
{
lean_object* v___x_3454_; 
v___x_3454_ = lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis(v_goal_3446_, v_h_3447_, v_md_3448_, v_a_3449_, v_a_3450_, v_a_3451_, v_a_3452_);
return v___x_3454_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_assertHypothesisS_tacticBuilder___redArg___boxed(lean_object* v_goal_3455_, lean_object* v_h_3456_, lean_object* v_md_3457_, lean_object* v_a_3458_, lean_object* v_a_3459_, lean_object* v_a_3460_, lean_object* v_a_3461_, lean_object* v_a_3462_){
_start:
{
uint8_t v_md_boxed_3463_; lean_object* v_res_3464_; 
v_md_boxed_3463_ = lean_unbox(v_md_3457_);
v_res_3464_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_assertHypothesisS_tacticBuilder___redArg(v_goal_3455_, v_h_3456_, v_md_boxed_3463_, v_a_3458_, v_a_3459_, v_a_3460_, v_a_3461_);
lean_dec(v_a_3461_);
lean_dec_ref(v_a_3460_);
lean_dec(v_a_3459_);
lean_dec_ref(v_a_3458_);
return v_res_3464_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_assertHypothesisS_tacticBuilder(lean_object* v_goal_3465_, lean_object* v_h_3466_, uint8_t v_md_3467_, lean_object* v_x_3468_, lean_object* v_a_3469_, lean_object* v_a_3470_, lean_object* v_a_3471_, lean_object* v_a_3472_){
_start:
{
lean_object* v___x_3474_; 
v___x_3474_ = lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis(v_goal_3465_, v_h_3466_, v_md_3467_, v_a_3469_, v_a_3470_, v_a_3471_, v_a_3472_);
return v___x_3474_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_assertHypothesisS_tacticBuilder___boxed(lean_object* v_goal_3475_, lean_object* v_h_3476_, lean_object* v_md_3477_, lean_object* v_x_3478_, lean_object* v_a_3479_, lean_object* v_a_3480_, lean_object* v_a_3481_, lean_object* v_a_3482_, lean_object* v_a_3483_){
_start:
{
uint8_t v_md_boxed_3484_; lean_object* v_res_3485_; 
v_md_boxed_3484_ = lean_unbox(v_md_3477_);
v_res_3485_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_assertHypothesisS_tacticBuilder(v_goal_3475_, v_h_3476_, v_md_boxed_3484_, v_x_3478_, v_a_3479_, v_a_3480_, v_a_3481_, v_a_3482_);
lean_dec(v_a_3482_);
lean_dec_ref(v_a_3481_);
lean_dec(v_a_3480_);
lean_dec_ref(v_a_3479_);
lean_dec_ref(v_x_3478_);
return v_res_3485_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_assertHypothesisS___lam__0(lean_object* v_goal_3486_, lean_object* v_h_3487_, uint8_t v_md_3488_, lean_object* v___y_3489_, lean_object* v___y_3490_, lean_object* v___y_3491_, lean_object* v___y_3492_, lean_object* v___y_3493_){
_start:
{
lean_object* v___x_3495_; 
v___x_3495_ = lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis(v_goal_3486_, v_h_3487_, v_md_3488_, v___y_3490_, v___y_3491_, v___y_3492_, v___y_3493_);
return v___x_3495_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_assertHypothesisS___lam__0___boxed(lean_object* v_goal_3496_, lean_object* v_h_3497_, lean_object* v_md_3498_, lean_object* v___y_3499_, lean_object* v___y_3500_, lean_object* v___y_3501_, lean_object* v___y_3502_, lean_object* v___y_3503_, lean_object* v___y_3504_){
_start:
{
uint8_t v_md_boxed_3505_; lean_object* v_res_3506_; 
v_md_boxed_3505_ = lean_unbox(v_md_3498_);
v_res_3506_ = lp_aesop_Aesop_assertHypothesisS___lam__0(v_goal_3496_, v_h_3497_, v_md_boxed_3505_, v___y_3499_, v___y_3500_, v___y_3501_, v___y_3502_, v___y_3503_);
lean_dec(v___y_3503_);
lean_dec_ref(v___y_3502_);
lean_dec(v___y_3501_);
lean_dec_ref(v___y_3500_);
lean_dec_ref(v___y_3499_);
return v_res_3506_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_assertHypothesisS___lam__1(lean_object* v_x_3507_){
_start:
{
lean_object* v_fst_3508_; lean_object* v___x_3509_; lean_object* v___x_3510_; lean_object* v___x_3511_; 
v_fst_3508_ = lean_ctor_get(v_x_3507_, 0);
lean_inc(v_fst_3508_);
lean_dec_ref(v_x_3507_);
v___x_3509_ = lean_unsigned_to_nat(1u);
v___x_3510_ = lean_mk_empty_array_with_capacity(v___x_3509_);
v___x_3511_ = lean_array_push(v___x_3510_, v_fst_3508_);
return v___x_3511_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_assertHypothesisS___lam__2(lean_object* v_x_3512_){
_start:
{
uint8_t v___x_3513_; 
v___x_3513_ = 1;
return v___x_3513_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_assertHypothesisS___lam__2___boxed(lean_object* v_x_3514_){
_start:
{
uint8_t v_res_3515_; lean_object* v_r_3516_; 
v_res_3515_ = lp_aesop_Aesop_assertHypothesisS___lam__2(v_x_3514_);
lean_dec_ref(v_x_3514_);
v_r_3516_ = lean_box(v_res_3515_);
return v_r_3516_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_assertHypothesisS___lam__3(uint8_t v_md_3517_, lean_object* v_goal_3518_, lean_object* v___x_3519_, lean_object* v___y_3520_, lean_object* v___y_3521_, lean_object* v___y_3522_, lean_object* v___y_3523_){
_start:
{
lean_object* v_keyedConfig_3525_; uint8_t v_trackZetaDelta_3526_; lean_object* v_zetaDeltaSet_3527_; lean_object* v_lctx_3528_; lean_object* v_localInstances_3529_; lean_object* v_defEqCtx_x3f_3530_; lean_object* v_synthPendingDepth_3531_; lean_object* v_customCanUnfoldPredicate_x3f_3532_; uint8_t v_univApprox_3533_; uint8_t v_inTypeClassResolution_3534_; uint8_t v_cacheInferType_3535_; lean_object* v___x_3537_; uint8_t v_isShared_3538_; uint8_t v_isSharedCheck_3569_; 
v_keyedConfig_3525_ = lean_ctor_get(v___y_3520_, 0);
v_trackZetaDelta_3526_ = lean_ctor_get_uint8(v___y_3520_, sizeof(void*)*7);
v_zetaDeltaSet_3527_ = lean_ctor_get(v___y_3520_, 1);
v_lctx_3528_ = lean_ctor_get(v___y_3520_, 2);
v_localInstances_3529_ = lean_ctor_get(v___y_3520_, 3);
v_defEqCtx_x3f_3530_ = lean_ctor_get(v___y_3520_, 4);
v_synthPendingDepth_3531_ = lean_ctor_get(v___y_3520_, 5);
v_customCanUnfoldPredicate_x3f_3532_ = lean_ctor_get(v___y_3520_, 6);
v_univApprox_3533_ = lean_ctor_get_uint8(v___y_3520_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_3534_ = lean_ctor_get_uint8(v___y_3520_, sizeof(void*)*7 + 2);
v_cacheInferType_3535_ = lean_ctor_get_uint8(v___y_3520_, sizeof(void*)*7 + 3);
v_isSharedCheck_3569_ = !lean_is_exclusive(v___y_3520_);
if (v_isSharedCheck_3569_ == 0)
{
v___x_3537_ = v___y_3520_;
v_isShared_3538_ = v_isSharedCheck_3569_;
goto v_resetjp_3536_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_3532_);
lean_inc(v_synthPendingDepth_3531_);
lean_inc(v_defEqCtx_x3f_3530_);
lean_inc(v_localInstances_3529_);
lean_inc(v_lctx_3528_);
lean_inc(v_zetaDeltaSet_3527_);
lean_inc(v_keyedConfig_3525_);
lean_dec(v___y_3520_);
v___x_3537_ = lean_box(0);
v_isShared_3538_ = v_isSharedCheck_3569_;
goto v_resetjp_3536_;
}
v_resetjp_3536_:
{
lean_object* v___x_3539_; lean_object* v___x_3541_; 
v___x_3539_ = l_Lean_Meta_ConfigWithKey_setTransparency(v_md_3517_, v_keyedConfig_3525_);
if (v_isShared_3538_ == 0)
{
lean_ctor_set(v___x_3537_, 0, v___x_3539_);
v___x_3541_ = v___x_3537_;
goto v_reusejp_3540_;
}
else
{
lean_object* v_reuseFailAlloc_3568_; 
v_reuseFailAlloc_3568_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_3568_, 0, v___x_3539_);
lean_ctor_set(v_reuseFailAlloc_3568_, 1, v_zetaDeltaSet_3527_);
lean_ctor_set(v_reuseFailAlloc_3568_, 2, v_lctx_3528_);
lean_ctor_set(v_reuseFailAlloc_3568_, 3, v_localInstances_3529_);
lean_ctor_set(v_reuseFailAlloc_3568_, 4, v_defEqCtx_x3f_3530_);
lean_ctor_set(v_reuseFailAlloc_3568_, 5, v_synthPendingDepth_3531_);
lean_ctor_set(v_reuseFailAlloc_3568_, 6, v_customCanUnfoldPredicate_x3f_3532_);
lean_ctor_set_uint8(v_reuseFailAlloc_3568_, sizeof(void*)*7, v_trackZetaDelta_3526_);
lean_ctor_set_uint8(v_reuseFailAlloc_3568_, sizeof(void*)*7 + 1, v_univApprox_3533_);
lean_ctor_set_uint8(v_reuseFailAlloc_3568_, sizeof(void*)*7 + 2, v_inTypeClassResolution_3534_);
lean_ctor_set_uint8(v_reuseFailAlloc_3568_, sizeof(void*)*7 + 3, v_cacheInferType_3535_);
v___x_3541_ = v_reuseFailAlloc_3568_;
goto v_reusejp_3540_;
}
v_reusejp_3540_:
{
lean_object* v___x_3542_; 
v___x_3542_ = l_Lean_MVarId_assertHypotheses(v_goal_3518_, v___x_3519_, v___x_3541_, v___y_3521_, v___y_3522_, v___y_3523_);
lean_dec_ref(v___x_3541_);
if (lean_obj_tag(v___x_3542_) == 0)
{
lean_object* v_a_3543_; lean_object* v___x_3545_; uint8_t v_isShared_3546_; uint8_t v_isSharedCheck_3559_; 
v_a_3543_ = lean_ctor_get(v___x_3542_, 0);
v_isSharedCheck_3559_ = !lean_is_exclusive(v___x_3542_);
if (v_isSharedCheck_3559_ == 0)
{
v___x_3545_ = v___x_3542_;
v_isShared_3546_ = v_isSharedCheck_3559_;
goto v_resetjp_3544_;
}
else
{
lean_inc(v_a_3543_);
lean_dec(v___x_3542_);
v___x_3545_ = lean_box(0);
v_isShared_3546_ = v_isSharedCheck_3559_;
goto v_resetjp_3544_;
}
v_resetjp_3544_:
{
lean_object* v_fst_3547_; lean_object* v_snd_3548_; lean_object* v___x_3550_; uint8_t v_isShared_3551_; uint8_t v_isSharedCheck_3558_; 
v_fst_3547_ = lean_ctor_get(v_a_3543_, 0);
v_snd_3548_ = lean_ctor_get(v_a_3543_, 1);
v_isSharedCheck_3558_ = !lean_is_exclusive(v_a_3543_);
if (v_isSharedCheck_3558_ == 0)
{
v___x_3550_ = v_a_3543_;
v_isShared_3551_ = v_isSharedCheck_3558_;
goto v_resetjp_3549_;
}
else
{
lean_inc(v_snd_3548_);
lean_inc(v_fst_3547_);
lean_dec(v_a_3543_);
v___x_3550_ = lean_box(0);
v_isShared_3551_ = v_isSharedCheck_3558_;
goto v_resetjp_3549_;
}
v_resetjp_3549_:
{
lean_object* v___x_3553_; 
if (v_isShared_3551_ == 0)
{
lean_ctor_set(v___x_3550_, 1, v_fst_3547_);
lean_ctor_set(v___x_3550_, 0, v_snd_3548_);
v___x_3553_ = v___x_3550_;
goto v_reusejp_3552_;
}
else
{
lean_object* v_reuseFailAlloc_3557_; 
v_reuseFailAlloc_3557_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3557_, 0, v_snd_3548_);
lean_ctor_set(v_reuseFailAlloc_3557_, 1, v_fst_3547_);
v___x_3553_ = v_reuseFailAlloc_3557_;
goto v_reusejp_3552_;
}
v_reusejp_3552_:
{
lean_object* v___x_3555_; 
if (v_isShared_3546_ == 0)
{
lean_ctor_set(v___x_3545_, 0, v___x_3553_);
v___x_3555_ = v___x_3545_;
goto v_reusejp_3554_;
}
else
{
lean_object* v_reuseFailAlloc_3556_; 
v_reuseFailAlloc_3556_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3556_, 0, v___x_3553_);
v___x_3555_ = v_reuseFailAlloc_3556_;
goto v_reusejp_3554_;
}
v_reusejp_3554_:
{
return v___x_3555_;
}
}
}
}
}
else
{
lean_object* v_a_3560_; lean_object* v___x_3562_; uint8_t v_isShared_3563_; uint8_t v_isSharedCheck_3567_; 
v_a_3560_ = lean_ctor_get(v___x_3542_, 0);
v_isSharedCheck_3567_ = !lean_is_exclusive(v___x_3542_);
if (v_isSharedCheck_3567_ == 0)
{
v___x_3562_ = v___x_3542_;
v_isShared_3563_ = v_isSharedCheck_3567_;
goto v_resetjp_3561_;
}
else
{
lean_inc(v_a_3560_);
lean_dec(v___x_3542_);
v___x_3562_ = lean_box(0);
v_isShared_3563_ = v_isSharedCheck_3567_;
goto v_resetjp_3561_;
}
v_resetjp_3561_:
{
lean_object* v___x_3565_; 
if (v_isShared_3563_ == 0)
{
v___x_3565_ = v___x_3562_;
goto v_reusejp_3564_;
}
else
{
lean_object* v_reuseFailAlloc_3566_; 
v_reuseFailAlloc_3566_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3566_, 0, v_a_3560_);
v___x_3565_ = v_reuseFailAlloc_3566_;
goto v_reusejp_3564_;
}
v_reusejp_3564_:
{
return v___x_3565_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_assertHypothesisS___lam__3___boxed(lean_object* v_md_3570_, lean_object* v_goal_3571_, lean_object* v___x_3572_, lean_object* v___y_3573_, lean_object* v___y_3574_, lean_object* v___y_3575_, lean_object* v___y_3576_, lean_object* v___y_3577_){
_start:
{
uint8_t v_md_boxed_3578_; lean_object* v_res_3579_; 
v_md_boxed_3578_ = lean_unbox(v_md_3570_);
v_res_3579_ = lp_aesop_Aesop_assertHypothesisS___lam__3(v_md_boxed_3578_, v_goal_3571_, v___x_3572_, v___y_3573_, v___y_3574_, v___y_3575_, v___y_3576_);
lean_dec(v___y_3576_);
lean_dec_ref(v___y_3575_);
lean_dec(v___y_3574_);
return v_res_3579_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_assertHypothesisS(lean_object* v_goal_3582_, lean_object* v_h_3583_, uint8_t v_md_3584_, lean_object* v_a_3585_, lean_object* v_a_3586_, lean_object* v_a_3587_, lean_object* v_a_3588_, lean_object* v_a_3589_, lean_object* v_a_3590_){
_start:
{
lean_object* v___x_3592_; lean_object* v___f_3593_; lean_object* v___f_3594_; lean_object* v___f_3595_; lean_object* v___x_3596_; lean_object* v___x_3597_; lean_object* v___x_3598_; lean_object* v___x_3599_; lean_object* v___f_3600_; lean_object* v___x_3601_; 
v___x_3592_ = lean_box(v_md_3584_);
lean_inc_ref(v_h_3583_);
lean_inc_n(v_goal_3582_, 2);
v___f_3593_ = lean_alloc_closure((void*)(lp_aesop_Aesop_assertHypothesisS___lam__0___boxed), 9, 3);
lean_closure_set(v___f_3593_, 0, v_goal_3582_);
lean_closure_set(v___f_3593_, 1, v_h_3583_);
lean_closure_set(v___f_3593_, 2, v___x_3592_);
v___f_3594_ = ((lean_object*)(lp_aesop_Aesop_assertHypothesisS___closed__0));
v___f_3595_ = ((lean_object*)(lp_aesop_Aesop_assertHypothesisS___closed__1));
v___x_3596_ = lean_unsigned_to_nat(1u);
v___x_3597_ = lean_mk_empty_array_with_capacity(v___x_3596_);
v___x_3598_ = lean_array_push(v___x_3597_, v_h_3583_);
v___x_3599_ = lean_box(v_md_3584_);
v___f_3600_ = lean_alloc_closure((void*)(lp_aesop_Aesop_assertHypothesisS___lam__3___boxed), 8, 3);
lean_closure_set(v___f_3600_, 0, v___x_3599_);
lean_closure_set(v___f_3600_, 1, v_goal_3582_);
lean_closure_set(v___f_3600_, 2, v___x_3598_);
v___x_3601_ = lp_aesop_Aesop_withScriptStep___redArg(v_goal_3582_, v___f_3594_, v___f_3595_, v___f_3593_, v___f_3600_, v_a_3585_, v_a_3586_, v_a_3587_, v_a_3588_, v_a_3589_, v_a_3590_);
return v___x_3601_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_assertHypothesisS___boxed(lean_object* v_goal_3602_, lean_object* v_h_3603_, lean_object* v_md_3604_, lean_object* v_a_3605_, lean_object* v_a_3606_, lean_object* v_a_3607_, lean_object* v_a_3608_, lean_object* v_a_3609_, lean_object* v_a_3610_, lean_object* v_a_3611_){
_start:
{
uint8_t v_md_boxed_3612_; lean_object* v_res_3613_; 
v_md_boxed_3612_ = lean_unbox(v_md_3604_);
v_res_3613_ = lp_aesop_Aesop_assertHypothesisS(v_goal_3602_, v_h_3603_, v_md_boxed_3612_, v_a_3605_, v_a_3606_, v_a_3607_, v_a_3608_, v_a_3609_, v_a_3610_);
lean_dec(v_a_3610_);
lean_dec_ref(v_a_3609_);
lean_dec(v_a_3608_);
lean_dec_ref(v_a_3607_);
lean_dec(v_a_3606_);
lean_dec(v_a_3605_);
return v_res_3613_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_applyS_tacticBuilder___redArg(lean_object* v_goal_3614_, lean_object* v_e_3615_, lean_object* v_eStx_x3f_3616_, uint8_t v_md_3617_, lean_object* v_a_3618_, lean_object* v_a_3619_, lean_object* v_a_3620_, lean_object* v_a_3621_){
_start:
{
if (lean_obj_tag(v_eStx_x3f_3616_) == 0)
{
lean_object* v___x_3623_; 
v___x_3623_ = lp_aesop_Aesop_Script_TacticBuilder_apply(v_goal_3614_, v_e_3615_, v_md_3617_, v_a_3618_, v_a_3619_, v_a_3620_, v_a_3621_);
return v___x_3623_;
}
else
{
lean_object* v_val_3624_; lean_object* v___x_3625_; 
lean_dec_ref(v_e_3615_);
lean_dec(v_goal_3614_);
v_val_3624_ = lean_ctor_get(v_eStx_x3f_3616_, 0);
lean_inc(v_val_3624_);
lean_dec_ref_known(v_eStx_x3f_3616_, 1);
v___x_3625_ = lp_aesop_Aesop_Script_TacticBuilder_applyStx___redArg(v_val_3624_, v_md_3617_, v_a_3620_);
return v___x_3625_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_applyS_tacticBuilder___redArg___boxed(lean_object* v_goal_3626_, lean_object* v_e_3627_, lean_object* v_eStx_x3f_3628_, lean_object* v_md_3629_, lean_object* v_a_3630_, lean_object* v_a_3631_, lean_object* v_a_3632_, lean_object* v_a_3633_, lean_object* v_a_3634_){
_start:
{
uint8_t v_md_boxed_3635_; lean_object* v_res_3636_; 
v_md_boxed_3635_ = lean_unbox(v_md_3629_);
v_res_3636_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_applyS_tacticBuilder___redArg(v_goal_3626_, v_e_3627_, v_eStx_x3f_3628_, v_md_boxed_3635_, v_a_3630_, v_a_3631_, v_a_3632_, v_a_3633_);
lean_dec(v_a_3633_);
lean_dec_ref(v_a_3632_);
lean_dec(v_a_3631_);
lean_dec_ref(v_a_3630_);
return v_res_3636_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_applyS_tacticBuilder(lean_object* v_goal_3637_, lean_object* v_e_3638_, lean_object* v_eStx_x3f_3639_, uint8_t v_md_3640_, lean_object* v_x_3641_, lean_object* v_a_3642_, lean_object* v_a_3643_, lean_object* v_a_3644_, lean_object* v_a_3645_){
_start:
{
lean_object* v___x_3647_; 
v___x_3647_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_applyS_tacticBuilder___redArg(v_goal_3637_, v_e_3638_, v_eStx_x3f_3639_, v_md_3640_, v_a_3642_, v_a_3643_, v_a_3644_, v_a_3645_);
return v___x_3647_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_applyS_tacticBuilder___boxed(lean_object* v_goal_3648_, lean_object* v_e_3649_, lean_object* v_eStx_x3f_3650_, lean_object* v_md_3651_, lean_object* v_x_3652_, lean_object* v_a_3653_, lean_object* v_a_3654_, lean_object* v_a_3655_, lean_object* v_a_3656_, lean_object* v_a_3657_){
_start:
{
uint8_t v_md_boxed_3658_; lean_object* v_res_3659_; 
v_md_boxed_3658_ = lean_unbox(v_md_3651_);
v_res_3659_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_applyS_tacticBuilder(v_goal_3648_, v_e_3649_, v_eStx_x3f_3650_, v_md_boxed_3658_, v_x_3652_, v_a_3653_, v_a_3654_, v_a_3655_, v_a_3656_);
lean_dec(v_a_3656_);
lean_dec_ref(v_a_3655_);
lean_dec(v_a_3654_);
lean_dec_ref(v_a_3653_);
lean_dec_ref(v_x_3652_);
return v_res_3659_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyS___lam__0(lean_object* v___y_3660_){
_start:
{
lean_inc_ref(v___y_3660_);
return v___y_3660_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyS___lam__0___boxed(lean_object* v___y_3661_){
_start:
{
lean_object* v_res_3662_; 
v_res_3662_ = lp_aesop_Aesop_applyS___lam__0(v___y_3661_);
lean_dec_ref(v___y_3661_);
return v_res_3662_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_applyS___lam__1(lean_object* v_x_3663_){
_start:
{
uint8_t v___x_3664_; 
v___x_3664_ = 1;
return v___x_3664_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyS___lam__1___boxed(lean_object* v_x_3665_){
_start:
{
uint8_t v_res_3666_; lean_object* v_r_3667_; 
v_res_3666_ = lp_aesop_Aesop_applyS___lam__1(v_x_3665_);
lean_dec_ref(v_x_3665_);
v_r_3667_ = lean_box(v_res_3666_);
return v_r_3667_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyS___lam__2(uint8_t v_md_3668_, lean_object* v_goal_3669_, lean_object* v_e_3670_, lean_object* v___x_3671_, lean_object* v___x_3672_, lean_object* v___y_3673_, lean_object* v___y_3674_, lean_object* v___y_3675_, lean_object* v___y_3676_){
_start:
{
lean_object* v_keyedConfig_3678_; uint8_t v_trackZetaDelta_3679_; lean_object* v_zetaDeltaSet_3680_; lean_object* v_lctx_3681_; lean_object* v_localInstances_3682_; lean_object* v_defEqCtx_x3f_3683_; lean_object* v_synthPendingDepth_3684_; lean_object* v_customCanUnfoldPredicate_x3f_3685_; uint8_t v_univApprox_3686_; uint8_t v_inTypeClassResolution_3687_; uint8_t v_cacheInferType_3688_; lean_object* v___x_3690_; uint8_t v_isShared_3691_; uint8_t v_isSharedCheck_3714_; 
v_keyedConfig_3678_ = lean_ctor_get(v___y_3673_, 0);
v_trackZetaDelta_3679_ = lean_ctor_get_uint8(v___y_3673_, sizeof(void*)*7);
v_zetaDeltaSet_3680_ = lean_ctor_get(v___y_3673_, 1);
v_lctx_3681_ = lean_ctor_get(v___y_3673_, 2);
v_localInstances_3682_ = lean_ctor_get(v___y_3673_, 3);
v_defEqCtx_x3f_3683_ = lean_ctor_get(v___y_3673_, 4);
v_synthPendingDepth_3684_ = lean_ctor_get(v___y_3673_, 5);
v_customCanUnfoldPredicate_x3f_3685_ = lean_ctor_get(v___y_3673_, 6);
v_univApprox_3686_ = lean_ctor_get_uint8(v___y_3673_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_3687_ = lean_ctor_get_uint8(v___y_3673_, sizeof(void*)*7 + 2);
v_cacheInferType_3688_ = lean_ctor_get_uint8(v___y_3673_, sizeof(void*)*7 + 3);
v_isSharedCheck_3714_ = !lean_is_exclusive(v___y_3673_);
if (v_isSharedCheck_3714_ == 0)
{
v___x_3690_ = v___y_3673_;
v_isShared_3691_ = v_isSharedCheck_3714_;
goto v_resetjp_3689_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_3685_);
lean_inc(v_synthPendingDepth_3684_);
lean_inc(v_defEqCtx_x3f_3683_);
lean_inc(v_localInstances_3682_);
lean_inc(v_lctx_3681_);
lean_inc(v_zetaDeltaSet_3680_);
lean_inc(v_keyedConfig_3678_);
lean_dec(v___y_3673_);
v___x_3690_ = lean_box(0);
v_isShared_3691_ = v_isSharedCheck_3714_;
goto v_resetjp_3689_;
}
v_resetjp_3689_:
{
lean_object* v___x_3692_; lean_object* v___x_3694_; 
v___x_3692_ = l_Lean_Meta_ConfigWithKey_setTransparency(v_md_3668_, v_keyedConfig_3678_);
if (v_isShared_3691_ == 0)
{
lean_ctor_set(v___x_3690_, 0, v___x_3692_);
v___x_3694_ = v___x_3690_;
goto v_reusejp_3693_;
}
else
{
lean_object* v_reuseFailAlloc_3713_; 
v_reuseFailAlloc_3713_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_3713_, 0, v___x_3692_);
lean_ctor_set(v_reuseFailAlloc_3713_, 1, v_zetaDeltaSet_3680_);
lean_ctor_set(v_reuseFailAlloc_3713_, 2, v_lctx_3681_);
lean_ctor_set(v_reuseFailAlloc_3713_, 3, v_localInstances_3682_);
lean_ctor_set(v_reuseFailAlloc_3713_, 4, v_defEqCtx_x3f_3683_);
lean_ctor_set(v_reuseFailAlloc_3713_, 5, v_synthPendingDepth_3684_);
lean_ctor_set(v_reuseFailAlloc_3713_, 6, v_customCanUnfoldPredicate_x3f_3685_);
lean_ctor_set_uint8(v_reuseFailAlloc_3713_, sizeof(void*)*7, v_trackZetaDelta_3679_);
lean_ctor_set_uint8(v_reuseFailAlloc_3713_, sizeof(void*)*7 + 1, v_univApprox_3686_);
lean_ctor_set_uint8(v_reuseFailAlloc_3713_, sizeof(void*)*7 + 2, v_inTypeClassResolution_3687_);
lean_ctor_set_uint8(v_reuseFailAlloc_3713_, sizeof(void*)*7 + 3, v_cacheInferType_3688_);
v___x_3694_ = v_reuseFailAlloc_3713_;
goto v_reusejp_3693_;
}
v_reusejp_3693_:
{
lean_object* v___x_3695_; 
v___x_3695_ = l_Lean_MVarId_apply(v_goal_3669_, v_e_3670_, v___x_3671_, v___x_3672_, v___x_3694_, v___y_3674_, v___y_3675_, v___y_3676_);
lean_dec_ref(v___x_3694_);
if (lean_obj_tag(v___x_3695_) == 0)
{
lean_object* v_a_3696_; lean_object* v___x_3698_; uint8_t v_isShared_3699_; uint8_t v_isSharedCheck_3704_; 
v_a_3696_ = lean_ctor_get(v___x_3695_, 0);
v_isSharedCheck_3704_ = !lean_is_exclusive(v___x_3695_);
if (v_isSharedCheck_3704_ == 0)
{
v___x_3698_ = v___x_3695_;
v_isShared_3699_ = v_isSharedCheck_3704_;
goto v_resetjp_3697_;
}
else
{
lean_inc(v_a_3696_);
lean_dec(v___x_3695_);
v___x_3698_ = lean_box(0);
v_isShared_3699_ = v_isSharedCheck_3704_;
goto v_resetjp_3697_;
}
v_resetjp_3697_:
{
lean_object* v___x_3700_; lean_object* v___x_3702_; 
v___x_3700_ = lean_array_mk(v_a_3696_);
if (v_isShared_3699_ == 0)
{
lean_ctor_set(v___x_3698_, 0, v___x_3700_);
v___x_3702_ = v___x_3698_;
goto v_reusejp_3701_;
}
else
{
lean_object* v_reuseFailAlloc_3703_; 
v_reuseFailAlloc_3703_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3703_, 0, v___x_3700_);
v___x_3702_ = v_reuseFailAlloc_3703_;
goto v_reusejp_3701_;
}
v_reusejp_3701_:
{
return v___x_3702_;
}
}
}
else
{
lean_object* v_a_3705_; lean_object* v___x_3707_; uint8_t v_isShared_3708_; uint8_t v_isSharedCheck_3712_; 
v_a_3705_ = lean_ctor_get(v___x_3695_, 0);
v_isSharedCheck_3712_ = !lean_is_exclusive(v___x_3695_);
if (v_isSharedCheck_3712_ == 0)
{
v___x_3707_ = v___x_3695_;
v_isShared_3708_ = v_isSharedCheck_3712_;
goto v_resetjp_3706_;
}
else
{
lean_inc(v_a_3705_);
lean_dec(v___x_3695_);
v___x_3707_ = lean_box(0);
v_isShared_3708_ = v_isSharedCheck_3712_;
goto v_resetjp_3706_;
}
v_resetjp_3706_:
{
lean_object* v___x_3710_; 
if (v_isShared_3708_ == 0)
{
v___x_3710_ = v___x_3707_;
goto v_reusejp_3709_;
}
else
{
lean_object* v_reuseFailAlloc_3711_; 
v_reuseFailAlloc_3711_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3711_, 0, v_a_3705_);
v___x_3710_ = v_reuseFailAlloc_3711_;
goto v_reusejp_3709_;
}
v_reusejp_3709_:
{
return v___x_3710_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyS___lam__2___boxed(lean_object* v_md_3715_, lean_object* v_goal_3716_, lean_object* v_e_3717_, lean_object* v___x_3718_, lean_object* v___x_3719_, lean_object* v___y_3720_, lean_object* v___y_3721_, lean_object* v___y_3722_, lean_object* v___y_3723_, lean_object* v___y_3724_){
_start:
{
uint8_t v_md_boxed_3725_; lean_object* v_res_3726_; 
v_md_boxed_3725_ = lean_unbox(v_md_3715_);
v_res_3726_ = lp_aesop_Aesop_applyS___lam__2(v_md_boxed_3725_, v_goal_3716_, v_e_3717_, v___x_3718_, v___x_3719_, v___y_3720_, v___y_3721_, v___y_3722_, v___y_3723_);
lean_dec(v___y_3723_);
lean_dec_ref(v___y_3722_);
lean_dec(v___y_3721_);
return v_res_3726_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyS(lean_object* v_goal_3733_, lean_object* v_e_3734_, lean_object* v_eStx_x3f_3735_, uint8_t v_md_3736_, lean_object* v_a_3737_, lean_object* v_a_3738_, lean_object* v_a_3739_, lean_object* v_a_3740_, lean_object* v_a_3741_, lean_object* v_a_3742_){
_start:
{
lean_object* v___f_3744_; lean_object* v___f_3745_; lean_object* v___x_3746_; lean_object* v___x_3747_; lean_object* v___x_3748_; lean_object* v___x_3749_; lean_object* v___x_3750_; lean_object* v___f_3751_; lean_object* v___x_3752_; 
v___f_3744_ = ((lean_object*)(lp_aesop_Aesop_applyS___closed__0));
v___f_3745_ = ((lean_object*)(lp_aesop_Aesop_applyS___closed__1));
v___x_3746_ = lean_box(v_md_3736_);
lean_inc_ref(v_e_3734_);
lean_inc_n(v_goal_3733_, 2);
v___x_3747_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_applyS_tacticBuilder___boxed), 10, 4);
lean_closure_set(v___x_3747_, 0, v_goal_3733_);
lean_closure_set(v___x_3747_, 1, v_e_3734_);
lean_closure_set(v___x_3747_, 2, v_eStx_x3f_3735_);
lean_closure_set(v___x_3747_, 3, v___x_3746_);
v___x_3748_ = ((lean_object*)(lp_aesop_Aesop_applyS___closed__2));
v___x_3749_ = lean_box(0);
v___x_3750_ = lean_box(v_md_3736_);
v___f_3751_ = lean_alloc_closure((void*)(lp_aesop_Aesop_applyS___lam__2___boxed), 10, 5);
lean_closure_set(v___f_3751_, 0, v___x_3750_);
lean_closure_set(v___f_3751_, 1, v_goal_3733_);
lean_closure_set(v___f_3751_, 2, v_e_3734_);
lean_closure_set(v___f_3751_, 3, v___x_3748_);
lean_closure_set(v___f_3751_, 4, v___x_3749_);
v___x_3752_ = lp_aesop_Aesop_withScriptStep___redArg(v_goal_3733_, v___f_3744_, v___f_3745_, v___x_3747_, v___f_3751_, v_a_3737_, v_a_3738_, v_a_3739_, v_a_3740_, v_a_3741_, v_a_3742_);
return v___x_3752_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_applyS___boxed(lean_object* v_goal_3753_, lean_object* v_e_3754_, lean_object* v_eStx_x3f_3755_, lean_object* v_md_3756_, lean_object* v_a_3757_, lean_object* v_a_3758_, lean_object* v_a_3759_, lean_object* v_a_3760_, lean_object* v_a_3761_, lean_object* v_a_3762_, lean_object* v_a_3763_){
_start:
{
uint8_t v_md_boxed_3764_; lean_object* v_res_3765_; 
v_md_boxed_3764_ = lean_unbox(v_md_3756_);
v_res_3765_ = lp_aesop_Aesop_applyS(v_goal_3753_, v_e_3754_, v_eStx_x3f_3755_, v_md_boxed_3764_, v_a_3757_, v_a_3758_, v_a_3759_, v_a_3760_, v_a_3761_, v_a_3762_);
lean_dec(v_a_3762_);
lean_dec_ref(v_a_3761_);
lean_dec(v_a_3760_);
lean_dec_ref(v_a_3759_);
lean_dec(v_a_3758_);
lean_dec(v_a_3757_);
return v_res_3765_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_replaceFVarS_tacticBuilder(lean_object* v_goal_3766_, lean_object* v_fvarId_3767_, lean_object* v_type_3768_, lean_object* v_proof_3769_, lean_object* v_x_3770_, lean_object* v_a_3771_, lean_object* v_a_3772_, lean_object* v_a_3773_, lean_object* v_a_3774_){
_start:
{
lean_object* v_fst_3776_; lean_object* v___x_3777_; 
v_fst_3776_ = lean_ctor_get(v_x_3770_, 0);
lean_inc(v_fst_3776_);
lean_dec_ref(v_x_3770_);
v___x_3777_ = lp_aesop_Aesop_Script_TacticBuilder_replace(v_goal_3766_, v_fst_3776_, v_fvarId_3767_, v_type_3768_, v_proof_3769_, v_a_3771_, v_a_3772_, v_a_3773_, v_a_3774_);
return v___x_3777_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_replaceFVarS_tacticBuilder___boxed(lean_object* v_goal_3778_, lean_object* v_fvarId_3779_, lean_object* v_type_3780_, lean_object* v_proof_3781_, lean_object* v_x_3782_, lean_object* v_a_3783_, lean_object* v_a_3784_, lean_object* v_a_3785_, lean_object* v_a_3786_, lean_object* v_a_3787_){
_start:
{
lean_object* v_res_3788_; 
v_res_3788_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_replaceFVarS_tacticBuilder(v_goal_3778_, v_fvarId_3779_, v_type_3780_, v_proof_3781_, v_x_3782_, v_a_3783_, v_a_3784_, v_a_3785_, v_a_3786_);
lean_dec(v_a_3786_);
lean_dec_ref(v_a_3785_);
lean_dec(v_a_3784_);
lean_dec_ref(v_a_3783_);
return v_res_3788_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_replaceFVarS___lam__0(lean_object* v_x_3789_){
_start:
{
lean_object* v_fst_3790_; lean_object* v___x_3791_; lean_object* v___x_3792_; lean_object* v___x_3793_; 
v_fst_3790_ = lean_ctor_get(v_x_3789_, 0);
lean_inc(v_fst_3790_);
lean_dec_ref(v_x_3789_);
v___x_3791_ = lean_unsigned_to_nat(1u);
v___x_3792_ = lean_mk_empty_array_with_capacity(v___x_3791_);
v___x_3793_ = lean_array_push(v___x_3792_, v_fst_3790_);
return v___x_3793_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_replaceFVarS___lam__1(lean_object* v_x_3794_){
_start:
{
uint8_t v___x_3795_; 
v___x_3795_ = 1;
return v___x_3795_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_replaceFVarS___lam__1___boxed(lean_object* v_x_3796_){
_start:
{
uint8_t v_res_3797_; lean_object* v_r_3798_; 
v_res_3797_ = lp_aesop_Aesop_replaceFVarS___lam__1(v_x_3796_);
lean_dec_ref(v_x_3796_);
v_r_3798_ = lean_box(v_res_3797_);
return v_r_3798_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_replaceFVarS(lean_object* v_goal_3801_, lean_object* v_fvarId_3802_, lean_object* v_type_3803_, lean_object* v_proof_3804_, lean_object* v_a_3805_, lean_object* v_a_3806_, lean_object* v_a_3807_, lean_object* v_a_3808_, lean_object* v_a_3809_, lean_object* v_a_3810_){
_start:
{
lean_object* v___f_3812_; lean_object* v___f_3813_; lean_object* v___x_3814_; lean_object* v___x_3815_; lean_object* v___x_3816_; 
v___f_3812_ = ((lean_object*)(lp_aesop_Aesop_replaceFVarS___closed__0));
v___f_3813_ = ((lean_object*)(lp_aesop_Aesop_replaceFVarS___closed__1));
lean_inc_ref(v_proof_3804_);
lean_inc_ref(v_type_3803_);
lean_inc(v_fvarId_3802_);
lean_inc_n(v_goal_3801_, 2);
v___x_3814_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_replaceFVarS_tacticBuilder___boxed), 10, 4);
lean_closure_set(v___x_3814_, 0, v_goal_3801_);
lean_closure_set(v___x_3814_, 1, v_fvarId_3802_);
lean_closure_set(v___x_3814_, 2, v_type_3803_);
lean_closure_set(v___x_3814_, 3, v_proof_3804_);
v___x_3815_ = lean_alloc_closure((void*)(lp_aesop_Aesop_replaceFVar___boxed), 9, 4);
lean_closure_set(v___x_3815_, 0, v_goal_3801_);
lean_closure_set(v___x_3815_, 1, v_fvarId_3802_);
lean_closure_set(v___x_3815_, 2, v_type_3803_);
lean_closure_set(v___x_3815_, 3, v_proof_3804_);
v___x_3816_ = lp_aesop_Aesop_withScriptStep___redArg(v_goal_3801_, v___f_3812_, v___f_3813_, v___x_3814_, v___x_3815_, v_a_3805_, v_a_3806_, v_a_3807_, v_a_3808_, v_a_3809_, v_a_3810_);
return v___x_3816_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_replaceFVarS___boxed(lean_object* v_goal_3817_, lean_object* v_fvarId_3818_, lean_object* v_type_3819_, lean_object* v_proof_3820_, lean_object* v_a_3821_, lean_object* v_a_3822_, lean_object* v_a_3823_, lean_object* v_a_3824_, lean_object* v_a_3825_, lean_object* v_a_3826_, lean_object* v_a_3827_){
_start:
{
lean_object* v_res_3828_; 
v_res_3828_ = lp_aesop_Aesop_replaceFVarS(v_goal_3817_, v_fvarId_3818_, v_type_3819_, v_proof_3820_, v_a_3821_, v_a_3822_, v_a_3823_, v_a_3824_, v_a_3825_, v_a_3826_);
lean_dec(v_a_3826_);
lean_dec_ref(v_a_3825_);
lean_dec(v_a_3824_);
lean_dec_ref(v_a_3823_);
lean_dec(v_a_3822_);
lean_dec(v_a_3821_);
return v_res_3828_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_clearS_tacticBuilder___redArg(lean_object* v_goal_3829_, lean_object* v_fvarId_3830_, lean_object* v_a_3831_, lean_object* v_a_3832_, lean_object* v_a_3833_, lean_object* v_a_3834_){
_start:
{
lean_object* v___x_3836_; lean_object* v___x_3837_; lean_object* v___x_3838_; lean_object* v___x_3839_; 
v___x_3836_ = lean_unsigned_to_nat(1u);
v___x_3837_ = lean_mk_empty_array_with_capacity(v___x_3836_);
v___x_3838_ = lean_array_push(v___x_3837_, v_fvarId_3830_);
v___x_3839_ = lp_aesop_Aesop_Script_TacticBuilder_clear(v_goal_3829_, v___x_3838_, v_a_3831_, v_a_3832_, v_a_3833_, v_a_3834_);
return v___x_3839_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_clearS_tacticBuilder___redArg___boxed(lean_object* v_goal_3840_, lean_object* v_fvarId_3841_, lean_object* v_a_3842_, lean_object* v_a_3843_, lean_object* v_a_3844_, lean_object* v_a_3845_, lean_object* v_a_3846_){
_start:
{
lean_object* v_res_3847_; 
v_res_3847_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_clearS_tacticBuilder___redArg(v_goal_3840_, v_fvarId_3841_, v_a_3842_, v_a_3843_, v_a_3844_, v_a_3845_);
lean_dec(v_a_3845_);
lean_dec_ref(v_a_3844_);
lean_dec(v_a_3843_);
lean_dec_ref(v_a_3842_);
return v_res_3847_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_clearS_tacticBuilder(lean_object* v_goal_3848_, lean_object* v_fvarId_3849_, lean_object* v_x_3850_, lean_object* v_a_3851_, lean_object* v_a_3852_, lean_object* v_a_3853_, lean_object* v_a_3854_){
_start:
{
lean_object* v___x_3856_; 
v___x_3856_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_clearS_tacticBuilder___redArg(v_goal_3848_, v_fvarId_3849_, v_a_3851_, v_a_3852_, v_a_3853_, v_a_3854_);
return v___x_3856_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_clearS_tacticBuilder___boxed(lean_object* v_goal_3857_, lean_object* v_fvarId_3858_, lean_object* v_x_3859_, lean_object* v_a_3860_, lean_object* v_a_3861_, lean_object* v_a_3862_, lean_object* v_a_3863_, lean_object* v_a_3864_){
_start:
{
lean_object* v_res_3865_; 
v_res_3865_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_clearS_tacticBuilder(v_goal_3857_, v_fvarId_3858_, v_x_3859_, v_a_3860_, v_a_3861_, v_a_3862_, v_a_3863_);
lean_dec(v_a_3863_);
lean_dec_ref(v_a_3862_);
lean_dec(v_a_3861_);
lean_dec_ref(v_a_3860_);
lean_dec(v_x_3859_);
return v_res_3865_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_clearS___lam__0(lean_object* v_x_3866_){
_start:
{
lean_object* v___x_3867_; lean_object* v___x_3868_; lean_object* v___x_3869_; 
v___x_3867_ = lean_unsigned_to_nat(1u);
v___x_3868_ = lean_mk_empty_array_with_capacity(v___x_3867_);
v___x_3869_ = lean_array_push(v___x_3868_, v_x_3866_);
return v___x_3869_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_clearS___lam__1(lean_object* v_x_3870_){
_start:
{
uint8_t v___x_3871_; 
v___x_3871_ = 1;
return v___x_3871_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_clearS___lam__1___boxed(lean_object* v_x_3872_){
_start:
{
uint8_t v_res_3873_; lean_object* v_r_3874_; 
v_res_3873_ = lp_aesop_Aesop_clearS___lam__1(v_x_3872_);
lean_dec(v_x_3872_);
v_r_3874_ = lean_box(v_res_3873_);
return v_r_3874_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_clearS(lean_object* v_goal_3877_, lean_object* v_fvarId_3878_, lean_object* v_a_3879_, lean_object* v_a_3880_, lean_object* v_a_3881_, lean_object* v_a_3882_, lean_object* v_a_3883_, lean_object* v_a_3884_){
_start:
{
lean_object* v___f_3886_; lean_object* v___f_3887_; lean_object* v___x_3888_; lean_object* v___x_3889_; lean_object* v___x_3890_; 
v___f_3886_ = ((lean_object*)(lp_aesop_Aesop_clearS___closed__0));
v___f_3887_ = ((lean_object*)(lp_aesop_Aesop_clearS___closed__1));
lean_inc(v_fvarId_3878_);
lean_inc_n(v_goal_3877_, 2);
v___x_3888_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_clearS_tacticBuilder___boxed), 8, 2);
lean_closure_set(v___x_3888_, 0, v_goal_3877_);
lean_closure_set(v___x_3888_, 1, v_fvarId_3878_);
v___x_3889_ = lean_alloc_closure((void*)(l_Lean_MVarId_clear___boxed), 7, 2);
lean_closure_set(v___x_3889_, 0, v_goal_3877_);
lean_closure_set(v___x_3889_, 1, v_fvarId_3878_);
v___x_3890_ = lp_aesop_Aesop_withScriptStep___redArg(v_goal_3877_, v___f_3886_, v___f_3887_, v___x_3888_, v___x_3889_, v_a_3879_, v_a_3880_, v_a_3881_, v_a_3882_, v_a_3883_, v_a_3884_);
return v___x_3890_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_clearS___boxed(lean_object* v_goal_3891_, lean_object* v_fvarId_3892_, lean_object* v_a_3893_, lean_object* v_a_3894_, lean_object* v_a_3895_, lean_object* v_a_3896_, lean_object* v_a_3897_, lean_object* v_a_3898_, lean_object* v_a_3899_){
_start:
{
lean_object* v_res_3900_; 
v_res_3900_ = lp_aesop_Aesop_clearS(v_goal_3891_, v_fvarId_3892_, v_a_3893_, v_a_3894_, v_a_3895_, v_a_3896_, v_a_3897_, v_a_3898_);
lean_dec(v_a_3898_);
lean_dec_ref(v_a_3897_);
lean_dec(v_a_3896_);
lean_dec_ref(v_a_3895_);
lean_dec(v_a_3894_);
lean_dec(v_a_3893_);
return v_res_3900_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_tryClearS_tacticBuilder___redArg(lean_object* v_goal_3901_, lean_object* v_fvarId_3902_, lean_object* v_a_3903_, lean_object* v_a_3904_, lean_object* v_a_3905_, lean_object* v_a_3906_){
_start:
{
lean_object* v___x_3908_; lean_object* v___x_3909_; lean_object* v___x_3910_; lean_object* v___x_3911_; 
v___x_3908_ = lean_unsigned_to_nat(1u);
v___x_3909_ = lean_mk_empty_array_with_capacity(v___x_3908_);
v___x_3910_ = lean_array_push(v___x_3909_, v_fvarId_3902_);
v___x_3911_ = lp_aesop_Aesop_Script_TacticBuilder_clear(v_goal_3901_, v___x_3910_, v_a_3903_, v_a_3904_, v_a_3905_, v_a_3906_);
return v___x_3911_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_tryClearS_tacticBuilder___redArg___boxed(lean_object* v_goal_3912_, lean_object* v_fvarId_3913_, lean_object* v_a_3914_, lean_object* v_a_3915_, lean_object* v_a_3916_, lean_object* v_a_3917_, lean_object* v_a_3918_){
_start:
{
lean_object* v_res_3919_; 
v_res_3919_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_tryClearS_tacticBuilder___redArg(v_goal_3912_, v_fvarId_3913_, v_a_3914_, v_a_3915_, v_a_3916_, v_a_3917_);
lean_dec(v_a_3917_);
lean_dec_ref(v_a_3916_);
lean_dec(v_a_3915_);
lean_dec_ref(v_a_3914_);
return v_res_3919_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_tryClearS_tacticBuilder(lean_object* v_goal_3920_, lean_object* v_fvarId_3921_, lean_object* v_x_3922_, lean_object* v_a_3923_, lean_object* v_a_3924_, lean_object* v_a_3925_, lean_object* v_a_3926_){
_start:
{
lean_object* v___x_3928_; 
v___x_3928_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_tryClearS_tacticBuilder___redArg(v_goal_3920_, v_fvarId_3921_, v_a_3923_, v_a_3924_, v_a_3925_, v_a_3926_);
return v___x_3928_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_tryClearS_tacticBuilder___boxed(lean_object* v_goal_3929_, lean_object* v_fvarId_3930_, lean_object* v_x_3931_, lean_object* v_a_3932_, lean_object* v_a_3933_, lean_object* v_a_3934_, lean_object* v_a_3935_, lean_object* v_a_3936_){
_start:
{
lean_object* v_res_3937_; 
v_res_3937_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_tryClearS_tacticBuilder(v_goal_3929_, v_fvarId_3930_, v_x_3931_, v_a_3932_, v_a_3933_, v_a_3934_, v_a_3935_);
lean_dec(v_a_3935_);
lean_dec_ref(v_a_3934_);
lean_dec(v_a_3933_);
lean_dec_ref(v_a_3932_);
lean_dec(v_x_3931_);
return v_res_3937_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryClearS___redArg___lam__1(lean_object* v_goal_3938_, lean_object* v_fvarId_3939_, lean_object* v___y_3940_, lean_object* v___y_3941_, lean_object* v___y_3942_, lean_object* v___y_3943_){
_start:
{
lean_object* v___x_3945_; 
v___x_3945_ = l_Lean_MVarId_tryClear(v_goal_3938_, v_fvarId_3939_, v___y_3940_, v___y_3941_, v___y_3942_, v___y_3943_);
if (lean_obj_tag(v___x_3945_) == 0)
{
lean_object* v_a_3946_; lean_object* v___x_3948_; uint8_t v_isShared_3949_; uint8_t v_isSharedCheck_3954_; 
v_a_3946_ = lean_ctor_get(v___x_3945_, 0);
v_isSharedCheck_3954_ = !lean_is_exclusive(v___x_3945_);
if (v_isSharedCheck_3954_ == 0)
{
v___x_3948_ = v___x_3945_;
v_isShared_3949_ = v_isSharedCheck_3954_;
goto v_resetjp_3947_;
}
else
{
lean_inc(v_a_3946_);
lean_dec(v___x_3945_);
v___x_3948_ = lean_box(0);
v_isShared_3949_ = v_isSharedCheck_3954_;
goto v_resetjp_3947_;
}
v_resetjp_3947_:
{
lean_object* v___x_3950_; lean_object* v___x_3952_; 
v___x_3950_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3950_, 0, v_a_3946_);
if (v_isShared_3949_ == 0)
{
lean_ctor_set(v___x_3948_, 0, v___x_3950_);
v___x_3952_ = v___x_3948_;
goto v_reusejp_3951_;
}
else
{
lean_object* v_reuseFailAlloc_3953_; 
v_reuseFailAlloc_3953_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3953_, 0, v___x_3950_);
v___x_3952_ = v_reuseFailAlloc_3953_;
goto v_reusejp_3951_;
}
v_reusejp_3951_:
{
return v___x_3952_;
}
}
}
else
{
lean_object* v_a_3955_; lean_object* v___x_3957_; uint8_t v_isShared_3958_; uint8_t v_isSharedCheck_3962_; 
v_a_3955_ = lean_ctor_get(v___x_3945_, 0);
v_isSharedCheck_3962_ = !lean_is_exclusive(v___x_3945_);
if (v_isSharedCheck_3962_ == 0)
{
v___x_3957_ = v___x_3945_;
v_isShared_3958_ = v_isSharedCheck_3962_;
goto v_resetjp_3956_;
}
else
{
lean_inc(v_a_3955_);
lean_dec(v___x_3945_);
v___x_3957_ = lean_box(0);
v_isShared_3958_ = v_isSharedCheck_3962_;
goto v_resetjp_3956_;
}
v_resetjp_3956_:
{
lean_object* v___x_3960_; 
if (v_isShared_3958_ == 0)
{
v___x_3960_ = v___x_3957_;
goto v_reusejp_3959_;
}
else
{
lean_object* v_reuseFailAlloc_3961_; 
v_reuseFailAlloc_3961_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3961_, 0, v_a_3955_);
v___x_3960_ = v_reuseFailAlloc_3961_;
goto v_reusejp_3959_;
}
v_reusejp_3959_:
{
return v___x_3960_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryClearS___redArg___lam__1___boxed(lean_object* v_goal_3963_, lean_object* v_fvarId_3964_, lean_object* v___y_3965_, lean_object* v___y_3966_, lean_object* v___y_3967_, lean_object* v___y_3968_, lean_object* v___y_3969_){
_start:
{
lean_object* v_res_3970_; 
v_res_3970_ = lp_aesop_Aesop_tryClearS___redArg___lam__1(v_goal_3963_, v_fvarId_3964_, v___y_3965_, v___y_3966_, v___y_3967_, v___y_3968_);
lean_dec(v___y_3968_);
lean_dec_ref(v___y_3967_);
lean_dec(v___y_3966_);
lean_dec_ref(v___y_3965_);
return v_res_3970_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryClearS___redArg(lean_object* v_goal_3971_, lean_object* v_fvarId_3972_, lean_object* v_a_3973_, lean_object* v_a_3974_, lean_object* v_a_3975_, lean_object* v_a_3976_, lean_object* v_a_3977_){
_start:
{
lean_object* v___f_3979_; lean_object* v___f_3980_; lean_object* v___x_3981_; lean_object* v___x_3982_; 
v___f_3979_ = ((lean_object*)(lp_aesop_Aesop_clearS___closed__0));
lean_inc(v_fvarId_3972_);
lean_inc_n(v_goal_3971_, 2);
v___f_3980_ = lean_alloc_closure((void*)(lp_aesop_Aesop_tryClearS___redArg___lam__1___boxed), 7, 2);
lean_closure_set(v___f_3980_, 0, v_goal_3971_);
lean_closure_set(v___f_3980_, 1, v_fvarId_3972_);
v___x_3981_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_tryClearS_tacticBuilder___boxed), 8, 2);
lean_closure_set(v___x_3981_, 0, v_goal_3971_);
lean_closure_set(v___x_3981_, 1, v_fvarId_3972_);
v___x_3982_ = lp_aesop_Aesop_withOptScriptStep___redArg(v_goal_3971_, v___f_3979_, v___x_3981_, v___f_3980_, v_a_3973_, v_a_3974_, v_a_3975_, v_a_3976_, v_a_3977_);
return v___x_3982_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryClearS___redArg___boxed(lean_object* v_goal_3983_, lean_object* v_fvarId_3984_, lean_object* v_a_3985_, lean_object* v_a_3986_, lean_object* v_a_3987_, lean_object* v_a_3988_, lean_object* v_a_3989_, lean_object* v_a_3990_){
_start:
{
lean_object* v_res_3991_; 
v_res_3991_ = lp_aesop_Aesop_tryClearS___redArg(v_goal_3983_, v_fvarId_3984_, v_a_3985_, v_a_3986_, v_a_3987_, v_a_3988_, v_a_3989_);
lean_dec(v_a_3989_);
lean_dec_ref(v_a_3988_);
lean_dec(v_a_3987_);
lean_dec_ref(v_a_3986_);
lean_dec(v_a_3985_);
return v_res_3991_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryClearS(lean_object* v_goal_3992_, lean_object* v_fvarId_3993_, lean_object* v_a_3994_, lean_object* v_a_3995_, lean_object* v_a_3996_, lean_object* v_a_3997_, lean_object* v_a_3998_, lean_object* v_a_3999_){
_start:
{
lean_object* v___x_4001_; 
v___x_4001_ = lp_aesop_Aesop_tryClearS___redArg(v_goal_3992_, v_fvarId_3993_, v_a_3994_, v_a_3996_, v_a_3997_, v_a_3998_, v_a_3999_);
return v___x_4001_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryClearS___boxed(lean_object* v_goal_4002_, lean_object* v_fvarId_4003_, lean_object* v_a_4004_, lean_object* v_a_4005_, lean_object* v_a_4006_, lean_object* v_a_4007_, lean_object* v_a_4008_, lean_object* v_a_4009_, lean_object* v_a_4010_){
_start:
{
lean_object* v_res_4011_; 
v_res_4011_ = lp_aesop_Aesop_tryClearS(v_goal_4002_, v_fvarId_4003_, v_a_4004_, v_a_4005_, v_a_4006_, v_a_4007_, v_a_4008_, v_a_4009_);
lean_dec(v_a_4009_);
lean_dec_ref(v_a_4008_);
lean_dec(v_a_4007_);
lean_dec_ref(v_a_4006_);
lean_dec(v_a_4005_);
lean_dec(v_a_4004_);
return v_res_4011_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_tryClearManyS_tacticBuilder(lean_object* v_goal_4012_, lean_object* v_x_4013_, lean_object* v_a_4014_, lean_object* v_a_4015_, lean_object* v_a_4016_, lean_object* v_a_4017_){
_start:
{
lean_object* v_snd_4019_; lean_object* v___x_4020_; 
v_snd_4019_ = lean_ctor_get(v_x_4013_, 1);
lean_inc(v_snd_4019_);
lean_dec_ref(v_x_4013_);
v___x_4020_ = lp_aesop_Aesop_Script_TacticBuilder_clear(v_goal_4012_, v_snd_4019_, v_a_4014_, v_a_4015_, v_a_4016_, v_a_4017_);
return v___x_4020_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_tryClearManyS_tacticBuilder___boxed(lean_object* v_goal_4021_, lean_object* v_x_4022_, lean_object* v_a_4023_, lean_object* v_a_4024_, lean_object* v_a_4025_, lean_object* v_a_4026_, lean_object* v_a_4027_){
_start:
{
lean_object* v_res_4028_; 
v_res_4028_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_tryClearManyS_tacticBuilder(v_goal_4021_, v_x_4022_, v_a_4023_, v_a_4024_, v_a_4025_, v_a_4026_);
lean_dec(v_a_4026_);
lean_dec_ref(v_a_4025_);
lean_dec(v_a_4024_);
lean_dec_ref(v_a_4023_);
return v_res_4028_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_tryClearManyS___lam__1(lean_object* v_x_4029_){
_start:
{
lean_object* v_snd_4030_; lean_object* v___x_4031_; lean_object* v___x_4032_; uint8_t v___x_4033_; 
v_snd_4030_ = lean_ctor_get(v_x_4029_, 1);
v___x_4031_ = lean_array_get_size(v_snd_4030_);
v___x_4032_ = lean_unsigned_to_nat(0u);
v___x_4033_ = lean_nat_dec_eq(v___x_4031_, v___x_4032_);
if (v___x_4033_ == 0)
{
uint8_t v___x_4034_; 
v___x_4034_ = 1;
return v___x_4034_;
}
else
{
uint8_t v___x_4035_; 
v___x_4035_ = 0;
return v___x_4035_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryClearManyS___lam__1___boxed(lean_object* v_x_4036_){
_start:
{
uint8_t v_res_4037_; lean_object* v_r_4038_; 
v_res_4037_ = lp_aesop_Aesop_tryClearManyS___lam__1(v_x_4036_);
lean_dec_ref(v_x_4036_);
v_r_4038_ = lean_box(v_res_4037_);
return v_r_4038_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryClearManyS(lean_object* v_goal_4040_, lean_object* v_fvarIds_4041_, lean_object* v_a_4042_, lean_object* v_a_4043_, lean_object* v_a_4044_, lean_object* v_a_4045_, lean_object* v_a_4046_, lean_object* v_a_4047_){
_start:
{
lean_object* v___f_4049_; lean_object* v___f_4050_; lean_object* v___x_4051_; lean_object* v___x_4052_; lean_object* v___x_4053_; 
v___f_4049_ = ((lean_object*)(lp_aesop_Aesop_assertHypothesisS___closed__0));
v___f_4050_ = ((lean_object*)(lp_aesop_Aesop_tryClearManyS___closed__0));
lean_inc_n(v_goal_4040_, 2);
v___x_4051_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_tryClearManyS_tacticBuilder___boxed), 7, 1);
lean_closure_set(v___x_4051_, 0, v_goal_4040_);
v___x_4052_ = lean_alloc_closure((void*)(l_Lean_MVarId_tryClearMany_x27___boxed), 7, 2);
lean_closure_set(v___x_4052_, 0, v_goal_4040_);
lean_closure_set(v___x_4052_, 1, v_fvarIds_4041_);
v___x_4053_ = lp_aesop_Aesop_withScriptStep___redArg(v_goal_4040_, v___f_4049_, v___f_4050_, v___x_4051_, v___x_4052_, v_a_4042_, v_a_4043_, v_a_4044_, v_a_4045_, v_a_4046_, v_a_4047_);
return v___x_4053_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryClearManyS___boxed(lean_object* v_goal_4054_, lean_object* v_fvarIds_4055_, lean_object* v_a_4056_, lean_object* v_a_4057_, lean_object* v_a_4058_, lean_object* v_a_4059_, lean_object* v_a_4060_, lean_object* v_a_4061_, lean_object* v_a_4062_){
_start:
{
lean_object* v_res_4063_; 
v_res_4063_ = lp_aesop_Aesop_tryClearManyS(v_goal_4054_, v_fvarIds_4055_, v_a_4056_, v_a_4057_, v_a_4058_, v_a_4059_, v_a_4060_, v_a_4061_);
lean_dec(v_a_4061_);
lean_dec_ref(v_a_4060_);
lean_dec(v_a_4059_);
lean_dec_ref(v_a_4058_);
lean_dec(v_a_4057_);
lean_dec(v_a_4056_);
return v_res_4063_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_tryCasesS_getUnusedCtorNames_spec__0(lean_object* v_as_4064_, size_t v_i_4065_, size_t v_stop_4066_, lean_object* v_b_4067_){
_start:
{
uint8_t v___x_4068_; 
v___x_4068_ = lean_usize_dec_eq(v_i_4065_, v_stop_4066_);
if (v___x_4068_ == 0)
{
lean_object* v_fst_4069_; lean_object* v_snd_4070_; lean_object* v___x_4071_; lean_object* v___x_4072_; lean_object* v_fst_4073_; lean_object* v_snd_4074_; lean_object* v___x_4076_; uint8_t v_isShared_4077_; uint8_t v_isSharedCheck_4085_; 
v_fst_4069_ = lean_ctor_get(v_b_4067_, 0);
lean_inc(v_fst_4069_);
v_snd_4070_ = lean_ctor_get(v_b_4067_, 1);
lean_inc(v_snd_4070_);
lean_dec_ref(v_b_4067_);
v___x_4071_ = lean_array_uget_borrowed(v_as_4064_, v_i_4065_);
lean_inc(v___x_4071_);
v___x_4072_ = lp_aesop_Aesop_CtorNames_mkFreshArgNames(v_snd_4070_, v___x_4071_);
v_fst_4073_ = lean_ctor_get(v___x_4072_, 0);
v_snd_4074_ = lean_ctor_get(v___x_4072_, 1);
v_isSharedCheck_4085_ = !lean_is_exclusive(v___x_4072_);
if (v_isSharedCheck_4085_ == 0)
{
v___x_4076_ = v___x_4072_;
v_isShared_4077_ = v_isSharedCheck_4085_;
goto v_resetjp_4075_;
}
else
{
lean_inc(v_snd_4074_);
lean_inc(v_fst_4073_);
lean_dec(v___x_4072_);
v___x_4076_ = lean_box(0);
v_isShared_4077_ = v_isSharedCheck_4085_;
goto v_resetjp_4075_;
}
v_resetjp_4075_:
{
lean_object* v___x_4078_; lean_object* v___x_4080_; 
v___x_4078_ = lean_array_push(v_fst_4069_, v_fst_4073_);
if (v_isShared_4077_ == 0)
{
lean_ctor_set(v___x_4076_, 0, v___x_4078_);
v___x_4080_ = v___x_4076_;
goto v_reusejp_4079_;
}
else
{
lean_object* v_reuseFailAlloc_4084_; 
v_reuseFailAlloc_4084_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4084_, 0, v___x_4078_);
lean_ctor_set(v_reuseFailAlloc_4084_, 1, v_snd_4074_);
v___x_4080_ = v_reuseFailAlloc_4084_;
goto v_reusejp_4079_;
}
v_reusejp_4079_:
{
size_t v___x_4081_; size_t v___x_4082_; 
v___x_4081_ = ((size_t)1ULL);
v___x_4082_ = lean_usize_add(v_i_4065_, v___x_4081_);
v_i_4065_ = v___x_4082_;
v_b_4067_ = v___x_4080_;
goto _start;
}
}
}
else
{
return v_b_4067_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_tryCasesS_getUnusedCtorNames_spec__0___boxed(lean_object* v_as_4086_, lean_object* v_i_4087_, lean_object* v_stop_4088_, lean_object* v_b_4089_){
_start:
{
size_t v_i_boxed_4090_; size_t v_stop_boxed_4091_; lean_object* v_res_4092_; 
v_i_boxed_4090_ = lean_unbox_usize(v_i_4087_);
lean_dec(v_i_4087_);
v_stop_boxed_4091_ = lean_unbox_usize(v_stop_4088_);
lean_dec(v_stop_4088_);
v_res_4092_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_tryCasesS_getUnusedCtorNames_spec__0(v_as_4086_, v_i_boxed_4090_, v_stop_boxed_4091_, v_b_4089_);
lean_dec_ref(v_as_4086_);
return v_res_4092_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_tryCasesS_getUnusedCtorNames(lean_object* v_ctorNames_4093_, lean_object* v_lctx_4094_){
_start:
{
lean_object* v___x_4095_; lean_object* v___x_4096_; lean_object* v___x_4097_; uint8_t v___x_4098_; 
v___x_4095_ = lean_array_get_size(v_ctorNames_4093_);
v___x_4096_ = lean_mk_empty_array_with_capacity(v___x_4095_);
v___x_4097_ = lean_unsigned_to_nat(0u);
v___x_4098_ = lean_nat_dec_lt(v___x_4097_, v___x_4095_);
if (v___x_4098_ == 0)
{
lean_dec_ref(v_lctx_4094_);
return v___x_4096_;
}
else
{
lean_object* v___x_4099_; uint8_t v___x_4100_; 
lean_inc_ref(v___x_4096_);
v___x_4099_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4099_, 0, v___x_4096_);
lean_ctor_set(v___x_4099_, 1, v_lctx_4094_);
v___x_4100_ = lean_nat_dec_le(v___x_4095_, v___x_4095_);
if (v___x_4100_ == 0)
{
if (v___x_4098_ == 0)
{
lean_dec_ref_known(v___x_4099_, 2);
return v___x_4096_;
}
else
{
size_t v___x_4101_; size_t v___x_4102_; lean_object* v___x_4103_; lean_object* v_fst_4104_; 
lean_dec_ref(v___x_4096_);
v___x_4101_ = ((size_t)0ULL);
v___x_4102_ = lean_usize_of_nat(v___x_4095_);
v___x_4103_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_tryCasesS_getUnusedCtorNames_spec__0(v_ctorNames_4093_, v___x_4101_, v___x_4102_, v___x_4099_);
v_fst_4104_ = lean_ctor_get(v___x_4103_, 0);
lean_inc(v_fst_4104_);
lean_dec_ref(v___x_4103_);
return v_fst_4104_;
}
}
else
{
size_t v___x_4105_; size_t v___x_4106_; lean_object* v___x_4107_; lean_object* v_fst_4108_; 
lean_dec_ref(v___x_4096_);
v___x_4105_ = ((size_t)0ULL);
v___x_4106_ = lean_usize_of_nat(v___x_4095_);
v___x_4107_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_tryCasesS_getUnusedCtorNames_spec__0(v_ctorNames_4093_, v___x_4105_, v___x_4106_, v___x_4099_);
v_fst_4108_ = lean_ctor_get(v___x_4107_, 0);
lean_inc(v_fst_4108_);
lean_dec_ref(v___x_4107_);
return v_fst_4108_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_tryCasesS_getUnusedCtorNames___boxed(lean_object* v_ctorNames_4109_, lean_object* v_lctx_4110_){
_start:
{
lean_object* v_res_4111_; 
v_res_4111_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_tryCasesS_getUnusedCtorNames(v_ctorNames_4109_, v_lctx_4110_);
lean_dec_ref(v_ctorNames_4109_);
return v_res_4111_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_observing_x3f___at___00Aesop_tryCasesS_spec__2___redArg(lean_object* v_x_4112_, lean_object* v___y_4113_, lean_object* v___y_4114_, lean_object* v___y_4115_, lean_object* v___y_4116_){
_start:
{
lean_object* v___x_4118_; 
v___x_4118_ = l_Lean_Meta_saveState___redArg(v___y_4114_, v___y_4116_);
if (lean_obj_tag(v___x_4118_) == 0)
{
lean_object* v_a_4119_; lean_object* v___x_4120_; 
v_a_4119_ = lean_ctor_get(v___x_4118_, 0);
lean_inc(v_a_4119_);
lean_dec_ref_known(v___x_4118_, 1);
lean_inc(v___y_4116_);
lean_inc_ref(v___y_4115_);
lean_inc(v___y_4114_);
lean_inc_ref(v___y_4113_);
v___x_4120_ = lean_apply_5(v_x_4112_, v___y_4113_, v___y_4114_, v___y_4115_, v___y_4116_, lean_box(0));
if (lean_obj_tag(v___x_4120_) == 0)
{
lean_object* v_a_4121_; lean_object* v___x_4123_; uint8_t v_isShared_4124_; uint8_t v_isSharedCheck_4129_; 
lean_dec(v_a_4119_);
v_a_4121_ = lean_ctor_get(v___x_4120_, 0);
v_isSharedCheck_4129_ = !lean_is_exclusive(v___x_4120_);
if (v_isSharedCheck_4129_ == 0)
{
v___x_4123_ = v___x_4120_;
v_isShared_4124_ = v_isSharedCheck_4129_;
goto v_resetjp_4122_;
}
else
{
lean_inc(v_a_4121_);
lean_dec(v___x_4120_);
v___x_4123_ = lean_box(0);
v_isShared_4124_ = v_isSharedCheck_4129_;
goto v_resetjp_4122_;
}
v_resetjp_4122_:
{
lean_object* v___x_4125_; lean_object* v___x_4127_; 
v___x_4125_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4125_, 0, v_a_4121_);
if (v_isShared_4124_ == 0)
{
lean_ctor_set(v___x_4123_, 0, v___x_4125_);
v___x_4127_ = v___x_4123_;
goto v_reusejp_4126_;
}
else
{
lean_object* v_reuseFailAlloc_4128_; 
v_reuseFailAlloc_4128_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4128_, 0, v___x_4125_);
v___x_4127_ = v_reuseFailAlloc_4128_;
goto v_reusejp_4126_;
}
v_reusejp_4126_:
{
return v___x_4127_;
}
}
}
else
{
lean_object* v_a_4130_; lean_object* v___x_4132_; uint8_t v_isShared_4133_; uint8_t v_isSharedCheck_4159_; 
v_a_4130_ = lean_ctor_get(v___x_4120_, 0);
v_isSharedCheck_4159_ = !lean_is_exclusive(v___x_4120_);
if (v_isSharedCheck_4159_ == 0)
{
v___x_4132_ = v___x_4120_;
v_isShared_4133_ = v_isSharedCheck_4159_;
goto v_resetjp_4131_;
}
else
{
lean_inc(v_a_4130_);
lean_dec(v___x_4120_);
v___x_4132_ = lean_box(0);
v_isShared_4133_ = v_isSharedCheck_4159_;
goto v_resetjp_4131_;
}
v_resetjp_4131_:
{
uint8_t v___y_4135_; uint8_t v___x_4157_; 
v___x_4157_ = l_Lean_Exception_isInterrupt(v_a_4130_);
if (v___x_4157_ == 0)
{
uint8_t v___x_4158_; 
lean_inc(v_a_4130_);
v___x_4158_ = l_Lean_Exception_isRuntime(v_a_4130_);
v___y_4135_ = v___x_4158_;
goto v___jp_4134_;
}
else
{
v___y_4135_ = v___x_4157_;
goto v___jp_4134_;
}
v___jp_4134_:
{
if (v___y_4135_ == 0)
{
lean_object* v___x_4136_; 
lean_del_object(v___x_4132_);
lean_dec(v_a_4130_);
v___x_4136_ = l_Lean_Meta_SavedState_restore___redArg(v_a_4119_, v___y_4114_, v___y_4116_);
lean_dec(v_a_4119_);
if (lean_obj_tag(v___x_4136_) == 0)
{
lean_object* v___x_4138_; uint8_t v_isShared_4139_; uint8_t v_isSharedCheck_4144_; 
v_isSharedCheck_4144_ = !lean_is_exclusive(v___x_4136_);
if (v_isSharedCheck_4144_ == 0)
{
lean_object* v_unused_4145_; 
v_unused_4145_ = lean_ctor_get(v___x_4136_, 0);
lean_dec(v_unused_4145_);
v___x_4138_ = v___x_4136_;
v_isShared_4139_ = v_isSharedCheck_4144_;
goto v_resetjp_4137_;
}
else
{
lean_dec(v___x_4136_);
v___x_4138_ = lean_box(0);
v_isShared_4139_ = v_isSharedCheck_4144_;
goto v_resetjp_4137_;
}
v_resetjp_4137_:
{
lean_object* v___x_4140_; lean_object* v___x_4142_; 
v___x_4140_ = lean_box(0);
if (v_isShared_4139_ == 0)
{
lean_ctor_set(v___x_4138_, 0, v___x_4140_);
v___x_4142_ = v___x_4138_;
goto v_reusejp_4141_;
}
else
{
lean_object* v_reuseFailAlloc_4143_; 
v_reuseFailAlloc_4143_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4143_, 0, v___x_4140_);
v___x_4142_ = v_reuseFailAlloc_4143_;
goto v_reusejp_4141_;
}
v_reusejp_4141_:
{
return v___x_4142_;
}
}
}
else
{
lean_object* v_a_4146_; lean_object* v___x_4148_; uint8_t v_isShared_4149_; uint8_t v_isSharedCheck_4153_; 
v_a_4146_ = lean_ctor_get(v___x_4136_, 0);
v_isSharedCheck_4153_ = !lean_is_exclusive(v___x_4136_);
if (v_isSharedCheck_4153_ == 0)
{
v___x_4148_ = v___x_4136_;
v_isShared_4149_ = v_isSharedCheck_4153_;
goto v_resetjp_4147_;
}
else
{
lean_inc(v_a_4146_);
lean_dec(v___x_4136_);
v___x_4148_ = lean_box(0);
v_isShared_4149_ = v_isSharedCheck_4153_;
goto v_resetjp_4147_;
}
v_resetjp_4147_:
{
lean_object* v___x_4151_; 
if (v_isShared_4149_ == 0)
{
v___x_4151_ = v___x_4148_;
goto v_reusejp_4150_;
}
else
{
lean_object* v_reuseFailAlloc_4152_; 
v_reuseFailAlloc_4152_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4152_, 0, v_a_4146_);
v___x_4151_ = v_reuseFailAlloc_4152_;
goto v_reusejp_4150_;
}
v_reusejp_4150_:
{
return v___x_4151_;
}
}
}
}
else
{
lean_object* v___x_4155_; 
lean_dec(v_a_4119_);
if (v_isShared_4133_ == 0)
{
v___x_4155_ = v___x_4132_;
goto v_reusejp_4154_;
}
else
{
lean_object* v_reuseFailAlloc_4156_; 
v_reuseFailAlloc_4156_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4156_, 0, v_a_4130_);
v___x_4155_ = v_reuseFailAlloc_4156_;
goto v_reusejp_4154_;
}
v_reusejp_4154_:
{
return v___x_4155_;
}
}
}
}
}
}
else
{
lean_object* v_a_4160_; lean_object* v___x_4162_; uint8_t v_isShared_4163_; uint8_t v_isSharedCheck_4167_; 
lean_dec_ref(v_x_4112_);
v_a_4160_ = lean_ctor_get(v___x_4118_, 0);
v_isSharedCheck_4167_ = !lean_is_exclusive(v___x_4118_);
if (v_isSharedCheck_4167_ == 0)
{
v___x_4162_ = v___x_4118_;
v_isShared_4163_ = v_isSharedCheck_4167_;
goto v_resetjp_4161_;
}
else
{
lean_inc(v_a_4160_);
lean_dec(v___x_4118_);
v___x_4162_ = lean_box(0);
v_isShared_4163_ = v_isSharedCheck_4167_;
goto v_resetjp_4161_;
}
v_resetjp_4161_:
{
lean_object* v___x_4165_; 
if (v_isShared_4163_ == 0)
{
v___x_4165_ = v___x_4162_;
goto v_reusejp_4164_;
}
else
{
lean_object* v_reuseFailAlloc_4166_; 
v_reuseFailAlloc_4166_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4166_, 0, v_a_4160_);
v___x_4165_ = v_reuseFailAlloc_4166_;
goto v_reusejp_4164_;
}
v_reusejp_4164_:
{
return v___x_4165_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_observing_x3f___at___00Aesop_tryCasesS_spec__2___redArg___boxed(lean_object* v_x_4168_, lean_object* v___y_4169_, lean_object* v___y_4170_, lean_object* v___y_4171_, lean_object* v___y_4172_, lean_object* v___y_4173_){
_start:
{
lean_object* v_res_4174_; 
v_res_4174_ = lp_aesop_Lean_observing_x3f___at___00Aesop_tryCasesS_spec__2___redArg(v_x_4168_, v___y_4169_, v___y_4170_, v___y_4171_, v___y_4172_);
lean_dec(v___y_4172_);
lean_dec_ref(v___y_4171_);
lean_dec(v___y_4170_);
lean_dec_ref(v___y_4169_);
return v_res_4174_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_observing_x3f___at___00Aesop_tryCasesS_spec__2(lean_object* v_00_u03b1_4175_, lean_object* v_x_4176_, lean_object* v___y_4177_, lean_object* v___y_4178_, lean_object* v___y_4179_, lean_object* v___y_4180_){
_start:
{
lean_object* v___x_4182_; 
v___x_4182_ = lp_aesop_Lean_observing_x3f___at___00Aesop_tryCasesS_spec__2___redArg(v_x_4176_, v___y_4177_, v___y_4178_, v___y_4179_, v___y_4180_);
return v___x_4182_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_observing_x3f___at___00Aesop_tryCasesS_spec__2___boxed(lean_object* v_00_u03b1_4183_, lean_object* v_x_4184_, lean_object* v___y_4185_, lean_object* v___y_4186_, lean_object* v___y_4187_, lean_object* v___y_4188_, lean_object* v___y_4189_){
_start:
{
lean_object* v_res_4190_; 
v_res_4190_ = lp_aesop_Lean_observing_x3f___at___00Aesop_tryCasesS_spec__2(v_00_u03b1_4183_, v_x_4184_, v___y_4185_, v___y_4186_, v___y_4187_, v___y_4188_);
lean_dec(v___y_4188_);
lean_dec_ref(v___y_4187_);
lean_dec(v___y_4186_);
lean_dec_ref(v___y_4185_);
return v_res_4190_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_tryCasesS_spec__0(size_t v_sz_4191_, size_t v_i_4192_, lean_object* v_bs_4193_){
_start:
{
uint8_t v___x_4194_; 
v___x_4194_ = lean_usize_dec_lt(v_i_4192_, v_sz_4191_);
if (v___x_4194_ == 0)
{
return v_bs_4193_;
}
else
{
lean_object* v_v_4195_; lean_object* v_toInductionSubgoal_4196_; lean_object* v_mvarId_4197_; lean_object* v___x_4198_; lean_object* v_bs_x27_4199_; size_t v___x_4200_; size_t v___x_4201_; lean_object* v___x_4202_; 
v_v_4195_ = lean_array_uget_borrowed(v_bs_4193_, v_i_4192_);
v_toInductionSubgoal_4196_ = lean_ctor_get(v_v_4195_, 0);
v_mvarId_4197_ = lean_ctor_get(v_toInductionSubgoal_4196_, 0);
lean_inc(v_mvarId_4197_);
v___x_4198_ = lean_unsigned_to_nat(0u);
v_bs_x27_4199_ = lean_array_uset(v_bs_4193_, v_i_4192_, v___x_4198_);
v___x_4200_ = ((size_t)1ULL);
v___x_4201_ = lean_usize_add(v_i_4192_, v___x_4200_);
v___x_4202_ = lean_array_uset(v_bs_x27_4199_, v_i_4192_, v_mvarId_4197_);
v_i_4192_ = v___x_4201_;
v_bs_4193_ = v___x_4202_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_tryCasesS_spec__0___boxed(lean_object* v_sz_4204_, lean_object* v_i_4205_, lean_object* v_bs_4206_){
_start:
{
size_t v_sz_boxed_4207_; size_t v_i_boxed_4208_; lean_object* v_res_4209_; 
v_sz_boxed_4207_ = lean_unbox_usize(v_sz_4204_);
lean_dec(v_sz_4204_);
v_i_boxed_4208_ = lean_unbox_usize(v_i_4205_);
lean_dec(v_i_4205_);
v_res_4209_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_tryCasesS_spec__0(v_sz_boxed_4207_, v_i_boxed_4208_, v_bs_4206_);
return v_res_4209_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryCasesS___redArg___lam__0(lean_object* v_x_4210_){
_start:
{
size_t v_sz_4211_; size_t v___x_4212_; lean_object* v___x_4213_; 
v_sz_4211_ = lean_array_size(v_x_4210_);
v___x_4212_ = ((size_t)0ULL);
v___x_4213_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_tryCasesS_spec__0(v_sz_4211_, v___x_4212_, v_x_4210_);
return v___x_4213_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryCasesS___redArg___lam__1(lean_object* v_fvarId_4214_, lean_object* v_goal_4215_, lean_object* v___x_4216_, lean_object* v_x_4217_, lean_object* v___y_4218_, lean_object* v___y_4219_, lean_object* v___y_4220_, lean_object* v___y_4221_){
_start:
{
lean_object* v___x_4223_; lean_object* v___x_4224_; 
v___x_4223_ = l_Lean_Expr_fvar___override(v_fvarId_4214_);
v___x_4224_ = lp_aesop_Aesop_Script_TacticBuilder_casesOrObtain(v_goal_4215_, v___x_4223_, v___x_4216_, v___y_4218_, v___y_4219_, v___y_4220_, v___y_4221_);
return v___x_4224_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryCasesS___redArg___lam__1___boxed(lean_object* v_fvarId_4225_, lean_object* v_goal_4226_, lean_object* v___x_4227_, lean_object* v_x_4228_, lean_object* v___y_4229_, lean_object* v___y_4230_, lean_object* v___y_4231_, lean_object* v___y_4232_, lean_object* v___y_4233_){
_start:
{
lean_object* v_res_4234_; 
v_res_4234_ = lp_aesop_Aesop_tryCasesS___redArg___lam__1(v_fvarId_4225_, v_goal_4226_, v___x_4227_, v_x_4228_, v___y_4229_, v___y_4230_, v___y_4231_, v___y_4232_);
lean_dec(v___y_4232_);
lean_dec_ref(v___y_4231_);
lean_dec(v___y_4230_);
lean_dec_ref(v___y_4229_);
lean_dec_ref(v_x_4228_);
return v_res_4234_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_tryCasesS_spec__1(size_t v_sz_4235_, size_t v_i_4236_, lean_object* v_bs_4237_){
_start:
{
uint8_t v___x_4238_; 
v___x_4238_ = lean_usize_dec_lt(v_i_4236_, v_sz_4235_);
if (v___x_4238_ == 0)
{
return v_bs_4237_;
}
else
{
lean_object* v_v_4239_; lean_object* v___x_4240_; lean_object* v_bs_x27_4241_; lean_object* v___x_4242_; size_t v___x_4243_; size_t v___x_4244_; lean_object* v___x_4245_; 
v_v_4239_ = lean_array_uget(v_bs_4237_, v_i_4236_);
v___x_4240_ = lean_unsigned_to_nat(0u);
v_bs_x27_4241_ = lean_array_uset(v_bs_4237_, v_i_4236_, v___x_4240_);
v___x_4242_ = lp_aesop_Aesop_CtorNames_toAltVarNames(v_v_4239_);
v___x_4243_ = ((size_t)1ULL);
v___x_4244_ = lean_usize_add(v_i_4236_, v___x_4243_);
v___x_4245_ = lean_array_uset(v_bs_x27_4241_, v_i_4236_, v___x_4242_);
v_i_4236_ = v___x_4244_;
v_bs_4237_ = v___x_4245_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_tryCasesS_spec__1___boxed(lean_object* v_sz_4247_, lean_object* v_i_4248_, lean_object* v_bs_4249_){
_start:
{
size_t v_sz_boxed_4250_; size_t v_i_boxed_4251_; lean_object* v_res_4252_; 
v_sz_boxed_4250_ = lean_unbox_usize(v_sz_4247_);
lean_dec(v_sz_4247_);
v_i_boxed_4251_ = lean_unbox_usize(v_i_4248_);
lean_dec(v_i_4248_);
v_res_4252_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_tryCasesS_spec__1(v_sz_boxed_4250_, v_i_boxed_4251_, v_bs_4249_);
return v_res_4252_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryCasesS___redArg(lean_object* v_goal_4254_, lean_object* v_fvarId_4255_, lean_object* v_ctorNames_4256_, lean_object* v_a_4257_, lean_object* v_a_4258_, lean_object* v_a_4259_, lean_object* v_a_4260_, lean_object* v_a_4261_){
_start:
{
lean_object* v___x_4263_; 
lean_inc(v_goal_4254_);
v___x_4263_ = l_Lean_MVarId_getDecl(v_goal_4254_, v_a_4258_, v_a_4259_, v_a_4260_, v_a_4261_);
if (lean_obj_tag(v___x_4263_) == 0)
{
lean_object* v_a_4264_; lean_object* v_lctx_4265_; lean_object* v___f_4266_; lean_object* v___x_4267_; lean_object* v___f_4268_; size_t v_sz_4269_; size_t v___x_4270_; lean_object* v___x_4271_; uint8_t v___x_4272_; lean_object* v___x_4273_; lean_object* v___x_4274_; lean_object* v___x_4275_; lean_object* v___x_4276_; lean_object* v___x_4277_; 
v_a_4264_ = lean_ctor_get(v___x_4263_, 0);
lean_inc(v_a_4264_);
lean_dec_ref_known(v___x_4263_, 1);
v_lctx_4265_ = lean_ctor_get(v_a_4264_, 1);
lean_inc_ref(v_lctx_4265_);
lean_dec(v_a_4264_);
v___f_4266_ = ((lean_object*)(lp_aesop_Aesop_tryCasesS___redArg___closed__0));
v___x_4267_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_tryCasesS_getUnusedCtorNames(v_ctorNames_4256_, v_lctx_4265_);
lean_inc_ref(v___x_4267_);
lean_inc_n(v_goal_4254_, 2);
lean_inc(v_fvarId_4255_);
v___f_4268_ = lean_alloc_closure((void*)(lp_aesop_Aesop_tryCasesS___redArg___lam__1___boxed), 9, 3);
lean_closure_set(v___f_4268_, 0, v_fvarId_4255_);
lean_closure_set(v___f_4268_, 1, v_goal_4254_);
lean_closure_set(v___f_4268_, 2, v___x_4267_);
v_sz_4269_ = lean_array_size(v___x_4267_);
v___x_4270_ = ((size_t)0ULL);
v___x_4271_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_tryCasesS_spec__1(v_sz_4269_, v___x_4270_, v___x_4267_);
v___x_4272_ = 1;
v___x_4273_ = lean_box(0);
v___x_4274_ = lean_box(v___x_4272_);
v___x_4275_ = lean_alloc_closure((void*)(l_Lean_MVarId_cases___boxed), 10, 5);
lean_closure_set(v___x_4275_, 0, v_goal_4254_);
lean_closure_set(v___x_4275_, 1, v_fvarId_4255_);
lean_closure_set(v___x_4275_, 2, v___x_4271_);
lean_closure_set(v___x_4275_, 3, v___x_4274_);
lean_closure_set(v___x_4275_, 4, v___x_4273_);
v___x_4276_ = lean_alloc_closure((void*)(lp_aesop_Lean_observing_x3f___at___00Aesop_tryCasesS_spec__2___boxed), 7, 2);
lean_closure_set(v___x_4276_, 0, lean_box(0));
lean_closure_set(v___x_4276_, 1, v___x_4275_);
v___x_4277_ = lp_aesop_Aesop_withOptScriptStep___redArg(v_goal_4254_, v___f_4266_, v___f_4268_, v___x_4276_, v_a_4257_, v_a_4258_, v_a_4259_, v_a_4260_, v_a_4261_);
return v___x_4277_;
}
else
{
lean_object* v_a_4278_; lean_object* v___x_4280_; uint8_t v_isShared_4281_; uint8_t v_isSharedCheck_4285_; 
lean_dec(v_fvarId_4255_);
lean_dec(v_goal_4254_);
v_a_4278_ = lean_ctor_get(v___x_4263_, 0);
v_isSharedCheck_4285_ = !lean_is_exclusive(v___x_4263_);
if (v_isSharedCheck_4285_ == 0)
{
v___x_4280_ = v___x_4263_;
v_isShared_4281_ = v_isSharedCheck_4285_;
goto v_resetjp_4279_;
}
else
{
lean_inc(v_a_4278_);
lean_dec(v___x_4263_);
v___x_4280_ = lean_box(0);
v_isShared_4281_ = v_isSharedCheck_4285_;
goto v_resetjp_4279_;
}
v_resetjp_4279_:
{
lean_object* v___x_4283_; 
if (v_isShared_4281_ == 0)
{
v___x_4283_ = v___x_4280_;
goto v_reusejp_4282_;
}
else
{
lean_object* v_reuseFailAlloc_4284_; 
v_reuseFailAlloc_4284_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4284_, 0, v_a_4278_);
v___x_4283_ = v_reuseFailAlloc_4284_;
goto v_reusejp_4282_;
}
v_reusejp_4282_:
{
return v___x_4283_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryCasesS___redArg___boxed(lean_object* v_goal_4286_, lean_object* v_fvarId_4287_, lean_object* v_ctorNames_4288_, lean_object* v_a_4289_, lean_object* v_a_4290_, lean_object* v_a_4291_, lean_object* v_a_4292_, lean_object* v_a_4293_, lean_object* v_a_4294_){
_start:
{
lean_object* v_res_4295_; 
v_res_4295_ = lp_aesop_Aesop_tryCasesS___redArg(v_goal_4286_, v_fvarId_4287_, v_ctorNames_4288_, v_a_4289_, v_a_4290_, v_a_4291_, v_a_4292_, v_a_4293_);
lean_dec(v_a_4293_);
lean_dec_ref(v_a_4292_);
lean_dec(v_a_4291_);
lean_dec_ref(v_a_4290_);
lean_dec(v_a_4289_);
lean_dec_ref(v_ctorNames_4288_);
return v_res_4295_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryCasesS(lean_object* v_goal_4296_, lean_object* v_fvarId_4297_, lean_object* v_ctorNames_4298_, lean_object* v_a_4299_, lean_object* v_a_4300_, lean_object* v_a_4301_, lean_object* v_a_4302_, lean_object* v_a_4303_, lean_object* v_a_4304_){
_start:
{
lean_object* v___x_4306_; 
v___x_4306_ = lp_aesop_Aesop_tryCasesS___redArg(v_goal_4296_, v_fvarId_4297_, v_ctorNames_4298_, v_a_4299_, v_a_4301_, v_a_4302_, v_a_4303_, v_a_4304_);
return v___x_4306_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryCasesS___boxed(lean_object* v_goal_4307_, lean_object* v_fvarId_4308_, lean_object* v_ctorNames_4309_, lean_object* v_a_4310_, lean_object* v_a_4311_, lean_object* v_a_4312_, lean_object* v_a_4313_, lean_object* v_a_4314_, lean_object* v_a_4315_, lean_object* v_a_4316_){
_start:
{
lean_object* v_res_4317_; 
v_res_4317_ = lp_aesop_Aesop_tryCasesS(v_goal_4307_, v_fvarId_4308_, v_ctorNames_4309_, v_a_4310_, v_a_4311_, v_a_4312_, v_a_4313_, v_a_4314_, v_a_4315_);
lean_dec(v_a_4315_);
lean_dec_ref(v_a_4314_);
lean_dec(v_a_4313_);
lean_dec_ref(v_a_4312_);
lean_dec(v_a_4311_);
lean_dec(v_a_4310_);
lean_dec_ref(v_ctorNames_4309_);
return v_res_4317_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_renameInaccessibleFVarsS_tacticBuilder(lean_object* v_x_4318_, lean_object* v_a_4319_, lean_object* v_a_4320_, lean_object* v_a_4321_, lean_object* v_a_4322_){
_start:
{
lean_object* v_fst_4324_; lean_object* v_snd_4325_; lean_object* v___x_4326_; 
v_fst_4324_ = lean_ctor_get(v_x_4318_, 0);
lean_inc(v_fst_4324_);
v_snd_4325_ = lean_ctor_get(v_x_4318_, 1);
lean_inc(v_snd_4325_);
lean_dec_ref(v_x_4318_);
v___x_4326_ = lp_aesop_Aesop_Script_TacticBuilder_renameInaccessibleFVars(v_fst_4324_, v_snd_4325_, v_a_4319_, v_a_4320_, v_a_4321_, v_a_4322_);
return v___x_4326_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_renameInaccessibleFVarsS_tacticBuilder___boxed(lean_object* v_x_4327_, lean_object* v_a_4328_, lean_object* v_a_4329_, lean_object* v_a_4330_, lean_object* v_a_4331_, lean_object* v_a_4332_){
_start:
{
lean_object* v_res_4333_; 
v_res_4333_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_renameInaccessibleFVarsS_tacticBuilder(v_x_4327_, v_a_4328_, v_a_4329_, v_a_4330_, v_a_4331_);
lean_dec(v_a_4331_);
lean_dec_ref(v_a_4330_);
lean_dec(v_a_4329_);
lean_dec_ref(v_a_4328_);
return v_res_4333_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_renameInaccessibleFVarsS(lean_object* v_goal_4335_, lean_object* v_a_4336_, lean_object* v_a_4337_, lean_object* v_a_4338_, lean_object* v_a_4339_, lean_object* v_a_4340_, lean_object* v_a_4341_){
_start:
{
lean_object* v___f_4343_; lean_object* v___f_4344_; lean_object* v___x_4345_; lean_object* v___x_4346_; lean_object* v___x_4347_; 
v___f_4343_ = ((lean_object*)(lp_aesop_Aesop_assertHypothesisS___closed__0));
v___f_4344_ = ((lean_object*)(lp_aesop_Aesop_tryClearManyS___closed__0));
v___x_4345_ = ((lean_object*)(lp_aesop_Aesop_renameInaccessibleFVarsS___closed__0));
lean_inc(v_goal_4335_);
v___x_4346_ = lean_alloc_closure((void*)(lp_batteries_Lean_MVarId_renameInaccessibleFVars___boxed), 6, 1);
lean_closure_set(v___x_4346_, 0, v_goal_4335_);
v___x_4347_ = lp_aesop_Aesop_withScriptStep___redArg(v_goal_4335_, v___f_4343_, v___f_4344_, v___x_4345_, v___x_4346_, v_a_4336_, v_a_4337_, v_a_4338_, v_a_4339_, v_a_4340_, v_a_4341_);
return v___x_4347_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_renameInaccessibleFVarsS___boxed(lean_object* v_goal_4348_, lean_object* v_a_4349_, lean_object* v_a_4350_, lean_object* v_a_4351_, lean_object* v_a_4352_, lean_object* v_a_4353_, lean_object* v_a_4354_, lean_object* v_a_4355_){
_start:
{
lean_object* v_res_4356_; 
v_res_4356_ = lp_aesop_Aesop_renameInaccessibleFVarsS(v_goal_4348_, v_a_4349_, v_a_4350_, v_a_4351_, v_a_4352_, v_a_4353_, v_a_4354_);
lean_dec(v_a_4354_);
lean_dec_ref(v_a_4353_);
lean_dec(v_a_4352_);
lean_dec_ref(v_a_4351_);
lean_dec(v_a_4350_);
lean_dec(v_a_4349_);
return v_res_4356_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00Aesop_unfoldManyTargetS_spec__0___redArg(lean_object* v_step_4357_, lean_object* v___y_4358_){
_start:
{
lean_object* v___x_4360_; lean_object* v___x_4361_; lean_object* v___x_4362_; lean_object* v___x_4363_; lean_object* v___x_4364_; 
v___x_4360_ = lean_st_ref_take(v___y_4358_);
v___x_4361_ = lean_array_push(v___x_4360_, v_step_4357_);
v___x_4362_ = lean_st_ref_set(v___y_4358_, v___x_4361_);
v___x_4363_ = lean_box(0);
v___x_4364_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4364_, 0, v___x_4363_);
return v___x_4364_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00Aesop_unfoldManyTargetS_spec__0___redArg___boxed(lean_object* v_step_4365_, lean_object* v___y_4366_, lean_object* v___y_4367_){
_start:
{
lean_object* v_res_4368_; 
v_res_4368_ = lp_aesop_Aesop_recordScriptStep___at___00Aesop_unfoldManyTargetS_spec__0___redArg(v_step_4365_, v___y_4366_);
lean_dec(v___y_4366_);
return v_res_4368_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00Aesop_unfoldManyTargetS_spec__0(lean_object* v_step_4369_, lean_object* v___y_4370_, lean_object* v___y_4371_, lean_object* v___y_4372_, lean_object* v___y_4373_, lean_object* v___y_4374_, lean_object* v___y_4375_){
_start:
{
lean_object* v___x_4377_; 
v___x_4377_ = lp_aesop_Aesop_recordScriptStep___at___00Aesop_unfoldManyTargetS_spec__0___redArg(v_step_4369_, v___y_4370_);
return v___x_4377_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptStep___at___00Aesop_unfoldManyTargetS_spec__0___boxed(lean_object* v_step_4378_, lean_object* v___y_4379_, lean_object* v___y_4380_, lean_object* v___y_4381_, lean_object* v___y_4382_, lean_object* v___y_4383_, lean_object* v___y_4384_, lean_object* v___y_4385_){
_start:
{
lean_object* v_res_4386_; 
v_res_4386_ = lp_aesop_Aesop_recordScriptStep___at___00Aesop_unfoldManyTargetS_spec__0(v_step_4378_, v___y_4379_, v___y_4380_, v___y_4381_, v___y_4382_, v___y_4383_, v___y_4384_);
lean_dec(v___y_4384_);
lean_dec_ref(v___y_4383_);
lean_dec(v___y_4382_);
lean_dec_ref(v___y_4381_);
lean_dec(v___y_4380_);
lean_dec(v___y_4379_);
return v_res_4386_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyTargetS(lean_object* v_unfold_x3f_4387_, lean_object* v_goal_4388_, lean_object* v_a_4389_, lean_object* v_a_4390_, lean_object* v_a_4391_, lean_object* v_a_4392_, lean_object* v_a_4393_, lean_object* v_a_4394_){
_start:
{
lean_object* v___x_4396_; 
v___x_4396_ = l_Lean_Meta_saveState___redArg(v_a_4392_, v_a_4394_);
if (lean_obj_tag(v___x_4396_) == 0)
{
lean_object* v_a_4397_; lean_object* v___x_4398_; 
v_a_4397_ = lean_ctor_get(v___x_4396_, 0);
lean_inc(v_a_4397_);
lean_dec_ref_known(v___x_4396_, 1);
lean_inc(v_goal_4388_);
v___x_4398_ = lp_aesop_Aesop_unfoldManyTarget(v_unfold_x3f_4387_, v_goal_4388_, v_a_4391_, v_a_4392_, v_a_4393_, v_a_4394_);
if (lean_obj_tag(v___x_4398_) == 0)
{
lean_object* v_a_4399_; lean_object* v___x_4401_; uint8_t v_isShared_4402_; uint8_t v_isSharedCheck_4443_; 
v_a_4399_ = lean_ctor_get(v___x_4398_, 0);
v_isSharedCheck_4443_ = !lean_is_exclusive(v___x_4398_);
if (v_isSharedCheck_4443_ == 0)
{
v___x_4401_ = v___x_4398_;
v_isShared_4402_ = v_isSharedCheck_4443_;
goto v_resetjp_4400_;
}
else
{
lean_inc(v_a_4399_);
lean_dec(v___x_4398_);
v___x_4401_ = lean_box(0);
v_isShared_4402_ = v_isSharedCheck_4443_;
goto v_resetjp_4400_;
}
v_resetjp_4400_:
{
if (lean_obj_tag(v_a_4399_) == 1)
{
lean_object* v_val_4403_; lean_object* v_fst_4404_; lean_object* v_snd_4405_; lean_object* v___x_4406_; 
lean_del_object(v___x_4401_);
v_val_4403_ = lean_ctor_get(v_a_4399_, 0);
v_fst_4404_ = lean_ctor_get(v_val_4403_, 0);
v_snd_4405_ = lean_ctor_get(v_val_4403_, 1);
v___x_4406_ = l_Lean_Meta_saveState___redArg(v_a_4392_, v_a_4394_);
if (lean_obj_tag(v___x_4406_) == 0)
{
lean_object* v_a_4407_; uint8_t v___x_4408_; lean_object* v___x_4409_; lean_object* v___x_4410_; uint8_t v___x_4411_; lean_object* v___x_4412_; lean_object* v___x_4413_; lean_object* v___x_4414_; lean_object* v___x_4415_; lean_object* v___x_4416_; lean_object* v___x_4417_; lean_object* v___x_4418_; lean_object* v___x_4419_; lean_object* v___x_4420_; lean_object* v___x_4421_; lean_object* v___x_4422_; lean_object* v___x_4424_; uint8_t v_isShared_4425_; uint8_t v_isSharedCheck_4429_; 
v_a_4407_ = lean_ctor_get(v___x_4406_, 0);
lean_inc(v_a_4407_);
lean_dec_ref_known(v___x_4406_, 1);
v___x_4408_ = 0;
v___x_4409_ = lean_box(v___x_4408_);
lean_inc_n(v_snd_4405_, 2);
v___x_4410_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticBuilder_unfold___boxed), 7, 2);
lean_closure_set(v___x_4410_, 0, v_snd_4405_);
lean_closure_set(v___x_4410_, 1, v___x_4409_);
v___x_4411_ = 1;
v___x_4412_ = lean_box(v___x_4411_);
v___x_4413_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticBuilder_unfold___boxed), 7, 2);
lean_closure_set(v___x_4413_, 0, v_snd_4405_);
lean_closure_set(v___x_4413_, 1, v___x_4412_);
v___x_4414_ = lean_unsigned_to_nat(2u);
v___x_4415_ = lean_mk_empty_array_with_capacity(v___x_4414_);
v___x_4416_ = lean_array_push(v___x_4415_, v___x_4410_);
v___x_4417_ = lean_array_push(v___x_4416_, v___x_4413_);
v___x_4418_ = lean_unsigned_to_nat(1u);
v___x_4419_ = lean_mk_empty_array_with_capacity(v___x_4418_);
lean_inc(v_fst_4404_);
v___x_4420_ = lean_array_push(v___x_4419_, v_fst_4404_);
v___x_4421_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_4421_, 0, v_a_4397_);
lean_ctor_set(v___x_4421_, 1, v_goal_4388_);
lean_ctor_set(v___x_4421_, 2, v___x_4417_);
lean_ctor_set(v___x_4421_, 3, v_a_4407_);
lean_ctor_set(v___x_4421_, 4, v___x_4420_);
v___x_4422_ = lp_aesop_Aesop_recordScriptStep___at___00Aesop_unfoldManyTargetS_spec__0___redArg(v___x_4421_, v_a_4389_);
v_isSharedCheck_4429_ = !lean_is_exclusive(v___x_4422_);
if (v_isSharedCheck_4429_ == 0)
{
lean_object* v_unused_4430_; 
v_unused_4430_ = lean_ctor_get(v___x_4422_, 0);
lean_dec(v_unused_4430_);
v___x_4424_ = v___x_4422_;
v_isShared_4425_ = v_isSharedCheck_4429_;
goto v_resetjp_4423_;
}
else
{
lean_dec(v___x_4422_);
v___x_4424_ = lean_box(0);
v_isShared_4425_ = v_isSharedCheck_4429_;
goto v_resetjp_4423_;
}
v_resetjp_4423_:
{
lean_object* v___x_4427_; 
if (v_isShared_4425_ == 0)
{
lean_ctor_set(v___x_4424_, 0, v_a_4399_);
v___x_4427_ = v___x_4424_;
goto v_reusejp_4426_;
}
else
{
lean_object* v_reuseFailAlloc_4428_; 
v_reuseFailAlloc_4428_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4428_, 0, v_a_4399_);
v___x_4427_ = v_reuseFailAlloc_4428_;
goto v_reusejp_4426_;
}
v_reusejp_4426_:
{
return v___x_4427_;
}
}
}
else
{
lean_object* v_a_4431_; lean_object* v___x_4433_; uint8_t v_isShared_4434_; uint8_t v_isSharedCheck_4438_; 
lean_dec_ref_known(v_a_4399_, 1);
lean_dec(v_a_4397_);
lean_dec(v_goal_4388_);
v_a_4431_ = lean_ctor_get(v___x_4406_, 0);
v_isSharedCheck_4438_ = !lean_is_exclusive(v___x_4406_);
if (v_isSharedCheck_4438_ == 0)
{
v___x_4433_ = v___x_4406_;
v_isShared_4434_ = v_isSharedCheck_4438_;
goto v_resetjp_4432_;
}
else
{
lean_inc(v_a_4431_);
lean_dec(v___x_4406_);
v___x_4433_ = lean_box(0);
v_isShared_4434_ = v_isSharedCheck_4438_;
goto v_resetjp_4432_;
}
v_resetjp_4432_:
{
lean_object* v___x_4436_; 
if (v_isShared_4434_ == 0)
{
v___x_4436_ = v___x_4433_;
goto v_reusejp_4435_;
}
else
{
lean_object* v_reuseFailAlloc_4437_; 
v_reuseFailAlloc_4437_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4437_, 0, v_a_4431_);
v___x_4436_ = v_reuseFailAlloc_4437_;
goto v_reusejp_4435_;
}
v_reusejp_4435_:
{
return v___x_4436_;
}
}
}
}
else
{
lean_object* v___x_4439_; lean_object* v___x_4441_; 
lean_dec(v_a_4399_);
lean_dec(v_a_4397_);
lean_dec(v_goal_4388_);
v___x_4439_ = lean_box(0);
if (v_isShared_4402_ == 0)
{
lean_ctor_set(v___x_4401_, 0, v___x_4439_);
v___x_4441_ = v___x_4401_;
goto v_reusejp_4440_;
}
else
{
lean_object* v_reuseFailAlloc_4442_; 
v_reuseFailAlloc_4442_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4442_, 0, v___x_4439_);
v___x_4441_ = v_reuseFailAlloc_4442_;
goto v_reusejp_4440_;
}
v_reusejp_4440_:
{
return v___x_4441_;
}
}
}
}
else
{
lean_dec(v_a_4397_);
lean_dec(v_goal_4388_);
return v___x_4398_;
}
}
else
{
lean_object* v_a_4444_; lean_object* v___x_4446_; uint8_t v_isShared_4447_; uint8_t v_isSharedCheck_4451_; 
lean_dec(v_goal_4388_);
lean_dec_ref(v_unfold_x3f_4387_);
v_a_4444_ = lean_ctor_get(v___x_4396_, 0);
v_isSharedCheck_4451_ = !lean_is_exclusive(v___x_4396_);
if (v_isSharedCheck_4451_ == 0)
{
v___x_4446_ = v___x_4396_;
v_isShared_4447_ = v_isSharedCheck_4451_;
goto v_resetjp_4445_;
}
else
{
lean_inc(v_a_4444_);
lean_dec(v___x_4396_);
v___x_4446_ = lean_box(0);
v_isShared_4447_ = v_isSharedCheck_4451_;
goto v_resetjp_4445_;
}
v_resetjp_4445_:
{
lean_object* v___x_4449_; 
if (v_isShared_4447_ == 0)
{
v___x_4449_ = v___x_4446_;
goto v_reusejp_4448_;
}
else
{
lean_object* v_reuseFailAlloc_4450_; 
v_reuseFailAlloc_4450_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4450_, 0, v_a_4444_);
v___x_4449_ = v_reuseFailAlloc_4450_;
goto v_reusejp_4448_;
}
v_reusejp_4448_:
{
return v___x_4449_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyTargetS___boxed(lean_object* v_unfold_x3f_4452_, lean_object* v_goal_4453_, lean_object* v_a_4454_, lean_object* v_a_4455_, lean_object* v_a_4456_, lean_object* v_a_4457_, lean_object* v_a_4458_, lean_object* v_a_4459_, lean_object* v_a_4460_){
_start:
{
lean_object* v_res_4461_; 
v_res_4461_ = lp_aesop_Aesop_unfoldManyTargetS(v_unfold_x3f_4452_, v_goal_4453_, v_a_4454_, v_a_4455_, v_a_4456_, v_a_4457_, v_a_4458_, v_a_4459_);
lean_dec(v_a_4459_);
lean_dec_ref(v_a_4458_);
lean_dec(v_a_4457_);
lean_dec_ref(v_a_4456_);
lean_dec(v_a_4455_);
lean_dec(v_a_4454_);
return v_res_4461_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyAtS___redArg(lean_object* v_unfold_x3f_4462_, lean_object* v_goal_4463_, lean_object* v_fvarId_4464_, lean_object* v_a_4465_, lean_object* v_a_4466_, lean_object* v_a_4467_, lean_object* v_a_4468_, lean_object* v_a_4469_){
_start:
{
lean_object* v___x_4471_; 
v___x_4471_ = l_Lean_Meta_saveState___redArg(v_a_4467_, v_a_4469_);
if (lean_obj_tag(v___x_4471_) == 0)
{
lean_object* v_a_4472_; lean_object* v___x_4473_; 
v_a_4472_ = lean_ctor_get(v___x_4471_, 0);
lean_inc(v_a_4472_);
lean_dec_ref_known(v___x_4471_, 1);
lean_inc(v_fvarId_4464_);
lean_inc(v_goal_4463_);
v___x_4473_ = lp_aesop_Aesop_unfoldManyAt(v_unfold_x3f_4462_, v_goal_4463_, v_fvarId_4464_, v_a_4466_, v_a_4467_, v_a_4468_, v_a_4469_);
if (lean_obj_tag(v___x_4473_) == 0)
{
lean_object* v_a_4474_; lean_object* v___x_4476_; uint8_t v_isShared_4477_; uint8_t v_isSharedCheck_4518_; 
v_a_4474_ = lean_ctor_get(v___x_4473_, 0);
v_isSharedCheck_4518_ = !lean_is_exclusive(v___x_4473_);
if (v_isSharedCheck_4518_ == 0)
{
v___x_4476_ = v___x_4473_;
v_isShared_4477_ = v_isSharedCheck_4518_;
goto v_resetjp_4475_;
}
else
{
lean_inc(v_a_4474_);
lean_dec(v___x_4473_);
v___x_4476_ = lean_box(0);
v_isShared_4477_ = v_isSharedCheck_4518_;
goto v_resetjp_4475_;
}
v_resetjp_4475_:
{
if (lean_obj_tag(v_a_4474_) == 1)
{
lean_object* v_val_4478_; lean_object* v_fst_4479_; lean_object* v_snd_4480_; lean_object* v___x_4481_; 
lean_del_object(v___x_4476_);
v_val_4478_ = lean_ctor_get(v_a_4474_, 0);
v_fst_4479_ = lean_ctor_get(v_val_4478_, 0);
v_snd_4480_ = lean_ctor_get(v_val_4478_, 1);
v___x_4481_ = l_Lean_Meta_saveState___redArg(v_a_4467_, v_a_4469_);
if (lean_obj_tag(v___x_4481_) == 0)
{
lean_object* v_a_4482_; uint8_t v___x_4483_; lean_object* v___x_4484_; lean_object* v___x_4485_; uint8_t v___x_4486_; lean_object* v___x_4487_; lean_object* v___x_4488_; lean_object* v___x_4489_; lean_object* v___x_4490_; lean_object* v___x_4491_; lean_object* v___x_4492_; lean_object* v___x_4493_; lean_object* v___x_4494_; lean_object* v___x_4495_; lean_object* v___x_4496_; lean_object* v___x_4497_; lean_object* v___x_4499_; uint8_t v_isShared_4500_; uint8_t v_isSharedCheck_4504_; 
v_a_4482_ = lean_ctor_get(v___x_4481_, 0);
lean_inc(v_a_4482_);
lean_dec_ref_known(v___x_4481_, 1);
v___x_4483_ = 0;
v___x_4484_ = lean_box(v___x_4483_);
lean_inc_n(v_snd_4480_, 2);
lean_inc(v_fvarId_4464_);
lean_inc_n(v_goal_4463_, 2);
v___x_4485_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___boxed), 9, 4);
lean_closure_set(v___x_4485_, 0, v_goal_4463_);
lean_closure_set(v___x_4485_, 1, v_fvarId_4464_);
lean_closure_set(v___x_4485_, 2, v_snd_4480_);
lean_closure_set(v___x_4485_, 3, v___x_4484_);
v___x_4486_ = 1;
v___x_4487_ = lean_box(v___x_4486_);
v___x_4488_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticBuilder_unfoldAt___boxed), 9, 4);
lean_closure_set(v___x_4488_, 0, v_goal_4463_);
lean_closure_set(v___x_4488_, 1, v_fvarId_4464_);
lean_closure_set(v___x_4488_, 2, v_snd_4480_);
lean_closure_set(v___x_4488_, 3, v___x_4487_);
v___x_4489_ = lean_unsigned_to_nat(2u);
v___x_4490_ = lean_mk_empty_array_with_capacity(v___x_4489_);
v___x_4491_ = lean_array_push(v___x_4490_, v___x_4485_);
v___x_4492_ = lean_array_push(v___x_4491_, v___x_4488_);
v___x_4493_ = lean_unsigned_to_nat(1u);
v___x_4494_ = lean_mk_empty_array_with_capacity(v___x_4493_);
lean_inc(v_fst_4479_);
v___x_4495_ = lean_array_push(v___x_4494_, v_fst_4479_);
v___x_4496_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_4496_, 0, v_a_4472_);
lean_ctor_set(v___x_4496_, 1, v_goal_4463_);
lean_ctor_set(v___x_4496_, 2, v___x_4492_);
lean_ctor_set(v___x_4496_, 3, v_a_4482_);
lean_ctor_set(v___x_4496_, 4, v___x_4495_);
v___x_4497_ = lp_aesop_Aesop_recordScriptStep___at___00Aesop_unfoldManyTargetS_spec__0___redArg(v___x_4496_, v_a_4465_);
v_isSharedCheck_4504_ = !lean_is_exclusive(v___x_4497_);
if (v_isSharedCheck_4504_ == 0)
{
lean_object* v_unused_4505_; 
v_unused_4505_ = lean_ctor_get(v___x_4497_, 0);
lean_dec(v_unused_4505_);
v___x_4499_ = v___x_4497_;
v_isShared_4500_ = v_isSharedCheck_4504_;
goto v_resetjp_4498_;
}
else
{
lean_dec(v___x_4497_);
v___x_4499_ = lean_box(0);
v_isShared_4500_ = v_isSharedCheck_4504_;
goto v_resetjp_4498_;
}
v_resetjp_4498_:
{
lean_object* v___x_4502_; 
if (v_isShared_4500_ == 0)
{
lean_ctor_set(v___x_4499_, 0, v_a_4474_);
v___x_4502_ = v___x_4499_;
goto v_reusejp_4501_;
}
else
{
lean_object* v_reuseFailAlloc_4503_; 
v_reuseFailAlloc_4503_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4503_, 0, v_a_4474_);
v___x_4502_ = v_reuseFailAlloc_4503_;
goto v_reusejp_4501_;
}
v_reusejp_4501_:
{
return v___x_4502_;
}
}
}
else
{
lean_object* v_a_4506_; lean_object* v___x_4508_; uint8_t v_isShared_4509_; uint8_t v_isSharedCheck_4513_; 
lean_dec_ref_known(v_a_4474_, 1);
lean_dec(v_a_4472_);
lean_dec(v_fvarId_4464_);
lean_dec(v_goal_4463_);
v_a_4506_ = lean_ctor_get(v___x_4481_, 0);
v_isSharedCheck_4513_ = !lean_is_exclusive(v___x_4481_);
if (v_isSharedCheck_4513_ == 0)
{
v___x_4508_ = v___x_4481_;
v_isShared_4509_ = v_isSharedCheck_4513_;
goto v_resetjp_4507_;
}
else
{
lean_inc(v_a_4506_);
lean_dec(v___x_4481_);
v___x_4508_ = lean_box(0);
v_isShared_4509_ = v_isSharedCheck_4513_;
goto v_resetjp_4507_;
}
v_resetjp_4507_:
{
lean_object* v___x_4511_; 
if (v_isShared_4509_ == 0)
{
v___x_4511_ = v___x_4508_;
goto v_reusejp_4510_;
}
else
{
lean_object* v_reuseFailAlloc_4512_; 
v_reuseFailAlloc_4512_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4512_, 0, v_a_4506_);
v___x_4511_ = v_reuseFailAlloc_4512_;
goto v_reusejp_4510_;
}
v_reusejp_4510_:
{
return v___x_4511_;
}
}
}
}
else
{
lean_object* v___x_4514_; lean_object* v___x_4516_; 
lean_dec(v_a_4474_);
lean_dec(v_a_4472_);
lean_dec(v_fvarId_4464_);
lean_dec(v_goal_4463_);
v___x_4514_ = lean_box(0);
if (v_isShared_4477_ == 0)
{
lean_ctor_set(v___x_4476_, 0, v___x_4514_);
v___x_4516_ = v___x_4476_;
goto v_reusejp_4515_;
}
else
{
lean_object* v_reuseFailAlloc_4517_; 
v_reuseFailAlloc_4517_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4517_, 0, v___x_4514_);
v___x_4516_ = v_reuseFailAlloc_4517_;
goto v_reusejp_4515_;
}
v_reusejp_4515_:
{
return v___x_4516_;
}
}
}
}
else
{
lean_dec(v_a_4472_);
lean_dec(v_fvarId_4464_);
lean_dec(v_goal_4463_);
return v___x_4473_;
}
}
else
{
lean_object* v_a_4519_; lean_object* v___x_4521_; uint8_t v_isShared_4522_; uint8_t v_isSharedCheck_4526_; 
lean_dec(v_fvarId_4464_);
lean_dec(v_goal_4463_);
lean_dec_ref(v_unfold_x3f_4462_);
v_a_4519_ = lean_ctor_get(v___x_4471_, 0);
v_isSharedCheck_4526_ = !lean_is_exclusive(v___x_4471_);
if (v_isSharedCheck_4526_ == 0)
{
v___x_4521_ = v___x_4471_;
v_isShared_4522_ = v_isSharedCheck_4526_;
goto v_resetjp_4520_;
}
else
{
lean_inc(v_a_4519_);
lean_dec(v___x_4471_);
v___x_4521_ = lean_box(0);
v_isShared_4522_ = v_isSharedCheck_4526_;
goto v_resetjp_4520_;
}
v_resetjp_4520_:
{
lean_object* v___x_4524_; 
if (v_isShared_4522_ == 0)
{
v___x_4524_ = v___x_4521_;
goto v_reusejp_4523_;
}
else
{
lean_object* v_reuseFailAlloc_4525_; 
v_reuseFailAlloc_4525_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4525_, 0, v_a_4519_);
v___x_4524_ = v_reuseFailAlloc_4525_;
goto v_reusejp_4523_;
}
v_reusejp_4523_:
{
return v___x_4524_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyAtS___redArg___boxed(lean_object* v_unfold_x3f_4527_, lean_object* v_goal_4528_, lean_object* v_fvarId_4529_, lean_object* v_a_4530_, lean_object* v_a_4531_, lean_object* v_a_4532_, lean_object* v_a_4533_, lean_object* v_a_4534_, lean_object* v_a_4535_){
_start:
{
lean_object* v_res_4536_; 
v_res_4536_ = lp_aesop_Aesop_unfoldManyAtS___redArg(v_unfold_x3f_4527_, v_goal_4528_, v_fvarId_4529_, v_a_4530_, v_a_4531_, v_a_4532_, v_a_4533_, v_a_4534_);
lean_dec(v_a_4534_);
lean_dec_ref(v_a_4533_);
lean_dec(v_a_4532_);
lean_dec_ref(v_a_4531_);
lean_dec(v_a_4530_);
return v_res_4536_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyAtS(lean_object* v_unfold_x3f_4537_, lean_object* v_goal_4538_, lean_object* v_fvarId_4539_, lean_object* v_a_4540_, lean_object* v_a_4541_, lean_object* v_a_4542_, lean_object* v_a_4543_, lean_object* v_a_4544_, lean_object* v_a_4545_){
_start:
{
lean_object* v___x_4547_; 
v___x_4547_ = lp_aesop_Aesop_unfoldManyAtS___redArg(v_unfold_x3f_4537_, v_goal_4538_, v_fvarId_4539_, v_a_4540_, v_a_4542_, v_a_4543_, v_a_4544_, v_a_4545_);
return v___x_4547_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyAtS___boxed(lean_object* v_unfold_x3f_4548_, lean_object* v_goal_4549_, lean_object* v_fvarId_4550_, lean_object* v_a_4551_, lean_object* v_a_4552_, lean_object* v_a_4553_, lean_object* v_a_4554_, lean_object* v_a_4555_, lean_object* v_a_4556_, lean_object* v_a_4557_){
_start:
{
lean_object* v_res_4558_; 
v_res_4558_ = lp_aesop_Aesop_unfoldManyAtS(v_unfold_x3f_4548_, v_goal_4549_, v_fvarId_4550_, v_a_4551_, v_a_4552_, v_a_4553_, v_a_4554_, v_a_4555_, v_a_4556_);
lean_dec(v_a_4556_);
lean_dec_ref(v_a_4555_);
lean_dec(v_a_4554_);
lean_dec_ref(v_a_4553_);
lean_dec(v_a_4552_);
lean_dec(v_a_4551_);
return v_res_4558_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyStarS_spec__1___redArg___lam__0(lean_object* v_x_4559_, lean_object* v___y_4560_, lean_object* v___y_4561_, lean_object* v___y_4562_, lean_object* v___y_4563_, lean_object* v___y_4564_, lean_object* v___y_4565_){
_start:
{
lean_object* v___x_4567_; 
lean_inc(v___y_4561_);
lean_inc(v___y_4560_);
v___x_4567_ = lean_apply_7(v_x_4559_, v___y_4560_, v___y_4561_, v___y_4562_, v___y_4563_, v___y_4564_, v___y_4565_, lean_box(0));
return v___x_4567_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyStarS_spec__1___redArg___lam__0___boxed(lean_object* v_x_4568_, lean_object* v___y_4569_, lean_object* v___y_4570_, lean_object* v___y_4571_, lean_object* v___y_4572_, lean_object* v___y_4573_, lean_object* v___y_4574_, lean_object* v___y_4575_){
_start:
{
lean_object* v_res_4576_; 
v_res_4576_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyStarS_spec__1___redArg___lam__0(v_x_4568_, v___y_4569_, v___y_4570_, v___y_4571_, v___y_4572_, v___y_4573_, v___y_4574_);
lean_dec(v___y_4570_);
lean_dec(v___y_4569_);
return v_res_4576_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyStarS_spec__1___redArg(lean_object* v_mvarId_4577_, lean_object* v_x_4578_, lean_object* v___y_4579_, lean_object* v___y_4580_, lean_object* v___y_4581_, lean_object* v___y_4582_, lean_object* v___y_4583_, lean_object* v___y_4584_){
_start:
{
lean_object* v___f_4586_; lean_object* v___x_4587_; 
lean_inc(v___y_4580_);
lean_inc(v___y_4579_);
v___f_4586_ = lean_alloc_closure((void*)(lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyStarS_spec__1___redArg___lam__0___boxed), 8, 3);
lean_closure_set(v___f_4586_, 0, v_x_4578_);
lean_closure_set(v___f_4586_, 1, v___y_4579_);
lean_closure_set(v___f_4586_, 2, v___y_4580_);
v___x_4587_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_4577_, v___f_4586_, v___y_4581_, v___y_4582_, v___y_4583_, v___y_4584_);
if (lean_obj_tag(v___x_4587_) == 0)
{
return v___x_4587_;
}
else
{
lean_object* v_a_4588_; lean_object* v___x_4590_; uint8_t v_isShared_4591_; uint8_t v_isSharedCheck_4595_; 
v_a_4588_ = lean_ctor_get(v___x_4587_, 0);
v_isSharedCheck_4595_ = !lean_is_exclusive(v___x_4587_);
if (v_isSharedCheck_4595_ == 0)
{
v___x_4590_ = v___x_4587_;
v_isShared_4591_ = v_isSharedCheck_4595_;
goto v_resetjp_4589_;
}
else
{
lean_inc(v_a_4588_);
lean_dec(v___x_4587_);
v___x_4590_ = lean_box(0);
v_isShared_4591_ = v_isSharedCheck_4595_;
goto v_resetjp_4589_;
}
v_resetjp_4589_:
{
lean_object* v___x_4593_; 
if (v_isShared_4591_ == 0)
{
v___x_4593_ = v___x_4590_;
goto v_reusejp_4592_;
}
else
{
lean_object* v_reuseFailAlloc_4594_; 
v_reuseFailAlloc_4594_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4594_, 0, v_a_4588_);
v___x_4593_ = v_reuseFailAlloc_4594_;
goto v_reusejp_4592_;
}
v_reusejp_4592_:
{
return v___x_4593_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyStarS_spec__1___redArg___boxed(lean_object* v_mvarId_4596_, lean_object* v_x_4597_, lean_object* v___y_4598_, lean_object* v___y_4599_, lean_object* v___y_4600_, lean_object* v___y_4601_, lean_object* v___y_4602_, lean_object* v___y_4603_, lean_object* v___y_4604_){
_start:
{
lean_object* v_res_4605_; 
v_res_4605_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyStarS_spec__1___redArg(v_mvarId_4596_, v_x_4597_, v___y_4598_, v___y_4599_, v___y_4600_, v___y_4601_, v___y_4602_, v___y_4603_);
lean_dec(v___y_4603_);
lean_dec_ref(v___y_4602_);
lean_dec(v___y_4601_);
lean_dec_ref(v___y_4600_);
lean_dec(v___y_4599_);
lean_dec(v___y_4598_);
return v_res_4605_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyStarS_spec__1(lean_object* v_00_u03b1_4606_, lean_object* v_mvarId_4607_, lean_object* v_x_4608_, lean_object* v___y_4609_, lean_object* v___y_4610_, lean_object* v___y_4611_, lean_object* v___y_4612_, lean_object* v___y_4613_, lean_object* v___y_4614_){
_start:
{
lean_object* v___x_4616_; 
v___x_4616_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyStarS_spec__1___redArg(v_mvarId_4607_, v_x_4608_, v___y_4609_, v___y_4610_, v___y_4611_, v___y_4612_, v___y_4613_, v___y_4614_);
return v___x_4616_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyStarS_spec__1___boxed(lean_object* v_00_u03b1_4617_, lean_object* v_mvarId_4618_, lean_object* v_x_4619_, lean_object* v___y_4620_, lean_object* v___y_4621_, lean_object* v___y_4622_, lean_object* v___y_4623_, lean_object* v___y_4624_, lean_object* v___y_4625_, lean_object* v___y_4626_){
_start:
{
lean_object* v_res_4627_; 
v_res_4627_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyStarS_spec__1(v_00_u03b1_4617_, v_mvarId_4618_, v_x_4619_, v___y_4620_, v___y_4621_, v___y_4622_, v___y_4623_, v___y_4624_, v___y_4625_);
lean_dec(v___y_4625_);
lean_dec_ref(v___y_4624_);
lean_dec(v___y_4623_);
lean_dec_ref(v___y_4622_);
lean_dec(v___y_4621_);
lean_dec(v___y_4620_);
return v_res_4627_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0_spec__3_spec__4___redArg(lean_object* v_unfold_x3f_4628_, lean_object* v_as_4629_, size_t v_sz_4630_, size_t v_i_4631_, lean_object* v_b_4632_, lean_object* v___y_4633_, lean_object* v___y_4634_, lean_object* v___y_4635_, lean_object* v___y_4636_, lean_object* v___y_4637_){
_start:
{
uint8_t v___x_4639_; 
v___x_4639_ = lean_usize_dec_lt(v_i_4631_, v_sz_4630_);
if (v___x_4639_ == 0)
{
lean_object* v___x_4640_; 
lean_dec_ref(v_unfold_x3f_4628_);
v___x_4640_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4640_, 0, v_b_4632_);
return v___x_4640_;
}
else
{
lean_object* v_snd_4641_; lean_object* v___x_4643_; uint8_t v_isShared_4644_; uint8_t v_isSharedCheck_4670_; 
v_snd_4641_ = lean_ctor_get(v_b_4632_, 1);
v_isSharedCheck_4670_ = !lean_is_exclusive(v_b_4632_);
if (v_isSharedCheck_4670_ == 0)
{
lean_object* v_unused_4671_; 
v_unused_4671_ = lean_ctor_get(v_b_4632_, 0);
lean_dec(v_unused_4671_);
v___x_4643_ = v_b_4632_;
v_isShared_4644_ = v_isSharedCheck_4670_;
goto v_resetjp_4642_;
}
else
{
lean_inc(v_snd_4641_);
lean_dec(v_b_4632_);
v___x_4643_ = lean_box(0);
v_isShared_4644_ = v_isSharedCheck_4670_;
goto v_resetjp_4642_;
}
v_resetjp_4642_:
{
lean_object* v___x_4645_; lean_object* v_a_4647_; lean_object* v_a_4654_; 
v___x_4645_ = lean_box(0);
v_a_4654_ = lean_array_uget_borrowed(v_as_4629_, v_i_4631_);
if (lean_obj_tag(v_a_4654_) == 0)
{
v_a_4647_ = v_snd_4641_;
goto v___jp_4646_;
}
else
{
lean_object* v_val_4655_; uint8_t v___x_4656_; 
v_val_4655_ = lean_ctor_get(v_a_4654_, 0);
v___x_4656_ = l_Lean_LocalDecl_isImplementationDetail(v_val_4655_);
if (v___x_4656_ == 0)
{
lean_object* v___x_4657_; lean_object* v___x_4658_; 
v___x_4657_ = l_Lean_LocalDecl_fvarId(v_val_4655_);
lean_inc(v_snd_4641_);
lean_inc_ref(v_unfold_x3f_4628_);
v___x_4658_ = lp_aesop_Aesop_unfoldManyAtS___redArg(v_unfold_x3f_4628_, v_snd_4641_, v___x_4657_, v___y_4633_, v___y_4634_, v___y_4635_, v___y_4636_, v___y_4637_);
if (lean_obj_tag(v___x_4658_) == 0)
{
lean_object* v_a_4659_; 
v_a_4659_ = lean_ctor_get(v___x_4658_, 0);
lean_inc(v_a_4659_);
lean_dec_ref_known(v___x_4658_, 1);
if (lean_obj_tag(v_a_4659_) == 1)
{
lean_object* v_val_4660_; lean_object* v_fst_4661_; 
lean_dec(v_snd_4641_);
v_val_4660_ = lean_ctor_get(v_a_4659_, 0);
lean_inc(v_val_4660_);
lean_dec_ref_known(v_a_4659_, 1);
v_fst_4661_ = lean_ctor_get(v_val_4660_, 0);
lean_inc(v_fst_4661_);
lean_dec(v_val_4660_);
v_a_4647_ = v_fst_4661_;
goto v___jp_4646_;
}
else
{
lean_dec(v_a_4659_);
v_a_4647_ = v_snd_4641_;
goto v___jp_4646_;
}
}
else
{
lean_object* v_a_4662_; lean_object* v___x_4664_; uint8_t v_isShared_4665_; uint8_t v_isSharedCheck_4669_; 
lean_del_object(v___x_4643_);
lean_dec(v_snd_4641_);
lean_dec_ref(v_unfold_x3f_4628_);
v_a_4662_ = lean_ctor_get(v___x_4658_, 0);
v_isSharedCheck_4669_ = !lean_is_exclusive(v___x_4658_);
if (v_isSharedCheck_4669_ == 0)
{
v___x_4664_ = v___x_4658_;
v_isShared_4665_ = v_isSharedCheck_4669_;
goto v_resetjp_4663_;
}
else
{
lean_inc(v_a_4662_);
lean_dec(v___x_4658_);
v___x_4664_ = lean_box(0);
v_isShared_4665_ = v_isSharedCheck_4669_;
goto v_resetjp_4663_;
}
v_resetjp_4663_:
{
lean_object* v___x_4667_; 
if (v_isShared_4665_ == 0)
{
v___x_4667_ = v___x_4664_;
goto v_reusejp_4666_;
}
else
{
lean_object* v_reuseFailAlloc_4668_; 
v_reuseFailAlloc_4668_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4668_, 0, v_a_4662_);
v___x_4667_ = v_reuseFailAlloc_4668_;
goto v_reusejp_4666_;
}
v_reusejp_4666_:
{
return v___x_4667_;
}
}
}
}
else
{
v_a_4647_ = v_snd_4641_;
goto v___jp_4646_;
}
}
v___jp_4646_:
{
lean_object* v___x_4649_; 
if (v_isShared_4644_ == 0)
{
lean_ctor_set(v___x_4643_, 1, v_a_4647_);
lean_ctor_set(v___x_4643_, 0, v___x_4645_);
v___x_4649_ = v___x_4643_;
goto v_reusejp_4648_;
}
else
{
lean_object* v_reuseFailAlloc_4653_; 
v_reuseFailAlloc_4653_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4653_, 0, v___x_4645_);
lean_ctor_set(v_reuseFailAlloc_4653_, 1, v_a_4647_);
v___x_4649_ = v_reuseFailAlloc_4653_;
goto v_reusejp_4648_;
}
v_reusejp_4648_:
{
size_t v___x_4650_; size_t v___x_4651_; 
v___x_4650_ = ((size_t)1ULL);
v___x_4651_ = lean_usize_add(v_i_4631_, v___x_4650_);
v_i_4631_ = v___x_4651_;
v_b_4632_ = v___x_4649_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0_spec__3_spec__4___redArg___boxed(lean_object* v_unfold_x3f_4672_, lean_object* v_as_4673_, lean_object* v_sz_4674_, lean_object* v_i_4675_, lean_object* v_b_4676_, lean_object* v___y_4677_, lean_object* v___y_4678_, lean_object* v___y_4679_, lean_object* v___y_4680_, lean_object* v___y_4681_, lean_object* v___y_4682_){
_start:
{
size_t v_sz_boxed_4683_; size_t v_i_boxed_4684_; lean_object* v_res_4685_; 
v_sz_boxed_4683_ = lean_unbox_usize(v_sz_4674_);
lean_dec(v_sz_4674_);
v_i_boxed_4684_ = lean_unbox_usize(v_i_4675_);
lean_dec(v_i_4675_);
v_res_4685_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0_spec__3_spec__4___redArg(v_unfold_x3f_4672_, v_as_4673_, v_sz_boxed_4683_, v_i_boxed_4684_, v_b_4676_, v___y_4677_, v___y_4678_, v___y_4679_, v___y_4680_, v___y_4681_);
lean_dec(v___y_4681_);
lean_dec_ref(v___y_4680_);
lean_dec(v___y_4679_);
lean_dec_ref(v___y_4678_);
lean_dec(v___y_4677_);
lean_dec_ref(v_as_4673_);
return v_res_4685_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0_spec__3(lean_object* v_unfold_x3f_4686_, lean_object* v_as_4687_, size_t v_sz_4688_, size_t v_i_4689_, lean_object* v_b_4690_, lean_object* v___y_4691_, lean_object* v___y_4692_, lean_object* v___y_4693_, lean_object* v___y_4694_, lean_object* v___y_4695_, lean_object* v___y_4696_){
_start:
{
uint8_t v___x_4698_; 
v___x_4698_ = lean_usize_dec_lt(v_i_4689_, v_sz_4688_);
if (v___x_4698_ == 0)
{
lean_object* v___x_4699_; 
lean_dec_ref(v_unfold_x3f_4686_);
v___x_4699_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4699_, 0, v_b_4690_);
return v___x_4699_;
}
else
{
lean_object* v_snd_4700_; lean_object* v___x_4702_; uint8_t v_isShared_4703_; uint8_t v_isSharedCheck_4729_; 
v_snd_4700_ = lean_ctor_get(v_b_4690_, 1);
v_isSharedCheck_4729_ = !lean_is_exclusive(v_b_4690_);
if (v_isSharedCheck_4729_ == 0)
{
lean_object* v_unused_4730_; 
v_unused_4730_ = lean_ctor_get(v_b_4690_, 0);
lean_dec(v_unused_4730_);
v___x_4702_ = v_b_4690_;
v_isShared_4703_ = v_isSharedCheck_4729_;
goto v_resetjp_4701_;
}
else
{
lean_inc(v_snd_4700_);
lean_dec(v_b_4690_);
v___x_4702_ = lean_box(0);
v_isShared_4703_ = v_isSharedCheck_4729_;
goto v_resetjp_4701_;
}
v_resetjp_4701_:
{
lean_object* v___x_4704_; lean_object* v_a_4706_; lean_object* v_a_4713_; 
v___x_4704_ = lean_box(0);
v_a_4713_ = lean_array_uget_borrowed(v_as_4687_, v_i_4689_);
if (lean_obj_tag(v_a_4713_) == 0)
{
v_a_4706_ = v_snd_4700_;
goto v___jp_4705_;
}
else
{
lean_object* v_val_4714_; uint8_t v___x_4715_; 
v_val_4714_ = lean_ctor_get(v_a_4713_, 0);
v___x_4715_ = l_Lean_LocalDecl_isImplementationDetail(v_val_4714_);
if (v___x_4715_ == 0)
{
lean_object* v___x_4716_; lean_object* v___x_4717_; 
v___x_4716_ = l_Lean_LocalDecl_fvarId(v_val_4714_);
lean_inc(v_snd_4700_);
lean_inc_ref(v_unfold_x3f_4686_);
v___x_4717_ = lp_aesop_Aesop_unfoldManyAtS___redArg(v_unfold_x3f_4686_, v_snd_4700_, v___x_4716_, v___y_4691_, v___y_4693_, v___y_4694_, v___y_4695_, v___y_4696_);
if (lean_obj_tag(v___x_4717_) == 0)
{
lean_object* v_a_4718_; 
v_a_4718_ = lean_ctor_get(v___x_4717_, 0);
lean_inc(v_a_4718_);
lean_dec_ref_known(v___x_4717_, 1);
if (lean_obj_tag(v_a_4718_) == 1)
{
lean_object* v_val_4719_; lean_object* v_fst_4720_; 
lean_dec(v_snd_4700_);
v_val_4719_ = lean_ctor_get(v_a_4718_, 0);
lean_inc(v_val_4719_);
lean_dec_ref_known(v_a_4718_, 1);
v_fst_4720_ = lean_ctor_get(v_val_4719_, 0);
lean_inc(v_fst_4720_);
lean_dec(v_val_4719_);
v_a_4706_ = v_fst_4720_;
goto v___jp_4705_;
}
else
{
lean_dec(v_a_4718_);
v_a_4706_ = v_snd_4700_;
goto v___jp_4705_;
}
}
else
{
lean_object* v_a_4721_; lean_object* v___x_4723_; uint8_t v_isShared_4724_; uint8_t v_isSharedCheck_4728_; 
lean_del_object(v___x_4702_);
lean_dec(v_snd_4700_);
lean_dec_ref(v_unfold_x3f_4686_);
v_a_4721_ = lean_ctor_get(v___x_4717_, 0);
v_isSharedCheck_4728_ = !lean_is_exclusive(v___x_4717_);
if (v_isSharedCheck_4728_ == 0)
{
v___x_4723_ = v___x_4717_;
v_isShared_4724_ = v_isSharedCheck_4728_;
goto v_resetjp_4722_;
}
else
{
lean_inc(v_a_4721_);
lean_dec(v___x_4717_);
v___x_4723_ = lean_box(0);
v_isShared_4724_ = v_isSharedCheck_4728_;
goto v_resetjp_4722_;
}
v_resetjp_4722_:
{
lean_object* v___x_4726_; 
if (v_isShared_4724_ == 0)
{
v___x_4726_ = v___x_4723_;
goto v_reusejp_4725_;
}
else
{
lean_object* v_reuseFailAlloc_4727_; 
v_reuseFailAlloc_4727_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4727_, 0, v_a_4721_);
v___x_4726_ = v_reuseFailAlloc_4727_;
goto v_reusejp_4725_;
}
v_reusejp_4725_:
{
return v___x_4726_;
}
}
}
}
else
{
v_a_4706_ = v_snd_4700_;
goto v___jp_4705_;
}
}
v___jp_4705_:
{
lean_object* v___x_4708_; 
if (v_isShared_4703_ == 0)
{
lean_ctor_set(v___x_4702_, 1, v_a_4706_);
lean_ctor_set(v___x_4702_, 0, v___x_4704_);
v___x_4708_ = v___x_4702_;
goto v_reusejp_4707_;
}
else
{
lean_object* v_reuseFailAlloc_4712_; 
v_reuseFailAlloc_4712_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4712_, 0, v___x_4704_);
lean_ctor_set(v_reuseFailAlloc_4712_, 1, v_a_4706_);
v___x_4708_ = v_reuseFailAlloc_4712_;
goto v_reusejp_4707_;
}
v_reusejp_4707_:
{
size_t v___x_4709_; size_t v___x_4710_; lean_object* v___x_4711_; 
v___x_4709_ = ((size_t)1ULL);
v___x_4710_ = lean_usize_add(v_i_4689_, v___x_4709_);
v___x_4711_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0_spec__3_spec__4___redArg(v_unfold_x3f_4686_, v_as_4687_, v_sz_4688_, v___x_4710_, v___x_4708_, v___y_4691_, v___y_4693_, v___y_4694_, v___y_4695_, v___y_4696_);
return v___x_4711_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0_spec__3___boxed(lean_object* v_unfold_x3f_4731_, lean_object* v_as_4732_, lean_object* v_sz_4733_, lean_object* v_i_4734_, lean_object* v_b_4735_, lean_object* v___y_4736_, lean_object* v___y_4737_, lean_object* v___y_4738_, lean_object* v___y_4739_, lean_object* v___y_4740_, lean_object* v___y_4741_, lean_object* v___y_4742_){
_start:
{
size_t v_sz_boxed_4743_; size_t v_i_boxed_4744_; lean_object* v_res_4745_; 
v_sz_boxed_4743_ = lean_unbox_usize(v_sz_4733_);
lean_dec(v_sz_4733_);
v_i_boxed_4744_ = lean_unbox_usize(v_i_4734_);
lean_dec(v_i_4734_);
v_res_4745_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0_spec__3(v_unfold_x3f_4731_, v_as_4732_, v_sz_boxed_4743_, v_i_boxed_4744_, v_b_4735_, v___y_4736_, v___y_4737_, v___y_4738_, v___y_4739_, v___y_4740_, v___y_4741_);
lean_dec(v___y_4741_);
lean_dec_ref(v___y_4740_);
lean_dec(v___y_4739_);
lean_dec_ref(v___y_4738_);
lean_dec(v___y_4737_);
lean_dec(v___y_4736_);
lean_dec_ref(v_as_4732_);
return v_res_4745_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0(lean_object* v_init_4746_, lean_object* v_unfold_x3f_4747_, lean_object* v_n_4748_, lean_object* v_b_4749_, lean_object* v___y_4750_, lean_object* v___y_4751_, lean_object* v___y_4752_, lean_object* v___y_4753_, lean_object* v___y_4754_, lean_object* v___y_4755_){
_start:
{
if (lean_obj_tag(v_n_4748_) == 0)
{
lean_object* v_cs_4757_; lean_object* v___x_4758_; lean_object* v___x_4759_; size_t v_sz_4760_; size_t v___x_4761_; lean_object* v___x_4762_; 
v_cs_4757_ = lean_ctor_get(v_n_4748_, 0);
v___x_4758_ = lean_box(0);
v___x_4759_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4759_, 0, v___x_4758_);
lean_ctor_set(v___x_4759_, 1, v_b_4749_);
v_sz_4760_ = lean_array_size(v_cs_4757_);
v___x_4761_ = ((size_t)0ULL);
v___x_4762_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0_spec__2(v_init_4746_, v_unfold_x3f_4747_, v_cs_4757_, v_sz_4760_, v___x_4761_, v___x_4759_, v___y_4750_, v___y_4751_, v___y_4752_, v___y_4753_, v___y_4754_, v___y_4755_);
if (lean_obj_tag(v___x_4762_) == 0)
{
lean_object* v_a_4763_; lean_object* v___x_4765_; uint8_t v_isShared_4766_; uint8_t v_isSharedCheck_4777_; 
v_a_4763_ = lean_ctor_get(v___x_4762_, 0);
v_isSharedCheck_4777_ = !lean_is_exclusive(v___x_4762_);
if (v_isSharedCheck_4777_ == 0)
{
v___x_4765_ = v___x_4762_;
v_isShared_4766_ = v_isSharedCheck_4777_;
goto v_resetjp_4764_;
}
else
{
lean_inc(v_a_4763_);
lean_dec(v___x_4762_);
v___x_4765_ = lean_box(0);
v_isShared_4766_ = v_isSharedCheck_4777_;
goto v_resetjp_4764_;
}
v_resetjp_4764_:
{
lean_object* v_fst_4767_; 
v_fst_4767_ = lean_ctor_get(v_a_4763_, 0);
if (lean_obj_tag(v_fst_4767_) == 0)
{
lean_object* v_snd_4768_; lean_object* v___x_4769_; lean_object* v___x_4771_; 
v_snd_4768_ = lean_ctor_get(v_a_4763_, 1);
lean_inc(v_snd_4768_);
lean_dec(v_a_4763_);
v___x_4769_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4769_, 0, v_snd_4768_);
if (v_isShared_4766_ == 0)
{
lean_ctor_set(v___x_4765_, 0, v___x_4769_);
v___x_4771_ = v___x_4765_;
goto v_reusejp_4770_;
}
else
{
lean_object* v_reuseFailAlloc_4772_; 
v_reuseFailAlloc_4772_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4772_, 0, v___x_4769_);
v___x_4771_ = v_reuseFailAlloc_4772_;
goto v_reusejp_4770_;
}
v_reusejp_4770_:
{
return v___x_4771_;
}
}
else
{
lean_object* v_val_4773_; lean_object* v___x_4775_; 
lean_inc_ref(v_fst_4767_);
lean_dec(v_a_4763_);
v_val_4773_ = lean_ctor_get(v_fst_4767_, 0);
lean_inc(v_val_4773_);
lean_dec_ref_known(v_fst_4767_, 1);
if (v_isShared_4766_ == 0)
{
lean_ctor_set(v___x_4765_, 0, v_val_4773_);
v___x_4775_ = v___x_4765_;
goto v_reusejp_4774_;
}
else
{
lean_object* v_reuseFailAlloc_4776_; 
v_reuseFailAlloc_4776_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4776_, 0, v_val_4773_);
v___x_4775_ = v_reuseFailAlloc_4776_;
goto v_reusejp_4774_;
}
v_reusejp_4774_:
{
return v___x_4775_;
}
}
}
}
else
{
lean_object* v_a_4778_; lean_object* v___x_4780_; uint8_t v_isShared_4781_; uint8_t v_isSharedCheck_4785_; 
v_a_4778_ = lean_ctor_get(v___x_4762_, 0);
v_isSharedCheck_4785_ = !lean_is_exclusive(v___x_4762_);
if (v_isSharedCheck_4785_ == 0)
{
v___x_4780_ = v___x_4762_;
v_isShared_4781_ = v_isSharedCheck_4785_;
goto v_resetjp_4779_;
}
else
{
lean_inc(v_a_4778_);
lean_dec(v___x_4762_);
v___x_4780_ = lean_box(0);
v_isShared_4781_ = v_isSharedCheck_4785_;
goto v_resetjp_4779_;
}
v_resetjp_4779_:
{
lean_object* v___x_4783_; 
if (v_isShared_4781_ == 0)
{
v___x_4783_ = v___x_4780_;
goto v_reusejp_4782_;
}
else
{
lean_object* v_reuseFailAlloc_4784_; 
v_reuseFailAlloc_4784_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4784_, 0, v_a_4778_);
v___x_4783_ = v_reuseFailAlloc_4784_;
goto v_reusejp_4782_;
}
v_reusejp_4782_:
{
return v___x_4783_;
}
}
}
}
else
{
lean_object* v_vs_4786_; lean_object* v___x_4787_; lean_object* v___x_4788_; size_t v_sz_4789_; size_t v___x_4790_; lean_object* v___x_4791_; 
v_vs_4786_ = lean_ctor_get(v_n_4748_, 0);
v___x_4787_ = lean_box(0);
v___x_4788_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4788_, 0, v___x_4787_);
lean_ctor_set(v___x_4788_, 1, v_b_4749_);
v_sz_4789_ = lean_array_size(v_vs_4786_);
v___x_4790_ = ((size_t)0ULL);
v___x_4791_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0_spec__3(v_unfold_x3f_4747_, v_vs_4786_, v_sz_4789_, v___x_4790_, v___x_4788_, v___y_4750_, v___y_4751_, v___y_4752_, v___y_4753_, v___y_4754_, v___y_4755_);
if (lean_obj_tag(v___x_4791_) == 0)
{
lean_object* v_a_4792_; lean_object* v___x_4794_; uint8_t v_isShared_4795_; uint8_t v_isSharedCheck_4806_; 
v_a_4792_ = lean_ctor_get(v___x_4791_, 0);
v_isSharedCheck_4806_ = !lean_is_exclusive(v___x_4791_);
if (v_isSharedCheck_4806_ == 0)
{
v___x_4794_ = v___x_4791_;
v_isShared_4795_ = v_isSharedCheck_4806_;
goto v_resetjp_4793_;
}
else
{
lean_inc(v_a_4792_);
lean_dec(v___x_4791_);
v___x_4794_ = lean_box(0);
v_isShared_4795_ = v_isSharedCheck_4806_;
goto v_resetjp_4793_;
}
v_resetjp_4793_:
{
lean_object* v_fst_4796_; 
v_fst_4796_ = lean_ctor_get(v_a_4792_, 0);
if (lean_obj_tag(v_fst_4796_) == 0)
{
lean_object* v_snd_4797_; lean_object* v___x_4798_; lean_object* v___x_4800_; 
v_snd_4797_ = lean_ctor_get(v_a_4792_, 1);
lean_inc(v_snd_4797_);
lean_dec(v_a_4792_);
v___x_4798_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4798_, 0, v_snd_4797_);
if (v_isShared_4795_ == 0)
{
lean_ctor_set(v___x_4794_, 0, v___x_4798_);
v___x_4800_ = v___x_4794_;
goto v_reusejp_4799_;
}
else
{
lean_object* v_reuseFailAlloc_4801_; 
v_reuseFailAlloc_4801_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4801_, 0, v___x_4798_);
v___x_4800_ = v_reuseFailAlloc_4801_;
goto v_reusejp_4799_;
}
v_reusejp_4799_:
{
return v___x_4800_;
}
}
else
{
lean_object* v_val_4802_; lean_object* v___x_4804_; 
lean_inc_ref(v_fst_4796_);
lean_dec(v_a_4792_);
v_val_4802_ = lean_ctor_get(v_fst_4796_, 0);
lean_inc(v_val_4802_);
lean_dec_ref_known(v_fst_4796_, 1);
if (v_isShared_4795_ == 0)
{
lean_ctor_set(v___x_4794_, 0, v_val_4802_);
v___x_4804_ = v___x_4794_;
goto v_reusejp_4803_;
}
else
{
lean_object* v_reuseFailAlloc_4805_; 
v_reuseFailAlloc_4805_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4805_, 0, v_val_4802_);
v___x_4804_ = v_reuseFailAlloc_4805_;
goto v_reusejp_4803_;
}
v_reusejp_4803_:
{
return v___x_4804_;
}
}
}
}
else
{
lean_object* v_a_4807_; lean_object* v___x_4809_; uint8_t v_isShared_4810_; uint8_t v_isSharedCheck_4814_; 
v_a_4807_ = lean_ctor_get(v___x_4791_, 0);
v_isSharedCheck_4814_ = !lean_is_exclusive(v___x_4791_);
if (v_isSharedCheck_4814_ == 0)
{
v___x_4809_ = v___x_4791_;
v_isShared_4810_ = v_isSharedCheck_4814_;
goto v_resetjp_4808_;
}
else
{
lean_inc(v_a_4807_);
lean_dec(v___x_4791_);
v___x_4809_ = lean_box(0);
v_isShared_4810_ = v_isSharedCheck_4814_;
goto v_resetjp_4808_;
}
v_resetjp_4808_:
{
lean_object* v___x_4812_; 
if (v_isShared_4810_ == 0)
{
v___x_4812_ = v___x_4809_;
goto v_reusejp_4811_;
}
else
{
lean_object* v_reuseFailAlloc_4813_; 
v_reuseFailAlloc_4813_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4813_, 0, v_a_4807_);
v___x_4812_ = v_reuseFailAlloc_4813_;
goto v_reusejp_4811_;
}
v_reusejp_4811_:
{
return v___x_4812_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0_spec__2(lean_object* v_init_4815_, lean_object* v_unfold_x3f_4816_, lean_object* v_as_4817_, size_t v_sz_4818_, size_t v_i_4819_, lean_object* v_b_4820_, lean_object* v___y_4821_, lean_object* v___y_4822_, lean_object* v___y_4823_, lean_object* v___y_4824_, lean_object* v___y_4825_, lean_object* v___y_4826_){
_start:
{
uint8_t v___x_4828_; 
v___x_4828_ = lean_usize_dec_lt(v_i_4819_, v_sz_4818_);
if (v___x_4828_ == 0)
{
lean_object* v___x_4829_; 
lean_dec_ref(v_unfold_x3f_4816_);
v___x_4829_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4829_, 0, v_b_4820_);
return v___x_4829_;
}
else
{
lean_object* v_snd_4830_; lean_object* v___x_4832_; uint8_t v_isShared_4833_; uint8_t v_isSharedCheck_4864_; 
v_snd_4830_ = lean_ctor_get(v_b_4820_, 1);
v_isSharedCheck_4864_ = !lean_is_exclusive(v_b_4820_);
if (v_isSharedCheck_4864_ == 0)
{
lean_object* v_unused_4865_; 
v_unused_4865_ = lean_ctor_get(v_b_4820_, 0);
lean_dec(v_unused_4865_);
v___x_4832_ = v_b_4820_;
v_isShared_4833_ = v_isSharedCheck_4864_;
goto v_resetjp_4831_;
}
else
{
lean_inc(v_snd_4830_);
lean_dec(v_b_4820_);
v___x_4832_ = lean_box(0);
v_isShared_4833_ = v_isSharedCheck_4864_;
goto v_resetjp_4831_;
}
v_resetjp_4831_:
{
lean_object* v_a_4834_; lean_object* v___x_4835_; 
v_a_4834_ = lean_array_uget_borrowed(v_as_4817_, v_i_4819_);
lean_inc(v_snd_4830_);
lean_inc_ref(v_unfold_x3f_4816_);
v___x_4835_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0(v_init_4815_, v_unfold_x3f_4816_, v_a_4834_, v_snd_4830_, v___y_4821_, v___y_4822_, v___y_4823_, v___y_4824_, v___y_4825_, v___y_4826_);
if (lean_obj_tag(v___x_4835_) == 0)
{
lean_object* v_a_4836_; lean_object* v___x_4838_; uint8_t v_isShared_4839_; uint8_t v_isSharedCheck_4855_; 
v_a_4836_ = lean_ctor_get(v___x_4835_, 0);
v_isSharedCheck_4855_ = !lean_is_exclusive(v___x_4835_);
if (v_isSharedCheck_4855_ == 0)
{
v___x_4838_ = v___x_4835_;
v_isShared_4839_ = v_isSharedCheck_4855_;
goto v_resetjp_4837_;
}
else
{
lean_inc(v_a_4836_);
lean_dec(v___x_4835_);
v___x_4838_ = lean_box(0);
v_isShared_4839_ = v_isSharedCheck_4855_;
goto v_resetjp_4837_;
}
v_resetjp_4837_:
{
if (lean_obj_tag(v_a_4836_) == 0)
{
lean_object* v___x_4840_; lean_object* v___x_4842_; 
lean_dec_ref(v_unfold_x3f_4816_);
v___x_4840_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4840_, 0, v_a_4836_);
if (v_isShared_4833_ == 0)
{
lean_ctor_set(v___x_4832_, 0, v___x_4840_);
v___x_4842_ = v___x_4832_;
goto v_reusejp_4841_;
}
else
{
lean_object* v_reuseFailAlloc_4846_; 
v_reuseFailAlloc_4846_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4846_, 0, v___x_4840_);
lean_ctor_set(v_reuseFailAlloc_4846_, 1, v_snd_4830_);
v___x_4842_ = v_reuseFailAlloc_4846_;
goto v_reusejp_4841_;
}
v_reusejp_4841_:
{
lean_object* v___x_4844_; 
if (v_isShared_4839_ == 0)
{
lean_ctor_set(v___x_4838_, 0, v___x_4842_);
v___x_4844_ = v___x_4838_;
goto v_reusejp_4843_;
}
else
{
lean_object* v_reuseFailAlloc_4845_; 
v_reuseFailAlloc_4845_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4845_, 0, v___x_4842_);
v___x_4844_ = v_reuseFailAlloc_4845_;
goto v_reusejp_4843_;
}
v_reusejp_4843_:
{
return v___x_4844_;
}
}
}
else
{
lean_object* v_a_4847_; lean_object* v___x_4848_; lean_object* v___x_4850_; 
lean_del_object(v___x_4838_);
lean_dec(v_snd_4830_);
v_a_4847_ = lean_ctor_get(v_a_4836_, 0);
lean_inc(v_a_4847_);
lean_dec_ref_known(v_a_4836_, 1);
v___x_4848_ = lean_box(0);
if (v_isShared_4833_ == 0)
{
lean_ctor_set(v___x_4832_, 1, v_a_4847_);
lean_ctor_set(v___x_4832_, 0, v___x_4848_);
v___x_4850_ = v___x_4832_;
goto v_reusejp_4849_;
}
else
{
lean_object* v_reuseFailAlloc_4854_; 
v_reuseFailAlloc_4854_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4854_, 0, v___x_4848_);
lean_ctor_set(v_reuseFailAlloc_4854_, 1, v_a_4847_);
v___x_4850_ = v_reuseFailAlloc_4854_;
goto v_reusejp_4849_;
}
v_reusejp_4849_:
{
size_t v___x_4851_; size_t v___x_4852_; 
v___x_4851_ = ((size_t)1ULL);
v___x_4852_ = lean_usize_add(v_i_4819_, v___x_4851_);
v_i_4819_ = v___x_4852_;
v_b_4820_ = v___x_4850_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_4856_; lean_object* v___x_4858_; uint8_t v_isShared_4859_; uint8_t v_isSharedCheck_4863_; 
lean_del_object(v___x_4832_);
lean_dec(v_snd_4830_);
lean_dec_ref(v_unfold_x3f_4816_);
v_a_4856_ = lean_ctor_get(v___x_4835_, 0);
v_isSharedCheck_4863_ = !lean_is_exclusive(v___x_4835_);
if (v_isSharedCheck_4863_ == 0)
{
v___x_4858_ = v___x_4835_;
v_isShared_4859_ = v_isSharedCheck_4863_;
goto v_resetjp_4857_;
}
else
{
lean_inc(v_a_4856_);
lean_dec(v___x_4835_);
v___x_4858_ = lean_box(0);
v_isShared_4859_ = v_isSharedCheck_4863_;
goto v_resetjp_4857_;
}
v_resetjp_4857_:
{
lean_object* v___x_4861_; 
if (v_isShared_4859_ == 0)
{
v___x_4861_ = v___x_4858_;
goto v_reusejp_4860_;
}
else
{
lean_object* v_reuseFailAlloc_4862_; 
v_reuseFailAlloc_4862_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4862_, 0, v_a_4856_);
v___x_4861_ = v_reuseFailAlloc_4862_;
goto v_reusejp_4860_;
}
v_reusejp_4860_:
{
return v___x_4861_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0_spec__2___boxed(lean_object* v_init_4866_, lean_object* v_unfold_x3f_4867_, lean_object* v_as_4868_, lean_object* v_sz_4869_, lean_object* v_i_4870_, lean_object* v_b_4871_, lean_object* v___y_4872_, lean_object* v___y_4873_, lean_object* v___y_4874_, lean_object* v___y_4875_, lean_object* v___y_4876_, lean_object* v___y_4877_, lean_object* v___y_4878_){
_start:
{
size_t v_sz_boxed_4879_; size_t v_i_boxed_4880_; lean_object* v_res_4881_; 
v_sz_boxed_4879_ = lean_unbox_usize(v_sz_4869_);
lean_dec(v_sz_4869_);
v_i_boxed_4880_ = lean_unbox_usize(v_i_4870_);
lean_dec(v_i_4870_);
v_res_4881_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0_spec__2(v_init_4866_, v_unfold_x3f_4867_, v_as_4868_, v_sz_boxed_4879_, v_i_boxed_4880_, v_b_4871_, v___y_4872_, v___y_4873_, v___y_4874_, v___y_4875_, v___y_4876_, v___y_4877_);
lean_dec(v___y_4877_);
lean_dec_ref(v___y_4876_);
lean_dec(v___y_4875_);
lean_dec_ref(v___y_4874_);
lean_dec(v___y_4873_);
lean_dec(v___y_4872_);
lean_dec_ref(v_as_4868_);
lean_dec(v_init_4866_);
return v_res_4881_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0___boxed(lean_object* v_init_4882_, lean_object* v_unfold_x3f_4883_, lean_object* v_n_4884_, lean_object* v_b_4885_, lean_object* v___y_4886_, lean_object* v___y_4887_, lean_object* v___y_4888_, lean_object* v___y_4889_, lean_object* v___y_4890_, lean_object* v___y_4891_, lean_object* v___y_4892_){
_start:
{
lean_object* v_res_4893_; 
v_res_4893_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0(v_init_4882_, v_unfold_x3f_4883_, v_n_4884_, v_b_4885_, v___y_4886_, v___y_4887_, v___y_4888_, v___y_4889_, v___y_4890_, v___y_4891_);
lean_dec(v___y_4891_);
lean_dec_ref(v___y_4890_);
lean_dec(v___y_4889_);
lean_dec_ref(v___y_4888_);
lean_dec(v___y_4887_);
lean_dec(v___y_4886_);
lean_dec_ref(v_n_4884_);
lean_dec(v_init_4882_);
return v_res_4893_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__1_spec__5___redArg(lean_object* v_unfold_x3f_4894_, lean_object* v_as_4895_, size_t v_sz_4896_, size_t v_i_4897_, lean_object* v_b_4898_, lean_object* v___y_4899_, lean_object* v___y_4900_, lean_object* v___y_4901_, lean_object* v___y_4902_, lean_object* v___y_4903_){
_start:
{
uint8_t v___x_4905_; 
v___x_4905_ = lean_usize_dec_lt(v_i_4897_, v_sz_4896_);
if (v___x_4905_ == 0)
{
lean_object* v___x_4906_; 
lean_dec_ref(v_unfold_x3f_4894_);
v___x_4906_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4906_, 0, v_b_4898_);
return v___x_4906_;
}
else
{
lean_object* v_snd_4907_; lean_object* v___x_4909_; uint8_t v_isShared_4910_; uint8_t v_isSharedCheck_4936_; 
v_snd_4907_ = lean_ctor_get(v_b_4898_, 1);
v_isSharedCheck_4936_ = !lean_is_exclusive(v_b_4898_);
if (v_isSharedCheck_4936_ == 0)
{
lean_object* v_unused_4937_; 
v_unused_4937_ = lean_ctor_get(v_b_4898_, 0);
lean_dec(v_unused_4937_);
v___x_4909_ = v_b_4898_;
v_isShared_4910_ = v_isSharedCheck_4936_;
goto v_resetjp_4908_;
}
else
{
lean_inc(v_snd_4907_);
lean_dec(v_b_4898_);
v___x_4909_ = lean_box(0);
v_isShared_4910_ = v_isSharedCheck_4936_;
goto v_resetjp_4908_;
}
v_resetjp_4908_:
{
lean_object* v___x_4911_; lean_object* v_a_4913_; lean_object* v_a_4920_; 
v___x_4911_ = lean_box(0);
v_a_4920_ = lean_array_uget_borrowed(v_as_4895_, v_i_4897_);
if (lean_obj_tag(v_a_4920_) == 0)
{
v_a_4913_ = v_snd_4907_;
goto v___jp_4912_;
}
else
{
lean_object* v_val_4921_; uint8_t v___x_4922_; 
v_val_4921_ = lean_ctor_get(v_a_4920_, 0);
v___x_4922_ = l_Lean_LocalDecl_isImplementationDetail(v_val_4921_);
if (v___x_4922_ == 0)
{
lean_object* v___x_4923_; lean_object* v___x_4924_; 
v___x_4923_ = l_Lean_LocalDecl_fvarId(v_val_4921_);
lean_inc(v_snd_4907_);
lean_inc_ref(v_unfold_x3f_4894_);
v___x_4924_ = lp_aesop_Aesop_unfoldManyAtS___redArg(v_unfold_x3f_4894_, v_snd_4907_, v___x_4923_, v___y_4899_, v___y_4900_, v___y_4901_, v___y_4902_, v___y_4903_);
if (lean_obj_tag(v___x_4924_) == 0)
{
lean_object* v_a_4925_; 
v_a_4925_ = lean_ctor_get(v___x_4924_, 0);
lean_inc(v_a_4925_);
lean_dec_ref_known(v___x_4924_, 1);
if (lean_obj_tag(v_a_4925_) == 1)
{
lean_object* v_val_4926_; lean_object* v_fst_4927_; 
lean_dec(v_snd_4907_);
v_val_4926_ = lean_ctor_get(v_a_4925_, 0);
lean_inc(v_val_4926_);
lean_dec_ref_known(v_a_4925_, 1);
v_fst_4927_ = lean_ctor_get(v_val_4926_, 0);
lean_inc(v_fst_4927_);
lean_dec(v_val_4926_);
v_a_4913_ = v_fst_4927_;
goto v___jp_4912_;
}
else
{
lean_dec(v_a_4925_);
v_a_4913_ = v_snd_4907_;
goto v___jp_4912_;
}
}
else
{
lean_object* v_a_4928_; lean_object* v___x_4930_; uint8_t v_isShared_4931_; uint8_t v_isSharedCheck_4935_; 
lean_del_object(v___x_4909_);
lean_dec(v_snd_4907_);
lean_dec_ref(v_unfold_x3f_4894_);
v_a_4928_ = lean_ctor_get(v___x_4924_, 0);
v_isSharedCheck_4935_ = !lean_is_exclusive(v___x_4924_);
if (v_isSharedCheck_4935_ == 0)
{
v___x_4930_ = v___x_4924_;
v_isShared_4931_ = v_isSharedCheck_4935_;
goto v_resetjp_4929_;
}
else
{
lean_inc(v_a_4928_);
lean_dec(v___x_4924_);
v___x_4930_ = lean_box(0);
v_isShared_4931_ = v_isSharedCheck_4935_;
goto v_resetjp_4929_;
}
v_resetjp_4929_:
{
lean_object* v___x_4933_; 
if (v_isShared_4931_ == 0)
{
v___x_4933_ = v___x_4930_;
goto v_reusejp_4932_;
}
else
{
lean_object* v_reuseFailAlloc_4934_; 
v_reuseFailAlloc_4934_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4934_, 0, v_a_4928_);
v___x_4933_ = v_reuseFailAlloc_4934_;
goto v_reusejp_4932_;
}
v_reusejp_4932_:
{
return v___x_4933_;
}
}
}
}
else
{
v_a_4913_ = v_snd_4907_;
goto v___jp_4912_;
}
}
v___jp_4912_:
{
lean_object* v___x_4915_; 
if (v_isShared_4910_ == 0)
{
lean_ctor_set(v___x_4909_, 1, v_a_4913_);
lean_ctor_set(v___x_4909_, 0, v___x_4911_);
v___x_4915_ = v___x_4909_;
goto v_reusejp_4914_;
}
else
{
lean_object* v_reuseFailAlloc_4919_; 
v_reuseFailAlloc_4919_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4919_, 0, v___x_4911_);
lean_ctor_set(v_reuseFailAlloc_4919_, 1, v_a_4913_);
v___x_4915_ = v_reuseFailAlloc_4919_;
goto v_reusejp_4914_;
}
v_reusejp_4914_:
{
size_t v___x_4916_; size_t v___x_4917_; 
v___x_4916_ = ((size_t)1ULL);
v___x_4917_ = lean_usize_add(v_i_4897_, v___x_4916_);
v_i_4897_ = v___x_4917_;
v_b_4898_ = v___x_4915_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__1_spec__5___redArg___boxed(lean_object* v_unfold_x3f_4938_, lean_object* v_as_4939_, lean_object* v_sz_4940_, lean_object* v_i_4941_, lean_object* v_b_4942_, lean_object* v___y_4943_, lean_object* v___y_4944_, lean_object* v___y_4945_, lean_object* v___y_4946_, lean_object* v___y_4947_, lean_object* v___y_4948_){
_start:
{
size_t v_sz_boxed_4949_; size_t v_i_boxed_4950_; lean_object* v_res_4951_; 
v_sz_boxed_4949_ = lean_unbox_usize(v_sz_4940_);
lean_dec(v_sz_4940_);
v_i_boxed_4950_ = lean_unbox_usize(v_i_4941_);
lean_dec(v_i_4941_);
v_res_4951_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__1_spec__5___redArg(v_unfold_x3f_4938_, v_as_4939_, v_sz_boxed_4949_, v_i_boxed_4950_, v_b_4942_, v___y_4943_, v___y_4944_, v___y_4945_, v___y_4946_, v___y_4947_);
lean_dec(v___y_4947_);
lean_dec_ref(v___y_4946_);
lean_dec(v___y_4945_);
lean_dec_ref(v___y_4944_);
lean_dec(v___y_4943_);
lean_dec_ref(v_as_4939_);
return v_res_4951_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__1(lean_object* v_unfold_x3f_4952_, lean_object* v_as_4953_, size_t v_sz_4954_, size_t v_i_4955_, lean_object* v_b_4956_, lean_object* v___y_4957_, lean_object* v___y_4958_, lean_object* v___y_4959_, lean_object* v___y_4960_, lean_object* v___y_4961_, lean_object* v___y_4962_){
_start:
{
uint8_t v___x_4964_; 
v___x_4964_ = lean_usize_dec_lt(v_i_4955_, v_sz_4954_);
if (v___x_4964_ == 0)
{
lean_object* v___x_4965_; 
lean_dec_ref(v_unfold_x3f_4952_);
v___x_4965_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4965_, 0, v_b_4956_);
return v___x_4965_;
}
else
{
lean_object* v_snd_4966_; lean_object* v___x_4968_; uint8_t v_isShared_4969_; uint8_t v_isSharedCheck_4995_; 
v_snd_4966_ = lean_ctor_get(v_b_4956_, 1);
v_isSharedCheck_4995_ = !lean_is_exclusive(v_b_4956_);
if (v_isSharedCheck_4995_ == 0)
{
lean_object* v_unused_4996_; 
v_unused_4996_ = lean_ctor_get(v_b_4956_, 0);
lean_dec(v_unused_4996_);
v___x_4968_ = v_b_4956_;
v_isShared_4969_ = v_isSharedCheck_4995_;
goto v_resetjp_4967_;
}
else
{
lean_inc(v_snd_4966_);
lean_dec(v_b_4956_);
v___x_4968_ = lean_box(0);
v_isShared_4969_ = v_isSharedCheck_4995_;
goto v_resetjp_4967_;
}
v_resetjp_4967_:
{
lean_object* v___x_4970_; lean_object* v_a_4972_; lean_object* v_a_4979_; 
v___x_4970_ = lean_box(0);
v_a_4979_ = lean_array_uget_borrowed(v_as_4953_, v_i_4955_);
if (lean_obj_tag(v_a_4979_) == 0)
{
v_a_4972_ = v_snd_4966_;
goto v___jp_4971_;
}
else
{
lean_object* v_val_4980_; uint8_t v___x_4981_; 
v_val_4980_ = lean_ctor_get(v_a_4979_, 0);
v___x_4981_ = l_Lean_LocalDecl_isImplementationDetail(v_val_4980_);
if (v___x_4981_ == 0)
{
lean_object* v___x_4982_; lean_object* v___x_4983_; 
v___x_4982_ = l_Lean_LocalDecl_fvarId(v_val_4980_);
lean_inc(v_snd_4966_);
lean_inc_ref(v_unfold_x3f_4952_);
v___x_4983_ = lp_aesop_Aesop_unfoldManyAtS___redArg(v_unfold_x3f_4952_, v_snd_4966_, v___x_4982_, v___y_4957_, v___y_4959_, v___y_4960_, v___y_4961_, v___y_4962_);
if (lean_obj_tag(v___x_4983_) == 0)
{
lean_object* v_a_4984_; 
v_a_4984_ = lean_ctor_get(v___x_4983_, 0);
lean_inc(v_a_4984_);
lean_dec_ref_known(v___x_4983_, 1);
if (lean_obj_tag(v_a_4984_) == 1)
{
lean_object* v_val_4985_; lean_object* v_fst_4986_; 
lean_dec(v_snd_4966_);
v_val_4985_ = lean_ctor_get(v_a_4984_, 0);
lean_inc(v_val_4985_);
lean_dec_ref_known(v_a_4984_, 1);
v_fst_4986_ = lean_ctor_get(v_val_4985_, 0);
lean_inc(v_fst_4986_);
lean_dec(v_val_4985_);
v_a_4972_ = v_fst_4986_;
goto v___jp_4971_;
}
else
{
lean_dec(v_a_4984_);
v_a_4972_ = v_snd_4966_;
goto v___jp_4971_;
}
}
else
{
lean_object* v_a_4987_; lean_object* v___x_4989_; uint8_t v_isShared_4990_; uint8_t v_isSharedCheck_4994_; 
lean_del_object(v___x_4968_);
lean_dec(v_snd_4966_);
lean_dec_ref(v_unfold_x3f_4952_);
v_a_4987_ = lean_ctor_get(v___x_4983_, 0);
v_isSharedCheck_4994_ = !lean_is_exclusive(v___x_4983_);
if (v_isSharedCheck_4994_ == 0)
{
v___x_4989_ = v___x_4983_;
v_isShared_4990_ = v_isSharedCheck_4994_;
goto v_resetjp_4988_;
}
else
{
lean_inc(v_a_4987_);
lean_dec(v___x_4983_);
v___x_4989_ = lean_box(0);
v_isShared_4990_ = v_isSharedCheck_4994_;
goto v_resetjp_4988_;
}
v_resetjp_4988_:
{
lean_object* v___x_4992_; 
if (v_isShared_4990_ == 0)
{
v___x_4992_ = v___x_4989_;
goto v_reusejp_4991_;
}
else
{
lean_object* v_reuseFailAlloc_4993_; 
v_reuseFailAlloc_4993_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4993_, 0, v_a_4987_);
v___x_4992_ = v_reuseFailAlloc_4993_;
goto v_reusejp_4991_;
}
v_reusejp_4991_:
{
return v___x_4992_;
}
}
}
}
else
{
v_a_4972_ = v_snd_4966_;
goto v___jp_4971_;
}
}
v___jp_4971_:
{
lean_object* v___x_4974_; 
if (v_isShared_4969_ == 0)
{
lean_ctor_set(v___x_4968_, 1, v_a_4972_);
lean_ctor_set(v___x_4968_, 0, v___x_4970_);
v___x_4974_ = v___x_4968_;
goto v_reusejp_4973_;
}
else
{
lean_object* v_reuseFailAlloc_4978_; 
v_reuseFailAlloc_4978_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4978_, 0, v___x_4970_);
lean_ctor_set(v_reuseFailAlloc_4978_, 1, v_a_4972_);
v___x_4974_ = v_reuseFailAlloc_4978_;
goto v_reusejp_4973_;
}
v_reusejp_4973_:
{
size_t v___x_4975_; size_t v___x_4976_; lean_object* v___x_4977_; 
v___x_4975_ = ((size_t)1ULL);
v___x_4976_ = lean_usize_add(v_i_4955_, v___x_4975_);
v___x_4977_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__1_spec__5___redArg(v_unfold_x3f_4952_, v_as_4953_, v_sz_4954_, v___x_4976_, v___x_4974_, v___y_4957_, v___y_4959_, v___y_4960_, v___y_4961_, v___y_4962_);
return v___x_4977_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__1___boxed(lean_object* v_unfold_x3f_4997_, lean_object* v_as_4998_, lean_object* v_sz_4999_, lean_object* v_i_5000_, lean_object* v_b_5001_, lean_object* v___y_5002_, lean_object* v___y_5003_, lean_object* v___y_5004_, lean_object* v___y_5005_, lean_object* v___y_5006_, lean_object* v___y_5007_, lean_object* v___y_5008_){
_start:
{
size_t v_sz_boxed_5009_; size_t v_i_boxed_5010_; lean_object* v_res_5011_; 
v_sz_boxed_5009_ = lean_unbox_usize(v_sz_4999_);
lean_dec(v_sz_4999_);
v_i_boxed_5010_ = lean_unbox_usize(v_i_5000_);
lean_dec(v_i_5000_);
v_res_5011_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__1(v_unfold_x3f_4997_, v_as_4998_, v_sz_boxed_5009_, v_i_boxed_5010_, v_b_5001_, v___y_5002_, v___y_5003_, v___y_5004_, v___y_5005_, v___y_5006_, v___y_5007_);
lean_dec(v___y_5007_);
lean_dec_ref(v___y_5006_);
lean_dec(v___y_5005_);
lean_dec_ref(v___y_5004_);
lean_dec(v___y_5003_);
lean_dec(v___y_5002_);
lean_dec_ref(v_as_4998_);
return v_res_5011_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0(lean_object* v_unfold_x3f_5012_, lean_object* v_t_5013_, lean_object* v_init_5014_, lean_object* v___y_5015_, lean_object* v___y_5016_, lean_object* v___y_5017_, lean_object* v___y_5018_, lean_object* v___y_5019_, lean_object* v___y_5020_){
_start:
{
lean_object* v_root_5022_; lean_object* v_tail_5023_; lean_object* v___x_5024_; 
v_root_5022_ = lean_ctor_get(v_t_5013_, 0);
v_tail_5023_ = lean_ctor_get(v_t_5013_, 1);
lean_inc_ref(v_unfold_x3f_5012_);
lean_inc(v_init_5014_);
v___x_5024_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0(v_init_5014_, v_unfold_x3f_5012_, v_root_5022_, v_init_5014_, v___y_5015_, v___y_5016_, v___y_5017_, v___y_5018_, v___y_5019_, v___y_5020_);
lean_dec(v_init_5014_);
if (lean_obj_tag(v___x_5024_) == 0)
{
lean_object* v_a_5025_; lean_object* v___x_5027_; uint8_t v_isShared_5028_; uint8_t v_isSharedCheck_5061_; 
v_a_5025_ = lean_ctor_get(v___x_5024_, 0);
v_isSharedCheck_5061_ = !lean_is_exclusive(v___x_5024_);
if (v_isSharedCheck_5061_ == 0)
{
v___x_5027_ = v___x_5024_;
v_isShared_5028_ = v_isSharedCheck_5061_;
goto v_resetjp_5026_;
}
else
{
lean_inc(v_a_5025_);
lean_dec(v___x_5024_);
v___x_5027_ = lean_box(0);
v_isShared_5028_ = v_isSharedCheck_5061_;
goto v_resetjp_5026_;
}
v_resetjp_5026_:
{
if (lean_obj_tag(v_a_5025_) == 0)
{
lean_object* v_a_5029_; lean_object* v___x_5031_; 
lean_dec_ref(v_unfold_x3f_5012_);
v_a_5029_ = lean_ctor_get(v_a_5025_, 0);
lean_inc(v_a_5029_);
lean_dec_ref_known(v_a_5025_, 1);
if (v_isShared_5028_ == 0)
{
lean_ctor_set(v___x_5027_, 0, v_a_5029_);
v___x_5031_ = v___x_5027_;
goto v_reusejp_5030_;
}
else
{
lean_object* v_reuseFailAlloc_5032_; 
v_reuseFailAlloc_5032_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5032_, 0, v_a_5029_);
v___x_5031_ = v_reuseFailAlloc_5032_;
goto v_reusejp_5030_;
}
v_reusejp_5030_:
{
return v___x_5031_;
}
}
else
{
lean_object* v_a_5033_; lean_object* v___x_5034_; lean_object* v___x_5035_; size_t v_sz_5036_; size_t v___x_5037_; lean_object* v___x_5038_; 
lean_del_object(v___x_5027_);
v_a_5033_ = lean_ctor_get(v_a_5025_, 0);
lean_inc(v_a_5033_);
lean_dec_ref_known(v_a_5025_, 1);
v___x_5034_ = lean_box(0);
v___x_5035_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5035_, 0, v___x_5034_);
lean_ctor_set(v___x_5035_, 1, v_a_5033_);
v_sz_5036_ = lean_array_size(v_tail_5023_);
v___x_5037_ = ((size_t)0ULL);
v___x_5038_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__1(v_unfold_x3f_5012_, v_tail_5023_, v_sz_5036_, v___x_5037_, v___x_5035_, v___y_5015_, v___y_5016_, v___y_5017_, v___y_5018_, v___y_5019_, v___y_5020_);
if (lean_obj_tag(v___x_5038_) == 0)
{
lean_object* v_a_5039_; lean_object* v___x_5041_; uint8_t v_isShared_5042_; uint8_t v_isSharedCheck_5052_; 
v_a_5039_ = lean_ctor_get(v___x_5038_, 0);
v_isSharedCheck_5052_ = !lean_is_exclusive(v___x_5038_);
if (v_isSharedCheck_5052_ == 0)
{
v___x_5041_ = v___x_5038_;
v_isShared_5042_ = v_isSharedCheck_5052_;
goto v_resetjp_5040_;
}
else
{
lean_inc(v_a_5039_);
lean_dec(v___x_5038_);
v___x_5041_ = lean_box(0);
v_isShared_5042_ = v_isSharedCheck_5052_;
goto v_resetjp_5040_;
}
v_resetjp_5040_:
{
lean_object* v_fst_5043_; 
v_fst_5043_ = lean_ctor_get(v_a_5039_, 0);
if (lean_obj_tag(v_fst_5043_) == 0)
{
lean_object* v_snd_5044_; lean_object* v___x_5046_; 
v_snd_5044_ = lean_ctor_get(v_a_5039_, 1);
lean_inc(v_snd_5044_);
lean_dec(v_a_5039_);
if (v_isShared_5042_ == 0)
{
lean_ctor_set(v___x_5041_, 0, v_snd_5044_);
v___x_5046_ = v___x_5041_;
goto v_reusejp_5045_;
}
else
{
lean_object* v_reuseFailAlloc_5047_; 
v_reuseFailAlloc_5047_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5047_, 0, v_snd_5044_);
v___x_5046_ = v_reuseFailAlloc_5047_;
goto v_reusejp_5045_;
}
v_reusejp_5045_:
{
return v___x_5046_;
}
}
else
{
lean_object* v_val_5048_; lean_object* v___x_5050_; 
lean_inc_ref(v_fst_5043_);
lean_dec(v_a_5039_);
v_val_5048_ = lean_ctor_get(v_fst_5043_, 0);
lean_inc(v_val_5048_);
lean_dec_ref_known(v_fst_5043_, 1);
if (v_isShared_5042_ == 0)
{
lean_ctor_set(v___x_5041_, 0, v_val_5048_);
v___x_5050_ = v___x_5041_;
goto v_reusejp_5049_;
}
else
{
lean_object* v_reuseFailAlloc_5051_; 
v_reuseFailAlloc_5051_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5051_, 0, v_val_5048_);
v___x_5050_ = v_reuseFailAlloc_5051_;
goto v_reusejp_5049_;
}
v_reusejp_5049_:
{
return v___x_5050_;
}
}
}
}
else
{
lean_object* v_a_5053_; lean_object* v___x_5055_; uint8_t v_isShared_5056_; uint8_t v_isSharedCheck_5060_; 
v_a_5053_ = lean_ctor_get(v___x_5038_, 0);
v_isSharedCheck_5060_ = !lean_is_exclusive(v___x_5038_);
if (v_isSharedCheck_5060_ == 0)
{
v___x_5055_ = v___x_5038_;
v_isShared_5056_ = v_isSharedCheck_5060_;
goto v_resetjp_5054_;
}
else
{
lean_inc(v_a_5053_);
lean_dec(v___x_5038_);
v___x_5055_ = lean_box(0);
v_isShared_5056_ = v_isSharedCheck_5060_;
goto v_resetjp_5054_;
}
v_resetjp_5054_:
{
lean_object* v___x_5058_; 
if (v_isShared_5056_ == 0)
{
v___x_5058_ = v___x_5055_;
goto v_reusejp_5057_;
}
else
{
lean_object* v_reuseFailAlloc_5059_; 
v_reuseFailAlloc_5059_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5059_, 0, v_a_5053_);
v___x_5058_ = v_reuseFailAlloc_5059_;
goto v_reusejp_5057_;
}
v_reusejp_5057_:
{
return v___x_5058_;
}
}
}
}
}
}
else
{
lean_object* v_a_5062_; lean_object* v___x_5064_; uint8_t v_isShared_5065_; uint8_t v_isSharedCheck_5069_; 
lean_dec_ref(v_unfold_x3f_5012_);
v_a_5062_ = lean_ctor_get(v___x_5024_, 0);
v_isSharedCheck_5069_ = !lean_is_exclusive(v___x_5024_);
if (v_isSharedCheck_5069_ == 0)
{
v___x_5064_ = v___x_5024_;
v_isShared_5065_ = v_isSharedCheck_5069_;
goto v_resetjp_5063_;
}
else
{
lean_inc(v_a_5062_);
lean_dec(v___x_5024_);
v___x_5064_ = lean_box(0);
v_isShared_5065_ = v_isSharedCheck_5069_;
goto v_resetjp_5063_;
}
v_resetjp_5063_:
{
lean_object* v___x_5067_; 
if (v_isShared_5065_ == 0)
{
v___x_5067_ = v___x_5064_;
goto v_reusejp_5066_;
}
else
{
lean_object* v_reuseFailAlloc_5068_; 
v_reuseFailAlloc_5068_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5068_, 0, v_a_5062_);
v___x_5067_ = v_reuseFailAlloc_5068_;
goto v_reusejp_5066_;
}
v_reusejp_5066_:
{
return v___x_5067_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0___boxed(lean_object* v_unfold_x3f_5070_, lean_object* v_t_5071_, lean_object* v_init_5072_, lean_object* v___y_5073_, lean_object* v___y_5074_, lean_object* v___y_5075_, lean_object* v___y_5076_, lean_object* v___y_5077_, lean_object* v___y_5078_, lean_object* v___y_5079_){
_start:
{
lean_object* v_res_5080_; 
v_res_5080_ = lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0(v_unfold_x3f_5070_, v_t_5071_, v_init_5072_, v___y_5073_, v___y_5074_, v___y_5075_, v___y_5076_, v___y_5077_, v___y_5078_);
lean_dec(v___y_5078_);
lean_dec_ref(v___y_5077_);
lean_dec(v___y_5076_);
lean_dec_ref(v___y_5075_);
lean_dec(v___y_5074_);
lean_dec(v___y_5073_);
lean_dec_ref(v_t_5071_);
return v_res_5080_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyStarS___lam__0(lean_object* v_unfold_x3f_5081_, lean_object* v_goal_5082_, lean_object* v___y_5083_, lean_object* v___y_5084_, lean_object* v___y_5085_, lean_object* v___y_5086_, lean_object* v___y_5087_, lean_object* v___y_5088_){
_start:
{
lean_object* v___x_5090_; 
lean_inc(v_goal_5082_);
lean_inc_ref(v_unfold_x3f_5081_);
v___x_5090_ = lp_aesop_Aesop_unfoldManyTargetS(v_unfold_x3f_5081_, v_goal_5082_, v___y_5083_, v___y_5084_, v___y_5085_, v___y_5086_, v___y_5087_, v___y_5088_);
if (lean_obj_tag(v___x_5090_) == 0)
{
lean_object* v_a_5091_; lean_object* v_goal_5093_; lean_object* v___y_5094_; lean_object* v___y_5095_; lean_object* v___y_5096_; lean_object* v___y_5097_; lean_object* v___y_5098_; lean_object* v___y_5099_; 
v_a_5091_ = lean_ctor_get(v___x_5090_, 0);
lean_inc(v_a_5091_);
lean_dec_ref_known(v___x_5090_, 1);
if (lean_obj_tag(v_a_5091_) == 1)
{
lean_object* v_val_5135_; lean_object* v_fst_5136_; 
v_val_5135_ = lean_ctor_get(v_a_5091_, 0);
lean_inc(v_val_5135_);
lean_dec_ref_known(v_a_5091_, 1);
v_fst_5136_ = lean_ctor_get(v_val_5135_, 0);
lean_inc(v_fst_5136_);
lean_dec(v_val_5135_);
v_goal_5093_ = v_fst_5136_;
v___y_5094_ = v___y_5083_;
v___y_5095_ = v___y_5084_;
v___y_5096_ = v___y_5085_;
v___y_5097_ = v___y_5086_;
v___y_5098_ = v___y_5087_;
v___y_5099_ = v___y_5088_;
goto v___jp_5092_;
}
else
{
lean_dec(v_a_5091_);
lean_inc(v_goal_5082_);
v_goal_5093_ = v_goal_5082_;
v___y_5094_ = v___y_5083_;
v___y_5095_ = v___y_5084_;
v___y_5096_ = v___y_5085_;
v___y_5097_ = v___y_5086_;
v___y_5098_ = v___y_5087_;
v___y_5099_ = v___y_5088_;
goto v___jp_5092_;
}
v___jp_5092_:
{
lean_object* v___x_5100_; 
lean_inc(v_goal_5093_);
v___x_5100_ = l_Lean_MVarId_getDecl(v_goal_5093_, v___y_5096_, v___y_5097_, v___y_5098_, v___y_5099_);
if (lean_obj_tag(v___x_5100_) == 0)
{
lean_object* v_a_5101_; lean_object* v_lctx_5102_; lean_object* v_decls_5103_; lean_object* v___x_5104_; 
v_a_5101_ = lean_ctor_get(v___x_5100_, 0);
lean_inc(v_a_5101_);
lean_dec_ref_known(v___x_5100_, 1);
v_lctx_5102_ = lean_ctor_get(v_a_5101_, 1);
lean_inc_ref(v_lctx_5102_);
lean_dec(v_a_5101_);
v_decls_5103_ = lean_ctor_get(v_lctx_5102_, 1);
lean_inc_ref(v_decls_5103_);
lean_dec_ref(v_lctx_5102_);
v___x_5104_ = lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0(v_unfold_x3f_5081_, v_decls_5103_, v_goal_5093_, v___y_5094_, v___y_5095_, v___y_5096_, v___y_5097_, v___y_5098_, v___y_5099_);
lean_dec_ref(v_decls_5103_);
if (lean_obj_tag(v___x_5104_) == 0)
{
lean_object* v_a_5105_; lean_object* v___x_5107_; uint8_t v_isShared_5108_; uint8_t v_isSharedCheck_5118_; 
v_a_5105_ = lean_ctor_get(v___x_5104_, 0);
v_isSharedCheck_5118_ = !lean_is_exclusive(v___x_5104_);
if (v_isSharedCheck_5118_ == 0)
{
v___x_5107_ = v___x_5104_;
v_isShared_5108_ = v_isSharedCheck_5118_;
goto v_resetjp_5106_;
}
else
{
lean_inc(v_a_5105_);
lean_dec(v___x_5104_);
v___x_5107_ = lean_box(0);
v_isShared_5108_ = v_isSharedCheck_5118_;
goto v_resetjp_5106_;
}
v_resetjp_5106_:
{
uint8_t v___x_5109_; 
v___x_5109_ = l_Lean_instBEqMVarId_beq(v_a_5105_, v_goal_5082_);
lean_dec(v_goal_5082_);
if (v___x_5109_ == 0)
{
lean_object* v___x_5110_; lean_object* v___x_5112_; 
v___x_5110_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5110_, 0, v_a_5105_);
if (v_isShared_5108_ == 0)
{
lean_ctor_set(v___x_5107_, 0, v___x_5110_);
v___x_5112_ = v___x_5107_;
goto v_reusejp_5111_;
}
else
{
lean_object* v_reuseFailAlloc_5113_; 
v_reuseFailAlloc_5113_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5113_, 0, v___x_5110_);
v___x_5112_ = v_reuseFailAlloc_5113_;
goto v_reusejp_5111_;
}
v_reusejp_5111_:
{
return v___x_5112_;
}
}
else
{
lean_object* v___x_5114_; lean_object* v___x_5116_; 
lean_dec(v_a_5105_);
v___x_5114_ = lean_box(0);
if (v_isShared_5108_ == 0)
{
lean_ctor_set(v___x_5107_, 0, v___x_5114_);
v___x_5116_ = v___x_5107_;
goto v_reusejp_5115_;
}
else
{
lean_object* v_reuseFailAlloc_5117_; 
v_reuseFailAlloc_5117_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5117_, 0, v___x_5114_);
v___x_5116_ = v_reuseFailAlloc_5117_;
goto v_reusejp_5115_;
}
v_reusejp_5115_:
{
return v___x_5116_;
}
}
}
}
else
{
lean_object* v_a_5119_; lean_object* v___x_5121_; uint8_t v_isShared_5122_; uint8_t v_isSharedCheck_5126_; 
lean_dec(v_goal_5082_);
v_a_5119_ = lean_ctor_get(v___x_5104_, 0);
v_isSharedCheck_5126_ = !lean_is_exclusive(v___x_5104_);
if (v_isSharedCheck_5126_ == 0)
{
v___x_5121_ = v___x_5104_;
v_isShared_5122_ = v_isSharedCheck_5126_;
goto v_resetjp_5120_;
}
else
{
lean_inc(v_a_5119_);
lean_dec(v___x_5104_);
v___x_5121_ = lean_box(0);
v_isShared_5122_ = v_isSharedCheck_5126_;
goto v_resetjp_5120_;
}
v_resetjp_5120_:
{
lean_object* v___x_5124_; 
if (v_isShared_5122_ == 0)
{
v___x_5124_ = v___x_5121_;
goto v_reusejp_5123_;
}
else
{
lean_object* v_reuseFailAlloc_5125_; 
v_reuseFailAlloc_5125_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5125_, 0, v_a_5119_);
v___x_5124_ = v_reuseFailAlloc_5125_;
goto v_reusejp_5123_;
}
v_reusejp_5123_:
{
return v___x_5124_;
}
}
}
}
else
{
lean_object* v_a_5127_; lean_object* v___x_5129_; uint8_t v_isShared_5130_; uint8_t v_isSharedCheck_5134_; 
lean_dec(v_goal_5093_);
lean_dec(v_goal_5082_);
lean_dec_ref(v_unfold_x3f_5081_);
v_a_5127_ = lean_ctor_get(v___x_5100_, 0);
v_isSharedCheck_5134_ = !lean_is_exclusive(v___x_5100_);
if (v_isSharedCheck_5134_ == 0)
{
v___x_5129_ = v___x_5100_;
v_isShared_5130_ = v_isSharedCheck_5134_;
goto v_resetjp_5128_;
}
else
{
lean_inc(v_a_5127_);
lean_dec(v___x_5100_);
v___x_5129_ = lean_box(0);
v_isShared_5130_ = v_isSharedCheck_5134_;
goto v_resetjp_5128_;
}
v_resetjp_5128_:
{
lean_object* v___x_5132_; 
if (v_isShared_5130_ == 0)
{
v___x_5132_ = v___x_5129_;
goto v_reusejp_5131_;
}
else
{
lean_object* v_reuseFailAlloc_5133_; 
v_reuseFailAlloc_5133_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5133_, 0, v_a_5127_);
v___x_5132_ = v_reuseFailAlloc_5133_;
goto v_reusejp_5131_;
}
v_reusejp_5131_:
{
return v___x_5132_;
}
}
}
}
}
else
{
lean_object* v_a_5137_; lean_object* v___x_5139_; uint8_t v_isShared_5140_; uint8_t v_isSharedCheck_5144_; 
lean_dec(v_goal_5082_);
lean_dec_ref(v_unfold_x3f_5081_);
v_a_5137_ = lean_ctor_get(v___x_5090_, 0);
v_isSharedCheck_5144_ = !lean_is_exclusive(v___x_5090_);
if (v_isSharedCheck_5144_ == 0)
{
v___x_5139_ = v___x_5090_;
v_isShared_5140_ = v_isSharedCheck_5144_;
goto v_resetjp_5138_;
}
else
{
lean_inc(v_a_5137_);
lean_dec(v___x_5090_);
v___x_5139_ = lean_box(0);
v_isShared_5140_ = v_isSharedCheck_5144_;
goto v_resetjp_5138_;
}
v_resetjp_5138_:
{
lean_object* v___x_5142_; 
if (v_isShared_5140_ == 0)
{
v___x_5142_ = v___x_5139_;
goto v_reusejp_5141_;
}
else
{
lean_object* v_reuseFailAlloc_5143_; 
v_reuseFailAlloc_5143_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5143_, 0, v_a_5137_);
v___x_5142_ = v_reuseFailAlloc_5143_;
goto v_reusejp_5141_;
}
v_reusejp_5141_:
{
return v___x_5142_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyStarS___lam__0___boxed(lean_object* v_unfold_x3f_5145_, lean_object* v_goal_5146_, lean_object* v___y_5147_, lean_object* v___y_5148_, lean_object* v___y_5149_, lean_object* v___y_5150_, lean_object* v___y_5151_, lean_object* v___y_5152_, lean_object* v___y_5153_){
_start:
{
lean_object* v_res_5154_; 
v_res_5154_ = lp_aesop_Aesop_unfoldManyStarS___lam__0(v_unfold_x3f_5145_, v_goal_5146_, v___y_5147_, v___y_5148_, v___y_5149_, v___y_5150_, v___y_5151_, v___y_5152_);
lean_dec(v___y_5152_);
lean_dec_ref(v___y_5151_);
lean_dec(v___y_5150_);
lean_dec_ref(v___y_5149_);
lean_dec(v___y_5148_);
lean_dec(v___y_5147_);
return v_res_5154_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyStarS(lean_object* v_goal_5155_, lean_object* v_unfold_x3f_5156_, lean_object* v_a_5157_, lean_object* v_a_5158_, lean_object* v_a_5159_, lean_object* v_a_5160_, lean_object* v_a_5161_, lean_object* v_a_5162_){
_start:
{
lean_object* v___f_5164_; lean_object* v___x_5165_; 
lean_inc(v_goal_5155_);
v___f_5164_ = lean_alloc_closure((void*)(lp_aesop_Aesop_unfoldManyStarS___lam__0___boxed), 9, 2);
lean_closure_set(v___f_5164_, 0, v_unfold_x3f_5156_);
lean_closure_set(v___f_5164_, 1, v_goal_5155_);
v___x_5165_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_unfoldManyStarS_spec__1___redArg(v_goal_5155_, v___f_5164_, v_a_5157_, v_a_5158_, v_a_5159_, v_a_5160_, v_a_5161_, v_a_5162_);
return v___x_5165_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_unfoldManyStarS___boxed(lean_object* v_goal_5166_, lean_object* v_unfold_x3f_5167_, lean_object* v_a_5168_, lean_object* v_a_5169_, lean_object* v_a_5170_, lean_object* v_a_5171_, lean_object* v_a_5172_, lean_object* v_a_5173_, lean_object* v_a_5174_){
_start:
{
lean_object* v_res_5175_; 
v_res_5175_ = lp_aesop_Aesop_unfoldManyStarS(v_goal_5166_, v_unfold_x3f_5167_, v_a_5168_, v_a_5169_, v_a_5170_, v_a_5171_, v_a_5172_, v_a_5173_);
lean_dec(v_a_5173_);
lean_dec_ref(v_a_5172_);
lean_dec(v_a_5171_);
lean_dec_ref(v_a_5170_);
lean_dec(v_a_5169_);
lean_dec(v_a_5168_);
return v_res_5175_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__1_spec__5(lean_object* v_unfold_x3f_5176_, lean_object* v_as_5177_, size_t v_sz_5178_, size_t v_i_5179_, lean_object* v_b_5180_, lean_object* v___y_5181_, lean_object* v___y_5182_, lean_object* v___y_5183_, lean_object* v___y_5184_, lean_object* v___y_5185_, lean_object* v___y_5186_){
_start:
{
lean_object* v___x_5188_; 
v___x_5188_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__1_spec__5___redArg(v_unfold_x3f_5176_, v_as_5177_, v_sz_5178_, v_i_5179_, v_b_5180_, v___y_5181_, v___y_5183_, v___y_5184_, v___y_5185_, v___y_5186_);
return v___x_5188_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__1_spec__5___boxed(lean_object* v_unfold_x3f_5189_, lean_object* v_as_5190_, lean_object* v_sz_5191_, lean_object* v_i_5192_, lean_object* v_b_5193_, lean_object* v___y_5194_, lean_object* v___y_5195_, lean_object* v___y_5196_, lean_object* v___y_5197_, lean_object* v___y_5198_, lean_object* v___y_5199_, lean_object* v___y_5200_){
_start:
{
size_t v_sz_boxed_5201_; size_t v_i_boxed_5202_; lean_object* v_res_5203_; 
v_sz_boxed_5201_ = lean_unbox_usize(v_sz_5191_);
lean_dec(v_sz_5191_);
v_i_boxed_5202_ = lean_unbox_usize(v_i_5192_);
lean_dec(v_i_5192_);
v_res_5203_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__1_spec__5(v_unfold_x3f_5189_, v_as_5190_, v_sz_boxed_5201_, v_i_boxed_5202_, v_b_5193_, v___y_5194_, v___y_5195_, v___y_5196_, v___y_5197_, v___y_5198_, v___y_5199_);
lean_dec(v___y_5199_);
lean_dec_ref(v___y_5198_);
lean_dec(v___y_5197_);
lean_dec_ref(v___y_5196_);
lean_dec(v___y_5195_);
lean_dec(v___y_5194_);
lean_dec_ref(v_as_5190_);
return v_res_5203_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0_spec__3_spec__4(lean_object* v_unfold_x3f_5204_, lean_object* v_as_5205_, size_t v_sz_5206_, size_t v_i_5207_, lean_object* v_b_5208_, lean_object* v___y_5209_, lean_object* v___y_5210_, lean_object* v___y_5211_, lean_object* v___y_5212_, lean_object* v___y_5213_, lean_object* v___y_5214_){
_start:
{
lean_object* v___x_5216_; 
v___x_5216_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0_spec__3_spec__4___redArg(v_unfold_x3f_5204_, v_as_5205_, v_sz_5206_, v_i_5207_, v_b_5208_, v___y_5209_, v___y_5211_, v___y_5212_, v___y_5213_, v___y_5214_);
return v___x_5216_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0_spec__3_spec__4___boxed(lean_object* v_unfold_x3f_5217_, lean_object* v_as_5218_, lean_object* v_sz_5219_, lean_object* v_i_5220_, lean_object* v_b_5221_, lean_object* v___y_5222_, lean_object* v___y_5223_, lean_object* v___y_5224_, lean_object* v___y_5225_, lean_object* v___y_5226_, lean_object* v___y_5227_, lean_object* v___y_5228_){
_start:
{
size_t v_sz_boxed_5229_; size_t v_i_boxed_5230_; lean_object* v_res_5231_; 
v_sz_boxed_5229_ = lean_unbox_usize(v_sz_5219_);
lean_dec(v_sz_5219_);
v_i_boxed_5230_ = lean_unbox_usize(v_i_5220_);
lean_dec(v_i_5220_);
v_res_5231_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_unfoldManyStarS_spec__0_spec__0_spec__3_spec__4(v_unfold_x3f_5217_, v_as_5218_, v_sz_boxed_5229_, v_i_boxed_5230_, v_b_5221_, v___y_5222_, v___y_5223_, v___y_5224_, v___y_5225_, v___y_5226_, v___y_5227_);
lean_dec(v___y_5227_);
lean_dec_ref(v___y_5226_);
lean_dec(v___y_5225_);
lean_dec_ref(v___y_5224_);
lean_dec(v___y_5223_);
lean_dec(v___y_5222_);
lean_dec_ref(v_as_5218_);
return v_res_5231_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_introsS_tacticBuilder(lean_object* v_x_5232_, lean_object* v_a_5233_, lean_object* v_a_5234_, lean_object* v_a_5235_, lean_object* v_a_5236_){
_start:
{
lean_object* v_fst_5238_; lean_object* v_snd_5239_; uint8_t v___x_5240_; lean_object* v___x_5241_; 
v_fst_5238_ = lean_ctor_get(v_x_5232_, 0);
lean_inc(v_fst_5238_);
v_snd_5239_ = lean_ctor_get(v_x_5232_, 1);
lean_inc(v_snd_5239_);
lean_dec_ref(v_x_5232_);
v___x_5240_ = 1;
v___x_5241_ = lp_aesop_Aesop_Script_TacticBuilder_intros(v_fst_5238_, v_snd_5239_, v___x_5240_, v_a_5233_, v_a_5234_, v_a_5235_, v_a_5236_);
return v___x_5241_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_introsS_tacticBuilder___boxed(lean_object* v_x_5242_, lean_object* v_a_5243_, lean_object* v_a_5244_, lean_object* v_a_5245_, lean_object* v_a_5246_, lean_object* v_a_5247_){
_start:
{
lean_object* v_res_5248_; 
v_res_5248_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_introsS_tacticBuilder(v_x_5242_, v_a_5243_, v_a_5244_, v_a_5245_, v_a_5246_);
lean_dec(v_a_5246_);
lean_dec_ref(v_a_5245_);
lean_dec(v_a_5244_);
lean_dec_ref(v_a_5243_);
return v_res_5248_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Options_set___at___00Lean_Option_set___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__0_spec__1(lean_object* v_o_5252_, lean_object* v_k_5253_, uint8_t v_v_5254_){
_start:
{
lean_object* v_map_5255_; uint8_t v_hasTrace_5256_; lean_object* v___x_5258_; uint8_t v_isShared_5259_; uint8_t v_isSharedCheck_5270_; 
v_map_5255_ = lean_ctor_get(v_o_5252_, 0);
v_hasTrace_5256_ = lean_ctor_get_uint8(v_o_5252_, sizeof(void*)*1);
v_isSharedCheck_5270_ = !lean_is_exclusive(v_o_5252_);
if (v_isSharedCheck_5270_ == 0)
{
v___x_5258_ = v_o_5252_;
v_isShared_5259_ = v_isSharedCheck_5270_;
goto v_resetjp_5257_;
}
else
{
lean_inc(v_map_5255_);
lean_dec(v_o_5252_);
v___x_5258_ = lean_box(0);
v_isShared_5259_ = v_isSharedCheck_5270_;
goto v_resetjp_5257_;
}
v_resetjp_5257_:
{
lean_object* v___x_5260_; lean_object* v___x_5261_; 
v___x_5260_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_5260_, 0, v_v_5254_);
lean_inc(v_k_5253_);
v___x_5261_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v_k_5253_, v___x_5260_, v_map_5255_);
if (v_hasTrace_5256_ == 0)
{
lean_object* v___x_5262_; uint8_t v___x_5263_; lean_object* v___x_5265_; 
v___x_5262_ = ((lean_object*)(lp_aesop_Lean_Options_set___at___00Lean_Option_set___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__0_spec__1___closed__1));
v___x_5263_ = l_Lean_Name_isPrefixOf(v___x_5262_, v_k_5253_);
lean_dec(v_k_5253_);
if (v_isShared_5259_ == 0)
{
lean_ctor_set(v___x_5258_, 0, v___x_5261_);
v___x_5265_ = v___x_5258_;
goto v_reusejp_5264_;
}
else
{
lean_object* v_reuseFailAlloc_5266_; 
v_reuseFailAlloc_5266_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_5266_, 0, v___x_5261_);
v___x_5265_ = v_reuseFailAlloc_5266_;
goto v_reusejp_5264_;
}
v_reusejp_5264_:
{
lean_ctor_set_uint8(v___x_5265_, sizeof(void*)*1, v___x_5263_);
return v___x_5265_;
}
}
else
{
lean_object* v___x_5268_; 
lean_dec(v_k_5253_);
if (v_isShared_5259_ == 0)
{
lean_ctor_set(v___x_5258_, 0, v___x_5261_);
v___x_5268_ = v___x_5258_;
goto v_reusejp_5267_;
}
else
{
lean_object* v_reuseFailAlloc_5269_; 
v_reuseFailAlloc_5269_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_5269_, 0, v___x_5261_);
lean_ctor_set_uint8(v_reuseFailAlloc_5269_, sizeof(void*)*1, v_hasTrace_5256_);
v___x_5268_ = v_reuseFailAlloc_5269_;
goto v_reusejp_5267_;
}
v_reusejp_5267_:
{
return v___x_5268_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Options_set___at___00Lean_Option_set___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__0_spec__1___boxed(lean_object* v_o_5271_, lean_object* v_k_5272_, lean_object* v_v_5273_){
_start:
{
uint8_t v_v_boxed_5274_; lean_object* v_res_5275_; 
v_v_boxed_5274_ = lean_unbox(v_v_5273_);
v_res_5275_ = lp_aesop_Lean_Options_set___at___00Lean_Option_set___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__0_spec__1(v_o_5271_, v_k_5272_, v_v_boxed_5274_);
return v_res_5275_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_set___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__0(lean_object* v_opts_5276_, lean_object* v_opt_5277_, uint8_t v_val_5278_){
_start:
{
lean_object* v_name_5279_; lean_object* v___x_5280_; 
v_name_5279_ = lean_ctor_get(v_opt_5277_, 0);
lean_inc(v_name_5279_);
lean_dec_ref(v_opt_5277_);
v___x_5280_ = lp_aesop_Lean_Options_set___at___00Lean_Option_set___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__0_spec__1(v_opts_5276_, v_name_5279_, v_val_5278_);
return v___x_5280_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_set___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__0___boxed(lean_object* v_opts_5281_, lean_object* v_opt_5282_, lean_object* v_val_5283_){
_start:
{
uint8_t v_val_boxed_5284_; lean_object* v_res_5285_; 
v_val_boxed_5284_ = lean_unbox(v_val_5283_);
v_res_5285_ = lp_aesop_Lean_Option_set___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__0(v_opts_5281_, v_opt_5282_, v_val_boxed_5284_);
return v_res_5285_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__2(lean_object* v_opts_5286_, lean_object* v_opt_5287_){
_start:
{
lean_object* v_name_5288_; lean_object* v_defValue_5289_; lean_object* v_map_5290_; lean_object* v___x_5291_; 
v_name_5288_ = lean_ctor_get(v_opt_5287_, 0);
v_defValue_5289_ = lean_ctor_get(v_opt_5287_, 1);
v_map_5290_ = lean_ctor_get(v_opts_5286_, 0);
v___x_5291_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_5290_, v_name_5288_);
if (lean_obj_tag(v___x_5291_) == 0)
{
lean_inc(v_defValue_5289_);
return v_defValue_5289_;
}
else
{
lean_object* v_val_5292_; 
v_val_5292_ = lean_ctor_get(v___x_5291_, 0);
lean_inc(v_val_5292_);
lean_dec_ref_known(v___x_5291_, 1);
if (lean_obj_tag(v_val_5292_) == 3)
{
lean_object* v_v_5293_; 
v_v_5293_ = lean_ctor_get(v_val_5292_, 0);
lean_inc(v_v_5293_);
lean_dec_ref_known(v_val_5292_, 1);
return v_v_5293_;
}
else
{
lean_dec(v_val_5292_);
lean_inc(v_defValue_5289_);
return v_defValue_5289_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__2___boxed(lean_object* v_opts_5294_, lean_object* v_opt_5295_){
_start:
{
lean_object* v_res_5296_; 
v_res_5296_ = lp_aesop_Lean_Option_get___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__2(v_opts_5294_, v_opt_5295_);
lean_dec_ref(v_opt_5295_);
lean_dec_ref(v_opts_5294_);
return v_res_5296_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__1(lean_object* v_opts_5297_, lean_object* v_opt_5298_){
_start:
{
lean_object* v_name_5299_; lean_object* v_defValue_5300_; lean_object* v_map_5301_; lean_object* v___x_5302_; 
v_name_5299_ = lean_ctor_get(v_opt_5298_, 0);
v_defValue_5300_ = lean_ctor_get(v_opt_5298_, 1);
v_map_5301_ = lean_ctor_get(v_opts_5297_, 0);
v___x_5302_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_5301_, v_name_5299_);
if (lean_obj_tag(v___x_5302_) == 0)
{
uint8_t v___x_5303_; 
v___x_5303_ = lean_unbox(v_defValue_5300_);
return v___x_5303_;
}
else
{
lean_object* v_val_5304_; 
v_val_5304_ = lean_ctor_get(v___x_5302_, 0);
lean_inc(v_val_5304_);
lean_dec_ref_known(v___x_5302_, 1);
if (lean_obj_tag(v_val_5304_) == 1)
{
uint8_t v_v_5305_; 
v_v_5305_ = lean_ctor_get_uint8(v_val_5304_, 0);
lean_dec_ref_known(v_val_5304_, 0);
return v_v_5305_;
}
else
{
uint8_t v___x_5306_; 
lean_dec(v_val_5304_);
v___x_5306_ = lean_unbox(v_defValue_5300_);
return v___x_5306_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__1___boxed(lean_object* v_opts_5307_, lean_object* v_opt_5308_){
_start:
{
uint8_t v_res_5309_; lean_object* v_r_5310_; 
v_res_5309_ = lp_aesop_Lean_Option_get___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__1(v_opts_5307_, v_opt_5308_);
lean_dec_ref(v_opt_5308_);
lean_dec_ref(v_opts_5307_);
v_r_5310_ = lean_box(v_res_5309_);
return v_r_5310_;
}
}
static lean_object* _init_lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_5311_; 
v___x_5311_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_5311_;
}
}
static lean_object* _init_lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg___closed__1(void){
_start:
{
lean_object* v___x_5312_; lean_object* v___x_5313_; 
v___x_5312_ = lean_obj_once(&lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg___closed__0, &lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg___closed__0_once, _init_lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg___closed__0);
v___x_5313_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5313_, 0, v___x_5312_);
return v___x_5313_;
}
}
static lean_object* _init_lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg___closed__2(void){
_start:
{
lean_object* v___x_5314_; lean_object* v___x_5315_; 
v___x_5314_ = lean_obj_once(&lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg___closed__1, &lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg___closed__1_once, _init_lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg___closed__1);
v___x_5315_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5315_, 0, v___x_5314_);
lean_ctor_set(v___x_5315_, 1, v___x_5314_);
return v___x_5315_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg(lean_object* v_x_5316_, lean_object* v___y_5317_, lean_object* v___y_5318_, lean_object* v___y_5319_, lean_object* v___y_5320_){
_start:
{
lean_object* v___x_5322_; lean_object* v_fileName_5323_; lean_object* v_fileMap_5324_; lean_object* v_options_5325_; lean_object* v_currRecDepth_5326_; lean_object* v_ref_5327_; lean_object* v_currNamespace_5328_; lean_object* v_openDecls_5329_; lean_object* v_initHeartbeats_5330_; lean_object* v_maxHeartbeats_5331_; lean_object* v_quotContext_5332_; lean_object* v_currMacroScope_5333_; lean_object* v_cancelTk_x3f_5334_; uint8_t v_suppressElabErrors_5335_; lean_object* v_inheritedTraceOptions_5336_; lean_object* v_env_5337_; lean_object* v___x_5338_; uint8_t v___x_5339_; lean_object* v___x_5340_; lean_object* v___x_5341_; uint8_t v___x_5342_; lean_object* v_fileName_5344_; lean_object* v_fileMap_5345_; lean_object* v_currRecDepth_5346_; lean_object* v_ref_5347_; lean_object* v_currNamespace_5348_; lean_object* v_openDecls_5349_; lean_object* v_initHeartbeats_5350_; lean_object* v_maxHeartbeats_5351_; lean_object* v_quotContext_5352_; lean_object* v_currMacroScope_5353_; lean_object* v_cancelTk_x3f_5354_; uint8_t v_suppressElabErrors_5355_; lean_object* v_inheritedTraceOptions_5356_; lean_object* v___y_5357_; uint8_t v___y_5363_; uint8_t v___x_5384_; 
v___x_5322_ = lean_st_ref_get(v___y_5320_);
v_fileName_5323_ = lean_ctor_get(v___y_5319_, 0);
v_fileMap_5324_ = lean_ctor_get(v___y_5319_, 1);
v_options_5325_ = lean_ctor_get(v___y_5319_, 2);
v_currRecDepth_5326_ = lean_ctor_get(v___y_5319_, 3);
v_ref_5327_ = lean_ctor_get(v___y_5319_, 5);
v_currNamespace_5328_ = lean_ctor_get(v___y_5319_, 6);
v_openDecls_5329_ = lean_ctor_get(v___y_5319_, 7);
v_initHeartbeats_5330_ = lean_ctor_get(v___y_5319_, 8);
v_maxHeartbeats_5331_ = lean_ctor_get(v___y_5319_, 9);
v_quotContext_5332_ = lean_ctor_get(v___y_5319_, 10);
v_currMacroScope_5333_ = lean_ctor_get(v___y_5319_, 11);
v_cancelTk_x3f_5334_ = lean_ctor_get(v___y_5319_, 12);
v_suppressElabErrors_5335_ = lean_ctor_get_uint8(v___y_5319_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_5336_ = lean_ctor_get(v___y_5319_, 13);
v_env_5337_ = lean_ctor_get(v___x_5322_, 0);
lean_inc_ref(v_env_5337_);
lean_dec(v___x_5322_);
v___x_5338_ = l_Lean_Meta_tactic_hygienic;
v___x_5339_ = 0;
lean_inc_ref(v_options_5325_);
v___x_5340_ = lp_aesop_Lean_Option_set___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__0(v_options_5325_, v___x_5338_, v___x_5339_);
v___x_5341_ = l_Lean_diagnostics;
v___x_5342_ = lp_aesop_Lean_Option_get___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__1(v___x_5340_, v___x_5341_);
v___x_5384_ = l_Lean_Kernel_isDiagnosticsEnabled(v_env_5337_);
lean_dec_ref(v_env_5337_);
if (v___x_5384_ == 0)
{
if (v___x_5342_ == 0)
{
v_fileName_5344_ = v_fileName_5323_;
v_fileMap_5345_ = v_fileMap_5324_;
v_currRecDepth_5346_ = v_currRecDepth_5326_;
v_ref_5347_ = v_ref_5327_;
v_currNamespace_5348_ = v_currNamespace_5328_;
v_openDecls_5349_ = v_openDecls_5329_;
v_initHeartbeats_5350_ = v_initHeartbeats_5330_;
v_maxHeartbeats_5351_ = v_maxHeartbeats_5331_;
v_quotContext_5352_ = v_quotContext_5332_;
v_currMacroScope_5353_ = v_currMacroScope_5333_;
v_cancelTk_x3f_5354_ = v_cancelTk_x3f_5334_;
v_suppressElabErrors_5355_ = v_suppressElabErrors_5335_;
v_inheritedTraceOptions_5356_ = v_inheritedTraceOptions_5336_;
v___y_5357_ = v___y_5320_;
goto v___jp_5343_;
}
else
{
v___y_5363_ = v___x_5384_;
goto v___jp_5362_;
}
}
else
{
v___y_5363_ = v___x_5342_;
goto v___jp_5362_;
}
v___jp_5343_:
{
lean_object* v___x_5358_; lean_object* v___x_5359_; lean_object* v___x_5360_; lean_object* v___x_5361_; 
v___x_5358_ = l_Lean_maxRecDepth;
v___x_5359_ = lp_aesop_Lean_Option_get___at___00Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0_spec__2(v___x_5340_, v___x_5358_);
lean_inc_ref(v_inheritedTraceOptions_5356_);
lean_inc(v_cancelTk_x3f_5354_);
lean_inc(v_currMacroScope_5353_);
lean_inc(v_quotContext_5352_);
lean_inc(v_maxHeartbeats_5351_);
lean_inc(v_initHeartbeats_5350_);
lean_inc(v_openDecls_5349_);
lean_inc(v_currNamespace_5348_);
lean_inc(v_ref_5347_);
lean_inc(v_currRecDepth_5346_);
lean_inc_ref(v_fileMap_5345_);
lean_inc_ref(v_fileName_5344_);
v___x_5360_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_5360_, 0, v_fileName_5344_);
lean_ctor_set(v___x_5360_, 1, v_fileMap_5345_);
lean_ctor_set(v___x_5360_, 2, v___x_5340_);
lean_ctor_set(v___x_5360_, 3, v_currRecDepth_5346_);
lean_ctor_set(v___x_5360_, 4, v___x_5359_);
lean_ctor_set(v___x_5360_, 5, v_ref_5347_);
lean_ctor_set(v___x_5360_, 6, v_currNamespace_5348_);
lean_ctor_set(v___x_5360_, 7, v_openDecls_5349_);
lean_ctor_set(v___x_5360_, 8, v_initHeartbeats_5350_);
lean_ctor_set(v___x_5360_, 9, v_maxHeartbeats_5351_);
lean_ctor_set(v___x_5360_, 10, v_quotContext_5352_);
lean_ctor_set(v___x_5360_, 11, v_currMacroScope_5353_);
lean_ctor_set(v___x_5360_, 12, v_cancelTk_x3f_5354_);
lean_ctor_set(v___x_5360_, 13, v_inheritedTraceOptions_5356_);
lean_ctor_set_uint8(v___x_5360_, sizeof(void*)*14, v___x_5342_);
lean_ctor_set_uint8(v___x_5360_, sizeof(void*)*14 + 1, v_suppressElabErrors_5355_);
lean_inc(v___y_5357_);
lean_inc(v___y_5318_);
lean_inc_ref(v___y_5317_);
v___x_5361_ = lean_apply_5(v_x_5316_, v___y_5317_, v___y_5318_, v___x_5360_, v___y_5357_, lean_box(0));
return v___x_5361_;
}
v___jp_5362_:
{
if (v___y_5363_ == 0)
{
lean_object* v___x_5364_; lean_object* v_env_5365_; lean_object* v_nextMacroScope_5366_; lean_object* v_ngen_5367_; lean_object* v_auxDeclNGen_5368_; lean_object* v_traceState_5369_; lean_object* v_messages_5370_; lean_object* v_infoState_5371_; lean_object* v_snapshotTasks_5372_; lean_object* v___x_5374_; uint8_t v_isShared_5375_; uint8_t v_isSharedCheck_5382_; 
v___x_5364_ = lean_st_ref_take(v___y_5320_);
v_env_5365_ = lean_ctor_get(v___x_5364_, 0);
v_nextMacroScope_5366_ = lean_ctor_get(v___x_5364_, 1);
v_ngen_5367_ = lean_ctor_get(v___x_5364_, 2);
v_auxDeclNGen_5368_ = lean_ctor_get(v___x_5364_, 3);
v_traceState_5369_ = lean_ctor_get(v___x_5364_, 4);
v_messages_5370_ = lean_ctor_get(v___x_5364_, 6);
v_infoState_5371_ = lean_ctor_get(v___x_5364_, 7);
v_snapshotTasks_5372_ = lean_ctor_get(v___x_5364_, 8);
v_isSharedCheck_5382_ = !lean_is_exclusive(v___x_5364_);
if (v_isSharedCheck_5382_ == 0)
{
lean_object* v_unused_5383_; 
v_unused_5383_ = lean_ctor_get(v___x_5364_, 5);
lean_dec(v_unused_5383_);
v___x_5374_ = v___x_5364_;
v_isShared_5375_ = v_isSharedCheck_5382_;
goto v_resetjp_5373_;
}
else
{
lean_inc(v_snapshotTasks_5372_);
lean_inc(v_infoState_5371_);
lean_inc(v_messages_5370_);
lean_inc(v_traceState_5369_);
lean_inc(v_auxDeclNGen_5368_);
lean_inc(v_ngen_5367_);
lean_inc(v_nextMacroScope_5366_);
lean_inc(v_env_5365_);
lean_dec(v___x_5364_);
v___x_5374_ = lean_box(0);
v_isShared_5375_ = v_isSharedCheck_5382_;
goto v_resetjp_5373_;
}
v_resetjp_5373_:
{
lean_object* v___x_5376_; lean_object* v___x_5377_; lean_object* v___x_5379_; 
v___x_5376_ = l_Lean_Kernel_enableDiag(v_env_5365_, v___x_5342_);
v___x_5377_ = lean_obj_once(&lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg___closed__2, &lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg___closed__2_once, _init_lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg___closed__2);
if (v_isShared_5375_ == 0)
{
lean_ctor_set(v___x_5374_, 5, v___x_5377_);
lean_ctor_set(v___x_5374_, 0, v___x_5376_);
v___x_5379_ = v___x_5374_;
goto v_reusejp_5378_;
}
else
{
lean_object* v_reuseFailAlloc_5381_; 
v_reuseFailAlloc_5381_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_5381_, 0, v___x_5376_);
lean_ctor_set(v_reuseFailAlloc_5381_, 1, v_nextMacroScope_5366_);
lean_ctor_set(v_reuseFailAlloc_5381_, 2, v_ngen_5367_);
lean_ctor_set(v_reuseFailAlloc_5381_, 3, v_auxDeclNGen_5368_);
lean_ctor_set(v_reuseFailAlloc_5381_, 4, v_traceState_5369_);
lean_ctor_set(v_reuseFailAlloc_5381_, 5, v___x_5377_);
lean_ctor_set(v_reuseFailAlloc_5381_, 6, v_messages_5370_);
lean_ctor_set(v_reuseFailAlloc_5381_, 7, v_infoState_5371_);
lean_ctor_set(v_reuseFailAlloc_5381_, 8, v_snapshotTasks_5372_);
v___x_5379_ = v_reuseFailAlloc_5381_;
goto v_reusejp_5378_;
}
v_reusejp_5378_:
{
lean_object* v___x_5380_; 
v___x_5380_ = lean_st_ref_set(v___y_5320_, v___x_5379_);
v_fileName_5344_ = v_fileName_5323_;
v_fileMap_5345_ = v_fileMap_5324_;
v_currRecDepth_5346_ = v_currRecDepth_5326_;
v_ref_5347_ = v_ref_5327_;
v_currNamespace_5348_ = v_currNamespace_5328_;
v_openDecls_5349_ = v_openDecls_5329_;
v_initHeartbeats_5350_ = v_initHeartbeats_5330_;
v_maxHeartbeats_5351_ = v_maxHeartbeats_5331_;
v_quotContext_5352_ = v_quotContext_5332_;
v_currMacroScope_5353_ = v_currMacroScope_5333_;
v_cancelTk_x3f_5354_ = v_cancelTk_x3f_5334_;
v_suppressElabErrors_5355_ = v_suppressElabErrors_5335_;
v_inheritedTraceOptions_5356_ = v_inheritedTraceOptions_5336_;
v___y_5357_ = v___y_5320_;
goto v___jp_5343_;
}
}
}
else
{
v_fileName_5344_ = v_fileName_5323_;
v_fileMap_5345_ = v_fileMap_5324_;
v_currRecDepth_5346_ = v_currRecDepth_5326_;
v_ref_5347_ = v_ref_5327_;
v_currNamespace_5348_ = v_currNamespace_5328_;
v_openDecls_5349_ = v_openDecls_5329_;
v_initHeartbeats_5350_ = v_initHeartbeats_5330_;
v_maxHeartbeats_5351_ = v_maxHeartbeats_5331_;
v_quotContext_5352_ = v_quotContext_5332_;
v_currMacroScope_5353_ = v_currMacroScope_5333_;
v_cancelTk_x3f_5354_ = v_cancelTk_x3f_5334_;
v_suppressElabErrors_5355_ = v_suppressElabErrors_5335_;
v_inheritedTraceOptions_5356_ = v_inheritedTraceOptions_5336_;
v___y_5357_ = v___y_5320_;
goto v___jp_5343_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg___boxed(lean_object* v_x_5385_, lean_object* v___y_5386_, lean_object* v___y_5387_, lean_object* v___y_5388_, lean_object* v___y_5389_, lean_object* v___y_5390_){
_start:
{
lean_object* v_res_5391_; 
v_res_5391_ = lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg(v_x_5385_, v___y_5386_, v___y_5387_, v___y_5388_, v___y_5389_);
lean_dec(v___y_5389_);
lean_dec_ref(v___y_5388_);
lean_dec(v___y_5387_);
lean_dec_ref(v___y_5386_);
return v_res_5391_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_introsS___lam__2(lean_object* v___x_5392_, lean_object* v___y_5393_, lean_object* v___y_5394_, lean_object* v___y_5395_, lean_object* v___y_5396_){
_start:
{
lean_object* v___x_5398_; 
v___x_5398_ = lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg(v___x_5392_, v___y_5393_, v___y_5394_, v___y_5395_, v___y_5396_);
if (lean_obj_tag(v___x_5398_) == 0)
{
lean_object* v_a_5399_; lean_object* v___x_5401_; uint8_t v_isShared_5402_; uint8_t v_isSharedCheck_5415_; 
v_a_5399_ = lean_ctor_get(v___x_5398_, 0);
v_isSharedCheck_5415_ = !lean_is_exclusive(v___x_5398_);
if (v_isSharedCheck_5415_ == 0)
{
v___x_5401_ = v___x_5398_;
v_isShared_5402_ = v_isSharedCheck_5415_;
goto v_resetjp_5400_;
}
else
{
lean_inc(v_a_5399_);
lean_dec(v___x_5398_);
v___x_5401_ = lean_box(0);
v_isShared_5402_ = v_isSharedCheck_5415_;
goto v_resetjp_5400_;
}
v_resetjp_5400_:
{
lean_object* v_fst_5403_; lean_object* v_snd_5404_; lean_object* v___x_5406_; uint8_t v_isShared_5407_; uint8_t v_isSharedCheck_5414_; 
v_fst_5403_ = lean_ctor_get(v_a_5399_, 0);
v_snd_5404_ = lean_ctor_get(v_a_5399_, 1);
v_isSharedCheck_5414_ = !lean_is_exclusive(v_a_5399_);
if (v_isSharedCheck_5414_ == 0)
{
v___x_5406_ = v_a_5399_;
v_isShared_5407_ = v_isSharedCheck_5414_;
goto v_resetjp_5405_;
}
else
{
lean_inc(v_snd_5404_);
lean_inc(v_fst_5403_);
lean_dec(v_a_5399_);
v___x_5406_ = lean_box(0);
v_isShared_5407_ = v_isSharedCheck_5414_;
goto v_resetjp_5405_;
}
v_resetjp_5405_:
{
lean_object* v___x_5409_; 
if (v_isShared_5407_ == 0)
{
lean_ctor_set(v___x_5406_, 1, v_fst_5403_);
lean_ctor_set(v___x_5406_, 0, v_snd_5404_);
v___x_5409_ = v___x_5406_;
goto v_reusejp_5408_;
}
else
{
lean_object* v_reuseFailAlloc_5413_; 
v_reuseFailAlloc_5413_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5413_, 0, v_snd_5404_);
lean_ctor_set(v_reuseFailAlloc_5413_, 1, v_fst_5403_);
v___x_5409_ = v_reuseFailAlloc_5413_;
goto v_reusejp_5408_;
}
v_reusejp_5408_:
{
lean_object* v___x_5411_; 
if (v_isShared_5402_ == 0)
{
lean_ctor_set(v___x_5401_, 0, v___x_5409_);
v___x_5411_ = v___x_5401_;
goto v_reusejp_5410_;
}
else
{
lean_object* v_reuseFailAlloc_5412_; 
v_reuseFailAlloc_5412_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5412_, 0, v___x_5409_);
v___x_5411_ = v_reuseFailAlloc_5412_;
goto v_reusejp_5410_;
}
v_reusejp_5410_:
{
return v___x_5411_;
}
}
}
}
}
else
{
lean_object* v_a_5416_; lean_object* v___x_5418_; uint8_t v_isShared_5419_; uint8_t v_isSharedCheck_5423_; 
v_a_5416_ = lean_ctor_get(v___x_5398_, 0);
v_isSharedCheck_5423_ = !lean_is_exclusive(v___x_5398_);
if (v_isSharedCheck_5423_ == 0)
{
v___x_5418_ = v___x_5398_;
v_isShared_5419_ = v_isSharedCheck_5423_;
goto v_resetjp_5417_;
}
else
{
lean_inc(v_a_5416_);
lean_dec(v___x_5398_);
v___x_5418_ = lean_box(0);
v_isShared_5419_ = v_isSharedCheck_5423_;
goto v_resetjp_5417_;
}
v_resetjp_5417_:
{
lean_object* v___x_5421_; 
if (v_isShared_5419_ == 0)
{
v___x_5421_ = v___x_5418_;
goto v_reusejp_5420_;
}
else
{
lean_object* v_reuseFailAlloc_5422_; 
v_reuseFailAlloc_5422_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5422_, 0, v_a_5416_);
v___x_5421_ = v_reuseFailAlloc_5422_;
goto v_reusejp_5420_;
}
v_reusejp_5420_:
{
return v___x_5421_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_introsS___lam__2___boxed(lean_object* v___x_5424_, lean_object* v___y_5425_, lean_object* v___y_5426_, lean_object* v___y_5427_, lean_object* v___y_5428_, lean_object* v___y_5429_){
_start:
{
lean_object* v_res_5430_; 
v_res_5430_ = lp_aesop_Aesop_introsS___lam__2(v___x_5424_, v___y_5425_, v___y_5426_, v___y_5427_, v___y_5428_);
lean_dec(v___y_5428_);
lean_dec_ref(v___y_5427_);
lean_dec(v___y_5426_);
lean_dec_ref(v___y_5425_);
return v_res_5430_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_introsS(lean_object* v_goal_5432_, lean_object* v_a_5433_, lean_object* v_a_5434_, lean_object* v_a_5435_, lean_object* v_a_5436_, lean_object* v_a_5437_, lean_object* v_a_5438_){
_start:
{
lean_object* v___f_5440_; lean_object* v___f_5441_; lean_object* v___x_5442_; lean_object* v___x_5443_; lean_object* v___f_5444_; lean_object* v___x_5445_; 
v___f_5440_ = ((lean_object*)(lp_aesop_Aesop_assertHypothesisS___closed__0));
v___f_5441_ = ((lean_object*)(lp_aesop_Aesop_tryClearManyS___closed__0));
v___x_5442_ = ((lean_object*)(lp_aesop_Aesop_introsS___closed__0));
lean_inc(v_goal_5432_);
v___x_5443_ = lean_alloc_closure((void*)(l_Lean_MVarId_intros___boxed), 6, 1);
lean_closure_set(v___x_5443_, 0, v_goal_5432_);
v___f_5444_ = lean_alloc_closure((void*)(lp_aesop_Aesop_introsS___lam__2___boxed), 6, 1);
lean_closure_set(v___f_5444_, 0, v___x_5443_);
v___x_5445_ = lp_aesop_Aesop_withScriptStep___redArg(v_goal_5432_, v___f_5440_, v___f_5441_, v___x_5442_, v___f_5444_, v_a_5433_, v_a_5434_, v_a_5435_, v_a_5436_, v_a_5437_, v_a_5438_);
return v___x_5445_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_introsS___boxed(lean_object* v_goal_5446_, lean_object* v_a_5447_, lean_object* v_a_5448_, lean_object* v_a_5449_, lean_object* v_a_5450_, lean_object* v_a_5451_, lean_object* v_a_5452_, lean_object* v_a_5453_){
_start:
{
lean_object* v_res_5454_; 
v_res_5454_ = lp_aesop_Aesop_introsS(v_goal_5446_, v_a_5447_, v_a_5448_, v_a_5449_, v_a_5450_, v_a_5451_, v_a_5452_);
lean_dec(v_a_5452_);
lean_dec_ref(v_a_5451_);
lean_dec(v_a_5450_);
lean_dec_ref(v_a_5449_);
lean_dec(v_a_5448_);
lean_dec(v_a_5447_);
return v_res_5454_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0(lean_object* v_00_u03b1_5455_, lean_object* v_x_5456_, lean_object* v___y_5457_, lean_object* v___y_5458_, lean_object* v___y_5459_, lean_object* v___y_5460_){
_start:
{
lean_object* v___x_5462_; 
v___x_5462_ = lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg(v_x_5456_, v___y_5457_, v___y_5458_, v___y_5459_, v___y_5460_);
return v___x_5462_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___boxed(lean_object* v_00_u03b1_5463_, lean_object* v_x_5464_, lean_object* v___y_5465_, lean_object* v___y_5466_, lean_object* v___y_5467_, lean_object* v___y_5468_, lean_object* v___y_5469_){
_start:
{
lean_object* v_res_5470_; 
v_res_5470_ = lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0(v_00_u03b1_5463_, v_x_5464_, v___y_5465_, v___y_5466_, v___y_5467_, v___y_5468_);
lean_dec(v___y_5468_);
lean_dec_ref(v___y_5467_);
lean_dec(v___y_5466_);
lean_dec_ref(v___y_5465_);
return v_res_5470_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_introsUnfoldingS_tacticBuilder(uint8_t v_md_5471_, lean_object* v_x_5472_, lean_object* v_a_5473_, lean_object* v_a_5474_, lean_object* v_a_5475_, lean_object* v_a_5476_){
_start:
{
lean_object* v_fst_5478_; lean_object* v_snd_5479_; lean_object* v___x_5480_; 
v_fst_5478_ = lean_ctor_get(v_x_5472_, 0);
lean_inc(v_fst_5478_);
v_snd_5479_ = lean_ctor_get(v_x_5472_, 1);
lean_inc(v_snd_5479_);
lean_dec_ref(v_x_5472_);
v___x_5480_ = lp_aesop_Aesop_Script_TacticBuilder_intros(v_fst_5478_, v_snd_5479_, v_md_5471_, v_a_5473_, v_a_5474_, v_a_5475_, v_a_5476_);
return v___x_5480_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_introsUnfoldingS_tacticBuilder___boxed(lean_object* v_md_5481_, lean_object* v_x_5482_, lean_object* v_a_5483_, lean_object* v_a_5484_, lean_object* v_a_5485_, lean_object* v_a_5486_, lean_object* v_a_5487_){
_start:
{
uint8_t v_md_boxed_5488_; lean_object* v_res_5489_; 
v_md_boxed_5488_ = lean_unbox(v_md_5481_);
v_res_5489_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_introsUnfoldingS_tacticBuilder(v_md_boxed_5488_, v_x_5482_, v_a_5483_, v_a_5484_, v_a_5485_, v_a_5486_);
lean_dec(v_a_5486_);
lean_dec_ref(v_a_5485_);
lean_dec(v_a_5484_);
lean_dec_ref(v_a_5483_);
return v_res_5489_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_introsUnfoldingS___lam__2(uint8_t v_md_5490_, lean_object* v___x_5491_, lean_object* v___y_5492_, lean_object* v___y_5493_, lean_object* v___y_5494_, lean_object* v___y_5495_){
_start:
{
lean_object* v_keyedConfig_5497_; uint8_t v_trackZetaDelta_5498_; lean_object* v_zetaDeltaSet_5499_; lean_object* v_lctx_5500_; lean_object* v_localInstances_5501_; lean_object* v_defEqCtx_x3f_5502_; lean_object* v_synthPendingDepth_5503_; lean_object* v_customCanUnfoldPredicate_x3f_5504_; uint8_t v_univApprox_5505_; uint8_t v_inTypeClassResolution_5506_; uint8_t v_cacheInferType_5507_; lean_object* v___x_5509_; uint8_t v_isShared_5510_; uint8_t v_isSharedCheck_5541_; 
v_keyedConfig_5497_ = lean_ctor_get(v___y_5492_, 0);
v_trackZetaDelta_5498_ = lean_ctor_get_uint8(v___y_5492_, sizeof(void*)*7);
v_zetaDeltaSet_5499_ = lean_ctor_get(v___y_5492_, 1);
v_lctx_5500_ = lean_ctor_get(v___y_5492_, 2);
v_localInstances_5501_ = lean_ctor_get(v___y_5492_, 3);
v_defEqCtx_x3f_5502_ = lean_ctor_get(v___y_5492_, 4);
v_synthPendingDepth_5503_ = lean_ctor_get(v___y_5492_, 5);
v_customCanUnfoldPredicate_x3f_5504_ = lean_ctor_get(v___y_5492_, 6);
v_univApprox_5505_ = lean_ctor_get_uint8(v___y_5492_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_5506_ = lean_ctor_get_uint8(v___y_5492_, sizeof(void*)*7 + 2);
v_cacheInferType_5507_ = lean_ctor_get_uint8(v___y_5492_, sizeof(void*)*7 + 3);
v_isSharedCheck_5541_ = !lean_is_exclusive(v___y_5492_);
if (v_isSharedCheck_5541_ == 0)
{
v___x_5509_ = v___y_5492_;
v_isShared_5510_ = v_isSharedCheck_5541_;
goto v_resetjp_5508_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_5504_);
lean_inc(v_synthPendingDepth_5503_);
lean_inc(v_defEqCtx_x3f_5502_);
lean_inc(v_localInstances_5501_);
lean_inc(v_lctx_5500_);
lean_inc(v_zetaDeltaSet_5499_);
lean_inc(v_keyedConfig_5497_);
lean_dec(v___y_5492_);
v___x_5509_ = lean_box(0);
v_isShared_5510_ = v_isSharedCheck_5541_;
goto v_resetjp_5508_;
}
v_resetjp_5508_:
{
lean_object* v___x_5511_; lean_object* v___x_5513_; 
v___x_5511_ = l_Lean_Meta_ConfigWithKey_setTransparency(v_md_5490_, v_keyedConfig_5497_);
if (v_isShared_5510_ == 0)
{
lean_ctor_set(v___x_5509_, 0, v___x_5511_);
v___x_5513_ = v___x_5509_;
goto v_reusejp_5512_;
}
else
{
lean_object* v_reuseFailAlloc_5540_; 
v_reuseFailAlloc_5540_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_5540_, 0, v___x_5511_);
lean_ctor_set(v_reuseFailAlloc_5540_, 1, v_zetaDeltaSet_5499_);
lean_ctor_set(v_reuseFailAlloc_5540_, 2, v_lctx_5500_);
lean_ctor_set(v_reuseFailAlloc_5540_, 3, v_localInstances_5501_);
lean_ctor_set(v_reuseFailAlloc_5540_, 4, v_defEqCtx_x3f_5502_);
lean_ctor_set(v_reuseFailAlloc_5540_, 5, v_synthPendingDepth_5503_);
lean_ctor_set(v_reuseFailAlloc_5540_, 6, v_customCanUnfoldPredicate_x3f_5504_);
lean_ctor_set_uint8(v_reuseFailAlloc_5540_, sizeof(void*)*7, v_trackZetaDelta_5498_);
lean_ctor_set_uint8(v_reuseFailAlloc_5540_, sizeof(void*)*7 + 1, v_univApprox_5505_);
lean_ctor_set_uint8(v_reuseFailAlloc_5540_, sizeof(void*)*7 + 2, v_inTypeClassResolution_5506_);
lean_ctor_set_uint8(v_reuseFailAlloc_5540_, sizeof(void*)*7 + 3, v_cacheInferType_5507_);
v___x_5513_ = v_reuseFailAlloc_5540_;
goto v_reusejp_5512_;
}
v_reusejp_5512_:
{
lean_object* v___x_5514_; 
v___x_5514_ = lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___redArg(v___x_5491_, v___x_5513_, v___y_5493_, v___y_5494_, v___y_5495_);
lean_dec_ref(v___x_5513_);
if (lean_obj_tag(v___x_5514_) == 0)
{
lean_object* v_a_5515_; lean_object* v___x_5517_; uint8_t v_isShared_5518_; uint8_t v_isSharedCheck_5531_; 
v_a_5515_ = lean_ctor_get(v___x_5514_, 0);
v_isSharedCheck_5531_ = !lean_is_exclusive(v___x_5514_);
if (v_isSharedCheck_5531_ == 0)
{
v___x_5517_ = v___x_5514_;
v_isShared_5518_ = v_isSharedCheck_5531_;
goto v_resetjp_5516_;
}
else
{
lean_inc(v_a_5515_);
lean_dec(v___x_5514_);
v___x_5517_ = lean_box(0);
v_isShared_5518_ = v_isSharedCheck_5531_;
goto v_resetjp_5516_;
}
v_resetjp_5516_:
{
lean_object* v_fst_5519_; lean_object* v_snd_5520_; lean_object* v___x_5522_; uint8_t v_isShared_5523_; uint8_t v_isSharedCheck_5530_; 
v_fst_5519_ = lean_ctor_get(v_a_5515_, 0);
v_snd_5520_ = lean_ctor_get(v_a_5515_, 1);
v_isSharedCheck_5530_ = !lean_is_exclusive(v_a_5515_);
if (v_isSharedCheck_5530_ == 0)
{
v___x_5522_ = v_a_5515_;
v_isShared_5523_ = v_isSharedCheck_5530_;
goto v_resetjp_5521_;
}
else
{
lean_inc(v_snd_5520_);
lean_inc(v_fst_5519_);
lean_dec(v_a_5515_);
v___x_5522_ = lean_box(0);
v_isShared_5523_ = v_isSharedCheck_5530_;
goto v_resetjp_5521_;
}
v_resetjp_5521_:
{
lean_object* v___x_5525_; 
if (v_isShared_5523_ == 0)
{
lean_ctor_set(v___x_5522_, 1, v_fst_5519_);
lean_ctor_set(v___x_5522_, 0, v_snd_5520_);
v___x_5525_ = v___x_5522_;
goto v_reusejp_5524_;
}
else
{
lean_object* v_reuseFailAlloc_5529_; 
v_reuseFailAlloc_5529_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5529_, 0, v_snd_5520_);
lean_ctor_set(v_reuseFailAlloc_5529_, 1, v_fst_5519_);
v___x_5525_ = v_reuseFailAlloc_5529_;
goto v_reusejp_5524_;
}
v_reusejp_5524_:
{
lean_object* v___x_5527_; 
if (v_isShared_5518_ == 0)
{
lean_ctor_set(v___x_5517_, 0, v___x_5525_);
v___x_5527_ = v___x_5517_;
goto v_reusejp_5526_;
}
else
{
lean_object* v_reuseFailAlloc_5528_; 
v_reuseFailAlloc_5528_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5528_, 0, v___x_5525_);
v___x_5527_ = v_reuseFailAlloc_5528_;
goto v_reusejp_5526_;
}
v_reusejp_5526_:
{
return v___x_5527_;
}
}
}
}
}
else
{
lean_object* v_a_5532_; lean_object* v___x_5534_; uint8_t v_isShared_5535_; uint8_t v_isSharedCheck_5539_; 
v_a_5532_ = lean_ctor_get(v___x_5514_, 0);
v_isSharedCheck_5539_ = !lean_is_exclusive(v___x_5514_);
if (v_isSharedCheck_5539_ == 0)
{
v___x_5534_ = v___x_5514_;
v_isShared_5535_ = v_isSharedCheck_5539_;
goto v_resetjp_5533_;
}
else
{
lean_inc(v_a_5532_);
lean_dec(v___x_5514_);
v___x_5534_ = lean_box(0);
v_isShared_5535_ = v_isSharedCheck_5539_;
goto v_resetjp_5533_;
}
v_resetjp_5533_:
{
lean_object* v___x_5537_; 
if (v_isShared_5535_ == 0)
{
v___x_5537_ = v___x_5534_;
goto v_reusejp_5536_;
}
else
{
lean_object* v_reuseFailAlloc_5538_; 
v_reuseFailAlloc_5538_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5538_, 0, v_a_5532_);
v___x_5537_ = v_reuseFailAlloc_5538_;
goto v_reusejp_5536_;
}
v_reusejp_5536_:
{
return v___x_5537_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_introsUnfoldingS___lam__2___boxed(lean_object* v_md_5542_, lean_object* v___x_5543_, lean_object* v___y_5544_, lean_object* v___y_5545_, lean_object* v___y_5546_, lean_object* v___y_5547_, lean_object* v___y_5548_){
_start:
{
uint8_t v_md_boxed_5549_; lean_object* v_res_5550_; 
v_md_boxed_5549_ = lean_unbox(v_md_5542_);
v_res_5550_ = lp_aesop_Aesop_introsUnfoldingS___lam__2(v_md_boxed_5549_, v___x_5543_, v___y_5544_, v___y_5545_, v___y_5546_, v___y_5547_);
lean_dec(v___y_5547_);
lean_dec_ref(v___y_5546_);
lean_dec(v___y_5545_);
return v_res_5550_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_introsUnfoldingS(lean_object* v_goal_5551_, uint8_t v_md_5552_, lean_object* v_a_5553_, lean_object* v_a_5554_, lean_object* v_a_5555_, lean_object* v_a_5556_, lean_object* v_a_5557_, lean_object* v_a_5558_){
_start:
{
lean_object* v___f_5560_; lean_object* v___f_5561_; lean_object* v___x_5562_; lean_object* v___x_5563_; lean_object* v___x_5564_; lean_object* v___x_5565_; lean_object* v___f_5566_; lean_object* v___x_5567_; 
v___f_5560_ = ((lean_object*)(lp_aesop_Aesop_assertHypothesisS___closed__0));
v___f_5561_ = ((lean_object*)(lp_aesop_Aesop_tryClearManyS___closed__0));
v___x_5562_ = lean_box(v_md_5552_);
v___x_5563_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_introsUnfoldingS_tacticBuilder___boxed), 7, 1);
lean_closure_set(v___x_5563_, 0, v___x_5562_);
lean_inc(v_goal_5551_);
v___x_5564_ = lean_alloc_closure((void*)(lp_aesop_Aesop_introsUnfolding___boxed), 6, 1);
lean_closure_set(v___x_5564_, 0, v_goal_5551_);
v___x_5565_ = lean_box(v_md_5552_);
v___f_5566_ = lean_alloc_closure((void*)(lp_aesop_Aesop_introsUnfoldingS___lam__2___boxed), 7, 2);
lean_closure_set(v___f_5566_, 0, v___x_5565_);
lean_closure_set(v___f_5566_, 1, v___x_5564_);
v___x_5567_ = lp_aesop_Aesop_withScriptStep___redArg(v_goal_5551_, v___f_5560_, v___f_5561_, v___x_5563_, v___f_5566_, v_a_5553_, v_a_5554_, v_a_5555_, v_a_5556_, v_a_5557_, v_a_5558_);
return v___x_5567_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_introsUnfoldingS___boxed(lean_object* v_goal_5568_, lean_object* v_md_5569_, lean_object* v_a_5570_, lean_object* v_a_5571_, lean_object* v_a_5572_, lean_object* v_a_5573_, lean_object* v_a_5574_, lean_object* v_a_5575_, lean_object* v_a_5576_){
_start:
{
uint8_t v_md_boxed_5577_; lean_object* v_res_5578_; 
v_md_boxed_5577_ = lean_unbox(v_md_5569_);
v_res_5578_ = lp_aesop_Aesop_introsUnfoldingS(v_goal_5568_, v_md_boxed_5577_, v_a_5570_, v_a_5571_, v_a_5572_, v_a_5573_, v_a_5574_, v_a_5575_);
lean_dec(v_a_5575_);
lean_dec_ref(v_a_5574_);
lean_dec(v_a_5573_);
lean_dec_ref(v_a_5572_);
lean_dec(v_a_5571_);
lean_dec(v_a_5570_);
return v_res_5578_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_straightLineExtS_tacticBuilder(lean_object* v_r_5579_, lean_object* v_a_5580_, lean_object* v_a_5581_, lean_object* v_a_5582_, lean_object* v_a_5583_){
_start:
{
lean_object* v___x_5585_; 
v___x_5585_ = lp_aesop_Aesop_Script_TacticBuilder_extN(v_r_5579_, v_a_5580_, v_a_5581_, v_a_5582_, v_a_5583_);
return v___x_5585_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_straightLineExtS_tacticBuilder___boxed(lean_object* v_r_5586_, lean_object* v_a_5587_, lean_object* v_a_5588_, lean_object* v_a_5589_, lean_object* v_a_5590_, lean_object* v_a_5591_){
_start:
{
lean_object* v_res_5592_; 
v_res_5592_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_straightLineExtS_tacticBuilder(v_r_5586_, v_a_5587_, v_a_5588_, v_a_5589_, v_a_5590_);
lean_dec(v_a_5590_);
lean_dec_ref(v_a_5589_);
lean_dec(v_a_5588_);
lean_dec_ref(v_a_5587_);
return v_res_5592_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_straightLineExtS_spec__0(size_t v_sz_5593_, size_t v_i_5594_, lean_object* v_bs_5595_){
_start:
{
uint8_t v___x_5596_; 
v___x_5596_ = lean_usize_dec_lt(v_i_5594_, v_sz_5593_);
if (v___x_5596_ == 0)
{
return v_bs_5595_;
}
else
{
lean_object* v_v_5597_; lean_object* v_fst_5598_; lean_object* v___x_5599_; lean_object* v_bs_x27_5600_; size_t v___x_5601_; size_t v___x_5602_; lean_object* v___x_5603_; 
v_v_5597_ = lean_array_uget_borrowed(v_bs_5595_, v_i_5594_);
v_fst_5598_ = lean_ctor_get(v_v_5597_, 0);
lean_inc(v_fst_5598_);
v___x_5599_ = lean_unsigned_to_nat(0u);
v_bs_x27_5600_ = lean_array_uset(v_bs_5595_, v_i_5594_, v___x_5599_);
v___x_5601_ = ((size_t)1ULL);
v___x_5602_ = lean_usize_add(v_i_5594_, v___x_5601_);
v___x_5603_ = lean_array_uset(v_bs_x27_5600_, v_i_5594_, v_fst_5598_);
v_i_5594_ = v___x_5602_;
v_bs_5595_ = v___x_5603_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_straightLineExtS_spec__0___boxed(lean_object* v_sz_5605_, lean_object* v_i_5606_, lean_object* v_bs_5607_){
_start:
{
size_t v_sz_boxed_5608_; size_t v_i_boxed_5609_; lean_object* v_res_5610_; 
v_sz_boxed_5608_ = lean_unbox_usize(v_sz_5605_);
lean_dec(v_sz_5605_);
v_i_boxed_5609_ = lean_unbox_usize(v_i_5606_);
lean_dec(v_i_5606_);
v_res_5610_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_straightLineExtS_spec__0(v_sz_boxed_5608_, v_i_boxed_5609_, v_bs_5607_);
return v_res_5610_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_straightLineExtS___lam__0(lean_object* v_x_5611_){
_start:
{
lean_object* v_goals_5612_; size_t v_sz_5613_; size_t v___x_5614_; lean_object* v___x_5615_; 
v_goals_5612_ = lean_ctor_get(v_x_5611_, 2);
lean_inc_ref(v_goals_5612_);
lean_dec_ref(v_x_5611_);
v_sz_5613_ = lean_array_size(v_goals_5612_);
v___x_5614_ = ((size_t)0ULL);
v___x_5615_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_straightLineExtS_spec__0(v_sz_5613_, v___x_5614_, v_goals_5612_);
return v___x_5615_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_straightLineExtS___lam__1(lean_object* v_x_5616_){
_start:
{
lean_object* v_depth_5617_; lean_object* v___x_5618_; uint8_t v___x_5619_; 
v_depth_5617_ = lean_ctor_get(v_x_5616_, 0);
v___x_5618_ = lean_unsigned_to_nat(0u);
v___x_5619_ = lean_nat_dec_eq(v_depth_5617_, v___x_5618_);
if (v___x_5619_ == 0)
{
uint8_t v___x_5620_; 
v___x_5620_ = 1;
return v___x_5620_;
}
else
{
uint8_t v___x_5621_; 
v___x_5621_ = 0;
return v___x_5621_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_straightLineExtS___lam__1___boxed(lean_object* v_x_5622_){
_start:
{
uint8_t v_res_5623_; lean_object* v_r_5624_; 
v_res_5623_ = lp_aesop_Aesop_straightLineExtS___lam__1(v_x_5622_);
lean_dec_ref(v_x_5622_);
v_r_5624_ = lean_box(v_res_5623_);
return v_r_5624_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_straightLineExtS(lean_object* v_goal_5628_, lean_object* v_a_5629_, lean_object* v_a_5630_, lean_object* v_a_5631_, lean_object* v_a_5632_, lean_object* v_a_5633_, lean_object* v_a_5634_){
_start:
{
lean_object* v___f_5636_; lean_object* v___f_5637_; lean_object* v___f_5638_; lean_object* v___x_5639_; lean_object* v___x_5640_; lean_object* v___x_5641_; 
v___f_5636_ = ((lean_object*)(lp_aesop_Aesop_straightLineExtS___closed__0));
v___f_5637_ = ((lean_object*)(lp_aesop_Aesop_straightLineExtS___closed__1));
v___f_5638_ = ((lean_object*)(lp_aesop_Aesop_straightLineExtS___closed__2));
lean_inc(v_goal_5628_);
v___x_5639_ = lean_alloc_closure((void*)(lp_aesop_Aesop_straightLineExt___boxed), 6, 1);
lean_closure_set(v___x_5639_, 0, v_goal_5628_);
v___x_5640_ = lean_alloc_closure((void*)(lp_aesop_Lean_Meta_unhygienic___at___00Aesop_introsS_spec__0___boxed), 7, 2);
lean_closure_set(v___x_5640_, 0, lean_box(0));
lean_closure_set(v___x_5640_, 1, v___x_5639_);
v___x_5641_ = lp_aesop_Aesop_withScriptStep___redArg(v_goal_5628_, v___f_5636_, v___f_5637_, v___f_5638_, v___x_5640_, v_a_5629_, v_a_5630_, v_a_5631_, v_a_5632_, v_a_5633_, v_a_5634_);
return v___x_5641_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_straightLineExtS___boxed(lean_object* v_goal_5642_, lean_object* v_a_5643_, lean_object* v_a_5644_, lean_object* v_a_5645_, lean_object* v_a_5646_, lean_object* v_a_5647_, lean_object* v_a_5648_, lean_object* v_a_5649_){
_start:
{
lean_object* v_res_5650_; 
v_res_5650_ = lp_aesop_Aesop_straightLineExtS(v_goal_5642_, v_a_5643_, v_a_5644_, v_a_5645_, v_a_5646_, v_a_5647_, v_a_5648_);
lean_dec(v_a_5648_);
lean_dec_ref(v_a_5647_);
lean_dec(v_a_5646_);
lean_dec_ref(v_a_5645_);
lean_dec(v_a_5644_);
lean_dec(v_a_5643_);
return v_res_5650_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1_spec__2_spec__3___redArg(lean_object* v_x_5651_, lean_object* v_x_5652_, lean_object* v_x_5653_, lean_object* v_x_5654_){
_start:
{
lean_object* v_ks_5655_; lean_object* v_vs_5656_; lean_object* v___x_5658_; uint8_t v_isShared_5659_; uint8_t v_isSharedCheck_5680_; 
v_ks_5655_ = lean_ctor_get(v_x_5651_, 0);
v_vs_5656_ = lean_ctor_get(v_x_5651_, 1);
v_isSharedCheck_5680_ = !lean_is_exclusive(v_x_5651_);
if (v_isSharedCheck_5680_ == 0)
{
v___x_5658_ = v_x_5651_;
v_isShared_5659_ = v_isSharedCheck_5680_;
goto v_resetjp_5657_;
}
else
{
lean_inc(v_vs_5656_);
lean_inc(v_ks_5655_);
lean_dec(v_x_5651_);
v___x_5658_ = lean_box(0);
v_isShared_5659_ = v_isSharedCheck_5680_;
goto v_resetjp_5657_;
}
v_resetjp_5657_:
{
lean_object* v___x_5660_; uint8_t v___x_5661_; 
v___x_5660_ = lean_array_get_size(v_ks_5655_);
v___x_5661_ = lean_nat_dec_lt(v_x_5652_, v___x_5660_);
if (v___x_5661_ == 0)
{
lean_object* v___x_5662_; lean_object* v___x_5663_; lean_object* v___x_5665_; 
lean_dec(v_x_5652_);
v___x_5662_ = lean_array_push(v_ks_5655_, v_x_5653_);
v___x_5663_ = lean_array_push(v_vs_5656_, v_x_5654_);
if (v_isShared_5659_ == 0)
{
lean_ctor_set(v___x_5658_, 1, v___x_5663_);
lean_ctor_set(v___x_5658_, 0, v___x_5662_);
v___x_5665_ = v___x_5658_;
goto v_reusejp_5664_;
}
else
{
lean_object* v_reuseFailAlloc_5666_; 
v_reuseFailAlloc_5666_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5666_, 0, v___x_5662_);
lean_ctor_set(v_reuseFailAlloc_5666_, 1, v___x_5663_);
v___x_5665_ = v_reuseFailAlloc_5666_;
goto v_reusejp_5664_;
}
v_reusejp_5664_:
{
return v___x_5665_;
}
}
else
{
lean_object* v_k_x27_5667_; uint8_t v___x_5668_; 
v_k_x27_5667_ = lean_array_fget_borrowed(v_ks_5655_, v_x_5652_);
v___x_5668_ = l_Lean_instBEqMVarId_beq(v_x_5653_, v_k_x27_5667_);
if (v___x_5668_ == 0)
{
lean_object* v___x_5670_; 
if (v_isShared_5659_ == 0)
{
v___x_5670_ = v___x_5658_;
goto v_reusejp_5669_;
}
else
{
lean_object* v_reuseFailAlloc_5674_; 
v_reuseFailAlloc_5674_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5674_, 0, v_ks_5655_);
lean_ctor_set(v_reuseFailAlloc_5674_, 1, v_vs_5656_);
v___x_5670_ = v_reuseFailAlloc_5674_;
goto v_reusejp_5669_;
}
v_reusejp_5669_:
{
lean_object* v___x_5671_; lean_object* v___x_5672_; 
v___x_5671_ = lean_unsigned_to_nat(1u);
v___x_5672_ = lean_nat_add(v_x_5652_, v___x_5671_);
lean_dec(v_x_5652_);
v_x_5651_ = v___x_5670_;
v_x_5652_ = v___x_5672_;
goto _start;
}
}
else
{
lean_object* v___x_5675_; lean_object* v___x_5676_; lean_object* v___x_5678_; 
v___x_5675_ = lean_array_fset(v_ks_5655_, v_x_5652_, v_x_5653_);
v___x_5676_ = lean_array_fset(v_vs_5656_, v_x_5652_, v_x_5654_);
lean_dec(v_x_5652_);
if (v_isShared_5659_ == 0)
{
lean_ctor_set(v___x_5658_, 1, v___x_5676_);
lean_ctor_set(v___x_5658_, 0, v___x_5675_);
v___x_5678_ = v___x_5658_;
goto v_reusejp_5677_;
}
else
{
lean_object* v_reuseFailAlloc_5679_; 
v_reuseFailAlloc_5679_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5679_, 0, v___x_5675_);
lean_ctor_set(v_reuseFailAlloc_5679_, 1, v___x_5676_);
v___x_5678_ = v_reuseFailAlloc_5679_;
goto v_reusejp_5677_;
}
v_reusejp_5677_:
{
return v___x_5678_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1_spec__2___redArg(lean_object* v_n_5681_, lean_object* v_k_5682_, lean_object* v_v_5683_){
_start:
{
lean_object* v___x_5684_; lean_object* v___x_5685_; 
v___x_5684_ = lean_unsigned_to_nat(0u);
v___x_5685_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1_spec__2_spec__3___redArg(v_n_5681_, v___x_5684_, v_k_5682_, v_v_5683_);
return v___x_5685_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_5686_; 
v___x_5686_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_5686_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1___redArg(lean_object* v_x_5687_, size_t v_x_5688_, size_t v_x_5689_, lean_object* v_x_5690_, lean_object* v_x_5691_){
_start:
{
if (lean_obj_tag(v_x_5687_) == 0)
{
lean_object* v_es_5692_; size_t v___x_5693_; size_t v___x_5694_; lean_object* v_j_5695_; lean_object* v___x_5696_; uint8_t v___x_5697_; 
v_es_5692_ = lean_ctor_get(v_x_5687_, 0);
v___x_5693_ = ((size_t)31ULL);
v___x_5694_ = lean_usize_land(v_x_5688_, v___x_5693_);
v_j_5695_ = lean_usize_to_nat(v___x_5694_);
v___x_5696_ = lean_array_get_size(v_es_5692_);
v___x_5697_ = lean_nat_dec_lt(v_j_5695_, v___x_5696_);
if (v___x_5697_ == 0)
{
lean_dec(v_j_5695_);
lean_dec(v_x_5691_);
lean_dec(v_x_5690_);
return v_x_5687_;
}
else
{
lean_object* v___x_5699_; uint8_t v_isShared_5700_; uint8_t v_isSharedCheck_5736_; 
lean_inc_ref(v_es_5692_);
v_isSharedCheck_5736_ = !lean_is_exclusive(v_x_5687_);
if (v_isSharedCheck_5736_ == 0)
{
lean_object* v_unused_5737_; 
v_unused_5737_ = lean_ctor_get(v_x_5687_, 0);
lean_dec(v_unused_5737_);
v___x_5699_ = v_x_5687_;
v_isShared_5700_ = v_isSharedCheck_5736_;
goto v_resetjp_5698_;
}
else
{
lean_dec(v_x_5687_);
v___x_5699_ = lean_box(0);
v_isShared_5700_ = v_isSharedCheck_5736_;
goto v_resetjp_5698_;
}
v_resetjp_5698_:
{
lean_object* v_v_5701_; lean_object* v___x_5702_; lean_object* v_xs_x27_5703_; lean_object* v___y_5705_; 
v_v_5701_ = lean_array_fget(v_es_5692_, v_j_5695_);
v___x_5702_ = lean_box(0);
v_xs_x27_5703_ = lean_array_fset(v_es_5692_, v_j_5695_, v___x_5702_);
switch(lean_obj_tag(v_v_5701_))
{
case 0:
{
lean_object* v_key_5710_; lean_object* v_val_5711_; lean_object* v___x_5713_; uint8_t v_isShared_5714_; uint8_t v_isSharedCheck_5721_; 
v_key_5710_ = lean_ctor_get(v_v_5701_, 0);
v_val_5711_ = lean_ctor_get(v_v_5701_, 1);
v_isSharedCheck_5721_ = !lean_is_exclusive(v_v_5701_);
if (v_isSharedCheck_5721_ == 0)
{
v___x_5713_ = v_v_5701_;
v_isShared_5714_ = v_isSharedCheck_5721_;
goto v_resetjp_5712_;
}
else
{
lean_inc(v_val_5711_);
lean_inc(v_key_5710_);
lean_dec(v_v_5701_);
v___x_5713_ = lean_box(0);
v_isShared_5714_ = v_isSharedCheck_5721_;
goto v_resetjp_5712_;
}
v_resetjp_5712_:
{
uint8_t v___x_5715_; 
v___x_5715_ = l_Lean_instBEqMVarId_beq(v_x_5690_, v_key_5710_);
if (v___x_5715_ == 0)
{
lean_object* v___x_5716_; lean_object* v___x_5717_; 
lean_del_object(v___x_5713_);
v___x_5716_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_5710_, v_val_5711_, v_x_5690_, v_x_5691_);
v___x_5717_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5717_, 0, v___x_5716_);
v___y_5705_ = v___x_5717_;
goto v___jp_5704_;
}
else
{
lean_object* v___x_5719_; 
lean_dec(v_val_5711_);
lean_dec(v_key_5710_);
if (v_isShared_5714_ == 0)
{
lean_ctor_set(v___x_5713_, 1, v_x_5691_);
lean_ctor_set(v___x_5713_, 0, v_x_5690_);
v___x_5719_ = v___x_5713_;
goto v_reusejp_5718_;
}
else
{
lean_object* v_reuseFailAlloc_5720_; 
v_reuseFailAlloc_5720_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5720_, 0, v_x_5690_);
lean_ctor_set(v_reuseFailAlloc_5720_, 1, v_x_5691_);
v___x_5719_ = v_reuseFailAlloc_5720_;
goto v_reusejp_5718_;
}
v_reusejp_5718_:
{
v___y_5705_ = v___x_5719_;
goto v___jp_5704_;
}
}
}
}
case 1:
{
lean_object* v_node_5722_; lean_object* v___x_5724_; uint8_t v_isShared_5725_; uint8_t v_isSharedCheck_5734_; 
v_node_5722_ = lean_ctor_get(v_v_5701_, 0);
v_isSharedCheck_5734_ = !lean_is_exclusive(v_v_5701_);
if (v_isSharedCheck_5734_ == 0)
{
v___x_5724_ = v_v_5701_;
v_isShared_5725_ = v_isSharedCheck_5734_;
goto v_resetjp_5723_;
}
else
{
lean_inc(v_node_5722_);
lean_dec(v_v_5701_);
v___x_5724_ = lean_box(0);
v_isShared_5725_ = v_isSharedCheck_5734_;
goto v_resetjp_5723_;
}
v_resetjp_5723_:
{
size_t v___x_5726_; size_t v___x_5727_; size_t v___x_5728_; size_t v___x_5729_; lean_object* v___x_5730_; lean_object* v___x_5732_; 
v___x_5726_ = ((size_t)5ULL);
v___x_5727_ = lean_usize_shift_right(v_x_5688_, v___x_5726_);
v___x_5728_ = ((size_t)1ULL);
v___x_5729_ = lean_usize_add(v_x_5689_, v___x_5728_);
v___x_5730_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1___redArg(v_node_5722_, v___x_5727_, v___x_5729_, v_x_5690_, v_x_5691_);
if (v_isShared_5725_ == 0)
{
lean_ctor_set(v___x_5724_, 0, v___x_5730_);
v___x_5732_ = v___x_5724_;
goto v_reusejp_5731_;
}
else
{
lean_object* v_reuseFailAlloc_5733_; 
v_reuseFailAlloc_5733_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5733_, 0, v___x_5730_);
v___x_5732_ = v_reuseFailAlloc_5733_;
goto v_reusejp_5731_;
}
v_reusejp_5731_:
{
v___y_5705_ = v___x_5732_;
goto v___jp_5704_;
}
}
}
default: 
{
lean_object* v___x_5735_; 
v___x_5735_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5735_, 0, v_x_5690_);
lean_ctor_set(v___x_5735_, 1, v_x_5691_);
v___y_5705_ = v___x_5735_;
goto v___jp_5704_;
}
}
v___jp_5704_:
{
lean_object* v___x_5706_; lean_object* v___x_5708_; 
v___x_5706_ = lean_array_fset(v_xs_x27_5703_, v_j_5695_, v___y_5705_);
lean_dec(v_j_5695_);
if (v_isShared_5700_ == 0)
{
lean_ctor_set(v___x_5699_, 0, v___x_5706_);
v___x_5708_ = v___x_5699_;
goto v_reusejp_5707_;
}
else
{
lean_object* v_reuseFailAlloc_5709_; 
v_reuseFailAlloc_5709_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5709_, 0, v___x_5706_);
v___x_5708_ = v_reuseFailAlloc_5709_;
goto v_reusejp_5707_;
}
v_reusejp_5707_:
{
return v___x_5708_;
}
}
}
}
}
else
{
lean_object* v_ks_5738_; lean_object* v_vs_5739_; lean_object* v___x_5741_; uint8_t v_isShared_5742_; uint8_t v_isSharedCheck_5759_; 
v_ks_5738_ = lean_ctor_get(v_x_5687_, 0);
v_vs_5739_ = lean_ctor_get(v_x_5687_, 1);
v_isSharedCheck_5759_ = !lean_is_exclusive(v_x_5687_);
if (v_isSharedCheck_5759_ == 0)
{
v___x_5741_ = v_x_5687_;
v_isShared_5742_ = v_isSharedCheck_5759_;
goto v_resetjp_5740_;
}
else
{
lean_inc(v_vs_5739_);
lean_inc(v_ks_5738_);
lean_dec(v_x_5687_);
v___x_5741_ = lean_box(0);
v_isShared_5742_ = v_isSharedCheck_5759_;
goto v_resetjp_5740_;
}
v_resetjp_5740_:
{
lean_object* v___x_5744_; 
if (v_isShared_5742_ == 0)
{
v___x_5744_ = v___x_5741_;
goto v_reusejp_5743_;
}
else
{
lean_object* v_reuseFailAlloc_5758_; 
v_reuseFailAlloc_5758_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_5758_, 0, v_ks_5738_);
lean_ctor_set(v_reuseFailAlloc_5758_, 1, v_vs_5739_);
v___x_5744_ = v_reuseFailAlloc_5758_;
goto v_reusejp_5743_;
}
v_reusejp_5743_:
{
lean_object* v_newNode_5745_; uint8_t v___y_5747_; size_t v___x_5753_; uint8_t v___x_5754_; 
v_newNode_5745_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1_spec__2___redArg(v___x_5744_, v_x_5690_, v_x_5691_);
v___x_5753_ = ((size_t)7ULL);
v___x_5754_ = lean_usize_dec_le(v___x_5753_, v_x_5689_);
if (v___x_5754_ == 0)
{
lean_object* v___x_5755_; lean_object* v___x_5756_; uint8_t v___x_5757_; 
v___x_5755_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_5745_);
v___x_5756_ = lean_unsigned_to_nat(4u);
v___x_5757_ = lean_nat_dec_lt(v___x_5755_, v___x_5756_);
lean_dec(v___x_5755_);
v___y_5747_ = v___x_5757_;
goto v___jp_5746_;
}
else
{
v___y_5747_ = v___x_5754_;
goto v___jp_5746_;
}
v___jp_5746_:
{
if (v___y_5747_ == 0)
{
lean_object* v_ks_5748_; lean_object* v_vs_5749_; lean_object* v___x_5750_; lean_object* v___x_5751_; lean_object* v___x_5752_; 
v_ks_5748_ = lean_ctor_get(v_newNode_5745_, 0);
lean_inc_ref(v_ks_5748_);
v_vs_5749_ = lean_ctor_get(v_newNode_5745_, 1);
lean_inc_ref(v_vs_5749_);
lean_dec_ref(v_newNode_5745_);
v___x_5750_ = lean_unsigned_to_nat(0u);
v___x_5751_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1___redArg___closed__0, &lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1___redArg___closed__0_once, _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1___redArg___closed__0);
v___x_5752_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1_spec__3___redArg(v_x_5689_, v_ks_5748_, v_vs_5749_, v___x_5750_, v___x_5751_);
lean_dec_ref(v_vs_5749_);
lean_dec_ref(v_ks_5748_);
return v___x_5752_;
}
else
{
return v_newNode_5745_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1_spec__3___redArg(size_t v_depth_5760_, lean_object* v_keys_5761_, lean_object* v_vals_5762_, lean_object* v_i_5763_, lean_object* v_entries_5764_){
_start:
{
lean_object* v___x_5765_; uint8_t v___x_5766_; 
v___x_5765_ = lean_array_get_size(v_keys_5761_);
v___x_5766_ = lean_nat_dec_lt(v_i_5763_, v___x_5765_);
if (v___x_5766_ == 0)
{
lean_dec(v_i_5763_);
return v_entries_5764_;
}
else
{
lean_object* v_k_5767_; lean_object* v_v_5768_; uint64_t v___x_5769_; size_t v_h_5770_; size_t v___x_5771_; lean_object* v___x_5772_; size_t v___x_5773_; size_t v___x_5774_; size_t v___x_5775_; size_t v_h_5776_; lean_object* v___x_5777_; lean_object* v___x_5778_; 
v_k_5767_ = lean_array_fget_borrowed(v_keys_5761_, v_i_5763_);
v_v_5768_ = lean_array_fget_borrowed(v_vals_5762_, v_i_5763_);
v___x_5769_ = l_Lean_instHashableMVarId_hash(v_k_5767_);
v_h_5770_ = lean_uint64_to_usize(v___x_5769_);
v___x_5771_ = ((size_t)5ULL);
v___x_5772_ = lean_unsigned_to_nat(1u);
v___x_5773_ = ((size_t)1ULL);
v___x_5774_ = lean_usize_sub(v_depth_5760_, v___x_5773_);
v___x_5775_ = lean_usize_mul(v___x_5771_, v___x_5774_);
v_h_5776_ = lean_usize_shift_right(v_h_5770_, v___x_5775_);
v___x_5777_ = lean_nat_add(v_i_5763_, v___x_5772_);
lean_dec(v_i_5763_);
lean_inc(v_v_5768_);
lean_inc(v_k_5767_);
v___x_5778_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1___redArg(v_entries_5764_, v_h_5776_, v_depth_5760_, v_k_5767_, v_v_5768_);
v_i_5763_ = v___x_5777_;
v_entries_5764_ = v___x_5778_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1_spec__3___redArg___boxed(lean_object* v_depth_5780_, lean_object* v_keys_5781_, lean_object* v_vals_5782_, lean_object* v_i_5783_, lean_object* v_entries_5784_){
_start:
{
size_t v_depth_boxed_5785_; lean_object* v_res_5786_; 
v_depth_boxed_5785_ = lean_unbox_usize(v_depth_5780_);
lean_dec(v_depth_5780_);
v_res_5786_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1_spec__3___redArg(v_depth_boxed_5785_, v_keys_5781_, v_vals_5782_, v_i_5783_, v_entries_5784_);
lean_dec_ref(v_vals_5782_);
lean_dec_ref(v_keys_5781_);
return v_res_5786_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_x_5787_, lean_object* v_x_5788_, lean_object* v_x_5789_, lean_object* v_x_5790_, lean_object* v_x_5791_){
_start:
{
size_t v_x_7008__boxed_5792_; size_t v_x_7009__boxed_5793_; lean_object* v_res_5794_; 
v_x_7008__boxed_5792_ = lean_unbox_usize(v_x_5788_);
lean_dec(v_x_5788_);
v_x_7009__boxed_5793_ = lean_unbox_usize(v_x_5789_);
lean_dec(v_x_5789_);
v_res_5794_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1___redArg(v_x_5787_, v_x_7008__boxed_5792_, v_x_7009__boxed_5793_, v_x_5790_, v_x_5791_);
return v_res_5794_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0___redArg(lean_object* v_x_5795_, lean_object* v_x_5796_, lean_object* v_x_5797_){
_start:
{
uint64_t v___x_5798_; size_t v___x_5799_; size_t v___x_5800_; lean_object* v___x_5801_; 
v___x_5798_ = l_Lean_instHashableMVarId_hash(v_x_5796_);
v___x_5799_ = lean_uint64_to_usize(v___x_5798_);
v___x_5800_ = ((size_t)1ULL);
v___x_5801_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1___redArg(v_x_5795_, v___x_5799_, v___x_5800_, v_x_5796_, v_x_5797_);
return v___x_5801_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0___redArg(lean_object* v_mvarId_5802_, lean_object* v_val_5803_, lean_object* v___y_5804_){
_start:
{
lean_object* v___x_5806_; lean_object* v_mctx_5807_; lean_object* v_cache_5808_; lean_object* v_zetaDeltaFVarIds_5809_; lean_object* v_postponed_5810_; lean_object* v_diag_5811_; lean_object* v___x_5813_; uint8_t v_isShared_5814_; uint8_t v_isSharedCheck_5839_; 
v___x_5806_ = lean_st_ref_take(v___y_5804_);
v_mctx_5807_ = lean_ctor_get(v___x_5806_, 0);
v_cache_5808_ = lean_ctor_get(v___x_5806_, 1);
v_zetaDeltaFVarIds_5809_ = lean_ctor_get(v___x_5806_, 2);
v_postponed_5810_ = lean_ctor_get(v___x_5806_, 3);
v_diag_5811_ = lean_ctor_get(v___x_5806_, 4);
v_isSharedCheck_5839_ = !lean_is_exclusive(v___x_5806_);
if (v_isSharedCheck_5839_ == 0)
{
v___x_5813_ = v___x_5806_;
v_isShared_5814_ = v_isSharedCheck_5839_;
goto v_resetjp_5812_;
}
else
{
lean_inc(v_diag_5811_);
lean_inc(v_postponed_5810_);
lean_inc(v_zetaDeltaFVarIds_5809_);
lean_inc(v_cache_5808_);
lean_inc(v_mctx_5807_);
lean_dec(v___x_5806_);
v___x_5813_ = lean_box(0);
v_isShared_5814_ = v_isSharedCheck_5839_;
goto v_resetjp_5812_;
}
v_resetjp_5812_:
{
lean_object* v_depth_5815_; lean_object* v_levelAssignDepth_5816_; lean_object* v_lmvarCounter_5817_; lean_object* v_mvarCounter_5818_; lean_object* v_lDecls_5819_; lean_object* v_decls_5820_; lean_object* v_userNames_5821_; lean_object* v_lAssignment_5822_; lean_object* v_eAssignment_5823_; lean_object* v_dAssignment_5824_; lean_object* v___x_5826_; uint8_t v_isShared_5827_; uint8_t v_isSharedCheck_5838_; 
v_depth_5815_ = lean_ctor_get(v_mctx_5807_, 0);
v_levelAssignDepth_5816_ = lean_ctor_get(v_mctx_5807_, 1);
v_lmvarCounter_5817_ = lean_ctor_get(v_mctx_5807_, 2);
v_mvarCounter_5818_ = lean_ctor_get(v_mctx_5807_, 3);
v_lDecls_5819_ = lean_ctor_get(v_mctx_5807_, 4);
v_decls_5820_ = lean_ctor_get(v_mctx_5807_, 5);
v_userNames_5821_ = lean_ctor_get(v_mctx_5807_, 6);
v_lAssignment_5822_ = lean_ctor_get(v_mctx_5807_, 7);
v_eAssignment_5823_ = lean_ctor_get(v_mctx_5807_, 8);
v_dAssignment_5824_ = lean_ctor_get(v_mctx_5807_, 9);
v_isSharedCheck_5838_ = !lean_is_exclusive(v_mctx_5807_);
if (v_isSharedCheck_5838_ == 0)
{
v___x_5826_ = v_mctx_5807_;
v_isShared_5827_ = v_isSharedCheck_5838_;
goto v_resetjp_5825_;
}
else
{
lean_inc(v_dAssignment_5824_);
lean_inc(v_eAssignment_5823_);
lean_inc(v_lAssignment_5822_);
lean_inc(v_userNames_5821_);
lean_inc(v_decls_5820_);
lean_inc(v_lDecls_5819_);
lean_inc(v_mvarCounter_5818_);
lean_inc(v_lmvarCounter_5817_);
lean_inc(v_levelAssignDepth_5816_);
lean_inc(v_depth_5815_);
lean_dec(v_mctx_5807_);
v___x_5826_ = lean_box(0);
v_isShared_5827_ = v_isSharedCheck_5838_;
goto v_resetjp_5825_;
}
v_resetjp_5825_:
{
lean_object* v___x_5828_; lean_object* v___x_5830_; 
v___x_5828_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0___redArg(v_eAssignment_5823_, v_mvarId_5802_, v_val_5803_);
if (v_isShared_5827_ == 0)
{
lean_ctor_set(v___x_5826_, 8, v___x_5828_);
v___x_5830_ = v___x_5826_;
goto v_reusejp_5829_;
}
else
{
lean_object* v_reuseFailAlloc_5837_; 
v_reuseFailAlloc_5837_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_5837_, 0, v_depth_5815_);
lean_ctor_set(v_reuseFailAlloc_5837_, 1, v_levelAssignDepth_5816_);
lean_ctor_set(v_reuseFailAlloc_5837_, 2, v_lmvarCounter_5817_);
lean_ctor_set(v_reuseFailAlloc_5837_, 3, v_mvarCounter_5818_);
lean_ctor_set(v_reuseFailAlloc_5837_, 4, v_lDecls_5819_);
lean_ctor_set(v_reuseFailAlloc_5837_, 5, v_decls_5820_);
lean_ctor_set(v_reuseFailAlloc_5837_, 6, v_userNames_5821_);
lean_ctor_set(v_reuseFailAlloc_5837_, 7, v_lAssignment_5822_);
lean_ctor_set(v_reuseFailAlloc_5837_, 8, v___x_5828_);
lean_ctor_set(v_reuseFailAlloc_5837_, 9, v_dAssignment_5824_);
v___x_5830_ = v_reuseFailAlloc_5837_;
goto v_reusejp_5829_;
}
v_reusejp_5829_:
{
lean_object* v___x_5832_; 
if (v_isShared_5814_ == 0)
{
lean_ctor_set(v___x_5813_, 0, v___x_5830_);
v___x_5832_ = v___x_5813_;
goto v_reusejp_5831_;
}
else
{
lean_object* v_reuseFailAlloc_5836_; 
v_reuseFailAlloc_5836_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_5836_, 0, v___x_5830_);
lean_ctor_set(v_reuseFailAlloc_5836_, 1, v_cache_5808_);
lean_ctor_set(v_reuseFailAlloc_5836_, 2, v_zetaDeltaFVarIds_5809_);
lean_ctor_set(v_reuseFailAlloc_5836_, 3, v_postponed_5810_);
lean_ctor_set(v_reuseFailAlloc_5836_, 4, v_diag_5811_);
v___x_5832_ = v_reuseFailAlloc_5836_;
goto v_reusejp_5831_;
}
v_reusejp_5831_:
{
lean_object* v___x_5833_; lean_object* v___x_5834_; lean_object* v___x_5835_; 
v___x_5833_ = lean_st_ref_set(v___y_5804_, v___x_5832_);
v___x_5834_ = lean_box(0);
v___x_5835_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5835_, 0, v___x_5834_);
return v___x_5835_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0___redArg___boxed(lean_object* v_mvarId_5840_, lean_object* v_val_5841_, lean_object* v___y_5842_, lean_object* v___y_5843_){
_start:
{
lean_object* v_res_5844_; 
v_res_5844_ = lp_aesop_Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0___redArg(v_mvarId_5840_, v_val_5841_, v___y_5842_);
lean_dec(v___y_5842_);
return v_res_5844_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryExactFVarS(lean_object* v_goal_5847_, lean_object* v_fvarId_5848_, uint8_t v_md_5849_, lean_object* v_a_5850_, lean_object* v_a_5851_, lean_object* v_a_5852_, lean_object* v_a_5853_, lean_object* v_a_5854_, lean_object* v_a_5855_){
_start:
{
lean_object* v___x_5857_; 
v___x_5857_ = l_Lean_Meta_saveState___redArg(v_a_5853_, v_a_5855_);
if (lean_obj_tag(v___x_5857_) == 0)
{
lean_object* v_a_5858_; lean_object* v___x_5859_; 
v_a_5858_ = lean_ctor_get(v___x_5857_, 0);
lean_inc(v_a_5858_);
lean_dec_ref_known(v___x_5857_, 1);
lean_inc(v_fvarId_5848_);
v___x_5859_ = l_Lean_FVarId_getDecl___redArg(v_fvarId_5848_, v_a_5852_, v_a_5854_, v_a_5855_);
if (lean_obj_tag(v___x_5859_) == 0)
{
lean_object* v_a_5860_; uint8_t v_a_5862_; lean_object* v___x_5910_; 
v_a_5860_ = lean_ctor_get(v___x_5859_, 0);
lean_inc(v_a_5860_);
lean_dec_ref_known(v___x_5859_, 1);
lean_inc(v_goal_5847_);
v___x_5910_ = l_Lean_MVarId_getType(v_goal_5847_, v_a_5852_, v_a_5853_, v_a_5854_, v_a_5855_);
if (lean_obj_tag(v___x_5910_) == 0)
{
lean_object* v_a_5911_; lean_object* v_keyedConfig_5912_; uint8_t v_trackZetaDelta_5913_; lean_object* v_zetaDeltaSet_5914_; lean_object* v_lctx_5915_; lean_object* v_localInstances_5916_; lean_object* v_defEqCtx_x3f_5917_; lean_object* v_synthPendingDepth_5918_; lean_object* v_customCanUnfoldPredicate_x3f_5919_; uint8_t v_univApprox_5920_; uint8_t v_inTypeClassResolution_5921_; uint8_t v_cacheInferType_5922_; lean_object* v___x_5923_; lean_object* v___x_5924_; lean_object* v___x_5925_; lean_object* v___x_5926_; 
v_a_5911_ = lean_ctor_get(v___x_5910_, 0);
lean_inc(v_a_5911_);
lean_dec_ref_known(v___x_5910_, 1);
v_keyedConfig_5912_ = lean_ctor_get(v_a_5852_, 0);
v_trackZetaDelta_5913_ = lean_ctor_get_uint8(v_a_5852_, sizeof(void*)*7);
v_zetaDeltaSet_5914_ = lean_ctor_get(v_a_5852_, 1);
v_lctx_5915_ = lean_ctor_get(v_a_5852_, 2);
v_localInstances_5916_ = lean_ctor_get(v_a_5852_, 3);
v_defEqCtx_x3f_5917_ = lean_ctor_get(v_a_5852_, 4);
v_synthPendingDepth_5918_ = lean_ctor_get(v_a_5852_, 5);
v_customCanUnfoldPredicate_x3f_5919_ = lean_ctor_get(v_a_5852_, 6);
v_univApprox_5920_ = lean_ctor_get_uint8(v_a_5852_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_5921_ = lean_ctor_get_uint8(v_a_5852_, sizeof(void*)*7 + 2);
v_cacheInferType_5922_ = lean_ctor_get_uint8(v_a_5852_, sizeof(void*)*7 + 3);
v___x_5923_ = l_Lean_LocalDecl_type(v_a_5860_);
lean_inc_ref(v_keyedConfig_5912_);
v___x_5924_ = l_Lean_Meta_ConfigWithKey_setTransparency(v_md_5849_, v_keyedConfig_5912_);
lean_inc(v_customCanUnfoldPredicate_x3f_5919_);
lean_inc(v_synthPendingDepth_5918_);
lean_inc(v_defEqCtx_x3f_5917_);
lean_inc_ref(v_localInstances_5916_);
lean_inc_ref(v_lctx_5915_);
lean_inc(v_zetaDeltaSet_5914_);
v___x_5925_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_5925_, 0, v___x_5924_);
lean_ctor_set(v___x_5925_, 1, v_zetaDeltaSet_5914_);
lean_ctor_set(v___x_5925_, 2, v_lctx_5915_);
lean_ctor_set(v___x_5925_, 3, v_localInstances_5916_);
lean_ctor_set(v___x_5925_, 4, v_defEqCtx_x3f_5917_);
lean_ctor_set(v___x_5925_, 5, v_synthPendingDepth_5918_);
lean_ctor_set(v___x_5925_, 6, v_customCanUnfoldPredicate_x3f_5919_);
lean_ctor_set_uint8(v___x_5925_, sizeof(void*)*7, v_trackZetaDelta_5913_);
lean_ctor_set_uint8(v___x_5925_, sizeof(void*)*7 + 1, v_univApprox_5920_);
lean_ctor_set_uint8(v___x_5925_, sizeof(void*)*7 + 2, v_inTypeClassResolution_5921_);
lean_ctor_set_uint8(v___x_5925_, sizeof(void*)*7 + 3, v_cacheInferType_5922_);
v___x_5926_ = l_Lean_Meta_isExprDefEq(v___x_5923_, v_a_5911_, v___x_5925_, v_a_5853_, v_a_5854_, v_a_5855_);
lean_dec_ref_known(v___x_5925_, 7);
if (lean_obj_tag(v___x_5926_) == 0)
{
lean_object* v_a_5927_; uint8_t v___x_5928_; 
v_a_5927_ = lean_ctor_get(v___x_5926_, 0);
lean_inc(v_a_5927_);
lean_dec_ref_known(v___x_5926_, 1);
v___x_5928_ = lean_unbox(v_a_5927_);
lean_dec(v_a_5927_);
v_a_5862_ = v___x_5928_;
goto v___jp_5861_;
}
else
{
if (lean_obj_tag(v___x_5926_) == 0)
{
lean_object* v_a_5929_; uint8_t v___x_5930_; 
v_a_5929_ = lean_ctor_get(v___x_5926_, 0);
lean_inc(v_a_5929_);
lean_dec_ref_known(v___x_5926_, 1);
v___x_5930_ = lean_unbox(v_a_5929_);
lean_dec(v_a_5929_);
v_a_5862_ = v___x_5930_;
goto v___jp_5861_;
}
else
{
lean_dec(v_a_5860_);
lean_dec(v_a_5858_);
lean_dec(v_fvarId_5848_);
lean_dec(v_goal_5847_);
return v___x_5926_;
}
}
}
else
{
lean_object* v_a_5931_; lean_object* v___x_5933_; uint8_t v_isShared_5934_; uint8_t v_isSharedCheck_5938_; 
lean_dec(v_a_5860_);
lean_dec(v_a_5858_);
lean_dec(v_fvarId_5848_);
lean_dec(v_goal_5847_);
v_a_5931_ = lean_ctor_get(v___x_5910_, 0);
v_isSharedCheck_5938_ = !lean_is_exclusive(v___x_5910_);
if (v_isSharedCheck_5938_ == 0)
{
v___x_5933_ = v___x_5910_;
v_isShared_5934_ = v_isSharedCheck_5938_;
goto v_resetjp_5932_;
}
else
{
lean_inc(v_a_5931_);
lean_dec(v___x_5910_);
v___x_5933_ = lean_box(0);
v_isShared_5934_ = v_isSharedCheck_5938_;
goto v_resetjp_5932_;
}
v_resetjp_5932_:
{
lean_object* v___x_5936_; 
if (v_isShared_5934_ == 0)
{
v___x_5936_ = v___x_5933_;
goto v_reusejp_5935_;
}
else
{
lean_object* v_reuseFailAlloc_5937_; 
v_reuseFailAlloc_5937_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5937_, 0, v_a_5931_);
v___x_5936_ = v_reuseFailAlloc_5937_;
goto v_reusejp_5935_;
}
v_reusejp_5935_:
{
return v___x_5936_;
}
}
}
v___jp_5861_:
{
if (v_a_5862_ == 0)
{
lean_object* v___x_5863_; 
lean_dec(v_a_5860_);
lean_dec(v_fvarId_5848_);
lean_dec(v_goal_5847_);
v___x_5863_ = l_Lean_Meta_SavedState_restore___redArg(v_a_5858_, v_a_5853_, v_a_5855_);
lean_dec(v_a_5858_);
if (lean_obj_tag(v___x_5863_) == 0)
{
lean_object* v___x_5865_; uint8_t v_isShared_5866_; uint8_t v_isSharedCheck_5871_; 
v_isSharedCheck_5871_ = !lean_is_exclusive(v___x_5863_);
if (v_isSharedCheck_5871_ == 0)
{
lean_object* v_unused_5872_; 
v_unused_5872_ = lean_ctor_get(v___x_5863_, 0);
lean_dec(v_unused_5872_);
v___x_5865_ = v___x_5863_;
v_isShared_5866_ = v_isSharedCheck_5871_;
goto v_resetjp_5864_;
}
else
{
lean_dec(v___x_5863_);
v___x_5865_ = lean_box(0);
v_isShared_5866_ = v_isSharedCheck_5871_;
goto v_resetjp_5864_;
}
v_resetjp_5864_:
{
lean_object* v___x_5867_; lean_object* v___x_5869_; 
v___x_5867_ = lean_box(v_a_5862_);
if (v_isShared_5866_ == 0)
{
lean_ctor_set(v___x_5865_, 0, v___x_5867_);
v___x_5869_ = v___x_5865_;
goto v_reusejp_5868_;
}
else
{
lean_object* v_reuseFailAlloc_5870_; 
v_reuseFailAlloc_5870_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5870_, 0, v___x_5867_);
v___x_5869_ = v_reuseFailAlloc_5870_;
goto v_reusejp_5868_;
}
v_reusejp_5868_:
{
return v___x_5869_;
}
}
}
else
{
lean_object* v_a_5873_; lean_object* v___x_5875_; uint8_t v_isShared_5876_; uint8_t v_isSharedCheck_5880_; 
v_a_5873_ = lean_ctor_get(v___x_5863_, 0);
v_isSharedCheck_5880_ = !lean_is_exclusive(v___x_5863_);
if (v_isSharedCheck_5880_ == 0)
{
v___x_5875_ = v___x_5863_;
v_isShared_5876_ = v_isSharedCheck_5880_;
goto v_resetjp_5874_;
}
else
{
lean_inc(v_a_5873_);
lean_dec(v___x_5863_);
v___x_5875_ = lean_box(0);
v_isShared_5876_ = v_isSharedCheck_5880_;
goto v_resetjp_5874_;
}
v_resetjp_5874_:
{
lean_object* v___x_5878_; 
if (v_isShared_5876_ == 0)
{
v___x_5878_ = v___x_5875_;
goto v_reusejp_5877_;
}
else
{
lean_object* v_reuseFailAlloc_5879_; 
v_reuseFailAlloc_5879_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5879_, 0, v_a_5873_);
v___x_5878_ = v_reuseFailAlloc_5879_;
goto v_reusejp_5877_;
}
v_reusejp_5877_:
{
return v___x_5878_;
}
}
}
}
else
{
lean_object* v___x_5881_; lean_object* v___x_5882_; lean_object* v___x_5883_; 
v___x_5881_ = l_Lean_LocalDecl_toExpr(v_a_5860_);
lean_inc(v_goal_5847_);
v___x_5882_ = lp_aesop_Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0___redArg(v_goal_5847_, v___x_5881_, v_a_5853_);
lean_dec_ref(v___x_5882_);
v___x_5883_ = l_Lean_Meta_saveState___redArg(v_a_5853_, v_a_5855_);
if (lean_obj_tag(v___x_5883_) == 0)
{
lean_object* v_a_5884_; lean_object* v___x_5885_; lean_object* v___x_5886_; lean_object* v___x_5887_; lean_object* v___x_5888_; lean_object* v___x_5889_; lean_object* v___x_5890_; lean_object* v___x_5891_; lean_object* v___x_5892_; lean_object* v___x_5894_; uint8_t v_isShared_5895_; uint8_t v_isSharedCheck_5900_; 
v_a_5884_ = lean_ctor_get(v___x_5883_, 0);
lean_inc(v_a_5884_);
lean_dec_ref_known(v___x_5883_, 1);
v___x_5885_ = lean_box(v_md_5849_);
lean_inc(v_goal_5847_);
v___x_5886_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Script_TacticBuilder_exactFVar___boxed), 8, 3);
lean_closure_set(v___x_5886_, 0, v_goal_5847_);
lean_closure_set(v___x_5886_, 1, v_fvarId_5848_);
lean_closure_set(v___x_5886_, 2, v___x_5885_);
v___x_5887_ = lean_unsigned_to_nat(1u);
v___x_5888_ = lean_mk_empty_array_with_capacity(v___x_5887_);
v___x_5889_ = lean_array_push(v___x_5888_, v___x_5886_);
v___x_5890_ = ((lean_object*)(lp_aesop_Aesop_tryExactFVarS___closed__0));
v___x_5891_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_5891_, 0, v_a_5858_);
lean_ctor_set(v___x_5891_, 1, v_goal_5847_);
lean_ctor_set(v___x_5891_, 2, v___x_5889_);
lean_ctor_set(v___x_5891_, 3, v_a_5884_);
lean_ctor_set(v___x_5891_, 4, v___x_5890_);
v___x_5892_ = lp_aesop_Aesop_recordScriptStep___at___00Aesop_unfoldManyTargetS_spec__0___redArg(v___x_5891_, v_a_5850_);
v_isSharedCheck_5900_ = !lean_is_exclusive(v___x_5892_);
if (v_isSharedCheck_5900_ == 0)
{
lean_object* v_unused_5901_; 
v_unused_5901_ = lean_ctor_get(v___x_5892_, 0);
lean_dec(v_unused_5901_);
v___x_5894_ = v___x_5892_;
v_isShared_5895_ = v_isSharedCheck_5900_;
goto v_resetjp_5893_;
}
else
{
lean_dec(v___x_5892_);
v___x_5894_ = lean_box(0);
v_isShared_5895_ = v_isSharedCheck_5900_;
goto v_resetjp_5893_;
}
v_resetjp_5893_:
{
lean_object* v___x_5896_; lean_object* v___x_5898_; 
v___x_5896_ = lean_box(v_a_5862_);
if (v_isShared_5895_ == 0)
{
lean_ctor_set(v___x_5894_, 0, v___x_5896_);
v___x_5898_ = v___x_5894_;
goto v_reusejp_5897_;
}
else
{
lean_object* v_reuseFailAlloc_5899_; 
v_reuseFailAlloc_5899_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5899_, 0, v___x_5896_);
v___x_5898_ = v_reuseFailAlloc_5899_;
goto v_reusejp_5897_;
}
v_reusejp_5897_:
{
return v___x_5898_;
}
}
}
else
{
lean_object* v_a_5902_; lean_object* v___x_5904_; uint8_t v_isShared_5905_; uint8_t v_isSharedCheck_5909_; 
lean_dec(v_a_5858_);
lean_dec(v_fvarId_5848_);
lean_dec(v_goal_5847_);
v_a_5902_ = lean_ctor_get(v___x_5883_, 0);
v_isSharedCheck_5909_ = !lean_is_exclusive(v___x_5883_);
if (v_isSharedCheck_5909_ == 0)
{
v___x_5904_ = v___x_5883_;
v_isShared_5905_ = v_isSharedCheck_5909_;
goto v_resetjp_5903_;
}
else
{
lean_inc(v_a_5902_);
lean_dec(v___x_5883_);
v___x_5904_ = lean_box(0);
v_isShared_5905_ = v_isSharedCheck_5909_;
goto v_resetjp_5903_;
}
v_resetjp_5903_:
{
lean_object* v___x_5907_; 
if (v_isShared_5905_ == 0)
{
v___x_5907_ = v___x_5904_;
goto v_reusejp_5906_;
}
else
{
lean_object* v_reuseFailAlloc_5908_; 
v_reuseFailAlloc_5908_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5908_, 0, v_a_5902_);
v___x_5907_ = v_reuseFailAlloc_5908_;
goto v_reusejp_5906_;
}
v_reusejp_5906_:
{
return v___x_5907_;
}
}
}
}
}
}
else
{
lean_object* v_a_5939_; lean_object* v___x_5941_; uint8_t v_isShared_5942_; uint8_t v_isSharedCheck_5946_; 
lean_dec(v_a_5858_);
lean_dec(v_fvarId_5848_);
lean_dec(v_goal_5847_);
v_a_5939_ = lean_ctor_get(v___x_5859_, 0);
v_isSharedCheck_5946_ = !lean_is_exclusive(v___x_5859_);
if (v_isSharedCheck_5946_ == 0)
{
v___x_5941_ = v___x_5859_;
v_isShared_5942_ = v_isSharedCheck_5946_;
goto v_resetjp_5940_;
}
else
{
lean_inc(v_a_5939_);
lean_dec(v___x_5859_);
v___x_5941_ = lean_box(0);
v_isShared_5942_ = v_isSharedCheck_5946_;
goto v_resetjp_5940_;
}
v_resetjp_5940_:
{
lean_object* v___x_5944_; 
if (v_isShared_5942_ == 0)
{
v___x_5944_ = v___x_5941_;
goto v_reusejp_5943_;
}
else
{
lean_object* v_reuseFailAlloc_5945_; 
v_reuseFailAlloc_5945_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5945_, 0, v_a_5939_);
v___x_5944_ = v_reuseFailAlloc_5945_;
goto v_reusejp_5943_;
}
v_reusejp_5943_:
{
return v___x_5944_;
}
}
}
}
else
{
lean_object* v_a_5947_; lean_object* v___x_5949_; uint8_t v_isShared_5950_; uint8_t v_isSharedCheck_5954_; 
lean_dec(v_fvarId_5848_);
lean_dec(v_goal_5847_);
v_a_5947_ = lean_ctor_get(v___x_5857_, 0);
v_isSharedCheck_5954_ = !lean_is_exclusive(v___x_5857_);
if (v_isSharedCheck_5954_ == 0)
{
v___x_5949_ = v___x_5857_;
v_isShared_5950_ = v_isSharedCheck_5954_;
goto v_resetjp_5948_;
}
else
{
lean_inc(v_a_5947_);
lean_dec(v___x_5857_);
v___x_5949_ = lean_box(0);
v_isShared_5950_ = v_isSharedCheck_5954_;
goto v_resetjp_5948_;
}
v_resetjp_5948_:
{
lean_object* v___x_5952_; 
if (v_isShared_5950_ == 0)
{
v___x_5952_ = v___x_5949_;
goto v_reusejp_5951_;
}
else
{
lean_object* v_reuseFailAlloc_5953_; 
v_reuseFailAlloc_5953_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5953_, 0, v_a_5947_);
v___x_5952_ = v_reuseFailAlloc_5953_;
goto v_reusejp_5951_;
}
v_reusejp_5951_:
{
return v___x_5952_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_tryExactFVarS___boxed(lean_object* v_goal_5955_, lean_object* v_fvarId_5956_, lean_object* v_md_5957_, lean_object* v_a_5958_, lean_object* v_a_5959_, lean_object* v_a_5960_, lean_object* v_a_5961_, lean_object* v_a_5962_, lean_object* v_a_5963_, lean_object* v_a_5964_){
_start:
{
uint8_t v_md_boxed_5965_; lean_object* v_res_5966_; 
v_md_boxed_5965_ = lean_unbox(v_md_5957_);
v_res_5966_ = lp_aesop_Aesop_tryExactFVarS(v_goal_5955_, v_fvarId_5956_, v_md_boxed_5965_, v_a_5958_, v_a_5959_, v_a_5960_, v_a_5961_, v_a_5962_, v_a_5963_);
lean_dec(v_a_5963_);
lean_dec_ref(v_a_5962_);
lean_dec(v_a_5961_);
lean_dec_ref(v_a_5960_);
lean_dec(v_a_5959_);
lean_dec(v_a_5958_);
return v_res_5966_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0(lean_object* v_mvarId_5967_, lean_object* v_val_5968_, lean_object* v___y_5969_, lean_object* v___y_5970_, lean_object* v___y_5971_, lean_object* v___y_5972_, lean_object* v___y_5973_, lean_object* v___y_5974_){
_start:
{
lean_object* v___x_5976_; 
v___x_5976_ = lp_aesop_Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0___redArg(v_mvarId_5967_, v_val_5968_, v___y_5972_);
return v___x_5976_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0___boxed(lean_object* v_mvarId_5977_, lean_object* v_val_5978_, lean_object* v___y_5979_, lean_object* v___y_5980_, lean_object* v___y_5981_, lean_object* v___y_5982_, lean_object* v___y_5983_, lean_object* v___y_5984_, lean_object* v___y_5985_){
_start:
{
lean_object* v_res_5986_; 
v_res_5986_ = lp_aesop_Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0(v_mvarId_5977_, v_val_5978_, v___y_5979_, v___y_5980_, v___y_5981_, v___y_5982_, v___y_5983_, v___y_5984_);
lean_dec(v___y_5984_);
lean_dec_ref(v___y_5983_);
lean_dec(v___y_5982_);
lean_dec_ref(v___y_5981_);
lean_dec(v___y_5980_);
lean_dec(v___y_5979_);
return v_res_5986_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0(lean_object* v_00_u03b2_5987_, lean_object* v_x_5988_, lean_object* v_x_5989_, lean_object* v_x_5990_){
_start:
{
lean_object* v___x_5991_; 
v___x_5991_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0___redArg(v_x_5988_, v_x_5989_, v_x_5990_);
return v___x_5991_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1(lean_object* v_00_u03b2_5992_, lean_object* v_x_5993_, size_t v_x_5994_, size_t v_x_5995_, lean_object* v_x_5996_, lean_object* v_x_5997_){
_start:
{
lean_object* v___x_5998_; 
v___x_5998_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1___redArg(v_x_5993_, v_x_5994_, v_x_5995_, v_x_5996_, v_x_5997_);
return v___x_5998_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b2_5999_, lean_object* v_x_6000_, lean_object* v_x_6001_, lean_object* v_x_6002_, lean_object* v_x_6003_, lean_object* v_x_6004_){
_start:
{
size_t v_x_7442__boxed_6005_; size_t v_x_7443__boxed_6006_; lean_object* v_res_6007_; 
v_x_7442__boxed_6005_ = lean_unbox_usize(v_x_6001_);
lean_dec(v_x_6001_);
v_x_7443__boxed_6006_ = lean_unbox_usize(v_x_6002_);
lean_dec(v_x_6002_);
v_res_6007_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1(v_00_u03b2_5999_, v_x_6000_, v_x_7442__boxed_6005_, v_x_7443__boxed_6006_, v_x_6003_, v_x_6004_);
return v_res_6007_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1_spec__2(lean_object* v_00_u03b2_6008_, lean_object* v_n_6009_, lean_object* v_k_6010_, lean_object* v_v_6011_){
_start:
{
lean_object* v___x_6012_; 
v___x_6012_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1_spec__2___redArg(v_n_6009_, v_k_6010_, v_v_6011_);
return v___x_6012_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1_spec__3(lean_object* v_00_u03b2_6013_, size_t v_depth_6014_, lean_object* v_keys_6015_, lean_object* v_vals_6016_, lean_object* v_heq_6017_, lean_object* v_i_6018_, lean_object* v_entries_6019_){
_start:
{
lean_object* v___x_6020_; 
v___x_6020_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1_spec__3___redArg(v_depth_6014_, v_keys_6015_, v_vals_6016_, v_i_6018_, v_entries_6019_);
return v___x_6020_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1_spec__3___boxed(lean_object* v_00_u03b2_6021_, lean_object* v_depth_6022_, lean_object* v_keys_6023_, lean_object* v_vals_6024_, lean_object* v_heq_6025_, lean_object* v_i_6026_, lean_object* v_entries_6027_){
_start:
{
size_t v_depth_boxed_6028_; lean_object* v_res_6029_; 
v_depth_boxed_6028_ = lean_unbox_usize(v_depth_6022_);
lean_dec(v_depth_6022_);
v_res_6029_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1_spec__3(v_00_u03b2_6021_, v_depth_boxed_6028_, v_keys_6023_, v_vals_6024_, v_heq_6025_, v_i_6026_, v_entries_6027_);
lean_dec_ref(v_vals_6024_);
lean_dec_ref(v_keys_6023_);
return v_res_6029_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1_spec__2_spec__3(lean_object* v_00_u03b2_6030_, lean_object* v_x_6031_, lean_object* v_x_6032_, lean_object* v_x_6033_, lean_object* v_x_6034_){
_start:
{
lean_object* v___x_6035_; 
v___x_6035_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_tryExactFVarS_spec__0_spec__0_spec__1_spec__2_spec__3___redArg(v_x_6031_, v_x_6032_, v_x_6033_, v_x_6034_);
return v___x_6035_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_renameInaccessibleFVarsS_x27_spec__0(size_t v_sz_6036_, size_t v_i_6037_, lean_object* v_bs_6038_, lean_object* v___y_6039_, lean_object* v___y_6040_, lean_object* v___y_6041_, lean_object* v___y_6042_, lean_object* v___y_6043_, lean_object* v___y_6044_){
_start:
{
uint8_t v___x_6046_; 
v___x_6046_ = lean_usize_dec_lt(v_i_6037_, v_sz_6036_);
if (v___x_6046_ == 0)
{
lean_object* v___x_6047_; 
v___x_6047_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_6047_, 0, v_bs_6038_);
return v___x_6047_;
}
else
{
lean_object* v_v_6048_; lean_object* v___x_6049_; 
v_v_6048_ = lean_array_uget_borrowed(v_bs_6038_, v_i_6037_);
lean_inc(v_v_6048_);
v___x_6049_ = lp_aesop_Aesop_renameInaccessibleFVarsS(v_v_6048_, v___y_6039_, v___y_6040_, v___y_6041_, v___y_6042_, v___y_6043_, v___y_6044_);
if (lean_obj_tag(v___x_6049_) == 0)
{
lean_object* v_a_6050_; lean_object* v_fst_6051_; lean_object* v___x_6052_; lean_object* v_bs_x27_6053_; size_t v___x_6054_; size_t v___x_6055_; lean_object* v___x_6056_; 
v_a_6050_ = lean_ctor_get(v___x_6049_, 0);
lean_inc(v_a_6050_);
lean_dec_ref_known(v___x_6049_, 1);
v_fst_6051_ = lean_ctor_get(v_a_6050_, 0);
lean_inc(v_fst_6051_);
lean_dec(v_a_6050_);
v___x_6052_ = lean_unsigned_to_nat(0u);
v_bs_x27_6053_ = lean_array_uset(v_bs_6038_, v_i_6037_, v___x_6052_);
v___x_6054_ = ((size_t)1ULL);
v___x_6055_ = lean_usize_add(v_i_6037_, v___x_6054_);
v___x_6056_ = lean_array_uset(v_bs_x27_6053_, v_i_6037_, v_fst_6051_);
v_i_6037_ = v___x_6055_;
v_bs_6038_ = v___x_6056_;
goto _start;
}
else
{
lean_object* v_a_6058_; lean_object* v___x_6060_; uint8_t v_isShared_6061_; uint8_t v_isSharedCheck_6065_; 
lean_dec_ref(v_bs_6038_);
v_a_6058_ = lean_ctor_get(v___x_6049_, 0);
v_isSharedCheck_6065_ = !lean_is_exclusive(v___x_6049_);
if (v_isSharedCheck_6065_ == 0)
{
v___x_6060_ = v___x_6049_;
v_isShared_6061_ = v_isSharedCheck_6065_;
goto v_resetjp_6059_;
}
else
{
lean_inc(v_a_6058_);
lean_dec(v___x_6049_);
v___x_6060_ = lean_box(0);
v_isShared_6061_ = v_isSharedCheck_6065_;
goto v_resetjp_6059_;
}
v_resetjp_6059_:
{
lean_object* v___x_6063_; 
if (v_isShared_6061_ == 0)
{
v___x_6063_ = v___x_6060_;
goto v_reusejp_6062_;
}
else
{
lean_object* v_reuseFailAlloc_6064_; 
v_reuseFailAlloc_6064_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6064_, 0, v_a_6058_);
v___x_6063_ = v_reuseFailAlloc_6064_;
goto v_reusejp_6062_;
}
v_reusejp_6062_:
{
return v___x_6063_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_renameInaccessibleFVarsS_x27_spec__0___boxed(lean_object* v_sz_6066_, lean_object* v_i_6067_, lean_object* v_bs_6068_, lean_object* v___y_6069_, lean_object* v___y_6070_, lean_object* v___y_6071_, lean_object* v___y_6072_, lean_object* v___y_6073_, lean_object* v___y_6074_, lean_object* v___y_6075_){
_start:
{
size_t v_sz_boxed_6076_; size_t v_i_boxed_6077_; lean_object* v_res_6078_; 
v_sz_boxed_6076_ = lean_unbox_usize(v_sz_6066_);
lean_dec(v_sz_6066_);
v_i_boxed_6077_ = lean_unbox_usize(v_i_6067_);
lean_dec(v_i_6067_);
v_res_6078_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_renameInaccessibleFVarsS_x27_spec__0(v_sz_boxed_6076_, v_i_boxed_6077_, v_bs_6068_, v___y_6069_, v___y_6070_, v___y_6071_, v___y_6072_, v___y_6073_, v___y_6074_);
lean_dec(v___y_6074_);
lean_dec_ref(v___y_6073_);
lean_dec(v___y_6072_);
lean_dec_ref(v___y_6071_);
lean_dec(v___y_6070_);
lean_dec(v___y_6069_);
return v_res_6078_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_renameInaccessibleFVarsS_x27(lean_object* v_goals_6079_, lean_object* v_a_6080_, lean_object* v_a_6081_, lean_object* v_a_6082_, lean_object* v_a_6083_, lean_object* v_a_6084_, lean_object* v_a_6085_){
_start:
{
size_t v_sz_6087_; size_t v___x_6088_; lean_object* v___x_6089_; 
v_sz_6087_ = lean_array_size(v_goals_6079_);
v___x_6088_ = ((size_t)0ULL);
v___x_6089_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Script_SpecificTactics_0__Aesop_renameInaccessibleFVarsS_x27_spec__0(v_sz_6087_, v___x_6088_, v_goals_6079_, v_a_6080_, v_a_6081_, v_a_6082_, v_a_6083_, v_a_6084_, v_a_6085_);
return v___x_6089_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_renameInaccessibleFVarsS_x27___boxed(lean_object* v_goals_6090_, lean_object* v_a_6091_, lean_object* v_a_6092_, lean_object* v_a_6093_, lean_object* v_a_6094_, lean_object* v_a_6095_, lean_object* v_a_6096_, lean_object* v_a_6097_){
_start:
{
lean_object* v_res_6098_; 
v_res_6098_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_renameInaccessibleFVarsS_x27(v_goals_6090_, v_a_6091_, v_a_6092_, v_a_6093_, v_a_6094_, v_a_6095_, v_a_6096_);
lean_dec(v_a_6096_);
lean_dec_ref(v_a_6095_);
lean_dec(v_a_6094_);
lean_dec_ref(v_a_6093_);
lean_dec(v_a_6092_);
lean_dec(v_a_6091_);
return v_res_6098_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_splitTargetS_x3f_tacticBuilder___redArg(lean_object* v_a_6099_){
_start:
{
lean_object* v___x_6101_; 
v___x_6101_ = lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg(v_a_6099_);
return v___x_6101_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_splitTargetS_x3f_tacticBuilder___redArg___boxed(lean_object* v_a_6102_, lean_object* v_a_6103_){
_start:
{
lean_object* v_res_6104_; 
v_res_6104_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_splitTargetS_x3f_tacticBuilder___redArg(v_a_6102_);
lean_dec_ref(v_a_6102_);
return v_res_6104_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_splitTargetS_x3f_tacticBuilder(lean_object* v_x_6105_, lean_object* v_a_6106_, lean_object* v_a_6107_, lean_object* v_a_6108_, lean_object* v_a_6109_){
_start:
{
lean_object* v___x_6111_; 
v___x_6111_ = lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg(v_a_6108_);
return v___x_6111_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_splitTargetS_x3f_tacticBuilder___boxed(lean_object* v_x_6112_, lean_object* v_a_6113_, lean_object* v_a_6114_, lean_object* v_a_6115_, lean_object* v_a_6116_, lean_object* v_a_6117_){
_start:
{
lean_object* v_res_6118_; 
v_res_6118_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_splitTargetS_x3f_tacticBuilder(v_x_6112_, v_a_6113_, v_a_6114_, v_a_6115_, v_a_6116_);
lean_dec(v_a_6116_);
lean_dec_ref(v_a_6115_);
lean_dec(v_a_6114_);
lean_dec_ref(v_a_6113_);
lean_dec_ref(v_x_6112_);
return v_res_6118_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_splitTargetS_x3f___lam__0(lean_object* v___y_6119_, lean_object* v___y_6120_, lean_object* v___y_6121_, lean_object* v___y_6122_, lean_object* v___y_6123_){
_start:
{
lean_object* v___x_6125_; 
v___x_6125_ = lp_aesop_Aesop_Script_TacticBuilder_splitTarget___redArg(v___y_6122_);
return v___x_6125_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_splitTargetS_x3f___lam__0___boxed(lean_object* v___y_6126_, lean_object* v___y_6127_, lean_object* v___y_6128_, lean_object* v___y_6129_, lean_object* v___y_6130_, lean_object* v___y_6131_){
_start:
{
lean_object* v_res_6132_; 
v_res_6132_ = lp_aesop_Aesop_splitTargetS_x3f___lam__0(v___y_6126_, v___y_6127_, v___y_6128_, v___y_6129_, v___y_6130_);
lean_dec(v___y_6130_);
lean_dec_ref(v___y_6129_);
lean_dec(v___y_6128_);
lean_dec_ref(v___y_6127_);
lean_dec_ref(v___y_6126_);
return v_res_6132_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_splitTargetS_x3f___lam__2(lean_object* v_goal_6133_, uint8_t v___x_6134_, uint8_t v___x_6135_, lean_object* v___y_6136_, lean_object* v___y_6137_, lean_object* v___y_6138_, lean_object* v___y_6139_){
_start:
{
lean_object* v___x_6141_; 
v___x_6141_ = l_Lean_Meta_splitTarget_x3f(v_goal_6133_, v___x_6134_, v___x_6135_, v___y_6136_, v___y_6137_, v___y_6138_, v___y_6139_);
if (lean_obj_tag(v___x_6141_) == 0)
{
lean_object* v_a_6142_; lean_object* v___x_6144_; uint8_t v_isShared_6145_; uint8_t v_isSharedCheck_6162_; 
v_a_6142_ = lean_ctor_get(v___x_6141_, 0);
v_isSharedCheck_6162_ = !lean_is_exclusive(v___x_6141_);
if (v_isSharedCheck_6162_ == 0)
{
v___x_6144_ = v___x_6141_;
v_isShared_6145_ = v_isSharedCheck_6162_;
goto v_resetjp_6143_;
}
else
{
lean_inc(v_a_6142_);
lean_dec(v___x_6141_);
v___x_6144_ = lean_box(0);
v_isShared_6145_ = v_isSharedCheck_6162_;
goto v_resetjp_6143_;
}
v_resetjp_6143_:
{
if (lean_obj_tag(v_a_6142_) == 0)
{
lean_object* v___x_6146_; lean_object* v___x_6148_; 
v___x_6146_ = lean_box(0);
if (v_isShared_6145_ == 0)
{
lean_ctor_set(v___x_6144_, 0, v___x_6146_);
v___x_6148_ = v___x_6144_;
goto v_reusejp_6147_;
}
else
{
lean_object* v_reuseFailAlloc_6149_; 
v_reuseFailAlloc_6149_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6149_, 0, v___x_6146_);
v___x_6148_ = v_reuseFailAlloc_6149_;
goto v_reusejp_6147_;
}
v_reusejp_6147_:
{
return v___x_6148_;
}
}
else
{
lean_object* v_val_6150_; lean_object* v___x_6152_; uint8_t v_isShared_6153_; uint8_t v_isSharedCheck_6161_; 
v_val_6150_ = lean_ctor_get(v_a_6142_, 0);
v_isSharedCheck_6161_ = !lean_is_exclusive(v_a_6142_);
if (v_isSharedCheck_6161_ == 0)
{
v___x_6152_ = v_a_6142_;
v_isShared_6153_ = v_isSharedCheck_6161_;
goto v_resetjp_6151_;
}
else
{
lean_inc(v_val_6150_);
lean_dec(v_a_6142_);
v___x_6152_ = lean_box(0);
v_isShared_6153_ = v_isSharedCheck_6161_;
goto v_resetjp_6151_;
}
v_resetjp_6151_:
{
lean_object* v___x_6154_; lean_object* v___x_6156_; 
v___x_6154_ = lean_array_mk(v_val_6150_);
if (v_isShared_6153_ == 0)
{
lean_ctor_set(v___x_6152_, 0, v___x_6154_);
v___x_6156_ = v___x_6152_;
goto v_reusejp_6155_;
}
else
{
lean_object* v_reuseFailAlloc_6160_; 
v_reuseFailAlloc_6160_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6160_, 0, v___x_6154_);
v___x_6156_ = v_reuseFailAlloc_6160_;
goto v_reusejp_6155_;
}
v_reusejp_6155_:
{
lean_object* v___x_6158_; 
if (v_isShared_6145_ == 0)
{
lean_ctor_set(v___x_6144_, 0, v___x_6156_);
v___x_6158_ = v___x_6144_;
goto v_reusejp_6157_;
}
else
{
lean_object* v_reuseFailAlloc_6159_; 
v_reuseFailAlloc_6159_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6159_, 0, v___x_6156_);
v___x_6158_ = v_reuseFailAlloc_6159_;
goto v_reusejp_6157_;
}
v_reusejp_6157_:
{
return v___x_6158_;
}
}
}
}
}
}
else
{
lean_object* v_a_6163_; lean_object* v___x_6165_; uint8_t v_isShared_6166_; uint8_t v_isSharedCheck_6170_; 
v_a_6163_ = lean_ctor_get(v___x_6141_, 0);
v_isSharedCheck_6170_ = !lean_is_exclusive(v___x_6141_);
if (v_isSharedCheck_6170_ == 0)
{
v___x_6165_ = v___x_6141_;
v_isShared_6166_ = v_isSharedCheck_6170_;
goto v_resetjp_6164_;
}
else
{
lean_inc(v_a_6163_);
lean_dec(v___x_6141_);
v___x_6165_ = lean_box(0);
v_isShared_6166_ = v_isSharedCheck_6170_;
goto v_resetjp_6164_;
}
v_resetjp_6164_:
{
lean_object* v___x_6168_; 
if (v_isShared_6166_ == 0)
{
v___x_6168_ = v___x_6165_;
goto v_reusejp_6167_;
}
else
{
lean_object* v_reuseFailAlloc_6169_; 
v_reuseFailAlloc_6169_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6169_, 0, v_a_6163_);
v___x_6168_ = v_reuseFailAlloc_6169_;
goto v_reusejp_6167_;
}
v_reusejp_6167_:
{
return v___x_6168_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_splitTargetS_x3f___lam__2___boxed(lean_object* v_goal_6171_, lean_object* v___x_6172_, lean_object* v___x_6173_, lean_object* v___y_6174_, lean_object* v___y_6175_, lean_object* v___y_6176_, lean_object* v___y_6177_, lean_object* v___y_6178_){
_start:
{
uint8_t v___x_1029__boxed_6179_; uint8_t v___x_1030__boxed_6180_; lean_object* v_res_6181_; 
v___x_1029__boxed_6179_ = lean_unbox(v___x_6172_);
v___x_1030__boxed_6180_ = lean_unbox(v___x_6173_);
v_res_6181_ = lp_aesop_Aesop_splitTargetS_x3f___lam__2(v_goal_6171_, v___x_1029__boxed_6179_, v___x_1030__boxed_6180_, v___y_6174_, v___y_6175_, v___y_6176_, v___y_6177_);
lean_dec(v___y_6177_);
lean_dec_ref(v___y_6176_);
lean_dec(v___y_6175_);
lean_dec_ref(v___y_6174_);
return v_res_6181_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_splitTargetS_x3f(lean_object* v_goal_6183_, lean_object* v_a_6184_, lean_object* v_a_6185_, lean_object* v_a_6186_, lean_object* v_a_6187_, lean_object* v_a_6188_, lean_object* v_a_6189_){
_start:
{
lean_object* v___f_6191_; lean_object* v___f_6192_; uint8_t v___x_6193_; uint8_t v___x_6194_; lean_object* v___x_6195_; lean_object* v___x_6196_; lean_object* v___f_6197_; lean_object* v___x_6198_; 
v___f_6191_ = ((lean_object*)(lp_aesop_Aesop_splitTargetS_x3f___closed__0));
v___f_6192_ = ((lean_object*)(lp_aesop_Aesop_applyS___closed__0));
v___x_6193_ = 1;
v___x_6194_ = 0;
v___x_6195_ = lean_box(v___x_6193_);
v___x_6196_ = lean_box(v___x_6194_);
lean_inc(v_goal_6183_);
v___f_6197_ = lean_alloc_closure((void*)(lp_aesop_Aesop_splitTargetS_x3f___lam__2___boxed), 8, 3);
lean_closure_set(v___f_6197_, 0, v_goal_6183_);
lean_closure_set(v___f_6197_, 1, v___x_6195_);
lean_closure_set(v___f_6197_, 2, v___x_6196_);
v___x_6198_ = lp_aesop_Aesop_withOptScriptStep___redArg(v_goal_6183_, v___f_6192_, v___f_6191_, v___f_6197_, v_a_6184_, v_a_6186_, v_a_6187_, v_a_6188_, v_a_6189_);
if (lean_obj_tag(v___x_6198_) == 0)
{
lean_object* v_a_6199_; lean_object* v___x_6201_; uint8_t v_isShared_6202_; uint8_t v_isSharedCheck_6232_; 
v_a_6199_ = lean_ctor_get(v___x_6198_, 0);
v_isSharedCheck_6232_ = !lean_is_exclusive(v___x_6198_);
if (v_isSharedCheck_6232_ == 0)
{
v___x_6201_ = v___x_6198_;
v_isShared_6202_ = v_isSharedCheck_6232_;
goto v_resetjp_6200_;
}
else
{
lean_inc(v_a_6199_);
lean_dec(v___x_6198_);
v___x_6201_ = lean_box(0);
v_isShared_6202_ = v_isSharedCheck_6232_;
goto v_resetjp_6200_;
}
v_resetjp_6200_:
{
if (lean_obj_tag(v_a_6199_) == 1)
{
lean_object* v_val_6203_; lean_object* v___x_6205_; uint8_t v_isShared_6206_; uint8_t v_isSharedCheck_6227_; 
lean_del_object(v___x_6201_);
v_val_6203_ = lean_ctor_get(v_a_6199_, 0);
v_isSharedCheck_6227_ = !lean_is_exclusive(v_a_6199_);
if (v_isSharedCheck_6227_ == 0)
{
v___x_6205_ = v_a_6199_;
v_isShared_6206_ = v_isSharedCheck_6227_;
goto v_resetjp_6204_;
}
else
{
lean_inc(v_val_6203_);
lean_dec(v_a_6199_);
v___x_6205_ = lean_box(0);
v_isShared_6206_ = v_isSharedCheck_6227_;
goto v_resetjp_6204_;
}
v_resetjp_6204_:
{
lean_object* v___x_6207_; 
v___x_6207_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_renameInaccessibleFVarsS_x27(v_val_6203_, v_a_6184_, v_a_6185_, v_a_6186_, v_a_6187_, v_a_6188_, v_a_6189_);
if (lean_obj_tag(v___x_6207_) == 0)
{
lean_object* v_a_6208_; lean_object* v___x_6210_; uint8_t v_isShared_6211_; uint8_t v_isSharedCheck_6218_; 
v_a_6208_ = lean_ctor_get(v___x_6207_, 0);
v_isSharedCheck_6218_ = !lean_is_exclusive(v___x_6207_);
if (v_isSharedCheck_6218_ == 0)
{
v___x_6210_ = v___x_6207_;
v_isShared_6211_ = v_isSharedCheck_6218_;
goto v_resetjp_6209_;
}
else
{
lean_inc(v_a_6208_);
lean_dec(v___x_6207_);
v___x_6210_ = lean_box(0);
v_isShared_6211_ = v_isSharedCheck_6218_;
goto v_resetjp_6209_;
}
v_resetjp_6209_:
{
lean_object* v___x_6213_; 
if (v_isShared_6206_ == 0)
{
lean_ctor_set(v___x_6205_, 0, v_a_6208_);
v___x_6213_ = v___x_6205_;
goto v_reusejp_6212_;
}
else
{
lean_object* v_reuseFailAlloc_6217_; 
v_reuseFailAlloc_6217_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6217_, 0, v_a_6208_);
v___x_6213_ = v_reuseFailAlloc_6217_;
goto v_reusejp_6212_;
}
v_reusejp_6212_:
{
lean_object* v___x_6215_; 
if (v_isShared_6211_ == 0)
{
lean_ctor_set(v___x_6210_, 0, v___x_6213_);
v___x_6215_ = v___x_6210_;
goto v_reusejp_6214_;
}
else
{
lean_object* v_reuseFailAlloc_6216_; 
v_reuseFailAlloc_6216_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6216_, 0, v___x_6213_);
v___x_6215_ = v_reuseFailAlloc_6216_;
goto v_reusejp_6214_;
}
v_reusejp_6214_:
{
return v___x_6215_;
}
}
}
}
else
{
lean_object* v_a_6219_; lean_object* v___x_6221_; uint8_t v_isShared_6222_; uint8_t v_isSharedCheck_6226_; 
lean_del_object(v___x_6205_);
v_a_6219_ = lean_ctor_get(v___x_6207_, 0);
v_isSharedCheck_6226_ = !lean_is_exclusive(v___x_6207_);
if (v_isSharedCheck_6226_ == 0)
{
v___x_6221_ = v___x_6207_;
v_isShared_6222_ = v_isSharedCheck_6226_;
goto v_resetjp_6220_;
}
else
{
lean_inc(v_a_6219_);
lean_dec(v___x_6207_);
v___x_6221_ = lean_box(0);
v_isShared_6222_ = v_isSharedCheck_6226_;
goto v_resetjp_6220_;
}
v_resetjp_6220_:
{
lean_object* v___x_6224_; 
if (v_isShared_6222_ == 0)
{
v___x_6224_ = v___x_6221_;
goto v_reusejp_6223_;
}
else
{
lean_object* v_reuseFailAlloc_6225_; 
v_reuseFailAlloc_6225_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6225_, 0, v_a_6219_);
v___x_6224_ = v_reuseFailAlloc_6225_;
goto v_reusejp_6223_;
}
v_reusejp_6223_:
{
return v___x_6224_;
}
}
}
}
}
else
{
lean_object* v___x_6228_; lean_object* v___x_6230_; 
lean_dec(v_a_6199_);
v___x_6228_ = lean_box(0);
if (v_isShared_6202_ == 0)
{
lean_ctor_set(v___x_6201_, 0, v___x_6228_);
v___x_6230_ = v___x_6201_;
goto v_reusejp_6229_;
}
else
{
lean_object* v_reuseFailAlloc_6231_; 
v_reuseFailAlloc_6231_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6231_, 0, v___x_6228_);
v___x_6230_ = v_reuseFailAlloc_6231_;
goto v_reusejp_6229_;
}
v_reusejp_6229_:
{
return v___x_6230_;
}
}
}
}
else
{
return v___x_6198_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_splitTargetS_x3f___boxed(lean_object* v_goal_6233_, lean_object* v_a_6234_, lean_object* v_a_6235_, lean_object* v_a_6236_, lean_object* v_a_6237_, lean_object* v_a_6238_, lean_object* v_a_6239_, lean_object* v_a_6240_){
_start:
{
lean_object* v_res_6241_; 
v_res_6241_ = lp_aesop_Aesop_splitTargetS_x3f(v_goal_6233_, v_a_6234_, v_a_6235_, v_a_6236_, v_a_6237_, v_a_6238_, v_a_6239_);
lean_dec(v_a_6239_);
lean_dec_ref(v_a_6238_);
lean_dec(v_a_6237_);
lean_dec_ref(v_a_6236_);
lean_dec(v_a_6235_);
lean_dec(v_a_6234_);
return v_res_6241_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_splitFirstHypothesisS_x3f_tacticBuilder(lean_object* v_goal_6242_, lean_object* v_x_6243_, lean_object* v_a_6244_, lean_object* v_a_6245_, lean_object* v_a_6246_, lean_object* v_a_6247_){
_start:
{
lean_object* v_snd_6249_; lean_object* v___x_6250_; 
v_snd_6249_ = lean_ctor_get(v_x_6243_, 1);
lean_inc(v_snd_6249_);
lean_dec_ref(v_x_6243_);
v___x_6250_ = lp_aesop_Aesop_Script_TacticBuilder_splitAt(v_goal_6242_, v_snd_6249_, v_a_6244_, v_a_6245_, v_a_6246_, v_a_6247_);
return v___x_6250_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_splitFirstHypothesisS_x3f_tacticBuilder___boxed(lean_object* v_goal_6251_, lean_object* v_x_6252_, lean_object* v_a_6253_, lean_object* v_a_6254_, lean_object* v_a_6255_, lean_object* v_a_6256_, lean_object* v_a_6257_){
_start:
{
lean_object* v_res_6258_; 
v_res_6258_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_splitFirstHypothesisS_x3f_tacticBuilder(v_goal_6251_, v_x_6252_, v_a_6253_, v_a_6254_, v_a_6255_, v_a_6256_);
lean_dec(v_a_6256_);
lean_dec_ref(v_a_6255_);
lean_dec(v_a_6254_);
lean_dec_ref(v_a_6253_);
return v_res_6258_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_splitFirstHypothesisS_x3f___lam__0(lean_object* v_x_6259_){
_start:
{
lean_object* v_fst_6260_; 
v_fst_6260_ = lean_ctor_get(v_x_6259_, 0);
lean_inc(v_fst_6260_);
return v_fst_6260_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_splitFirstHypothesisS_x3f___lam__0___boxed(lean_object* v_x_6261_){
_start:
{
lean_object* v_res_6262_; 
v_res_6262_ = lp_aesop_Aesop_splitFirstHypothesisS_x3f___lam__0(v_x_6261_);
lean_dec_ref(v_x_6261_);
return v_res_6262_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0_spec__2_spec__3(lean_object* v_goal_6266_, lean_object* v_as_6267_, size_t v_sz_6268_, size_t v_i_6269_, lean_object* v_b_6270_, lean_object* v___y_6271_, lean_object* v___y_6272_, lean_object* v___y_6273_, lean_object* v___y_6274_){
_start:
{
uint8_t v___x_6276_; 
v___x_6276_ = lean_usize_dec_lt(v_i_6269_, v_sz_6268_);
if (v___x_6276_ == 0)
{
lean_object* v___x_6277_; 
lean_dec(v_goal_6266_);
v___x_6277_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_6277_, 0, v_b_6270_);
return v___x_6277_;
}
else
{
lean_object* v_snd_6278_; lean_object* v___x_6280_; uint8_t v_isShared_6281_; uint8_t v_isSharedCheck_6335_; 
v_snd_6278_ = lean_ctor_get(v_b_6270_, 1);
v_isSharedCheck_6335_ = !lean_is_exclusive(v_b_6270_);
if (v_isSharedCheck_6335_ == 0)
{
lean_object* v_unused_6336_; 
v_unused_6336_ = lean_ctor_get(v_b_6270_, 0);
lean_dec(v_unused_6336_);
v___x_6280_ = v_b_6270_;
v_isShared_6281_ = v_isSharedCheck_6335_;
goto v_resetjp_6279_;
}
else
{
lean_inc(v_snd_6278_);
lean_dec(v_b_6270_);
v___x_6280_ = lean_box(0);
v_isShared_6281_ = v_isSharedCheck_6335_;
goto v_resetjp_6279_;
}
v_resetjp_6279_:
{
lean_object* v___x_6282_; lean_object* v_a_6284_; lean_object* v_a_6291_; 
v___x_6282_ = lean_box(0);
v_a_6291_ = lean_array_uget(v_as_6267_, v_i_6269_);
if (lean_obj_tag(v_a_6291_) == 0)
{
v_a_6284_ = v_snd_6278_;
goto v___jp_6283_;
}
else
{
lean_object* v_val_6292_; lean_object* v___x_6294_; uint8_t v_isShared_6295_; uint8_t v_isSharedCheck_6334_; 
v_val_6292_ = lean_ctor_get(v_a_6291_, 0);
v_isSharedCheck_6334_ = !lean_is_exclusive(v_a_6291_);
if (v_isSharedCheck_6334_ == 0)
{
v___x_6294_ = v_a_6291_;
v_isShared_6295_ = v_isSharedCheck_6334_;
goto v_resetjp_6293_;
}
else
{
lean_inc(v_val_6292_);
lean_dec(v_a_6291_);
v___x_6294_ = lean_box(0);
v_isShared_6295_ = v_isSharedCheck_6334_;
goto v_resetjp_6293_;
}
v_resetjp_6293_:
{
lean_object* v___x_6296_; lean_object* v___x_6297_; uint8_t v___x_6298_; 
v___x_6296_ = lean_box(0);
v___x_6297_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0_spec__2_spec__3___closed__0));
v___x_6298_ = l_Lean_LocalDecl_isImplementationDetail(v_val_6292_);
if (v___x_6298_ == 0)
{
lean_object* v___x_6299_; lean_object* v___x_6300_; 
v___x_6299_ = l_Lean_LocalDecl_fvarId(v_val_6292_);
lean_dec(v_val_6292_);
lean_inc(v___x_6299_);
lean_inc(v_goal_6266_);
v___x_6300_ = l_Lean_Meta_splitLocalDecl_x3f(v_goal_6266_, v___x_6299_, v___y_6271_, v___y_6272_, v___y_6273_, v___y_6274_);
if (lean_obj_tag(v___x_6300_) == 0)
{
lean_object* v_a_6301_; lean_object* v___x_6303_; uint8_t v_isShared_6304_; uint8_t v_isSharedCheck_6325_; 
v_a_6301_ = lean_ctor_get(v___x_6300_, 0);
v_isSharedCheck_6325_ = !lean_is_exclusive(v___x_6300_);
if (v_isSharedCheck_6325_ == 0)
{
v___x_6303_ = v___x_6300_;
v_isShared_6304_ = v_isSharedCheck_6325_;
goto v_resetjp_6302_;
}
else
{
lean_inc(v_a_6301_);
lean_dec(v___x_6300_);
v___x_6303_ = lean_box(0);
v_isShared_6304_ = v_isSharedCheck_6325_;
goto v_resetjp_6302_;
}
v_resetjp_6302_:
{
if (lean_obj_tag(v_a_6301_) == 1)
{
lean_object* v_val_6305_; lean_object* v___x_6307_; uint8_t v_isShared_6308_; uint8_t v_isSharedCheck_6324_; 
lean_del_object(v___x_6280_);
lean_dec(v_goal_6266_);
v_val_6305_ = lean_ctor_get(v_a_6301_, 0);
v_isSharedCheck_6324_ = !lean_is_exclusive(v_a_6301_);
if (v_isSharedCheck_6324_ == 0)
{
v___x_6307_ = v_a_6301_;
v_isShared_6308_ = v_isSharedCheck_6324_;
goto v_resetjp_6306_;
}
else
{
lean_inc(v_val_6305_);
lean_dec(v_a_6301_);
v___x_6307_ = lean_box(0);
v_isShared_6308_ = v_isSharedCheck_6324_;
goto v_resetjp_6306_;
}
v_resetjp_6306_:
{
lean_object* v___x_6309_; lean_object* v___x_6310_; lean_object* v___x_6312_; 
v___x_6309_ = lean_array_mk(v_val_6305_);
v___x_6310_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6310_, 0, v___x_6309_);
lean_ctor_set(v___x_6310_, 1, v___x_6299_);
if (v_isShared_6308_ == 0)
{
lean_ctor_set(v___x_6307_, 0, v___x_6310_);
v___x_6312_ = v___x_6307_;
goto v_reusejp_6311_;
}
else
{
lean_object* v_reuseFailAlloc_6323_; 
v_reuseFailAlloc_6323_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6323_, 0, v___x_6310_);
v___x_6312_ = v_reuseFailAlloc_6323_;
goto v_reusejp_6311_;
}
v_reusejp_6311_:
{
lean_object* v___x_6314_; 
if (v_isShared_6295_ == 0)
{
lean_ctor_set(v___x_6294_, 0, v___x_6312_);
v___x_6314_ = v___x_6294_;
goto v_reusejp_6313_;
}
else
{
lean_object* v_reuseFailAlloc_6322_; 
v_reuseFailAlloc_6322_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6322_, 0, v___x_6312_);
v___x_6314_ = v_reuseFailAlloc_6322_;
goto v_reusejp_6313_;
}
v_reusejp_6313_:
{
lean_object* v___x_6315_; lean_object* v___x_6316_; lean_object* v___x_6317_; lean_object* v___x_6318_; lean_object* v___x_6320_; 
v___x_6315_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6315_, 0, v___x_6314_);
lean_ctor_set(v___x_6315_, 1, v___x_6296_);
v___x_6316_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_6316_, 0, v___x_6315_);
v___x_6317_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_6317_, 0, v___x_6316_);
v___x_6318_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6318_, 0, v___x_6317_);
lean_ctor_set(v___x_6318_, 1, v_snd_6278_);
if (v_isShared_6304_ == 0)
{
lean_ctor_set(v___x_6303_, 0, v___x_6318_);
v___x_6320_ = v___x_6303_;
goto v_reusejp_6319_;
}
else
{
lean_object* v_reuseFailAlloc_6321_; 
v_reuseFailAlloc_6321_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6321_, 0, v___x_6318_);
v___x_6320_ = v_reuseFailAlloc_6321_;
goto v_reusejp_6319_;
}
v_reusejp_6319_:
{
return v___x_6320_;
}
}
}
}
}
else
{
lean_del_object(v___x_6303_);
lean_dec(v_a_6301_);
lean_dec(v___x_6299_);
lean_del_object(v___x_6294_);
lean_dec(v_snd_6278_);
v_a_6284_ = v___x_6297_;
goto v___jp_6283_;
}
}
}
else
{
lean_object* v_a_6326_; lean_object* v___x_6328_; uint8_t v_isShared_6329_; uint8_t v_isSharedCheck_6333_; 
lean_dec(v___x_6299_);
lean_del_object(v___x_6294_);
lean_del_object(v___x_6280_);
lean_dec(v_snd_6278_);
lean_dec(v_goal_6266_);
v_a_6326_ = lean_ctor_get(v___x_6300_, 0);
v_isSharedCheck_6333_ = !lean_is_exclusive(v___x_6300_);
if (v_isSharedCheck_6333_ == 0)
{
v___x_6328_ = v___x_6300_;
v_isShared_6329_ = v_isSharedCheck_6333_;
goto v_resetjp_6327_;
}
else
{
lean_inc(v_a_6326_);
lean_dec(v___x_6300_);
v___x_6328_ = lean_box(0);
v_isShared_6329_ = v_isSharedCheck_6333_;
goto v_resetjp_6327_;
}
v_resetjp_6327_:
{
lean_object* v___x_6331_; 
if (v_isShared_6329_ == 0)
{
v___x_6331_ = v___x_6328_;
goto v_reusejp_6330_;
}
else
{
lean_object* v_reuseFailAlloc_6332_; 
v_reuseFailAlloc_6332_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6332_, 0, v_a_6326_);
v___x_6331_ = v_reuseFailAlloc_6332_;
goto v_reusejp_6330_;
}
v_reusejp_6330_:
{
return v___x_6331_;
}
}
}
}
else
{
lean_del_object(v___x_6294_);
lean_dec(v_val_6292_);
lean_dec(v_snd_6278_);
v_a_6284_ = v___x_6297_;
goto v___jp_6283_;
}
}
}
v___jp_6283_:
{
lean_object* v___x_6286_; 
if (v_isShared_6281_ == 0)
{
lean_ctor_set(v___x_6280_, 1, v_a_6284_);
lean_ctor_set(v___x_6280_, 0, v___x_6282_);
v___x_6286_ = v___x_6280_;
goto v_reusejp_6285_;
}
else
{
lean_object* v_reuseFailAlloc_6290_; 
v_reuseFailAlloc_6290_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_6290_, 0, v___x_6282_);
lean_ctor_set(v_reuseFailAlloc_6290_, 1, v_a_6284_);
v___x_6286_ = v_reuseFailAlloc_6290_;
goto v_reusejp_6285_;
}
v_reusejp_6285_:
{
size_t v___x_6287_; size_t v___x_6288_; 
v___x_6287_ = ((size_t)1ULL);
v___x_6288_ = lean_usize_add(v_i_6269_, v___x_6287_);
v_i_6269_ = v___x_6288_;
v_b_6270_ = v___x_6286_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0_spec__2_spec__3___boxed(lean_object* v_goal_6337_, lean_object* v_as_6338_, lean_object* v_sz_6339_, lean_object* v_i_6340_, lean_object* v_b_6341_, lean_object* v___y_6342_, lean_object* v___y_6343_, lean_object* v___y_6344_, lean_object* v___y_6345_, lean_object* v___y_6346_){
_start:
{
size_t v_sz_boxed_6347_; size_t v_i_boxed_6348_; lean_object* v_res_6349_; 
v_sz_boxed_6347_ = lean_unbox_usize(v_sz_6339_);
lean_dec(v_sz_6339_);
v_i_boxed_6348_ = lean_unbox_usize(v_i_6340_);
lean_dec(v_i_6340_);
v_res_6349_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0_spec__2_spec__3(v_goal_6337_, v_as_6338_, v_sz_boxed_6347_, v_i_boxed_6348_, v_b_6341_, v___y_6342_, v___y_6343_, v___y_6344_, v___y_6345_);
lean_dec(v___y_6345_);
lean_dec_ref(v___y_6344_);
lean_dec(v___y_6343_);
lean_dec_ref(v___y_6342_);
lean_dec_ref(v_as_6338_);
return v_res_6349_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0_spec__2(lean_object* v_goal_6350_, lean_object* v_as_6351_, size_t v_sz_6352_, size_t v_i_6353_, lean_object* v_b_6354_, lean_object* v___y_6355_, lean_object* v___y_6356_, lean_object* v___y_6357_, lean_object* v___y_6358_){
_start:
{
uint8_t v___x_6360_; 
v___x_6360_ = lean_usize_dec_lt(v_i_6353_, v_sz_6352_);
if (v___x_6360_ == 0)
{
lean_object* v___x_6361_; 
lean_dec(v_goal_6350_);
v___x_6361_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_6361_, 0, v_b_6354_);
return v___x_6361_;
}
else
{
lean_object* v_snd_6362_; lean_object* v___x_6364_; uint8_t v_isShared_6365_; uint8_t v_isSharedCheck_6419_; 
v_snd_6362_ = lean_ctor_get(v_b_6354_, 1);
v_isSharedCheck_6419_ = !lean_is_exclusive(v_b_6354_);
if (v_isSharedCheck_6419_ == 0)
{
lean_object* v_unused_6420_; 
v_unused_6420_ = lean_ctor_get(v_b_6354_, 0);
lean_dec(v_unused_6420_);
v___x_6364_ = v_b_6354_;
v_isShared_6365_ = v_isSharedCheck_6419_;
goto v_resetjp_6363_;
}
else
{
lean_inc(v_snd_6362_);
lean_dec(v_b_6354_);
v___x_6364_ = lean_box(0);
v_isShared_6365_ = v_isSharedCheck_6419_;
goto v_resetjp_6363_;
}
v_resetjp_6363_:
{
lean_object* v___x_6366_; lean_object* v_a_6368_; lean_object* v_a_6375_; 
v___x_6366_ = lean_box(0);
v_a_6375_ = lean_array_uget(v_as_6351_, v_i_6353_);
if (lean_obj_tag(v_a_6375_) == 0)
{
v_a_6368_ = v_snd_6362_;
goto v___jp_6367_;
}
else
{
lean_object* v_val_6376_; lean_object* v___x_6378_; uint8_t v_isShared_6379_; uint8_t v_isSharedCheck_6418_; 
v_val_6376_ = lean_ctor_get(v_a_6375_, 0);
v_isSharedCheck_6418_ = !lean_is_exclusive(v_a_6375_);
if (v_isSharedCheck_6418_ == 0)
{
v___x_6378_ = v_a_6375_;
v_isShared_6379_ = v_isSharedCheck_6418_;
goto v_resetjp_6377_;
}
else
{
lean_inc(v_val_6376_);
lean_dec(v_a_6375_);
v___x_6378_ = lean_box(0);
v_isShared_6379_ = v_isSharedCheck_6418_;
goto v_resetjp_6377_;
}
v_resetjp_6377_:
{
lean_object* v___x_6380_; lean_object* v___x_6381_; uint8_t v___x_6382_; 
v___x_6380_ = lean_box(0);
v___x_6381_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0_spec__2_spec__3___closed__0));
v___x_6382_ = l_Lean_LocalDecl_isImplementationDetail(v_val_6376_);
if (v___x_6382_ == 0)
{
lean_object* v___x_6383_; lean_object* v___x_6384_; 
v___x_6383_ = l_Lean_LocalDecl_fvarId(v_val_6376_);
lean_dec(v_val_6376_);
lean_inc(v___x_6383_);
lean_inc(v_goal_6350_);
v___x_6384_ = l_Lean_Meta_splitLocalDecl_x3f(v_goal_6350_, v___x_6383_, v___y_6355_, v___y_6356_, v___y_6357_, v___y_6358_);
if (lean_obj_tag(v___x_6384_) == 0)
{
lean_object* v_a_6385_; lean_object* v___x_6387_; uint8_t v_isShared_6388_; uint8_t v_isSharedCheck_6409_; 
v_a_6385_ = lean_ctor_get(v___x_6384_, 0);
v_isSharedCheck_6409_ = !lean_is_exclusive(v___x_6384_);
if (v_isSharedCheck_6409_ == 0)
{
v___x_6387_ = v___x_6384_;
v_isShared_6388_ = v_isSharedCheck_6409_;
goto v_resetjp_6386_;
}
else
{
lean_inc(v_a_6385_);
lean_dec(v___x_6384_);
v___x_6387_ = lean_box(0);
v_isShared_6388_ = v_isSharedCheck_6409_;
goto v_resetjp_6386_;
}
v_resetjp_6386_:
{
if (lean_obj_tag(v_a_6385_) == 1)
{
lean_object* v_val_6389_; lean_object* v___x_6391_; uint8_t v_isShared_6392_; uint8_t v_isSharedCheck_6408_; 
lean_del_object(v___x_6364_);
lean_dec(v_goal_6350_);
v_val_6389_ = lean_ctor_get(v_a_6385_, 0);
v_isSharedCheck_6408_ = !lean_is_exclusive(v_a_6385_);
if (v_isSharedCheck_6408_ == 0)
{
v___x_6391_ = v_a_6385_;
v_isShared_6392_ = v_isSharedCheck_6408_;
goto v_resetjp_6390_;
}
else
{
lean_inc(v_val_6389_);
lean_dec(v_a_6385_);
v___x_6391_ = lean_box(0);
v_isShared_6392_ = v_isSharedCheck_6408_;
goto v_resetjp_6390_;
}
v_resetjp_6390_:
{
lean_object* v___x_6393_; lean_object* v___x_6394_; lean_object* v___x_6396_; 
v___x_6393_ = lean_array_mk(v_val_6389_);
v___x_6394_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6394_, 0, v___x_6393_);
lean_ctor_set(v___x_6394_, 1, v___x_6383_);
if (v_isShared_6392_ == 0)
{
lean_ctor_set(v___x_6391_, 0, v___x_6394_);
v___x_6396_ = v___x_6391_;
goto v_reusejp_6395_;
}
else
{
lean_object* v_reuseFailAlloc_6407_; 
v_reuseFailAlloc_6407_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6407_, 0, v___x_6394_);
v___x_6396_ = v_reuseFailAlloc_6407_;
goto v_reusejp_6395_;
}
v_reusejp_6395_:
{
lean_object* v___x_6398_; 
if (v_isShared_6379_ == 0)
{
lean_ctor_set(v___x_6378_, 0, v___x_6396_);
v___x_6398_ = v___x_6378_;
goto v_reusejp_6397_;
}
else
{
lean_object* v_reuseFailAlloc_6406_; 
v_reuseFailAlloc_6406_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6406_, 0, v___x_6396_);
v___x_6398_ = v_reuseFailAlloc_6406_;
goto v_reusejp_6397_;
}
v_reusejp_6397_:
{
lean_object* v___x_6399_; lean_object* v___x_6400_; lean_object* v___x_6401_; lean_object* v___x_6402_; lean_object* v___x_6404_; 
v___x_6399_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6399_, 0, v___x_6398_);
lean_ctor_set(v___x_6399_, 1, v___x_6380_);
v___x_6400_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_6400_, 0, v___x_6399_);
v___x_6401_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_6401_, 0, v___x_6400_);
v___x_6402_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6402_, 0, v___x_6401_);
lean_ctor_set(v___x_6402_, 1, v_snd_6362_);
if (v_isShared_6388_ == 0)
{
lean_ctor_set(v___x_6387_, 0, v___x_6402_);
v___x_6404_ = v___x_6387_;
goto v_reusejp_6403_;
}
else
{
lean_object* v_reuseFailAlloc_6405_; 
v_reuseFailAlloc_6405_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6405_, 0, v___x_6402_);
v___x_6404_ = v_reuseFailAlloc_6405_;
goto v_reusejp_6403_;
}
v_reusejp_6403_:
{
return v___x_6404_;
}
}
}
}
}
else
{
lean_del_object(v___x_6387_);
lean_dec(v_a_6385_);
lean_dec(v___x_6383_);
lean_del_object(v___x_6378_);
lean_dec(v_snd_6362_);
v_a_6368_ = v___x_6381_;
goto v___jp_6367_;
}
}
}
else
{
lean_object* v_a_6410_; lean_object* v___x_6412_; uint8_t v_isShared_6413_; uint8_t v_isSharedCheck_6417_; 
lean_dec(v___x_6383_);
lean_del_object(v___x_6378_);
lean_del_object(v___x_6364_);
lean_dec(v_snd_6362_);
lean_dec(v_goal_6350_);
v_a_6410_ = lean_ctor_get(v___x_6384_, 0);
v_isSharedCheck_6417_ = !lean_is_exclusive(v___x_6384_);
if (v_isSharedCheck_6417_ == 0)
{
v___x_6412_ = v___x_6384_;
v_isShared_6413_ = v_isSharedCheck_6417_;
goto v_resetjp_6411_;
}
else
{
lean_inc(v_a_6410_);
lean_dec(v___x_6384_);
v___x_6412_ = lean_box(0);
v_isShared_6413_ = v_isSharedCheck_6417_;
goto v_resetjp_6411_;
}
v_resetjp_6411_:
{
lean_object* v___x_6415_; 
if (v_isShared_6413_ == 0)
{
v___x_6415_ = v___x_6412_;
goto v_reusejp_6414_;
}
else
{
lean_object* v_reuseFailAlloc_6416_; 
v_reuseFailAlloc_6416_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6416_, 0, v_a_6410_);
v___x_6415_ = v_reuseFailAlloc_6416_;
goto v_reusejp_6414_;
}
v_reusejp_6414_:
{
return v___x_6415_;
}
}
}
}
else
{
lean_del_object(v___x_6378_);
lean_dec(v_val_6376_);
lean_dec(v_snd_6362_);
v_a_6368_ = v___x_6381_;
goto v___jp_6367_;
}
}
}
v___jp_6367_:
{
lean_object* v___x_6370_; 
if (v_isShared_6365_ == 0)
{
lean_ctor_set(v___x_6364_, 1, v_a_6368_);
lean_ctor_set(v___x_6364_, 0, v___x_6366_);
v___x_6370_ = v___x_6364_;
goto v_reusejp_6369_;
}
else
{
lean_object* v_reuseFailAlloc_6374_; 
v_reuseFailAlloc_6374_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_6374_, 0, v___x_6366_);
lean_ctor_set(v_reuseFailAlloc_6374_, 1, v_a_6368_);
v___x_6370_ = v_reuseFailAlloc_6374_;
goto v_reusejp_6369_;
}
v_reusejp_6369_:
{
size_t v___x_6371_; size_t v___x_6372_; lean_object* v___x_6373_; 
v___x_6371_ = ((size_t)1ULL);
v___x_6372_ = lean_usize_add(v_i_6353_, v___x_6371_);
v___x_6373_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0_spec__2_spec__3(v_goal_6350_, v_as_6351_, v_sz_6352_, v___x_6372_, v___x_6370_, v___y_6355_, v___y_6356_, v___y_6357_, v___y_6358_);
return v___x_6373_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0_spec__2___boxed(lean_object* v_goal_6421_, lean_object* v_as_6422_, lean_object* v_sz_6423_, lean_object* v_i_6424_, lean_object* v_b_6425_, lean_object* v___y_6426_, lean_object* v___y_6427_, lean_object* v___y_6428_, lean_object* v___y_6429_, lean_object* v___y_6430_){
_start:
{
size_t v_sz_boxed_6431_; size_t v_i_boxed_6432_; lean_object* v_res_6433_; 
v_sz_boxed_6431_ = lean_unbox_usize(v_sz_6423_);
lean_dec(v_sz_6423_);
v_i_boxed_6432_ = lean_unbox_usize(v_i_6424_);
lean_dec(v_i_6424_);
v_res_6433_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0_spec__2(v_goal_6421_, v_as_6422_, v_sz_boxed_6431_, v_i_boxed_6432_, v_b_6425_, v___y_6426_, v___y_6427_, v___y_6428_, v___y_6429_);
lean_dec(v___y_6429_);
lean_dec_ref(v___y_6428_);
lean_dec(v___y_6427_);
lean_dec_ref(v___y_6426_);
lean_dec_ref(v_as_6422_);
return v_res_6433_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0(lean_object* v_init_6434_, lean_object* v_goal_6435_, lean_object* v_n_6436_, lean_object* v_b_6437_, lean_object* v___y_6438_, lean_object* v___y_6439_, lean_object* v___y_6440_, lean_object* v___y_6441_){
_start:
{
if (lean_obj_tag(v_n_6436_) == 0)
{
lean_object* v_cs_6443_; lean_object* v___x_6444_; lean_object* v___x_6445_; size_t v_sz_6446_; size_t v___x_6447_; lean_object* v___x_6448_; 
v_cs_6443_ = lean_ctor_get(v_n_6436_, 0);
v___x_6444_ = lean_box(0);
v___x_6445_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6445_, 0, v___x_6444_);
lean_ctor_set(v___x_6445_, 1, v_b_6437_);
v_sz_6446_ = lean_array_size(v_cs_6443_);
v___x_6447_ = ((size_t)0ULL);
v___x_6448_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0_spec__1(v_init_6434_, v_goal_6435_, v_cs_6443_, v_sz_6446_, v___x_6447_, v___x_6445_, v___y_6438_, v___y_6439_, v___y_6440_, v___y_6441_);
if (lean_obj_tag(v___x_6448_) == 0)
{
lean_object* v_a_6449_; lean_object* v___x_6451_; uint8_t v_isShared_6452_; uint8_t v_isSharedCheck_6463_; 
v_a_6449_ = lean_ctor_get(v___x_6448_, 0);
v_isSharedCheck_6463_ = !lean_is_exclusive(v___x_6448_);
if (v_isSharedCheck_6463_ == 0)
{
v___x_6451_ = v___x_6448_;
v_isShared_6452_ = v_isSharedCheck_6463_;
goto v_resetjp_6450_;
}
else
{
lean_inc(v_a_6449_);
lean_dec(v___x_6448_);
v___x_6451_ = lean_box(0);
v_isShared_6452_ = v_isSharedCheck_6463_;
goto v_resetjp_6450_;
}
v_resetjp_6450_:
{
lean_object* v_fst_6453_; 
v_fst_6453_ = lean_ctor_get(v_a_6449_, 0);
if (lean_obj_tag(v_fst_6453_) == 0)
{
lean_object* v_snd_6454_; lean_object* v___x_6455_; lean_object* v___x_6457_; 
v_snd_6454_ = lean_ctor_get(v_a_6449_, 1);
lean_inc(v_snd_6454_);
lean_dec(v_a_6449_);
v___x_6455_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_6455_, 0, v_snd_6454_);
if (v_isShared_6452_ == 0)
{
lean_ctor_set(v___x_6451_, 0, v___x_6455_);
v___x_6457_ = v___x_6451_;
goto v_reusejp_6456_;
}
else
{
lean_object* v_reuseFailAlloc_6458_; 
v_reuseFailAlloc_6458_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6458_, 0, v___x_6455_);
v___x_6457_ = v_reuseFailAlloc_6458_;
goto v_reusejp_6456_;
}
v_reusejp_6456_:
{
return v___x_6457_;
}
}
else
{
lean_object* v_val_6459_; lean_object* v___x_6461_; 
lean_inc_ref(v_fst_6453_);
lean_dec(v_a_6449_);
v_val_6459_ = lean_ctor_get(v_fst_6453_, 0);
lean_inc(v_val_6459_);
lean_dec_ref_known(v_fst_6453_, 1);
if (v_isShared_6452_ == 0)
{
lean_ctor_set(v___x_6451_, 0, v_val_6459_);
v___x_6461_ = v___x_6451_;
goto v_reusejp_6460_;
}
else
{
lean_object* v_reuseFailAlloc_6462_; 
v_reuseFailAlloc_6462_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6462_, 0, v_val_6459_);
v___x_6461_ = v_reuseFailAlloc_6462_;
goto v_reusejp_6460_;
}
v_reusejp_6460_:
{
return v___x_6461_;
}
}
}
}
else
{
lean_object* v_a_6464_; lean_object* v___x_6466_; uint8_t v_isShared_6467_; uint8_t v_isSharedCheck_6471_; 
v_a_6464_ = lean_ctor_get(v___x_6448_, 0);
v_isSharedCheck_6471_ = !lean_is_exclusive(v___x_6448_);
if (v_isSharedCheck_6471_ == 0)
{
v___x_6466_ = v___x_6448_;
v_isShared_6467_ = v_isSharedCheck_6471_;
goto v_resetjp_6465_;
}
else
{
lean_inc(v_a_6464_);
lean_dec(v___x_6448_);
v___x_6466_ = lean_box(0);
v_isShared_6467_ = v_isSharedCheck_6471_;
goto v_resetjp_6465_;
}
v_resetjp_6465_:
{
lean_object* v___x_6469_; 
if (v_isShared_6467_ == 0)
{
v___x_6469_ = v___x_6466_;
goto v_reusejp_6468_;
}
else
{
lean_object* v_reuseFailAlloc_6470_; 
v_reuseFailAlloc_6470_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6470_, 0, v_a_6464_);
v___x_6469_ = v_reuseFailAlloc_6470_;
goto v_reusejp_6468_;
}
v_reusejp_6468_:
{
return v___x_6469_;
}
}
}
}
else
{
lean_object* v_vs_6472_; lean_object* v___x_6473_; lean_object* v___x_6474_; size_t v_sz_6475_; size_t v___x_6476_; lean_object* v___x_6477_; 
v_vs_6472_ = lean_ctor_get(v_n_6436_, 0);
v___x_6473_ = lean_box(0);
v___x_6474_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6474_, 0, v___x_6473_);
lean_ctor_set(v___x_6474_, 1, v_b_6437_);
v_sz_6475_ = lean_array_size(v_vs_6472_);
v___x_6476_ = ((size_t)0ULL);
v___x_6477_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0_spec__2(v_goal_6435_, v_vs_6472_, v_sz_6475_, v___x_6476_, v___x_6474_, v___y_6438_, v___y_6439_, v___y_6440_, v___y_6441_);
if (lean_obj_tag(v___x_6477_) == 0)
{
lean_object* v_a_6478_; lean_object* v___x_6480_; uint8_t v_isShared_6481_; uint8_t v_isSharedCheck_6492_; 
v_a_6478_ = lean_ctor_get(v___x_6477_, 0);
v_isSharedCheck_6492_ = !lean_is_exclusive(v___x_6477_);
if (v_isSharedCheck_6492_ == 0)
{
v___x_6480_ = v___x_6477_;
v_isShared_6481_ = v_isSharedCheck_6492_;
goto v_resetjp_6479_;
}
else
{
lean_inc(v_a_6478_);
lean_dec(v___x_6477_);
v___x_6480_ = lean_box(0);
v_isShared_6481_ = v_isSharedCheck_6492_;
goto v_resetjp_6479_;
}
v_resetjp_6479_:
{
lean_object* v_fst_6482_; 
v_fst_6482_ = lean_ctor_get(v_a_6478_, 0);
if (lean_obj_tag(v_fst_6482_) == 0)
{
lean_object* v_snd_6483_; lean_object* v___x_6484_; lean_object* v___x_6486_; 
v_snd_6483_ = lean_ctor_get(v_a_6478_, 1);
lean_inc(v_snd_6483_);
lean_dec(v_a_6478_);
v___x_6484_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_6484_, 0, v_snd_6483_);
if (v_isShared_6481_ == 0)
{
lean_ctor_set(v___x_6480_, 0, v___x_6484_);
v___x_6486_ = v___x_6480_;
goto v_reusejp_6485_;
}
else
{
lean_object* v_reuseFailAlloc_6487_; 
v_reuseFailAlloc_6487_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6487_, 0, v___x_6484_);
v___x_6486_ = v_reuseFailAlloc_6487_;
goto v_reusejp_6485_;
}
v_reusejp_6485_:
{
return v___x_6486_;
}
}
else
{
lean_object* v_val_6488_; lean_object* v___x_6490_; 
lean_inc_ref(v_fst_6482_);
lean_dec(v_a_6478_);
v_val_6488_ = lean_ctor_get(v_fst_6482_, 0);
lean_inc(v_val_6488_);
lean_dec_ref_known(v_fst_6482_, 1);
if (v_isShared_6481_ == 0)
{
lean_ctor_set(v___x_6480_, 0, v_val_6488_);
v___x_6490_ = v___x_6480_;
goto v_reusejp_6489_;
}
else
{
lean_object* v_reuseFailAlloc_6491_; 
v_reuseFailAlloc_6491_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6491_, 0, v_val_6488_);
v___x_6490_ = v_reuseFailAlloc_6491_;
goto v_reusejp_6489_;
}
v_reusejp_6489_:
{
return v___x_6490_;
}
}
}
}
else
{
lean_object* v_a_6493_; lean_object* v___x_6495_; uint8_t v_isShared_6496_; uint8_t v_isSharedCheck_6500_; 
v_a_6493_ = lean_ctor_get(v___x_6477_, 0);
v_isSharedCheck_6500_ = !lean_is_exclusive(v___x_6477_);
if (v_isSharedCheck_6500_ == 0)
{
v___x_6495_ = v___x_6477_;
v_isShared_6496_ = v_isSharedCheck_6500_;
goto v_resetjp_6494_;
}
else
{
lean_inc(v_a_6493_);
lean_dec(v___x_6477_);
v___x_6495_ = lean_box(0);
v_isShared_6496_ = v_isSharedCheck_6500_;
goto v_resetjp_6494_;
}
v_resetjp_6494_:
{
lean_object* v___x_6498_; 
if (v_isShared_6496_ == 0)
{
v___x_6498_ = v___x_6495_;
goto v_reusejp_6497_;
}
else
{
lean_object* v_reuseFailAlloc_6499_; 
v_reuseFailAlloc_6499_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6499_, 0, v_a_6493_);
v___x_6498_ = v_reuseFailAlloc_6499_;
goto v_reusejp_6497_;
}
v_reusejp_6497_:
{
return v___x_6498_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0_spec__1(lean_object* v_init_6501_, lean_object* v_goal_6502_, lean_object* v_as_6503_, size_t v_sz_6504_, size_t v_i_6505_, lean_object* v_b_6506_, lean_object* v___y_6507_, lean_object* v___y_6508_, lean_object* v___y_6509_, lean_object* v___y_6510_){
_start:
{
uint8_t v___x_6512_; 
v___x_6512_ = lean_usize_dec_lt(v_i_6505_, v_sz_6504_);
if (v___x_6512_ == 0)
{
lean_object* v___x_6513_; 
lean_dec(v_goal_6502_);
v___x_6513_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_6513_, 0, v_b_6506_);
return v___x_6513_;
}
else
{
lean_object* v_snd_6514_; lean_object* v___x_6516_; uint8_t v_isShared_6517_; uint8_t v_isSharedCheck_6548_; 
v_snd_6514_ = lean_ctor_get(v_b_6506_, 1);
v_isSharedCheck_6548_ = !lean_is_exclusive(v_b_6506_);
if (v_isSharedCheck_6548_ == 0)
{
lean_object* v_unused_6549_; 
v_unused_6549_ = lean_ctor_get(v_b_6506_, 0);
lean_dec(v_unused_6549_);
v___x_6516_ = v_b_6506_;
v_isShared_6517_ = v_isSharedCheck_6548_;
goto v_resetjp_6515_;
}
else
{
lean_inc(v_snd_6514_);
lean_dec(v_b_6506_);
v___x_6516_ = lean_box(0);
v_isShared_6517_ = v_isSharedCheck_6548_;
goto v_resetjp_6515_;
}
v_resetjp_6515_:
{
lean_object* v_a_6518_; lean_object* v___x_6519_; 
v_a_6518_ = lean_array_uget_borrowed(v_as_6503_, v_i_6505_);
lean_inc(v_snd_6514_);
lean_inc(v_goal_6502_);
v___x_6519_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0(v_init_6501_, v_goal_6502_, v_a_6518_, v_snd_6514_, v___y_6507_, v___y_6508_, v___y_6509_, v___y_6510_);
if (lean_obj_tag(v___x_6519_) == 0)
{
lean_object* v_a_6520_; lean_object* v___x_6522_; uint8_t v_isShared_6523_; uint8_t v_isSharedCheck_6539_; 
v_a_6520_ = lean_ctor_get(v___x_6519_, 0);
v_isSharedCheck_6539_ = !lean_is_exclusive(v___x_6519_);
if (v_isSharedCheck_6539_ == 0)
{
v___x_6522_ = v___x_6519_;
v_isShared_6523_ = v_isSharedCheck_6539_;
goto v_resetjp_6521_;
}
else
{
lean_inc(v_a_6520_);
lean_dec(v___x_6519_);
v___x_6522_ = lean_box(0);
v_isShared_6523_ = v_isSharedCheck_6539_;
goto v_resetjp_6521_;
}
v_resetjp_6521_:
{
if (lean_obj_tag(v_a_6520_) == 0)
{
lean_object* v___x_6524_; lean_object* v___x_6526_; 
lean_dec(v_goal_6502_);
v___x_6524_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_6524_, 0, v_a_6520_);
if (v_isShared_6517_ == 0)
{
lean_ctor_set(v___x_6516_, 0, v___x_6524_);
v___x_6526_ = v___x_6516_;
goto v_reusejp_6525_;
}
else
{
lean_object* v_reuseFailAlloc_6530_; 
v_reuseFailAlloc_6530_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_6530_, 0, v___x_6524_);
lean_ctor_set(v_reuseFailAlloc_6530_, 1, v_snd_6514_);
v___x_6526_ = v_reuseFailAlloc_6530_;
goto v_reusejp_6525_;
}
v_reusejp_6525_:
{
lean_object* v___x_6528_; 
if (v_isShared_6523_ == 0)
{
lean_ctor_set(v___x_6522_, 0, v___x_6526_);
v___x_6528_ = v___x_6522_;
goto v_reusejp_6527_;
}
else
{
lean_object* v_reuseFailAlloc_6529_; 
v_reuseFailAlloc_6529_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6529_, 0, v___x_6526_);
v___x_6528_ = v_reuseFailAlloc_6529_;
goto v_reusejp_6527_;
}
v_reusejp_6527_:
{
return v___x_6528_;
}
}
}
else
{
lean_object* v_a_6531_; lean_object* v___x_6532_; lean_object* v___x_6534_; 
lean_del_object(v___x_6522_);
lean_dec(v_snd_6514_);
v_a_6531_ = lean_ctor_get(v_a_6520_, 0);
lean_inc(v_a_6531_);
lean_dec_ref_known(v_a_6520_, 1);
v___x_6532_ = lean_box(0);
if (v_isShared_6517_ == 0)
{
lean_ctor_set(v___x_6516_, 1, v_a_6531_);
lean_ctor_set(v___x_6516_, 0, v___x_6532_);
v___x_6534_ = v___x_6516_;
goto v_reusejp_6533_;
}
else
{
lean_object* v_reuseFailAlloc_6538_; 
v_reuseFailAlloc_6538_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_6538_, 0, v___x_6532_);
lean_ctor_set(v_reuseFailAlloc_6538_, 1, v_a_6531_);
v___x_6534_ = v_reuseFailAlloc_6538_;
goto v_reusejp_6533_;
}
v_reusejp_6533_:
{
size_t v___x_6535_; size_t v___x_6536_; 
v___x_6535_ = ((size_t)1ULL);
v___x_6536_ = lean_usize_add(v_i_6505_, v___x_6535_);
v_i_6505_ = v___x_6536_;
v_b_6506_ = v___x_6534_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_6540_; lean_object* v___x_6542_; uint8_t v_isShared_6543_; uint8_t v_isSharedCheck_6547_; 
lean_del_object(v___x_6516_);
lean_dec(v_snd_6514_);
lean_dec(v_goal_6502_);
v_a_6540_ = lean_ctor_get(v___x_6519_, 0);
v_isSharedCheck_6547_ = !lean_is_exclusive(v___x_6519_);
if (v_isSharedCheck_6547_ == 0)
{
v___x_6542_ = v___x_6519_;
v_isShared_6543_ = v_isSharedCheck_6547_;
goto v_resetjp_6541_;
}
else
{
lean_inc(v_a_6540_);
lean_dec(v___x_6519_);
v___x_6542_ = lean_box(0);
v_isShared_6543_ = v_isSharedCheck_6547_;
goto v_resetjp_6541_;
}
v_resetjp_6541_:
{
lean_object* v___x_6545_; 
if (v_isShared_6543_ == 0)
{
v___x_6545_ = v___x_6542_;
goto v_reusejp_6544_;
}
else
{
lean_object* v_reuseFailAlloc_6546_; 
v_reuseFailAlloc_6546_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6546_, 0, v_a_6540_);
v___x_6545_ = v_reuseFailAlloc_6546_;
goto v_reusejp_6544_;
}
v_reusejp_6544_:
{
return v___x_6545_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0_spec__1___boxed(lean_object* v_init_6550_, lean_object* v_goal_6551_, lean_object* v_as_6552_, lean_object* v_sz_6553_, lean_object* v_i_6554_, lean_object* v_b_6555_, lean_object* v___y_6556_, lean_object* v___y_6557_, lean_object* v___y_6558_, lean_object* v___y_6559_, lean_object* v___y_6560_){
_start:
{
size_t v_sz_boxed_6561_; size_t v_i_boxed_6562_; lean_object* v_res_6563_; 
v_sz_boxed_6561_ = lean_unbox_usize(v_sz_6553_);
lean_dec(v_sz_6553_);
v_i_boxed_6562_ = lean_unbox_usize(v_i_6554_);
lean_dec(v_i_6554_);
v_res_6563_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0_spec__1(v_init_6550_, v_goal_6551_, v_as_6552_, v_sz_boxed_6561_, v_i_boxed_6562_, v_b_6555_, v___y_6556_, v___y_6557_, v___y_6558_, v___y_6559_);
lean_dec(v___y_6559_);
lean_dec_ref(v___y_6558_);
lean_dec(v___y_6557_);
lean_dec_ref(v___y_6556_);
lean_dec_ref(v_as_6552_);
lean_dec_ref(v_init_6550_);
return v_res_6563_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0___boxed(lean_object* v_init_6564_, lean_object* v_goal_6565_, lean_object* v_n_6566_, lean_object* v_b_6567_, lean_object* v___y_6568_, lean_object* v___y_6569_, lean_object* v___y_6570_, lean_object* v___y_6571_, lean_object* v___y_6572_){
_start:
{
lean_object* v_res_6573_; 
v_res_6573_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0(v_init_6564_, v_goal_6565_, v_n_6566_, v_b_6567_, v___y_6568_, v___y_6569_, v___y_6570_, v___y_6571_);
lean_dec(v___y_6571_);
lean_dec_ref(v___y_6570_);
lean_dec(v___y_6569_);
lean_dec_ref(v___y_6568_);
lean_dec_ref(v_n_6566_);
lean_dec_ref(v_init_6564_);
return v_res_6573_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__1_spec__4(lean_object* v_goal_6577_, lean_object* v_as_6578_, size_t v_sz_6579_, size_t v_i_6580_, lean_object* v_b_6581_, lean_object* v___y_6582_, lean_object* v___y_6583_, lean_object* v___y_6584_, lean_object* v___y_6585_){
_start:
{
uint8_t v___x_6587_; 
v___x_6587_ = lean_usize_dec_lt(v_i_6580_, v_sz_6579_);
if (v___x_6587_ == 0)
{
lean_object* v___x_6588_; 
lean_dec(v_goal_6577_);
v___x_6588_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_6588_, 0, v_b_6581_);
return v___x_6588_;
}
else
{
lean_object* v_snd_6589_; lean_object* v___x_6591_; uint8_t v_isShared_6592_; uint8_t v_isSharedCheck_6645_; 
v_snd_6589_ = lean_ctor_get(v_b_6581_, 1);
v_isSharedCheck_6645_ = !lean_is_exclusive(v_b_6581_);
if (v_isSharedCheck_6645_ == 0)
{
lean_object* v_unused_6646_; 
v_unused_6646_ = lean_ctor_get(v_b_6581_, 0);
lean_dec(v_unused_6646_);
v___x_6591_ = v_b_6581_;
v_isShared_6592_ = v_isSharedCheck_6645_;
goto v_resetjp_6590_;
}
else
{
lean_inc(v_snd_6589_);
lean_dec(v_b_6581_);
v___x_6591_ = lean_box(0);
v_isShared_6592_ = v_isSharedCheck_6645_;
goto v_resetjp_6590_;
}
v_resetjp_6590_:
{
lean_object* v___x_6593_; lean_object* v_a_6595_; lean_object* v_a_6602_; 
v___x_6593_ = lean_box(0);
v_a_6602_ = lean_array_uget(v_as_6578_, v_i_6580_);
if (lean_obj_tag(v_a_6602_) == 0)
{
v_a_6595_ = v_snd_6589_;
goto v___jp_6594_;
}
else
{
lean_object* v_val_6603_; lean_object* v___x_6605_; uint8_t v_isShared_6606_; uint8_t v_isSharedCheck_6644_; 
v_val_6603_ = lean_ctor_get(v_a_6602_, 0);
v_isSharedCheck_6644_ = !lean_is_exclusive(v_a_6602_);
if (v_isSharedCheck_6644_ == 0)
{
v___x_6605_ = v_a_6602_;
v_isShared_6606_ = v_isSharedCheck_6644_;
goto v_resetjp_6604_;
}
else
{
lean_inc(v_val_6603_);
lean_dec(v_a_6602_);
v___x_6605_ = lean_box(0);
v_isShared_6606_ = v_isSharedCheck_6644_;
goto v_resetjp_6604_;
}
v_resetjp_6604_:
{
lean_object* v___x_6607_; lean_object* v___x_6608_; uint8_t v___x_6609_; 
v___x_6607_ = lean_box(0);
v___x_6608_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__1_spec__4___closed__0));
v___x_6609_ = l_Lean_LocalDecl_isImplementationDetail(v_val_6603_);
if (v___x_6609_ == 0)
{
lean_object* v___x_6610_; lean_object* v___x_6611_; 
v___x_6610_ = l_Lean_LocalDecl_fvarId(v_val_6603_);
lean_dec(v_val_6603_);
lean_inc(v___x_6610_);
lean_inc(v_goal_6577_);
v___x_6611_ = l_Lean_Meta_splitLocalDecl_x3f(v_goal_6577_, v___x_6610_, v___y_6582_, v___y_6583_, v___y_6584_, v___y_6585_);
if (lean_obj_tag(v___x_6611_) == 0)
{
lean_object* v_a_6612_; lean_object* v___x_6614_; uint8_t v_isShared_6615_; uint8_t v_isSharedCheck_6635_; 
v_a_6612_ = lean_ctor_get(v___x_6611_, 0);
v_isSharedCheck_6635_ = !lean_is_exclusive(v___x_6611_);
if (v_isSharedCheck_6635_ == 0)
{
v___x_6614_ = v___x_6611_;
v_isShared_6615_ = v_isSharedCheck_6635_;
goto v_resetjp_6613_;
}
else
{
lean_inc(v_a_6612_);
lean_dec(v___x_6611_);
v___x_6614_ = lean_box(0);
v_isShared_6615_ = v_isSharedCheck_6635_;
goto v_resetjp_6613_;
}
v_resetjp_6613_:
{
if (lean_obj_tag(v_a_6612_) == 1)
{
lean_object* v_val_6616_; lean_object* v___x_6618_; uint8_t v_isShared_6619_; uint8_t v_isSharedCheck_6634_; 
lean_del_object(v___x_6591_);
lean_dec(v_goal_6577_);
v_val_6616_ = lean_ctor_get(v_a_6612_, 0);
v_isSharedCheck_6634_ = !lean_is_exclusive(v_a_6612_);
if (v_isSharedCheck_6634_ == 0)
{
v___x_6618_ = v_a_6612_;
v_isShared_6619_ = v_isSharedCheck_6634_;
goto v_resetjp_6617_;
}
else
{
lean_inc(v_val_6616_);
lean_dec(v_a_6612_);
v___x_6618_ = lean_box(0);
v_isShared_6619_ = v_isSharedCheck_6634_;
goto v_resetjp_6617_;
}
v_resetjp_6617_:
{
lean_object* v___x_6620_; lean_object* v___x_6621_; lean_object* v___x_6623_; 
v___x_6620_ = lean_array_mk(v_val_6616_);
v___x_6621_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6621_, 0, v___x_6620_);
lean_ctor_set(v___x_6621_, 1, v___x_6610_);
if (v_isShared_6619_ == 0)
{
lean_ctor_set(v___x_6618_, 0, v___x_6621_);
v___x_6623_ = v___x_6618_;
goto v_reusejp_6622_;
}
else
{
lean_object* v_reuseFailAlloc_6633_; 
v_reuseFailAlloc_6633_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6633_, 0, v___x_6621_);
v___x_6623_ = v_reuseFailAlloc_6633_;
goto v_reusejp_6622_;
}
v_reusejp_6622_:
{
lean_object* v___x_6625_; 
if (v_isShared_6606_ == 0)
{
lean_ctor_set(v___x_6605_, 0, v___x_6623_);
v___x_6625_ = v___x_6605_;
goto v_reusejp_6624_;
}
else
{
lean_object* v_reuseFailAlloc_6632_; 
v_reuseFailAlloc_6632_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6632_, 0, v___x_6623_);
v___x_6625_ = v_reuseFailAlloc_6632_;
goto v_reusejp_6624_;
}
v_reusejp_6624_:
{
lean_object* v___x_6626_; lean_object* v___x_6627_; lean_object* v___x_6628_; lean_object* v___x_6630_; 
v___x_6626_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6626_, 0, v___x_6625_);
lean_ctor_set(v___x_6626_, 1, v___x_6607_);
v___x_6627_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_6627_, 0, v___x_6626_);
v___x_6628_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6628_, 0, v___x_6627_);
lean_ctor_set(v___x_6628_, 1, v_snd_6589_);
if (v_isShared_6615_ == 0)
{
lean_ctor_set(v___x_6614_, 0, v___x_6628_);
v___x_6630_ = v___x_6614_;
goto v_reusejp_6629_;
}
else
{
lean_object* v_reuseFailAlloc_6631_; 
v_reuseFailAlloc_6631_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6631_, 0, v___x_6628_);
v___x_6630_ = v_reuseFailAlloc_6631_;
goto v_reusejp_6629_;
}
v_reusejp_6629_:
{
return v___x_6630_;
}
}
}
}
}
else
{
lean_del_object(v___x_6614_);
lean_dec(v_a_6612_);
lean_dec(v___x_6610_);
lean_del_object(v___x_6605_);
lean_dec(v_snd_6589_);
v_a_6595_ = v___x_6608_;
goto v___jp_6594_;
}
}
}
else
{
lean_object* v_a_6636_; lean_object* v___x_6638_; uint8_t v_isShared_6639_; uint8_t v_isSharedCheck_6643_; 
lean_dec(v___x_6610_);
lean_del_object(v___x_6605_);
lean_del_object(v___x_6591_);
lean_dec(v_snd_6589_);
lean_dec(v_goal_6577_);
v_a_6636_ = lean_ctor_get(v___x_6611_, 0);
v_isSharedCheck_6643_ = !lean_is_exclusive(v___x_6611_);
if (v_isSharedCheck_6643_ == 0)
{
v___x_6638_ = v___x_6611_;
v_isShared_6639_ = v_isSharedCheck_6643_;
goto v_resetjp_6637_;
}
else
{
lean_inc(v_a_6636_);
lean_dec(v___x_6611_);
v___x_6638_ = lean_box(0);
v_isShared_6639_ = v_isSharedCheck_6643_;
goto v_resetjp_6637_;
}
v_resetjp_6637_:
{
lean_object* v___x_6641_; 
if (v_isShared_6639_ == 0)
{
v___x_6641_ = v___x_6638_;
goto v_reusejp_6640_;
}
else
{
lean_object* v_reuseFailAlloc_6642_; 
v_reuseFailAlloc_6642_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6642_, 0, v_a_6636_);
v___x_6641_ = v_reuseFailAlloc_6642_;
goto v_reusejp_6640_;
}
v_reusejp_6640_:
{
return v___x_6641_;
}
}
}
}
else
{
lean_del_object(v___x_6605_);
lean_dec(v_val_6603_);
lean_dec(v_snd_6589_);
v_a_6595_ = v___x_6608_;
goto v___jp_6594_;
}
}
}
v___jp_6594_:
{
lean_object* v___x_6597_; 
if (v_isShared_6592_ == 0)
{
lean_ctor_set(v___x_6591_, 1, v_a_6595_);
lean_ctor_set(v___x_6591_, 0, v___x_6593_);
v___x_6597_ = v___x_6591_;
goto v_reusejp_6596_;
}
else
{
lean_object* v_reuseFailAlloc_6601_; 
v_reuseFailAlloc_6601_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_6601_, 0, v___x_6593_);
lean_ctor_set(v_reuseFailAlloc_6601_, 1, v_a_6595_);
v___x_6597_ = v_reuseFailAlloc_6601_;
goto v_reusejp_6596_;
}
v_reusejp_6596_:
{
size_t v___x_6598_; size_t v___x_6599_; 
v___x_6598_ = ((size_t)1ULL);
v___x_6599_ = lean_usize_add(v_i_6580_, v___x_6598_);
v_i_6580_ = v___x_6599_;
v_b_6581_ = v___x_6597_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__1_spec__4___boxed(lean_object* v_goal_6647_, lean_object* v_as_6648_, lean_object* v_sz_6649_, lean_object* v_i_6650_, lean_object* v_b_6651_, lean_object* v___y_6652_, lean_object* v___y_6653_, lean_object* v___y_6654_, lean_object* v___y_6655_, lean_object* v___y_6656_){
_start:
{
size_t v_sz_boxed_6657_; size_t v_i_boxed_6658_; lean_object* v_res_6659_; 
v_sz_boxed_6657_ = lean_unbox_usize(v_sz_6649_);
lean_dec(v_sz_6649_);
v_i_boxed_6658_ = lean_unbox_usize(v_i_6650_);
lean_dec(v_i_6650_);
v_res_6659_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__1_spec__4(v_goal_6647_, v_as_6648_, v_sz_boxed_6657_, v_i_boxed_6658_, v_b_6651_, v___y_6652_, v___y_6653_, v___y_6654_, v___y_6655_);
lean_dec(v___y_6655_);
lean_dec_ref(v___y_6654_);
lean_dec(v___y_6653_);
lean_dec_ref(v___y_6652_);
lean_dec_ref(v_as_6648_);
return v_res_6659_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__1(lean_object* v_goal_6660_, lean_object* v_as_6661_, size_t v_sz_6662_, size_t v_i_6663_, lean_object* v_b_6664_, lean_object* v___y_6665_, lean_object* v___y_6666_, lean_object* v___y_6667_, lean_object* v___y_6668_){
_start:
{
uint8_t v___x_6670_; 
v___x_6670_ = lean_usize_dec_lt(v_i_6663_, v_sz_6662_);
if (v___x_6670_ == 0)
{
lean_object* v___x_6671_; 
lean_dec(v_goal_6660_);
v___x_6671_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_6671_, 0, v_b_6664_);
return v___x_6671_;
}
else
{
lean_object* v_snd_6672_; lean_object* v___x_6674_; uint8_t v_isShared_6675_; uint8_t v_isSharedCheck_6728_; 
v_snd_6672_ = lean_ctor_get(v_b_6664_, 1);
v_isSharedCheck_6728_ = !lean_is_exclusive(v_b_6664_);
if (v_isSharedCheck_6728_ == 0)
{
lean_object* v_unused_6729_; 
v_unused_6729_ = lean_ctor_get(v_b_6664_, 0);
lean_dec(v_unused_6729_);
v___x_6674_ = v_b_6664_;
v_isShared_6675_ = v_isSharedCheck_6728_;
goto v_resetjp_6673_;
}
else
{
lean_inc(v_snd_6672_);
lean_dec(v_b_6664_);
v___x_6674_ = lean_box(0);
v_isShared_6675_ = v_isSharedCheck_6728_;
goto v_resetjp_6673_;
}
v_resetjp_6673_:
{
lean_object* v___x_6676_; lean_object* v_a_6678_; lean_object* v_a_6685_; 
v___x_6676_ = lean_box(0);
v_a_6685_ = lean_array_uget(v_as_6661_, v_i_6663_);
if (lean_obj_tag(v_a_6685_) == 0)
{
v_a_6678_ = v_snd_6672_;
goto v___jp_6677_;
}
else
{
lean_object* v_val_6686_; lean_object* v___x_6688_; uint8_t v_isShared_6689_; uint8_t v_isSharedCheck_6727_; 
v_val_6686_ = lean_ctor_get(v_a_6685_, 0);
v_isSharedCheck_6727_ = !lean_is_exclusive(v_a_6685_);
if (v_isSharedCheck_6727_ == 0)
{
v___x_6688_ = v_a_6685_;
v_isShared_6689_ = v_isSharedCheck_6727_;
goto v_resetjp_6687_;
}
else
{
lean_inc(v_val_6686_);
lean_dec(v_a_6685_);
v___x_6688_ = lean_box(0);
v_isShared_6689_ = v_isSharedCheck_6727_;
goto v_resetjp_6687_;
}
v_resetjp_6687_:
{
lean_object* v___x_6690_; lean_object* v___x_6691_; uint8_t v___x_6692_; 
v___x_6690_ = lean_box(0);
v___x_6691_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__1_spec__4___closed__0));
v___x_6692_ = l_Lean_LocalDecl_isImplementationDetail(v_val_6686_);
if (v___x_6692_ == 0)
{
lean_object* v___x_6693_; lean_object* v___x_6694_; 
v___x_6693_ = l_Lean_LocalDecl_fvarId(v_val_6686_);
lean_dec(v_val_6686_);
lean_inc(v___x_6693_);
lean_inc(v_goal_6660_);
v___x_6694_ = l_Lean_Meta_splitLocalDecl_x3f(v_goal_6660_, v___x_6693_, v___y_6665_, v___y_6666_, v___y_6667_, v___y_6668_);
if (lean_obj_tag(v___x_6694_) == 0)
{
lean_object* v_a_6695_; lean_object* v___x_6697_; uint8_t v_isShared_6698_; uint8_t v_isSharedCheck_6718_; 
v_a_6695_ = lean_ctor_get(v___x_6694_, 0);
v_isSharedCheck_6718_ = !lean_is_exclusive(v___x_6694_);
if (v_isSharedCheck_6718_ == 0)
{
v___x_6697_ = v___x_6694_;
v_isShared_6698_ = v_isSharedCheck_6718_;
goto v_resetjp_6696_;
}
else
{
lean_inc(v_a_6695_);
lean_dec(v___x_6694_);
v___x_6697_ = lean_box(0);
v_isShared_6698_ = v_isSharedCheck_6718_;
goto v_resetjp_6696_;
}
v_resetjp_6696_:
{
if (lean_obj_tag(v_a_6695_) == 1)
{
lean_object* v_val_6699_; lean_object* v___x_6701_; uint8_t v_isShared_6702_; uint8_t v_isSharedCheck_6717_; 
lean_del_object(v___x_6674_);
lean_dec(v_goal_6660_);
v_val_6699_ = lean_ctor_get(v_a_6695_, 0);
v_isSharedCheck_6717_ = !lean_is_exclusive(v_a_6695_);
if (v_isSharedCheck_6717_ == 0)
{
v___x_6701_ = v_a_6695_;
v_isShared_6702_ = v_isSharedCheck_6717_;
goto v_resetjp_6700_;
}
else
{
lean_inc(v_val_6699_);
lean_dec(v_a_6695_);
v___x_6701_ = lean_box(0);
v_isShared_6702_ = v_isSharedCheck_6717_;
goto v_resetjp_6700_;
}
v_resetjp_6700_:
{
lean_object* v___x_6703_; lean_object* v___x_6704_; lean_object* v___x_6706_; 
v___x_6703_ = lean_array_mk(v_val_6699_);
v___x_6704_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6704_, 0, v___x_6703_);
lean_ctor_set(v___x_6704_, 1, v___x_6693_);
if (v_isShared_6702_ == 0)
{
lean_ctor_set(v___x_6701_, 0, v___x_6704_);
v___x_6706_ = v___x_6701_;
goto v_reusejp_6705_;
}
else
{
lean_object* v_reuseFailAlloc_6716_; 
v_reuseFailAlloc_6716_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6716_, 0, v___x_6704_);
v___x_6706_ = v_reuseFailAlloc_6716_;
goto v_reusejp_6705_;
}
v_reusejp_6705_:
{
lean_object* v___x_6708_; 
if (v_isShared_6689_ == 0)
{
lean_ctor_set(v___x_6688_, 0, v___x_6706_);
v___x_6708_ = v___x_6688_;
goto v_reusejp_6707_;
}
else
{
lean_object* v_reuseFailAlloc_6715_; 
v_reuseFailAlloc_6715_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6715_, 0, v___x_6706_);
v___x_6708_ = v_reuseFailAlloc_6715_;
goto v_reusejp_6707_;
}
v_reusejp_6707_:
{
lean_object* v___x_6709_; lean_object* v___x_6710_; lean_object* v___x_6711_; lean_object* v___x_6713_; 
v___x_6709_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6709_, 0, v___x_6708_);
lean_ctor_set(v___x_6709_, 1, v___x_6690_);
v___x_6710_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_6710_, 0, v___x_6709_);
v___x_6711_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6711_, 0, v___x_6710_);
lean_ctor_set(v___x_6711_, 1, v_snd_6672_);
if (v_isShared_6698_ == 0)
{
lean_ctor_set(v___x_6697_, 0, v___x_6711_);
v___x_6713_ = v___x_6697_;
goto v_reusejp_6712_;
}
else
{
lean_object* v_reuseFailAlloc_6714_; 
v_reuseFailAlloc_6714_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6714_, 0, v___x_6711_);
v___x_6713_ = v_reuseFailAlloc_6714_;
goto v_reusejp_6712_;
}
v_reusejp_6712_:
{
return v___x_6713_;
}
}
}
}
}
else
{
lean_del_object(v___x_6697_);
lean_dec(v_a_6695_);
lean_dec(v___x_6693_);
lean_del_object(v___x_6688_);
lean_dec(v_snd_6672_);
v_a_6678_ = v___x_6691_;
goto v___jp_6677_;
}
}
}
else
{
lean_object* v_a_6719_; lean_object* v___x_6721_; uint8_t v_isShared_6722_; uint8_t v_isSharedCheck_6726_; 
lean_dec(v___x_6693_);
lean_del_object(v___x_6688_);
lean_del_object(v___x_6674_);
lean_dec(v_snd_6672_);
lean_dec(v_goal_6660_);
v_a_6719_ = lean_ctor_get(v___x_6694_, 0);
v_isSharedCheck_6726_ = !lean_is_exclusive(v___x_6694_);
if (v_isSharedCheck_6726_ == 0)
{
v___x_6721_ = v___x_6694_;
v_isShared_6722_ = v_isSharedCheck_6726_;
goto v_resetjp_6720_;
}
else
{
lean_inc(v_a_6719_);
lean_dec(v___x_6694_);
v___x_6721_ = lean_box(0);
v_isShared_6722_ = v_isSharedCheck_6726_;
goto v_resetjp_6720_;
}
v_resetjp_6720_:
{
lean_object* v___x_6724_; 
if (v_isShared_6722_ == 0)
{
v___x_6724_ = v___x_6721_;
goto v_reusejp_6723_;
}
else
{
lean_object* v_reuseFailAlloc_6725_; 
v_reuseFailAlloc_6725_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6725_, 0, v_a_6719_);
v___x_6724_ = v_reuseFailAlloc_6725_;
goto v_reusejp_6723_;
}
v_reusejp_6723_:
{
return v___x_6724_;
}
}
}
}
else
{
lean_del_object(v___x_6688_);
lean_dec(v_val_6686_);
lean_dec(v_snd_6672_);
v_a_6678_ = v___x_6691_;
goto v___jp_6677_;
}
}
}
v___jp_6677_:
{
lean_object* v___x_6680_; 
if (v_isShared_6675_ == 0)
{
lean_ctor_set(v___x_6674_, 1, v_a_6678_);
lean_ctor_set(v___x_6674_, 0, v___x_6676_);
v___x_6680_ = v___x_6674_;
goto v_reusejp_6679_;
}
else
{
lean_object* v_reuseFailAlloc_6684_; 
v_reuseFailAlloc_6684_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_6684_, 0, v___x_6676_);
lean_ctor_set(v_reuseFailAlloc_6684_, 1, v_a_6678_);
v___x_6680_ = v_reuseFailAlloc_6684_;
goto v_reusejp_6679_;
}
v_reusejp_6679_:
{
size_t v___x_6681_; size_t v___x_6682_; lean_object* v___x_6683_; 
v___x_6681_ = ((size_t)1ULL);
v___x_6682_ = lean_usize_add(v_i_6663_, v___x_6681_);
v___x_6683_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__1_spec__4(v_goal_6660_, v_as_6661_, v_sz_6662_, v___x_6682_, v___x_6680_, v___y_6665_, v___y_6666_, v___y_6667_, v___y_6668_);
return v___x_6683_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__1___boxed(lean_object* v_goal_6730_, lean_object* v_as_6731_, lean_object* v_sz_6732_, lean_object* v_i_6733_, lean_object* v_b_6734_, lean_object* v___y_6735_, lean_object* v___y_6736_, lean_object* v___y_6737_, lean_object* v___y_6738_, lean_object* v___y_6739_){
_start:
{
size_t v_sz_boxed_6740_; size_t v_i_boxed_6741_; lean_object* v_res_6742_; 
v_sz_boxed_6740_ = lean_unbox_usize(v_sz_6732_);
lean_dec(v_sz_6732_);
v_i_boxed_6741_ = lean_unbox_usize(v_i_6733_);
lean_dec(v_i_6733_);
v_res_6742_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__1(v_goal_6730_, v_as_6731_, v_sz_boxed_6740_, v_i_boxed_6741_, v_b_6734_, v___y_6735_, v___y_6736_, v___y_6737_, v___y_6738_);
lean_dec(v___y_6738_);
lean_dec_ref(v___y_6737_);
lean_dec(v___y_6736_);
lean_dec_ref(v___y_6735_);
lean_dec_ref(v_as_6731_);
return v_res_6742_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0(lean_object* v_goal_6743_, lean_object* v_t_6744_, lean_object* v_init_6745_, lean_object* v___y_6746_, lean_object* v___y_6747_, lean_object* v___y_6748_, lean_object* v___y_6749_){
_start:
{
lean_object* v_root_6751_; lean_object* v_tail_6752_; lean_object* v___x_6753_; 
v_root_6751_ = lean_ctor_get(v_t_6744_, 0);
v_tail_6752_ = lean_ctor_get(v_t_6744_, 1);
lean_inc(v_goal_6743_);
lean_inc_ref(v_init_6745_);
v___x_6753_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__0(v_init_6745_, v_goal_6743_, v_root_6751_, v_init_6745_, v___y_6746_, v___y_6747_, v___y_6748_, v___y_6749_);
lean_dec_ref(v_init_6745_);
if (lean_obj_tag(v___x_6753_) == 0)
{
lean_object* v_a_6754_; lean_object* v___x_6756_; uint8_t v_isShared_6757_; uint8_t v_isSharedCheck_6790_; 
v_a_6754_ = lean_ctor_get(v___x_6753_, 0);
v_isSharedCheck_6790_ = !lean_is_exclusive(v___x_6753_);
if (v_isSharedCheck_6790_ == 0)
{
v___x_6756_ = v___x_6753_;
v_isShared_6757_ = v_isSharedCheck_6790_;
goto v_resetjp_6755_;
}
else
{
lean_inc(v_a_6754_);
lean_dec(v___x_6753_);
v___x_6756_ = lean_box(0);
v_isShared_6757_ = v_isSharedCheck_6790_;
goto v_resetjp_6755_;
}
v_resetjp_6755_:
{
if (lean_obj_tag(v_a_6754_) == 0)
{
lean_object* v_a_6758_; lean_object* v___x_6760_; 
lean_dec(v_goal_6743_);
v_a_6758_ = lean_ctor_get(v_a_6754_, 0);
lean_inc(v_a_6758_);
lean_dec_ref_known(v_a_6754_, 1);
if (v_isShared_6757_ == 0)
{
lean_ctor_set(v___x_6756_, 0, v_a_6758_);
v___x_6760_ = v___x_6756_;
goto v_reusejp_6759_;
}
else
{
lean_object* v_reuseFailAlloc_6761_; 
v_reuseFailAlloc_6761_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6761_, 0, v_a_6758_);
v___x_6760_ = v_reuseFailAlloc_6761_;
goto v_reusejp_6759_;
}
v_reusejp_6759_:
{
return v___x_6760_;
}
}
else
{
lean_object* v_a_6762_; lean_object* v___x_6763_; lean_object* v___x_6764_; size_t v_sz_6765_; size_t v___x_6766_; lean_object* v___x_6767_; 
lean_del_object(v___x_6756_);
v_a_6762_ = lean_ctor_get(v_a_6754_, 0);
lean_inc(v_a_6762_);
lean_dec_ref_known(v_a_6754_, 1);
v___x_6763_ = lean_box(0);
v___x_6764_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6764_, 0, v___x_6763_);
lean_ctor_set(v___x_6764_, 1, v_a_6762_);
v_sz_6765_ = lean_array_size(v_tail_6752_);
v___x_6766_ = ((size_t)0ULL);
v___x_6767_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0_spec__1(v_goal_6743_, v_tail_6752_, v_sz_6765_, v___x_6766_, v___x_6764_, v___y_6746_, v___y_6747_, v___y_6748_, v___y_6749_);
if (lean_obj_tag(v___x_6767_) == 0)
{
lean_object* v_a_6768_; lean_object* v___x_6770_; uint8_t v_isShared_6771_; uint8_t v_isSharedCheck_6781_; 
v_a_6768_ = lean_ctor_get(v___x_6767_, 0);
v_isSharedCheck_6781_ = !lean_is_exclusive(v___x_6767_);
if (v_isSharedCheck_6781_ == 0)
{
v___x_6770_ = v___x_6767_;
v_isShared_6771_ = v_isSharedCheck_6781_;
goto v_resetjp_6769_;
}
else
{
lean_inc(v_a_6768_);
lean_dec(v___x_6767_);
v___x_6770_ = lean_box(0);
v_isShared_6771_ = v_isSharedCheck_6781_;
goto v_resetjp_6769_;
}
v_resetjp_6769_:
{
lean_object* v_fst_6772_; 
v_fst_6772_ = lean_ctor_get(v_a_6768_, 0);
if (lean_obj_tag(v_fst_6772_) == 0)
{
lean_object* v_snd_6773_; lean_object* v___x_6775_; 
v_snd_6773_ = lean_ctor_get(v_a_6768_, 1);
lean_inc(v_snd_6773_);
lean_dec(v_a_6768_);
if (v_isShared_6771_ == 0)
{
lean_ctor_set(v___x_6770_, 0, v_snd_6773_);
v___x_6775_ = v___x_6770_;
goto v_reusejp_6774_;
}
else
{
lean_object* v_reuseFailAlloc_6776_; 
v_reuseFailAlloc_6776_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6776_, 0, v_snd_6773_);
v___x_6775_ = v_reuseFailAlloc_6776_;
goto v_reusejp_6774_;
}
v_reusejp_6774_:
{
return v___x_6775_;
}
}
else
{
lean_object* v_val_6777_; lean_object* v___x_6779_; 
lean_inc_ref(v_fst_6772_);
lean_dec(v_a_6768_);
v_val_6777_ = lean_ctor_get(v_fst_6772_, 0);
lean_inc(v_val_6777_);
lean_dec_ref_known(v_fst_6772_, 1);
if (v_isShared_6771_ == 0)
{
lean_ctor_set(v___x_6770_, 0, v_val_6777_);
v___x_6779_ = v___x_6770_;
goto v_reusejp_6778_;
}
else
{
lean_object* v_reuseFailAlloc_6780_; 
v_reuseFailAlloc_6780_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6780_, 0, v_val_6777_);
v___x_6779_ = v_reuseFailAlloc_6780_;
goto v_reusejp_6778_;
}
v_reusejp_6778_:
{
return v___x_6779_;
}
}
}
}
else
{
lean_object* v_a_6782_; lean_object* v___x_6784_; uint8_t v_isShared_6785_; uint8_t v_isSharedCheck_6789_; 
v_a_6782_ = lean_ctor_get(v___x_6767_, 0);
v_isSharedCheck_6789_ = !lean_is_exclusive(v___x_6767_);
if (v_isSharedCheck_6789_ == 0)
{
v___x_6784_ = v___x_6767_;
v_isShared_6785_ = v_isSharedCheck_6789_;
goto v_resetjp_6783_;
}
else
{
lean_inc(v_a_6782_);
lean_dec(v___x_6767_);
v___x_6784_ = lean_box(0);
v_isShared_6785_ = v_isSharedCheck_6789_;
goto v_resetjp_6783_;
}
v_resetjp_6783_:
{
lean_object* v___x_6787_; 
if (v_isShared_6785_ == 0)
{
v___x_6787_ = v___x_6784_;
goto v_reusejp_6786_;
}
else
{
lean_object* v_reuseFailAlloc_6788_; 
v_reuseFailAlloc_6788_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6788_, 0, v_a_6782_);
v___x_6787_ = v_reuseFailAlloc_6788_;
goto v_reusejp_6786_;
}
v_reusejp_6786_:
{
return v___x_6787_;
}
}
}
}
}
}
else
{
lean_object* v_a_6791_; lean_object* v___x_6793_; uint8_t v_isShared_6794_; uint8_t v_isSharedCheck_6798_; 
lean_dec(v_goal_6743_);
v_a_6791_ = lean_ctor_get(v___x_6753_, 0);
v_isSharedCheck_6798_ = !lean_is_exclusive(v___x_6753_);
if (v_isSharedCheck_6798_ == 0)
{
v___x_6793_ = v___x_6753_;
v_isShared_6794_ = v_isSharedCheck_6798_;
goto v_resetjp_6792_;
}
else
{
lean_inc(v_a_6791_);
lean_dec(v___x_6753_);
v___x_6793_ = lean_box(0);
v_isShared_6794_ = v_isSharedCheck_6798_;
goto v_resetjp_6792_;
}
v_resetjp_6792_:
{
lean_object* v___x_6796_; 
if (v_isShared_6794_ == 0)
{
v___x_6796_ = v___x_6793_;
goto v_reusejp_6795_;
}
else
{
lean_object* v_reuseFailAlloc_6797_; 
v_reuseFailAlloc_6797_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6797_, 0, v_a_6791_);
v___x_6796_ = v_reuseFailAlloc_6797_;
goto v_reusejp_6795_;
}
v_reusejp_6795_:
{
return v___x_6796_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0___boxed(lean_object* v_goal_6799_, lean_object* v_t_6800_, lean_object* v_init_6801_, lean_object* v___y_6802_, lean_object* v___y_6803_, lean_object* v___y_6804_, lean_object* v___y_6805_, lean_object* v___y_6806_){
_start:
{
lean_object* v_res_6807_; 
v_res_6807_ = lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0(v_goal_6799_, v_t_6800_, v_init_6801_, v___y_6802_, v___y_6803_, v___y_6804_, v___y_6805_);
lean_dec(v___y_6805_);
lean_dec_ref(v___y_6804_);
lean_dec(v___y_6803_);
lean_dec_ref(v___y_6802_);
lean_dec_ref(v_t_6800_);
return v_res_6807_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_splitFirstHypothesisS_x3f___lam__1(lean_object* v_goal_6811_, lean_object* v___y_6812_, lean_object* v___y_6813_, lean_object* v___y_6814_, lean_object* v___y_6815_){
_start:
{
lean_object* v___x_6817_; 
lean_inc(v_goal_6811_);
v___x_6817_ = l_Lean_MVarId_getDecl(v_goal_6811_, v___y_6812_, v___y_6813_, v___y_6814_, v___y_6815_);
if (lean_obj_tag(v___x_6817_) == 0)
{
lean_object* v_a_6818_; lean_object* v_lctx_6819_; lean_object* v_decls_6820_; lean_object* v___x_6821_; lean_object* v___x_6822_; lean_object* v___x_6823_; 
v_a_6818_ = lean_ctor_get(v___x_6817_, 0);
lean_inc(v_a_6818_);
lean_dec_ref_known(v___x_6817_, 1);
v_lctx_6819_ = lean_ctor_get(v_a_6818_, 1);
lean_inc_ref(v_lctx_6819_);
lean_dec(v_a_6818_);
v_decls_6820_ = lean_ctor_get(v_lctx_6819_, 1);
lean_inc_ref(v_decls_6820_);
lean_dec_ref(v_lctx_6819_);
v___x_6821_ = lean_box(0);
v___x_6822_ = ((lean_object*)(lp_aesop_Aesop_splitFirstHypothesisS_x3f___lam__1___closed__0));
v___x_6823_ = lp_aesop_Lean_PersistentArray_forIn___at___00Aesop_splitFirstHypothesisS_x3f_spec__0(v_goal_6811_, v_decls_6820_, v___x_6822_, v___y_6812_, v___y_6813_, v___y_6814_, v___y_6815_);
lean_dec_ref(v_decls_6820_);
if (lean_obj_tag(v___x_6823_) == 0)
{
lean_object* v_a_6824_; lean_object* v___x_6826_; uint8_t v_isShared_6827_; uint8_t v_isSharedCheck_6836_; 
v_a_6824_ = lean_ctor_get(v___x_6823_, 0);
v_isSharedCheck_6836_ = !lean_is_exclusive(v___x_6823_);
if (v_isSharedCheck_6836_ == 0)
{
v___x_6826_ = v___x_6823_;
v_isShared_6827_ = v_isSharedCheck_6836_;
goto v_resetjp_6825_;
}
else
{
lean_inc(v_a_6824_);
lean_dec(v___x_6823_);
v___x_6826_ = lean_box(0);
v_isShared_6827_ = v_isSharedCheck_6836_;
goto v_resetjp_6825_;
}
v_resetjp_6825_:
{
lean_object* v_fst_6828_; 
v_fst_6828_ = lean_ctor_get(v_a_6824_, 0);
lean_inc(v_fst_6828_);
lean_dec(v_a_6824_);
if (lean_obj_tag(v_fst_6828_) == 0)
{
lean_object* v___x_6830_; 
if (v_isShared_6827_ == 0)
{
lean_ctor_set(v___x_6826_, 0, v___x_6821_);
v___x_6830_ = v___x_6826_;
goto v_reusejp_6829_;
}
else
{
lean_object* v_reuseFailAlloc_6831_; 
v_reuseFailAlloc_6831_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6831_, 0, v___x_6821_);
v___x_6830_ = v_reuseFailAlloc_6831_;
goto v_reusejp_6829_;
}
v_reusejp_6829_:
{
return v___x_6830_;
}
}
else
{
lean_object* v_val_6832_; lean_object* v___x_6834_; 
v_val_6832_ = lean_ctor_get(v_fst_6828_, 0);
lean_inc(v_val_6832_);
lean_dec_ref_known(v_fst_6828_, 1);
if (v_isShared_6827_ == 0)
{
lean_ctor_set(v___x_6826_, 0, v_val_6832_);
v___x_6834_ = v___x_6826_;
goto v_reusejp_6833_;
}
else
{
lean_object* v_reuseFailAlloc_6835_; 
v_reuseFailAlloc_6835_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6835_, 0, v_val_6832_);
v___x_6834_ = v_reuseFailAlloc_6835_;
goto v_reusejp_6833_;
}
v_reusejp_6833_:
{
return v___x_6834_;
}
}
}
}
else
{
lean_object* v_a_6837_; lean_object* v___x_6839_; uint8_t v_isShared_6840_; uint8_t v_isSharedCheck_6844_; 
v_a_6837_ = lean_ctor_get(v___x_6823_, 0);
v_isSharedCheck_6844_ = !lean_is_exclusive(v___x_6823_);
if (v_isSharedCheck_6844_ == 0)
{
v___x_6839_ = v___x_6823_;
v_isShared_6840_ = v_isSharedCheck_6844_;
goto v_resetjp_6838_;
}
else
{
lean_inc(v_a_6837_);
lean_dec(v___x_6823_);
v___x_6839_ = lean_box(0);
v_isShared_6840_ = v_isSharedCheck_6844_;
goto v_resetjp_6838_;
}
v_resetjp_6838_:
{
lean_object* v___x_6842_; 
if (v_isShared_6840_ == 0)
{
v___x_6842_ = v___x_6839_;
goto v_reusejp_6841_;
}
else
{
lean_object* v_reuseFailAlloc_6843_; 
v_reuseFailAlloc_6843_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6843_, 0, v_a_6837_);
v___x_6842_ = v_reuseFailAlloc_6843_;
goto v_reusejp_6841_;
}
v_reusejp_6841_:
{
return v___x_6842_;
}
}
}
}
else
{
lean_object* v_a_6845_; lean_object* v___x_6847_; uint8_t v_isShared_6848_; uint8_t v_isSharedCheck_6852_; 
lean_dec(v_goal_6811_);
v_a_6845_ = lean_ctor_get(v___x_6817_, 0);
v_isSharedCheck_6852_ = !lean_is_exclusive(v___x_6817_);
if (v_isSharedCheck_6852_ == 0)
{
v___x_6847_ = v___x_6817_;
v_isShared_6848_ = v_isSharedCheck_6852_;
goto v_resetjp_6846_;
}
else
{
lean_inc(v_a_6845_);
lean_dec(v___x_6817_);
v___x_6847_ = lean_box(0);
v_isShared_6848_ = v_isSharedCheck_6852_;
goto v_resetjp_6846_;
}
v_resetjp_6846_:
{
lean_object* v___x_6850_; 
if (v_isShared_6848_ == 0)
{
v___x_6850_ = v___x_6847_;
goto v_reusejp_6849_;
}
else
{
lean_object* v_reuseFailAlloc_6851_; 
v_reuseFailAlloc_6851_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6851_, 0, v_a_6845_);
v___x_6850_ = v_reuseFailAlloc_6851_;
goto v_reusejp_6849_;
}
v_reusejp_6849_:
{
return v___x_6850_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_splitFirstHypothesisS_x3f___lam__1___boxed(lean_object* v_goal_6853_, lean_object* v___y_6854_, lean_object* v___y_6855_, lean_object* v___y_6856_, lean_object* v___y_6857_, lean_object* v___y_6858_){
_start:
{
lean_object* v_res_6859_; 
v_res_6859_ = lp_aesop_Aesop_splitFirstHypothesisS_x3f___lam__1(v_goal_6853_, v___y_6854_, v___y_6855_, v___y_6856_, v___y_6857_);
lean_dec(v___y_6857_);
lean_dec_ref(v___y_6856_);
lean_dec(v___y_6855_);
lean_dec_ref(v___y_6854_);
return v_res_6859_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_splitFirstHypothesisS_x3f(lean_object* v_goal_6861_, lean_object* v_a_6862_, lean_object* v_a_6863_, lean_object* v_a_6864_, lean_object* v_a_6865_, lean_object* v_a_6866_, lean_object* v_a_6867_){
_start:
{
lean_object* v___f_6869_; lean_object* v___f_6870_; lean_object* v___x_6871_; lean_object* v___x_6872_; 
v___f_6869_ = ((lean_object*)(lp_aesop_Aesop_splitFirstHypothesisS_x3f___closed__0));
lean_inc_n(v_goal_6861_, 2);
v___f_6870_ = lean_alloc_closure((void*)(lp_aesop_Aesop_splitFirstHypothesisS_x3f___lam__1___boxed), 6, 1);
lean_closure_set(v___f_6870_, 0, v_goal_6861_);
v___x_6871_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_splitFirstHypothesisS_x3f_tacticBuilder___boxed), 7, 1);
lean_closure_set(v___x_6871_, 0, v_goal_6861_);
v___x_6872_ = lp_aesop_Aesop_withOptScriptStep___redArg(v_goal_6861_, v___f_6869_, v___x_6871_, v___f_6870_, v_a_6862_, v_a_6864_, v_a_6865_, v_a_6866_, v_a_6867_);
if (lean_obj_tag(v___x_6872_) == 0)
{
lean_object* v_a_6873_; lean_object* v___x_6875_; uint8_t v_isShared_6876_; uint8_t v_isSharedCheck_6907_; 
v_a_6873_ = lean_ctor_get(v___x_6872_, 0);
v_isSharedCheck_6907_ = !lean_is_exclusive(v___x_6872_);
if (v_isSharedCheck_6907_ == 0)
{
v___x_6875_ = v___x_6872_;
v_isShared_6876_ = v_isSharedCheck_6907_;
goto v_resetjp_6874_;
}
else
{
lean_inc(v_a_6873_);
lean_dec(v___x_6872_);
v___x_6875_ = lean_box(0);
v_isShared_6876_ = v_isSharedCheck_6907_;
goto v_resetjp_6874_;
}
v_resetjp_6874_:
{
if (lean_obj_tag(v_a_6873_) == 1)
{
lean_object* v_val_6877_; lean_object* v___x_6879_; uint8_t v_isShared_6880_; uint8_t v_isSharedCheck_6902_; 
lean_del_object(v___x_6875_);
v_val_6877_ = lean_ctor_get(v_a_6873_, 0);
v_isSharedCheck_6902_ = !lean_is_exclusive(v_a_6873_);
if (v_isSharedCheck_6902_ == 0)
{
v___x_6879_ = v_a_6873_;
v_isShared_6880_ = v_isSharedCheck_6902_;
goto v_resetjp_6878_;
}
else
{
lean_inc(v_val_6877_);
lean_dec(v_a_6873_);
v___x_6879_ = lean_box(0);
v_isShared_6880_ = v_isSharedCheck_6902_;
goto v_resetjp_6878_;
}
v_resetjp_6878_:
{
lean_object* v_fst_6881_; lean_object* v___x_6882_; 
v_fst_6881_ = lean_ctor_get(v_val_6877_, 0);
lean_inc(v_fst_6881_);
lean_dec(v_val_6877_);
v___x_6882_ = lp_aesop___private_Aesop_Script_SpecificTactics_0__Aesop_renameInaccessibleFVarsS_x27(v_fst_6881_, v_a_6862_, v_a_6863_, v_a_6864_, v_a_6865_, v_a_6866_, v_a_6867_);
if (lean_obj_tag(v___x_6882_) == 0)
{
lean_object* v_a_6883_; lean_object* v___x_6885_; uint8_t v_isShared_6886_; uint8_t v_isSharedCheck_6893_; 
v_a_6883_ = lean_ctor_get(v___x_6882_, 0);
v_isSharedCheck_6893_ = !lean_is_exclusive(v___x_6882_);
if (v_isSharedCheck_6893_ == 0)
{
v___x_6885_ = v___x_6882_;
v_isShared_6886_ = v_isSharedCheck_6893_;
goto v_resetjp_6884_;
}
else
{
lean_inc(v_a_6883_);
lean_dec(v___x_6882_);
v___x_6885_ = lean_box(0);
v_isShared_6886_ = v_isSharedCheck_6893_;
goto v_resetjp_6884_;
}
v_resetjp_6884_:
{
lean_object* v___x_6888_; 
if (v_isShared_6880_ == 0)
{
lean_ctor_set(v___x_6879_, 0, v_a_6883_);
v___x_6888_ = v___x_6879_;
goto v_reusejp_6887_;
}
else
{
lean_object* v_reuseFailAlloc_6892_; 
v_reuseFailAlloc_6892_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6892_, 0, v_a_6883_);
v___x_6888_ = v_reuseFailAlloc_6892_;
goto v_reusejp_6887_;
}
v_reusejp_6887_:
{
lean_object* v___x_6890_; 
if (v_isShared_6886_ == 0)
{
lean_ctor_set(v___x_6885_, 0, v___x_6888_);
v___x_6890_ = v___x_6885_;
goto v_reusejp_6889_;
}
else
{
lean_object* v_reuseFailAlloc_6891_; 
v_reuseFailAlloc_6891_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6891_, 0, v___x_6888_);
v___x_6890_ = v_reuseFailAlloc_6891_;
goto v_reusejp_6889_;
}
v_reusejp_6889_:
{
return v___x_6890_;
}
}
}
}
else
{
lean_object* v_a_6894_; lean_object* v___x_6896_; uint8_t v_isShared_6897_; uint8_t v_isSharedCheck_6901_; 
lean_del_object(v___x_6879_);
v_a_6894_ = lean_ctor_get(v___x_6882_, 0);
v_isSharedCheck_6901_ = !lean_is_exclusive(v___x_6882_);
if (v_isSharedCheck_6901_ == 0)
{
v___x_6896_ = v___x_6882_;
v_isShared_6897_ = v_isSharedCheck_6901_;
goto v_resetjp_6895_;
}
else
{
lean_inc(v_a_6894_);
lean_dec(v___x_6882_);
v___x_6896_ = lean_box(0);
v_isShared_6897_ = v_isSharedCheck_6901_;
goto v_resetjp_6895_;
}
v_resetjp_6895_:
{
lean_object* v___x_6899_; 
if (v_isShared_6897_ == 0)
{
v___x_6899_ = v___x_6896_;
goto v_reusejp_6898_;
}
else
{
lean_object* v_reuseFailAlloc_6900_; 
v_reuseFailAlloc_6900_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6900_, 0, v_a_6894_);
v___x_6899_ = v_reuseFailAlloc_6900_;
goto v_reusejp_6898_;
}
v_reusejp_6898_:
{
return v___x_6899_;
}
}
}
}
}
else
{
lean_object* v___x_6903_; lean_object* v___x_6905_; 
lean_dec(v_a_6873_);
v___x_6903_ = lean_box(0);
if (v_isShared_6876_ == 0)
{
lean_ctor_set(v___x_6875_, 0, v___x_6903_);
v___x_6905_ = v___x_6875_;
goto v_reusejp_6904_;
}
else
{
lean_object* v_reuseFailAlloc_6906_; 
v_reuseFailAlloc_6906_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6906_, 0, v___x_6903_);
v___x_6905_ = v_reuseFailAlloc_6906_;
goto v_reusejp_6904_;
}
v_reusejp_6904_:
{
return v___x_6905_;
}
}
}
}
else
{
lean_object* v_a_6908_; lean_object* v___x_6910_; uint8_t v_isShared_6911_; uint8_t v_isSharedCheck_6915_; 
v_a_6908_ = lean_ctor_get(v___x_6872_, 0);
v_isSharedCheck_6915_ = !lean_is_exclusive(v___x_6872_);
if (v_isSharedCheck_6915_ == 0)
{
v___x_6910_ = v___x_6872_;
v_isShared_6911_ = v_isSharedCheck_6915_;
goto v_resetjp_6909_;
}
else
{
lean_inc(v_a_6908_);
lean_dec(v___x_6872_);
v___x_6910_ = lean_box(0);
v_isShared_6911_ = v_isSharedCheck_6915_;
goto v_resetjp_6909_;
}
v_resetjp_6909_:
{
lean_object* v___x_6913_; 
if (v_isShared_6911_ == 0)
{
v___x_6913_ = v___x_6910_;
goto v_reusejp_6912_;
}
else
{
lean_object* v_reuseFailAlloc_6914_; 
v_reuseFailAlloc_6914_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6914_, 0, v_a_6908_);
v___x_6913_ = v_reuseFailAlloc_6914_;
goto v_reusejp_6912_;
}
v_reusejp_6912_:
{
return v___x_6913_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_splitFirstHypothesisS_x3f___boxed(lean_object* v_goal_6916_, lean_object* v_a_6917_, lean_object* v_a_6918_, lean_object* v_a_6919_, lean_object* v_a_6920_, lean_object* v_a_6921_, lean_object* v_a_6922_, lean_object* v_a_6923_){
_start:
{
lean_object* v_res_6924_; 
v_res_6924_ = lp_aesop_Aesop_splitFirstHypothesisS_x3f(v_goal_6916_, v_a_6917_, v_a_6918_, v_a_6919_, v_a_6920_, v_a_6921_, v_a_6922_);
lean_dec(v_a_6922_);
lean_dec_ref(v_a_6921_);
lean_dec(v_a_6920_);
lean_dec_ref(v_a_6919_);
lean_dec(v_a_6918_);
lean_dec(v_a_6917_);
return v_res_6924_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Cases(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Simp_Types(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Util_Tactic_Ext(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Script_CtorNames(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Script_ScriptM(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_Inaccessible(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Util_Tactic(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Util_Tactic_Unfold(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Util_Unfold(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_Basic(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_RCases(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Simp(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Split(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Script_SpecificTactics(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Cases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Simp_Types(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Util_Tactic_Ext(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_CtorNames(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_ScriptM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_Inaccessible(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Util_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Util_Tactic_Unfold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Util_Unfold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_RCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Split(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_Script_Tactic_skip = _init_lp_aesop_Aesop_Script_Tactic_skip();
lean_mark_persistent(lp_aesop_Aesop_Script_Tactic_skip);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Script_SpecificTactics(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Cases(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Simp_Types(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Util_Tactic_Ext(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Script_CtorNames(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Script_ScriptM(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_Meta_Inaccessible(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Util_Tactic(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Util_Tactic_Unfold(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Util_Unfold(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_Meta_Basic(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_RCases(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Simp(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Split(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Script_SpecificTactics(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Cases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Simp_Types(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Util_Tactic_Ext(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_CtorNames(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_ScriptM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Meta_Inaccessible(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Util_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Util_Tactic_Unfold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Util_Unfold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_RCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Split(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_SpecificTactics(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Script_SpecificTactics(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Script_SpecificTactics(builtin);
}
#ifdef __cplusplus
}
#endif
