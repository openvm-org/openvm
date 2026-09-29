// Lean compiler output
// Module: Qq.Commands
// Imports: public import Init public meta import Init public import Qq.Macro public meta import Lean.Meta.Eval
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
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Meta_evalExpr___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_abortTermExceptionId;
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l_Lean_instInhabitedPersistentArrayNode_default(lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_left(size_t, size_t);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
size_t lean_usize_add(size_t, size_t);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* l_Lean_LocalDecl_index(lean_object*);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
lean_object* l_Lean_LocalDecl_userName(lean_object*);
lean_object* l_Lean_LocalDecl_type(lean_object*);
uint8_t l_Lean_LocalDecl_kind(lean_object*);
lean_object* l_Lean_LocalContext_addDecl(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTermEnsuringType(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_synthesizeSyntheticMVarsNoPostponing(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l_Lean_Meta_getMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_logUnassignedUsingErrorInfos(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_LocalContext_empty;
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLetFVars___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_getLevelNames___redArg(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lp_Qq_Qq_Impl_quoteLCtx(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* lp_Qq_Qq_Impl_quoteExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_withExpectedType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_addTermInfo_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_LocalContext_mkLocalDecl(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t);
lean_object* l_Lean_Expr_mvar___override(lean_object*);
lean_object* lp_Qq_toExprExpr(lean_object*);
lean_object* l_Lean_Elab_Tactic_SavedState_restore___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Elab_Tactic_saveState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__8(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__9(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__7_spec__9(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__7_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__9___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__7___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__7___closed__0;
LEAN_EXPORT lean_object* lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__7(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__2_spec__4(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_Qq___private_Qq_Commands_0__Qq_mkLetFVarsFromValues___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_Qq___private_Qq_Commands_0__Qq_mkLetFVarsFromValues___closed__0 = (const lean_object*)&lp_Qq___private_Qq_Commands_0__Qq_mkLetFVarsFromValues___closed__0_value;
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Commands_0__Qq_mkLetFVarsFromValues(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Commands_0__Qq_mkLetFVarsFromValues___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Qq_termBy__elabq___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Qq"};
static const lean_object* lp_Qq_Qq_termBy__elabq___00__closed__0 = (const lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__0_value;
static const lean_string_object lp_Qq_Qq_termBy__elabq___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "termBy_elabq_"};
static const lean_object* lp_Qq_Qq_termBy__elabq___00__closed__1 = (const lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__1_value;
static const lean_ctor_object lp_Qq_Qq_termBy__elabq___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(172, 246, 49, 188, 140, 116, 47, 174)}};
static const lean_ctor_object lp_Qq_Qq_termBy__elabq___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__2_value_aux_0),((lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(183, 86, 140, 161, 155, 24, 174, 233)}};
static const lean_object* lp_Qq_Qq_termBy__elabq___00__closed__2 = (const lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__2_value;
static const lean_string_object lp_Qq_Qq_termBy__elabq___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_Qq_Qq_termBy__elabq___00__closed__3 = (const lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__3_value;
static const lean_ctor_object lp_Qq_Qq_termBy__elabq___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_Qq_Qq_termBy__elabq___00__closed__4 = (const lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__4_value;
static const lean_string_object lp_Qq_Qq_termBy__elabq___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "by_elabq"};
static const lean_object* lp_Qq_Qq_termBy__elabq___00__closed__5 = (const lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__5_value;
static const lean_ctor_object lp_Qq_Qq_termBy__elabq___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__5_value)}};
static const lean_object* lp_Qq_Qq_termBy__elabq___00__closed__6 = (const lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__6_value;
static const lean_string_object lp_Qq_Qq_termBy__elabq___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "doSeq"};
static const lean_object* lp_Qq_Qq_termBy__elabq___00__closed__7 = (const lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__7_value;
static const lean_ctor_object lp_Qq_Qq_termBy__elabq___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(87, 108, 208, 147, 238, 58, 86, 179)}};
static const lean_object* lp_Qq_Qq_termBy__elabq___00__closed__8 = (const lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__8_value;
static const lean_ctor_object lp_Qq_Qq_termBy__elabq___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__8_value)}};
static const lean_object* lp_Qq_Qq_termBy__elabq___00__closed__9 = (const lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__9_value;
static const lean_ctor_object lp_Qq_Qq_termBy__elabq___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__4_value),((lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__6_value),((lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__9_value)}};
static const lean_object* lp_Qq_Qq_termBy__elabq___00__closed__10 = (const lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__10_value;
static const lean_ctor_object lp_Qq_Qq_termBy__elabq___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__2_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__10_value)}};
static const lean_object* lp_Qq_Qq_termBy__elabq___00__closed__11 = (const lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_Qq_Qq_termBy__elabq__ = (const lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__11_value;
static const lean_string_object lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__0 = (const lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__0_value;
static const lean_string_object lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__1 = (const lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__1_value;
static const lean_string_object lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__2 = (const lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__2_value;
static const lean_string_object lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "TermElabM"};
static const lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__3 = (const lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__3_value;
static const lean_ctor_object lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__4_value_aux_0),((lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__1_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__4_value_aux_1),((lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__2_value),LEAN_SCALAR_PTR_LITERAL(252, 225, 247, 249, 114, 131, 135, 109)}};
static const lean_ctor_object lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__4_value_aux_2),((lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__3_value),LEAN_SCALAR_PTR_LITERAL(85, 85, 78, 208, 80, 136, 131, 165)}};
static const lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__4 = (const lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__4_value;
static lean_once_cell_t lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__5;
static const lean_string_object lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Expr"};
static const lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__6 = (const lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__6_value;
static const lean_ctor_object lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__7_value_aux_0),((lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__6_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__7 = (const lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__7_value;
static lean_once_cell_t lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__8;
static lean_once_cell_t lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__9;
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__3___redArg___closed__0;
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__3___redArg();
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__3___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__4___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "do"};
static const lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__1___closed__0 = (const lean_object*)&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__1(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__0;
static lean_once_cell_t lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__1;
static const lean_array_object lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__2 = (const lean_object*)&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__2_value;
static lean_once_cell_t lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__3;
static const lean_string_object lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__4 = (const lean_object*)&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__4_value;
static lean_once_cell_t lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__5;
static lean_once_cell_t lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__6;
static lean_once_cell_t lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__7;
static const lean_string_object lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Quoted"};
static const lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__8 = (const lean_object*)&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__8_value;
static const lean_ctor_object lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(172, 246, 49, 188, 140, 116, 47, 174)}};
static const lean_ctor_object lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__9_value_aux_0),((lean_object*)&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__8_value),LEAN_SCALAR_PTR_LITERAL(115, 104, 38, 134, 26, 50, 120, 141)}};
static const lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__9 = (const lean_object*)&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__9_value;
static lean_once_cell_t lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__10;
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticRun_tacq_=>_"};
static const lean_object* lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__0 = (const lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__0_value;
static const lean_ctor_object lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(172, 246, 49, 188, 140, 116, 47, 174)}};
static const lean_ctor_object lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__1_value_aux_0),((lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(215, 234, 91, 8, 158, 244, 245, 38)}};
static const lean_object* lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__1 = (const lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__1_value;
static const lean_string_object lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "run_tacq"};
static const lean_object* lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__2 = (const lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__2_value;
static const lean_ctor_object lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__3 = (const lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__3_value;
static const lean_string_object lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__4 = (const lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__4_value;
static const lean_ctor_object lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__5 = (const lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__5_value;
static const lean_string_object lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "atomic"};
static const lean_object* lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__6 = (const lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__6_value;
static const lean_ctor_object lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(56, 145, 113, 208, 127, 167, 216, 55)}};
static const lean_object* lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__7 = (const lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__7_value;
static const lean_string_object lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__8 = (const lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__8_value;
static const lean_ctor_object lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__8_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__9 = (const lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__9_value;
static const lean_ctor_object lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__9_value)}};
static const lean_object* lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__10 = (const lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__10_value;
static const lean_string_object lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__11 = (const lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__11_value;
static const lean_ctor_object lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__11_value)}};
static const lean_object* lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__12 = (const lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__12_value;
static const lean_ctor_object lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__4_value),((lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__10_value),((lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__12_value)}};
static const lean_object* lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__13 = (const lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__13_value;
static const lean_ctor_object lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__7_value),((lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__13_value)}};
static const lean_object* lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__14 = (const lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__14_value;
static const lean_ctor_object lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__5_value),((lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__14_value)}};
static const lean_object* lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__15 = (const lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__15_value;
static const lean_ctor_object lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__4_value),((lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__3_value),((lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__15_value)}};
static const lean_object* lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__16 = (const lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__16_value;
static const lean_ctor_object lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__4_value),((lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__16_value),((lean_object*)&lp_Qq_Qq_termBy__elabq___00__closed__9_value)}};
static const lean_object* lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__17 = (const lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__17_value;
static const lean_ctor_object lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__17_value)}};
static const lean_object* lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__18 = (const lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__18_value;
LEAN_EXPORT const lean_object* lp_Qq_Qq_tacticRun__tacq___x3d_x3e__ = (const lean_object*)&lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__18_value;
static const lean_string_object lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__0 = (const lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__0_value;
static const lean_string_object lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "TacticM"};
static const lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__1 = (const lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__1_value;
static const lean_ctor_object lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__2_value_aux_0),((lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__1_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__2_value_aux_1),((lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__0_value),LEAN_SCALAR_PTR_LITERAL(161, 230, 229, 85, 182, 144, 182, 176)}};
static const lean_ctor_object lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__2_value_aux_2),((lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__1_value),LEAN_SCALAR_PTR_LITERAL(143, 63, 151, 54, 27, 84, 190, 214)}};
static const lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__2 = (const lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__2_value;
static lean_once_cell_t lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__3;
static const lean_string_object lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Unit"};
static const lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__4 = (const lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__4_value;
static const lean_ctor_object lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__4_value),LEAN_SCALAR_PTR_LITERAL(230, 84, 106, 234, 91, 210, 120, 136)}};
static const lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__5 = (const lean_object*)&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__5_value;
static lean_once_cell_t lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__6;
static lean_once_cell_t lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__7;
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__2___redArg();
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__2___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_MVarId_withContext___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_MVarId_withContext___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_MVarId_withContext___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_MVarId_withContext___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__0 = (const lean_object*)&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__0_value;
static const lean_string_object lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "discard"};
static const lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__1 = (const lean_object*)&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__1_value;
static lean_once_cell_t lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__2;
static const lean_ctor_object lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(64, 238, 40, 96, 153, 56, 105, 203)}};
static const lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__3 = (const lean_object*)&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__3_value;
static const lean_string_object lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Functor"};
static const lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__4 = (const lean_object*)&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__4_value;
static const lean_ctor_object lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(39, 234, 35, 88, 204, 30, 230, 30)}};
static const lean_ctor_object lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__5_value_aux_0),((lean_object*)&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(12, 185, 213, 31, 185, 7, 183, 6)}};
static const lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__5 = (const lean_object*)&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__5_value;
static const lean_ctor_object lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__6 = (const lean_object*)&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__6_value;
static const lean_ctor_object lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__6_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__7 = (const lean_object*)&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__7_value;
static const lean_string_object lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__8 = (const lean_object*)&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__8_value;
static const lean_ctor_object lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__9 = (const lean_object*)&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__9_value;
static lean_once_cell_t lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__10;
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__1(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_mkFreshId___at___00Lean_mkFreshFVarId___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__4_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_mkFreshId___at___00Lean_mkFreshFVarId___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__4_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_mkFreshFVarId___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_mkFreshFVarId___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__2___closed__0;
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__2___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__6_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__6_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "no open goal, run_tacq requires main goal"};
static const lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___closed__0 = (const lean_object*)&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___closed__0_value;
static lean_once_cell_t lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___closed__1;
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_mkFreshId___at___00Lean_mkFreshFVarId___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__4_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_mkFreshId___at___00Lean_mkFreshFVarId___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__4_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__2___redArg(lean_object* v_lctx_1_, lean_object* v_localInsts_2_, lean_object* v_x_3_, lean_object* v___y_4_, lean_object* v___y_5_, lean_object* v___y_6_, lean_object* v___y_7_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_box(0), v_lctx_1_, v_localInsts_2_, v_x_3_, v___y_4_, v___y_5_, v___y_6_, v___y_7_);
if (lean_obj_tag(v___x_9_) == 0)
{
lean_object* v_a_10_; lean_object* v___x_12_; uint8_t v_isShared_13_; uint8_t v_isSharedCheck_17_; 
v_a_10_ = lean_ctor_get(v___x_9_, 0);
v_isSharedCheck_17_ = !lean_is_exclusive(v___x_9_);
if (v_isSharedCheck_17_ == 0)
{
v___x_12_ = v___x_9_;
v_isShared_13_ = v_isSharedCheck_17_;
goto v_resetjp_11_;
}
else
{
lean_inc(v_a_10_);
lean_dec(v___x_9_);
v___x_12_ = lean_box(0);
v_isShared_13_ = v_isSharedCheck_17_;
goto v_resetjp_11_;
}
v_resetjp_11_:
{
lean_object* v___x_15_; 
if (v_isShared_13_ == 0)
{
v___x_15_ = v___x_12_;
goto v_reusejp_14_;
}
else
{
lean_object* v_reuseFailAlloc_16_; 
v_reuseFailAlloc_16_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_16_, 0, v_a_10_);
v___x_15_ = v_reuseFailAlloc_16_;
goto v_reusejp_14_;
}
v_reusejp_14_:
{
return v___x_15_;
}
}
}
else
{
lean_object* v_a_18_; lean_object* v___x_20_; uint8_t v_isShared_21_; uint8_t v_isSharedCheck_25_; 
v_a_18_ = lean_ctor_get(v___x_9_, 0);
v_isSharedCheck_25_ = !lean_is_exclusive(v___x_9_);
if (v_isSharedCheck_25_ == 0)
{
v___x_20_ = v___x_9_;
v_isShared_21_ = v_isSharedCheck_25_;
goto v_resetjp_19_;
}
else
{
lean_inc(v_a_18_);
lean_dec(v___x_9_);
v___x_20_ = lean_box(0);
v_isShared_21_ = v_isSharedCheck_25_;
goto v_resetjp_19_;
}
v_resetjp_19_:
{
lean_object* v___x_23_; 
if (v_isShared_21_ == 0)
{
v___x_23_ = v___x_20_;
goto v_reusejp_22_;
}
else
{
lean_object* v_reuseFailAlloc_24_; 
v_reuseFailAlloc_24_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_24_, 0, v_a_18_);
v___x_23_ = v_reuseFailAlloc_24_;
goto v_reusejp_22_;
}
v_reusejp_22_:
{
return v___x_23_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__2___redArg___boxed(lean_object* v_lctx_26_, lean_object* v_localInsts_27_, lean_object* v_x_28_, lean_object* v___y_29_, lean_object* v___y_30_, lean_object* v___y_31_, lean_object* v___y_32_, lean_object* v___y_33_){
_start:
{
lean_object* v_res_34_; 
v_res_34_ = lp_Qq_Lean_Meta_withLCtx___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__2___redArg(v_lctx_26_, v_localInsts_27_, v_x_28_, v___y_29_, v___y_30_, v___y_31_, v___y_32_);
lean_dec(v___y_32_);
lean_dec_ref(v___y_31_);
lean_dec(v___y_30_);
lean_dec_ref(v___y_29_);
return v_res_34_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__2(lean_object* v_00_u03b1_35_, lean_object* v_lctx_36_, lean_object* v_localInsts_37_, lean_object* v_x_38_, lean_object* v___y_39_, lean_object* v___y_40_, lean_object* v___y_41_, lean_object* v___y_42_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lp_Qq_Lean_Meta_withLCtx___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__2___redArg(v_lctx_36_, v_localInsts_37_, v_x_38_, v___y_39_, v___y_40_, v___y_41_, v___y_42_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__2___boxed(lean_object* v_00_u03b1_45_, lean_object* v_lctx_46_, lean_object* v_localInsts_47_, lean_object* v_x_48_, lean_object* v___y_49_, lean_object* v___y_50_, lean_object* v___y_51_, lean_object* v___y_52_, lean_object* v___y_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_Qq_Lean_Meta_withLCtx___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__2(v_00_u03b1_45_, v_lctx_46_, v_localInsts_47_, v_x_48_, v___y_49_, v___y_50_, v___y_51_, v___y_52_);
lean_dec(v___y_52_);
lean_dec_ref(v___y_51_);
lean_dec(v___y_50_);
lean_dec_ref(v___y_49_);
return v_res_54_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__8(lean_object* v_as_55_, size_t v_i_56_, size_t v_stop_57_, lean_object* v_b_58_){
_start:
{
lean_object* v___y_60_; uint8_t v___x_64_; 
v___x_64_ = lean_usize_dec_eq(v_i_56_, v_stop_57_);
if (v___x_64_ == 0)
{
lean_object* v___x_65_; 
v___x_65_ = lean_array_uget_borrowed(v_as_55_, v_i_56_);
if (lean_obj_tag(v___x_65_) == 0)
{
v___y_60_ = v_b_58_;
goto v___jp_59_;
}
else
{
lean_object* v_val_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; 
v_val_66_ = lean_ctor_get(v___x_65_, 0);
v___x_67_ = l_Lean_LocalDecl_fvarId(v_val_66_);
v___x_68_ = l_Lean_Expr_fvar___override(v___x_67_);
v___x_69_ = lean_array_push(v_b_58_, v___x_68_);
v___y_60_ = v___x_69_;
goto v___jp_59_;
}
}
else
{
return v_b_58_;
}
v___jp_59_:
{
size_t v___x_61_; size_t v___x_62_; 
v___x_61_ = ((size_t)1ULL);
v___x_62_ = lean_usize_add(v_i_56_, v___x_61_);
v_i_56_ = v___x_62_;
v_b_58_ = v___y_60_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__8___boxed(lean_object* v_as_70_, lean_object* v_i_71_, lean_object* v_stop_72_, lean_object* v_b_73_){
_start:
{
size_t v_i_boxed_74_; size_t v_stop_boxed_75_; lean_object* v_res_76_; 
v_i_boxed_74_ = lean_unbox_usize(v_i_71_);
lean_dec(v_i_71_);
v_stop_boxed_75_ = lean_unbox_usize(v_stop_72_);
lean_dec(v_stop_72_);
v_res_76_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__8(v_as_70_, v_i_boxed_74_, v_stop_boxed_75_, v_b_73_);
lean_dec_ref(v_as_70_);
return v_res_76_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__9(lean_object* v_x_77_, lean_object* v_x_78_){
_start:
{
if (lean_obj_tag(v_x_77_) == 0)
{
lean_object* v_cs_79_; lean_object* v___x_80_; lean_object* v___x_81_; uint8_t v___x_82_; 
v_cs_79_ = lean_ctor_get(v_x_77_, 0);
v___x_80_ = lean_unsigned_to_nat(0u);
v___x_81_ = lean_array_get_size(v_cs_79_);
v___x_82_ = lean_nat_dec_lt(v___x_80_, v___x_81_);
if (v___x_82_ == 0)
{
return v_x_78_;
}
else
{
uint8_t v___x_83_; 
v___x_83_ = lean_nat_dec_le(v___x_81_, v___x_81_);
if (v___x_83_ == 0)
{
if (v___x_82_ == 0)
{
return v_x_78_;
}
else
{
size_t v___x_84_; size_t v___x_85_; lean_object* v___x_86_; 
v___x_84_ = ((size_t)0ULL);
v___x_85_ = lean_usize_of_nat(v___x_81_);
v___x_86_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__7_spec__9(v_cs_79_, v___x_84_, v___x_85_, v_x_78_);
return v___x_86_;
}
}
else
{
size_t v___x_87_; size_t v___x_88_; lean_object* v___x_89_; 
v___x_87_ = ((size_t)0ULL);
v___x_88_ = lean_usize_of_nat(v___x_81_);
v___x_89_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__7_spec__9(v_cs_79_, v___x_87_, v___x_88_, v_x_78_);
return v___x_89_;
}
}
}
else
{
lean_object* v_vs_90_; lean_object* v___x_91_; lean_object* v___x_92_; uint8_t v___x_93_; 
v_vs_90_ = lean_ctor_get(v_x_77_, 0);
v___x_91_ = lean_unsigned_to_nat(0u);
v___x_92_ = lean_array_get_size(v_vs_90_);
v___x_93_ = lean_nat_dec_lt(v___x_91_, v___x_92_);
if (v___x_93_ == 0)
{
return v_x_78_;
}
else
{
uint8_t v___x_94_; 
v___x_94_ = lean_nat_dec_le(v___x_92_, v___x_92_);
if (v___x_94_ == 0)
{
if (v___x_93_ == 0)
{
return v_x_78_;
}
else
{
size_t v___x_95_; size_t v___x_96_; lean_object* v___x_97_; 
v___x_95_ = ((size_t)0ULL);
v___x_96_ = lean_usize_of_nat(v___x_92_);
v___x_97_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__8(v_vs_90_, v___x_95_, v___x_96_, v_x_78_);
return v___x_97_;
}
}
else
{
size_t v___x_98_; size_t v___x_99_; lean_object* v___x_100_; 
v___x_98_ = ((size_t)0ULL);
v___x_99_ = lean_usize_of_nat(v___x_92_);
v___x_100_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__8(v_vs_90_, v___x_98_, v___x_99_, v_x_78_);
return v___x_100_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__7_spec__9(lean_object* v_as_101_, size_t v_i_102_, size_t v_stop_103_, lean_object* v_b_104_){
_start:
{
uint8_t v___x_105_; 
v___x_105_ = lean_usize_dec_eq(v_i_102_, v_stop_103_);
if (v___x_105_ == 0)
{
lean_object* v___x_106_; lean_object* v___x_107_; size_t v___x_108_; size_t v___x_109_; 
v___x_106_ = lean_array_uget_borrowed(v_as_101_, v_i_102_);
v___x_107_ = lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__9(v___x_106_, v_b_104_);
v___x_108_ = ((size_t)1ULL);
v___x_109_ = lean_usize_add(v_i_102_, v___x_108_);
v_i_102_ = v___x_109_;
v_b_104_ = v___x_107_;
goto _start;
}
else
{
return v_b_104_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__7_spec__9___boxed(lean_object* v_as_111_, lean_object* v_i_112_, lean_object* v_stop_113_, lean_object* v_b_114_){
_start:
{
size_t v_i_boxed_115_; size_t v_stop_boxed_116_; lean_object* v_res_117_; 
v_i_boxed_115_ = lean_unbox_usize(v_i_112_);
lean_dec(v_i_112_);
v_stop_boxed_116_ = lean_unbox_usize(v_stop_113_);
lean_dec(v_stop_113_);
v_res_117_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__7_spec__9(v_as_111_, v_i_boxed_115_, v_stop_boxed_116_, v_b_114_);
lean_dec_ref(v_as_111_);
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__9___boxed(lean_object* v_x_118_, lean_object* v_x_119_){
_start:
{
lean_object* v_res_120_; 
v_res_120_ = lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__9(v_x_118_, v_x_119_);
lean_dec_ref(v_x_118_);
return v_res_120_;
}
}
static lean_object* _init_lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__7___closed__0(void){
_start:
{
lean_object* v___x_121_; 
v___x_121_ = l_Lean_instInhabitedPersistentArrayNode_default(lean_box(0));
return v___x_121_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__7(lean_object* v_x_122_, size_t v_x_123_, size_t v_x_124_, lean_object* v_x_125_){
_start:
{
if (lean_obj_tag(v_x_122_) == 0)
{
lean_object* v_cs_126_; lean_object* v___x_127_; size_t v___x_128_; lean_object* v_j_129_; lean_object* v___x_130_; size_t v___x_131_; size_t v___x_132_; size_t v___x_133_; size_t v___x_134_; size_t v___x_135_; size_t v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; uint8_t v___x_141_; 
v_cs_126_ = lean_ctor_get(v_x_122_, 0);
v___x_127_ = lean_obj_once(&lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__7___closed__0, &lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__7___closed__0_once, _init_lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__7___closed__0);
v___x_128_ = lean_usize_shift_right(v_x_123_, v_x_124_);
v_j_129_ = lean_usize_to_nat(v___x_128_);
v___x_130_ = lean_array_get_borrowed(v___x_127_, v_cs_126_, v_j_129_);
v___x_131_ = ((size_t)1ULL);
v___x_132_ = lean_usize_shift_left(v___x_131_, v_x_124_);
v___x_133_ = lean_usize_sub(v___x_132_, v___x_131_);
v___x_134_ = lean_usize_land(v_x_123_, v___x_133_);
v___x_135_ = ((size_t)5ULL);
v___x_136_ = lean_usize_sub(v_x_124_, v___x_135_);
v___x_137_ = lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__7(v___x_130_, v___x_134_, v___x_136_, v_x_125_);
v___x_138_ = lean_unsigned_to_nat(1u);
v___x_139_ = lean_nat_add(v_j_129_, v___x_138_);
lean_dec(v_j_129_);
v___x_140_ = lean_array_get_size(v_cs_126_);
v___x_141_ = lean_nat_dec_lt(v___x_139_, v___x_140_);
if (v___x_141_ == 0)
{
lean_dec(v___x_139_);
return v___x_137_;
}
else
{
uint8_t v___x_142_; 
v___x_142_ = lean_nat_dec_le(v___x_140_, v___x_140_);
if (v___x_142_ == 0)
{
if (v___x_141_ == 0)
{
lean_dec(v___x_139_);
return v___x_137_;
}
else
{
size_t v___x_143_; size_t v___x_144_; lean_object* v___x_145_; 
v___x_143_ = lean_usize_of_nat(v___x_139_);
lean_dec(v___x_139_);
v___x_144_ = lean_usize_of_nat(v___x_140_);
v___x_145_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__7_spec__9(v_cs_126_, v___x_143_, v___x_144_, v___x_137_);
return v___x_145_;
}
}
else
{
size_t v___x_146_; size_t v___x_147_; lean_object* v___x_148_; 
v___x_146_ = lean_usize_of_nat(v___x_139_);
lean_dec(v___x_139_);
v___x_147_ = lean_usize_of_nat(v___x_140_);
v___x_148_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__7_spec__9(v_cs_126_, v___x_146_, v___x_147_, v___x_137_);
return v___x_148_;
}
}
}
else
{
lean_object* v_vs_149_; lean_object* v___x_150_; lean_object* v___x_151_; uint8_t v___x_152_; 
v_vs_149_ = lean_ctor_get(v_x_122_, 0);
v___x_150_ = lean_usize_to_nat(v_x_123_);
v___x_151_ = lean_array_get_size(v_vs_149_);
v___x_152_ = lean_nat_dec_lt(v___x_150_, v___x_151_);
if (v___x_152_ == 0)
{
lean_dec(v___x_150_);
return v_x_125_;
}
else
{
uint8_t v___x_153_; 
v___x_153_ = lean_nat_dec_le(v___x_151_, v___x_151_);
if (v___x_153_ == 0)
{
if (v___x_152_ == 0)
{
lean_dec(v___x_150_);
return v_x_125_;
}
else
{
size_t v___x_154_; size_t v___x_155_; lean_object* v___x_156_; 
v___x_154_ = lean_usize_of_nat(v___x_150_);
lean_dec(v___x_150_);
v___x_155_ = lean_usize_of_nat(v___x_151_);
v___x_156_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__8(v_vs_149_, v___x_154_, v___x_155_, v_x_125_);
return v___x_156_;
}
}
else
{
size_t v___x_157_; size_t v___x_158_; lean_object* v___x_159_; 
v___x_157_ = lean_usize_of_nat(v___x_150_);
lean_dec(v___x_150_);
v___x_158_ = lean_usize_of_nat(v___x_151_);
v___x_159_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__8(v_vs_149_, v___x_157_, v___x_158_, v_x_125_);
return v___x_159_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__7___boxed(lean_object* v_x_160_, lean_object* v_x_161_, lean_object* v_x_162_, lean_object* v_x_163_){
_start:
{
size_t v_x_3542__boxed_164_; size_t v_x_3543__boxed_165_; lean_object* v_res_166_; 
v_x_3542__boxed_164_ = lean_unbox_usize(v_x_161_);
lean_dec(v_x_161_);
v_x_3543__boxed_165_ = lean_unbox_usize(v_x_162_);
lean_dec(v_x_162_);
v_res_166_ = lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__7(v_x_160_, v_x_3542__boxed_164_, v_x_3543__boxed_165_, v_x_163_);
lean_dec_ref(v_x_160_);
return v_res_166_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2(lean_object* v_t_167_, lean_object* v_init_168_, lean_object* v_start_169_){
_start:
{
lean_object* v___x_170_; uint8_t v___x_171_; 
v___x_170_ = lean_unsigned_to_nat(0u);
v___x_171_ = lean_nat_dec_eq(v_start_169_, v___x_170_);
if (v___x_171_ == 0)
{
lean_object* v_root_172_; lean_object* v_tail_173_; size_t v_shift_174_; lean_object* v_tailOff_175_; uint8_t v___x_176_; 
v_root_172_ = lean_ctor_get(v_t_167_, 0);
v_tail_173_ = lean_ctor_get(v_t_167_, 1);
v_shift_174_ = lean_ctor_get_usize(v_t_167_, 4);
v_tailOff_175_ = lean_ctor_get(v_t_167_, 3);
v___x_176_ = lean_nat_dec_le(v_tailOff_175_, v_start_169_);
if (v___x_176_ == 0)
{
size_t v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; uint8_t v___x_180_; 
v___x_177_ = lean_usize_of_nat(v_start_169_);
v___x_178_ = lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__7(v_root_172_, v___x_177_, v_shift_174_, v_init_168_);
v___x_179_ = lean_array_get_size(v_tail_173_);
v___x_180_ = lean_nat_dec_lt(v___x_170_, v___x_179_);
if (v___x_180_ == 0)
{
return v___x_178_;
}
else
{
uint8_t v___x_181_; 
v___x_181_ = lean_nat_dec_le(v___x_179_, v___x_179_);
if (v___x_181_ == 0)
{
if (v___x_180_ == 0)
{
return v___x_178_;
}
else
{
size_t v___x_182_; size_t v___x_183_; lean_object* v___x_184_; 
v___x_182_ = ((size_t)0ULL);
v___x_183_ = lean_usize_of_nat(v___x_179_);
v___x_184_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__8(v_tail_173_, v___x_182_, v___x_183_, v___x_178_);
return v___x_184_;
}
}
else
{
size_t v___x_185_; size_t v___x_186_; lean_object* v___x_187_; 
v___x_185_ = ((size_t)0ULL);
v___x_186_ = lean_usize_of_nat(v___x_179_);
v___x_187_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__8(v_tail_173_, v___x_185_, v___x_186_, v___x_178_);
return v___x_187_;
}
}
}
else
{
lean_object* v___x_188_; lean_object* v___x_189_; uint8_t v___x_190_; 
v___x_188_ = lean_nat_sub(v_start_169_, v_tailOff_175_);
v___x_189_ = lean_array_get_size(v_tail_173_);
v___x_190_ = lean_nat_dec_lt(v___x_188_, v___x_189_);
if (v___x_190_ == 0)
{
lean_dec(v___x_188_);
return v_init_168_;
}
else
{
uint8_t v___x_191_; 
v___x_191_ = lean_nat_dec_le(v___x_189_, v___x_189_);
if (v___x_191_ == 0)
{
if (v___x_190_ == 0)
{
lean_dec(v___x_188_);
return v_init_168_;
}
else
{
size_t v___x_192_; size_t v___x_193_; lean_object* v___x_194_; 
v___x_192_ = lean_usize_of_nat(v___x_188_);
lean_dec(v___x_188_);
v___x_193_ = lean_usize_of_nat(v___x_189_);
v___x_194_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__8(v_tail_173_, v___x_192_, v___x_193_, v_init_168_);
return v___x_194_;
}
}
else
{
size_t v___x_195_; size_t v___x_196_; lean_object* v___x_197_; 
v___x_195_ = lean_usize_of_nat(v___x_188_);
lean_dec(v___x_188_);
v___x_196_ = lean_usize_of_nat(v___x_189_);
v___x_197_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__8(v_tail_173_, v___x_195_, v___x_196_, v_init_168_);
return v___x_197_;
}
}
}
}
else
{
lean_object* v_root_198_; lean_object* v_tail_199_; lean_object* v___x_200_; lean_object* v___x_201_; uint8_t v___x_202_; 
v_root_198_ = lean_ctor_get(v_t_167_, 0);
v_tail_199_ = lean_ctor_get(v_t_167_, 1);
v___x_200_ = lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__9(v_root_198_, v_init_168_);
v___x_201_ = lean_array_get_size(v_tail_199_);
v___x_202_ = lean_nat_dec_lt(v___x_170_, v___x_201_);
if (v___x_202_ == 0)
{
return v___x_200_;
}
else
{
uint8_t v___x_203_; 
v___x_203_ = lean_nat_dec_le(v___x_201_, v___x_201_);
if (v___x_203_ == 0)
{
if (v___x_202_ == 0)
{
return v___x_200_;
}
else
{
size_t v___x_204_; size_t v___x_205_; lean_object* v___x_206_; 
v___x_204_ = ((size_t)0ULL);
v___x_205_ = lean_usize_of_nat(v___x_201_);
v___x_206_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__8(v_tail_199_, v___x_204_, v___x_205_, v___x_200_);
return v___x_206_;
}
}
else
{
size_t v___x_207_; size_t v___x_208_; lean_object* v___x_209_; 
v___x_207_ = ((size_t)0ULL);
v___x_208_ = lean_usize_of_nat(v___x_201_);
v___x_209_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__8(v_tail_199_, v___x_207_, v___x_208_, v___x_200_);
return v___x_209_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2___boxed(lean_object* v_t_210_, lean_object* v_init_211_, lean_object* v_start_212_){
_start:
{
lean_object* v_res_213_; 
v_res_213_ = lp_Qq_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2(v_t_210_, v_init_211_, v_start_212_);
lean_dec(v_start_212_);
lean_dec_ref(v_t_210_);
return v_res_213_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1(lean_object* v_lctx_214_, lean_object* v_init_215_, lean_object* v_start_216_){
_start:
{
lean_object* v_decls_217_; lean_object* v___x_218_; 
v_decls_217_ = lean_ctor_get(v_lctx_214_, 1);
v___x_218_ = lp_Qq_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2(v_decls_217_, v_init_215_, v_start_216_);
return v___x_218_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1___boxed(lean_object* v_lctx_219_, lean_object* v_init_220_, lean_object* v_start_221_){
_start:
{
lean_object* v_res_222_; 
v_res_222_ = lp_Qq_Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1(v_lctx_219_, v_init_220_, v_start_221_);
lean_dec(v_start_221_);
lean_dec_ref(v_lctx_219_);
return v_res_222_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__3(lean_object* v_values_223_, lean_object* v_as_224_, size_t v_i_225_, size_t v_stop_226_, lean_object* v_b_227_){
_start:
{
lean_object* v___y_229_; uint8_t v___x_233_; 
v___x_233_ = lean_usize_dec_eq(v_i_225_, v_stop_226_);
if (v___x_233_ == 0)
{
lean_object* v___x_234_; 
v___x_234_ = lean_array_uget_borrowed(v_as_224_, v_i_225_);
if (lean_obj_tag(v___x_234_) == 0)
{
v___y_229_ = v_b_227_;
goto v___jp_228_;
}
else
{
lean_object* v_val_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; uint8_t v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; 
v_val_235_ = lean_ctor_get(v___x_234_, 0);
v___x_236_ = l_Lean_instInhabitedExpr;
v___x_237_ = l_Lean_LocalDecl_index(v_val_235_);
v___x_238_ = l_Lean_LocalDecl_fvarId(v_val_235_);
v___x_239_ = l_Lean_LocalDecl_userName(v_val_235_);
v___x_240_ = l_Lean_LocalDecl_type(v_val_235_);
v___x_241_ = lean_array_get_borrowed(v___x_236_, v_values_223_, v___x_237_);
v___x_242_ = l_Lean_LocalDecl_kind(v_val_235_);
lean_inc(v___x_241_);
v___x_243_ = lean_alloc_ctor(1, 5, 2);
lean_ctor_set(v___x_243_, 0, v___x_237_);
lean_ctor_set(v___x_243_, 1, v___x_238_);
lean_ctor_set(v___x_243_, 2, v___x_239_);
lean_ctor_set(v___x_243_, 3, v___x_240_);
lean_ctor_set(v___x_243_, 4, v___x_241_);
lean_ctor_set_uint8(v___x_243_, sizeof(void*)*5, v___x_233_);
lean_ctor_set_uint8(v___x_243_, sizeof(void*)*5 + 1, v___x_242_);
v___x_244_ = l_Lean_LocalContext_addDecl(v_b_227_, v___x_243_);
v___y_229_ = v___x_244_;
goto v___jp_228_;
}
}
else
{
return v_b_227_;
}
v___jp_228_:
{
size_t v___x_230_; size_t v___x_231_; 
v___x_230_ = ((size_t)1ULL);
v___x_231_ = lean_usize_add(v_i_225_, v___x_230_);
v_i_225_ = v___x_231_;
v_b_227_ = v___y_229_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__3___boxed(lean_object* v_values_245_, lean_object* v_as_246_, lean_object* v_i_247_, lean_object* v_stop_248_, lean_object* v_b_249_){
_start:
{
size_t v_i_boxed_250_; size_t v_stop_boxed_251_; lean_object* v_res_252_; 
v_i_boxed_250_ = lean_unbox_usize(v_i_247_);
lean_dec(v_i_247_);
v_stop_boxed_251_ = lean_unbox_usize(v_stop_248_);
lean_dec(v_stop_248_);
v_res_252_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__3(v_values_245_, v_as_246_, v_i_boxed_250_, v_stop_boxed_251_, v_b_249_);
lean_dec_ref(v_as_246_);
lean_dec_ref(v_values_245_);
return v_res_252_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__4(lean_object* v_values_253_, lean_object* v_x_254_, lean_object* v_x_255_){
_start:
{
if (lean_obj_tag(v_x_254_) == 0)
{
lean_object* v_cs_256_; lean_object* v___x_257_; lean_object* v___x_258_; uint8_t v___x_259_; 
v_cs_256_ = lean_ctor_get(v_x_254_, 0);
v___x_257_ = lean_unsigned_to_nat(0u);
v___x_258_ = lean_array_get_size(v_cs_256_);
v___x_259_ = lean_nat_dec_lt(v___x_257_, v___x_258_);
if (v___x_259_ == 0)
{
return v_x_255_;
}
else
{
uint8_t v___x_260_; 
v___x_260_ = lean_nat_dec_le(v___x_258_, v___x_258_);
if (v___x_260_ == 0)
{
if (v___x_259_ == 0)
{
return v_x_255_;
}
else
{
size_t v___x_261_; size_t v___x_262_; lean_object* v___x_263_; 
v___x_261_ = ((size_t)0ULL);
v___x_262_ = lean_usize_of_nat(v___x_258_);
v___x_263_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__2_spec__4(v_values_253_, v_cs_256_, v___x_261_, v___x_262_, v_x_255_);
return v___x_263_;
}
}
else
{
size_t v___x_264_; size_t v___x_265_; lean_object* v___x_266_; 
v___x_264_ = ((size_t)0ULL);
v___x_265_ = lean_usize_of_nat(v___x_258_);
v___x_266_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__2_spec__4(v_values_253_, v_cs_256_, v___x_264_, v___x_265_, v_x_255_);
return v___x_266_;
}
}
}
else
{
lean_object* v_vs_267_; lean_object* v___x_268_; lean_object* v___x_269_; uint8_t v___x_270_; 
v_vs_267_ = lean_ctor_get(v_x_254_, 0);
v___x_268_ = lean_unsigned_to_nat(0u);
v___x_269_ = lean_array_get_size(v_vs_267_);
v___x_270_ = lean_nat_dec_lt(v___x_268_, v___x_269_);
if (v___x_270_ == 0)
{
return v_x_255_;
}
else
{
uint8_t v___x_271_; 
v___x_271_ = lean_nat_dec_le(v___x_269_, v___x_269_);
if (v___x_271_ == 0)
{
if (v___x_270_ == 0)
{
return v_x_255_;
}
else
{
size_t v___x_272_; size_t v___x_273_; lean_object* v___x_274_; 
v___x_272_ = ((size_t)0ULL);
v___x_273_ = lean_usize_of_nat(v___x_269_);
v___x_274_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__3(v_values_253_, v_vs_267_, v___x_272_, v___x_273_, v_x_255_);
return v___x_274_;
}
}
else
{
size_t v___x_275_; size_t v___x_276_; lean_object* v___x_277_; 
v___x_275_ = ((size_t)0ULL);
v___x_276_ = lean_usize_of_nat(v___x_269_);
v___x_277_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__3(v_values_253_, v_vs_267_, v___x_275_, v___x_276_, v_x_255_);
return v___x_277_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__2_spec__4(lean_object* v_values_278_, lean_object* v_as_279_, size_t v_i_280_, size_t v_stop_281_, lean_object* v_b_282_){
_start:
{
uint8_t v___x_283_; 
v___x_283_ = lean_usize_dec_eq(v_i_280_, v_stop_281_);
if (v___x_283_ == 0)
{
lean_object* v___x_284_; lean_object* v___x_285_; size_t v___x_286_; size_t v___x_287_; 
v___x_284_ = lean_array_uget_borrowed(v_as_279_, v_i_280_);
v___x_285_ = lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__4(v_values_278_, v___x_284_, v_b_282_);
v___x_286_ = ((size_t)1ULL);
v___x_287_ = lean_usize_add(v_i_280_, v___x_286_);
v_i_280_ = v___x_287_;
v_b_282_ = v___x_285_;
goto _start;
}
else
{
return v_b_282_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__2_spec__4___boxed(lean_object* v_values_289_, lean_object* v_as_290_, lean_object* v_i_291_, lean_object* v_stop_292_, lean_object* v_b_293_){
_start:
{
size_t v_i_boxed_294_; size_t v_stop_boxed_295_; lean_object* v_res_296_; 
v_i_boxed_294_ = lean_unbox_usize(v_i_291_);
lean_dec(v_i_291_);
v_stop_boxed_295_ = lean_unbox_usize(v_stop_292_);
lean_dec(v_stop_292_);
v_res_296_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__2_spec__4(v_values_289_, v_as_290_, v_i_boxed_294_, v_stop_boxed_295_, v_b_293_);
lean_dec_ref(v_as_290_);
lean_dec_ref(v_values_289_);
return v_res_296_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__4___boxed(lean_object* v_values_297_, lean_object* v_x_298_, lean_object* v_x_299_){
_start:
{
lean_object* v_res_300_; 
v_res_300_ = lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__4(v_values_297_, v_x_298_, v_x_299_);
lean_dec_ref(v_x_298_);
lean_dec_ref(v_values_297_);
return v_res_300_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__2(lean_object* v_values_301_, lean_object* v_x_302_, size_t v_x_303_, size_t v_x_304_, lean_object* v_x_305_){
_start:
{
if (lean_obj_tag(v_x_302_) == 0)
{
lean_object* v_cs_306_; lean_object* v___x_307_; size_t v___x_308_; lean_object* v_j_309_; lean_object* v___x_310_; size_t v___x_311_; size_t v___x_312_; size_t v___x_313_; size_t v___x_314_; size_t v___x_315_; size_t v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; uint8_t v___x_321_; 
v_cs_306_ = lean_ctor_get(v_x_302_, 0);
v___x_307_ = lean_obj_once(&lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__7___closed__0, &lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__7___closed__0_once, _init_lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1_spec__2_spec__7___closed__0);
v___x_308_ = lean_usize_shift_right(v_x_303_, v_x_304_);
v_j_309_ = lean_usize_to_nat(v___x_308_);
v___x_310_ = lean_array_get_borrowed(v___x_307_, v_cs_306_, v_j_309_);
v___x_311_ = ((size_t)1ULL);
v___x_312_ = lean_usize_shift_left(v___x_311_, v_x_304_);
v___x_313_ = lean_usize_sub(v___x_312_, v___x_311_);
v___x_314_ = lean_usize_land(v_x_303_, v___x_313_);
v___x_315_ = ((size_t)5ULL);
v___x_316_ = lean_usize_sub(v_x_304_, v___x_315_);
v___x_317_ = lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__2(v_values_301_, v___x_310_, v___x_314_, v___x_316_, v_x_305_);
v___x_318_ = lean_unsigned_to_nat(1u);
v___x_319_ = lean_nat_add(v_j_309_, v___x_318_);
lean_dec(v_j_309_);
v___x_320_ = lean_array_get_size(v_cs_306_);
v___x_321_ = lean_nat_dec_lt(v___x_319_, v___x_320_);
if (v___x_321_ == 0)
{
lean_dec(v___x_319_);
return v___x_317_;
}
else
{
uint8_t v___x_322_; 
v___x_322_ = lean_nat_dec_le(v___x_320_, v___x_320_);
if (v___x_322_ == 0)
{
if (v___x_321_ == 0)
{
lean_dec(v___x_319_);
return v___x_317_;
}
else
{
size_t v___x_323_; size_t v___x_324_; lean_object* v___x_325_; 
v___x_323_ = lean_usize_of_nat(v___x_319_);
lean_dec(v___x_319_);
v___x_324_ = lean_usize_of_nat(v___x_320_);
v___x_325_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__2_spec__4(v_values_301_, v_cs_306_, v___x_323_, v___x_324_, v___x_317_);
return v___x_325_;
}
}
else
{
size_t v___x_326_; size_t v___x_327_; lean_object* v___x_328_; 
v___x_326_ = lean_usize_of_nat(v___x_319_);
lean_dec(v___x_319_);
v___x_327_ = lean_usize_of_nat(v___x_320_);
v___x_328_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__2_spec__4(v_values_301_, v_cs_306_, v___x_326_, v___x_327_, v___x_317_);
return v___x_328_;
}
}
}
else
{
lean_object* v_vs_329_; lean_object* v___x_330_; lean_object* v___x_331_; uint8_t v___x_332_; 
v_vs_329_ = lean_ctor_get(v_x_302_, 0);
v___x_330_ = lean_usize_to_nat(v_x_303_);
v___x_331_ = lean_array_get_size(v_vs_329_);
v___x_332_ = lean_nat_dec_lt(v___x_330_, v___x_331_);
if (v___x_332_ == 0)
{
lean_dec(v___x_330_);
return v_x_305_;
}
else
{
uint8_t v___x_333_; 
v___x_333_ = lean_nat_dec_le(v___x_331_, v___x_331_);
if (v___x_333_ == 0)
{
if (v___x_332_ == 0)
{
lean_dec(v___x_330_);
return v_x_305_;
}
else
{
size_t v___x_334_; size_t v___x_335_; lean_object* v___x_336_; 
v___x_334_ = lean_usize_of_nat(v___x_330_);
lean_dec(v___x_330_);
v___x_335_ = lean_usize_of_nat(v___x_331_);
v___x_336_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__3(v_values_301_, v_vs_329_, v___x_334_, v___x_335_, v_x_305_);
return v___x_336_;
}
}
else
{
size_t v___x_337_; size_t v___x_338_; lean_object* v___x_339_; 
v___x_337_ = lean_usize_of_nat(v___x_330_);
lean_dec(v___x_330_);
v___x_338_ = lean_usize_of_nat(v___x_331_);
v___x_339_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__3(v_values_301_, v_vs_329_, v___x_337_, v___x_338_, v_x_305_);
return v___x_339_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__2___boxed(lean_object* v_values_340_, lean_object* v_x_341_, lean_object* v_x_342_, lean_object* v_x_343_, lean_object* v_x_344_){
_start:
{
size_t v_x_3788__boxed_345_; size_t v_x_3789__boxed_346_; lean_object* v_res_347_; 
v_x_3788__boxed_345_ = lean_unbox_usize(v_x_342_);
lean_dec(v_x_342_);
v_x_3789__boxed_346_ = lean_unbox_usize(v_x_343_);
lean_dec(v_x_343_);
v_res_347_ = lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__2(v_values_340_, v_x_341_, v_x_3788__boxed_345_, v_x_3789__boxed_346_, v_x_344_);
lean_dec_ref(v_x_341_);
lean_dec_ref(v_values_340_);
return v_res_347_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0(lean_object* v_values_348_, lean_object* v_t_349_, lean_object* v_init_350_, lean_object* v_start_351_){
_start:
{
lean_object* v___x_352_; uint8_t v___x_353_; 
v___x_352_ = lean_unsigned_to_nat(0u);
v___x_353_ = lean_nat_dec_eq(v_start_351_, v___x_352_);
if (v___x_353_ == 0)
{
lean_object* v_root_354_; lean_object* v_tail_355_; size_t v_shift_356_; lean_object* v_tailOff_357_; uint8_t v___x_358_; 
v_root_354_ = lean_ctor_get(v_t_349_, 0);
v_tail_355_ = lean_ctor_get(v_t_349_, 1);
v_shift_356_ = lean_ctor_get_usize(v_t_349_, 4);
v_tailOff_357_ = lean_ctor_get(v_t_349_, 3);
v___x_358_ = lean_nat_dec_le(v_tailOff_357_, v_start_351_);
if (v___x_358_ == 0)
{
size_t v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; uint8_t v___x_362_; 
v___x_359_ = lean_usize_of_nat(v_start_351_);
v___x_360_ = lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__2(v_values_348_, v_root_354_, v___x_359_, v_shift_356_, v_init_350_);
v___x_361_ = lean_array_get_size(v_tail_355_);
v___x_362_ = lean_nat_dec_lt(v___x_352_, v___x_361_);
if (v___x_362_ == 0)
{
return v___x_360_;
}
else
{
uint8_t v___x_363_; 
v___x_363_ = lean_nat_dec_le(v___x_361_, v___x_361_);
if (v___x_363_ == 0)
{
if (v___x_362_ == 0)
{
return v___x_360_;
}
else
{
size_t v___x_364_; size_t v___x_365_; lean_object* v___x_366_; 
v___x_364_ = ((size_t)0ULL);
v___x_365_ = lean_usize_of_nat(v___x_361_);
v___x_366_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__3(v_values_348_, v_tail_355_, v___x_364_, v___x_365_, v___x_360_);
return v___x_366_;
}
}
else
{
size_t v___x_367_; size_t v___x_368_; lean_object* v___x_369_; 
v___x_367_ = ((size_t)0ULL);
v___x_368_ = lean_usize_of_nat(v___x_361_);
v___x_369_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__3(v_values_348_, v_tail_355_, v___x_367_, v___x_368_, v___x_360_);
return v___x_369_;
}
}
}
else
{
lean_object* v___x_370_; lean_object* v___x_371_; uint8_t v___x_372_; 
v___x_370_ = lean_nat_sub(v_start_351_, v_tailOff_357_);
v___x_371_ = lean_array_get_size(v_tail_355_);
v___x_372_ = lean_nat_dec_lt(v___x_370_, v___x_371_);
if (v___x_372_ == 0)
{
lean_dec(v___x_370_);
return v_init_350_;
}
else
{
uint8_t v___x_373_; 
v___x_373_ = lean_nat_dec_le(v___x_371_, v___x_371_);
if (v___x_373_ == 0)
{
if (v___x_372_ == 0)
{
lean_dec(v___x_370_);
return v_init_350_;
}
else
{
size_t v___x_374_; size_t v___x_375_; lean_object* v___x_376_; 
v___x_374_ = lean_usize_of_nat(v___x_370_);
lean_dec(v___x_370_);
v___x_375_ = lean_usize_of_nat(v___x_371_);
v___x_376_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__3(v_values_348_, v_tail_355_, v___x_374_, v___x_375_, v_init_350_);
return v___x_376_;
}
}
else
{
size_t v___x_377_; size_t v___x_378_; lean_object* v___x_379_; 
v___x_377_ = lean_usize_of_nat(v___x_370_);
lean_dec(v___x_370_);
v___x_378_ = lean_usize_of_nat(v___x_371_);
v___x_379_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__3(v_values_348_, v_tail_355_, v___x_377_, v___x_378_, v_init_350_);
return v___x_379_;
}
}
}
}
else
{
lean_object* v_root_380_; lean_object* v_tail_381_; lean_object* v___x_382_; lean_object* v___x_383_; uint8_t v___x_384_; 
v_root_380_ = lean_ctor_get(v_t_349_, 0);
v_tail_381_ = lean_ctor_get(v_t_349_, 1);
v___x_382_ = lp_Qq___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__4(v_values_348_, v_root_380_, v_init_350_);
v___x_383_ = lean_array_get_size(v_tail_381_);
v___x_384_ = lean_nat_dec_lt(v___x_352_, v___x_383_);
if (v___x_384_ == 0)
{
return v___x_382_;
}
else
{
uint8_t v___x_385_; 
v___x_385_ = lean_nat_dec_le(v___x_383_, v___x_383_);
if (v___x_385_ == 0)
{
if (v___x_384_ == 0)
{
return v___x_382_;
}
else
{
size_t v___x_386_; size_t v___x_387_; lean_object* v___x_388_; 
v___x_386_ = ((size_t)0ULL);
v___x_387_ = lean_usize_of_nat(v___x_383_);
v___x_388_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__3(v_values_348_, v_tail_381_, v___x_386_, v___x_387_, v___x_382_);
return v___x_388_;
}
}
else
{
size_t v___x_389_; size_t v___x_390_; lean_object* v___x_391_; 
v___x_389_ = ((size_t)0ULL);
v___x_390_ = lean_usize_of_nat(v___x_383_);
v___x_391_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0_spec__3(v_values_348_, v_tail_381_, v___x_389_, v___x_390_, v___x_382_);
return v___x_391_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0___boxed(lean_object* v_values_392_, lean_object* v_t_393_, lean_object* v_init_394_, lean_object* v_start_395_){
_start:
{
lean_object* v_res_396_; 
v_res_396_ = lp_Qq_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0(v_values_392_, v_t_393_, v_init_394_, v_start_395_);
lean_dec(v_start_395_);
lean_dec_ref(v_t_393_);
lean_dec_ref(v_values_392_);
return v_res_396_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0(lean_object* v_values_397_, lean_object* v_lctx_398_, lean_object* v_init_399_, lean_object* v_start_400_){
_start:
{
lean_object* v_decls_401_; lean_object* v___x_402_; 
v_decls_401_ = lean_ctor_get(v_lctx_398_, 1);
v___x_402_ = lp_Qq_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0_spec__0(v_values_397_, v_decls_401_, v_init_399_, v_start_400_);
return v___x_402_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0___boxed(lean_object* v_values_403_, lean_object* v_lctx_404_, lean_object* v_init_405_, lean_object* v_start_406_){
_start:
{
lean_object* v_res_407_; 
v_res_407_ = lp_Qq_Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0(v_values_403_, v_lctx_404_, v_init_405_, v_start_406_);
lean_dec(v_start_406_);
lean_dec_ref(v_lctx_404_);
lean_dec_ref(v_values_403_);
return v_res_407_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Commands_0__Qq_mkLetFVarsFromValues(lean_object* v_values_410_, lean_object* v_body_411_, lean_object* v_a_412_, lean_object* v_a_413_, lean_object* v_a_414_, lean_object* v_a_415_){
_start:
{
lean_object* v_lctx_417_; lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; uint8_t v___x_423_; uint8_t v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; 
v_lctx_417_ = lean_ctor_get(v_a_412_, 2);
v___x_418_ = l_Lean_LocalContext_empty;
v___x_419_ = lean_unsigned_to_nat(0u);
v___x_420_ = lp_Qq_Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__0(v_values_410_, v_lctx_417_, v___x_418_, v___x_419_);
v___x_421_ = ((lean_object*)(lp_Qq___private_Qq_Commands_0__Qq_mkLetFVarsFromValues___closed__0));
v___x_422_ = lp_Qq_Lean_LocalContext_foldlM___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__1(v_lctx_417_, v___x_421_, v___x_419_);
v___x_423_ = 1;
v___x_424_ = 1;
v___x_425_ = lean_box(v___x_423_);
v___x_426_ = lean_box(v___x_423_);
v___x_427_ = lean_box(v___x_424_);
v___x_428_ = lean_alloc_closure((void*)(l_Lean_Meta_mkLetFVars___boxed), 10, 5);
lean_closure_set(v___x_428_, 0, v___x_422_);
lean_closure_set(v___x_428_, 1, v_body_411_);
lean_closure_set(v___x_428_, 2, v___x_425_);
lean_closure_set(v___x_428_, 3, v___x_426_);
lean_closure_set(v___x_428_, 4, v___x_427_);
v___x_429_ = lp_Qq_Lean_Meta_withLCtx___at___00__private_Qq_Commands_0__Qq_mkLetFVarsFromValues_spec__2___redArg(v___x_420_, v___x_421_, v___x_428_, v_a_412_, v_a_413_, v_a_414_, v_a_415_);
return v___x_429_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Commands_0__Qq_mkLetFVarsFromValues___boxed(lean_object* v_values_430_, lean_object* v_body_431_, lean_object* v_a_432_, lean_object* v_a_433_, lean_object* v_a_434_, lean_object* v_a_435_, lean_object* v_a_436_){
_start:
{
lean_object* v_res_437_; 
v_res_437_ = lp_Qq___private_Qq_Commands_0__Qq_mkLetFVarsFromValues(v_values_430_, v_body_431_, v_a_432_, v_a_433_, v_a_434_, v_a_435_);
lean_dec(v_a_435_);
lean_dec_ref(v_a_434_);
lean_dec(v_a_433_);
lean_dec_ref(v_a_432_);
lean_dec_ref(v_values_430_);
return v_res_437_;
}
}
static lean_object* _init_lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__5(void){
_start:
{
lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; 
v___x_472_ = lean_box(0);
v___x_473_ = ((lean_object*)(lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__4));
v___x_474_ = l_Lean_Expr_const___override(v___x_473_, v___x_472_);
return v___x_474_;
}
}
static lean_object* _init_lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__8(void){
_start:
{
lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; 
v___x_479_ = lean_box(0);
v___x_480_ = ((lean_object*)(lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__7));
v___x_481_ = l_Lean_Expr_const___override(v___x_480_, v___x_479_);
return v___x_481_;
}
}
static lean_object* _init_lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__9(void){
_start:
{
lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; 
v___x_482_ = lean_obj_once(&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__8, &lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__8_once, _init_lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__8);
v___x_483_ = lean_obj_once(&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__5, &lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__5_once, _init_lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__5);
v___x_484_ = l_Lean_Expr_app___override(v___x_483_, v___x_482_);
return v___x_484_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3(lean_object* v_codeExpr_485_, lean_object* v_a_486_, lean_object* v_a_487_, lean_object* v_a_488_, lean_object* v_a_489_){
_start:
{
lean_object* v___x_491_; uint8_t v___x_492_; uint8_t v___x_493_; lean_object* v___x_494_; 
v___x_491_ = lean_obj_once(&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__9, &lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__9_once, _init_lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__9);
v___x_492_ = 1;
v___x_493_ = 1;
v___x_494_ = l_Lean_Meta_evalExpr___redArg(v___x_491_, v_codeExpr_485_, v___x_492_, v___x_493_, v_a_486_, v_a_487_, v_a_488_, v_a_489_);
return v___x_494_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___boxed(lean_object* v_codeExpr_495_, lean_object* v_a_496_, lean_object* v_a_497_, lean_object* v_a_498_, lean_object* v_a_499_, lean_object* v_a_500_){
_start:
{
lean_object* v_res_501_; 
v_res_501_ = lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3(v_codeExpr_495_, v_a_496_, v_a_497_, v_a_498_, v_a_499_);
lean_dec(v_a_499_);
lean_dec_ref(v_a_498_);
lean_dec(v_a_497_);
lean_dec_ref(v_a_496_);
return v_res_501_;
}
}
static lean_object* _init_lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; 
v___x_502_ = lean_box(0);
v___x_503_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_504_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_504_, 0, v___x_503_);
lean_ctor_set(v___x_504_, 1, v___x_502_);
return v___x_504_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__0___redArg(){
_start:
{
lean_object* v___x_506_; lean_object* v___x_507_; 
v___x_506_ = lean_obj_once(&lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__0___redArg___closed__0, &lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__0___redArg___closed__0_once, _init_lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__0___redArg___closed__0);
v___x_507_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_507_, 0, v___x_506_);
return v___x_507_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__0___redArg___boxed(lean_object* v___y_508_){
_start:
{
lean_object* v_res_509_; 
v_res_509_ = lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__0___redArg();
return v_res_509_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__0(lean_object* v_00_u03b1_510_, lean_object* v___y_511_, lean_object* v___y_512_, lean_object* v___y_513_, lean_object* v___y_514_, lean_object* v___y_515_, lean_object* v___y_516_){
_start:
{
lean_object* v___x_518_; 
v___x_518_ = lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__0___redArg();
return v___x_518_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__0___boxed(lean_object* v_00_u03b1_519_, lean_object* v___y_520_, lean_object* v___y_521_, lean_object* v___y_522_, lean_object* v___y_523_, lean_object* v___y_524_, lean_object* v___y_525_, lean_object* v___y_526_){
_start:
{
lean_object* v_res_527_; 
v_res_527_ = lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__0(v_00_u03b1_519_, v___y_520_, v___y_521_, v___y_522_, v___y_523_, v___y_524_, v___y_525_);
lean_dec(v___y_525_);
lean_dec_ref(v___y_524_);
lean_dec(v___y_523_);
lean_dec_ref(v___y_522_);
lean_dec(v___y_521_);
lean_dec_ref(v___y_520_);
return v_res_527_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__1___redArg(lean_object* v_e_528_, lean_object* v___y_529_, lean_object* v___y_530_){
_start:
{
uint8_t v___x_532_; 
v___x_532_ = l_Lean_Expr_hasMVar(v_e_528_);
if (v___x_532_ == 0)
{
lean_object* v___x_533_; lean_object* v___x_534_; 
v___x_533_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_533_, 0, v_e_528_);
lean_ctor_set(v___x_533_, 1, v___y_529_);
v___x_534_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_534_, 0, v___x_533_);
return v___x_534_;
}
else
{
lean_object* v___x_535_; lean_object* v_mctx_536_; lean_object* v___x_537_; lean_object* v_fst_538_; lean_object* v_snd_539_; lean_object* v___x_541_; uint8_t v_isShared_542_; uint8_t v_isSharedCheck_561_; 
v___x_535_ = lean_st_ref_get(v___y_530_);
v_mctx_536_ = lean_ctor_get(v___x_535_, 0);
lean_inc_ref(v_mctx_536_);
lean_dec(v___x_535_);
v___x_537_ = l_Lean_instantiateMVarsCore(v_mctx_536_, v_e_528_);
v_fst_538_ = lean_ctor_get(v___x_537_, 0);
v_snd_539_ = lean_ctor_get(v___x_537_, 1);
v_isSharedCheck_561_ = !lean_is_exclusive(v___x_537_);
if (v_isSharedCheck_561_ == 0)
{
v___x_541_ = v___x_537_;
v_isShared_542_ = v_isSharedCheck_561_;
goto v_resetjp_540_;
}
else
{
lean_inc(v_snd_539_);
lean_inc(v_fst_538_);
lean_dec(v___x_537_);
v___x_541_ = lean_box(0);
v_isShared_542_ = v_isSharedCheck_561_;
goto v_resetjp_540_;
}
v_resetjp_540_:
{
lean_object* v___x_543_; lean_object* v_cache_544_; lean_object* v_zetaDeltaFVarIds_545_; lean_object* v_postponed_546_; lean_object* v_diag_547_; lean_object* v___x_549_; uint8_t v_isShared_550_; uint8_t v_isSharedCheck_559_; 
v___x_543_ = lean_st_ref_take(v___y_530_);
v_cache_544_ = lean_ctor_get(v___x_543_, 1);
v_zetaDeltaFVarIds_545_ = lean_ctor_get(v___x_543_, 2);
v_postponed_546_ = lean_ctor_get(v___x_543_, 3);
v_diag_547_ = lean_ctor_get(v___x_543_, 4);
v_isSharedCheck_559_ = !lean_is_exclusive(v___x_543_);
if (v_isSharedCheck_559_ == 0)
{
lean_object* v_unused_560_; 
v_unused_560_ = lean_ctor_get(v___x_543_, 0);
lean_dec(v_unused_560_);
v___x_549_ = v___x_543_;
v_isShared_550_ = v_isSharedCheck_559_;
goto v_resetjp_548_;
}
else
{
lean_inc(v_diag_547_);
lean_inc(v_postponed_546_);
lean_inc(v_zetaDeltaFVarIds_545_);
lean_inc(v_cache_544_);
lean_dec(v___x_543_);
v___x_549_ = lean_box(0);
v_isShared_550_ = v_isSharedCheck_559_;
goto v_resetjp_548_;
}
v_resetjp_548_:
{
lean_object* v___x_552_; 
if (v_isShared_550_ == 0)
{
lean_ctor_set(v___x_549_, 0, v_snd_539_);
v___x_552_ = v___x_549_;
goto v_reusejp_551_;
}
else
{
lean_object* v_reuseFailAlloc_558_; 
v_reuseFailAlloc_558_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_558_, 0, v_snd_539_);
lean_ctor_set(v_reuseFailAlloc_558_, 1, v_cache_544_);
lean_ctor_set(v_reuseFailAlloc_558_, 2, v_zetaDeltaFVarIds_545_);
lean_ctor_set(v_reuseFailAlloc_558_, 3, v_postponed_546_);
lean_ctor_set(v_reuseFailAlloc_558_, 4, v_diag_547_);
v___x_552_ = v_reuseFailAlloc_558_;
goto v_reusejp_551_;
}
v_reusejp_551_:
{
lean_object* v___x_553_; lean_object* v___x_555_; 
v___x_553_ = lean_st_ref_set(v___y_530_, v___x_552_);
if (v_isShared_542_ == 0)
{
lean_ctor_set(v___x_541_, 1, v___y_529_);
v___x_555_ = v___x_541_;
goto v_reusejp_554_;
}
else
{
lean_object* v_reuseFailAlloc_557_; 
v_reuseFailAlloc_557_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_557_, 0, v_fst_538_);
lean_ctor_set(v_reuseFailAlloc_557_, 1, v___y_529_);
v___x_555_ = v_reuseFailAlloc_557_;
goto v_reusejp_554_;
}
v_reusejp_554_:
{
lean_object* v___x_556_; 
v___x_556_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_556_, 0, v___x_555_);
return v___x_556_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__1___redArg___boxed(lean_object* v_e_562_, lean_object* v___y_563_, lean_object* v___y_564_, lean_object* v___y_565_){
_start:
{
lean_object* v_res_566_; 
v_res_566_ = lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__1___redArg(v_e_562_, v___y_563_, v___y_564_);
lean_dec(v___y_564_);
return v_res_566_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__1(lean_object* v_e_567_, lean_object* v___y_568_, lean_object* v___y_569_, lean_object* v___y_570_, lean_object* v___y_571_, lean_object* v___y_572_){
_start:
{
lean_object* v___x_574_; 
v___x_574_ = lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__1___redArg(v_e_567_, v___y_568_, v___y_570_);
return v___x_574_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__1___boxed(lean_object* v_e_575_, lean_object* v___y_576_, lean_object* v___y_577_, lean_object* v___y_578_, lean_object* v___y_579_, lean_object* v___y_580_, lean_object* v___y_581_){
_start:
{
lean_object* v_res_582_; 
v_res_582_ = lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__1(v_e_575_, v___y_576_, v___y_577_, v___y_578_, v___y_579_, v___y_580_);
lean_dec(v___y_580_);
lean_dec_ref(v___y_579_);
lean_dec(v___y_578_);
lean_dec_ref(v___y_577_);
return v_res_582_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__2___redArg(lean_object* v_e_583_, lean_object* v___y_584_){
_start:
{
uint8_t v___x_586_; 
v___x_586_ = l_Lean_Expr_hasMVar(v_e_583_);
if (v___x_586_ == 0)
{
lean_object* v___x_587_; 
v___x_587_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_587_, 0, v_e_583_);
return v___x_587_;
}
else
{
lean_object* v___x_588_; lean_object* v_mctx_589_; lean_object* v___x_590_; lean_object* v_fst_591_; lean_object* v_snd_592_; lean_object* v___x_593_; lean_object* v_cache_594_; lean_object* v_zetaDeltaFVarIds_595_; lean_object* v_postponed_596_; lean_object* v_diag_597_; lean_object* v___x_599_; uint8_t v_isShared_600_; uint8_t v_isSharedCheck_606_; 
v___x_588_ = lean_st_ref_get(v___y_584_);
v_mctx_589_ = lean_ctor_get(v___x_588_, 0);
lean_inc_ref(v_mctx_589_);
lean_dec(v___x_588_);
v___x_590_ = l_Lean_instantiateMVarsCore(v_mctx_589_, v_e_583_);
v_fst_591_ = lean_ctor_get(v___x_590_, 0);
lean_inc(v_fst_591_);
v_snd_592_ = lean_ctor_get(v___x_590_, 1);
lean_inc(v_snd_592_);
lean_dec_ref(v___x_590_);
v___x_593_ = lean_st_ref_take(v___y_584_);
v_cache_594_ = lean_ctor_get(v___x_593_, 1);
v_zetaDeltaFVarIds_595_ = lean_ctor_get(v___x_593_, 2);
v_postponed_596_ = lean_ctor_get(v___x_593_, 3);
v_diag_597_ = lean_ctor_get(v___x_593_, 4);
v_isSharedCheck_606_ = !lean_is_exclusive(v___x_593_);
if (v_isSharedCheck_606_ == 0)
{
lean_object* v_unused_607_; 
v_unused_607_ = lean_ctor_get(v___x_593_, 0);
lean_dec(v_unused_607_);
v___x_599_ = v___x_593_;
v_isShared_600_ = v_isSharedCheck_606_;
goto v_resetjp_598_;
}
else
{
lean_inc(v_diag_597_);
lean_inc(v_postponed_596_);
lean_inc(v_zetaDeltaFVarIds_595_);
lean_inc(v_cache_594_);
lean_dec(v___x_593_);
v___x_599_ = lean_box(0);
v_isShared_600_ = v_isSharedCheck_606_;
goto v_resetjp_598_;
}
v_resetjp_598_:
{
lean_object* v___x_602_; 
if (v_isShared_600_ == 0)
{
lean_ctor_set(v___x_599_, 0, v_snd_592_);
v___x_602_ = v___x_599_;
goto v_reusejp_601_;
}
else
{
lean_object* v_reuseFailAlloc_605_; 
v_reuseFailAlloc_605_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_605_, 0, v_snd_592_);
lean_ctor_set(v_reuseFailAlloc_605_, 1, v_cache_594_);
lean_ctor_set(v_reuseFailAlloc_605_, 2, v_zetaDeltaFVarIds_595_);
lean_ctor_set(v_reuseFailAlloc_605_, 3, v_postponed_596_);
lean_ctor_set(v_reuseFailAlloc_605_, 4, v_diag_597_);
v___x_602_ = v_reuseFailAlloc_605_;
goto v_reusejp_601_;
}
v_reusejp_601_:
{
lean_object* v___x_603_; lean_object* v___x_604_; 
v___x_603_ = lean_st_ref_set(v___y_584_, v___x_602_);
v___x_604_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_604_, 0, v_fst_591_);
return v___x_604_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__2___redArg___boxed(lean_object* v_e_608_, lean_object* v___y_609_, lean_object* v___y_610_){
_start:
{
lean_object* v_res_611_; 
v_res_611_ = lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__2___redArg(v_e_608_, v___y_609_);
lean_dec(v___y_609_);
return v_res_611_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__2(lean_object* v_e_612_, lean_object* v___y_613_, lean_object* v___y_614_, lean_object* v___y_615_, lean_object* v___y_616_, lean_object* v___y_617_, lean_object* v___y_618_){
_start:
{
lean_object* v___x_620_; 
v___x_620_ = lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__2___redArg(v_e_612_, v___y_616_);
return v___x_620_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__2___boxed(lean_object* v_e_621_, lean_object* v___y_622_, lean_object* v___y_623_, lean_object* v___y_624_, lean_object* v___y_625_, lean_object* v___y_626_, lean_object* v___y_627_, lean_object* v___y_628_){
_start:
{
lean_object* v_res_629_; 
v_res_629_ = lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__2(v_e_621_, v___y_622_, v___y_623_, v___y_624_, v___y_625_, v___y_626_, v___y_627_);
lean_dec(v___y_627_);
lean_dec_ref(v___y_626_);
lean_dec(v___y_625_);
lean_dec_ref(v___y_624_);
lean_dec(v___y_623_);
lean_dec_ref(v___y_622_);
return v_res_629_;
}
}
static lean_object* _init_lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_632_; 
v___x_630_ = lean_box(0);
v___x_631_ = l_Lean_Elab_abortTermExceptionId;
v___x_632_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_632_, 0, v___x_631_);
lean_ctor_set(v___x_632_, 1, v___x_630_);
return v___x_632_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__3___redArg(){
_start:
{
lean_object* v___x_634_; lean_object* v___x_635_; 
v___x_634_ = lean_obj_once(&lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__3___redArg___closed__0, &lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__3___redArg___closed__0_once, _init_lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__3___redArg___closed__0);
v___x_635_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_635_, 0, v___x_634_);
return v___x_635_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__3___redArg___boxed(lean_object* v___y_636_){
_start:
{
lean_object* v_res_637_; 
v_res_637_ = lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__3___redArg();
return v_res_637_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__3(lean_object* v_00_u03b1_638_, lean_object* v___y_639_, lean_object* v___y_640_, lean_object* v___y_641_, lean_object* v___y_642_, lean_object* v___y_643_, lean_object* v___y_644_){
_start:
{
lean_object* v___x_646_; 
v___x_646_ = lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__3___redArg();
return v___x_646_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__3___boxed(lean_object* v_00_u03b1_647_, lean_object* v___y_648_, lean_object* v___y_649_, lean_object* v___y_650_, lean_object* v___y_651_, lean_object* v___y_652_, lean_object* v___y_653_, lean_object* v___y_654_){
_start:
{
lean_object* v_res_655_; 
v_res_655_ = lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__3(v_00_u03b1_647_, v___y_648_, v___y_649_, v___y_650_, v___y_651_, v___y_652_, v___y_653_);
lean_dec(v___y_653_);
lean_dec_ref(v___y_652_);
lean_dec(v___y_651_);
lean_dec_ref(v___y_650_);
lean_dec(v___y_649_);
lean_dec_ref(v___y_648_);
return v_res_655_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__4___redArg___lam__0(lean_object* v_x_656_, lean_object* v___y_657_, lean_object* v___y_658_, lean_object* v___y_659_, lean_object* v___y_660_, lean_object* v___y_661_, lean_object* v___y_662_){
_start:
{
lean_object* v___x_664_; 
lean_inc(v___y_658_);
lean_inc_ref(v___y_657_);
v___x_664_ = lean_apply_7(v_x_656_, v___y_657_, v___y_658_, v___y_659_, v___y_660_, v___y_661_, v___y_662_, lean_box(0));
return v___x_664_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__4___redArg___lam__0___boxed(lean_object* v_x_665_, lean_object* v___y_666_, lean_object* v___y_667_, lean_object* v___y_668_, lean_object* v___y_669_, lean_object* v___y_670_, lean_object* v___y_671_, lean_object* v___y_672_){
_start:
{
lean_object* v_res_673_; 
v_res_673_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__4___redArg___lam__0(v_x_665_, v___y_666_, v___y_667_, v___y_668_, v___y_669_, v___y_670_, v___y_671_);
lean_dec(v___y_667_);
lean_dec_ref(v___y_666_);
return v_res_673_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__4___redArg(lean_object* v_lctx_674_, lean_object* v_localInsts_675_, lean_object* v_x_676_, lean_object* v___y_677_, lean_object* v___y_678_, lean_object* v___y_679_, lean_object* v___y_680_, lean_object* v___y_681_, lean_object* v___y_682_){
_start:
{
lean_object* v___f_684_; lean_object* v___x_685_; 
lean_inc(v___y_678_);
lean_inc_ref(v___y_677_);
v___f_684_ = lean_alloc_closure((void*)(lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__4___redArg___lam__0___boxed), 8, 3);
lean_closure_set(v___f_684_, 0, v_x_676_);
lean_closure_set(v___f_684_, 1, v___y_677_);
lean_closure_set(v___f_684_, 2, v___y_678_);
v___x_685_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_box(0), v_lctx_674_, v_localInsts_675_, v___f_684_, v___y_679_, v___y_680_, v___y_681_, v___y_682_);
if (lean_obj_tag(v___x_685_) == 0)
{
return v___x_685_;
}
else
{
lean_object* v_a_686_; lean_object* v___x_688_; uint8_t v_isShared_689_; uint8_t v_isSharedCheck_693_; 
v_a_686_ = lean_ctor_get(v___x_685_, 0);
v_isSharedCheck_693_ = !lean_is_exclusive(v___x_685_);
if (v_isSharedCheck_693_ == 0)
{
v___x_688_ = v___x_685_;
v_isShared_689_ = v_isSharedCheck_693_;
goto v_resetjp_687_;
}
else
{
lean_inc(v_a_686_);
lean_dec(v___x_685_);
v___x_688_ = lean_box(0);
v_isShared_689_ = v_isSharedCheck_693_;
goto v_resetjp_687_;
}
v_resetjp_687_:
{
lean_object* v___x_691_; 
if (v_isShared_689_ == 0)
{
v___x_691_ = v___x_688_;
goto v_reusejp_690_;
}
else
{
lean_object* v_reuseFailAlloc_692_; 
v_reuseFailAlloc_692_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_692_, 0, v_a_686_);
v___x_691_ = v_reuseFailAlloc_692_;
goto v_reusejp_690_;
}
v_reusejp_690_:
{
return v___x_691_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__4___redArg___boxed(lean_object* v_lctx_694_, lean_object* v_localInsts_695_, lean_object* v_x_696_, lean_object* v___y_697_, lean_object* v___y_698_, lean_object* v___y_699_, lean_object* v___y_700_, lean_object* v___y_701_, lean_object* v___y_702_, lean_object* v___y_703_){
_start:
{
lean_object* v_res_704_; 
v_res_704_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__4___redArg(v_lctx_694_, v_localInsts_695_, v_x_696_, v___y_697_, v___y_698_, v___y_699_, v___y_700_, v___y_701_, v___y_702_);
lean_dec(v___y_702_);
lean_dec_ref(v___y_701_);
lean_dec(v___y_700_);
lean_dec_ref(v___y_699_);
lean_dec(v___y_698_);
lean_dec_ref(v___y_697_);
return v_res_704_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__4(lean_object* v_00_u03b1_705_, lean_object* v_lctx_706_, lean_object* v_localInsts_707_, lean_object* v_x_708_, lean_object* v___y_709_, lean_object* v___y_710_, lean_object* v___y_711_, lean_object* v___y_712_, lean_object* v___y_713_, lean_object* v___y_714_){
_start:
{
lean_object* v___x_716_; 
v___x_716_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__4___redArg(v_lctx_706_, v_localInsts_707_, v_x_708_, v___y_709_, v___y_710_, v___y_711_, v___y_712_, v___y_713_, v___y_714_);
return v___x_716_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__4___boxed(lean_object* v_00_u03b1_717_, lean_object* v_lctx_718_, lean_object* v_localInsts_719_, lean_object* v_x_720_, lean_object* v___y_721_, lean_object* v___y_722_, lean_object* v___y_723_, lean_object* v___y_724_, lean_object* v___y_725_, lean_object* v___y_726_, lean_object* v___y_727_){
_start:
{
lean_object* v_res_728_; 
v_res_728_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__4(v_00_u03b1_717_, v_lctx_718_, v_localInsts_719_, v_x_720_, v___y_721_, v___y_722_, v___y_723_, v___y_724_, v___y_725_, v___y_726_);
lean_dec(v___y_726_);
lean_dec_ref(v___y_725_);
lean_dec(v___y_724_);
lean_dec_ref(v___y_723_);
lean_dec(v___y_722_);
lean_dec_ref(v___y_721_);
return v_res_728_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__0(lean_object* v_snd_729_, lean_object* v_fst_730_, lean_object* v_quotedGoal_731_, lean_object* v___y_732_, lean_object* v___y_733_, lean_object* v___y_734_, lean_object* v___y_735_, lean_object* v___y_736_){
_start:
{
lean_object* v___x_738_; lean_object* v___x_739_; lean_object* v___x_740_; lean_object* v___x_741_; 
v___x_738_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_738_, 0, v_snd_729_);
lean_ctor_set(v___x_738_, 1, v_quotedGoal_731_);
v___x_739_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_739_, 0, v_fst_730_);
lean_ctor_set(v___x_739_, 1, v___x_738_);
v___x_740_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_740_, 0, v___x_739_);
lean_ctor_set(v___x_740_, 1, v___y_732_);
v___x_741_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_741_, 0, v___x_740_);
return v___x_741_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__0___boxed(lean_object* v_snd_742_, lean_object* v_fst_743_, lean_object* v_quotedGoal_744_, lean_object* v___y_745_, lean_object* v___y_746_, lean_object* v___y_747_, lean_object* v___y_748_, lean_object* v___y_749_, lean_object* v___y_750_){
_start:
{
lean_object* v_res_751_; 
v_res_751_ = lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__0(v_snd_742_, v_fst_743_, v_quotedGoal_744_, v___y_745_, v___y_746_, v___y_747_, v___y_748_, v___y_749_);
lean_dec(v___y_749_);
lean_dec_ref(v___y_748_);
lean_dec(v___y_747_);
lean_dec_ref(v___y_746_);
return v_res_751_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__1(uint8_t v___x_753_, lean_object* v___x_754_, lean_object* v___x_755_, lean_object* v___x_756_, lean_object* v___x_757_, lean_object* v_snd_758_, uint8_t v___x_759_, lean_object* v_fst_760_, lean_object* v___y_761_, lean_object* v___y_762_, lean_object* v___y_763_, lean_object* v___y_764_, lean_object* v___y_765_, lean_object* v___y_766_){
_start:
{
lean_object* v_ref_768_; lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_771_; lean_object* v___x_772_; lean_object* v___x_773_; lean_object* v___x_774_; lean_object* v___x_775_; lean_object* v___x_776_; 
v_ref_768_ = lean_ctor_get(v___y_765_, 5);
v___x_769_ = l_Lean_SourceInfo_fromRef(v_ref_768_, v___x_753_);
v___x_770_ = ((lean_object*)(lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__1___closed__0));
v___x_771_ = l_Lean_Name_mkStr4(v___x_754_, v___x_755_, v___x_756_, v___x_770_);
lean_inc(v___x_769_);
v___x_772_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_772_, 0, v___x_769_);
lean_ctor_set(v___x_772_, 1, v___x_770_);
v___x_773_ = l_Lean_Syntax_node2(v___x_769_, v___x_771_, v___x_772_, v___x_757_);
v___x_774_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_774_, 0, v_snd_758_);
v___x_775_ = lean_box(0);
v___x_776_ = l_Lean_Elab_Term_elabTermEnsuringType(v___x_773_, v___x_774_, v___x_759_, v___x_759_, v___x_775_, v___y_761_, v___y_762_, v___y_763_, v___y_764_, v___y_765_, v___y_766_);
if (lean_obj_tag(v___x_776_) == 0)
{
lean_object* v_a_777_; lean_object* v___x_778_; 
v_a_777_ = lean_ctor_get(v___x_776_, 0);
lean_inc(v_a_777_);
lean_dec_ref_known(v___x_776_, 1);
v___x_778_ = l_Lean_Elab_Term_synthesizeSyntheticMVarsNoPostponing(v___x_753_, v___y_761_, v___y_762_, v___y_763_, v___y_764_, v___y_765_, v___y_766_);
if (lean_obj_tag(v___x_778_) == 0)
{
lean_object* v___x_779_; lean_object* v_a_780_; lean_object* v___x_781_; 
lean_dec_ref_known(v___x_778_, 1);
v___x_779_ = lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__2___redArg(v_a_777_, v___y_764_);
v_a_780_ = lean_ctor_get(v___x_779_, 0);
lean_inc_n(v_a_780_, 2);
lean_dec_ref(v___x_779_);
v___x_781_ = l_Lean_Meta_getMVars(v_a_780_, v___y_763_, v___y_764_, v___y_765_, v___y_766_);
if (lean_obj_tag(v___x_781_) == 0)
{
lean_object* v_a_782_; lean_object* v___x_783_; 
v_a_782_ = lean_ctor_get(v___x_781_, 0);
lean_inc(v_a_782_);
lean_dec_ref_known(v___x_781_, 1);
v___x_783_ = l_Lean_Elab_Term_logUnassignedUsingErrorInfos(v_a_782_, v___x_775_, v___y_761_, v___y_762_, v___y_763_, v___y_764_, v___y_765_, v___y_766_);
lean_dec(v_a_782_);
if (lean_obj_tag(v___x_783_) == 0)
{
lean_object* v_a_784_; uint8_t v___x_785_; 
v_a_784_ = lean_ctor_get(v___x_783_, 0);
lean_inc(v_a_784_);
lean_dec_ref_known(v___x_783_, 1);
v___x_785_ = lean_unbox(v_a_784_);
lean_dec(v_a_784_);
if (v___x_785_ == 0)
{
lean_object* v___x_786_; 
v___x_786_ = lp_Qq___private_Qq_Commands_0__Qq_mkLetFVarsFromValues(v_fst_760_, v_a_780_, v___y_763_, v___y_764_, v___y_765_, v___y_766_);
return v___x_786_;
}
else
{
lean_object* v___x_787_; lean_object* v_a_788_; lean_object* v___x_790_; uint8_t v_isShared_791_; uint8_t v_isSharedCheck_795_; 
lean_dec(v_a_780_);
v___x_787_ = lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__3___redArg();
v_a_788_ = lean_ctor_get(v___x_787_, 0);
v_isSharedCheck_795_ = !lean_is_exclusive(v___x_787_);
if (v_isSharedCheck_795_ == 0)
{
v___x_790_ = v___x_787_;
v_isShared_791_ = v_isSharedCheck_795_;
goto v_resetjp_789_;
}
else
{
lean_inc(v_a_788_);
lean_dec(v___x_787_);
v___x_790_ = lean_box(0);
v_isShared_791_ = v_isSharedCheck_795_;
goto v_resetjp_789_;
}
v_resetjp_789_:
{
lean_object* v___x_793_; 
if (v_isShared_791_ == 0)
{
v___x_793_ = v___x_790_;
goto v_reusejp_792_;
}
else
{
lean_object* v_reuseFailAlloc_794_; 
v_reuseFailAlloc_794_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_794_, 0, v_a_788_);
v___x_793_ = v_reuseFailAlloc_794_;
goto v_reusejp_792_;
}
v_reusejp_792_:
{
return v___x_793_;
}
}
}
}
else
{
lean_object* v_a_796_; lean_object* v___x_798_; uint8_t v_isShared_799_; uint8_t v_isSharedCheck_803_; 
lean_dec(v_a_780_);
v_a_796_ = lean_ctor_get(v___x_783_, 0);
v_isSharedCheck_803_ = !lean_is_exclusive(v___x_783_);
if (v_isSharedCheck_803_ == 0)
{
v___x_798_ = v___x_783_;
v_isShared_799_ = v_isSharedCheck_803_;
goto v_resetjp_797_;
}
else
{
lean_inc(v_a_796_);
lean_dec(v___x_783_);
v___x_798_ = lean_box(0);
v_isShared_799_ = v_isSharedCheck_803_;
goto v_resetjp_797_;
}
v_resetjp_797_:
{
lean_object* v___x_801_; 
if (v_isShared_799_ == 0)
{
v___x_801_ = v___x_798_;
goto v_reusejp_800_;
}
else
{
lean_object* v_reuseFailAlloc_802_; 
v_reuseFailAlloc_802_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_802_, 0, v_a_796_);
v___x_801_ = v_reuseFailAlloc_802_;
goto v_reusejp_800_;
}
v_reusejp_800_:
{
return v___x_801_;
}
}
}
}
else
{
lean_object* v_a_804_; lean_object* v___x_806_; uint8_t v_isShared_807_; uint8_t v_isSharedCheck_811_; 
lean_dec(v_a_780_);
v_a_804_ = lean_ctor_get(v___x_781_, 0);
v_isSharedCheck_811_ = !lean_is_exclusive(v___x_781_);
if (v_isSharedCheck_811_ == 0)
{
v___x_806_ = v___x_781_;
v_isShared_807_ = v_isSharedCheck_811_;
goto v_resetjp_805_;
}
else
{
lean_inc(v_a_804_);
lean_dec(v___x_781_);
v___x_806_ = lean_box(0);
v_isShared_807_ = v_isSharedCheck_811_;
goto v_resetjp_805_;
}
v_resetjp_805_:
{
lean_object* v___x_809_; 
if (v_isShared_807_ == 0)
{
v___x_809_ = v___x_806_;
goto v_reusejp_808_;
}
else
{
lean_object* v_reuseFailAlloc_810_; 
v_reuseFailAlloc_810_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_810_, 0, v_a_804_);
v___x_809_ = v_reuseFailAlloc_810_;
goto v_reusejp_808_;
}
v_reusejp_808_:
{
return v___x_809_;
}
}
}
}
else
{
lean_object* v_a_812_; lean_object* v___x_814_; uint8_t v_isShared_815_; uint8_t v_isSharedCheck_819_; 
lean_dec(v_a_777_);
v_a_812_ = lean_ctor_get(v___x_778_, 0);
v_isSharedCheck_819_ = !lean_is_exclusive(v___x_778_);
if (v_isSharedCheck_819_ == 0)
{
v___x_814_ = v___x_778_;
v_isShared_815_ = v_isSharedCheck_819_;
goto v_resetjp_813_;
}
else
{
lean_inc(v_a_812_);
lean_dec(v___x_778_);
v___x_814_ = lean_box(0);
v_isShared_815_ = v_isSharedCheck_819_;
goto v_resetjp_813_;
}
v_resetjp_813_:
{
lean_object* v___x_817_; 
if (v_isShared_815_ == 0)
{
v___x_817_ = v___x_814_;
goto v_reusejp_816_;
}
else
{
lean_object* v_reuseFailAlloc_818_; 
v_reuseFailAlloc_818_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_818_, 0, v_a_812_);
v___x_817_ = v_reuseFailAlloc_818_;
goto v_reusejp_816_;
}
v_reusejp_816_:
{
return v___x_817_;
}
}
}
}
else
{
return v___x_776_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__1___boxed(lean_object* v___x_820_, lean_object* v___x_821_, lean_object* v___x_822_, lean_object* v___x_823_, lean_object* v___x_824_, lean_object* v_snd_825_, lean_object* v___x_826_, lean_object* v_fst_827_, lean_object* v___y_828_, lean_object* v___y_829_, lean_object* v___y_830_, lean_object* v___y_831_, lean_object* v___y_832_, lean_object* v___y_833_, lean_object* v___y_834_){
_start:
{
uint8_t v___x_12215__boxed_835_; uint8_t v___x_12221__boxed_836_; lean_object* v_res_837_; 
v___x_12215__boxed_835_ = lean_unbox(v___x_820_);
v___x_12221__boxed_836_ = lean_unbox(v___x_826_);
v_res_837_ = lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__1(v___x_12215__boxed_835_, v___x_821_, v___x_822_, v___x_823_, v___x_824_, v_snd_825_, v___x_12221__boxed_836_, v_fst_827_, v___y_828_, v___y_829_, v___y_830_, v___y_831_, v___y_832_, v___y_833_);
lean_dec(v___y_833_);
lean_dec_ref(v___y_832_);
lean_dec(v___y_831_);
lean_dec_ref(v___y_830_);
lean_dec(v___y_829_);
lean_dec_ref(v___y_828_);
lean_dec_ref(v_fst_827_);
return v_res_837_;
}
}
static lean_object* _init_lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__0(void){
_start:
{
lean_object* v___x_838_; lean_object* v___x_839_; lean_object* v___x_840_; 
v___x_838_ = lean_box(0);
v___x_839_ = lean_unsigned_to_nat(16u);
v___x_840_ = lean_mk_array(v___x_839_, v___x_838_);
return v___x_840_;
}
}
static lean_object* _init_lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__1(void){
_start:
{
lean_object* v___x_841_; lean_object* v___x_842_; lean_object* v___x_843_; 
v___x_841_ = lean_obj_once(&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__0, &lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__0_once, _init_lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__0);
v___x_842_ = lean_unsigned_to_nat(0u);
v___x_843_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_843_, 0, v___x_842_);
lean_ctor_set(v___x_843_, 1, v___x_841_);
return v___x_843_;
}
}
static lean_object* _init_lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__3(void){
_start:
{
uint8_t v___x_846_; lean_object* v___x_847_; lean_object* v___x_848_; lean_object* v___x_849_; lean_object* v___x_850_; lean_object* v___x_851_; 
v___x_846_ = 0;
v___x_847_ = ((lean_object*)(lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__2));
v___x_848_ = l_Lean_LocalContext_empty;
v___x_849_ = lean_obj_once(&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__1, &lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__1_once, _init_lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__1);
v___x_850_ = lean_box(0);
v___x_851_ = lean_alloc_ctor(0, 8, 1);
lean_ctor_set(v___x_851_, 0, v___x_850_);
lean_ctor_set(v___x_851_, 1, v___x_849_);
lean_ctor_set(v___x_851_, 2, v___x_849_);
lean_ctor_set(v___x_851_, 3, v___x_848_);
lean_ctor_set(v___x_851_, 4, v___x_849_);
lean_ctor_set(v___x_851_, 5, v___x_849_);
lean_ctor_set(v___x_851_, 6, v___x_847_);
lean_ctor_set(v___x_851_, 7, v___x_850_);
lean_ctor_set_uint8(v___x_851_, sizeof(void*)*8, v___x_846_);
return v___x_851_;
}
}
static lean_object* _init_lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__5(void){
_start:
{
lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v___x_855_; 
v___x_853_ = lean_box(0);
v___x_854_ = ((lean_object*)(lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__4));
v___x_855_ = l_Lean_Expr_const___override(v___x_854_, v___x_853_);
return v___x_855_;
}
}
static lean_object* _init_lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__6(void){
_start:
{
lean_object* v___x_856_; lean_object* v___x_857_; lean_object* v___x_858_; 
v___x_856_ = lean_box(0);
v___x_857_ = ((lean_object*)(lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__7));
v___x_858_ = l_Lean_Expr_const___override(v___x_857_, v___x_856_);
return v___x_858_;
}
}
static lean_object* _init_lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__7(void){
_start:
{
lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_861_; 
v___x_859_ = lean_obj_once(&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__6, &lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__6_once, _init_lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__6);
v___x_860_ = lean_obj_once(&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__5, &lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__5_once, _init_lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__5);
v___x_861_ = l_Lean_Expr_app___override(v___x_860_, v___x_859_);
return v___x_861_;
}
}
static lean_object* _init_lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__10(void){
_start:
{
lean_object* v___x_866_; lean_object* v___x_867_; lean_object* v___x_868_; 
v___x_866_ = lean_box(0);
v___x_867_ = ((lean_object*)(lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__9));
v___x_868_ = l_Lean_Expr_const___override(v___x_867_, v___x_866_);
return v___x_868_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2(lean_object* v_stx_869_, lean_object* v_expectedType_870_, lean_object* v___y_871_, lean_object* v___y_872_, lean_object* v___y_873_, lean_object* v___y_874_, lean_object* v___y_875_, lean_object* v___y_876_){
_start:
{
lean_object* v___x_878_; uint8_t v___x_879_; 
v___x_878_ = ((lean_object*)(lp_Qq_Qq_termBy__elabq___00__closed__2));
lean_inc(v_stx_869_);
v___x_879_ = l_Lean_Syntax_isOfKind(v_stx_869_, v___x_878_);
if (v___x_879_ == 0)
{
lean_object* v___x_880_; 
lean_dec_ref(v_expectedType_870_);
lean_dec(v_stx_869_);
v___x_880_ = lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__0___redArg();
return v___x_880_;
}
else
{
lean_object* v___x_881_; 
v___x_881_ = l_Lean_Elab_Term_getLevelNames___redArg(v___y_872_);
if (lean_obj_tag(v___x_881_) == 0)
{
lean_object* v_a_882_; lean_object* v_lctx_883_; lean_object* v___x_884_; lean_object* v___x_885_; uint8_t v___x_886_; lean_object* v___x_887_; lean_object* v___x_888_; 
v_a_882_ = lean_ctor_get(v___x_881_, 0);
lean_inc(v_a_882_);
lean_dec_ref_known(v___x_881_, 1);
v_lctx_883_ = lean_ctor_get(v___y_873_, 2);
v___x_884_ = l_List_reverse___redArg(v_a_882_);
v___x_885_ = ((lean_object*)(lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__2));
v___x_886_ = 0;
v___x_887_ = lean_obj_once(&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__3, &lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__3_once, _init_lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__3);
v___x_888_ = lp_Qq_Qq_Impl_quoteLCtx(v_lctx_883_, v___x_884_, v___x_887_, v___y_873_, v___y_874_, v___y_875_, v___y_876_);
lean_dec(v___x_884_);
if (lean_obj_tag(v___x_888_) == 0)
{
lean_object* v_a_889_; lean_object* v_fst_890_; lean_object* v_snd_891_; lean_object* v_fst_892_; lean_object* v_snd_893_; lean_object* v___x_894_; lean_object* v_a_895_; lean_object* v_fst_896_; lean_object* v_snd_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; lean_object* v___x_901_; lean_object* v___x_902_; lean_object* v___y_904_; uint8_t v___x_929_; 
v_a_889_ = lean_ctor_get(v___x_888_, 0);
lean_inc(v_a_889_);
lean_dec_ref_known(v___x_888_, 1);
v_fst_890_ = lean_ctor_get(v_a_889_, 0);
lean_inc(v_fst_890_);
v_snd_891_ = lean_ctor_get(v_a_889_, 1);
lean_inc(v_snd_891_);
lean_dec(v_a_889_);
v_fst_892_ = lean_ctor_get(v_fst_890_, 0);
lean_inc(v_fst_892_);
v_snd_893_ = lean_ctor_get(v_fst_890_, 1);
lean_inc(v_snd_893_);
lean_dec(v_fst_890_);
v___x_894_ = lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__1___redArg(v_expectedType_870_, v_snd_891_, v___y_874_);
v_a_895_ = lean_ctor_get(v___x_894_, 0);
lean_inc(v_a_895_);
lean_dec_ref(v___x_894_);
v_fst_896_ = lean_ctor_get(v_a_895_, 0);
lean_inc(v_fst_896_);
v_snd_897_ = lean_ctor_get(v_a_895_, 1);
lean_inc(v_snd_897_);
lean_dec(v_a_895_);
v___x_898_ = lean_unsigned_to_nat(1u);
v___x_899_ = l_Lean_Syntax_getArg(v_stx_869_, v___x_898_);
lean_dec(v_stx_869_);
v___x_900_ = ((lean_object*)(lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__0));
v___x_901_ = ((lean_object*)(lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__4));
v___x_902_ = ((lean_object*)(lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__2));
v___x_929_ = l_Lean_Expr_hasMVar(v_fst_896_);
if (v___x_929_ == 0)
{
lean_object* v___x_930_; 
v___x_930_ = lp_Qq_Qq_Impl_quoteExpr(v_fst_896_, v_snd_897_, v___y_873_, v___y_874_, v___y_875_, v___y_876_);
if (lean_obj_tag(v___x_930_) == 0)
{
lean_object* v_a_931_; lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; lean_object* v___x_935_; lean_object* v___x_936_; 
v_a_931_ = lean_ctor_get(v___x_930_, 0);
lean_inc(v_a_931_);
lean_dec_ref_known(v___x_930_, 1);
v___x_932_ = lean_obj_once(&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__5, &lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__5_once, _init_lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__5);
v___x_933_ = lean_obj_once(&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__10, &lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__10_once, _init_lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__10);
v___x_934_ = l_Lean_Expr_app___override(v___x_933_, v_a_931_);
v___x_935_ = l_Lean_Expr_app___override(v___x_932_, v___x_934_);
v___x_936_ = lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__0(v_snd_893_, v_fst_892_, v___x_935_, v_snd_897_, v___y_873_, v___y_874_, v___y_875_, v___y_876_);
v___y_904_ = v___x_936_;
goto v___jp_903_;
}
else
{
lean_dec(v___x_899_);
lean_dec(v_snd_897_);
lean_dec(v_snd_893_);
lean_dec(v_fst_892_);
return v___x_930_;
}
}
else
{
lean_object* v___x_937_; lean_object* v___x_938_; 
lean_dec(v_fst_896_);
v___x_937_ = lean_obj_once(&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__7, &lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__7_once, _init_lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__7);
v___x_938_ = lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__0(v_snd_893_, v_fst_892_, v___x_937_, v_snd_897_, v___y_873_, v___y_874_, v___y_875_, v___y_876_);
v___y_904_ = v___x_938_;
goto v___jp_903_;
}
v___jp_903_:
{
lean_object* v_a_905_; lean_object* v_fst_906_; lean_object* v_snd_907_; lean_object* v_fst_908_; lean_object* v_fst_909_; lean_object* v_snd_910_; lean_object* v___x_911_; lean_object* v___x_912_; lean_object* v___f_913_; lean_object* v___x_914_; 
v_a_905_ = lean_ctor_get(v___y_904_, 0);
lean_inc(v_a_905_);
lean_dec_ref(v___y_904_);
v_fst_906_ = lean_ctor_get(v_a_905_, 0);
lean_inc(v_fst_906_);
lean_dec(v_a_905_);
v_snd_907_ = lean_ctor_get(v_fst_906_, 1);
lean_inc(v_snd_907_);
v_fst_908_ = lean_ctor_get(v_fst_906_, 0);
lean_inc(v_fst_908_);
lean_dec(v_fst_906_);
v_fst_909_ = lean_ctor_get(v_snd_907_, 0);
lean_inc(v_fst_909_);
v_snd_910_ = lean_ctor_get(v_snd_907_, 1);
lean_inc(v_snd_910_);
lean_dec(v_snd_907_);
v___x_911_ = lean_box(v___x_886_);
v___x_912_ = lean_box(v___x_879_);
v___f_913_ = lean_alloc_closure((void*)(lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__1___boxed), 15, 8);
lean_closure_set(v___f_913_, 0, v___x_911_);
lean_closure_set(v___f_913_, 1, v___x_900_);
lean_closure_set(v___f_913_, 2, v___x_901_);
lean_closure_set(v___f_913_, 3, v___x_902_);
lean_closure_set(v___f_913_, 4, v___x_899_);
lean_closure_set(v___f_913_, 5, v_snd_910_);
lean_closure_set(v___f_913_, 6, v___x_912_);
lean_closure_set(v___f_913_, 7, v_fst_909_);
v___x_914_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__4___redArg(v_fst_908_, v___x_885_, v___f_913_, v___y_871_, v___y_872_, v___y_873_, v___y_874_, v___y_875_, v___y_876_);
if (lean_obj_tag(v___x_914_) == 0)
{
lean_object* v_a_915_; lean_object* v___x_916_; uint8_t v___x_917_; lean_object* v___x_918_; 
v_a_915_ = lean_ctor_get(v___x_914_, 0);
lean_inc(v_a_915_);
lean_dec_ref_known(v___x_914_, 1);
v___x_916_ = lean_obj_once(&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__7, &lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__7_once, _init_lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__7);
v___x_917_ = 1;
v___x_918_ = l_Lean_Meta_evalExpr___redArg(v___x_916_, v_a_915_, v___x_917_, v___x_879_, v___y_873_, v___y_874_, v___y_875_, v___y_876_);
if (lean_obj_tag(v___x_918_) == 0)
{
lean_object* v_a_919_; lean_object* v___x_920_; 
v_a_919_ = lean_ctor_get(v___x_918_, 0);
lean_inc(v_a_919_);
lean_dec_ref_known(v___x_918_, 1);
lean_inc(v___y_876_);
lean_inc_ref(v___y_875_);
lean_inc(v___y_874_);
lean_inc_ref(v___y_873_);
lean_inc(v___y_872_);
lean_inc_ref(v___y_871_);
v___x_920_ = lean_apply_7(v_a_919_, v___y_871_, v___y_872_, v___y_873_, v___y_874_, v___y_875_, v___y_876_, lean_box(0));
return v___x_920_;
}
else
{
lean_object* v_a_921_; lean_object* v___x_923_; uint8_t v_isShared_924_; uint8_t v_isSharedCheck_928_; 
v_a_921_ = lean_ctor_get(v___x_918_, 0);
v_isSharedCheck_928_ = !lean_is_exclusive(v___x_918_);
if (v_isSharedCheck_928_ == 0)
{
v___x_923_ = v___x_918_;
v_isShared_924_ = v_isSharedCheck_928_;
goto v_resetjp_922_;
}
else
{
lean_inc(v_a_921_);
lean_dec(v___x_918_);
v___x_923_ = lean_box(0);
v_isShared_924_ = v_isSharedCheck_928_;
goto v_resetjp_922_;
}
v_resetjp_922_:
{
lean_object* v___x_926_; 
if (v_isShared_924_ == 0)
{
v___x_926_ = v___x_923_;
goto v_reusejp_925_;
}
else
{
lean_object* v_reuseFailAlloc_927_; 
v_reuseFailAlloc_927_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_927_, 0, v_a_921_);
v___x_926_ = v_reuseFailAlloc_927_;
goto v_reusejp_925_;
}
v_reusejp_925_:
{
return v___x_926_;
}
}
}
}
else
{
return v___x_914_;
}
}
}
else
{
lean_object* v_a_939_; lean_object* v___x_941_; uint8_t v_isShared_942_; uint8_t v_isSharedCheck_946_; 
lean_dec_ref(v_expectedType_870_);
lean_dec(v_stx_869_);
v_a_939_ = lean_ctor_get(v___x_888_, 0);
v_isSharedCheck_946_ = !lean_is_exclusive(v___x_888_);
if (v_isSharedCheck_946_ == 0)
{
v___x_941_ = v___x_888_;
v_isShared_942_ = v_isSharedCheck_946_;
goto v_resetjp_940_;
}
else
{
lean_inc(v_a_939_);
lean_dec(v___x_888_);
v___x_941_ = lean_box(0);
v_isShared_942_ = v_isSharedCheck_946_;
goto v_resetjp_940_;
}
v_resetjp_940_:
{
lean_object* v___x_944_; 
if (v_isShared_942_ == 0)
{
v___x_944_ = v___x_941_;
goto v_reusejp_943_;
}
else
{
lean_object* v_reuseFailAlloc_945_; 
v_reuseFailAlloc_945_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_945_, 0, v_a_939_);
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
lean_dec_ref(v_expectedType_870_);
lean_dec(v_stx_869_);
v_a_947_ = lean_ctor_get(v___x_881_, 0);
v_isSharedCheck_954_ = !lean_is_exclusive(v___x_881_);
if (v_isSharedCheck_954_ == 0)
{
v___x_949_ = v___x_881_;
v_isShared_950_ = v_isSharedCheck_954_;
goto v_resetjp_948_;
}
else
{
lean_inc(v_a_947_);
lean_dec(v___x_881_);
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
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___boxed(lean_object* v_stx_955_, lean_object* v_expectedType_956_, lean_object* v___y_957_, lean_object* v___y_958_, lean_object* v___y_959_, lean_object* v___y_960_, lean_object* v___y_961_, lean_object* v___y_962_, lean_object* v___y_963_){
_start:
{
lean_object* v_res_964_; 
v_res_964_ = lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2(v_stx_955_, v_expectedType_956_, v___y_957_, v___y_958_, v___y_959_, v___y_960_, v___y_961_, v___y_962_);
lean_dec(v___y_962_);
lean_dec_ref(v___y_961_);
lean_dec(v___y_960_);
lean_dec_ref(v___y_959_);
lean_dec(v___y_958_);
lean_dec_ref(v___y_957_);
return v_res_964_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1(lean_object* v_stx_965_, lean_object* v_expectedType_x3f_966_, lean_object* v_a_967_, lean_object* v_a_968_, lean_object* v_a_969_, lean_object* v_a_970_, lean_object* v_a_971_, lean_object* v_a_972_){
_start:
{
lean_object* v___f_974_; lean_object* v___x_975_; 
v___f_974_ = lean_alloc_closure((void*)(lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___boxed), 9, 1);
lean_closure_set(v___f_974_, 0, v_stx_965_);
v___x_975_ = l_Lean_Elab_Term_withExpectedType(v_expectedType_x3f_966_, v___f_974_, v_a_967_, v_a_968_, v_a_969_, v_a_970_, v_a_971_, v_a_972_);
return v___x_975_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___boxed(lean_object* v_stx_976_, lean_object* v_expectedType_x3f_977_, lean_object* v_a_978_, lean_object* v_a_979_, lean_object* v_a_980_, lean_object* v_a_981_, lean_object* v_a_982_, lean_object* v_a_983_, lean_object* v_a_984_){
_start:
{
lean_object* v_res_985_; 
v_res_985_ = lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1(v_stx_976_, v_expectedType_x3f_977_, v_a_978_, v_a_979_, v_a_980_, v_a_981_, v_a_982_, v_a_983_);
lean_dec(v_a_983_);
lean_dec_ref(v_a_982_);
lean_dec(v_a_981_);
lean_dec_ref(v_a_980_);
lean_dec(v_a_979_);
lean_dec_ref(v_a_978_);
return v_res_985_;
}
}
static lean_object* _init_lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__3(void){
_start:
{
lean_object* v___x_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; 
v___x_1038_ = lean_box(0);
v___x_1039_ = ((lean_object*)(lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__2));
v___x_1040_ = l_Lean_Expr_const___override(v___x_1039_, v___x_1038_);
return v___x_1040_;
}
}
static lean_object* _init_lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__6(void){
_start:
{
lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; 
v___x_1044_ = lean_box(0);
v___x_1045_ = ((lean_object*)(lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__5));
v___x_1046_ = l_Lean_Expr_const___override(v___x_1045_, v___x_1044_);
return v___x_1046_;
}
}
static lean_object* _init_lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__7(void){
_start:
{
lean_object* v___x_1047_; lean_object* v___x_1048_; lean_object* v___x_1049_; 
v___x_1047_ = lean_obj_once(&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__6, &lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__6_once, _init_lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__6);
v___x_1048_ = lean_obj_once(&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__3, &lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__3_once, _init_lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__3);
v___x_1049_ = l_Lean_Expr_app___override(v___x_1048_, v___x_1047_);
return v___x_1049_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6(lean_object* v_codeExpr_1050_, lean_object* v_a_1051_, lean_object* v_a_1052_, lean_object* v_a_1053_, lean_object* v_a_1054_){
_start:
{
lean_object* v___x_1056_; uint8_t v___x_1057_; uint8_t v___x_1058_; lean_object* v___x_1059_; 
v___x_1056_ = lean_obj_once(&lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__7, &lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__7_once, _init_lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__7);
v___x_1057_ = 1;
v___x_1058_ = 1;
v___x_1059_ = l_Lean_Meta_evalExpr___redArg(v___x_1056_, v_codeExpr_1050_, v___x_1057_, v___x_1058_, v_a_1051_, v_a_1052_, v_a_1053_, v_a_1054_);
return v___x_1059_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___boxed(lean_object* v_codeExpr_1060_, lean_object* v_a_1061_, lean_object* v_a_1062_, lean_object* v_a_1063_, lean_object* v_a_1064_, lean_object* v_a_1065_){
_start:
{
lean_object* v_res_1066_; 
v_res_1066_ = lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6(v_codeExpr_1060_, v_a_1061_, v_a_1062_, v_a_1063_, v_a_1064_);
lean_dec(v_a_1064_);
lean_dec_ref(v_a_1063_);
lean_dec(v_a_1062_);
lean_dec_ref(v_a_1061_);
return v_res_1066_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__0___redArg(){
_start:
{
lean_object* v___x_1068_; lean_object* v___x_1069_; 
v___x_1068_ = lean_obj_once(&lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__0___redArg___closed__0, &lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__0___redArg___closed__0_once, _init_lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__0___redArg___closed__0);
v___x_1069_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1069_, 0, v___x_1068_);
return v___x_1069_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__0___redArg___boxed(lean_object* v___y_1070_){
_start:
{
lean_object* v_res_1071_; 
v_res_1071_ = lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__0___redArg();
return v_res_1071_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__0(lean_object* v_00_u03b1_1072_, lean_object* v___y_1073_, lean_object* v___y_1074_, lean_object* v___y_1075_, lean_object* v___y_1076_, lean_object* v___y_1077_, lean_object* v___y_1078_, lean_object* v___y_1079_, lean_object* v___y_1080_){
_start:
{
lean_object* v___x_1082_; 
v___x_1082_ = lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__0___redArg();
return v___x_1082_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__0___boxed(lean_object* v_00_u03b1_1083_, lean_object* v___y_1084_, lean_object* v___y_1085_, lean_object* v___y_1086_, lean_object* v___y_1087_, lean_object* v___y_1088_, lean_object* v___y_1089_, lean_object* v___y_1090_, lean_object* v___y_1091_, lean_object* v___y_1092_){
_start:
{
lean_object* v_res_1093_; 
v_res_1093_ = lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__0(v_00_u03b1_1083_, v___y_1084_, v___y_1085_, v___y_1086_, v___y_1087_, v___y_1088_, v___y_1089_, v___y_1090_, v___y_1091_);
lean_dec(v___y_1091_);
lean_dec_ref(v___y_1090_);
lean_dec(v___y_1089_);
lean_dec_ref(v___y_1088_);
lean_dec(v___y_1087_);
lean_dec_ref(v___y_1086_);
lean_dec(v___y_1085_);
lean_dec_ref(v___y_1084_);
return v_res_1093_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__1___redArg(lean_object* v_e_1094_, lean_object* v___y_1095_){
_start:
{
uint8_t v___x_1097_; 
v___x_1097_ = l_Lean_Expr_hasMVar(v_e_1094_);
if (v___x_1097_ == 0)
{
lean_object* v___x_1098_; 
v___x_1098_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1098_, 0, v_e_1094_);
return v___x_1098_;
}
else
{
lean_object* v___x_1099_; lean_object* v_mctx_1100_; lean_object* v___x_1101_; lean_object* v_fst_1102_; lean_object* v_snd_1103_; lean_object* v___x_1104_; lean_object* v_cache_1105_; lean_object* v_zetaDeltaFVarIds_1106_; lean_object* v_postponed_1107_; lean_object* v_diag_1108_; lean_object* v___x_1110_; uint8_t v_isShared_1111_; uint8_t v_isSharedCheck_1117_; 
v___x_1099_ = lean_st_ref_get(v___y_1095_);
v_mctx_1100_ = lean_ctor_get(v___x_1099_, 0);
lean_inc_ref(v_mctx_1100_);
lean_dec(v___x_1099_);
v___x_1101_ = l_Lean_instantiateMVarsCore(v_mctx_1100_, v_e_1094_);
v_fst_1102_ = lean_ctor_get(v___x_1101_, 0);
lean_inc(v_fst_1102_);
v_snd_1103_ = lean_ctor_get(v___x_1101_, 1);
lean_inc(v_snd_1103_);
lean_dec_ref(v___x_1101_);
v___x_1104_ = lean_st_ref_take(v___y_1095_);
v_cache_1105_ = lean_ctor_get(v___x_1104_, 1);
v_zetaDeltaFVarIds_1106_ = lean_ctor_get(v___x_1104_, 2);
v_postponed_1107_ = lean_ctor_get(v___x_1104_, 3);
v_diag_1108_ = lean_ctor_get(v___x_1104_, 4);
v_isSharedCheck_1117_ = !lean_is_exclusive(v___x_1104_);
if (v_isSharedCheck_1117_ == 0)
{
lean_object* v_unused_1118_; 
v_unused_1118_ = lean_ctor_get(v___x_1104_, 0);
lean_dec(v_unused_1118_);
v___x_1110_ = v___x_1104_;
v_isShared_1111_ = v_isSharedCheck_1117_;
goto v_resetjp_1109_;
}
else
{
lean_inc(v_diag_1108_);
lean_inc(v_postponed_1107_);
lean_inc(v_zetaDeltaFVarIds_1106_);
lean_inc(v_cache_1105_);
lean_dec(v___x_1104_);
v___x_1110_ = lean_box(0);
v_isShared_1111_ = v_isSharedCheck_1117_;
goto v_resetjp_1109_;
}
v_resetjp_1109_:
{
lean_object* v___x_1113_; 
if (v_isShared_1111_ == 0)
{
lean_ctor_set(v___x_1110_, 0, v_snd_1103_);
v___x_1113_ = v___x_1110_;
goto v_reusejp_1112_;
}
else
{
lean_object* v_reuseFailAlloc_1116_; 
v_reuseFailAlloc_1116_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1116_, 0, v_snd_1103_);
lean_ctor_set(v_reuseFailAlloc_1116_, 1, v_cache_1105_);
lean_ctor_set(v_reuseFailAlloc_1116_, 2, v_zetaDeltaFVarIds_1106_);
lean_ctor_set(v_reuseFailAlloc_1116_, 3, v_postponed_1107_);
lean_ctor_set(v_reuseFailAlloc_1116_, 4, v_diag_1108_);
v___x_1113_ = v_reuseFailAlloc_1116_;
goto v_reusejp_1112_;
}
v_reusejp_1112_:
{
lean_object* v___x_1114_; lean_object* v___x_1115_; 
v___x_1114_ = lean_st_ref_set(v___y_1095_, v___x_1113_);
v___x_1115_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1115_, 0, v_fst_1102_);
return v___x_1115_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__1___redArg___boxed(lean_object* v_e_1119_, lean_object* v___y_1120_, lean_object* v___y_1121_){
_start:
{
lean_object* v_res_1122_; 
v_res_1122_ = lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__1___redArg(v_e_1119_, v___y_1120_);
lean_dec(v___y_1120_);
return v_res_1122_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__1(lean_object* v_e_1123_, lean_object* v___y_1124_, lean_object* v___y_1125_, lean_object* v___y_1126_, lean_object* v___y_1127_, lean_object* v___y_1128_, lean_object* v___y_1129_, lean_object* v___y_1130_, lean_object* v___y_1131_){
_start:
{
lean_object* v___x_1133_; 
v___x_1133_ = lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__1___redArg(v_e_1123_, v___y_1129_);
return v___x_1133_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__1___boxed(lean_object* v_e_1134_, lean_object* v___y_1135_, lean_object* v___y_1136_, lean_object* v___y_1137_, lean_object* v___y_1138_, lean_object* v___y_1139_, lean_object* v___y_1140_, lean_object* v___y_1141_, lean_object* v___y_1142_, lean_object* v___y_1143_){
_start:
{
lean_object* v_res_1144_; 
v_res_1144_ = lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__1(v_e_1134_, v___y_1135_, v___y_1136_, v___y_1137_, v___y_1138_, v___y_1139_, v___y_1140_, v___y_1141_, v___y_1142_);
lean_dec(v___y_1142_);
lean_dec_ref(v___y_1141_);
lean_dec(v___y_1140_);
lean_dec_ref(v___y_1139_);
lean_dec(v___y_1138_);
lean_dec_ref(v___y_1137_);
lean_dec(v___y_1136_);
lean_dec_ref(v___y_1135_);
return v_res_1144_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__2___redArg(){
_start:
{
lean_object* v___x_1146_; lean_object* v___x_1147_; 
v___x_1146_ = lean_obj_once(&lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__3___redArg___closed__0, &lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__3___redArg___closed__0_once, _init_lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_spec__3___redArg___closed__0);
v___x_1147_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1147_, 0, v___x_1146_);
return v___x_1147_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__2___redArg___boxed(lean_object* v___y_1148_){
_start:
{
lean_object* v_res_1149_; 
v_res_1149_ = lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__2___redArg();
return v_res_1149_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__2(lean_object* v_00_u03b1_1150_, lean_object* v___y_1151_, lean_object* v___y_1152_, lean_object* v___y_1153_, lean_object* v___y_1154_, lean_object* v___y_1155_, lean_object* v___y_1156_, lean_object* v___y_1157_, lean_object* v___y_1158_){
_start:
{
lean_object* v___x_1160_; 
v___x_1160_ = lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__2___redArg();
return v___x_1160_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__2___boxed(lean_object* v_00_u03b1_1161_, lean_object* v___y_1162_, lean_object* v___y_1163_, lean_object* v___y_1164_, lean_object* v___y_1165_, lean_object* v___y_1166_, lean_object* v___y_1167_, lean_object* v___y_1168_, lean_object* v___y_1169_, lean_object* v___y_1170_){
_start:
{
lean_object* v_res_1171_; 
v_res_1171_ = lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__2(v_00_u03b1_1161_, v___y_1162_, v___y_1163_, v___y_1164_, v___y_1165_, v___y_1166_, v___y_1167_, v___y_1168_, v___y_1169_);
lean_dec(v___y_1169_);
lean_dec_ref(v___y_1168_);
lean_dec(v___y_1167_);
lean_dec_ref(v___y_1166_);
lean_dec(v___y_1165_);
lean_dec_ref(v___y_1164_);
lean_dec(v___y_1163_);
lean_dec_ref(v___y_1162_);
return v_res_1171_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__3___redArg___lam__0(lean_object* v_x_1172_, lean_object* v___y_1173_, lean_object* v___y_1174_, lean_object* v___y_1175_, lean_object* v___y_1176_, lean_object* v___y_1177_, lean_object* v___y_1178_, lean_object* v___y_1179_, lean_object* v___y_1180_){
_start:
{
lean_object* v___x_1182_; 
lean_inc(v___y_1176_);
lean_inc_ref(v___y_1175_);
lean_inc(v___y_1174_);
lean_inc_ref(v___y_1173_);
v___x_1182_ = lean_apply_9(v_x_1172_, v___y_1173_, v___y_1174_, v___y_1175_, v___y_1176_, v___y_1177_, v___y_1178_, v___y_1179_, v___y_1180_, lean_box(0));
return v___x_1182_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__3___redArg___lam__0___boxed(lean_object* v_x_1183_, lean_object* v___y_1184_, lean_object* v___y_1185_, lean_object* v___y_1186_, lean_object* v___y_1187_, lean_object* v___y_1188_, lean_object* v___y_1189_, lean_object* v___y_1190_, lean_object* v___y_1191_, lean_object* v___y_1192_){
_start:
{
lean_object* v_res_1193_; 
v_res_1193_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__3___redArg___lam__0(v_x_1183_, v___y_1184_, v___y_1185_, v___y_1186_, v___y_1187_, v___y_1188_, v___y_1189_, v___y_1190_, v___y_1191_);
lean_dec(v___y_1187_);
lean_dec_ref(v___y_1186_);
lean_dec(v___y_1185_);
lean_dec_ref(v___y_1184_);
return v_res_1193_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__3___redArg(lean_object* v_lctx_1194_, lean_object* v_localInsts_1195_, lean_object* v_x_1196_, lean_object* v___y_1197_, lean_object* v___y_1198_, lean_object* v___y_1199_, lean_object* v___y_1200_, lean_object* v___y_1201_, lean_object* v___y_1202_, lean_object* v___y_1203_, lean_object* v___y_1204_){
_start:
{
lean_object* v___f_1206_; lean_object* v___x_1207_; 
lean_inc(v___y_1200_);
lean_inc_ref(v___y_1199_);
lean_inc(v___y_1198_);
lean_inc_ref(v___y_1197_);
v___f_1206_ = lean_alloc_closure((void*)(lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__3___redArg___lam__0___boxed), 10, 5);
lean_closure_set(v___f_1206_, 0, v_x_1196_);
lean_closure_set(v___f_1206_, 1, v___y_1197_);
lean_closure_set(v___f_1206_, 2, v___y_1198_);
lean_closure_set(v___f_1206_, 3, v___y_1199_);
lean_closure_set(v___f_1206_, 4, v___y_1200_);
v___x_1207_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_box(0), v_lctx_1194_, v_localInsts_1195_, v___f_1206_, v___y_1201_, v___y_1202_, v___y_1203_, v___y_1204_);
if (lean_obj_tag(v___x_1207_) == 0)
{
return v___x_1207_;
}
else
{
lean_object* v_a_1208_; lean_object* v___x_1210_; uint8_t v_isShared_1211_; uint8_t v_isSharedCheck_1215_; 
v_a_1208_ = lean_ctor_get(v___x_1207_, 0);
v_isSharedCheck_1215_ = !lean_is_exclusive(v___x_1207_);
if (v_isSharedCheck_1215_ == 0)
{
v___x_1210_ = v___x_1207_;
v_isShared_1211_ = v_isSharedCheck_1215_;
goto v_resetjp_1209_;
}
else
{
lean_inc(v_a_1208_);
lean_dec(v___x_1207_);
v___x_1210_ = lean_box(0);
v_isShared_1211_ = v_isSharedCheck_1215_;
goto v_resetjp_1209_;
}
v_resetjp_1209_:
{
lean_object* v___x_1213_; 
if (v_isShared_1211_ == 0)
{
v___x_1213_ = v___x_1210_;
goto v_reusejp_1212_;
}
else
{
lean_object* v_reuseFailAlloc_1214_; 
v_reuseFailAlloc_1214_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1214_, 0, v_a_1208_);
v___x_1213_ = v_reuseFailAlloc_1214_;
goto v_reusejp_1212_;
}
v_reusejp_1212_:
{
return v___x_1213_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__3___redArg___boxed(lean_object* v_lctx_1216_, lean_object* v_localInsts_1217_, lean_object* v_x_1218_, lean_object* v___y_1219_, lean_object* v___y_1220_, lean_object* v___y_1221_, lean_object* v___y_1222_, lean_object* v___y_1223_, lean_object* v___y_1224_, lean_object* v___y_1225_, lean_object* v___y_1226_, lean_object* v___y_1227_){
_start:
{
lean_object* v_res_1228_; 
v_res_1228_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__3___redArg(v_lctx_1216_, v_localInsts_1217_, v_x_1218_, v___y_1219_, v___y_1220_, v___y_1221_, v___y_1222_, v___y_1223_, v___y_1224_, v___y_1225_, v___y_1226_);
lean_dec(v___y_1226_);
lean_dec_ref(v___y_1225_);
lean_dec(v___y_1224_);
lean_dec_ref(v___y_1223_);
lean_dec(v___y_1222_);
lean_dec_ref(v___y_1221_);
lean_dec(v___y_1220_);
lean_dec_ref(v___y_1219_);
return v_res_1228_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__3(lean_object* v_00_u03b1_1229_, lean_object* v_lctx_1230_, lean_object* v_localInsts_1231_, lean_object* v_x_1232_, lean_object* v___y_1233_, lean_object* v___y_1234_, lean_object* v___y_1235_, lean_object* v___y_1236_, lean_object* v___y_1237_, lean_object* v___y_1238_, lean_object* v___y_1239_, lean_object* v___y_1240_){
_start:
{
lean_object* v___x_1242_; 
v___x_1242_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__3___redArg(v_lctx_1230_, v_localInsts_1231_, v_x_1232_, v___y_1233_, v___y_1234_, v___y_1235_, v___y_1236_, v___y_1237_, v___y_1238_, v___y_1239_, v___y_1240_);
return v___x_1242_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__3___boxed(lean_object* v_00_u03b1_1243_, lean_object* v_lctx_1244_, lean_object* v_localInsts_1245_, lean_object* v_x_1246_, lean_object* v___y_1247_, lean_object* v___y_1248_, lean_object* v___y_1249_, lean_object* v___y_1250_, lean_object* v___y_1251_, lean_object* v___y_1252_, lean_object* v___y_1253_, lean_object* v___y_1254_, lean_object* v___y_1255_){
_start:
{
lean_object* v_res_1256_; 
v_res_1256_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__3(v_00_u03b1_1243_, v_lctx_1244_, v_localInsts_1245_, v_x_1246_, v___y_1247_, v___y_1248_, v___y_1249_, v___y_1250_, v___y_1251_, v___y_1252_, v___y_1253_, v___y_1254_);
lean_dec(v___y_1254_);
lean_dec_ref(v___y_1253_);
lean_dec(v___y_1252_);
lean_dec_ref(v___y_1251_);
lean_dec(v___y_1250_);
lean_dec_ref(v___y_1249_);
lean_dec(v___y_1248_);
lean_dec_ref(v___y_1247_);
return v_res_1256_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_MVarId_withContext___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__5___redArg(lean_object* v_mvarId_1257_, lean_object* v_x_1258_, lean_object* v___y_1259_, lean_object* v___y_1260_, lean_object* v___y_1261_, lean_object* v___y_1262_, lean_object* v___y_1263_, lean_object* v___y_1264_, lean_object* v___y_1265_, lean_object* v___y_1266_){
_start:
{
lean_object* v___f_1268_; lean_object* v___x_1269_; 
lean_inc(v___y_1262_);
lean_inc_ref(v___y_1261_);
lean_inc(v___y_1260_);
lean_inc_ref(v___y_1259_);
v___f_1268_ = lean_alloc_closure((void*)(lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__3___redArg___lam__0___boxed), 10, 5);
lean_closure_set(v___f_1268_, 0, v_x_1258_);
lean_closure_set(v___f_1268_, 1, v___y_1259_);
lean_closure_set(v___f_1268_, 2, v___y_1260_);
lean_closure_set(v___f_1268_, 3, v___y_1261_);
lean_closure_set(v___f_1268_, 4, v___y_1262_);
v___x_1269_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_1257_, v___f_1268_, v___y_1263_, v___y_1264_, v___y_1265_, v___y_1266_);
if (lean_obj_tag(v___x_1269_) == 0)
{
return v___x_1269_;
}
else
{
lean_object* v_a_1270_; lean_object* v___x_1272_; uint8_t v_isShared_1273_; uint8_t v_isSharedCheck_1277_; 
v_a_1270_ = lean_ctor_get(v___x_1269_, 0);
v_isSharedCheck_1277_ = !lean_is_exclusive(v___x_1269_);
if (v_isSharedCheck_1277_ == 0)
{
v___x_1272_ = v___x_1269_;
v_isShared_1273_ = v_isSharedCheck_1277_;
goto v_resetjp_1271_;
}
else
{
lean_inc(v_a_1270_);
lean_dec(v___x_1269_);
v___x_1272_ = lean_box(0);
v_isShared_1273_ = v_isSharedCheck_1277_;
goto v_resetjp_1271_;
}
v_resetjp_1271_:
{
lean_object* v___x_1275_; 
if (v_isShared_1273_ == 0)
{
v___x_1275_ = v___x_1272_;
goto v_reusejp_1274_;
}
else
{
lean_object* v_reuseFailAlloc_1276_; 
v_reuseFailAlloc_1276_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1276_, 0, v_a_1270_);
v___x_1275_ = v_reuseFailAlloc_1276_;
goto v_reusejp_1274_;
}
v_reusejp_1274_:
{
return v___x_1275_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_MVarId_withContext___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__5___redArg___boxed(lean_object* v_mvarId_1278_, lean_object* v_x_1279_, lean_object* v___y_1280_, lean_object* v___y_1281_, lean_object* v___y_1282_, lean_object* v___y_1283_, lean_object* v___y_1284_, lean_object* v___y_1285_, lean_object* v___y_1286_, lean_object* v___y_1287_, lean_object* v___y_1288_){
_start:
{
lean_object* v_res_1289_; 
v_res_1289_ = lp_Qq_Lean_MVarId_withContext___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__5___redArg(v_mvarId_1278_, v_x_1279_, v___y_1280_, v___y_1281_, v___y_1282_, v___y_1283_, v___y_1284_, v___y_1285_, v___y_1286_, v___y_1287_);
lean_dec(v___y_1287_);
lean_dec_ref(v___y_1286_);
lean_dec(v___y_1285_);
lean_dec_ref(v___y_1284_);
lean_dec(v___y_1283_);
lean_dec_ref(v___y_1282_);
lean_dec(v___y_1281_);
lean_dec_ref(v___y_1280_);
return v_res_1289_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_MVarId_withContext___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__5(lean_object* v_00_u03b1_1290_, lean_object* v_mvarId_1291_, lean_object* v_x_1292_, lean_object* v___y_1293_, lean_object* v___y_1294_, lean_object* v___y_1295_, lean_object* v___y_1296_, lean_object* v___y_1297_, lean_object* v___y_1298_, lean_object* v___y_1299_, lean_object* v___y_1300_){
_start:
{
lean_object* v___x_1302_; 
v___x_1302_ = lp_Qq_Lean_MVarId_withContext___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__5___redArg(v_mvarId_1291_, v_x_1292_, v___y_1293_, v___y_1294_, v___y_1295_, v___y_1296_, v___y_1297_, v___y_1298_, v___y_1299_, v___y_1300_);
return v___x_1302_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_MVarId_withContext___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__5___boxed(lean_object* v_00_u03b1_1303_, lean_object* v_mvarId_1304_, lean_object* v_x_1305_, lean_object* v___y_1306_, lean_object* v___y_1307_, lean_object* v___y_1308_, lean_object* v___y_1309_, lean_object* v___y_1310_, lean_object* v___y_1311_, lean_object* v___y_1312_, lean_object* v___y_1313_, lean_object* v___y_1314_){
_start:
{
lean_object* v_res_1315_; 
v_res_1315_ = lp_Qq_Lean_MVarId_withContext___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__5(v_00_u03b1_1303_, v_mvarId_1304_, v_x_1305_, v___y_1306_, v___y_1307_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_, v___y_1312_, v___y_1313_);
lean_dec(v___y_1313_);
lean_dec_ref(v___y_1312_);
lean_dec(v___y_1311_);
lean_dec_ref(v___y_1310_);
lean_dec(v___y_1309_);
lean_dec_ref(v___y_1308_);
lean_dec(v___y_1307_);
lean_dec_ref(v___y_1306_);
return v_res_1315_;
}
}
static lean_object* _init_lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__2(void){
_start:
{
lean_object* v___x_1318_; lean_object* v___x_1319_; 
v___x_1318_ = ((lean_object*)(lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__1));
v___x_1319_ = l_String_toRawSubstring_x27(v___x_1318_);
return v___x_1319_;
}
}
static lean_object* _init_lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__10(void){
_start:
{
lean_object* v___x_1335_; lean_object* v___x_1336_; lean_object* v___x_1337_; 
v___x_1335_ = lean_box(0);
v___x_1336_ = ((lean_object*)(lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__5));
v___x_1337_ = l_Lean_Expr_const___override(v___x_1336_, v___x_1335_);
return v___x_1337_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0(uint8_t v___x_1338_, lean_object* v___x_1339_, lean_object* v___x_1340_, lean_object* v___x_1341_, lean_object* v___x_1342_, uint8_t v___x_1343_, lean_object* v_fst_1344_, lean_object* v_____r_1345_, lean_object* v___y_1346_, lean_object* v___y_1347_, lean_object* v___y_1348_, lean_object* v___y_1349_, lean_object* v___y_1350_, lean_object* v___y_1351_, lean_object* v___y_1352_, lean_object* v___y_1353_){
_start:
{
lean_object* v_ref_1355_; lean_object* v_quotContext_1356_; lean_object* v_currMacroScope_1357_; lean_object* v___x_1358_; lean_object* v___x_1359_; lean_object* v___x_1360_; lean_object* v___x_1361_; lean_object* v___x_1362_; lean_object* v___x_1363_; lean_object* v___x_1364_; lean_object* v___x_1365_; lean_object* v___x_1366_; lean_object* v___x_1367_; lean_object* v___x_1368_; lean_object* v___x_1369_; lean_object* v___x_1370_; lean_object* v___x_1371_; lean_object* v___x_1372_; lean_object* v___x_1373_; lean_object* v___x_1374_; lean_object* v___x_1375_; lean_object* v___x_1376_; lean_object* v___x_1377_; lean_object* v___x_1378_; lean_object* v___x_1379_; lean_object* v___x_1380_; lean_object* v___x_1381_; lean_object* v___x_1382_; lean_object* v___x_1383_; 
v_ref_1355_ = lean_ctor_get(v___y_1352_, 5);
v_quotContext_1356_ = lean_ctor_get(v___y_1352_, 10);
v_currMacroScope_1357_ = lean_ctor_get(v___y_1352_, 11);
v___x_1358_ = l_Lean_SourceInfo_fromRef(v_ref_1355_, v___x_1338_);
v___x_1359_ = ((lean_object*)(lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__0));
lean_inc_ref(v___x_1341_);
lean_inc_ref(v___x_1340_);
lean_inc_ref_n(v___x_1339_, 2);
v___x_1360_ = l_Lean_Name_mkStr4(v___x_1339_, v___x_1340_, v___x_1341_, v___x_1359_);
v___x_1361_ = lean_obj_once(&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__2, &lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__2_once, _init_lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__2);
v___x_1362_ = ((lean_object*)(lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__3));
lean_inc(v_currMacroScope_1357_);
lean_inc(v_quotContext_1356_);
v___x_1363_ = l_Lean_addMacroScope(v_quotContext_1356_, v___x_1362_, v_currMacroScope_1357_);
v___x_1364_ = lean_box(0);
v___x_1365_ = ((lean_object*)(lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__7));
lean_inc_n(v___x_1358_, 4);
v___x_1366_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1366_, 0, v___x_1358_);
lean_ctor_set(v___x_1366_, 1, v___x_1361_);
lean_ctor_set(v___x_1366_, 2, v___x_1363_);
lean_ctor_set(v___x_1366_, 3, v___x_1365_);
v___x_1367_ = ((lean_object*)(lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__9));
v___x_1368_ = ((lean_object*)(lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__1___closed__0));
v___x_1369_ = l_Lean_Name_mkStr4(v___x_1339_, v___x_1340_, v___x_1341_, v___x_1368_);
v___x_1370_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1370_, 0, v___x_1358_);
lean_ctor_set(v___x_1370_, 1, v___x_1368_);
v___x_1371_ = l_Lean_Syntax_node2(v___x_1358_, v___x_1369_, v___x_1370_, v___x_1342_);
v___x_1372_ = l_Lean_Syntax_node1(v___x_1358_, v___x_1367_, v___x_1371_);
v___x_1373_ = l_Lean_Syntax_node2(v___x_1358_, v___x_1360_, v___x_1366_, v___x_1372_);
v___x_1374_ = ((lean_object*)(lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__1));
v___x_1375_ = ((lean_object*)(lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__0));
v___x_1376_ = ((lean_object*)(lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__1));
v___x_1377_ = l_Lean_Name_mkStr4(v___x_1339_, v___x_1374_, v___x_1375_, v___x_1376_);
v___x_1378_ = l_Lean_Expr_const___override(v___x_1377_, v___x_1364_);
v___x_1379_ = lean_obj_once(&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__10, &lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__10_once, _init_lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___closed__10);
v___x_1380_ = l_Lean_Expr_app___override(v___x_1378_, v___x_1379_);
v___x_1381_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1381_, 0, v___x_1380_);
v___x_1382_ = lean_box(0);
v___x_1383_ = l_Lean_Elab_Term_elabTermEnsuringType(v___x_1373_, v___x_1381_, v___x_1343_, v___x_1343_, v___x_1382_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1383_) == 0)
{
lean_object* v_a_1384_; lean_object* v___x_1385_; 
v_a_1384_ = lean_ctor_get(v___x_1383_, 0);
lean_inc(v_a_1384_);
lean_dec_ref_known(v___x_1383_, 1);
v___x_1385_ = l_Lean_Elab_Term_synthesizeSyntheticMVarsNoPostponing(v___x_1338_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1385_) == 0)
{
lean_object* v___x_1386_; lean_object* v_a_1387_; lean_object* v___x_1388_; 
lean_dec_ref_known(v___x_1385_, 1);
v___x_1386_ = lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__1___redArg(v_a_1384_, v___y_1351_);
v_a_1387_ = lean_ctor_get(v___x_1386_, 0);
lean_inc_n(v_a_1387_, 2);
lean_dec_ref(v___x_1386_);
v___x_1388_ = l_Lean_Meta_getMVars(v_a_1387_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1388_) == 0)
{
lean_object* v_a_1389_; lean_object* v___x_1390_; 
v_a_1389_ = lean_ctor_get(v___x_1388_, 0);
lean_inc(v_a_1389_);
lean_dec_ref_known(v___x_1388_, 1);
v___x_1390_ = l_Lean_Elab_Term_logUnassignedUsingErrorInfos(v_a_1389_, v___x_1382_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
lean_dec(v_a_1389_);
if (lean_obj_tag(v___x_1390_) == 0)
{
lean_object* v_a_1391_; uint8_t v___x_1392_; 
v_a_1391_ = lean_ctor_get(v___x_1390_, 0);
lean_inc(v_a_1391_);
lean_dec_ref_known(v___x_1390_, 1);
v___x_1392_ = lean_unbox(v_a_1391_);
lean_dec(v_a_1391_);
if (v___x_1392_ == 0)
{
lean_object* v___x_1393_; 
v___x_1393_ = lp_Qq___private_Qq_Commands_0__Qq_mkLetFVarsFromValues(v_fst_1344_, v_a_1387_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
return v___x_1393_;
}
else
{
lean_object* v___x_1394_; lean_object* v_a_1395_; lean_object* v___x_1397_; uint8_t v_isShared_1398_; uint8_t v_isSharedCheck_1402_; 
lean_dec(v_a_1387_);
v___x_1394_ = lp_Qq_Lean_Elab_throwAbortTerm___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__2___redArg();
v_a_1395_ = lean_ctor_get(v___x_1394_, 0);
v_isSharedCheck_1402_ = !lean_is_exclusive(v___x_1394_);
if (v_isSharedCheck_1402_ == 0)
{
v___x_1397_ = v___x_1394_;
v_isShared_1398_ = v_isSharedCheck_1402_;
goto v_resetjp_1396_;
}
else
{
lean_inc(v_a_1395_);
lean_dec(v___x_1394_);
v___x_1397_ = lean_box(0);
v_isShared_1398_ = v_isSharedCheck_1402_;
goto v_resetjp_1396_;
}
v_resetjp_1396_:
{
lean_object* v___x_1400_; 
if (v_isShared_1398_ == 0)
{
v___x_1400_ = v___x_1397_;
goto v_reusejp_1399_;
}
else
{
lean_object* v_reuseFailAlloc_1401_; 
v_reuseFailAlloc_1401_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1401_, 0, v_a_1395_);
v___x_1400_ = v_reuseFailAlloc_1401_;
goto v_reusejp_1399_;
}
v_reusejp_1399_:
{
return v___x_1400_;
}
}
}
}
else
{
lean_object* v_a_1403_; lean_object* v___x_1405_; uint8_t v_isShared_1406_; uint8_t v_isSharedCheck_1410_; 
lean_dec(v_a_1387_);
v_a_1403_ = lean_ctor_get(v___x_1390_, 0);
v_isSharedCheck_1410_ = !lean_is_exclusive(v___x_1390_);
if (v_isSharedCheck_1410_ == 0)
{
v___x_1405_ = v___x_1390_;
v_isShared_1406_ = v_isSharedCheck_1410_;
goto v_resetjp_1404_;
}
else
{
lean_inc(v_a_1403_);
lean_dec(v___x_1390_);
v___x_1405_ = lean_box(0);
v_isShared_1406_ = v_isSharedCheck_1410_;
goto v_resetjp_1404_;
}
v_resetjp_1404_:
{
lean_object* v___x_1408_; 
if (v_isShared_1406_ == 0)
{
v___x_1408_ = v___x_1405_;
goto v_reusejp_1407_;
}
else
{
lean_object* v_reuseFailAlloc_1409_; 
v_reuseFailAlloc_1409_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1409_, 0, v_a_1403_);
v___x_1408_ = v_reuseFailAlloc_1409_;
goto v_reusejp_1407_;
}
v_reusejp_1407_:
{
return v___x_1408_;
}
}
}
}
else
{
lean_object* v_a_1411_; lean_object* v___x_1413_; uint8_t v_isShared_1414_; uint8_t v_isSharedCheck_1418_; 
lean_dec(v_a_1387_);
v_a_1411_ = lean_ctor_get(v___x_1388_, 0);
v_isSharedCheck_1418_ = !lean_is_exclusive(v___x_1388_);
if (v_isSharedCheck_1418_ == 0)
{
v___x_1413_ = v___x_1388_;
v_isShared_1414_ = v_isSharedCheck_1418_;
goto v_resetjp_1412_;
}
else
{
lean_inc(v_a_1411_);
lean_dec(v___x_1388_);
v___x_1413_ = lean_box(0);
v_isShared_1414_ = v_isSharedCheck_1418_;
goto v_resetjp_1412_;
}
v_resetjp_1412_:
{
lean_object* v___x_1416_; 
if (v_isShared_1414_ == 0)
{
v___x_1416_ = v___x_1413_;
goto v_reusejp_1415_;
}
else
{
lean_object* v_reuseFailAlloc_1417_; 
v_reuseFailAlloc_1417_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1417_, 0, v_a_1411_);
v___x_1416_ = v_reuseFailAlloc_1417_;
goto v_reusejp_1415_;
}
v_reusejp_1415_:
{
return v___x_1416_;
}
}
}
}
else
{
lean_object* v_a_1419_; lean_object* v___x_1421_; uint8_t v_isShared_1422_; uint8_t v_isSharedCheck_1426_; 
lean_dec(v_a_1384_);
v_a_1419_ = lean_ctor_get(v___x_1385_, 0);
v_isSharedCheck_1426_ = !lean_is_exclusive(v___x_1385_);
if (v_isSharedCheck_1426_ == 0)
{
v___x_1421_ = v___x_1385_;
v_isShared_1422_ = v_isSharedCheck_1426_;
goto v_resetjp_1420_;
}
else
{
lean_inc(v_a_1419_);
lean_dec(v___x_1385_);
v___x_1421_ = lean_box(0);
v_isShared_1422_ = v_isSharedCheck_1426_;
goto v_resetjp_1420_;
}
v_resetjp_1420_:
{
lean_object* v___x_1424_; 
if (v_isShared_1422_ == 0)
{
v___x_1424_ = v___x_1421_;
goto v_reusejp_1423_;
}
else
{
lean_object* v_reuseFailAlloc_1425_; 
v_reuseFailAlloc_1425_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1425_, 0, v_a_1419_);
v___x_1424_ = v_reuseFailAlloc_1425_;
goto v_reusejp_1423_;
}
v_reusejp_1423_:
{
return v___x_1424_;
}
}
}
}
else
{
return v___x_1383_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___boxed(lean_object** _args){
lean_object* v___x_1427_ = _args[0];
lean_object* v___x_1428_ = _args[1];
lean_object* v___x_1429_ = _args[2];
lean_object* v___x_1430_ = _args[3];
lean_object* v___x_1431_ = _args[4];
lean_object* v___x_1432_ = _args[5];
lean_object* v_fst_1433_ = _args[6];
lean_object* v_____r_1434_ = _args[7];
lean_object* v___y_1435_ = _args[8];
lean_object* v___y_1436_ = _args[9];
lean_object* v___y_1437_ = _args[10];
lean_object* v___y_1438_ = _args[11];
lean_object* v___y_1439_ = _args[12];
lean_object* v___y_1440_ = _args[13];
lean_object* v___y_1441_ = _args[14];
lean_object* v___y_1442_ = _args[15];
lean_object* v___y_1443_ = _args[16];
_start:
{
uint8_t v___x_22497__boxed_1444_; uint8_t v___x_22502__boxed_1445_; lean_object* v_res_1446_; 
v___x_22497__boxed_1444_ = lean_unbox(v___x_1427_);
v___x_22502__boxed_1445_ = lean_unbox(v___x_1432_);
v_res_1446_ = lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0(v___x_22497__boxed_1444_, v___x_1428_, v___x_1429_, v___x_1430_, v___x_1431_, v___x_22502__boxed_1445_, v_fst_1433_, v_____r_1434_, v___y_1435_, v___y_1436_, v___y_1437_, v___y_1438_, v___y_1439_, v___y_1440_, v___y_1441_, v___y_1442_);
lean_dec(v___y_1442_);
lean_dec_ref(v___y_1441_);
lean_dec(v___y_1440_);
lean_dec_ref(v___y_1439_);
lean_dec(v___y_1438_);
lean_dec_ref(v___y_1437_);
lean_dec(v___y_1436_);
lean_dec_ref(v___y_1435_);
lean_dec_ref(v_fst_1433_);
return v_res_1446_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__1(lean_object* v_snd_1447_, uint8_t v___x_1448_, uint8_t v___x_1449_, lean_object* v___f_1450_, lean_object* v___y_1451_, lean_object* v___y_1452_, lean_object* v___y_1453_, lean_object* v___y_1454_, lean_object* v___y_1455_, lean_object* v___y_1456_, lean_object* v___y_1457_, lean_object* v___y_1458_){
_start:
{
if (lean_obj_tag(v_snd_1447_) == 1)
{
lean_object* v_val_1460_; lean_object* v_fst_1461_; lean_object* v_snd_1462_; lean_object* v___x_1463_; lean_object* v___x_1464_; lean_object* v___x_1465_; lean_object* v___x_1466_; 
v_val_1460_ = lean_ctor_get(v_snd_1447_, 0);
lean_inc(v_val_1460_);
lean_dec_ref_known(v_snd_1447_, 1);
v_fst_1461_ = lean_ctor_get(v_val_1460_, 0);
lean_inc(v_fst_1461_);
v_snd_1462_ = lean_ctor_get(v_val_1460_, 1);
lean_inc(v_snd_1462_);
lean_dec(v_val_1460_);
v___x_1463_ = l_Lean_Expr_fvar___override(v_snd_1462_);
v___x_1464_ = lean_box(0);
v___x_1465_ = lean_box(0);
v___x_1466_ = l_Lean_Elab_Term_addTermInfo_x27(v_fst_1461_, v___x_1463_, v___x_1464_, v___x_1464_, v___x_1465_, v___x_1448_, v___x_1449_, v___y_1453_, v___y_1454_, v___y_1455_, v___y_1456_, v___y_1457_, v___y_1458_);
if (lean_obj_tag(v___x_1466_) == 0)
{
lean_object* v___x_1467_; lean_object* v___x_1468_; 
lean_dec_ref_known(v___x_1466_, 1);
v___x_1467_ = lean_box(0);
lean_inc(v___y_1458_);
lean_inc_ref(v___y_1457_);
lean_inc(v___y_1456_);
lean_inc_ref(v___y_1455_);
lean_inc(v___y_1454_);
lean_inc_ref(v___y_1453_);
lean_inc(v___y_1452_);
lean_inc_ref(v___y_1451_);
v___x_1468_ = lean_apply_10(v___f_1450_, v___x_1467_, v___y_1451_, v___y_1452_, v___y_1453_, v___y_1454_, v___y_1455_, v___y_1456_, v___y_1457_, v___y_1458_, lean_box(0));
return v___x_1468_;
}
else
{
if (lean_obj_tag(v___x_1466_) == 0)
{
lean_object* v_a_1469_; lean_object* v___x_1470_; 
v_a_1469_ = lean_ctor_get(v___x_1466_, 0);
lean_inc(v_a_1469_);
lean_dec_ref_known(v___x_1466_, 1);
lean_inc(v___y_1458_);
lean_inc_ref(v___y_1457_);
lean_inc(v___y_1456_);
lean_inc_ref(v___y_1455_);
lean_inc(v___y_1454_);
lean_inc_ref(v___y_1453_);
lean_inc(v___y_1452_);
lean_inc_ref(v___y_1451_);
v___x_1470_ = lean_apply_10(v___f_1450_, v_a_1469_, v___y_1451_, v___y_1452_, v___y_1453_, v___y_1454_, v___y_1455_, v___y_1456_, v___y_1457_, v___y_1458_, lean_box(0));
return v___x_1470_;
}
else
{
lean_object* v_a_1471_; lean_object* v___x_1473_; uint8_t v_isShared_1474_; uint8_t v_isSharedCheck_1478_; 
lean_dec_ref(v___f_1450_);
v_a_1471_ = lean_ctor_get(v___x_1466_, 0);
v_isSharedCheck_1478_ = !lean_is_exclusive(v___x_1466_);
if (v_isSharedCheck_1478_ == 0)
{
v___x_1473_ = v___x_1466_;
v_isShared_1474_ = v_isSharedCheck_1478_;
goto v_resetjp_1472_;
}
else
{
lean_inc(v_a_1471_);
lean_dec(v___x_1466_);
v___x_1473_ = lean_box(0);
v_isShared_1474_ = v_isSharedCheck_1478_;
goto v_resetjp_1472_;
}
v_resetjp_1472_:
{
lean_object* v___x_1476_; 
if (v_isShared_1474_ == 0)
{
v___x_1476_ = v___x_1473_;
goto v_reusejp_1475_;
}
else
{
lean_object* v_reuseFailAlloc_1477_; 
v_reuseFailAlloc_1477_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1477_, 0, v_a_1471_);
v___x_1476_ = v_reuseFailAlloc_1477_;
goto v_reusejp_1475_;
}
v_reusejp_1475_:
{
return v___x_1476_;
}
}
}
}
}
else
{
lean_object* v___x_1479_; lean_object* v___x_1480_; 
lean_dec(v_snd_1447_);
v___x_1479_ = lean_box(0);
lean_inc(v___y_1458_);
lean_inc_ref(v___y_1457_);
lean_inc(v___y_1456_);
lean_inc_ref(v___y_1455_);
lean_inc(v___y_1454_);
lean_inc_ref(v___y_1453_);
lean_inc(v___y_1452_);
lean_inc_ref(v___y_1451_);
v___x_1480_ = lean_apply_10(v___f_1450_, v___x_1479_, v___y_1451_, v___y_1452_, v___y_1453_, v___y_1454_, v___y_1455_, v___y_1456_, v___y_1457_, v___y_1458_, lean_box(0));
return v___x_1480_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__1___boxed(lean_object* v_snd_1481_, lean_object* v___x_1482_, lean_object* v___x_1483_, lean_object* v___f_1484_, lean_object* v___y_1485_, lean_object* v___y_1486_, lean_object* v___y_1487_, lean_object* v___y_1488_, lean_object* v___y_1489_, lean_object* v___y_1490_, lean_object* v___y_1491_, lean_object* v___y_1492_, lean_object* v___y_1493_){
_start:
{
uint8_t v___x_22713__boxed_1494_; uint8_t v___x_22714__boxed_1495_; lean_object* v_res_1496_; 
v___x_22713__boxed_1494_ = lean_unbox(v___x_1482_);
v___x_22714__boxed_1495_ = lean_unbox(v___x_1483_);
v_res_1496_ = lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__1(v_snd_1481_, v___x_22713__boxed_1494_, v___x_22714__boxed_1495_, v___f_1484_, v___y_1485_, v___y_1486_, v___y_1487_, v___y_1488_, v___y_1489_, v___y_1490_, v___y_1491_, v___y_1492_);
lean_dec(v___y_1492_);
lean_dec_ref(v___y_1491_);
lean_dec(v___y_1490_);
lean_dec_ref(v___y_1489_);
lean_dec(v___y_1488_);
lean_dec_ref(v___y_1487_);
lean_dec(v___y_1486_);
lean_dec_ref(v___y_1485_);
return v_res_1496_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_mkFreshId___at___00Lean_mkFreshFVarId___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__4_spec__4___redArg(lean_object* v___y_1497_, lean_object* v___y_1498_){
_start:
{
lean_object* v___x_1500_; lean_object* v_ngen_1501_; lean_object* v_namePrefix_1502_; lean_object* v_idx_1503_; lean_object* v___x_1505_; uint8_t v_isShared_1506_; uint8_t v_isSharedCheck_1533_; 
v___x_1500_ = lean_st_ref_get(v___y_1498_);
v_ngen_1501_ = lean_ctor_get(v___x_1500_, 2);
lean_inc_ref(v_ngen_1501_);
lean_dec(v___x_1500_);
v_namePrefix_1502_ = lean_ctor_get(v_ngen_1501_, 0);
v_idx_1503_ = lean_ctor_get(v_ngen_1501_, 1);
v_isSharedCheck_1533_ = !lean_is_exclusive(v_ngen_1501_);
if (v_isSharedCheck_1533_ == 0)
{
v___x_1505_ = v_ngen_1501_;
v_isShared_1506_ = v_isSharedCheck_1533_;
goto v_resetjp_1504_;
}
else
{
lean_inc(v_idx_1503_);
lean_inc(v_namePrefix_1502_);
lean_dec(v_ngen_1501_);
v___x_1505_ = lean_box(0);
v_isShared_1506_ = v_isSharedCheck_1533_;
goto v_resetjp_1504_;
}
v_resetjp_1504_:
{
lean_object* v___x_1507_; lean_object* v_env_1508_; lean_object* v_nextMacroScope_1509_; lean_object* v_auxDeclNGen_1510_; lean_object* v_traceState_1511_; lean_object* v_cache_1512_; lean_object* v_messages_1513_; lean_object* v_infoState_1514_; lean_object* v_snapshotTasks_1515_; lean_object* v___x_1517_; uint8_t v_isShared_1518_; uint8_t v_isSharedCheck_1531_; 
v___x_1507_ = lean_st_ref_take(v___y_1498_);
v_env_1508_ = lean_ctor_get(v___x_1507_, 0);
v_nextMacroScope_1509_ = lean_ctor_get(v___x_1507_, 1);
v_auxDeclNGen_1510_ = lean_ctor_get(v___x_1507_, 3);
v_traceState_1511_ = lean_ctor_get(v___x_1507_, 4);
v_cache_1512_ = lean_ctor_get(v___x_1507_, 5);
v_messages_1513_ = lean_ctor_get(v___x_1507_, 6);
v_infoState_1514_ = lean_ctor_get(v___x_1507_, 7);
v_snapshotTasks_1515_ = lean_ctor_get(v___x_1507_, 8);
v_isSharedCheck_1531_ = !lean_is_exclusive(v___x_1507_);
if (v_isSharedCheck_1531_ == 0)
{
lean_object* v_unused_1532_; 
v_unused_1532_ = lean_ctor_get(v___x_1507_, 2);
lean_dec(v_unused_1532_);
v___x_1517_ = v___x_1507_;
v_isShared_1518_ = v_isSharedCheck_1531_;
goto v_resetjp_1516_;
}
else
{
lean_inc(v_snapshotTasks_1515_);
lean_inc(v_infoState_1514_);
lean_inc(v_messages_1513_);
lean_inc(v_cache_1512_);
lean_inc(v_traceState_1511_);
lean_inc(v_auxDeclNGen_1510_);
lean_inc(v_nextMacroScope_1509_);
lean_inc(v_env_1508_);
lean_dec(v___x_1507_);
v___x_1517_ = lean_box(0);
v_isShared_1518_ = v_isSharedCheck_1531_;
goto v_resetjp_1516_;
}
v_resetjp_1516_:
{
lean_object* v_r_1519_; lean_object* v___x_1520_; lean_object* v___x_1521_; lean_object* v___x_1523_; 
lean_inc(v_idx_1503_);
lean_inc(v_namePrefix_1502_);
v_r_1519_ = l_Lean_Name_num___override(v_namePrefix_1502_, v_idx_1503_);
v___x_1520_ = lean_unsigned_to_nat(1u);
v___x_1521_ = lean_nat_add(v_idx_1503_, v___x_1520_);
lean_dec(v_idx_1503_);
if (v_isShared_1506_ == 0)
{
lean_ctor_set(v___x_1505_, 1, v___x_1521_);
v___x_1523_ = v___x_1505_;
goto v_reusejp_1522_;
}
else
{
lean_object* v_reuseFailAlloc_1530_; 
v_reuseFailAlloc_1530_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1530_, 0, v_namePrefix_1502_);
lean_ctor_set(v_reuseFailAlloc_1530_, 1, v___x_1521_);
v___x_1523_ = v_reuseFailAlloc_1530_;
goto v_reusejp_1522_;
}
v_reusejp_1522_:
{
lean_object* v___x_1525_; 
if (v_isShared_1518_ == 0)
{
lean_ctor_set(v___x_1517_, 2, v___x_1523_);
v___x_1525_ = v___x_1517_;
goto v_reusejp_1524_;
}
else
{
lean_object* v_reuseFailAlloc_1529_; 
v_reuseFailAlloc_1529_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1529_, 0, v_env_1508_);
lean_ctor_set(v_reuseFailAlloc_1529_, 1, v_nextMacroScope_1509_);
lean_ctor_set(v_reuseFailAlloc_1529_, 2, v___x_1523_);
lean_ctor_set(v_reuseFailAlloc_1529_, 3, v_auxDeclNGen_1510_);
lean_ctor_set(v_reuseFailAlloc_1529_, 4, v_traceState_1511_);
lean_ctor_set(v_reuseFailAlloc_1529_, 5, v_cache_1512_);
lean_ctor_set(v_reuseFailAlloc_1529_, 6, v_messages_1513_);
lean_ctor_set(v_reuseFailAlloc_1529_, 7, v_infoState_1514_);
lean_ctor_set(v_reuseFailAlloc_1529_, 8, v_snapshotTasks_1515_);
v___x_1525_ = v_reuseFailAlloc_1529_;
goto v_reusejp_1524_;
}
v_reusejp_1524_:
{
lean_object* v___x_1526_; lean_object* v___x_1527_; lean_object* v___x_1528_; 
v___x_1526_ = lean_st_ref_set(v___y_1498_, v___x_1525_);
v___x_1527_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1527_, 0, v_r_1519_);
lean_ctor_set(v___x_1527_, 1, v___y_1497_);
v___x_1528_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1528_, 0, v___x_1527_);
return v___x_1528_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_mkFreshId___at___00Lean_mkFreshFVarId___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__4_spec__4___redArg___boxed(lean_object* v___y_1534_, lean_object* v___y_1535_, lean_object* v___y_1536_){
_start:
{
lean_object* v_res_1537_; 
v_res_1537_ = lp_Qq_Lean_mkFreshId___at___00Lean_mkFreshFVarId___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__4_spec__4___redArg(v___y_1534_, v___y_1535_);
lean_dec(v___y_1535_);
return v_res_1537_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_mkFreshFVarId___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__4(lean_object* v___y_1538_, lean_object* v___y_1539_, lean_object* v___y_1540_, lean_object* v___y_1541_, lean_object* v___y_1542_){
_start:
{
lean_object* v___x_1544_; lean_object* v_a_1545_; lean_object* v___x_1547_; uint8_t v_isShared_1548_; uint8_t v_isSharedCheck_1561_; 
v___x_1544_ = lp_Qq_Lean_mkFreshId___at___00Lean_mkFreshFVarId___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__4_spec__4___redArg(v___y_1538_, v___y_1542_);
v_a_1545_ = lean_ctor_get(v___x_1544_, 0);
v_isSharedCheck_1561_ = !lean_is_exclusive(v___x_1544_);
if (v_isSharedCheck_1561_ == 0)
{
v___x_1547_ = v___x_1544_;
v_isShared_1548_ = v_isSharedCheck_1561_;
goto v_resetjp_1546_;
}
else
{
lean_inc(v_a_1545_);
lean_dec(v___x_1544_);
v___x_1547_ = lean_box(0);
v_isShared_1548_ = v_isSharedCheck_1561_;
goto v_resetjp_1546_;
}
v_resetjp_1546_:
{
lean_object* v_fst_1549_; lean_object* v_snd_1550_; lean_object* v___x_1552_; uint8_t v_isShared_1553_; uint8_t v_isSharedCheck_1560_; 
v_fst_1549_ = lean_ctor_get(v_a_1545_, 0);
v_snd_1550_ = lean_ctor_get(v_a_1545_, 1);
v_isSharedCheck_1560_ = !lean_is_exclusive(v_a_1545_);
if (v_isSharedCheck_1560_ == 0)
{
v___x_1552_ = v_a_1545_;
v_isShared_1553_ = v_isSharedCheck_1560_;
goto v_resetjp_1551_;
}
else
{
lean_inc(v_snd_1550_);
lean_inc(v_fst_1549_);
lean_dec(v_a_1545_);
v___x_1552_ = lean_box(0);
v_isShared_1553_ = v_isSharedCheck_1560_;
goto v_resetjp_1551_;
}
v_resetjp_1551_:
{
lean_object* v___x_1555_; 
if (v_isShared_1553_ == 0)
{
v___x_1555_ = v___x_1552_;
goto v_reusejp_1554_;
}
else
{
lean_object* v_reuseFailAlloc_1559_; 
v_reuseFailAlloc_1559_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1559_, 0, v_fst_1549_);
lean_ctor_set(v_reuseFailAlloc_1559_, 1, v_snd_1550_);
v___x_1555_ = v_reuseFailAlloc_1559_;
goto v_reusejp_1554_;
}
v_reusejp_1554_:
{
lean_object* v___x_1557_; 
if (v_isShared_1548_ == 0)
{
lean_ctor_set(v___x_1547_, 0, v___x_1555_);
v___x_1557_ = v___x_1547_;
goto v_reusejp_1556_;
}
else
{
lean_object* v_reuseFailAlloc_1558_; 
v_reuseFailAlloc_1558_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1558_, 0, v___x_1555_);
v___x_1557_ = v_reuseFailAlloc_1558_;
goto v_reusejp_1556_;
}
v_reusejp_1556_:
{
return v___x_1557_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_mkFreshFVarId___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__4___boxed(lean_object* v___y_1562_, lean_object* v___y_1563_, lean_object* v___y_1564_, lean_object* v___y_1565_, lean_object* v___y_1566_, lean_object* v___y_1567_){
_start:
{
lean_object* v_res_1568_; 
v_res_1568_ = lp_Qq_Lean_mkFreshFVarId___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__4(v___y_1562_, v___y_1563_, v___y_1564_, v___y_1565_, v___y_1566_);
lean_dec(v___y_1566_);
lean_dec_ref(v___y_1565_);
lean_dec(v___y_1564_);
lean_dec_ref(v___y_1563_);
return v_res_1568_;
}
}
static lean_object* _init_lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__2___closed__0(void){
_start:
{
lean_object* v___x_1569_; lean_object* v___x_1570_; lean_object* v___x_1571_; 
v___x_1569_ = lean_box(0);
v___x_1570_ = ((lean_object*)(lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__5));
v___x_1571_ = l_Lean_Expr_const___override(v___x_1570_, v___x_1569_);
return v___x_1571_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__2(lean_object* v_a_1572_, lean_object* v___x_1573_, lean_object* v___x_1574_, lean_object* v___x_1575_, lean_object* v___x_1576_, lean_object* v___x_1577_, uint8_t v___x_1578_, lean_object* v_gi_1579_, lean_object* v___x_1580_, lean_object* v___y_1581_, lean_object* v___y_1582_, lean_object* v___y_1583_, lean_object* v___y_1584_, lean_object* v___y_1585_, lean_object* v___y_1586_, lean_object* v___y_1587_, lean_object* v___y_1588_){
_start:
{
lean_object* v___x_1590_; 
v___x_1590_ = l_Lean_Elab_Term_getLevelNames___redArg(v___y_1584_);
if (lean_obj_tag(v___x_1590_) == 0)
{
lean_object* v_a_1591_; lean_object* v___x_1592_; 
v_a_1591_ = lean_ctor_get(v___x_1590_, 0);
lean_inc(v_a_1591_);
lean_dec_ref_known(v___x_1590_, 1);
lean_inc(v_a_1572_);
v___x_1592_ = l_Lean_MVarId_getType(v_a_1572_, v___y_1585_, v___y_1586_, v___y_1587_, v___y_1588_);
if (lean_obj_tag(v___x_1592_) == 0)
{
lean_object* v_a_1593_; lean_object* v_lctx_1594_; lean_object* v___x_1595_; lean_object* v_a_1596_; lean_object* v___x_1597_; lean_object* v___x_1598_; lean_object* v___x_1599_; lean_object* v___x_1600_; lean_object* v___x_1601_; uint8_t v___x_1602_; lean_object* v_fst_1604_; lean_object* v_fst_1605_; lean_object* v_snd_1606_; lean_object* v___x_1642_; lean_object* v___x_1643_; 
v_a_1593_ = lean_ctor_get(v___x_1592_, 0);
lean_inc(v_a_1593_);
lean_dec_ref_known(v___x_1592_, 1);
v_lctx_1594_ = lean_ctor_get(v___y_1585_, 2);
v___x_1595_ = lp_Qq_Lean_instantiateMVars___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__1___redArg(v_a_1593_, v___y_1586_);
v_a_1596_ = lean_ctor_get(v___x_1595_, 0);
lean_inc(v_a_1596_);
lean_dec_ref(v___x_1595_);
v___x_1597_ = l_List_reverse___redArg(v_a_1591_);
v___x_1598_ = lean_box(0);
v___x_1599_ = lean_obj_once(&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__1, &lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__1_once, _init_lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__1);
v___x_1600_ = l_Lean_LocalContext_empty;
v___x_1601_ = ((lean_object*)(lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__2));
v___x_1602_ = 0;
v___x_1642_ = lean_alloc_ctor(0, 8, 1);
lean_ctor_set(v___x_1642_, 0, v___x_1598_);
lean_ctor_set(v___x_1642_, 1, v___x_1599_);
lean_ctor_set(v___x_1642_, 2, v___x_1599_);
lean_ctor_set(v___x_1642_, 3, v___x_1600_);
lean_ctor_set(v___x_1642_, 4, v___x_1599_);
lean_ctor_set(v___x_1642_, 5, v___x_1599_);
lean_ctor_set(v___x_1642_, 6, v___x_1601_);
lean_ctor_set(v___x_1642_, 7, v___x_1573_);
lean_ctor_set_uint8(v___x_1642_, sizeof(void*)*8, v___x_1602_);
v___x_1643_ = lp_Qq_Qq_Impl_quoteLCtx(v_lctx_1594_, v___x_1597_, v___x_1642_, v___y_1585_, v___y_1586_, v___y_1587_, v___y_1588_);
lean_dec(v___x_1597_);
if (lean_obj_tag(v___x_1643_) == 0)
{
lean_object* v_a_1644_; lean_object* v_fst_1645_; 
v_a_1644_ = lean_ctor_get(v___x_1643_, 0);
lean_inc(v_a_1644_);
lean_dec_ref_known(v___x_1643_, 1);
v_fst_1645_ = lean_ctor_get(v_a_1644_, 0);
lean_inc(v_fst_1645_);
if (lean_obj_tag(v_gi_1579_) == 0)
{
lean_object* v_fst_1646_; lean_object* v_snd_1647_; lean_object* v___x_1648_; 
lean_dec(v_a_1644_);
lean_dec(v_a_1596_);
lean_dec_ref(v___x_1580_);
lean_dec(v_a_1572_);
v_fst_1646_ = lean_ctor_get(v_fst_1645_, 0);
lean_inc(v_fst_1646_);
v_snd_1647_ = lean_ctor_get(v_fst_1645_, 1);
lean_inc(v_snd_1647_);
lean_dec(v_fst_1645_);
v___x_1648_ = lean_box(0);
v_fst_1604_ = v_fst_1646_;
v_fst_1605_ = v_snd_1647_;
v_snd_1606_ = v___x_1648_;
goto v___jp_1603_;
}
else
{
lean_object* v_snd_1649_; lean_object* v_fst_1650_; lean_object* v_snd_1651_; lean_object* v_val_1652_; lean_object* v___x_1654_; uint8_t v_isShared_1655_; uint8_t v_isSharedCheck_1699_; 
v_snd_1649_ = lean_ctor_get(v_a_1644_, 1);
lean_inc(v_snd_1649_);
lean_dec(v_a_1644_);
v_fst_1650_ = lean_ctor_get(v_fst_1645_, 0);
lean_inc(v_fst_1650_);
v_snd_1651_ = lean_ctor_get(v_fst_1645_, 1);
lean_inc(v_snd_1651_);
lean_dec(v_fst_1645_);
v_val_1652_ = lean_ctor_get(v_gi_1579_, 0);
v_isSharedCheck_1699_ = !lean_is_exclusive(v_gi_1579_);
if (v_isSharedCheck_1699_ == 0)
{
v___x_1654_ = v_gi_1579_;
v_isShared_1655_ = v_isSharedCheck_1699_;
goto v_resetjp_1653_;
}
else
{
lean_inc(v_val_1652_);
lean_dec(v_gi_1579_);
v___x_1654_ = lean_box(0);
v_isShared_1655_ = v_isSharedCheck_1699_;
goto v_resetjp_1653_;
}
v_resetjp_1653_:
{
lean_object* v___x_1656_; 
v___x_1656_ = lp_Qq_Qq_Impl_quoteExpr(v_a_1596_, v_snd_1649_, v___y_1585_, v___y_1586_, v___y_1587_, v___y_1588_);
if (lean_obj_tag(v___x_1656_) == 0)
{
lean_object* v_a_1657_; lean_object* v___x_1658_; 
v_a_1657_ = lean_ctor_get(v___x_1656_, 0);
lean_inc(v_a_1657_);
lean_dec_ref_known(v___x_1656_, 1);
v___x_1658_ = lp_Qq_Lean_mkFreshFVarId___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__4(v_snd_1649_, v___y_1585_, v___y_1586_, v___y_1587_, v___y_1588_);
if (lean_obj_tag(v___x_1658_) == 0)
{
lean_object* v_a_1659_; lean_object* v_fst_1660_; lean_object* v___x_1662_; uint8_t v_isShared_1663_; uint8_t v_isSharedCheck_1681_; 
v_a_1659_ = lean_ctor_get(v___x_1658_, 0);
lean_inc(v_a_1659_);
lean_dec_ref_known(v___x_1658_, 1);
v_fst_1660_ = lean_ctor_get(v_a_1659_, 0);
v_isSharedCheck_1681_ = !lean_is_exclusive(v_a_1659_);
if (v_isSharedCheck_1681_ == 0)
{
lean_object* v_unused_1682_; 
v_unused_1682_ = lean_ctor_get(v_a_1659_, 1);
lean_dec(v_unused_1682_);
v___x_1662_ = v_a_1659_;
v_isShared_1663_ = v_isSharedCheck_1681_;
goto v_resetjp_1661_;
}
else
{
lean_inc(v_fst_1660_);
lean_dec(v_a_1659_);
v___x_1662_ = lean_box(0);
v_isShared_1663_ = v_isSharedCheck_1681_;
goto v_resetjp_1661_;
}
v_resetjp_1661_:
{
lean_object* v___x_1664_; lean_object* v___x_1665_; lean_object* v___x_1666_; lean_object* v___x_1667_; lean_object* v___x_1668_; uint8_t v___x_1669_; uint8_t v___x_1670_; lean_object* v___x_1671_; lean_object* v___x_1672_; lean_object* v___x_1673_; lean_object* v___x_1674_; lean_object* v___x_1676_; 
v___x_1664_ = l_Lean_TSyntax_getId(v_val_1652_);
v___x_1665_ = ((lean_object*)(lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__8));
v___x_1666_ = l_Lean_Name_mkStr2(v___x_1580_, v___x_1665_);
v___x_1667_ = l_Lean_Expr_const___override(v___x_1666_, v___x_1598_);
v___x_1668_ = l_Lean_Expr_app___override(v___x_1667_, v_a_1657_);
v___x_1669_ = 0;
v___x_1670_ = 0;
lean_inc(v_fst_1660_);
v___x_1671_ = l_Lean_LocalContext_mkLocalDecl(v_fst_1650_, v_fst_1660_, v___x_1664_, v___x_1668_, v___x_1669_, v___x_1670_);
v___x_1672_ = l_Lean_Expr_mvar___override(v_a_1572_);
v___x_1673_ = lp_Qq_toExprExpr(v___x_1672_);
v___x_1674_ = lean_array_push(v_snd_1651_, v___x_1673_);
if (v_isShared_1663_ == 0)
{
lean_ctor_set(v___x_1662_, 1, v_fst_1660_);
lean_ctor_set(v___x_1662_, 0, v_val_1652_);
v___x_1676_ = v___x_1662_;
goto v_reusejp_1675_;
}
else
{
lean_object* v_reuseFailAlloc_1680_; 
v_reuseFailAlloc_1680_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1680_, 0, v_val_1652_);
lean_ctor_set(v_reuseFailAlloc_1680_, 1, v_fst_1660_);
v___x_1676_ = v_reuseFailAlloc_1680_;
goto v_reusejp_1675_;
}
v_reusejp_1675_:
{
lean_object* v___x_1678_; 
if (v_isShared_1655_ == 0)
{
lean_ctor_set(v___x_1654_, 0, v___x_1676_);
v___x_1678_ = v___x_1654_;
goto v_reusejp_1677_;
}
else
{
lean_object* v_reuseFailAlloc_1679_; 
v_reuseFailAlloc_1679_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1679_, 0, v___x_1676_);
v___x_1678_ = v_reuseFailAlloc_1679_;
goto v_reusejp_1677_;
}
v_reusejp_1677_:
{
v_fst_1604_ = v___x_1671_;
v_fst_1605_ = v___x_1674_;
v_snd_1606_ = v___x_1678_;
goto v___jp_1603_;
}
}
}
}
else
{
lean_object* v_a_1683_; lean_object* v___x_1685_; uint8_t v_isShared_1686_; uint8_t v_isSharedCheck_1690_; 
lean_dec(v_a_1657_);
lean_del_object(v___x_1654_);
lean_dec(v_val_1652_);
lean_dec(v_snd_1651_);
lean_dec(v_fst_1650_);
lean_dec(v___y_1588_);
lean_dec_ref(v___y_1587_);
lean_dec(v___y_1586_);
lean_dec_ref(v___y_1585_);
lean_dec(v___y_1584_);
lean_dec_ref(v___y_1583_);
lean_dec(v___y_1582_);
lean_dec_ref(v___y_1581_);
lean_dec_ref(v___x_1580_);
lean_dec(v___x_1577_);
lean_dec_ref(v___x_1576_);
lean_dec_ref(v___x_1575_);
lean_dec_ref(v___x_1574_);
lean_dec(v_a_1572_);
v_a_1683_ = lean_ctor_get(v___x_1658_, 0);
v_isSharedCheck_1690_ = !lean_is_exclusive(v___x_1658_);
if (v_isSharedCheck_1690_ == 0)
{
v___x_1685_ = v___x_1658_;
v_isShared_1686_ = v_isSharedCheck_1690_;
goto v_resetjp_1684_;
}
else
{
lean_inc(v_a_1683_);
lean_dec(v___x_1658_);
v___x_1685_ = lean_box(0);
v_isShared_1686_ = v_isSharedCheck_1690_;
goto v_resetjp_1684_;
}
v_resetjp_1684_:
{
lean_object* v___x_1688_; 
if (v_isShared_1686_ == 0)
{
v___x_1688_ = v___x_1685_;
goto v_reusejp_1687_;
}
else
{
lean_object* v_reuseFailAlloc_1689_; 
v_reuseFailAlloc_1689_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1689_, 0, v_a_1683_);
v___x_1688_ = v_reuseFailAlloc_1689_;
goto v_reusejp_1687_;
}
v_reusejp_1687_:
{
return v___x_1688_;
}
}
}
}
else
{
lean_object* v_a_1691_; lean_object* v___x_1693_; uint8_t v_isShared_1694_; uint8_t v_isSharedCheck_1698_; 
lean_del_object(v___x_1654_);
lean_dec(v_val_1652_);
lean_dec(v_snd_1651_);
lean_dec(v_fst_1650_);
lean_dec(v_snd_1649_);
lean_dec(v___y_1588_);
lean_dec_ref(v___y_1587_);
lean_dec(v___y_1586_);
lean_dec_ref(v___y_1585_);
lean_dec(v___y_1584_);
lean_dec_ref(v___y_1583_);
lean_dec(v___y_1582_);
lean_dec_ref(v___y_1581_);
lean_dec_ref(v___x_1580_);
lean_dec(v___x_1577_);
lean_dec_ref(v___x_1576_);
lean_dec_ref(v___x_1575_);
lean_dec_ref(v___x_1574_);
lean_dec(v_a_1572_);
v_a_1691_ = lean_ctor_get(v___x_1656_, 0);
v_isSharedCheck_1698_ = !lean_is_exclusive(v___x_1656_);
if (v_isSharedCheck_1698_ == 0)
{
v___x_1693_ = v___x_1656_;
v_isShared_1694_ = v_isSharedCheck_1698_;
goto v_resetjp_1692_;
}
else
{
lean_inc(v_a_1691_);
lean_dec(v___x_1656_);
v___x_1693_ = lean_box(0);
v_isShared_1694_ = v_isSharedCheck_1698_;
goto v_resetjp_1692_;
}
v_resetjp_1692_:
{
lean_object* v___x_1696_; 
if (v_isShared_1694_ == 0)
{
v___x_1696_ = v___x_1693_;
goto v_reusejp_1695_;
}
else
{
lean_object* v_reuseFailAlloc_1697_; 
v_reuseFailAlloc_1697_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1697_, 0, v_a_1691_);
v___x_1696_ = v_reuseFailAlloc_1697_;
goto v_reusejp_1695_;
}
v_reusejp_1695_:
{
return v___x_1696_;
}
}
}
}
}
}
else
{
lean_object* v_a_1700_; lean_object* v___x_1702_; uint8_t v_isShared_1703_; uint8_t v_isSharedCheck_1707_; 
lean_dec(v_a_1596_);
lean_dec(v___y_1588_);
lean_dec_ref(v___y_1587_);
lean_dec(v___y_1586_);
lean_dec_ref(v___y_1585_);
lean_dec(v___y_1584_);
lean_dec_ref(v___y_1583_);
lean_dec(v___y_1582_);
lean_dec_ref(v___y_1581_);
lean_dec_ref(v___x_1580_);
lean_dec(v_gi_1579_);
lean_dec(v___x_1577_);
lean_dec_ref(v___x_1576_);
lean_dec_ref(v___x_1575_);
lean_dec_ref(v___x_1574_);
lean_dec(v_a_1572_);
v_a_1700_ = lean_ctor_get(v___x_1643_, 0);
v_isSharedCheck_1707_ = !lean_is_exclusive(v___x_1643_);
if (v_isSharedCheck_1707_ == 0)
{
v___x_1702_ = v___x_1643_;
v_isShared_1703_ = v_isSharedCheck_1707_;
goto v_resetjp_1701_;
}
else
{
lean_inc(v_a_1700_);
lean_dec(v___x_1643_);
v___x_1702_ = lean_box(0);
v_isShared_1703_ = v_isSharedCheck_1707_;
goto v_resetjp_1701_;
}
v_resetjp_1701_:
{
lean_object* v___x_1705_; 
if (v_isShared_1703_ == 0)
{
v___x_1705_ = v___x_1702_;
goto v_reusejp_1704_;
}
else
{
lean_object* v_reuseFailAlloc_1706_; 
v_reuseFailAlloc_1706_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1706_, 0, v_a_1700_);
v___x_1705_ = v_reuseFailAlloc_1706_;
goto v_reusejp_1704_;
}
v_reusejp_1704_:
{
return v___x_1705_;
}
}
}
v___jp_1603_:
{
lean_object* v___x_1607_; lean_object* v___x_1608_; lean_object* v___f_1609_; lean_object* v___x_1610_; lean_object* v___x_1611_; lean_object* v___y_1612_; lean_object* v___x_1613_; 
v___x_1607_ = lean_box(v___x_1602_);
v___x_1608_ = lean_box(v___x_1578_);
lean_inc_ref(v___x_1574_);
v___f_1609_ = lean_alloc_closure((void*)(lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__0___boxed), 17, 7);
lean_closure_set(v___f_1609_, 0, v___x_1607_);
lean_closure_set(v___f_1609_, 1, v___x_1574_);
lean_closure_set(v___f_1609_, 2, v___x_1575_);
lean_closure_set(v___f_1609_, 3, v___x_1576_);
lean_closure_set(v___f_1609_, 4, v___x_1577_);
lean_closure_set(v___f_1609_, 5, v___x_1608_);
lean_closure_set(v___f_1609_, 6, v_fst_1605_);
v___x_1610_ = lean_box(v___x_1578_);
v___x_1611_ = lean_box(v___x_1602_);
v___y_1612_ = lean_alloc_closure((void*)(lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__1___boxed), 13, 4);
lean_closure_set(v___y_1612_, 0, v_snd_1606_);
lean_closure_set(v___y_1612_, 1, v___x_1610_);
lean_closure_set(v___y_1612_, 2, v___x_1611_);
lean_closure_set(v___y_1612_, 3, v___f_1609_);
v___x_1613_ = lp_Qq_Lean_Meta_withLCtx___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__3___redArg(v_fst_1604_, v___x_1601_, v___y_1612_, v___y_1581_, v___y_1582_, v___y_1583_, v___y_1584_, v___y_1585_, v___y_1586_, v___y_1587_, v___y_1588_);
if (lean_obj_tag(v___x_1613_) == 0)
{
lean_object* v_a_1614_; lean_object* v___x_1615_; lean_object* v___x_1616_; lean_object* v___x_1617_; lean_object* v___x_1618_; lean_object* v___x_1619_; lean_object* v___x_1620_; lean_object* v___x_1621_; uint8_t v___x_1622_; lean_object* v___x_1623_; 
v_a_1614_ = lean_ctor_get(v___x_1613_, 0);
lean_inc(v_a_1614_);
lean_dec_ref_known(v___x_1613_, 1);
v___x_1615_ = ((lean_object*)(lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__1));
v___x_1616_ = ((lean_object*)(lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__0));
v___x_1617_ = ((lean_object*)(lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_unsafe__6___closed__1));
v___x_1618_ = l_Lean_Name_mkStr4(v___x_1574_, v___x_1615_, v___x_1616_, v___x_1617_);
v___x_1619_ = l_Lean_Expr_const___override(v___x_1618_, v___x_1598_);
v___x_1620_ = lean_obj_once(&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__2___closed__0, &lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__2___closed__0_once, _init_lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__2___closed__0);
v___x_1621_ = l_Lean_Expr_app___override(v___x_1619_, v___x_1620_);
v___x_1622_ = 1;
v___x_1623_ = l_Lean_Meta_evalExpr___redArg(v___x_1621_, v_a_1614_, v___x_1622_, v___x_1578_, v___y_1585_, v___y_1586_, v___y_1587_, v___y_1588_);
if (lean_obj_tag(v___x_1623_) == 0)
{
lean_object* v_a_1624_; lean_object* v___x_1625_; 
v_a_1624_ = lean_ctor_get(v___x_1623_, 0);
lean_inc(v_a_1624_);
lean_dec_ref_known(v___x_1623_, 1);
v___x_1625_ = lean_apply_9(v_a_1624_, v___y_1581_, v___y_1582_, v___y_1583_, v___y_1584_, v___y_1585_, v___y_1586_, v___y_1587_, v___y_1588_, lean_box(0));
return v___x_1625_;
}
else
{
lean_object* v_a_1626_; lean_object* v___x_1628_; uint8_t v_isShared_1629_; uint8_t v_isSharedCheck_1633_; 
lean_dec(v___y_1588_);
lean_dec_ref(v___y_1587_);
lean_dec(v___y_1586_);
lean_dec_ref(v___y_1585_);
lean_dec(v___y_1584_);
lean_dec_ref(v___y_1583_);
lean_dec(v___y_1582_);
lean_dec_ref(v___y_1581_);
v_a_1626_ = lean_ctor_get(v___x_1623_, 0);
v_isSharedCheck_1633_ = !lean_is_exclusive(v___x_1623_);
if (v_isSharedCheck_1633_ == 0)
{
v___x_1628_ = v___x_1623_;
v_isShared_1629_ = v_isSharedCheck_1633_;
goto v_resetjp_1627_;
}
else
{
lean_inc(v_a_1626_);
lean_dec(v___x_1623_);
v___x_1628_ = lean_box(0);
v_isShared_1629_ = v_isSharedCheck_1633_;
goto v_resetjp_1627_;
}
v_resetjp_1627_:
{
lean_object* v___x_1631_; 
if (v_isShared_1629_ == 0)
{
v___x_1631_ = v___x_1628_;
goto v_reusejp_1630_;
}
else
{
lean_object* v_reuseFailAlloc_1632_; 
v_reuseFailAlloc_1632_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1632_, 0, v_a_1626_);
v___x_1631_ = v_reuseFailAlloc_1632_;
goto v_reusejp_1630_;
}
v_reusejp_1630_:
{
return v___x_1631_;
}
}
}
}
else
{
lean_object* v_a_1634_; lean_object* v___x_1636_; uint8_t v_isShared_1637_; uint8_t v_isSharedCheck_1641_; 
lean_dec(v___y_1588_);
lean_dec_ref(v___y_1587_);
lean_dec(v___y_1586_);
lean_dec_ref(v___y_1585_);
lean_dec(v___y_1584_);
lean_dec_ref(v___y_1583_);
lean_dec(v___y_1582_);
lean_dec_ref(v___y_1581_);
lean_dec_ref(v___x_1574_);
v_a_1634_ = lean_ctor_get(v___x_1613_, 0);
v_isSharedCheck_1641_ = !lean_is_exclusive(v___x_1613_);
if (v_isSharedCheck_1641_ == 0)
{
v___x_1636_ = v___x_1613_;
v_isShared_1637_ = v_isSharedCheck_1641_;
goto v_resetjp_1635_;
}
else
{
lean_inc(v_a_1634_);
lean_dec(v___x_1613_);
v___x_1636_ = lean_box(0);
v_isShared_1637_ = v_isSharedCheck_1641_;
goto v_resetjp_1635_;
}
v_resetjp_1635_:
{
lean_object* v___x_1639_; 
if (v_isShared_1637_ == 0)
{
v___x_1639_ = v___x_1636_;
goto v_reusejp_1638_;
}
else
{
lean_object* v_reuseFailAlloc_1640_; 
v_reuseFailAlloc_1640_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1640_, 0, v_a_1634_);
v___x_1639_ = v_reuseFailAlloc_1640_;
goto v_reusejp_1638_;
}
v_reusejp_1638_:
{
return v___x_1639_;
}
}
}
}
}
else
{
lean_object* v_a_1708_; lean_object* v___x_1710_; uint8_t v_isShared_1711_; uint8_t v_isSharedCheck_1715_; 
lean_dec(v_a_1591_);
lean_dec(v___y_1588_);
lean_dec_ref(v___y_1587_);
lean_dec(v___y_1586_);
lean_dec_ref(v___y_1585_);
lean_dec(v___y_1584_);
lean_dec_ref(v___y_1583_);
lean_dec(v___y_1582_);
lean_dec_ref(v___y_1581_);
lean_dec_ref(v___x_1580_);
lean_dec(v_gi_1579_);
lean_dec(v___x_1577_);
lean_dec_ref(v___x_1576_);
lean_dec_ref(v___x_1575_);
lean_dec_ref(v___x_1574_);
lean_dec(v___x_1573_);
lean_dec(v_a_1572_);
v_a_1708_ = lean_ctor_get(v___x_1592_, 0);
v_isSharedCheck_1715_ = !lean_is_exclusive(v___x_1592_);
if (v_isSharedCheck_1715_ == 0)
{
v___x_1710_ = v___x_1592_;
v_isShared_1711_ = v_isSharedCheck_1715_;
goto v_resetjp_1709_;
}
else
{
lean_inc(v_a_1708_);
lean_dec(v___x_1592_);
v___x_1710_ = lean_box(0);
v_isShared_1711_ = v_isSharedCheck_1715_;
goto v_resetjp_1709_;
}
v_resetjp_1709_:
{
lean_object* v___x_1713_; 
if (v_isShared_1711_ == 0)
{
v___x_1713_ = v___x_1710_;
goto v_reusejp_1712_;
}
else
{
lean_object* v_reuseFailAlloc_1714_; 
v_reuseFailAlloc_1714_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1714_, 0, v_a_1708_);
v___x_1713_ = v_reuseFailAlloc_1714_;
goto v_reusejp_1712_;
}
v_reusejp_1712_:
{
return v___x_1713_;
}
}
}
}
else
{
lean_object* v_a_1716_; lean_object* v___x_1718_; uint8_t v_isShared_1719_; uint8_t v_isSharedCheck_1723_; 
lean_dec(v___y_1588_);
lean_dec_ref(v___y_1587_);
lean_dec(v___y_1586_);
lean_dec_ref(v___y_1585_);
lean_dec(v___y_1584_);
lean_dec_ref(v___y_1583_);
lean_dec(v___y_1582_);
lean_dec_ref(v___y_1581_);
lean_dec_ref(v___x_1580_);
lean_dec(v_gi_1579_);
lean_dec(v___x_1577_);
lean_dec_ref(v___x_1576_);
lean_dec_ref(v___x_1575_);
lean_dec_ref(v___x_1574_);
lean_dec(v___x_1573_);
lean_dec(v_a_1572_);
v_a_1716_ = lean_ctor_get(v___x_1590_, 0);
v_isSharedCheck_1723_ = !lean_is_exclusive(v___x_1590_);
if (v_isSharedCheck_1723_ == 0)
{
v___x_1718_ = v___x_1590_;
v_isShared_1719_ = v_isSharedCheck_1723_;
goto v_resetjp_1717_;
}
else
{
lean_inc(v_a_1716_);
lean_dec(v___x_1590_);
v___x_1718_ = lean_box(0);
v_isShared_1719_ = v_isSharedCheck_1723_;
goto v_resetjp_1717_;
}
v_resetjp_1717_:
{
lean_object* v___x_1721_; 
if (v_isShared_1719_ == 0)
{
v___x_1721_ = v___x_1718_;
goto v_reusejp_1720_;
}
else
{
lean_object* v_reuseFailAlloc_1722_; 
v_reuseFailAlloc_1722_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1722_, 0, v_a_1716_);
v___x_1721_ = v_reuseFailAlloc_1722_;
goto v_reusejp_1720_;
}
v_reusejp_1720_:
{
return v___x_1721_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__2___boxed(lean_object** _args){
lean_object* v_a_1724_ = _args[0];
lean_object* v___x_1725_ = _args[1];
lean_object* v___x_1726_ = _args[2];
lean_object* v___x_1727_ = _args[3];
lean_object* v___x_1728_ = _args[4];
lean_object* v___x_1729_ = _args[5];
lean_object* v___x_1730_ = _args[6];
lean_object* v_gi_1731_ = _args[7];
lean_object* v___x_1732_ = _args[8];
lean_object* v___y_1733_ = _args[9];
lean_object* v___y_1734_ = _args[10];
lean_object* v___y_1735_ = _args[11];
lean_object* v___y_1736_ = _args[12];
lean_object* v___y_1737_ = _args[13];
lean_object* v___y_1738_ = _args[14];
lean_object* v___y_1739_ = _args[15];
lean_object* v___y_1740_ = _args[16];
lean_object* v___y_1741_ = _args[17];
_start:
{
uint8_t v___x_22929__boxed_1742_; lean_object* v_res_1743_; 
v___x_22929__boxed_1742_ = lean_unbox(v___x_1730_);
v_res_1743_ = lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__2(v_a_1724_, v___x_1725_, v___x_1726_, v___x_1727_, v___x_1728_, v___x_1729_, v___x_22929__boxed_1742_, v_gi_1731_, v___x_1732_, v___y_1733_, v___y_1734_, v___y_1735_, v___y_1736_, v___y_1737_, v___y_1738_, v___y_1739_, v___y_1740_);
return v_res_1743_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__6_spec__7(lean_object* v_msgData_1744_, lean_object* v___y_1745_, lean_object* v___y_1746_, lean_object* v___y_1747_, lean_object* v___y_1748_){
_start:
{
lean_object* v___x_1750_; lean_object* v_env_1751_; lean_object* v___x_1752_; lean_object* v_mctx_1753_; lean_object* v_lctx_1754_; lean_object* v_options_1755_; lean_object* v___x_1756_; lean_object* v___x_1757_; lean_object* v___x_1758_; 
v___x_1750_ = lean_st_ref_get(v___y_1748_);
v_env_1751_ = lean_ctor_get(v___x_1750_, 0);
lean_inc_ref(v_env_1751_);
lean_dec(v___x_1750_);
v___x_1752_ = lean_st_ref_get(v___y_1746_);
v_mctx_1753_ = lean_ctor_get(v___x_1752_, 0);
lean_inc_ref(v_mctx_1753_);
lean_dec(v___x_1752_);
v_lctx_1754_ = lean_ctor_get(v___y_1745_, 2);
v_options_1755_ = lean_ctor_get(v___y_1747_, 2);
lean_inc_ref(v_options_1755_);
lean_inc_ref(v_lctx_1754_);
v___x_1756_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1756_, 0, v_env_1751_);
lean_ctor_set(v___x_1756_, 1, v_mctx_1753_);
lean_ctor_set(v___x_1756_, 2, v_lctx_1754_);
lean_ctor_set(v___x_1756_, 3, v_options_1755_);
v___x_1757_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1757_, 0, v___x_1756_);
lean_ctor_set(v___x_1757_, 1, v_msgData_1744_);
v___x_1758_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1758_, 0, v___x_1757_);
return v___x_1758_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__6_spec__7___boxed(lean_object* v_msgData_1759_, lean_object* v___y_1760_, lean_object* v___y_1761_, lean_object* v___y_1762_, lean_object* v___y_1763_, lean_object* v___y_1764_){
_start:
{
lean_object* v_res_1765_; 
v_res_1765_ = lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__6_spec__7(v_msgData_1759_, v___y_1760_, v___y_1761_, v___y_1762_, v___y_1763_);
lean_dec(v___y_1763_);
lean_dec_ref(v___y_1762_);
lean_dec(v___y_1761_);
lean_dec_ref(v___y_1760_);
return v_res_1765_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__6___redArg(lean_object* v_msg_1766_, lean_object* v___y_1767_, lean_object* v___y_1768_, lean_object* v___y_1769_, lean_object* v___y_1770_){
_start:
{
lean_object* v_ref_1772_; lean_object* v___x_1773_; lean_object* v_a_1774_; lean_object* v___x_1776_; uint8_t v_isShared_1777_; uint8_t v_isSharedCheck_1782_; 
v_ref_1772_ = lean_ctor_get(v___y_1769_, 5);
v___x_1773_ = lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__6_spec__7(v_msg_1766_, v___y_1767_, v___y_1768_, v___y_1769_, v___y_1770_);
v_a_1774_ = lean_ctor_get(v___x_1773_, 0);
v_isSharedCheck_1782_ = !lean_is_exclusive(v___x_1773_);
if (v_isSharedCheck_1782_ == 0)
{
v___x_1776_ = v___x_1773_;
v_isShared_1777_ = v_isSharedCheck_1782_;
goto v_resetjp_1775_;
}
else
{
lean_inc(v_a_1774_);
lean_dec(v___x_1773_);
v___x_1776_ = lean_box(0);
v_isShared_1777_ = v_isSharedCheck_1782_;
goto v_resetjp_1775_;
}
v_resetjp_1775_:
{
lean_object* v___x_1778_; lean_object* v___x_1780_; 
lean_inc(v_ref_1772_);
v___x_1778_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1778_, 0, v_ref_1772_);
lean_ctor_set(v___x_1778_, 1, v_a_1774_);
if (v_isShared_1777_ == 0)
{
lean_ctor_set_tag(v___x_1776_, 1);
lean_ctor_set(v___x_1776_, 0, v___x_1778_);
v___x_1780_ = v___x_1776_;
goto v_reusejp_1779_;
}
else
{
lean_object* v_reuseFailAlloc_1781_; 
v_reuseFailAlloc_1781_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1781_, 0, v___x_1778_);
v___x_1780_ = v_reuseFailAlloc_1781_;
goto v_reusejp_1779_;
}
v_reusejp_1779_:
{
return v___x_1780_;
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__6___redArg___boxed(lean_object* v_msg_1783_, lean_object* v___y_1784_, lean_object* v___y_1785_, lean_object* v___y_1786_, lean_object* v___y_1787_, lean_object* v___y_1788_){
_start:
{
lean_object* v_res_1789_; 
v_res_1789_ = lp_Qq_Lean_throwError___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__6___redArg(v_msg_1783_, v___y_1784_, v___y_1785_, v___y_1786_, v___y_1787_);
lean_dec(v___y_1787_);
lean_dec_ref(v___y_1786_);
lean_dec(v___y_1785_);
lean_dec_ref(v___y_1784_);
return v_res_1789_;
}
}
static lean_object* _init_lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___closed__1(void){
_start:
{
lean_object* v___x_1791_; lean_object* v___x_1792_; 
v___x_1791_ = ((lean_object*)(lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___closed__0));
v___x_1792_ = l_Lean_stringToMessageData(v___x_1791_);
return v___x_1792_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1(lean_object* v_x_1793_, lean_object* v_a_1794_, lean_object* v_a_1795_, lean_object* v_a_1796_, lean_object* v_a_1797_, lean_object* v_a_1798_, lean_object* v_a_1799_, lean_object* v_a_1800_, lean_object* v_a_1801_){
_start:
{
lean_object* v___x_1803_; lean_object* v___x_1804_; uint8_t v___x_1805_; lean_object* v___y_1807_; lean_object* v___y_1808_; lean_object* v___y_1809_; lean_object* v___y_1810_; lean_object* v___y_1811_; lean_object* v___y_1812_; lean_object* v___y_1813_; lean_object* v___y_1814_; lean_object* v___y_1815_; lean_object* v___y_1816_; lean_object* v___y_1817_; lean_object* v___y_1818_; lean_object* v___y_1819_; lean_object* v___y_1820_; lean_object* v___y_1821_; lean_object* v___y_1835_; lean_object* v___y_1836_; lean_object* v___y_1837_; lean_object* v___y_1838_; lean_object* v___y_1839_; lean_object* v___y_1840_; lean_object* v___y_1841_; lean_object* v___y_1842_; lean_object* v___y_1843_; lean_object* v___y_1844_; lean_object* v___y_1845_; lean_object* v___y_1846_; lean_object* v___y_1847_; lean_object* v___y_1848_; lean_object* v___y_1849_; lean_object* v___y_1850_; uint8_t v___y_1851_; lean_object* v_gi_1856_; lean_object* v___y_1857_; lean_object* v___y_1858_; lean_object* v___y_1859_; lean_object* v___y_1860_; lean_object* v___y_1861_; lean_object* v___y_1862_; lean_object* v___y_1863_; lean_object* v___y_1864_; 
v___x_1803_ = ((lean_object*)(lp_Qq_Qq_termBy__elabq___00__closed__0));
v___x_1804_ = ((lean_object*)(lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__1));
lean_inc(v_x_1793_);
v___x_1805_ = l_Lean_Syntax_isOfKind(v_x_1793_, v___x_1804_);
if (v___x_1805_ == 0)
{
lean_object* v___x_1885_; 
lean_dec(v_x_1793_);
v___x_1885_ = lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__0___redArg();
return v___x_1885_;
}
else
{
lean_object* v___x_1886_; lean_object* v___x_1887_; uint8_t v___x_1888_; 
v___x_1886_ = lean_unsigned_to_nat(1u);
v___x_1887_ = l_Lean_Syntax_getArg(v_x_1793_, v___x_1886_);
v___x_1888_ = l_Lean_Syntax_isNone(v___x_1887_);
if (v___x_1888_ == 0)
{
lean_object* v___x_1889_; uint8_t v___x_1890_; 
v___x_1889_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_1887_);
v___x_1890_ = l_Lean_Syntax_matchesNull(v___x_1887_, v___x_1889_);
if (v___x_1890_ == 0)
{
lean_object* v___x_1891_; 
lean_dec(v___x_1887_);
lean_dec(v_x_1793_);
v___x_1891_ = lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__0___redArg();
return v___x_1891_;
}
else
{
lean_object* v___x_1892_; lean_object* v_gi_1893_; lean_object* v___x_1894_; uint8_t v___x_1895_; 
v___x_1892_ = lean_unsigned_to_nat(0u);
v_gi_1893_ = l_Lean_Syntax_getArg(v___x_1887_, v___x_1892_);
lean_dec(v___x_1887_);
v___x_1894_ = ((lean_object*)(lp_Qq_Qq_tacticRun__tacq___x3d_x3e___00__closed__9));
lean_inc(v_gi_1893_);
v___x_1895_ = l_Lean_Syntax_isOfKind(v_gi_1893_, v___x_1894_);
if (v___x_1895_ == 0)
{
lean_object* v___x_1896_; 
lean_dec(v_gi_1893_);
lean_dec(v_x_1793_);
v___x_1896_ = lp_Qq_Lean_Elab_throwUnsupportedSyntax___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__0___redArg();
return v___x_1896_;
}
else
{
lean_object* v___x_1897_; 
v___x_1897_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1897_, 0, v_gi_1893_);
v_gi_1856_ = v___x_1897_;
v___y_1857_ = v_a_1794_;
v___y_1858_ = v_a_1795_;
v___y_1859_ = v_a_1796_;
v___y_1860_ = v_a_1797_;
v___y_1861_ = v_a_1798_;
v___y_1862_ = v_a_1799_;
v___y_1863_ = v_a_1800_;
v___y_1864_ = v_a_1801_;
goto v___jp_1855_;
}
}
}
else
{
lean_object* v___x_1898_; 
lean_dec(v___x_1887_);
v___x_1898_ = lean_box(0);
v_gi_1856_ = v___x_1898_;
v___y_1857_ = v_a_1794_;
v___y_1858_ = v_a_1795_;
v___y_1859_ = v_a_1796_;
v___y_1860_ = v_a_1797_;
v___y_1861_ = v_a_1798_;
v___y_1862_ = v_a_1799_;
v___y_1863_ = v_a_1800_;
v___y_1864_ = v_a_1801_;
goto v___jp_1855_;
}
}
v___jp_1806_:
{
if (lean_obj_tag(v___y_1821_) == 0)
{
lean_object* v_a_1822_; lean_object* v___x_1823_; lean_object* v___f_1824_; lean_object* v___x_1825_; 
v_a_1822_ = lean_ctor_get(v___y_1821_, 0);
lean_inc_n(v_a_1822_, 2);
lean_dec_ref_known(v___y_1821_, 1);
v___x_1823_ = lean_box(v___x_1805_);
lean_inc_ref(v___y_1808_);
lean_inc_ref(v___y_1809_);
lean_inc_ref(v___y_1810_);
lean_inc(v___y_1812_);
v___f_1824_ = lean_alloc_closure((void*)(lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___lam__2___boxed), 18, 9);
lean_closure_set(v___f_1824_, 0, v_a_1822_);
lean_closure_set(v___f_1824_, 1, v___y_1812_);
lean_closure_set(v___f_1824_, 2, v___y_1810_);
lean_closure_set(v___f_1824_, 3, v___y_1809_);
lean_closure_set(v___f_1824_, 4, v___y_1808_);
lean_closure_set(v___f_1824_, 5, v___y_1807_);
lean_closure_set(v___f_1824_, 6, v___x_1823_);
lean_closure_set(v___f_1824_, 7, v___y_1811_);
lean_closure_set(v___f_1824_, 8, v___x_1803_);
v___x_1825_ = lp_Qq_Lean_MVarId_withContext___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__5___redArg(v_a_1822_, v___f_1824_, v___y_1820_, v___y_1819_, v___y_1818_, v___y_1813_, v___y_1817_, v___y_1816_, v___y_1814_, v___y_1815_);
return v___x_1825_;
}
else
{
lean_object* v_a_1826_; lean_object* v___x_1828_; uint8_t v_isShared_1829_; uint8_t v_isSharedCheck_1833_; 
lean_dec(v___y_1811_);
lean_dec(v___y_1807_);
v_a_1826_ = lean_ctor_get(v___y_1821_, 0);
v_isSharedCheck_1833_ = !lean_is_exclusive(v___y_1821_);
if (v_isSharedCheck_1833_ == 0)
{
v___x_1828_ = v___y_1821_;
v_isShared_1829_ = v_isSharedCheck_1833_;
goto v_resetjp_1827_;
}
else
{
lean_inc(v_a_1826_);
lean_dec(v___y_1821_);
v___x_1828_ = lean_box(0);
v_isShared_1829_ = v_isSharedCheck_1833_;
goto v_resetjp_1827_;
}
v_resetjp_1827_:
{
lean_object* v___x_1831_; 
if (v_isShared_1829_ == 0)
{
v___x_1831_ = v___x_1828_;
goto v_reusejp_1830_;
}
else
{
lean_object* v_reuseFailAlloc_1832_; 
v_reuseFailAlloc_1832_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1832_, 0, v_a_1826_);
v___x_1831_ = v_reuseFailAlloc_1832_;
goto v_reusejp_1830_;
}
v_reusejp_1830_:
{
return v___x_1831_;
}
}
}
}
v___jp_1834_:
{
if (v___y_1851_ == 0)
{
lean_object* v___x_1852_; 
lean_dec_ref(v___y_1849_);
v___x_1852_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v___y_1842_, v___y_1851_, v___y_1848_, v___y_1847_, v___y_1841_, v___y_1846_, v___y_1845_, v___y_1843_, v___y_1844_);
if (lean_obj_tag(v___x_1852_) == 0)
{
lean_object* v___x_1853_; lean_object* v___x_1854_; 
lean_dec_ref_known(v___x_1852_, 1);
v___x_1853_ = lean_obj_once(&lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___closed__1, &lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___closed__1_once, _init_lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___closed__1);
v___x_1854_ = lp_Qq_Lean_throwError___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__6___redArg(v___x_1853_, v___y_1846_, v___y_1845_, v___y_1843_, v___y_1844_);
v___y_1807_ = v___y_1835_;
v___y_1808_ = v___y_1836_;
v___y_1809_ = v___y_1837_;
v___y_1810_ = v___y_1838_;
v___y_1811_ = v___y_1839_;
v___y_1812_ = v___y_1840_;
v___y_1813_ = v___y_1841_;
v___y_1814_ = v___y_1843_;
v___y_1815_ = v___y_1844_;
v___y_1816_ = v___y_1845_;
v___y_1817_ = v___y_1846_;
v___y_1818_ = v___y_1847_;
v___y_1819_ = v___y_1848_;
v___y_1820_ = v___y_1850_;
v___y_1821_ = v___x_1854_;
goto v___jp_1806_;
}
else
{
lean_dec(v___y_1839_);
lean_dec(v___y_1835_);
return v___x_1852_;
}
}
else
{
lean_dec_ref(v___y_1842_);
v___y_1807_ = v___y_1835_;
v___y_1808_ = v___y_1836_;
v___y_1809_ = v___y_1837_;
v___y_1810_ = v___y_1838_;
v___y_1811_ = v___y_1839_;
v___y_1812_ = v___y_1840_;
v___y_1813_ = v___y_1841_;
v___y_1814_ = v___y_1843_;
v___y_1815_ = v___y_1844_;
v___y_1816_ = v___y_1845_;
v___y_1817_ = v___y_1846_;
v___y_1818_ = v___y_1847_;
v___y_1819_ = v___y_1848_;
v___y_1820_ = v___y_1850_;
v___y_1821_ = v___y_1849_;
goto v___jp_1806_;
}
}
v___jp_1855_:
{
lean_object* v___x_1865_; 
v___x_1865_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_1858_, v___y_1860_, v___y_1862_, v___y_1864_);
if (lean_obj_tag(v___x_1865_) == 0)
{
lean_object* v_a_1866_; lean_object* v___x_1867_; lean_object* v___x_1868_; lean_object* v___x_1869_; lean_object* v___x_1870_; lean_object* v___x_1871_; lean_object* v___x_1872_; lean_object* v___x_1873_; 
v_a_1866_ = lean_ctor_get(v___x_1865_, 0);
lean_inc(v_a_1866_);
lean_dec_ref_known(v___x_1865_, 1);
v___x_1867_ = lean_unsigned_to_nat(2u);
v___x_1868_ = l_Lean_Syntax_getArg(v_x_1793_, v___x_1867_);
lean_dec(v_x_1793_);
v___x_1869_ = ((lean_object*)(lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__0));
v___x_1870_ = ((lean_object*)(lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1___lam__2___closed__4));
v___x_1871_ = ((lean_object*)(lp_Qq___private_Qq_Commands_0__Qq___aux__Qq__Commands______elabRules__Qq__termBy__elabq____1_unsafe__3___closed__2));
v___x_1872_ = lean_box(0);
v___x_1873_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_1858_, v___y_1861_, v___y_1862_, v___y_1863_, v___y_1864_);
if (lean_obj_tag(v___x_1873_) == 0)
{
lean_dec(v_a_1866_);
v___y_1807_ = v___x_1868_;
v___y_1808_ = v___x_1871_;
v___y_1809_ = v___x_1870_;
v___y_1810_ = v___x_1869_;
v___y_1811_ = v_gi_1856_;
v___y_1812_ = v___x_1872_;
v___y_1813_ = v___y_1860_;
v___y_1814_ = v___y_1863_;
v___y_1815_ = v___y_1864_;
v___y_1816_ = v___y_1862_;
v___y_1817_ = v___y_1861_;
v___y_1818_ = v___y_1859_;
v___y_1819_ = v___y_1858_;
v___y_1820_ = v___y_1857_;
v___y_1821_ = v___x_1873_;
goto v___jp_1806_;
}
else
{
lean_object* v_a_1874_; uint8_t v___x_1875_; 
v_a_1874_ = lean_ctor_get(v___x_1873_, 0);
lean_inc(v_a_1874_);
v___x_1875_ = l_Lean_Exception_isInterrupt(v_a_1874_);
if (v___x_1875_ == 0)
{
uint8_t v___x_1876_; 
v___x_1876_ = l_Lean_Exception_isRuntime(v_a_1874_);
v___y_1835_ = v___x_1868_;
v___y_1836_ = v___x_1871_;
v___y_1837_ = v___x_1870_;
v___y_1838_ = v___x_1869_;
v___y_1839_ = v_gi_1856_;
v___y_1840_ = v___x_1872_;
v___y_1841_ = v___y_1860_;
v___y_1842_ = v_a_1866_;
v___y_1843_ = v___y_1863_;
v___y_1844_ = v___y_1864_;
v___y_1845_ = v___y_1862_;
v___y_1846_ = v___y_1861_;
v___y_1847_ = v___y_1859_;
v___y_1848_ = v___y_1858_;
v___y_1849_ = v___x_1873_;
v___y_1850_ = v___y_1857_;
v___y_1851_ = v___x_1876_;
goto v___jp_1834_;
}
else
{
lean_dec(v_a_1874_);
v___y_1835_ = v___x_1868_;
v___y_1836_ = v___x_1871_;
v___y_1837_ = v___x_1870_;
v___y_1838_ = v___x_1869_;
v___y_1839_ = v_gi_1856_;
v___y_1840_ = v___x_1872_;
v___y_1841_ = v___y_1860_;
v___y_1842_ = v_a_1866_;
v___y_1843_ = v___y_1863_;
v___y_1844_ = v___y_1864_;
v___y_1845_ = v___y_1862_;
v___y_1846_ = v___y_1861_;
v___y_1847_ = v___y_1859_;
v___y_1848_ = v___y_1858_;
v___y_1849_ = v___x_1873_;
v___y_1850_ = v___y_1857_;
v___y_1851_ = v___x_1875_;
goto v___jp_1834_;
}
}
}
else
{
lean_object* v_a_1877_; lean_object* v___x_1879_; uint8_t v_isShared_1880_; uint8_t v_isSharedCheck_1884_; 
lean_dec(v_gi_1856_);
lean_dec(v_x_1793_);
v_a_1877_ = lean_ctor_get(v___x_1865_, 0);
v_isSharedCheck_1884_ = !lean_is_exclusive(v___x_1865_);
if (v_isSharedCheck_1884_ == 0)
{
v___x_1879_ = v___x_1865_;
v_isShared_1880_ = v_isSharedCheck_1884_;
goto v_resetjp_1878_;
}
else
{
lean_inc(v_a_1877_);
lean_dec(v___x_1865_);
v___x_1879_ = lean_box(0);
v_isShared_1880_ = v_isSharedCheck_1884_;
goto v_resetjp_1878_;
}
v_resetjp_1878_:
{
lean_object* v___x_1882_; 
if (v_isShared_1880_ == 0)
{
v___x_1882_ = v___x_1879_;
goto v_reusejp_1881_;
}
else
{
lean_object* v_reuseFailAlloc_1883_; 
v_reuseFailAlloc_1883_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1883_, 0, v_a_1877_);
v___x_1882_ = v_reuseFailAlloc_1883_;
goto v_reusejp_1881_;
}
v_reusejp_1881_:
{
return v___x_1882_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1___boxed(lean_object* v_x_1899_, lean_object* v_a_1900_, lean_object* v_a_1901_, lean_object* v_a_1902_, lean_object* v_a_1903_, lean_object* v_a_1904_, lean_object* v_a_1905_, lean_object* v_a_1906_, lean_object* v_a_1907_, lean_object* v_a_1908_){
_start:
{
lean_object* v_res_1909_; 
v_res_1909_ = lp_Qq_Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1(v_x_1899_, v_a_1900_, v_a_1901_, v_a_1902_, v_a_1903_, v_a_1904_, v_a_1905_, v_a_1906_, v_a_1907_);
lean_dec(v_a_1907_);
lean_dec_ref(v_a_1906_);
lean_dec(v_a_1905_);
lean_dec_ref(v_a_1904_);
lean_dec(v_a_1903_);
lean_dec_ref(v_a_1902_);
lean_dec(v_a_1901_);
lean_dec_ref(v_a_1900_);
return v_res_1909_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_mkFreshId___at___00Lean_mkFreshFVarId___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__4_spec__4(lean_object* v___y_1910_, lean_object* v___y_1911_, lean_object* v___y_1912_, lean_object* v___y_1913_, lean_object* v___y_1914_){
_start:
{
lean_object* v___x_1916_; 
v___x_1916_ = lp_Qq_Lean_mkFreshId___at___00Lean_mkFreshFVarId___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__4_spec__4___redArg(v___y_1910_, v___y_1914_);
return v___x_1916_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_mkFreshId___at___00Lean_mkFreshFVarId___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__4_spec__4___boxed(lean_object* v___y_1917_, lean_object* v___y_1918_, lean_object* v___y_1919_, lean_object* v___y_1920_, lean_object* v___y_1921_, lean_object* v___y_1922_){
_start:
{
lean_object* v_res_1923_; 
v_res_1923_ = lp_Qq_Lean_mkFreshId___at___00Lean_mkFreshFVarId___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__4_spec__4(v___y_1917_, v___y_1918_, v___y_1919_, v___y_1920_, v___y_1921_);
lean_dec(v___y_1921_);
lean_dec_ref(v___y_1920_);
lean_dec(v___y_1919_);
lean_dec_ref(v___y_1918_);
return v_res_1923_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__6(lean_object* v_00_u03b1_1924_, lean_object* v_msg_1925_, lean_object* v___y_1926_, lean_object* v___y_1927_, lean_object* v___y_1928_, lean_object* v___y_1929_, lean_object* v___y_1930_, lean_object* v___y_1931_, lean_object* v___y_1932_, lean_object* v___y_1933_){
_start:
{
lean_object* v___x_1935_; 
v___x_1935_ = lp_Qq_Lean_throwError___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__6___redArg(v_msg_1925_, v___y_1930_, v___y_1931_, v___y_1932_, v___y_1933_);
return v___x_1935_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__6___boxed(lean_object* v_00_u03b1_1936_, lean_object* v_msg_1937_, lean_object* v___y_1938_, lean_object* v___y_1939_, lean_object* v___y_1940_, lean_object* v___y_1941_, lean_object* v___y_1942_, lean_object* v___y_1943_, lean_object* v___y_1944_, lean_object* v___y_1945_, lean_object* v___y_1946_){
_start:
{
lean_object* v_res_1947_; 
v_res_1947_ = lp_Qq_Lean_throwError___at___00Qq___aux__Qq__Commands______elabRules__Qq__tacticRun__tacq___x3d_x3e____1_spec__6(v_00_u03b1_1936_, v_msg_1937_, v___y_1938_, v___y_1939_, v___y_1940_, v___y_1941_, v___y_1942_, v___y_1943_, v___y_1944_, v___y_1945_);
lean_dec(v___y_1945_);
lean_dec_ref(v___y_1944_);
lean_dec(v___y_1943_);
lean_dec_ref(v___y_1942_);
lean_dec(v___y_1941_);
lean_dec_ref(v___y_1940_);
lean_dec(v___y_1939_);
lean_dec_ref(v___y_1938_);
return v_res_1947_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Qq_Qq_Macro(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_Qq_Qq_Commands(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_Macro(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Eval(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_Qq_Qq_Commands(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Eval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Qq_Qq_Macro(uint8_t builtin);
lean_object* initialize_Lean_Meta_Eval(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_Qq_Qq_Commands(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Qq_Qq_Macro(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Eval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_Commands(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_Qq_Qq_Commands(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_Qq_Qq_Commands(builtin);
}
#ifdef __cplusplus
}
#endif
