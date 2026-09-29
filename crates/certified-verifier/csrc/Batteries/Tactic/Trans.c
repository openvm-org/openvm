// Lean compiler output
// Module: Batteries.Tactic.Trans
// Imports: public import Init public meta import Init public meta import Lean.Elab.Tactic.ElabTerm
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
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_createNodes(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_Meta_DiscrTree_Key_lt(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_insertIdx_loop(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
uint8_t l_Lean_Name_isAnonymous(lean_object*);
lean_object* l_Lean_Environment_setExporting(lean_object*, uint8_t);
uint8_t l_Lean_Environment_contains(lean_object*, lean_object*, uint8_t);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_EnvironmentHeader_moduleNames(lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_isPrivateName(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
extern lean_object* l_Lean_unknownIdentifierMessageTag;
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint64_t l_Lean_Meta_DiscrTree_Key_hash(lean_object*);
uint8_t l_Lean_Meta_DiscrTree_instBEqKey_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_isUnaryNode___redArg(lean_object*);
lean_object* l_Array_eraseIdx___redArg(lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_instInhabited(lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_registerSimpleScopedEnvExtension___redArg(lean_object*);
lean_object* l_Lean_ScopedEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_getUnify___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* l_Lean_Elab_Tactic_SavedState_restore___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Environment_findConstVal_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_mkLevelParam(lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingImp(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppOptM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_Elab_Tactic_saveState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
lean_object* l_Lean_ScopedEnvExtension_addCore___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_cleanupAnnotations(lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainTag___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_elabTermWithHoles(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Expr_bvar___override(lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_lam___override(lean_object*, lean_object*, lean_object*, uint8_t);
uint8_t l_Lean_Expr_hasLooseBVars(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l_Lean_Environment_find_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_ConstantInfo_type(lean_object*);
lean_object* l_Lean_Meta_forallMetaTelescopeReducing(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_pop(lean_object*);
lean_object* l_Lean_Meta_DiscrTree_mkPath(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_registerBuiltinAttribute(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_registerTraceClass(lean_object*, uint8_t, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Trans_simple___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Trans_simple(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__0_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__0_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__0_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__1_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trans"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__1_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__1_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__2_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__0_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(186, 205, 46, 93, 234, 75, 44, 75)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__2_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__2_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__1_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(12, 212, 212, 166, 74, 116, 15, 20)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__2_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__2_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__3_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__3_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__3_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__4_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__3_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__4_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__4_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__5_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__5_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__5_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__6_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__4_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__5_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(90, 175, 18, 163, 178, 203, 59, 243)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__6_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__6_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__7_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__6_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__0_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(47, 92, 120, 195, 85, 23, 119, 138)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__7_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__7_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__8_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Trans"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__8_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__8_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__9_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__7_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__8_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(205, 124, 60, 112, 24, 102, 73, 134)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__9_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__9_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__10_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__9_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(56, 4, 13, 3, 145, 203, 254, 156)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__10_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__10_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__11_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__10_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__5_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(49, 0, 128, 73, 65, 87, 13, 202)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__11_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__11_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__12_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__11_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__0_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(184, 231, 187, 22, 3, 113, 41, 115)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__12_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__12_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__13_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__13_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__13_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__14_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__12_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__13_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(245, 116, 239, 18, 32, 166, 132, 72)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__14_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__14_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__15_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__15_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__15_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__16_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__14_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__15_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(8, 48, 171, 179, 58, 92, 143, 53)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__16_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__16_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__17_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__16_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__5_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(225, 194, 131, 212, 29, 206, 194, 107)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__17_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__17_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__18_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__17_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__0_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(168, 180, 203, 67, 48, 198, 178, 192)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__18_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__18_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__19_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__18_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__8_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(54, 79, 204, 243, 251, 251, 64, 146)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__19_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__19_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__20_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__20_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__21_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__21_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__21_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__22_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__22_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__23_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__23_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__23_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__24_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__24_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__25_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__25_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2____boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__2___closed__0;
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__2(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__4_spec__8(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__4_spec__8___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__10_spec__11___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__10___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__11___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal_loop___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__1___boxed(lean_object*, lean_object*);
static const lean_array_object lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0___closed__0 = (const lean_object*)&lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0___closed__0_value;
static const lean_ctor_object lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0___closed__0_value),((lean_object*)&lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0___closed__0_value)}};
static const lean_object* lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0___closed__1 = (const lean_object*)&lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Lean.Meta.DiscrTree.Basic"};
static const lean_object* lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0___closed__0 = (const lean_object*)&lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0___closed__0_value;
static const lean_string_object lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "Lean.Meta.DiscrTree.insertKeyValue"};
static const lean_object* lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0___closed__1 = (const lean_object*)&lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0___closed__1_value;
static const lean_string_object lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "invalid key sequence"};
static const lean_object* lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0___closed__2 = (const lean_object*)&lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0___closed__2_value;
static lean_once_cell_t lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0___closed__3;
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__2_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__2_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2____boxed(lean_object*);
static const lean_closure_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__0_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2____boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__0_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__0_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__value;
static const lean_closure_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__1_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__1_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__1_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__value;
static const lean_closure_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__2_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__2_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__2_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__2_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__3_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "transExt"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__3_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__3_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__value;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__4_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__5_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__4_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__4_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__0_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__4_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__4_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__3_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(249, 118, 28, 234, 27, 97, 109, 170)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__4_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__4_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__5_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__5_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__6_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__6_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__7_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__7_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__10(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__11(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__10_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_transExt;
static lean_once_cell_t lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__0;
static lean_once_cell_t lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__1;
static lean_once_cell_t lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__2;
static lean_once_cell_t lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__3;
LEAN_EXPORT lean_object* lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__0;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__1;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__2;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__3;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__4;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__5;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "A private declaration `"};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__6 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__6_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__7;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 79, .m_capacity = 79, .m_length = 78, .m_data = "` (from the current module) exists but would need to be public to access here."};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__8 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__8_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__9;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "A public declaration `"};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__10 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__10_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__11;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 68, .m_capacity = 68, .m_length = 67, .m_data = "` exists but is imported privately; consider adding `public import "};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__12 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__12_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__13;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`."};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__14 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__14_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__15;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "` (from `"};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__16 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__16_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__17;
static const lean_string_object lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "`) exists but would need to be public to access here."};
static const lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__18 = (const lean_object*)&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__18_value;
static lean_once_cell_t lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__19;
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__9___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Unknown constant `"};
static const lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg___closed__0 = (const lean_object*)&lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg___closed__0_value;
static lean_once_cell_t lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg___closed__1;
static const lean_string_object lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg___closed__2 = (const lean_object*)&lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg___closed__2_value;
static lean_once_cell_t lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg___closed__3;
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__0_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 24, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 1, 1, 0),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 1, 1, 1, 2, 1),LEAN_SCALAR_PTR_LITERAL(1, 1, 1, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__0_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__0_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__1_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__1_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__2_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__2_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__3_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__3_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__4_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__4_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__5_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__5_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__6_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__6_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__7_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__7_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__8_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 94, .m_capacity = 94, .m_length = 83, .m_data = "@[trans] attribute only applies to lemmas proving\n      x ∼ y → y ∼ z → x ∼ z, got "};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__8_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__8_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__9_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__9_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__10_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = " with target "};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__10_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__10_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__11_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__11_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__3_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1___closed__0_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Attribute `["};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1___closed__0_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1___closed__0_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1___closed__1_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1___closed__1_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1___closed__2_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "]` cannot be erased"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1___closed__2_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1___closed__2_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1___closed__3_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1___closed__3_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__0_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2____boxed, .m_arity = 8, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1))} };
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__0_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__0_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__1_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__1_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__2_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__2_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__3_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__3_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__4_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__4_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_;
static const lean_ctor_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__5_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__1_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(161, 148, 39, 37, 235, 128, 152, 21)}};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__5_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__5_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__value;
static const lean_closure_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__6_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2____boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__5_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__value)} };
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__6_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__6_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__value;
static const lean_string_object lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__7_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "transitive relation"};
static const lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__7_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_ = (const lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__7_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__value;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__8_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__8_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_;
static lean_once_cell_t lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__9_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__9_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_getExplicitFuncArg_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_getExplicitFuncArg_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_getExplicitRelArg_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_getExplicitRelArg_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_getExplicitRelArgCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_getExplicitRelArgCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_TransRelation_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_TransRelation_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_TransRelation_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_TransRelation_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_TransRelation_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_TransRelation_app_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_TransRelation_app_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_TransRelation_implies_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_TransRelation_implies_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_getRel(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_getRel___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "tacticTrans___"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__5_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__1_value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__0_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(2, 230, 252, 84, 184, 107, 13, 25)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__1_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__4_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__5_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__6_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__8_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__8_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__9_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__10_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__10_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__11 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__11_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__11_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__12 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__12_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__9_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__12_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__13 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__13_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__14 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__14_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__14_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__15 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__15_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__15_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__16 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__16_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__13_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__16_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__17 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__17_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__6_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__17_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__18 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__18_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__4_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__18_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__19 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__19_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__19_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__20 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__20_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_tacticTrans______ = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__20_value;
static lean_once_cell_t lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instantiateMVars___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instantiateMVars___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instantiateMVars___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_instantiateMVars___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__3___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__3(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__0___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__0___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__0___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__4(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg___closed__0;
static const lean_string_object lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg___closed__1 = (const lean_object*)&lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg___closed__1_value;
static const lean_array_object lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg___closed__2 = (const lean_object*)&lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7_spec__13_spec__15___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7_spec__13___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7_spec__14___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7_spec__14___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__2_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkConstWithLevelParams___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_mkConstWithLevelParams___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 13, .m_data = "obtained g₁: "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__0_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__1;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 13, .m_data = "obtained g₂: "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__2 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__2_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__3;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "z:  "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__4 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__4_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__5;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "x:"};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__6 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__6_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__7;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "rel: "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__8 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__8_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__9;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "obtained y: "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__10 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__10_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__11;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "lemma-type: "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__12 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__12_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__13;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "arity: "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__14 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__14_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__15;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "trying lemma "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__0___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__0___closed__0_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__0___boxed(lean_object**);
static const lean_closure_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__0_value;
static const lean_closure_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__1___boxed, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__2_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value)} };
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__1 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__1_value;
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__2 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__2_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__3;
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "failed: "};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__4 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__4_value;
static lean_once_cell_t lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__5;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__2(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__3___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___boxed(lean_object**);
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "HEq"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__1___closed__0_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 44, .m_capacity = 44, .m_length = 43, .m_data = "no applicable transitivity lemma found for "};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__1___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__1___closed__1_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__1___closed__2;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__1___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__9___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__9___lam__1___boxed(lean_object**);
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__9___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__9___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__9___closed__0_value;
static const lean_ctor_object lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__9___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__9___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__9___closed__1 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__9___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__9___boxed(lean_object**);
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "simple"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__2___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__2___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__2___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__8_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(81, 102, 87, 41, 87, 171, 69, 129)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__2___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(93, 120, 121, 50, 153, 199, 2, 178)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__2___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__2___closed__1_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__2___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__2___closed__2_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__3(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__3___boxed(lean_object**);
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "trying heterogeneous case"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__0_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__1;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "trying homogeneous case"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__2_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__3;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "z: "};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__4_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__5;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "x: "};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__6_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__7;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "goal decomposed"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__8_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__9;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 48, .m_capacity = 48, .m_length = 47, .m_data = "`trans` is not implemented for dependent arrows"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__10 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__10_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__11;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 56, .m_capacity = 56, .m_length = 55, .m_data = "transitivity lemmas only apply to binary relations and "};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__12 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__12_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__13;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "non-dependent arrows, not "};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__14 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__14_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__15;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7_spec__13(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7_spec__14(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7_spec__13_spec__15(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "tacticTransitivity___"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__5_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__1_value_aux_0),((lean_object*)&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__0_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(219, 16, 21, 14, 34, 244, 116, 98)}};
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(125, 15, 152, 221, 144, 46, 161, 155)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__1_value;
static const lean_string_object lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "transitivity"};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__18_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__4_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__5_value;
LEAN_EXPORT const lean_object* lp_batteries_Batteries_Tactic_tacticTransitivity______ = (const lean_object*)&lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__5_value;
static const lean_string_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______macroRules__Batteries__Tactic__tacticTransitivity________1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______macroRules__Batteries__Tactic__tacticTransitivity________1___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______macroRules__Batteries__Tactic__tacticTransitivity________1___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______macroRules__Batteries__Tactic__tacticTransitivity________1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______macroRules__Batteries__Tactic__tacticTransitivity________1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______macroRules__Batteries__Tactic__tacticTransitivity________1___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______macroRules__Batteries__Tactic__tacticTransitivity________1___closed__1_value;
static lean_once_cell_t lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______macroRules__Batteries__Tactic__tacticTransitivity________1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______macroRules__Batteries__Tactic__tacticTransitivity________1___closed__2;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______macroRules__Batteries__Tactic__tacticTransitivity________1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______macroRules__Batteries__Tactic__tacticTransitivity________1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Trans_simple___redArg(lean_object* v_a_1_, lean_object* v_b_2_, lean_object* v_c_3_, lean_object* v_inst_4_, lean_object* v_a_5_, lean_object* v_a_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lean_apply_5(v_inst_4_, v_a_1_, v_b_2_, v_c_3_, v_a_5_, v_a_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Trans_simple(lean_object* v_00_u03b1_8_, lean_object* v_a_9_, lean_object* v_b_10_, lean_object* v_c_11_, lean_object* v_r_12_, lean_object* v_inst_13_, lean_object* v_a_14_, lean_object* v_a_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lean_apply_5(v_inst_13_, v_a_9_, v_b_10_, v_c_11_, v_a_14_, v_a_15_);
return v___x_16_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__20_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; 
v___x_63_ = lean_unsigned_to_nat(2429574326u);
v___x_64_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__19_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_));
v___x_65_ = l_Lean_Name_num___override(v___x_64_, v___x_63_);
return v___x_65_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__22_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; 
v___x_67_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__21_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_));
v___x_68_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__20_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__20_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__20_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_);
v___x_69_ = l_Lean_Name_str___override(v___x_68_, v___x_67_);
return v___x_69_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__24_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; 
v___x_71_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__23_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_));
v___x_72_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__22_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__22_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__22_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_);
v___x_73_ = l_Lean_Name_str___override(v___x_72_, v___x_71_);
return v___x_73_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__25_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; 
v___x_74_ = lean_unsigned_to_nat(2u);
v___x_75_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__24_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__24_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__24_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_);
v___x_76_ = l_Lean_Name_num___override(v___x_75_, v___x_74_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_78_; uint8_t v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_78_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__2_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_));
v___x_79_ = 0;
v___x_80_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__25_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__25_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__25_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_);
v___x_81_ = l_Lean_registerTraceClass(v___x_78_, v___x_79_, v___x_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2____boxed(lean_object* v_a_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_();
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_(lean_object* v_x_84_, lean_object* v_a_85_){
_start:
{
lean_object* v___x_86_; lean_object* v___x_87_; 
v___x_86_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_86_, 0, v_a_85_);
lean_inc_ref_n(v___x_86_, 2);
v___x_87_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_87_, 0, v___x_86_);
lean_ctor_set(v___x_87_, 1, v___x_86_);
lean_ctor_set(v___x_87_, 2, v___x_86_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2____boxed(lean_object* v_x_88_, lean_object* v_a_89_){
_start:
{
lean_object* v_res_90_; 
v_res_90_ = lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_(v_x_88_, v_a_89_);
lean_dec_ref(v_x_88_);
return v_res_90_;
}
}
static lean_object* _init_lp_batteries_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__2___closed__0(void){
_start:
{
lean_object* v___x_91_; 
v___x_91_ = l_Lean_Meta_DiscrTree_instInhabited(lean_box(0));
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__2(lean_object* v_msg_92_){
_start:
{
lean_object* v___x_93_; lean_object* v___x_94_; 
v___x_93_ = lean_obj_once(&lp_batteries_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__2___closed__0, &lp_batteries_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__2___closed__0_once, _init_lp_batteries_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__2___closed__0);
v___x_94_ = lean_panic_fn_borrowed(v___x_93_, v_msg_92_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__4_spec__8(lean_object* v_xs_95_, lean_object* v_v_96_, lean_object* v_i_97_){
_start:
{
lean_object* v___x_98_; uint8_t v___x_99_; 
v___x_98_ = lean_array_get_size(v_xs_95_);
v___x_99_ = lean_nat_dec_lt(v_i_97_, v___x_98_);
if (v___x_99_ == 0)
{
lean_object* v___x_100_; 
lean_dec(v_i_97_);
v___x_100_ = lean_box(0);
return v___x_100_;
}
else
{
lean_object* v___x_101_; uint8_t v___x_102_; 
v___x_101_ = lean_array_fget_borrowed(v_xs_95_, v_i_97_);
v___x_102_ = l_Lean_Meta_DiscrTree_instBEqKey_beq(v___x_101_, v_v_96_);
if (v___x_102_ == 0)
{
lean_object* v___x_103_; lean_object* v___x_104_; 
v___x_103_ = lean_unsigned_to_nat(1u);
v___x_104_ = lean_nat_add(v_i_97_, v___x_103_);
lean_dec(v_i_97_);
v_i_97_ = v___x_104_;
goto _start;
}
else
{
lean_object* v___x_106_; 
v___x_106_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_106_, 0, v_i_97_);
return v___x_106_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__4_spec__8___boxed(lean_object* v_xs_107_, lean_object* v_v_108_, lean_object* v_i_109_){
_start:
{
lean_object* v_res_110_; 
v_res_110_ = lp_batteries_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__4_spec__8(v_xs_107_, v_v_108_, v_i_109_);
lean_dec(v_v_108_);
lean_dec_ref(v_xs_107_);
return v_res_110_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__4(lean_object* v_xs_111_, lean_object* v_v_112_){
_start:
{
lean_object* v___x_113_; lean_object* v___x_114_; 
v___x_113_ = lean_unsigned_to_nat(0u);
v___x_114_ = lp_batteries_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__4_spec__8(v_xs_111_, v_v_112_, v___x_113_);
return v___x_114_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__4___boxed(lean_object* v_xs_115_, lean_object* v_v_116_){
_start:
{
lean_object* v_res_117_; 
v_res_117_ = lp_batteries_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__4(v_xs_115_, v_v_116_);
lean_dec(v_v_116_);
lean_dec_ref(v_xs_115_);
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__10_spec__11___redArg(lean_object* v_x_118_, lean_object* v_x_119_, lean_object* v_x_120_, lean_object* v_x_121_){
_start:
{
lean_object* v_ks_122_; lean_object* v_vs_123_; lean_object* v___x_125_; uint8_t v_isShared_126_; uint8_t v_isSharedCheck_147_; 
v_ks_122_ = lean_ctor_get(v_x_118_, 0);
v_vs_123_ = lean_ctor_get(v_x_118_, 1);
v_isSharedCheck_147_ = !lean_is_exclusive(v_x_118_);
if (v_isSharedCheck_147_ == 0)
{
v___x_125_ = v_x_118_;
v_isShared_126_ = v_isSharedCheck_147_;
goto v_resetjp_124_;
}
else
{
lean_inc(v_vs_123_);
lean_inc(v_ks_122_);
lean_dec(v_x_118_);
v___x_125_ = lean_box(0);
v_isShared_126_ = v_isSharedCheck_147_;
goto v_resetjp_124_;
}
v_resetjp_124_:
{
lean_object* v___x_127_; uint8_t v___x_128_; 
v___x_127_ = lean_array_get_size(v_ks_122_);
v___x_128_ = lean_nat_dec_lt(v_x_119_, v___x_127_);
if (v___x_128_ == 0)
{
lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_132_; 
lean_dec(v_x_119_);
v___x_129_ = lean_array_push(v_ks_122_, v_x_120_);
v___x_130_ = lean_array_push(v_vs_123_, v_x_121_);
if (v_isShared_126_ == 0)
{
lean_ctor_set(v___x_125_, 1, v___x_130_);
lean_ctor_set(v___x_125_, 0, v___x_129_);
v___x_132_ = v___x_125_;
goto v_reusejp_131_;
}
else
{
lean_object* v_reuseFailAlloc_133_; 
v_reuseFailAlloc_133_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_133_, 0, v___x_129_);
lean_ctor_set(v_reuseFailAlloc_133_, 1, v___x_130_);
v___x_132_ = v_reuseFailAlloc_133_;
goto v_reusejp_131_;
}
v_reusejp_131_:
{
return v___x_132_;
}
}
else
{
lean_object* v_k_x27_134_; uint8_t v___x_135_; 
v_k_x27_134_ = lean_array_fget_borrowed(v_ks_122_, v_x_119_);
v___x_135_ = l_Lean_Meta_DiscrTree_instBEqKey_beq(v_x_120_, v_k_x27_134_);
if (v___x_135_ == 0)
{
lean_object* v___x_137_; 
if (v_isShared_126_ == 0)
{
v___x_137_ = v___x_125_;
goto v_reusejp_136_;
}
else
{
lean_object* v_reuseFailAlloc_141_; 
v_reuseFailAlloc_141_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_141_, 0, v_ks_122_);
lean_ctor_set(v_reuseFailAlloc_141_, 1, v_vs_123_);
v___x_137_ = v_reuseFailAlloc_141_;
goto v_reusejp_136_;
}
v_reusejp_136_:
{
lean_object* v___x_138_; lean_object* v___x_139_; 
v___x_138_ = lean_unsigned_to_nat(1u);
v___x_139_ = lean_nat_add(v_x_119_, v___x_138_);
lean_dec(v_x_119_);
v_x_118_ = v___x_137_;
v_x_119_ = v___x_139_;
goto _start;
}
}
else
{
lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_145_; 
v___x_142_ = lean_array_fset(v_ks_122_, v_x_119_, v_x_120_);
v___x_143_ = lean_array_fset(v_vs_123_, v_x_119_, v_x_121_);
lean_dec(v_x_119_);
if (v_isShared_126_ == 0)
{
lean_ctor_set(v___x_125_, 1, v___x_143_);
lean_ctor_set(v___x_125_, 0, v___x_142_);
v___x_145_ = v___x_125_;
goto v_reusejp_144_;
}
else
{
lean_object* v_reuseFailAlloc_146_; 
v_reuseFailAlloc_146_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_146_, 0, v___x_142_);
lean_ctor_set(v_reuseFailAlloc_146_, 1, v___x_143_);
v___x_145_ = v_reuseFailAlloc_146_;
goto v_reusejp_144_;
}
v_reusejp_144_:
{
return v___x_145_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__10___redArg(lean_object* v_n_148_, lean_object* v_k_149_, lean_object* v_v_150_){
_start:
{
lean_object* v___x_151_; lean_object* v___x_152_; 
v___x_151_ = lean_unsigned_to_nat(0u);
v___x_152_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__10_spec__11___redArg(v_n_148_, v___x_151_, v_k_149_, v_v_150_);
return v___x_152_;
}
}
static lean_object* _init_lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg___closed__0(void){
_start:
{
lean_object* v___x_153_; 
v___x_153_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg(lean_object* v_x_154_, size_t v_x_155_, size_t v_x_156_, lean_object* v_x_157_, lean_object* v_x_158_){
_start:
{
if (lean_obj_tag(v_x_154_) == 0)
{
lean_object* v_es_159_; size_t v___x_160_; size_t v___x_161_; lean_object* v_j_162_; lean_object* v___x_163_; uint8_t v___x_164_; 
v_es_159_ = lean_ctor_get(v_x_154_, 0);
v___x_160_ = ((size_t)31ULL);
v___x_161_ = lean_usize_land(v_x_155_, v___x_160_);
v_j_162_ = lean_usize_to_nat(v___x_161_);
v___x_163_ = lean_array_get_size(v_es_159_);
v___x_164_ = lean_nat_dec_lt(v_j_162_, v___x_163_);
if (v___x_164_ == 0)
{
lean_dec(v_j_162_);
lean_dec(v_x_158_);
lean_dec(v_x_157_);
return v_x_154_;
}
else
{
lean_object* v___x_166_; uint8_t v_isShared_167_; uint8_t v_isSharedCheck_203_; 
lean_inc_ref(v_es_159_);
v_isSharedCheck_203_ = !lean_is_exclusive(v_x_154_);
if (v_isSharedCheck_203_ == 0)
{
lean_object* v_unused_204_; 
v_unused_204_ = lean_ctor_get(v_x_154_, 0);
lean_dec(v_unused_204_);
v___x_166_ = v_x_154_;
v_isShared_167_ = v_isSharedCheck_203_;
goto v_resetjp_165_;
}
else
{
lean_dec(v_x_154_);
v___x_166_ = lean_box(0);
v_isShared_167_ = v_isSharedCheck_203_;
goto v_resetjp_165_;
}
v_resetjp_165_:
{
lean_object* v_v_168_; lean_object* v___x_169_; lean_object* v_xs_x27_170_; lean_object* v___y_172_; 
v_v_168_ = lean_array_fget(v_es_159_, v_j_162_);
v___x_169_ = lean_box(0);
v_xs_x27_170_ = lean_array_fset(v_es_159_, v_j_162_, v___x_169_);
switch(lean_obj_tag(v_v_168_))
{
case 0:
{
lean_object* v_key_177_; lean_object* v_val_178_; lean_object* v___x_180_; uint8_t v_isShared_181_; uint8_t v_isSharedCheck_188_; 
v_key_177_ = lean_ctor_get(v_v_168_, 0);
v_val_178_ = lean_ctor_get(v_v_168_, 1);
v_isSharedCheck_188_ = !lean_is_exclusive(v_v_168_);
if (v_isSharedCheck_188_ == 0)
{
v___x_180_ = v_v_168_;
v_isShared_181_ = v_isSharedCheck_188_;
goto v_resetjp_179_;
}
else
{
lean_inc(v_val_178_);
lean_inc(v_key_177_);
lean_dec(v_v_168_);
v___x_180_ = lean_box(0);
v_isShared_181_ = v_isSharedCheck_188_;
goto v_resetjp_179_;
}
v_resetjp_179_:
{
uint8_t v___x_182_; 
v___x_182_ = l_Lean_Meta_DiscrTree_instBEqKey_beq(v_x_157_, v_key_177_);
if (v___x_182_ == 0)
{
lean_object* v___x_183_; lean_object* v___x_184_; 
lean_del_object(v___x_180_);
v___x_183_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_177_, v_val_178_, v_x_157_, v_x_158_);
v___x_184_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_184_, 0, v___x_183_);
v___y_172_ = v___x_184_;
goto v___jp_171_;
}
else
{
lean_object* v___x_186_; 
lean_dec(v_val_178_);
lean_dec(v_key_177_);
if (v_isShared_181_ == 0)
{
lean_ctor_set(v___x_180_, 1, v_x_158_);
lean_ctor_set(v___x_180_, 0, v_x_157_);
v___x_186_ = v___x_180_;
goto v_reusejp_185_;
}
else
{
lean_object* v_reuseFailAlloc_187_; 
v_reuseFailAlloc_187_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_187_, 0, v_x_157_);
lean_ctor_set(v_reuseFailAlloc_187_, 1, v_x_158_);
v___x_186_ = v_reuseFailAlloc_187_;
goto v_reusejp_185_;
}
v_reusejp_185_:
{
v___y_172_ = v___x_186_;
goto v___jp_171_;
}
}
}
}
case 1:
{
lean_object* v_node_189_; lean_object* v___x_191_; uint8_t v_isShared_192_; uint8_t v_isSharedCheck_201_; 
v_node_189_ = lean_ctor_get(v_v_168_, 0);
v_isSharedCheck_201_ = !lean_is_exclusive(v_v_168_);
if (v_isSharedCheck_201_ == 0)
{
v___x_191_ = v_v_168_;
v_isShared_192_ = v_isSharedCheck_201_;
goto v_resetjp_190_;
}
else
{
lean_inc(v_node_189_);
lean_dec(v_v_168_);
v___x_191_ = lean_box(0);
v_isShared_192_ = v_isSharedCheck_201_;
goto v_resetjp_190_;
}
v_resetjp_190_:
{
size_t v___x_193_; size_t v___x_194_; size_t v___x_195_; size_t v___x_196_; lean_object* v___x_197_; lean_object* v___x_199_; 
v___x_193_ = ((size_t)5ULL);
v___x_194_ = lean_usize_shift_right(v_x_155_, v___x_193_);
v___x_195_ = ((size_t)1ULL);
v___x_196_ = lean_usize_add(v_x_156_, v___x_195_);
v___x_197_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg(v_node_189_, v___x_194_, v___x_196_, v_x_157_, v_x_158_);
if (v_isShared_192_ == 0)
{
lean_ctor_set(v___x_191_, 0, v___x_197_);
v___x_199_ = v___x_191_;
goto v_reusejp_198_;
}
else
{
lean_object* v_reuseFailAlloc_200_; 
v_reuseFailAlloc_200_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_200_, 0, v___x_197_);
v___x_199_ = v_reuseFailAlloc_200_;
goto v_reusejp_198_;
}
v_reusejp_198_:
{
v___y_172_ = v___x_199_;
goto v___jp_171_;
}
}
}
default: 
{
lean_object* v___x_202_; 
v___x_202_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_202_, 0, v_x_157_);
lean_ctor_set(v___x_202_, 1, v_x_158_);
v___y_172_ = v___x_202_;
goto v___jp_171_;
}
}
v___jp_171_:
{
lean_object* v___x_173_; lean_object* v___x_175_; 
v___x_173_ = lean_array_fset(v_xs_x27_170_, v_j_162_, v___y_172_);
lean_dec(v_j_162_);
if (v_isShared_167_ == 0)
{
lean_ctor_set(v___x_166_, 0, v___x_173_);
v___x_175_ = v___x_166_;
goto v_reusejp_174_;
}
else
{
lean_object* v_reuseFailAlloc_176_; 
v_reuseFailAlloc_176_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_176_, 0, v___x_173_);
v___x_175_ = v_reuseFailAlloc_176_;
goto v_reusejp_174_;
}
v_reusejp_174_:
{
return v___x_175_;
}
}
}
}
}
else
{
lean_object* v_ks_205_; lean_object* v_vs_206_; lean_object* v___x_208_; uint8_t v_isShared_209_; uint8_t v_isSharedCheck_226_; 
v_ks_205_ = lean_ctor_get(v_x_154_, 0);
v_vs_206_ = lean_ctor_get(v_x_154_, 1);
v_isSharedCheck_226_ = !lean_is_exclusive(v_x_154_);
if (v_isSharedCheck_226_ == 0)
{
v___x_208_ = v_x_154_;
v_isShared_209_ = v_isSharedCheck_226_;
goto v_resetjp_207_;
}
else
{
lean_inc(v_vs_206_);
lean_inc(v_ks_205_);
lean_dec(v_x_154_);
v___x_208_ = lean_box(0);
v_isShared_209_ = v_isSharedCheck_226_;
goto v_resetjp_207_;
}
v_resetjp_207_:
{
lean_object* v___x_211_; 
if (v_isShared_209_ == 0)
{
v___x_211_ = v___x_208_;
goto v_reusejp_210_;
}
else
{
lean_object* v_reuseFailAlloc_225_; 
v_reuseFailAlloc_225_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_225_, 0, v_ks_205_);
lean_ctor_set(v_reuseFailAlloc_225_, 1, v_vs_206_);
v___x_211_ = v_reuseFailAlloc_225_;
goto v_reusejp_210_;
}
v_reusejp_210_:
{
lean_object* v_newNode_212_; uint8_t v___y_214_; size_t v___x_220_; uint8_t v___x_221_; 
v_newNode_212_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__10___redArg(v___x_211_, v_x_157_, v_x_158_);
v___x_220_ = ((size_t)7ULL);
v___x_221_ = lean_usize_dec_le(v___x_220_, v_x_156_);
if (v___x_221_ == 0)
{
lean_object* v___x_222_; lean_object* v___x_223_; uint8_t v___x_224_; 
v___x_222_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_212_);
v___x_223_ = lean_unsigned_to_nat(4u);
v___x_224_ = lean_nat_dec_lt(v___x_222_, v___x_223_);
lean_dec(v___x_222_);
v___y_214_ = v___x_224_;
goto v___jp_213_;
}
else
{
v___y_214_ = v___x_221_;
goto v___jp_213_;
}
v___jp_213_:
{
if (v___y_214_ == 0)
{
lean_object* v_ks_215_; lean_object* v_vs_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; 
v_ks_215_ = lean_ctor_get(v_newNode_212_, 0);
lean_inc_ref(v_ks_215_);
v_vs_216_ = lean_ctor_get(v_newNode_212_, 1);
lean_inc_ref(v_vs_216_);
lean_dec_ref(v_newNode_212_);
v___x_217_ = lean_unsigned_to_nat(0u);
v___x_218_ = lean_obj_once(&lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg___closed__0, &lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg___closed__0_once, _init_lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg___closed__0);
v___x_219_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__11___redArg(v_x_156_, v_ks_215_, v_vs_216_, v___x_217_, v___x_218_);
lean_dec_ref(v_vs_216_);
lean_dec_ref(v_ks_215_);
return v___x_219_;
}
else
{
return v_newNode_212_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__11___redArg(size_t v_depth_227_, lean_object* v_keys_228_, lean_object* v_vals_229_, lean_object* v_i_230_, lean_object* v_entries_231_){
_start:
{
lean_object* v___x_232_; uint8_t v___x_233_; 
v___x_232_ = lean_array_get_size(v_keys_228_);
v___x_233_ = lean_nat_dec_lt(v_i_230_, v___x_232_);
if (v___x_233_ == 0)
{
lean_dec(v_i_230_);
return v_entries_231_;
}
else
{
lean_object* v_k_234_; lean_object* v_v_235_; uint64_t v___x_236_; size_t v_h_237_; size_t v___x_238_; lean_object* v___x_239_; size_t v___x_240_; size_t v___x_241_; size_t v___x_242_; size_t v_h_243_; lean_object* v___x_244_; lean_object* v___x_245_; 
v_k_234_ = lean_array_fget_borrowed(v_keys_228_, v_i_230_);
v_v_235_ = lean_array_fget_borrowed(v_vals_229_, v_i_230_);
v___x_236_ = l_Lean_Meta_DiscrTree_Key_hash(v_k_234_);
v_h_237_ = lean_uint64_to_usize(v___x_236_);
v___x_238_ = ((size_t)5ULL);
v___x_239_ = lean_unsigned_to_nat(1u);
v___x_240_ = ((size_t)1ULL);
v___x_241_ = lean_usize_sub(v_depth_227_, v___x_240_);
v___x_242_ = lean_usize_mul(v___x_238_, v___x_241_);
v_h_243_ = lean_usize_shift_right(v_h_237_, v___x_242_);
v___x_244_ = lean_nat_add(v_i_230_, v___x_239_);
lean_dec(v_i_230_);
lean_inc(v_v_235_);
lean_inc(v_k_234_);
v___x_245_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg(v_entries_231_, v_h_243_, v_depth_227_, v_k_234_, v_v_235_);
v_i_230_ = v___x_244_;
v_entries_231_ = v___x_245_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__11___redArg___boxed(lean_object* v_depth_247_, lean_object* v_keys_248_, lean_object* v_vals_249_, lean_object* v_i_250_, lean_object* v_entries_251_){
_start:
{
size_t v_depth_boxed_252_; lean_object* v_res_253_; 
v_depth_boxed_252_ = lean_unbox_usize(v_depth_247_);
lean_dec(v_depth_247_);
v_res_253_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__11___redArg(v_depth_boxed_252_, v_keys_248_, v_vals_249_, v_i_250_, v_entries_251_);
lean_dec_ref(v_vals_249_);
lean_dec_ref(v_keys_248_);
return v_res_253_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg___boxed(lean_object* v_x_254_, lean_object* v_x_255_, lean_object* v_x_256_, lean_object* v_x_257_, lean_object* v_x_258_){
_start:
{
size_t v_x_1706__boxed_259_; size_t v_x_1707__boxed_260_; lean_object* v_res_261_; 
v_x_1706__boxed_259_ = lean_unbox_usize(v_x_255_);
lean_dec(v_x_255_);
v_x_1707__boxed_260_ = lean_unbox_usize(v_x_256_);
lean_dec(v_x_256_);
v_res_261_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg(v_x_254_, v_x_1706__boxed_259_, v_x_1707__boxed_260_, v_x_257_, v_x_258_);
return v_res_261_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal_loop___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__3(lean_object* v_vs_262_, lean_object* v_v_263_, lean_object* v_i_264_){
_start:
{
lean_object* v___x_265_; uint8_t v___x_266_; 
v___x_265_ = lean_array_get_size(v_vs_262_);
v___x_266_ = lean_nat_dec_lt(v_i_264_, v___x_265_);
if (v___x_266_ == 0)
{
lean_object* v___x_267_; 
lean_dec(v_i_264_);
v___x_267_ = lean_array_push(v_vs_262_, v_v_263_);
return v___x_267_;
}
else
{
lean_object* v___x_268_; uint8_t v___x_269_; 
v___x_268_ = lean_array_fget_borrowed(v_vs_262_, v_i_264_);
v___x_269_ = lean_name_eq(v_v_263_, v___x_268_);
if (v___x_269_ == 0)
{
lean_object* v___x_270_; lean_object* v___x_271_; 
v___x_270_ = lean_unsigned_to_nat(1u);
v___x_271_ = lean_nat_add(v_i_264_, v___x_270_);
lean_dec(v_i_264_);
v_i_264_ = v___x_271_;
goto _start;
}
else
{
lean_object* v___x_273_; 
v___x_273_ = lean_array_fset(v_vs_262_, v_i_264_, v_v_263_);
lean_dec(v_i_264_);
return v___x_273_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__1(lean_object* v_vs_274_, lean_object* v_v_275_){
_start:
{
lean_object* v___x_276_; lean_object* v___x_277_; 
v___x_276_ = lean_unsigned_to_nat(0u);
v___x_277_ = lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal_loop___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__3(v_vs_274_, v_v_275_, v___x_276_);
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__0(lean_object* v_x_278_, lean_object* v_keys_279_, lean_object* v_v_280_, lean_object* v_k_281_, lean_object* v_x_282_){
_start:
{
lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v_c_285_; lean_object* v___x_286_; 
v___x_283_ = lean_unsigned_to_nat(1u);
v___x_284_ = lean_nat_add(v_x_278_, v___x_283_);
v_c_285_ = l___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_createNodes(lean_box(0), v_keys_279_, v_v_280_, v___x_284_);
lean_dec(v___x_284_);
v___x_286_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_286_, 0, v_k_281_);
lean_ctor_set(v___x_286_, 1, v_c_285_);
return v___x_286_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__0___boxed(lean_object* v_x_287_, lean_object* v_keys_288_, lean_object* v_v_289_, lean_object* v_k_290_, lean_object* v_x_291_){
_start:
{
lean_object* v_res_292_; 
v_res_292_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__0(v_x_287_, v_keys_288_, v_v_289_, v_k_290_, v_x_291_);
lean_dec_ref(v_keys_288_);
lean_dec(v_x_287_);
return v_res_292_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__1(lean_object* v_a_293_, lean_object* v_b_294_){
_start:
{
lean_object* v_fst_295_; lean_object* v_fst_296_; uint8_t v___x_297_; 
v_fst_295_ = lean_ctor_get(v_a_293_, 0);
v_fst_296_ = lean_ctor_get(v_b_294_, 0);
v___x_297_ = l_Lean_Meta_DiscrTree_Key_lt(v_fst_295_, v_fst_296_);
return v___x_297_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__1___boxed(lean_object* v_a_298_, lean_object* v_b_299_){
_start:
{
uint8_t v_res_300_; lean_object* v_r_301_; 
v_res_300_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__1(v_a_298_, v_b_299_);
lean_dec_ref(v_b_299_);
lean_dec_ref(v_a_298_);
v_r_301_ = lean_box(v_res_300_);
return v_r_301_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__5___redArg(lean_object* v_x_306_, lean_object* v_keys_307_, lean_object* v_v_308_, lean_object* v_k_309_, lean_object* v_as_310_, lean_object* v_k_311_, lean_object* v_x_312_, lean_object* v_x_313_){
_start:
{
lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v_mid_316_; lean_object* v_midVal_317_; uint8_t v___x_318_; 
v___x_314_ = lean_nat_add(v_x_312_, v_x_313_);
v___x_315_ = lean_unsigned_to_nat(1u);
v_mid_316_ = lean_nat_shiftr(v___x_314_, v___x_315_);
lean_dec(v___x_314_);
v_midVal_317_ = lean_array_fget(v_as_310_, v_mid_316_);
v___x_318_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__1(v_midVal_317_, v_k_311_);
if (v___x_318_ == 0)
{
uint8_t v___x_319_; 
lean_dec(v_x_313_);
v___x_319_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__1(v_k_311_, v_midVal_317_);
if (v___x_319_ == 0)
{
lean_object* v___x_320_; uint8_t v___x_321_; 
lean_dec(v_x_312_);
v___x_320_ = lean_array_get_size(v_as_310_);
v___x_321_ = lean_nat_dec_lt(v_mid_316_, v___x_320_);
if (v___x_321_ == 0)
{
lean_dec(v_midVal_317_);
lean_dec(v_mid_316_);
lean_dec(v_k_309_);
lean_dec(v_v_308_);
return v_as_310_;
}
else
{
lean_object* v_snd_322_; lean_object* v___x_324_; uint8_t v_isShared_325_; uint8_t v_isSharedCheck_334_; 
v_snd_322_ = lean_ctor_get(v_midVal_317_, 1);
v_isSharedCheck_334_ = !lean_is_exclusive(v_midVal_317_);
if (v_isSharedCheck_334_ == 0)
{
lean_object* v_unused_335_; 
v_unused_335_ = lean_ctor_get(v_midVal_317_, 0);
lean_dec(v_unused_335_);
v___x_324_ = v_midVal_317_;
v_isShared_325_ = v_isSharedCheck_334_;
goto v_resetjp_323_;
}
else
{
lean_inc(v_snd_322_);
lean_dec(v_midVal_317_);
v___x_324_ = lean_box(0);
v_isShared_325_ = v_isSharedCheck_334_;
goto v_resetjp_323_;
}
v_resetjp_323_:
{
lean_object* v___x_326_; lean_object* v_xs_x27_327_; lean_object* v___x_328_; lean_object* v_c_329_; lean_object* v___x_331_; 
v___x_326_ = lean_box(0);
v_xs_x27_327_ = lean_array_fset(v_as_310_, v_mid_316_, v___x_326_);
v___x_328_ = lean_nat_add(v_x_306_, v___x_315_);
v_c_329_ = lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0(v_keys_307_, v_v_308_, v___x_328_, v_snd_322_);
lean_dec(v___x_328_);
if (v_isShared_325_ == 0)
{
lean_ctor_set(v___x_324_, 1, v_c_329_);
lean_ctor_set(v___x_324_, 0, v_k_309_);
v___x_331_ = v___x_324_;
goto v_reusejp_330_;
}
else
{
lean_object* v_reuseFailAlloc_333_; 
v_reuseFailAlloc_333_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_333_, 0, v_k_309_);
lean_ctor_set(v_reuseFailAlloc_333_, 1, v_c_329_);
v___x_331_ = v_reuseFailAlloc_333_;
goto v_reusejp_330_;
}
v_reusejp_330_:
{
lean_object* v___x_332_; 
v___x_332_ = lean_array_fset(v_xs_x27_327_, v_mid_316_, v___x_331_);
lean_dec(v_mid_316_);
return v___x_332_;
}
}
}
}
else
{
lean_dec(v_midVal_317_);
v_x_313_ = v_mid_316_;
goto _start;
}
}
else
{
uint8_t v___x_337_; 
lean_dec(v_midVal_317_);
v___x_337_ = lean_nat_dec_eq(v_mid_316_, v_x_312_);
if (v___x_337_ == 0)
{
lean_dec(v_x_312_);
v_x_312_ = v_mid_316_;
goto _start;
}
else
{
lean_object* v___x_339_; lean_object* v_c_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v_j_343_; lean_object* v_as_344_; lean_object* v___x_345_; 
lean_dec(v_mid_316_);
lean_dec(v_x_313_);
v___x_339_ = lean_nat_add(v_x_306_, v___x_315_);
v_c_340_ = l___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_createNodes(lean_box(0), v_keys_307_, v_v_308_, v___x_339_);
lean_dec(v___x_339_);
v___x_341_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_341_, 0, v_k_309_);
lean_ctor_set(v___x_341_, 1, v_c_340_);
v___x_342_ = lean_nat_add(v_x_312_, v___x_315_);
lean_dec(v_x_312_);
v_j_343_ = lean_array_get_size(v_as_310_);
v_as_344_ = lean_array_push(v_as_310_, v___x_341_);
v___x_345_ = l___private_Init_Data_Array_Basic_0__Array_insertIdx_loop(lean_box(0), v___x_342_, v_as_344_, v_j_343_);
lean_dec(v___x_342_);
return v___x_345_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2(lean_object* v_x_346_, lean_object* v_keys_347_, lean_object* v_v_348_, lean_object* v_k_349_, lean_object* v_as_350_, lean_object* v_k_351_){
_start:
{
lean_object* v___x_352_; lean_object* v___x_353_; uint8_t v___x_354_; 
v___x_352_ = lean_array_get_size(v_as_350_);
v___x_353_ = lean_unsigned_to_nat(0u);
v___x_354_ = lean_nat_dec_eq(v___x_352_, v___x_353_);
if (v___x_354_ == 0)
{
lean_object* v___x_355_; uint8_t v___x_356_; 
v___x_355_ = lean_array_fget_borrowed(v_as_350_, v___x_353_);
v___x_356_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__1(v_k_351_, v___x_355_);
if (v___x_356_ == 0)
{
uint8_t v___x_357_; 
v___x_357_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__1(v___x_355_, v_k_351_);
if (v___x_357_ == 0)
{
uint8_t v___x_358_; 
v___x_358_ = lean_nat_dec_lt(v___x_353_, v___x_352_);
if (v___x_358_ == 0)
{
lean_dec(v_k_349_);
lean_dec(v_v_348_);
return v_as_350_;
}
else
{
lean_object* v___x_359_; lean_object* v_xs_x27_360_; lean_object* v___x_361_; lean_object* v___x_362_; 
lean_inc(v___x_355_);
v___x_359_ = lean_box(0);
v_xs_x27_360_ = lean_array_fset(v_as_350_, v___x_353_, v___x_359_);
v___x_361_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__2(v_x_346_, v_keys_347_, v_v_348_, v_k_349_, v___x_355_);
v___x_362_ = lean_array_fset(v_xs_x27_360_, v___x_353_, v___x_361_);
return v___x_362_;
}
}
else
{
lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; uint8_t v___x_366_; 
v___x_363_ = lean_unsigned_to_nat(1u);
v___x_364_ = lean_nat_sub(v___x_352_, v___x_363_);
v___x_365_ = lean_array_fget_borrowed(v_as_350_, v___x_364_);
v___x_366_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__1(v___x_365_, v_k_351_);
if (v___x_366_ == 0)
{
uint8_t v___x_367_; 
v___x_367_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__1(v_k_351_, v___x_365_);
if (v___x_367_ == 0)
{
uint8_t v___x_368_; 
v___x_368_ = lean_nat_dec_lt(v___x_364_, v___x_352_);
if (v___x_368_ == 0)
{
lean_dec(v___x_364_);
lean_dec(v_k_349_);
lean_dec(v_v_348_);
return v_as_350_;
}
else
{
lean_object* v___x_369_; lean_object* v_xs_x27_370_; lean_object* v___x_371_; lean_object* v___x_372_; 
lean_inc(v___x_365_);
v___x_369_ = lean_box(0);
v_xs_x27_370_ = lean_array_fset(v_as_350_, v___x_364_, v___x_369_);
v___x_371_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__2(v_x_346_, v_keys_347_, v_v_348_, v_k_349_, v___x_365_);
v___x_372_ = lean_array_fset(v_xs_x27_370_, v___x_364_, v___x_371_);
lean_dec(v___x_364_);
return v___x_372_;
}
}
else
{
lean_object* v___x_373_; 
v___x_373_ = lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__5___redArg(v_x_346_, v_keys_347_, v_v_348_, v_k_349_, v_as_350_, v_k_351_, v___x_353_, v___x_364_);
return v___x_373_;
}
}
else
{
lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; 
lean_dec(v___x_364_);
v___x_374_ = lean_box(0);
v___x_375_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__0(v_x_346_, v_keys_347_, v_v_348_, v_k_349_, v___x_374_);
v___x_376_ = lean_array_push(v_as_350_, v___x_375_);
return v___x_376_;
}
}
}
else
{
lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v_as_379_; lean_object* v___x_380_; 
v___x_377_ = lean_box(0);
v___x_378_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__0(v_x_346_, v_keys_347_, v_v_348_, v_k_349_, v___x_377_);
v_as_379_ = lean_array_push(v_as_350_, v___x_378_);
v___x_380_ = l___private_Init_Data_Array_Basic_0__Array_insertIdx_loop(lean_box(0), v___x_353_, v_as_379_, v___x_352_);
return v___x_380_;
}
}
else
{
lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; 
v___x_381_ = lean_box(0);
v___x_382_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__0(v_x_346_, v_keys_347_, v_v_348_, v_k_349_, v___x_381_);
v___x_383_ = lean_array_push(v_as_350_, v___x_382_);
return v___x_383_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0(lean_object* v_keys_384_, lean_object* v_v_385_, lean_object* v_x_386_, lean_object* v_x_387_){
_start:
{
lean_object* v_vs_388_; lean_object* v_children_389_; lean_object* v___x_391_; uint8_t v_isShared_392_; uint8_t v_isSharedCheck_406_; 
v_vs_388_ = lean_ctor_get(v_x_387_, 0);
v_children_389_ = lean_ctor_get(v_x_387_, 1);
v_isSharedCheck_406_ = !lean_is_exclusive(v_x_387_);
if (v_isSharedCheck_406_ == 0)
{
v___x_391_ = v_x_387_;
v_isShared_392_ = v_isSharedCheck_406_;
goto v_resetjp_390_;
}
else
{
lean_inc(v_children_389_);
lean_inc(v_vs_388_);
lean_dec(v_x_387_);
v___x_391_ = lean_box(0);
v_isShared_392_ = v_isSharedCheck_406_;
goto v_resetjp_390_;
}
v_resetjp_390_:
{
lean_object* v___x_393_; uint8_t v___x_394_; 
v___x_393_ = lean_array_get_size(v_keys_384_);
v___x_394_ = lean_nat_dec_lt(v_x_386_, v___x_393_);
if (v___x_394_ == 0)
{
lean_object* v___x_395_; lean_object* v___x_397_; 
v___x_395_ = lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertVal___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__1(v_vs_388_, v_v_385_);
if (v_isShared_392_ == 0)
{
lean_ctor_set(v___x_391_, 0, v___x_395_);
v___x_397_ = v___x_391_;
goto v_reusejp_396_;
}
else
{
lean_object* v_reuseFailAlloc_398_; 
v_reuseFailAlloc_398_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_398_, 0, v___x_395_);
lean_ctor_set(v_reuseFailAlloc_398_, 1, v_children_389_);
v___x_397_ = v_reuseFailAlloc_398_;
goto v_reusejp_396_;
}
v_reusejp_396_:
{
return v___x_397_;
}
}
else
{
lean_object* v_k_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v_c_402_; lean_object* v___x_404_; 
v_k_399_ = lean_array_fget_borrowed(v_keys_384_, v_x_386_);
v___x_400_ = ((lean_object*)(lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0___closed__1));
lean_inc_n(v_k_399_, 2);
v___x_401_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_401_, 0, v_k_399_);
lean_ctor_set(v___x_401_, 1, v___x_400_);
v_c_402_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2(v_x_386_, v_keys_384_, v_v_385_, v_k_399_, v_children_389_, v___x_401_);
lean_dec_ref_known(v___x_401_, 2);
if (v_isShared_392_ == 0)
{
lean_ctor_set(v___x_391_, 1, v_c_402_);
v___x_404_ = v___x_391_;
goto v_reusejp_403_;
}
else
{
lean_object* v_reuseFailAlloc_405_; 
v_reuseFailAlloc_405_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_405_, 0, v_vs_388_);
lean_ctor_set(v_reuseFailAlloc_405_, 1, v_c_402_);
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
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__2(lean_object* v_x_407_, lean_object* v_keys_408_, lean_object* v_v_409_, lean_object* v_k_410_, lean_object* v_x_411_){
_start:
{
lean_object* v_snd_412_; lean_object* v___x_414_; uint8_t v_isShared_415_; uint8_t v_isSharedCheck_422_; 
v_snd_412_ = lean_ctor_get(v_x_411_, 1);
v_isSharedCheck_422_ = !lean_is_exclusive(v_x_411_);
if (v_isSharedCheck_422_ == 0)
{
lean_object* v_unused_423_; 
v_unused_423_ = lean_ctor_get(v_x_411_, 0);
lean_dec(v_unused_423_);
v___x_414_ = v_x_411_;
v_isShared_415_ = v_isSharedCheck_422_;
goto v_resetjp_413_;
}
else
{
lean_inc(v_snd_412_);
lean_dec(v_x_411_);
v___x_414_ = lean_box(0);
v_isShared_415_ = v_isSharedCheck_422_;
goto v_resetjp_413_;
}
v_resetjp_413_:
{
lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v_c_418_; lean_object* v___x_420_; 
v___x_416_ = lean_unsigned_to_nat(1u);
v___x_417_ = lean_nat_add(v_x_407_, v___x_416_);
v_c_418_ = lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0(v_keys_408_, v_v_409_, v___x_417_, v_snd_412_);
lean_dec(v___x_417_);
if (v_isShared_415_ == 0)
{
lean_ctor_set(v___x_414_, 1, v_c_418_);
lean_ctor_set(v___x_414_, 0, v_k_410_);
v___x_420_ = v___x_414_;
goto v_reusejp_419_;
}
else
{
lean_object* v_reuseFailAlloc_421_; 
v_reuseFailAlloc_421_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_421_, 0, v_k_410_);
lean_ctor_set(v_reuseFailAlloc_421_, 1, v_c_418_);
v___x_420_ = v_reuseFailAlloc_421_;
goto v_reusejp_419_;
}
v_reusejp_419_:
{
return v___x_420_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__2___boxed(lean_object* v_x_424_, lean_object* v_keys_425_, lean_object* v_v_426_, lean_object* v_k_427_, lean_object* v_x_428_){
_start:
{
lean_object* v_res_429_; 
v_res_429_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___lam__2(v_x_424_, v_keys_425_, v_v_426_, v_k_427_, v_x_428_);
lean_dec_ref(v_keys_425_);
lean_dec(v_x_424_);
return v_res_429_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object* v_keys_430_, lean_object* v_v_431_, lean_object* v_x_432_, lean_object* v_x_433_){
_start:
{
lean_object* v_res_434_; 
v_res_434_ = lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0(v_keys_430_, v_v_431_, v_x_432_, v_x_433_);
lean_dec(v_x_432_);
lean_dec_ref(v_keys_430_);
return v_res_434_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__5___redArg___boxed(lean_object* v_x_435_, lean_object* v_keys_436_, lean_object* v_v_437_, lean_object* v_k_438_, lean_object* v_as_439_, lean_object* v_k_440_, lean_object* v_x_441_, lean_object* v_x_442_){
_start:
{
lean_object* v_res_443_; 
v_res_443_ = lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__5___redArg(v_x_435_, v_keys_436_, v_v_437_, v_k_438_, v_as_439_, v_k_440_, v_x_441_, v_x_442_);
lean_dec_ref(v_k_440_);
lean_dec_ref(v_keys_436_);
lean_dec(v_x_435_);
return v_res_443_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2___boxed(lean_object* v_x_444_, lean_object* v_keys_445_, lean_object* v_v_446_, lean_object* v_k_447_, lean_object* v_as_448_, lean_object* v_k_449_){
_start:
{
lean_object* v_res_450_; 
v_res_450_ = lp_batteries_Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2(v_x_444_, v_keys_445_, v_v_446_, v_k_447_, v_as_448_, v_k_449_);
lean_dec_ref(v_k_449_);
lean_dec_ref(v_keys_445_);
lean_dec(v_x_444_);
return v_res_450_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1___lam__0(lean_object* v_keys_451_, lean_object* v_v_452_, lean_object* v_x_453_){
_start:
{
if (lean_obj_tag(v_x_453_) == 0)
{
lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; 
v___x_454_ = lean_unsigned_to_nat(1u);
v___x_455_ = l___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_createNodes(lean_box(0), v_keys_451_, v_v_452_, v___x_454_);
v___x_456_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_456_, 0, v___x_455_);
return v___x_456_;
}
else
{
lean_object* v_val_457_; lean_object* v___x_459_; uint8_t v_isShared_460_; uint8_t v_isSharedCheck_466_; 
v_val_457_ = lean_ctor_get(v_x_453_, 0);
v_isSharedCheck_466_ = !lean_is_exclusive(v_x_453_);
if (v_isSharedCheck_466_ == 0)
{
v___x_459_ = v_x_453_;
v_isShared_460_ = v_isSharedCheck_466_;
goto v_resetjp_458_;
}
else
{
lean_inc(v_val_457_);
lean_dec(v_x_453_);
v___x_459_ = lean_box(0);
v_isShared_460_ = v_isSharedCheck_466_;
goto v_resetjp_458_;
}
v_resetjp_458_:
{
lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_464_; 
v___x_461_ = lean_unsigned_to_nat(1u);
v___x_462_ = lp_batteries___private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0(v_keys_451_, v_v_452_, v___x_461_, v_val_457_);
if (v_isShared_460_ == 0)
{
lean_ctor_set(v___x_459_, 0, v___x_462_);
v___x_464_ = v___x_459_;
goto v_reusejp_463_;
}
else
{
lean_object* v_reuseFailAlloc_465_; 
v_reuseFailAlloc_465_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_465_, 0, v___x_462_);
v___x_464_ = v_reuseFailAlloc_465_;
goto v_reusejp_463_;
}
v_reusejp_463_:
{
return v___x_464_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1___lam__0___boxed(lean_object* v_keys_467_, lean_object* v_v_468_, lean_object* v_x_469_){
_start:
{
lean_object* v_res_470_; 
v_res_470_ = lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1___lam__0(v_keys_467_, v_v_468_, v_x_469_);
lean_dec_ref(v_keys_467_);
return v_res_470_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1(lean_object* v_keys_471_, lean_object* v_v_472_, lean_object* v_x_473_, size_t v_x_474_, size_t v_x_475_, lean_object* v_x_476_){
_start:
{
if (lean_obj_tag(v_x_473_) == 0)
{
lean_object* v_es_477_; size_t v___x_478_; size_t v___x_479_; lean_object* v_j_480_; lean_object* v___x_481_; uint8_t v___x_482_; 
v_es_477_ = lean_ctor_get(v_x_473_, 0);
v___x_478_ = ((size_t)31ULL);
v___x_479_ = lean_usize_land(v_x_474_, v___x_478_);
v_j_480_ = lean_usize_to_nat(v___x_479_);
v___x_481_ = lean_array_get_size(v_es_477_);
v___x_482_ = lean_nat_dec_lt(v_j_480_, v___x_481_);
if (v___x_482_ == 0)
{
lean_dec(v_j_480_);
lean_dec(v_x_476_);
lean_dec(v_v_472_);
return v_x_473_;
}
else
{
lean_object* v___x_484_; uint8_t v_isShared_485_; uint8_t v_isSharedCheck_550_; 
lean_inc_ref(v_es_477_);
v_isSharedCheck_550_ = !lean_is_exclusive(v_x_473_);
if (v_isSharedCheck_550_ == 0)
{
lean_object* v_unused_551_; 
v_unused_551_ = lean_ctor_get(v_x_473_, 0);
lean_dec(v_unused_551_);
v___x_484_ = v_x_473_;
v_isShared_485_ = v_isSharedCheck_550_;
goto v_resetjp_483_;
}
else
{
lean_dec(v_x_473_);
v___x_484_ = lean_box(0);
v_isShared_485_ = v_isSharedCheck_550_;
goto v_resetjp_483_;
}
v_resetjp_483_:
{
lean_object* v_v_486_; lean_object* v___x_487_; lean_object* v_xs_x27_488_; lean_object* v___y_490_; 
v_v_486_ = lean_array_fget(v_es_477_, v_j_480_);
v___x_487_ = lean_box(0);
v_xs_x27_488_ = lean_array_fset(v_es_477_, v_j_480_, v___x_487_);
switch(lean_obj_tag(v_v_486_))
{
case 0:
{
lean_object* v_key_495_; lean_object* v_val_496_; uint8_t v___x_497_; 
v_key_495_ = lean_ctor_get(v_v_486_, 0);
v_val_496_ = lean_ctor_get(v_v_486_, 1);
v___x_497_ = l_Lean_Meta_DiscrTree_instBEqKey_beq(v_x_476_, v_key_495_);
if (v___x_497_ == 0)
{
lean_object* v___x_498_; lean_object* v___x_499_; 
v___x_498_ = lean_box(0);
v___x_499_ = lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1___lam__0(v_keys_471_, v_v_472_, v___x_498_);
if (lean_obj_tag(v___x_499_) == 0)
{
lean_dec(v_x_476_);
v___y_490_ = v_v_486_;
goto v___jp_489_;
}
else
{
lean_object* v_val_500_; lean_object* v___x_502_; uint8_t v_isShared_503_; uint8_t v_isSharedCheck_508_; 
lean_inc(v_val_496_);
lean_inc(v_key_495_);
lean_dec_ref_known(v_v_486_, 2);
v_val_500_ = lean_ctor_get(v___x_499_, 0);
v_isSharedCheck_508_ = !lean_is_exclusive(v___x_499_);
if (v_isSharedCheck_508_ == 0)
{
v___x_502_ = v___x_499_;
v_isShared_503_ = v_isSharedCheck_508_;
goto v_resetjp_501_;
}
else
{
lean_inc(v_val_500_);
lean_dec(v___x_499_);
v___x_502_ = lean_box(0);
v_isShared_503_ = v_isSharedCheck_508_;
goto v_resetjp_501_;
}
v_resetjp_501_:
{
lean_object* v___x_504_; lean_object* v___x_506_; 
v___x_504_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_495_, v_val_496_, v_x_476_, v_val_500_);
if (v_isShared_503_ == 0)
{
lean_ctor_set(v___x_502_, 0, v___x_504_);
v___x_506_ = v___x_502_;
goto v_reusejp_505_;
}
else
{
lean_object* v_reuseFailAlloc_507_; 
v_reuseFailAlloc_507_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_507_, 0, v___x_504_);
v___x_506_ = v_reuseFailAlloc_507_;
goto v_reusejp_505_;
}
v_reusejp_505_:
{
v___y_490_ = v___x_506_;
goto v___jp_489_;
}
}
}
}
else
{
lean_object* v___x_510_; uint8_t v_isShared_511_; uint8_t v_isSharedCheck_519_; 
lean_inc(v_val_496_);
v_isSharedCheck_519_ = !lean_is_exclusive(v_v_486_);
if (v_isSharedCheck_519_ == 0)
{
lean_object* v_unused_520_; lean_object* v_unused_521_; 
v_unused_520_ = lean_ctor_get(v_v_486_, 1);
lean_dec(v_unused_520_);
v_unused_521_ = lean_ctor_get(v_v_486_, 0);
lean_dec(v_unused_521_);
v___x_510_ = v_v_486_;
v_isShared_511_ = v_isSharedCheck_519_;
goto v_resetjp_509_;
}
else
{
lean_dec(v_v_486_);
v___x_510_ = lean_box(0);
v_isShared_511_ = v_isSharedCheck_519_;
goto v_resetjp_509_;
}
v_resetjp_509_:
{
lean_object* v___x_512_; lean_object* v___x_513_; 
v___x_512_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_512_, 0, v_val_496_);
v___x_513_ = lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1___lam__0(v_keys_471_, v_v_472_, v___x_512_);
if (lean_obj_tag(v___x_513_) == 0)
{
lean_object* v___x_514_; 
lean_del_object(v___x_510_);
lean_dec(v_x_476_);
v___x_514_ = lean_box(2);
v___y_490_ = v___x_514_;
goto v___jp_489_;
}
else
{
lean_object* v_val_515_; lean_object* v___x_517_; 
v_val_515_ = lean_ctor_get(v___x_513_, 0);
lean_inc(v_val_515_);
lean_dec_ref_known(v___x_513_, 1);
if (v_isShared_511_ == 0)
{
lean_ctor_set(v___x_510_, 1, v_val_515_);
lean_ctor_set(v___x_510_, 0, v_x_476_);
v___x_517_ = v___x_510_;
goto v_reusejp_516_;
}
else
{
lean_object* v_reuseFailAlloc_518_; 
v_reuseFailAlloc_518_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_518_, 0, v_x_476_);
lean_ctor_set(v_reuseFailAlloc_518_, 1, v_val_515_);
v___x_517_ = v_reuseFailAlloc_518_;
goto v_reusejp_516_;
}
v_reusejp_516_:
{
v___y_490_ = v___x_517_;
goto v___jp_489_;
}
}
}
}
}
case 1:
{
lean_object* v_node_522_; lean_object* v___x_524_; uint8_t v_isShared_525_; uint8_t v_isSharedCheck_545_; 
v_node_522_ = lean_ctor_get(v_v_486_, 0);
v_isSharedCheck_545_ = !lean_is_exclusive(v_v_486_);
if (v_isSharedCheck_545_ == 0)
{
v___x_524_ = v_v_486_;
v_isShared_525_ = v_isSharedCheck_545_;
goto v_resetjp_523_;
}
else
{
lean_inc(v_node_522_);
lean_dec(v_v_486_);
v___x_524_ = lean_box(0);
v_isShared_525_ = v_isSharedCheck_545_;
goto v_resetjp_523_;
}
v_resetjp_523_:
{
size_t v___x_526_; size_t v___x_527_; size_t v___x_528_; size_t v___x_529_; lean_object* v_newNode_530_; lean_object* v___x_531_; 
v___x_526_ = ((size_t)5ULL);
v___x_527_ = lean_usize_shift_right(v_x_474_, v___x_526_);
v___x_528_ = ((size_t)1ULL);
v___x_529_ = lean_usize_add(v_x_475_, v___x_528_);
v_newNode_530_ = lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1(v_keys_471_, v_v_472_, v_node_522_, v___x_527_, v___x_529_, v_x_476_);
lean_inc_ref(v_newNode_530_);
v___x_531_ = l_Lean_PersistentHashMap_isUnaryNode___redArg(v_newNode_530_);
if (lean_obj_tag(v___x_531_) == 0)
{
lean_object* v___x_533_; 
if (v_isShared_525_ == 0)
{
lean_ctor_set(v___x_524_, 0, v_newNode_530_);
v___x_533_ = v___x_524_;
goto v_reusejp_532_;
}
else
{
lean_object* v_reuseFailAlloc_534_; 
v_reuseFailAlloc_534_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_534_, 0, v_newNode_530_);
v___x_533_ = v_reuseFailAlloc_534_;
goto v_reusejp_532_;
}
v_reusejp_532_:
{
v___y_490_ = v___x_533_;
goto v___jp_489_;
}
}
else
{
lean_object* v_val_535_; lean_object* v_fst_536_; lean_object* v_snd_537_; lean_object* v___x_539_; uint8_t v_isShared_540_; uint8_t v_isSharedCheck_544_; 
lean_dec_ref(v_newNode_530_);
lean_del_object(v___x_524_);
v_val_535_ = lean_ctor_get(v___x_531_, 0);
lean_inc(v_val_535_);
lean_dec_ref_known(v___x_531_, 1);
v_fst_536_ = lean_ctor_get(v_val_535_, 0);
v_snd_537_ = lean_ctor_get(v_val_535_, 1);
v_isSharedCheck_544_ = !lean_is_exclusive(v_val_535_);
if (v_isSharedCheck_544_ == 0)
{
v___x_539_ = v_val_535_;
v_isShared_540_ = v_isSharedCheck_544_;
goto v_resetjp_538_;
}
else
{
lean_inc(v_snd_537_);
lean_inc(v_fst_536_);
lean_dec(v_val_535_);
v___x_539_ = lean_box(0);
v_isShared_540_ = v_isSharedCheck_544_;
goto v_resetjp_538_;
}
v_resetjp_538_:
{
lean_object* v___x_542_; 
if (v_isShared_540_ == 0)
{
v___x_542_ = v___x_539_;
goto v_reusejp_541_;
}
else
{
lean_object* v_reuseFailAlloc_543_; 
v_reuseFailAlloc_543_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_543_, 0, v_fst_536_);
lean_ctor_set(v_reuseFailAlloc_543_, 1, v_snd_537_);
v___x_542_ = v_reuseFailAlloc_543_;
goto v_reusejp_541_;
}
v_reusejp_541_:
{
v___y_490_ = v___x_542_;
goto v___jp_489_;
}
}
}
}
}
default: 
{
lean_object* v___x_546_; lean_object* v___x_547_; 
v___x_546_ = lean_box(0);
v___x_547_ = lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1___lam__0(v_keys_471_, v_v_472_, v___x_546_);
if (lean_obj_tag(v___x_547_) == 0)
{
lean_dec(v_x_476_);
v___y_490_ = v_v_486_;
goto v___jp_489_;
}
else
{
lean_object* v_val_548_; lean_object* v___x_549_; 
v_val_548_ = lean_ctor_get(v___x_547_, 0);
lean_inc(v_val_548_);
lean_dec_ref_known(v___x_547_, 1);
v___x_549_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_549_, 0, v_x_476_);
lean_ctor_set(v___x_549_, 1, v_val_548_);
v___y_490_ = v___x_549_;
goto v___jp_489_;
}
}
}
v___jp_489_:
{
lean_object* v___x_491_; lean_object* v___x_493_; 
v___x_491_ = lean_array_fset(v_xs_x27_488_, v_j_480_, v___y_490_);
lean_dec(v_j_480_);
if (v_isShared_485_ == 0)
{
lean_ctor_set(v___x_484_, 0, v___x_491_);
v___x_493_ = v___x_484_;
goto v_reusejp_492_;
}
else
{
lean_object* v_reuseFailAlloc_494_; 
v_reuseFailAlloc_494_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_494_, 0, v___x_491_);
v___x_493_ = v_reuseFailAlloc_494_;
goto v_reusejp_492_;
}
v_reusejp_492_:
{
return v___x_493_;
}
}
}
}
}
else
{
lean_object* v_ks_552_; lean_object* v_vs_553_; lean_object* v___x_555_; uint8_t v_isShared_556_; uint8_t v_isSharedCheck_586_; 
v_ks_552_ = lean_ctor_get(v_x_473_, 0);
v_vs_553_ = lean_ctor_get(v_x_473_, 1);
v_isSharedCheck_586_ = !lean_is_exclusive(v_x_473_);
if (v_isSharedCheck_586_ == 0)
{
v___x_555_ = v_x_473_;
v_isShared_556_ = v_isSharedCheck_586_;
goto v_resetjp_554_;
}
else
{
lean_inc(v_vs_553_);
lean_inc(v_ks_552_);
lean_dec(v_x_473_);
v___x_555_ = lean_box(0);
v_isShared_556_ = v_isSharedCheck_586_;
goto v_resetjp_554_;
}
v_resetjp_554_:
{
lean_object* v___x_557_; 
v___x_557_ = lp_batteries_Array_finIdxOf_x3f___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__4(v_ks_552_, v_x_476_);
if (lean_obj_tag(v___x_557_) == 0)
{
lean_object* v___x_559_; 
if (v_isShared_556_ == 0)
{
v___x_559_ = v___x_555_;
goto v_reusejp_558_;
}
else
{
lean_object* v_reuseFailAlloc_564_; 
v_reuseFailAlloc_564_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_564_, 0, v_ks_552_);
lean_ctor_set(v_reuseFailAlloc_564_, 1, v_vs_553_);
v___x_559_ = v_reuseFailAlloc_564_;
goto v_reusejp_558_;
}
v_reusejp_558_:
{
lean_object* v___x_560_; lean_object* v___x_561_; 
v___x_560_ = lean_box(0);
v___x_561_ = lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1___lam__0(v_keys_471_, v_v_472_, v___x_560_);
if (lean_obj_tag(v___x_561_) == 0)
{
lean_dec(v_x_476_);
return v___x_559_;
}
else
{
lean_object* v_val_562_; lean_object* v___x_563_; 
v_val_562_ = lean_ctor_get(v___x_561_, 0);
lean_inc(v_val_562_);
lean_dec_ref_known(v___x_561_, 1);
v___x_563_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg(v___x_559_, v_x_474_, v_x_475_, v_x_476_, v_val_562_);
return v___x_563_;
}
}
}
else
{
lean_object* v_val_565_; lean_object* v___x_567_; uint8_t v_isShared_568_; uint8_t v_isSharedCheck_585_; 
v_val_565_ = lean_ctor_get(v___x_557_, 0);
v_isSharedCheck_585_ = !lean_is_exclusive(v___x_557_);
if (v_isSharedCheck_585_ == 0)
{
v___x_567_ = v___x_557_;
v_isShared_568_ = v_isSharedCheck_585_;
goto v_resetjp_566_;
}
else
{
lean_inc(v_val_565_);
lean_dec(v___x_557_);
v___x_567_ = lean_box(0);
v_isShared_568_ = v_isSharedCheck_585_;
goto v_resetjp_566_;
}
v_resetjp_566_:
{
lean_object* v_v_x27_569_; lean_object* v_keys_570_; lean_object* v_vals_571_; lean_object* v___x_573_; 
v_v_x27_569_ = lean_array_fget(v_vs_553_, v_val_565_);
lean_inc(v_val_565_);
v_keys_570_ = l_Array_eraseIdx___redArg(v_ks_552_, v_val_565_);
v_vals_571_ = l_Array_eraseIdx___redArg(v_vs_553_, v_val_565_);
if (v_isShared_568_ == 0)
{
lean_ctor_set(v___x_567_, 0, v_v_x27_569_);
v___x_573_ = v___x_567_;
goto v_reusejp_572_;
}
else
{
lean_object* v_reuseFailAlloc_584_; 
v_reuseFailAlloc_584_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_584_, 0, v_v_x27_569_);
v___x_573_ = v_reuseFailAlloc_584_;
goto v_reusejp_572_;
}
v_reusejp_572_:
{
lean_object* v___x_574_; 
v___x_574_ = lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1___lam__0(v_keys_471_, v_v_472_, v___x_573_);
if (lean_obj_tag(v___x_574_) == 0)
{
lean_object* v___x_576_; 
lean_dec(v_x_476_);
if (v_isShared_556_ == 0)
{
lean_ctor_set(v___x_555_, 1, v_vals_571_);
lean_ctor_set(v___x_555_, 0, v_keys_570_);
v___x_576_ = v___x_555_;
goto v_reusejp_575_;
}
else
{
lean_object* v_reuseFailAlloc_577_; 
v_reuseFailAlloc_577_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_577_, 0, v_keys_570_);
lean_ctor_set(v_reuseFailAlloc_577_, 1, v_vals_571_);
v___x_576_ = v_reuseFailAlloc_577_;
goto v_reusejp_575_;
}
v_reusejp_575_:
{
return v___x_576_;
}
}
else
{
lean_object* v_val_578_; lean_object* v_keys_579_; lean_object* v_vals_580_; lean_object* v___x_582_; 
v_val_578_ = lean_ctor_get(v___x_574_, 0);
lean_inc(v_val_578_);
lean_dec_ref_known(v___x_574_, 1);
v_keys_579_ = lean_array_push(v_keys_570_, v_x_476_);
v_vals_580_ = lean_array_push(v_vals_571_, v_val_578_);
if (v_isShared_556_ == 0)
{
lean_ctor_set(v___x_555_, 1, v_vals_580_);
lean_ctor_set(v___x_555_, 0, v_keys_579_);
v___x_582_ = v___x_555_;
goto v_reusejp_581_;
}
else
{
lean_object* v_reuseFailAlloc_583_; 
v_reuseFailAlloc_583_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_583_, 0, v_keys_579_);
lean_ctor_set(v_reuseFailAlloc_583_, 1, v_vals_580_);
v___x_582_ = v_reuseFailAlloc_583_;
goto v_reusejp_581_;
}
v_reusejp_581_:
{
return v___x_582_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1___boxed(lean_object* v_keys_587_, lean_object* v_v_588_, lean_object* v_x_589_, lean_object* v_x_590_, lean_object* v_x_591_, lean_object* v_x_592_){
_start:
{
size_t v_x_2131__boxed_593_; size_t v_x_2132__boxed_594_; lean_object* v_res_595_; 
v_x_2131__boxed_593_ = lean_unbox_usize(v_x_590_);
lean_dec(v_x_590_);
v_x_2132__boxed_594_ = lean_unbox_usize(v_x_591_);
lean_dec(v_x_591_);
v_res_595_ = lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1(v_keys_587_, v_v_588_, v_x_589_, v_x_2131__boxed_593_, v_x_2132__boxed_594_, v_x_592_);
lean_dec_ref(v_keys_587_);
return v_res_595_;
}
}
static lean_object* _init_lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0___closed__3(void){
_start:
{
lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; 
v___x_599_ = ((lean_object*)(lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0___closed__2));
v___x_600_ = lean_unsigned_to_nat(23u);
v___x_601_ = lean_unsigned_to_nat(166u);
v___x_602_ = ((lean_object*)(lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0___closed__1));
v___x_603_ = ((lean_object*)(lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0___closed__0));
v___x_604_ = l_mkPanicMessageWithDecl(v___x_603_, v___x_602_, v___x_601_, v___x_600_, v___x_599_);
return v___x_604_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0(lean_object* v_d_605_, lean_object* v_keys_606_, lean_object* v_v_607_){
_start:
{
lean_object* v___x_608_; lean_object* v___x_609_; uint8_t v___x_610_; 
v___x_608_ = lean_array_get_size(v_keys_606_);
v___x_609_ = lean_unsigned_to_nat(0u);
v___x_610_ = lean_nat_dec_eq(v___x_608_, v___x_609_);
if (v___x_610_ == 0)
{
lean_object* v___x_611_; lean_object* v_k_612_; uint64_t v___x_613_; size_t v_h_614_; size_t v___x_615_; lean_object* v___x_616_; 
v___x_611_ = lean_box(0);
v_k_612_ = lean_array_get_borrowed(v___x_611_, v_keys_606_, v___x_609_);
v___x_613_ = l_Lean_Meta_DiscrTree_Key_hash(v_k_612_);
v_h_614_ = lean_uint64_to_usize(v___x_613_);
v___x_615_ = ((size_t)1ULL);
lean_inc(v_k_612_);
v___x_616_ = lp_batteries_Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1(v_keys_606_, v_v_607_, v_d_605_, v_h_614_, v___x_615_, v_k_612_);
return v___x_616_;
}
else
{
lean_object* v___x_617_; lean_object* v___x_618_; 
lean_dec(v_v_607_);
lean_dec_ref(v_d_605_);
v___x_617_ = lean_obj_once(&lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0___closed__3, &lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0___closed__3_once, _init_lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0___closed__3);
v___x_618_ = lp_batteries_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__2(v___x_617_);
return v___x_618_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0___boxed(lean_object* v_d_619_, lean_object* v_keys_620_, lean_object* v_v_621_){
_start:
{
lean_object* v_res_622_; 
v_res_622_ = lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0(v_d_619_, v_keys_620_, v_v_621_);
lean_dec_ref(v_keys_620_);
return v_res_622_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_(lean_object* v_dt_623_, lean_object* v_x_624_){
_start:
{
lean_object* v_fst_625_; lean_object* v_snd_626_; lean_object* v___x_627_; 
v_fst_625_ = lean_ctor_get(v_x_624_, 0);
lean_inc(v_fst_625_);
v_snd_626_ = lean_ctor_get(v_x_624_, 1);
lean_inc(v_snd_626_);
lean_dec_ref(v_x_624_);
v___x_627_ = lp_batteries_Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0(v_dt_623_, v_snd_626_, v_fst_625_);
lean_dec(v_snd_626_);
return v___x_627_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__2_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_(lean_object* v___y_628_){
_start:
{
lean_inc_ref(v___y_628_);
return v___y_628_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__2_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2____boxed(lean_object* v___y_629_){
_start:
{
lean_object* v_res_630_; 
v_res_630_ = lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__2_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_(v___y_629_);
lean_dec_ref(v___y_629_);
return v_res_630_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__5_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_639_; 
v___x_639_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_639_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__6_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_640_; lean_object* v___x_641_; 
v___x_640_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__5_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__5_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__5_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_);
v___x_641_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_641_, 0, v___x_640_);
return v___x_641_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__7_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___f_642_; lean_object* v___f_643_; lean_object* v___x_644_; lean_object* v___f_645_; lean_object* v___x_646_; lean_object* v___x_647_; 
v___f_642_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__0_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_));
v___f_643_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__2_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_));
v___x_644_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__6_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__6_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__6_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_);
v___f_645_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__1_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_));
v___x_646_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__4_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_));
v___x_647_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_647_, 0, v___x_646_);
lean_ctor_set(v___x_647_, 1, v___f_645_);
lean_ctor_set(v___x_647_, 2, v___x_644_);
lean_ctor_set(v___x_647_, 3, v___f_643_);
lean_ctor_set(v___x_647_, 4, v___f_642_);
return v___x_647_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_649_; lean_object* v___x_650_; 
v___x_649_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__7_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__7_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__7_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_);
v___x_650_ = l_Lean_registerSimpleScopedEnvExtension___redArg(v___x_649_);
return v___x_650_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2____boxed(lean_object* v_a_651_){
_start:
{
lean_object* v_res_652_; 
v_res_652_ = lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_();
return v_res_652_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5(lean_object* v_00_u03b2_653_, lean_object* v_x_654_, size_t v_x_655_, size_t v_x_656_, lean_object* v_x_657_, lean_object* v_x_658_){
_start:
{
lean_object* v___x_659_; 
v___x_659_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5___redArg(v_x_654_, v_x_655_, v_x_656_, v_x_657_, v_x_658_);
return v___x_659_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5___boxed(lean_object* v_00_u03b2_660_, lean_object* v_x_661_, lean_object* v_x_662_, lean_object* v_x_663_, lean_object* v_x_664_, lean_object* v_x_665_){
_start:
{
size_t v_x_2461__boxed_666_; size_t v_x_2462__boxed_667_; lean_object* v_res_668_; 
v_x_2461__boxed_666_ = lean_unbox_usize(v_x_662_);
lean_dec(v_x_662_);
v_x_2462__boxed_667_ = lean_unbox_usize(v_x_663_);
lean_dec(v_x_663_);
v_res_668_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5(v_00_u03b2_660_, v_x_661_, v_x_2461__boxed_666_, v_x_2462__boxed_667_, v_x_664_, v_x_665_);
return v_res_668_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__5(lean_object* v_x_669_, lean_object* v_keys_670_, lean_object* v_v_671_, lean_object* v_k_672_, lean_object* v_as_673_, lean_object* v_k_674_, lean_object* v_x_675_, lean_object* v_x_676_, lean_object* v_x_677_, lean_object* v_x_678_){
_start:
{
lean_object* v___x_679_; 
v___x_679_ = lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__5___redArg(v_x_669_, v_keys_670_, v_v_671_, v_k_672_, v_as_673_, v_k_674_, v_x_675_, v_x_676_);
return v___x_679_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__5___boxed(lean_object* v_x_680_, lean_object* v_keys_681_, lean_object* v_v_682_, lean_object* v_k_683_, lean_object* v_as_684_, lean_object* v_k_685_, lean_object* v_x_686_, lean_object* v_x_687_, lean_object* v_x_688_, lean_object* v_x_689_){
_start:
{
lean_object* v_res_690_; 
v_res_690_ = lp_batteries___private_Init_Data_Array_BinSearch_0__Array_binInsertAux___at___00Array_binInsertM___at___00__private_Lean_Meta_DiscrTree_Basic_0__Lean_Meta_DiscrTree_insertAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__5(v_x_680_, v_keys_681_, v_v_682_, v_k_683_, v_as_684_, v_k_685_, v_x_686_, v_x_687_, v_x_688_, v_x_689_);
lean_dec_ref(v_k_685_);
lean_dec_ref(v_keys_681_);
lean_dec(v_x_680_);
return v_res_690_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__10(lean_object* v_00_u03b2_691_, lean_object* v_n_692_, lean_object* v_k_693_, lean_object* v_v_694_){
_start:
{
lean_object* v___x_695_; 
v___x_695_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__10___redArg(v_n_692_, v_k_693_, v_v_694_);
return v___x_695_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__11(lean_object* v_00_u03b2_696_, size_t v_depth_697_, lean_object* v_keys_698_, lean_object* v_vals_699_, lean_object* v_heq_700_, lean_object* v_i_701_, lean_object* v_entries_702_){
_start:
{
lean_object* v___x_703_; 
v___x_703_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__11___redArg(v_depth_697_, v_keys_698_, v_vals_699_, v_i_701_, v_entries_702_);
return v___x_703_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__11___boxed(lean_object* v_00_u03b2_704_, lean_object* v_depth_705_, lean_object* v_keys_706_, lean_object* v_vals_707_, lean_object* v_heq_708_, lean_object* v_i_709_, lean_object* v_entries_710_){
_start:
{
size_t v_depth_boxed_711_; lean_object* v_res_712_; 
v_depth_boxed_711_ = lean_unbox_usize(v_depth_705_);
lean_dec(v_depth_705_);
v_res_712_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__11(v_00_u03b2_704_, v_depth_boxed_711_, v_keys_706_, v_vals_707_, v_heq_708_, v_i_709_, v_entries_710_);
lean_dec_ref(v_vals_707_);
lean_dec_ref(v_keys_706_);
return v_res_712_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__10_spec__11(lean_object* v_00_u03b2_713_, lean_object* v_x_714_, lean_object* v_x_715_, lean_object* v_x_716_, lean_object* v_x_717_){
_start:
{
lean_object* v___x_718_; 
v___x_718_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_alterAux___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__1_spec__5_spec__10_spec__11___redArg(v_x_714_, v_x_715_, v_x_716_, v_x_717_);
return v___x_718_;
}
}
static lean_object* _init_lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_719_; 
v___x_719_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_719_;
}
}
static lean_object* _init_lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__1(void){
_start:
{
lean_object* v___x_720_; lean_object* v___x_721_; 
v___x_720_ = lean_obj_once(&lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__0, &lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__0_once, _init_lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__0);
v___x_721_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_721_, 0, v___x_720_);
return v___x_721_;
}
}
static lean_object* _init_lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__2(void){
_start:
{
lean_object* v___x_722_; lean_object* v___x_723_; 
v___x_722_ = lean_obj_once(&lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__1, &lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__1_once, _init_lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__1);
v___x_723_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_723_, 0, v___x_722_);
lean_ctor_set(v___x_723_, 1, v___x_722_);
return v___x_723_;
}
}
static lean_object* _init_lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__3(void){
_start:
{
lean_object* v___x_724_; lean_object* v___x_725_; 
v___x_724_ = lean_obj_once(&lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__1, &lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__1_once, _init_lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__1);
v___x_725_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_725_, 0, v___x_724_);
lean_ctor_set(v___x_725_, 1, v___x_724_);
lean_ctor_set(v___x_725_, 2, v___x_724_);
lean_ctor_set(v___x_725_, 3, v___x_724_);
lean_ctor_set(v___x_725_, 4, v___x_724_);
lean_ctor_set(v___x_725_, 5, v___x_724_);
return v___x_725_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg(lean_object* v_ext_726_, lean_object* v_b_727_, uint8_t v_kind_728_, lean_object* v___y_729_, lean_object* v___y_730_, lean_object* v___y_731_){
_start:
{
lean_object* v_currNamespace_733_; lean_object* v___x_734_; lean_object* v_env_735_; lean_object* v_nextMacroScope_736_; lean_object* v_ngen_737_; lean_object* v_auxDeclNGen_738_; lean_object* v_traceState_739_; lean_object* v_messages_740_; lean_object* v_infoState_741_; lean_object* v_snapshotTasks_742_; lean_object* v___x_744_; uint8_t v_isShared_745_; uint8_t v_isSharedCheck_769_; 
v_currNamespace_733_ = lean_ctor_get(v___y_730_, 6);
v___x_734_ = lean_st_ref_take(v___y_731_);
v_env_735_ = lean_ctor_get(v___x_734_, 0);
v_nextMacroScope_736_ = lean_ctor_get(v___x_734_, 1);
v_ngen_737_ = lean_ctor_get(v___x_734_, 2);
v_auxDeclNGen_738_ = lean_ctor_get(v___x_734_, 3);
v_traceState_739_ = lean_ctor_get(v___x_734_, 4);
v_messages_740_ = lean_ctor_get(v___x_734_, 6);
v_infoState_741_ = lean_ctor_get(v___x_734_, 7);
v_snapshotTasks_742_ = lean_ctor_get(v___x_734_, 8);
v_isSharedCheck_769_ = !lean_is_exclusive(v___x_734_);
if (v_isSharedCheck_769_ == 0)
{
lean_object* v_unused_770_; 
v_unused_770_ = lean_ctor_get(v___x_734_, 5);
lean_dec(v_unused_770_);
v___x_744_ = v___x_734_;
v_isShared_745_ = v_isSharedCheck_769_;
goto v_resetjp_743_;
}
else
{
lean_inc(v_snapshotTasks_742_);
lean_inc(v_infoState_741_);
lean_inc(v_messages_740_);
lean_inc(v_traceState_739_);
lean_inc(v_auxDeclNGen_738_);
lean_inc(v_ngen_737_);
lean_inc(v_nextMacroScope_736_);
lean_inc(v_env_735_);
lean_dec(v___x_734_);
v___x_744_ = lean_box(0);
v_isShared_745_ = v_isSharedCheck_769_;
goto v_resetjp_743_;
}
v_resetjp_743_:
{
lean_object* v___x_746_; lean_object* v___x_747_; lean_object* v___x_749_; 
lean_inc(v_currNamespace_733_);
v___x_746_ = l_Lean_ScopedEnvExtension_addCore___redArg(v_env_735_, v_ext_726_, v_b_727_, v_kind_728_, v_currNamespace_733_);
v___x_747_ = lean_obj_once(&lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__2, &lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__2_once, _init_lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__2);
if (v_isShared_745_ == 0)
{
lean_ctor_set(v___x_744_, 5, v___x_747_);
lean_ctor_set(v___x_744_, 0, v___x_746_);
v___x_749_ = v___x_744_;
goto v_reusejp_748_;
}
else
{
lean_object* v_reuseFailAlloc_768_; 
v_reuseFailAlloc_768_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_768_, 0, v___x_746_);
lean_ctor_set(v_reuseFailAlloc_768_, 1, v_nextMacroScope_736_);
lean_ctor_set(v_reuseFailAlloc_768_, 2, v_ngen_737_);
lean_ctor_set(v_reuseFailAlloc_768_, 3, v_auxDeclNGen_738_);
lean_ctor_set(v_reuseFailAlloc_768_, 4, v_traceState_739_);
lean_ctor_set(v_reuseFailAlloc_768_, 5, v___x_747_);
lean_ctor_set(v_reuseFailAlloc_768_, 6, v_messages_740_);
lean_ctor_set(v_reuseFailAlloc_768_, 7, v_infoState_741_);
lean_ctor_set(v_reuseFailAlloc_768_, 8, v_snapshotTasks_742_);
v___x_749_ = v_reuseFailAlloc_768_;
goto v_reusejp_748_;
}
v_reusejp_748_:
{
lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v_mctx_752_; lean_object* v_zetaDeltaFVarIds_753_; lean_object* v_postponed_754_; lean_object* v_diag_755_; lean_object* v___x_757_; uint8_t v_isShared_758_; uint8_t v_isSharedCheck_766_; 
v___x_750_ = lean_st_ref_set(v___y_731_, v___x_749_);
v___x_751_ = lean_st_ref_take(v___y_729_);
v_mctx_752_ = lean_ctor_get(v___x_751_, 0);
v_zetaDeltaFVarIds_753_ = lean_ctor_get(v___x_751_, 2);
v_postponed_754_ = lean_ctor_get(v___x_751_, 3);
v_diag_755_ = lean_ctor_get(v___x_751_, 4);
v_isSharedCheck_766_ = !lean_is_exclusive(v___x_751_);
if (v_isSharedCheck_766_ == 0)
{
lean_object* v_unused_767_; 
v_unused_767_ = lean_ctor_get(v___x_751_, 1);
lean_dec(v_unused_767_);
v___x_757_ = v___x_751_;
v_isShared_758_ = v_isSharedCheck_766_;
goto v_resetjp_756_;
}
else
{
lean_inc(v_diag_755_);
lean_inc(v_postponed_754_);
lean_inc(v_zetaDeltaFVarIds_753_);
lean_inc(v_mctx_752_);
lean_dec(v___x_751_);
v___x_757_ = lean_box(0);
v_isShared_758_ = v_isSharedCheck_766_;
goto v_resetjp_756_;
}
v_resetjp_756_:
{
lean_object* v___x_759_; lean_object* v___x_761_; 
v___x_759_ = lean_obj_once(&lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__3, &lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__3_once, _init_lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___closed__3);
if (v_isShared_758_ == 0)
{
lean_ctor_set(v___x_757_, 1, v___x_759_);
v___x_761_ = v___x_757_;
goto v_reusejp_760_;
}
else
{
lean_object* v_reuseFailAlloc_765_; 
v_reuseFailAlloc_765_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_765_, 0, v_mctx_752_);
lean_ctor_set(v_reuseFailAlloc_765_, 1, v___x_759_);
lean_ctor_set(v_reuseFailAlloc_765_, 2, v_zetaDeltaFVarIds_753_);
lean_ctor_set(v_reuseFailAlloc_765_, 3, v_postponed_754_);
lean_ctor_set(v_reuseFailAlloc_765_, 4, v_diag_755_);
v___x_761_ = v_reuseFailAlloc_765_;
goto v_reusejp_760_;
}
v_reusejp_760_:
{
lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; 
v___x_762_ = lean_st_ref_set(v___y_729_, v___x_761_);
v___x_763_ = lean_box(0);
v___x_764_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_764_, 0, v___x_763_);
return v___x_764_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg___boxed(lean_object* v_ext_771_, lean_object* v_b_772_, lean_object* v_kind_773_, lean_object* v___y_774_, lean_object* v___y_775_, lean_object* v___y_776_, lean_object* v___y_777_){
_start:
{
uint8_t v_kind_boxed_778_; lean_object* v_res_779_; 
v_kind_boxed_778_ = lean_unbox(v_kind_773_);
v_res_779_ = lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg(v_ext_771_, v_b_772_, v_kind_boxed_778_, v___y_774_, v___y_775_, v___y_776_);
lean_dec(v___y_776_);
lean_dec_ref(v___y_775_);
lean_dec(v___y_774_);
return v_res_779_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2(lean_object* v_00_u03b1_780_, lean_object* v_00_u03b2_781_, lean_object* v_00_u03c3_782_, lean_object* v_ext_783_, lean_object* v_b_784_, uint8_t v_kind_785_, lean_object* v___y_786_, lean_object* v___y_787_, lean_object* v___y_788_, lean_object* v___y_789_){
_start:
{
lean_object* v___x_791_; 
v___x_791_ = lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg(v_ext_783_, v_b_784_, v_kind_785_, v___y_787_, v___y_788_, v___y_789_);
return v___x_791_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___boxed(lean_object* v_00_u03b1_792_, lean_object* v_00_u03b2_793_, lean_object* v_00_u03c3_794_, lean_object* v_ext_795_, lean_object* v_b_796_, lean_object* v_kind_797_, lean_object* v___y_798_, lean_object* v___y_799_, lean_object* v___y_800_, lean_object* v___y_801_, lean_object* v___y_802_){
_start:
{
uint8_t v_kind_boxed_803_; lean_object* v_res_804_; 
v_kind_boxed_803_ = lean_unbox(v_kind_797_);
v_res_804_ = lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2(v_00_u03b1_792_, v_00_u03b2_793_, v_00_u03c3_794_, v_ext_795_, v_b_796_, v_kind_boxed_803_, v___y_798_, v___y_799_, v___y_800_, v___y_801_);
lean_dec(v___y_801_);
lean_dec_ref(v___y_800_);
lean_dec(v___y_799_);
lean_dec_ref(v___y_798_);
return v_res_804_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__0(void){
_start:
{
lean_object* v___x_805_; 
v___x_805_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_805_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__1(void){
_start:
{
lean_object* v___x_806_; lean_object* v___x_807_; 
v___x_806_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__0, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__0_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__0);
v___x_807_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_807_, 0, v___x_806_);
return v___x_807_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__2(void){
_start:
{
lean_object* v___x_808_; lean_object* v___x_809_; lean_object* v___x_810_; 
v___x_808_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__1, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__1_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__1);
v___x_809_ = lean_unsigned_to_nat(0u);
v___x_810_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_810_, 0, v___x_809_);
lean_ctor_set(v___x_810_, 1, v___x_809_);
lean_ctor_set(v___x_810_, 2, v___x_809_);
lean_ctor_set(v___x_810_, 3, v___x_809_);
lean_ctor_set(v___x_810_, 4, v___x_808_);
lean_ctor_set(v___x_810_, 5, v___x_808_);
lean_ctor_set(v___x_810_, 6, v___x_808_);
lean_ctor_set(v___x_810_, 7, v___x_808_);
lean_ctor_set(v___x_810_, 8, v___x_808_);
lean_ctor_set(v___x_810_, 9, v___x_808_);
return v___x_810_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__3(void){
_start:
{
lean_object* v___x_811_; lean_object* v___x_812_; lean_object* v___x_813_; 
v___x_811_ = lean_unsigned_to_nat(32u);
v___x_812_ = lean_mk_empty_array_with_capacity(v___x_811_);
v___x_813_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_813_, 0, v___x_812_);
return v___x_813_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__4(void){
_start:
{
size_t v___x_814_; lean_object* v___x_815_; lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; 
v___x_814_ = ((size_t)5ULL);
v___x_815_ = lean_unsigned_to_nat(0u);
v___x_816_ = lean_unsigned_to_nat(32u);
v___x_817_ = lean_mk_empty_array_with_capacity(v___x_816_);
v___x_818_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__3, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__3_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__3);
v___x_819_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_819_, 0, v___x_818_);
lean_ctor_set(v___x_819_, 1, v___x_817_);
lean_ctor_set(v___x_819_, 2, v___x_815_);
lean_ctor_set(v___x_819_, 3, v___x_815_);
lean_ctor_set_usize(v___x_819_, 4, v___x_814_);
return v___x_819_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__5(void){
_start:
{
lean_object* v___x_820_; lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v___x_823_; 
v___x_820_ = lean_box(1);
v___x_821_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__4, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__4_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__4);
v___x_822_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__1, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__1_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__1);
v___x_823_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_823_, 0, v___x_822_);
lean_ctor_set(v___x_823_, 1, v___x_821_);
lean_ctor_set(v___x_823_, 2, v___x_820_);
return v___x_823_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__7(void){
_start:
{
lean_object* v___x_825_; lean_object* v___x_826_; 
v___x_825_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__6));
v___x_826_ = l_Lean_stringToMessageData(v___x_825_);
return v___x_826_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__9(void){
_start:
{
lean_object* v___x_828_; lean_object* v___x_829_; 
v___x_828_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__8));
v___x_829_ = l_Lean_stringToMessageData(v___x_828_);
return v___x_829_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__11(void){
_start:
{
lean_object* v___x_831_; lean_object* v___x_832_; 
v___x_831_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__10));
v___x_832_ = l_Lean_stringToMessageData(v___x_831_);
return v___x_832_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__13(void){
_start:
{
lean_object* v___x_834_; lean_object* v___x_835_; 
v___x_834_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__12));
v___x_835_ = l_Lean_stringToMessageData(v___x_834_);
return v___x_835_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__15(void){
_start:
{
lean_object* v___x_837_; lean_object* v___x_838_; 
v___x_837_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__14));
v___x_838_ = l_Lean_stringToMessageData(v___x_837_);
return v___x_838_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__17(void){
_start:
{
lean_object* v___x_840_; lean_object* v___x_841_; 
v___x_840_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__16));
v___x_841_ = l_Lean_stringToMessageData(v___x_840_);
return v___x_841_;
}
}
static lean_object* _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__19(void){
_start:
{
lean_object* v___x_843_; lean_object* v___x_844_; 
v___x_843_ = ((lean_object*)(lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__18));
v___x_844_ = l_Lean_stringToMessageData(v___x_843_);
return v___x_844_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg(lean_object* v_msg_845_, lean_object* v_declHint_846_, lean_object* v___y_847_){
_start:
{
lean_object* v___x_849_; lean_object* v_env_850_; uint8_t v___x_851_; 
v___x_849_ = lean_st_ref_get(v___y_847_);
v_env_850_ = lean_ctor_get(v___x_849_, 0);
lean_inc_ref(v_env_850_);
lean_dec(v___x_849_);
v___x_851_ = l_Lean_Name_isAnonymous(v_declHint_846_);
if (v___x_851_ == 0)
{
uint8_t v_isExporting_852_; 
v_isExporting_852_ = lean_ctor_get_uint8(v_env_850_, sizeof(void*)*8);
if (v_isExporting_852_ == 0)
{
lean_object* v___x_853_; 
lean_dec_ref(v_env_850_);
lean_dec(v_declHint_846_);
v___x_853_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_853_, 0, v_msg_845_);
return v___x_853_;
}
else
{
lean_object* v___x_854_; uint8_t v___x_855_; 
lean_inc_ref(v_env_850_);
v___x_854_ = l_Lean_Environment_setExporting(v_env_850_, v___x_851_);
lean_inc(v_declHint_846_);
lean_inc_ref(v___x_854_);
v___x_855_ = l_Lean_Environment_contains(v___x_854_, v_declHint_846_, v_isExporting_852_);
if (v___x_855_ == 0)
{
lean_object* v___x_856_; 
lean_dec_ref(v___x_854_);
lean_dec_ref(v_env_850_);
lean_dec(v_declHint_846_);
v___x_856_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_856_, 0, v_msg_845_);
return v___x_856_;
}
else
{
lean_object* v___x_857_; lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_861_; lean_object* v_c_862_; lean_object* v___x_863_; 
v___x_857_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__2, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__2_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__2);
v___x_858_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__5, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__5_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__5);
v___x_859_ = l_Lean_Options_empty;
v___x_860_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_860_, 0, v___x_854_);
lean_ctor_set(v___x_860_, 1, v___x_857_);
lean_ctor_set(v___x_860_, 2, v___x_858_);
lean_ctor_set(v___x_860_, 3, v___x_859_);
lean_inc(v_declHint_846_);
v___x_861_ = l_Lean_MessageData_ofConstName(v_declHint_846_, v___x_851_);
v_c_862_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_c_862_, 0, v___x_860_);
lean_ctor_set(v_c_862_, 1, v___x_861_);
v___x_863_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_850_, v_declHint_846_);
if (lean_obj_tag(v___x_863_) == 0)
{
lean_object* v___x_864_; lean_object* v___x_865_; lean_object* v___x_866_; lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; lean_object* v___x_870_; 
lean_dec_ref(v_env_850_);
lean_dec(v_declHint_846_);
v___x_864_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__7, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__7_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__7);
v___x_865_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_865_, 0, v___x_864_);
lean_ctor_set(v___x_865_, 1, v_c_862_);
v___x_866_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__9, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__9_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__9);
v___x_867_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_867_, 0, v___x_865_);
lean_ctor_set(v___x_867_, 1, v___x_866_);
v___x_868_ = l_Lean_MessageData_note(v___x_867_);
v___x_869_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_869_, 0, v_msg_845_);
lean_ctor_set(v___x_869_, 1, v___x_868_);
v___x_870_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_870_, 0, v___x_869_);
return v___x_870_;
}
else
{
lean_object* v_val_871_; lean_object* v___x_873_; uint8_t v_isShared_874_; uint8_t v_isSharedCheck_906_; 
v_val_871_ = lean_ctor_get(v___x_863_, 0);
v_isSharedCheck_906_ = !lean_is_exclusive(v___x_863_);
if (v_isSharedCheck_906_ == 0)
{
v___x_873_ = v___x_863_;
v_isShared_874_ = v_isSharedCheck_906_;
goto v_resetjp_872_;
}
else
{
lean_inc(v_val_871_);
lean_dec(v___x_863_);
v___x_873_ = lean_box(0);
v_isShared_874_ = v_isSharedCheck_906_;
goto v_resetjp_872_;
}
v_resetjp_872_:
{
lean_object* v___x_875_; lean_object* v___x_876_; lean_object* v___x_877_; lean_object* v_mod_878_; uint8_t v___x_879_; 
v___x_875_ = lean_box(0);
v___x_876_ = l_Lean_Environment_header(v_env_850_);
lean_dec_ref(v_env_850_);
v___x_877_ = l_Lean_EnvironmentHeader_moduleNames(v___x_876_);
v_mod_878_ = lean_array_get(v___x_875_, v___x_877_, v_val_871_);
lean_dec(v_val_871_);
lean_dec_ref(v___x_877_);
v___x_879_ = l_Lean_isPrivateName(v_declHint_846_);
lean_dec(v_declHint_846_);
if (v___x_879_ == 0)
{
lean_object* v___x_880_; lean_object* v___x_881_; lean_object* v___x_882_; lean_object* v___x_883_; lean_object* v___x_884_; lean_object* v___x_885_; lean_object* v___x_886_; lean_object* v___x_887_; lean_object* v___x_888_; lean_object* v___x_889_; lean_object* v___x_891_; 
v___x_880_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__11, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__11_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__11);
v___x_881_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_881_, 0, v___x_880_);
lean_ctor_set(v___x_881_, 1, v_c_862_);
v___x_882_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__13, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__13_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__13);
v___x_883_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_883_, 0, v___x_881_);
lean_ctor_set(v___x_883_, 1, v___x_882_);
v___x_884_ = l_Lean_MessageData_ofName(v_mod_878_);
v___x_885_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_885_, 0, v___x_883_);
lean_ctor_set(v___x_885_, 1, v___x_884_);
v___x_886_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__15, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__15_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__15);
v___x_887_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_887_, 0, v___x_885_);
lean_ctor_set(v___x_887_, 1, v___x_886_);
v___x_888_ = l_Lean_MessageData_note(v___x_887_);
v___x_889_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_889_, 0, v_msg_845_);
lean_ctor_set(v___x_889_, 1, v___x_888_);
if (v_isShared_874_ == 0)
{
lean_ctor_set_tag(v___x_873_, 0);
lean_ctor_set(v___x_873_, 0, v___x_889_);
v___x_891_ = v___x_873_;
goto v_reusejp_890_;
}
else
{
lean_object* v_reuseFailAlloc_892_; 
v_reuseFailAlloc_892_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_892_, 0, v___x_889_);
v___x_891_ = v_reuseFailAlloc_892_;
goto v_reusejp_890_;
}
v_reusejp_890_:
{
return v___x_891_;
}
}
else
{
lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; lean_object* v___x_901_; lean_object* v___x_902_; lean_object* v___x_904_; 
v___x_893_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__7, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__7_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__7);
v___x_894_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_894_, 0, v___x_893_);
lean_ctor_set(v___x_894_, 1, v_c_862_);
v___x_895_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__17, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__17_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__17);
v___x_896_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_896_, 0, v___x_894_);
lean_ctor_set(v___x_896_, 1, v___x_895_);
v___x_897_ = l_Lean_MessageData_ofName(v_mod_878_);
v___x_898_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_898_, 0, v___x_896_);
lean_ctor_set(v___x_898_, 1, v___x_897_);
v___x_899_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__19, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__19_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__19);
v___x_900_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_900_, 0, v___x_898_);
lean_ctor_set(v___x_900_, 1, v___x_899_);
v___x_901_ = l_Lean_MessageData_note(v___x_900_);
v___x_902_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_902_, 0, v_msg_845_);
lean_ctor_set(v___x_902_, 1, v___x_901_);
if (v_isShared_874_ == 0)
{
lean_ctor_set_tag(v___x_873_, 0);
lean_ctor_set(v___x_873_, 0, v___x_902_);
v___x_904_ = v___x_873_;
goto v_reusejp_903_;
}
else
{
lean_object* v_reuseFailAlloc_905_; 
v_reuseFailAlloc_905_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_905_, 0, v___x_902_);
v___x_904_ = v_reuseFailAlloc_905_;
goto v_reusejp_903_;
}
v_reusejp_903_:
{
return v___x_904_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_907_; 
lean_dec_ref(v_env_850_);
lean_dec(v_declHint_846_);
v___x_907_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_907_, 0, v_msg_845_);
return v___x_907_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___boxed(lean_object* v_msg_908_, lean_object* v_declHint_909_, lean_object* v___y_910_, lean_object* v___y_911_){
_start:
{
lean_object* v_res_912_; 
v_res_912_ = lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg(v_msg_908_, v_declHint_909_, v___y_910_);
lean_dec(v___y_910_);
return v_res_912_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8(lean_object* v_msg_913_, lean_object* v_declHint_914_, lean_object* v___y_915_, lean_object* v___y_916_, lean_object* v___y_917_, lean_object* v___y_918_){
_start:
{
lean_object* v___x_920_; lean_object* v_a_921_; lean_object* v___x_923_; uint8_t v_isShared_924_; uint8_t v_isSharedCheck_930_; 
v___x_920_ = lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg(v_msg_913_, v_declHint_914_, v___y_918_);
v_a_921_ = lean_ctor_get(v___x_920_, 0);
v_isSharedCheck_930_ = !lean_is_exclusive(v___x_920_);
if (v_isSharedCheck_930_ == 0)
{
v___x_923_ = v___x_920_;
v_isShared_924_ = v_isSharedCheck_930_;
goto v_resetjp_922_;
}
else
{
lean_inc(v_a_921_);
lean_dec(v___x_920_);
v___x_923_ = lean_box(0);
v_isShared_924_ = v_isSharedCheck_930_;
goto v_resetjp_922_;
}
v_resetjp_922_:
{
lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_928_; 
v___x_925_ = l_Lean_unknownIdentifierMessageTag;
v___x_926_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_926_, 0, v___x_925_);
lean_ctor_set(v___x_926_, 1, v_a_921_);
if (v_isShared_924_ == 0)
{
lean_ctor_set(v___x_923_, 0, v___x_926_);
v___x_928_ = v___x_923_;
goto v_reusejp_927_;
}
else
{
lean_object* v_reuseFailAlloc_929_; 
v_reuseFailAlloc_929_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_929_, 0, v___x_926_);
v___x_928_ = v_reuseFailAlloc_929_;
goto v_reusejp_927_;
}
v_reusejp_927_:
{
return v___x_928_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8___boxed(lean_object* v_msg_931_, lean_object* v_declHint_932_, lean_object* v___y_933_, lean_object* v___y_934_, lean_object* v___y_935_, lean_object* v___y_936_, lean_object* v___y_937_){
_start:
{
lean_object* v_res_938_; 
v_res_938_ = lp_batteries_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8(v_msg_931_, v_declHint_932_, v___y_933_, v___y_934_, v___y_935_, v___y_936_);
lean_dec(v___y_936_);
lean_dec_ref(v___y_935_);
lean_dec(v___y_934_);
lean_dec_ref(v___y_933_);
return v_res_938_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1_spec__2(lean_object* v_msgData_939_, lean_object* v___y_940_, lean_object* v___y_941_, lean_object* v___y_942_, lean_object* v___y_943_){
_start:
{
lean_object* v___x_945_; lean_object* v_env_946_; lean_object* v___x_947_; lean_object* v_mctx_948_; lean_object* v_lctx_949_; lean_object* v_options_950_; lean_object* v___x_951_; lean_object* v___x_952_; lean_object* v___x_953_; 
v___x_945_ = lean_st_ref_get(v___y_943_);
v_env_946_ = lean_ctor_get(v___x_945_, 0);
lean_inc_ref(v_env_946_);
lean_dec(v___x_945_);
v___x_947_ = lean_st_ref_get(v___y_941_);
v_mctx_948_ = lean_ctor_get(v___x_947_, 0);
lean_inc_ref(v_mctx_948_);
lean_dec(v___x_947_);
v_lctx_949_ = lean_ctor_get(v___y_940_, 2);
v_options_950_ = lean_ctor_get(v___y_942_, 2);
lean_inc_ref(v_options_950_);
lean_inc_ref(v_lctx_949_);
v___x_951_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_951_, 0, v_env_946_);
lean_ctor_set(v___x_951_, 1, v_mctx_948_);
lean_ctor_set(v___x_951_, 2, v_lctx_949_);
lean_ctor_set(v___x_951_, 3, v_options_950_);
v___x_952_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_952_, 0, v___x_951_);
lean_ctor_set(v___x_952_, 1, v_msgData_939_);
v___x_953_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_953_, 0, v___x_952_);
return v___x_953_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1_spec__2___boxed(lean_object* v_msgData_954_, lean_object* v___y_955_, lean_object* v___y_956_, lean_object* v___y_957_, lean_object* v___y_958_, lean_object* v___y_959_){
_start:
{
lean_object* v_res_960_; 
v_res_960_ = lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1_spec__2(v_msgData_954_, v___y_955_, v___y_956_, v___y_957_, v___y_958_);
lean_dec(v___y_958_);
lean_dec_ref(v___y_957_);
lean_dec(v___y_956_);
lean_dec_ref(v___y_955_);
return v_res_960_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1___redArg(lean_object* v_msg_961_, lean_object* v___y_962_, lean_object* v___y_963_, lean_object* v___y_964_, lean_object* v___y_965_){
_start:
{
lean_object* v_ref_967_; lean_object* v___x_968_; lean_object* v_a_969_; lean_object* v___x_971_; uint8_t v_isShared_972_; uint8_t v_isSharedCheck_977_; 
v_ref_967_ = lean_ctor_get(v___y_964_, 5);
v___x_968_ = lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1_spec__2(v_msg_961_, v___y_962_, v___y_963_, v___y_964_, v___y_965_);
v_a_969_ = lean_ctor_get(v___x_968_, 0);
v_isSharedCheck_977_ = !lean_is_exclusive(v___x_968_);
if (v_isSharedCheck_977_ == 0)
{
v___x_971_ = v___x_968_;
v_isShared_972_ = v_isSharedCheck_977_;
goto v_resetjp_970_;
}
else
{
lean_inc(v_a_969_);
lean_dec(v___x_968_);
v___x_971_ = lean_box(0);
v_isShared_972_ = v_isSharedCheck_977_;
goto v_resetjp_970_;
}
v_resetjp_970_:
{
lean_object* v___x_973_; lean_object* v___x_975_; 
lean_inc(v_ref_967_);
v___x_973_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_973_, 0, v_ref_967_);
lean_ctor_set(v___x_973_, 1, v_a_969_);
if (v_isShared_972_ == 0)
{
lean_ctor_set_tag(v___x_971_, 1);
lean_ctor_set(v___x_971_, 0, v___x_973_);
v___x_975_ = v___x_971_;
goto v_reusejp_974_;
}
else
{
lean_object* v_reuseFailAlloc_976_; 
v_reuseFailAlloc_976_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_976_, 0, v___x_973_);
v___x_975_ = v_reuseFailAlloc_976_;
goto v_reusejp_974_;
}
v_reusejp_974_:
{
return v___x_975_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1___redArg___boxed(lean_object* v_msg_978_, lean_object* v___y_979_, lean_object* v___y_980_, lean_object* v___y_981_, lean_object* v___y_982_, lean_object* v___y_983_){
_start:
{
lean_object* v_res_984_; 
v_res_984_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1___redArg(v_msg_978_, v___y_979_, v___y_980_, v___y_981_, v___y_982_);
lean_dec(v___y_982_);
lean_dec_ref(v___y_981_);
lean_dec(v___y_980_);
lean_dec_ref(v___y_979_);
return v_res_984_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__9___redArg(lean_object* v_ref_985_, lean_object* v_msg_986_, lean_object* v___y_987_, lean_object* v___y_988_, lean_object* v___y_989_, lean_object* v___y_990_){
_start:
{
lean_object* v_fileName_992_; lean_object* v_fileMap_993_; lean_object* v_options_994_; lean_object* v_currRecDepth_995_; lean_object* v_maxRecDepth_996_; lean_object* v_ref_997_; lean_object* v_currNamespace_998_; lean_object* v_openDecls_999_; lean_object* v_initHeartbeats_1000_; lean_object* v_maxHeartbeats_1001_; lean_object* v_quotContext_1002_; lean_object* v_currMacroScope_1003_; uint8_t v_diag_1004_; lean_object* v_cancelTk_x3f_1005_; uint8_t v_suppressElabErrors_1006_; lean_object* v_inheritedTraceOptions_1007_; lean_object* v_ref_1008_; lean_object* v___x_1009_; lean_object* v___x_1010_; 
v_fileName_992_ = lean_ctor_get(v___y_989_, 0);
v_fileMap_993_ = lean_ctor_get(v___y_989_, 1);
v_options_994_ = lean_ctor_get(v___y_989_, 2);
v_currRecDepth_995_ = lean_ctor_get(v___y_989_, 3);
v_maxRecDepth_996_ = lean_ctor_get(v___y_989_, 4);
v_ref_997_ = lean_ctor_get(v___y_989_, 5);
v_currNamespace_998_ = lean_ctor_get(v___y_989_, 6);
v_openDecls_999_ = lean_ctor_get(v___y_989_, 7);
v_initHeartbeats_1000_ = lean_ctor_get(v___y_989_, 8);
v_maxHeartbeats_1001_ = lean_ctor_get(v___y_989_, 9);
v_quotContext_1002_ = lean_ctor_get(v___y_989_, 10);
v_currMacroScope_1003_ = lean_ctor_get(v___y_989_, 11);
v_diag_1004_ = lean_ctor_get_uint8(v___y_989_, sizeof(void*)*14);
v_cancelTk_x3f_1005_ = lean_ctor_get(v___y_989_, 12);
v_suppressElabErrors_1006_ = lean_ctor_get_uint8(v___y_989_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1007_ = lean_ctor_get(v___y_989_, 13);
v_ref_1008_ = l_Lean_replaceRef(v_ref_985_, v_ref_997_);
lean_inc_ref(v_inheritedTraceOptions_1007_);
lean_inc(v_cancelTk_x3f_1005_);
lean_inc(v_currMacroScope_1003_);
lean_inc(v_quotContext_1002_);
lean_inc(v_maxHeartbeats_1001_);
lean_inc(v_initHeartbeats_1000_);
lean_inc(v_openDecls_999_);
lean_inc(v_currNamespace_998_);
lean_inc(v_maxRecDepth_996_);
lean_inc(v_currRecDepth_995_);
lean_inc_ref(v_options_994_);
lean_inc_ref(v_fileMap_993_);
lean_inc_ref(v_fileName_992_);
v___x_1009_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1009_, 0, v_fileName_992_);
lean_ctor_set(v___x_1009_, 1, v_fileMap_993_);
lean_ctor_set(v___x_1009_, 2, v_options_994_);
lean_ctor_set(v___x_1009_, 3, v_currRecDepth_995_);
lean_ctor_set(v___x_1009_, 4, v_maxRecDepth_996_);
lean_ctor_set(v___x_1009_, 5, v_ref_1008_);
lean_ctor_set(v___x_1009_, 6, v_currNamespace_998_);
lean_ctor_set(v___x_1009_, 7, v_openDecls_999_);
lean_ctor_set(v___x_1009_, 8, v_initHeartbeats_1000_);
lean_ctor_set(v___x_1009_, 9, v_maxHeartbeats_1001_);
lean_ctor_set(v___x_1009_, 10, v_quotContext_1002_);
lean_ctor_set(v___x_1009_, 11, v_currMacroScope_1003_);
lean_ctor_set(v___x_1009_, 12, v_cancelTk_x3f_1005_);
lean_ctor_set(v___x_1009_, 13, v_inheritedTraceOptions_1007_);
lean_ctor_set_uint8(v___x_1009_, sizeof(void*)*14, v_diag_1004_);
lean_ctor_set_uint8(v___x_1009_, sizeof(void*)*14 + 1, v_suppressElabErrors_1006_);
v___x_1010_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1___redArg(v_msg_986_, v___y_987_, v___y_988_, v___x_1009_, v___y_990_);
lean_dec_ref_known(v___x_1009_, 14);
return v___x_1010_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__9___redArg___boxed(lean_object* v_ref_1011_, lean_object* v_msg_1012_, lean_object* v___y_1013_, lean_object* v___y_1014_, lean_object* v___y_1015_, lean_object* v___y_1016_, lean_object* v___y_1017_){
_start:
{
lean_object* v_res_1018_; 
v_res_1018_ = lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__9___redArg(v_ref_1011_, v_msg_1012_, v___y_1013_, v___y_1014_, v___y_1015_, v___y_1016_);
lean_dec(v___y_1016_);
lean_dec_ref(v___y_1015_);
lean_dec(v___y_1014_);
lean_dec_ref(v___y_1013_);
lean_dec(v_ref_1011_);
return v_res_1018_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7___redArg(lean_object* v_ref_1019_, lean_object* v_msg_1020_, lean_object* v_declHint_1021_, lean_object* v___y_1022_, lean_object* v___y_1023_, lean_object* v___y_1024_, lean_object* v___y_1025_){
_start:
{
lean_object* v___x_1027_; lean_object* v_a_1028_; lean_object* v___x_1029_; 
v___x_1027_ = lp_batteries_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8(v_msg_1020_, v_declHint_1021_, v___y_1022_, v___y_1023_, v___y_1024_, v___y_1025_);
v_a_1028_ = lean_ctor_get(v___x_1027_, 0);
lean_inc(v_a_1028_);
lean_dec_ref(v___x_1027_);
v___x_1029_ = lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__9___redArg(v_ref_1019_, v_a_1028_, v___y_1022_, v___y_1023_, v___y_1024_, v___y_1025_);
return v___x_1029_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7___redArg___boxed(lean_object* v_ref_1030_, lean_object* v_msg_1031_, lean_object* v_declHint_1032_, lean_object* v___y_1033_, lean_object* v___y_1034_, lean_object* v___y_1035_, lean_object* v___y_1036_, lean_object* v___y_1037_){
_start:
{
lean_object* v_res_1038_; 
v_res_1038_ = lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7___redArg(v_ref_1030_, v_msg_1031_, v_declHint_1032_, v___y_1033_, v___y_1034_, v___y_1035_, v___y_1036_);
lean_dec(v___y_1036_);
lean_dec_ref(v___y_1035_);
lean_dec(v___y_1034_);
lean_dec_ref(v___y_1033_);
lean_dec(v_ref_1030_);
return v_res_1038_;
}
}
static lean_object* _init_lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg___closed__1(void){
_start:
{
lean_object* v___x_1040_; lean_object* v___x_1041_; 
v___x_1040_ = ((lean_object*)(lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg___closed__0));
v___x_1041_ = l_Lean_stringToMessageData(v___x_1040_);
return v___x_1041_;
}
}
static lean_object* _init_lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg___closed__3(void){
_start:
{
lean_object* v___x_1043_; lean_object* v___x_1044_; 
v___x_1043_ = ((lean_object*)(lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg___closed__2));
v___x_1044_ = l_Lean_stringToMessageData(v___x_1043_);
return v___x_1044_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg(lean_object* v_ref_1045_, lean_object* v_constName_1046_, lean_object* v___y_1047_, lean_object* v___y_1048_, lean_object* v___y_1049_, lean_object* v___y_1050_){
_start:
{
lean_object* v___x_1052_; uint8_t v___x_1053_; lean_object* v___x_1054_; lean_object* v___x_1055_; lean_object* v___x_1056_; lean_object* v___x_1057_; lean_object* v___x_1058_; 
v___x_1052_ = lean_obj_once(&lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg___closed__1, &lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg___closed__1_once, _init_lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg___closed__1);
v___x_1053_ = 0;
lean_inc(v_constName_1046_);
v___x_1054_ = l_Lean_MessageData_ofConstName(v_constName_1046_, v___x_1053_);
v___x_1055_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1055_, 0, v___x_1052_);
lean_ctor_set(v___x_1055_, 1, v___x_1054_);
v___x_1056_ = lean_obj_once(&lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg___closed__3, &lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg___closed__3_once, _init_lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg___closed__3);
v___x_1057_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1057_, 0, v___x_1055_);
lean_ctor_set(v___x_1057_, 1, v___x_1056_);
v___x_1058_ = lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7___redArg(v_ref_1045_, v___x_1057_, v_constName_1046_, v___y_1047_, v___y_1048_, v___y_1049_, v___y_1050_);
return v___x_1058_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg___boxed(lean_object* v_ref_1059_, lean_object* v_constName_1060_, lean_object* v___y_1061_, lean_object* v___y_1062_, lean_object* v___y_1063_, lean_object* v___y_1064_, lean_object* v___y_1065_){
_start:
{
lean_object* v_res_1066_; 
v_res_1066_ = lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg(v_ref_1059_, v_constName_1060_, v___y_1061_, v___y_1062_, v___y_1063_, v___y_1064_);
lean_dec(v___y_1064_);
lean_dec_ref(v___y_1063_);
lean_dec(v___y_1062_);
lean_dec_ref(v___y_1061_);
lean_dec(v_ref_1059_);
return v_res_1066_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0___redArg(lean_object* v_constName_1067_, lean_object* v___y_1068_, lean_object* v___y_1069_, lean_object* v___y_1070_, lean_object* v___y_1071_){
_start:
{
lean_object* v_ref_1073_; lean_object* v___x_1074_; 
v_ref_1073_ = lean_ctor_get(v___y_1070_, 5);
v___x_1074_ = lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg(v_ref_1073_, v_constName_1067_, v___y_1068_, v___y_1069_, v___y_1070_, v___y_1071_);
return v___x_1074_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0___redArg___boxed(lean_object* v_constName_1075_, lean_object* v___y_1076_, lean_object* v___y_1077_, lean_object* v___y_1078_, lean_object* v___y_1079_, lean_object* v___y_1080_){
_start:
{
lean_object* v_res_1081_; 
v_res_1081_ = lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0___redArg(v_constName_1075_, v___y_1076_, v___y_1077_, v___y_1078_, v___y_1079_);
lean_dec(v___y_1079_);
lean_dec_ref(v___y_1078_);
lean_dec(v___y_1077_);
lean_dec_ref(v___y_1076_);
return v_res_1081_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0(lean_object* v_constName_1082_, lean_object* v___y_1083_, lean_object* v___y_1084_, lean_object* v___y_1085_, lean_object* v___y_1086_){
_start:
{
lean_object* v___x_1088_; lean_object* v_env_1089_; uint8_t v___x_1090_; lean_object* v___x_1091_; 
v___x_1088_ = lean_st_ref_get(v___y_1086_);
v_env_1089_ = lean_ctor_get(v___x_1088_, 0);
lean_inc_ref(v_env_1089_);
lean_dec(v___x_1088_);
v___x_1090_ = 0;
lean_inc(v_constName_1082_);
v___x_1091_ = l_Lean_Environment_find_x3f(v_env_1089_, v_constName_1082_, v___x_1090_);
if (lean_obj_tag(v___x_1091_) == 0)
{
lean_object* v___x_1092_; 
v___x_1092_ = lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0___redArg(v_constName_1082_, v___y_1083_, v___y_1084_, v___y_1085_, v___y_1086_);
return v___x_1092_;
}
else
{
lean_object* v_val_1093_; lean_object* v___x_1095_; uint8_t v_isShared_1096_; uint8_t v_isSharedCheck_1100_; 
lean_dec(v_constName_1082_);
v_val_1093_ = lean_ctor_get(v___x_1091_, 0);
v_isSharedCheck_1100_ = !lean_is_exclusive(v___x_1091_);
if (v_isSharedCheck_1100_ == 0)
{
v___x_1095_ = v___x_1091_;
v_isShared_1096_ = v_isSharedCheck_1100_;
goto v_resetjp_1094_;
}
else
{
lean_inc(v_val_1093_);
lean_dec(v___x_1091_);
v___x_1095_ = lean_box(0);
v_isShared_1096_ = v_isSharedCheck_1100_;
goto v_resetjp_1094_;
}
v_resetjp_1094_:
{
lean_object* v___x_1098_; 
if (v_isShared_1096_ == 0)
{
lean_ctor_set_tag(v___x_1095_, 0);
v___x_1098_ = v___x_1095_;
goto v_reusejp_1097_;
}
else
{
lean_object* v_reuseFailAlloc_1099_; 
v_reuseFailAlloc_1099_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1099_, 0, v_val_1093_);
v___x_1098_ = v_reuseFailAlloc_1099_;
goto v_reusejp_1097_;
}
v_reusejp_1097_:
{
return v___x_1098_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0___boxed(lean_object* v_constName_1101_, lean_object* v___y_1102_, lean_object* v___y_1103_, lean_object* v___y_1104_, lean_object* v___y_1105_, lean_object* v___y_1106_){
_start:
{
lean_object* v_res_1107_; 
v_res_1107_ = lp_batteries_Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0(v_constName_1101_, v___y_1102_, v___y_1103_, v___y_1104_, v___y_1105_);
lean_dec(v___y_1105_);
lean_dec_ref(v___y_1104_);
lean_dec(v___y_1103_);
lean_dec_ref(v___y_1102_);
return v_res_1107_;
}
}
static uint64_t _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__1_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1114_; uint64_t v___x_1115_; 
v___x_1114_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__0_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_));
v___x_1115_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_1114_);
return v___x_1115_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__2_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_(void){
_start:
{
uint64_t v___x_1116_; lean_object* v___x_1117_; lean_object* v___x_1118_; 
v___x_1116_ = lean_uint64_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__1_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__1_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__1_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_);
v___x_1117_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__0_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_));
v___x_1118_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_1118_, 0, v___x_1117_);
lean_ctor_set_uint64(v___x_1118_, sizeof(void*)*1, v___x_1116_);
return v___x_1118_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__3_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1119_; 
v___x_1119_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1119_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__4_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1120_; lean_object* v___x_1121_; 
v___x_1120_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__3_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__3_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__3_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_);
v___x_1121_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1121_, 0, v___x_1120_);
return v___x_1121_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__5_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1122_; lean_object* v___x_1123_; 
v___x_1122_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__4_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__4_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__4_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_);
v___x_1123_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1123_, 0, v___x_1122_);
lean_ctor_set(v___x_1123_, 1, v___x_1122_);
lean_ctor_set(v___x_1123_, 2, v___x_1122_);
lean_ctor_set(v___x_1123_, 3, v___x_1122_);
lean_ctor_set(v___x_1123_, 4, v___x_1122_);
lean_ctor_set(v___x_1123_, 5, v___x_1122_);
return v___x_1123_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__6_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1124_; lean_object* v___x_1125_; 
v___x_1124_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__4_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__4_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__4_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_);
v___x_1125_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1125_, 0, v___x_1124_);
lean_ctor_set(v___x_1125_, 1, v___x_1124_);
lean_ctor_set(v___x_1125_, 2, v___x_1124_);
lean_ctor_set(v___x_1125_, 3, v___x_1124_);
lean_ctor_set(v___x_1125_, 4, v___x_1124_);
return v___x_1125_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__7_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1126_; uint8_t v___x_1127_; lean_object* v___x_1128_; 
v___x_1126_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__2_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__2_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__2_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_);
v___x_1127_ = 2;
v___x_1128_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1127_, v___x_1126_);
return v___x_1128_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__9_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1130_; lean_object* v___x_1131_; 
v___x_1130_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__8_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_));
v___x_1131_ = l_Lean_stringToMessageData(v___x_1130_);
return v___x_1131_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__11_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1133_; lean_object* v___x_1134_; 
v___x_1133_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__10_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_));
v___x_1134_ = l_Lean_stringToMessageData(v___x_1133_);
return v___x_1134_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_(lean_object* v___x_1135_, lean_object* v___x_1136_, lean_object* v_decl_1137_, lean_object* v_x_1138_, uint8_t v_kind_1139_, lean_object* v___y_1140_, lean_object* v___y_1141_){
_start:
{
uint8_t v___x_1143_; uint8_t v___x_1144_; lean_object* v___x_1145_; lean_object* v___x_1146_; lean_object* v___x_1147_; lean_object* v___x_1148_; lean_object* v___x_1149_; size_t v___x_1150_; lean_object* v___x_1151_; lean_object* v___x_1152_; lean_object* v___x_1153_; lean_object* v___x_1154_; lean_object* v___x_1155_; lean_object* v___x_1156_; lean_object* v___x_1157_; lean_object* v___x_1158_; lean_object* v___x_1159_; lean_object* v___x_1160_; lean_object* v___x_1161_; lean_object* v___y_1163_; lean_object* v___x_1173_; 
v___x_1143_ = 0;
v___x_1144_ = 1;
v___x_1145_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__2_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__2_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__2_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_);
v___x_1146_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__4_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__4_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__4_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_);
v___x_1147_ = lean_unsigned_to_nat(32u);
v___x_1148_ = lean_mk_empty_array_with_capacity(v___x_1147_);
v___x_1149_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__3, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__3_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__3);
v___x_1150_ = ((size_t)5ULL);
lean_inc_n(v___x_1135_, 7);
v___x_1151_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1151_, 0, v___x_1149_);
lean_ctor_set(v___x_1151_, 1, v___x_1148_);
lean_ctor_set(v___x_1151_, 2, v___x_1135_);
lean_ctor_set(v___x_1151_, 3, v___x_1135_);
lean_ctor_set_usize(v___x_1151_, 4, v___x_1150_);
v___x_1152_ = lean_box(1);
lean_inc_ref(v___x_1151_);
v___x_1153_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1153_, 0, v___x_1146_);
lean_ctor_set(v___x_1153_, 1, v___x_1151_);
lean_ctor_set(v___x_1153_, 2, v___x_1152_);
v___x_1154_ = lean_mk_empty_array_with_capacity(v___x_1135_);
v___x_1155_ = lean_box(0);
lean_inc_ref(v___x_1154_);
lean_inc_ref(v___x_1153_);
lean_inc_n(v___x_1136_, 2);
v___x_1156_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_1156_, 0, v___x_1145_);
lean_ctor_set(v___x_1156_, 1, v___x_1136_);
lean_ctor_set(v___x_1156_, 2, v___x_1153_);
lean_ctor_set(v___x_1156_, 3, v___x_1154_);
lean_ctor_set(v___x_1156_, 4, v___x_1155_);
lean_ctor_set(v___x_1156_, 5, v___x_1135_);
lean_ctor_set(v___x_1156_, 6, v___x_1155_);
lean_ctor_set_uint8(v___x_1156_, sizeof(void*)*7, v___x_1143_);
lean_ctor_set_uint8(v___x_1156_, sizeof(void*)*7 + 1, v___x_1143_);
lean_ctor_set_uint8(v___x_1156_, sizeof(void*)*7 + 2, v___x_1143_);
lean_ctor_set_uint8(v___x_1156_, sizeof(void*)*7 + 3, v___x_1144_);
v___x_1157_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1157_, 0, v___x_1135_);
lean_ctor_set(v___x_1157_, 1, v___x_1135_);
lean_ctor_set(v___x_1157_, 2, v___x_1135_);
lean_ctor_set(v___x_1157_, 3, v___x_1135_);
lean_ctor_set(v___x_1157_, 4, v___x_1146_);
lean_ctor_set(v___x_1157_, 5, v___x_1146_);
lean_ctor_set(v___x_1157_, 6, v___x_1146_);
lean_ctor_set(v___x_1157_, 7, v___x_1146_);
lean_ctor_set(v___x_1157_, 8, v___x_1146_);
lean_ctor_set(v___x_1157_, 9, v___x_1146_);
v___x_1158_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__5_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__5_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__5_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_);
v___x_1159_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__6_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__6_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__6_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_);
v___x_1160_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1160_, 0, v___x_1157_);
lean_ctor_set(v___x_1160_, 1, v___x_1158_);
lean_ctor_set(v___x_1160_, 2, v___x_1136_);
lean_ctor_set(v___x_1160_, 3, v___x_1151_);
lean_ctor_set(v___x_1160_, 4, v___x_1159_);
v___x_1161_ = lean_st_mk_ref(v___x_1160_);
lean_inc(v_decl_1137_);
v___x_1173_ = lp_batteries_Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0(v_decl_1137_, v___x_1156_, v___x_1161_, v___y_1140_, v___y_1141_);
if (lean_obj_tag(v___x_1173_) == 0)
{
lean_object* v_a_1174_; lean_object* v___x_1175_; uint8_t v___x_1176_; lean_object* v___x_1177_; lean_object* v___x_1178_; lean_object* v___x_1179_; 
v_a_1174_ = lean_ctor_get(v___x_1173_, 0);
lean_inc(v_a_1174_);
lean_dec_ref_known(v___x_1173_, 1);
v___x_1175_ = l_Lean_ConstantInfo_type(v_a_1174_);
lean_dec(v_a_1174_);
v___x_1176_ = 0;
v___x_1177_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__7_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__7_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__7_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_);
v___x_1178_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_1178_, 0, v___x_1177_);
lean_ctor_set(v___x_1178_, 1, v___x_1136_);
lean_ctor_set(v___x_1178_, 2, v___x_1153_);
lean_ctor_set(v___x_1178_, 3, v___x_1154_);
lean_ctor_set(v___x_1178_, 4, v___x_1155_);
lean_ctor_set(v___x_1178_, 5, v___x_1135_);
lean_ctor_set(v___x_1178_, 6, v___x_1155_);
lean_ctor_set_uint8(v___x_1178_, sizeof(void*)*7, v___x_1143_);
lean_ctor_set_uint8(v___x_1178_, sizeof(void*)*7 + 1, v___x_1143_);
lean_ctor_set_uint8(v___x_1178_, sizeof(void*)*7 + 2, v___x_1143_);
lean_ctor_set_uint8(v___x_1178_, sizeof(void*)*7 + 3, v___x_1144_);
lean_inc_ref(v___x_1175_);
v___x_1179_ = l_Lean_Meta_forallMetaTelescopeReducing(v___x_1175_, v___x_1155_, v___x_1176_, v___x_1178_, v___x_1161_, v___y_1140_, v___y_1141_);
if (lean_obj_tag(v___x_1179_) == 0)
{
lean_object* v_a_1180_; lean_object* v_snd_1181_; lean_object* v_fst_1182_; lean_object* v___x_1184_; uint8_t v_isShared_1185_; uint8_t v_isSharedCheck_1258_; 
v_a_1180_ = lean_ctor_get(v___x_1179_, 0);
lean_inc(v_a_1180_);
lean_dec_ref_known(v___x_1179_, 1);
v_snd_1181_ = lean_ctor_get(v_a_1180_, 1);
v_fst_1182_ = lean_ctor_get(v_a_1180_, 0);
v_isSharedCheck_1258_ = !lean_is_exclusive(v_a_1180_);
if (v_isSharedCheck_1258_ == 0)
{
v___x_1184_ = v_a_1180_;
v_isShared_1185_ = v_isSharedCheck_1258_;
goto v_resetjp_1183_;
}
else
{
lean_inc(v_snd_1181_);
lean_inc(v_fst_1182_);
lean_dec(v_a_1180_);
v___x_1184_ = lean_box(0);
v_isShared_1185_ = v_isSharedCheck_1258_;
goto v_resetjp_1183_;
}
v_resetjp_1183_:
{
lean_object* v_snd_1186_; lean_object* v___x_1188_; uint8_t v_isShared_1189_; uint8_t v_isSharedCheck_1256_; 
v_snd_1186_ = lean_ctor_get(v_snd_1181_, 1);
v_isSharedCheck_1256_ = !lean_is_exclusive(v_snd_1181_);
if (v_isSharedCheck_1256_ == 0)
{
lean_object* v_unused_1257_; 
v_unused_1257_ = lean_ctor_get(v_snd_1181_, 0);
lean_dec(v_unused_1257_);
v___x_1188_ = v_snd_1181_;
v_isShared_1189_ = v_isSharedCheck_1256_;
goto v_resetjp_1187_;
}
else
{
lean_inc(v_snd_1186_);
lean_dec(v_snd_1181_);
v___x_1188_ = lean_box(0);
v_isShared_1189_ = v_isSharedCheck_1256_;
goto v_resetjp_1187_;
}
v_resetjp_1187_:
{
lean_object* v___x_1190_; lean_object* v___x_1191_; lean_object* v___x_1193_; 
v___x_1190_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__9_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__9_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__9_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_);
v___x_1191_ = l_Lean_indentExpr(v___x_1175_);
if (v_isShared_1185_ == 0)
{
lean_ctor_set_tag(v___x_1184_, 7);
lean_ctor_set(v___x_1184_, 1, v___x_1191_);
lean_ctor_set(v___x_1184_, 0, v___x_1190_);
v___x_1193_ = v___x_1184_;
goto v_reusejp_1192_;
}
else
{
lean_object* v_reuseFailAlloc_1255_; 
v_reuseFailAlloc_1255_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1255_, 0, v___x_1190_);
lean_ctor_set(v_reuseFailAlloc_1255_, 1, v___x_1191_);
v___x_1193_ = v_reuseFailAlloc_1255_;
goto v_reusejp_1192_;
}
v_reusejp_1192_:
{
lean_object* v___x_1194_; lean_object* v___x_1195_; lean_object* v___x_1196_; lean_object* v___x_1197_; 
v___x_1194_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__11_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__11_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0___closed__11_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_);
v___x_1195_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1195_, 0, v___x_1193_);
lean_ctor_set(v___x_1195_, 1, v___x_1194_);
lean_inc(v_snd_1186_);
v___x_1196_ = l_Lean_indentExpr(v_snd_1186_);
v___x_1197_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1197_, 0, v___x_1195_);
lean_ctor_set(v___x_1197_, 1, v___x_1196_);
if (lean_obj_tag(v_snd_1186_) == 5)
{
lean_object* v_fn_1198_; 
v_fn_1198_ = lean_ctor_get(v_snd_1186_, 0);
lean_inc_ref(v_fn_1198_);
lean_dec_ref_known(v_snd_1186_, 2);
if (lean_obj_tag(v_fn_1198_) == 5)
{
lean_object* v_fn_1199_; lean_object* v___x_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; uint8_t v___x_1203_; 
v_fn_1199_ = lean_ctor_get(v_fn_1198_, 0);
lean_inc_ref(v_fn_1199_);
lean_dec_ref_known(v_fn_1198_, 2);
v___x_1200_ = lean_array_get_size(v_fst_1182_);
v___x_1201_ = lean_unsigned_to_nat(1u);
v___x_1202_ = lean_nat_sub(v___x_1200_, v___x_1201_);
v___x_1203_ = lean_nat_dec_lt(v___x_1202_, v___x_1200_);
if (v___x_1203_ == 0)
{
lean_object* v___x_1204_; 
lean_dec(v___x_1202_);
lean_dec_ref(v_fn_1199_);
lean_del_object(v___x_1188_);
lean_dec(v_fst_1182_);
lean_dec_ref_known(v___x_1178_, 7);
lean_dec(v_decl_1137_);
v___x_1204_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1___redArg(v___x_1197_, v___x_1156_, v___x_1161_, v___y_1140_, v___y_1141_);
lean_dec_ref_known(v___x_1156_, 7);
v___y_1163_ = v___x_1204_;
goto v___jp_1162_;
}
else
{
lean_object* v___x_1205_; lean_object* v___x_1206_; lean_object* v___x_1207_; lean_object* v___x_1208_; uint8_t v___x_1209_; 
v___x_1205_ = lean_array_fget(v_fst_1182_, v___x_1202_);
lean_dec(v___x_1202_);
v___x_1206_ = lean_array_pop(v_fst_1182_);
v___x_1207_ = lean_array_get_size(v___x_1206_);
v___x_1208_ = lean_nat_sub(v___x_1207_, v___x_1201_);
v___x_1209_ = lean_nat_dec_lt(v___x_1208_, v___x_1207_);
if (v___x_1209_ == 0)
{
lean_object* v___x_1210_; 
lean_dec(v___x_1208_);
lean_dec_ref(v___x_1206_);
lean_dec(v___x_1205_);
lean_dec_ref(v_fn_1199_);
lean_del_object(v___x_1188_);
lean_dec_ref_known(v___x_1178_, 7);
lean_dec(v_decl_1137_);
v___x_1210_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1___redArg(v___x_1197_, v___x_1156_, v___x_1161_, v___y_1140_, v___y_1141_);
lean_dec_ref_known(v___x_1156_, 7);
v___y_1163_ = v___x_1210_;
goto v___jp_1162_;
}
else
{
lean_object* v___x_1211_; 
lean_inc(v___y_1141_);
lean_inc_ref(v___y_1140_);
lean_inc(v___x_1161_);
lean_inc_ref(v___x_1156_);
v___x_1211_ = lean_infer_type(v___x_1205_, v___x_1156_, v___x_1161_, v___y_1140_, v___y_1141_);
if (lean_obj_tag(v___x_1211_) == 0)
{
lean_object* v_a_1212_; 
v_a_1212_ = lean_ctor_get(v___x_1211_, 0);
lean_inc(v_a_1212_);
lean_dec_ref_known(v___x_1211_, 1);
if (lean_obj_tag(v_a_1212_) == 5)
{
lean_object* v_fn_1213_; 
v_fn_1213_ = lean_ctor_get(v_a_1212_, 0);
lean_inc_ref(v_fn_1213_);
lean_dec_ref_known(v_a_1212_, 2);
if (lean_obj_tag(v_fn_1213_) == 5)
{
lean_object* v___x_1214_; lean_object* v___x_1215_; 
lean_dec_ref_known(v_fn_1213_, 2);
v___x_1214_ = lean_array_fget(v___x_1206_, v___x_1208_);
lean_dec(v___x_1208_);
lean_dec_ref(v___x_1206_);
lean_inc(v___y_1141_);
lean_inc_ref(v___y_1140_);
lean_inc(v___x_1161_);
lean_inc_ref(v___x_1156_);
v___x_1215_ = lean_infer_type(v___x_1214_, v___x_1156_, v___x_1161_, v___y_1140_, v___y_1141_);
if (lean_obj_tag(v___x_1215_) == 0)
{
lean_object* v_a_1216_; 
v_a_1216_ = lean_ctor_get(v___x_1215_, 0);
lean_inc(v_a_1216_);
lean_dec_ref_known(v___x_1215_, 1);
if (lean_obj_tag(v_a_1216_) == 5)
{
lean_object* v_fn_1217_; 
v_fn_1217_ = lean_ctor_get(v_a_1216_, 0);
lean_inc_ref(v_fn_1217_);
lean_dec_ref_known(v_a_1216_, 2);
if (lean_obj_tag(v_fn_1217_) == 5)
{
lean_object* v___x_1218_; 
lean_dec_ref_known(v_fn_1217_, 2);
lean_dec_ref_known(v___x_1197_, 2);
lean_dec_ref_known(v___x_1156_, 7);
v___x_1218_ = l_Lean_Meta_DiscrTree_mkPath(v_fn_1199_, v___x_1143_, v___x_1178_, v___x_1161_, v___y_1140_, v___y_1141_);
lean_dec_ref_known(v___x_1178_, 7);
if (lean_obj_tag(v___x_1218_) == 0)
{
lean_object* v_a_1219_; lean_object* v___x_1220_; lean_object* v___x_1222_; 
v_a_1219_ = lean_ctor_get(v___x_1218_, 0);
lean_inc(v_a_1219_);
lean_dec_ref_known(v___x_1218_, 1);
v___x_1220_ = lp_batteries_Batteries_Tactic_transExt;
if (v_isShared_1189_ == 0)
{
lean_ctor_set(v___x_1188_, 1, v_a_1219_);
lean_ctor_set(v___x_1188_, 0, v_decl_1137_);
v___x_1222_ = v___x_1188_;
goto v_reusejp_1221_;
}
else
{
lean_object* v_reuseFailAlloc_1224_; 
v_reuseFailAlloc_1224_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1224_, 0, v_decl_1137_);
lean_ctor_set(v_reuseFailAlloc_1224_, 1, v_a_1219_);
v___x_1222_ = v_reuseFailAlloc_1224_;
goto v_reusejp_1221_;
}
v_reusejp_1221_:
{
lean_object* v___x_1223_; 
v___x_1223_ = lp_batteries_Lean_ScopedEnvExtension_add___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__2___redArg(v___x_1220_, v___x_1222_, v_kind_1139_, v___x_1161_, v___y_1140_, v___y_1141_);
v___y_1163_ = v___x_1223_;
goto v___jp_1162_;
}
}
else
{
lean_object* v_a_1225_; lean_object* v___x_1227_; uint8_t v_isShared_1228_; uint8_t v_isSharedCheck_1232_; 
lean_del_object(v___x_1188_);
lean_dec(v___x_1161_);
lean_dec(v_decl_1137_);
v_a_1225_ = lean_ctor_get(v___x_1218_, 0);
v_isSharedCheck_1232_ = !lean_is_exclusive(v___x_1218_);
if (v_isSharedCheck_1232_ == 0)
{
v___x_1227_ = v___x_1218_;
v_isShared_1228_ = v_isSharedCheck_1232_;
goto v_resetjp_1226_;
}
else
{
lean_inc(v_a_1225_);
lean_dec(v___x_1218_);
v___x_1227_ = lean_box(0);
v_isShared_1228_ = v_isSharedCheck_1232_;
goto v_resetjp_1226_;
}
v_resetjp_1226_:
{
lean_object* v___x_1230_; 
if (v_isShared_1228_ == 0)
{
v___x_1230_ = v___x_1227_;
goto v_reusejp_1229_;
}
else
{
lean_object* v_reuseFailAlloc_1231_; 
v_reuseFailAlloc_1231_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1231_, 0, v_a_1225_);
v___x_1230_ = v_reuseFailAlloc_1231_;
goto v_reusejp_1229_;
}
v_reusejp_1229_:
{
return v___x_1230_;
}
}
}
}
else
{
lean_object* v___x_1233_; 
lean_dec_ref(v_fn_1217_);
lean_dec_ref(v_fn_1199_);
lean_del_object(v___x_1188_);
lean_dec_ref_known(v___x_1178_, 7);
lean_dec(v_decl_1137_);
v___x_1233_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1___redArg(v___x_1197_, v___x_1156_, v___x_1161_, v___y_1140_, v___y_1141_);
lean_dec_ref_known(v___x_1156_, 7);
v___y_1163_ = v___x_1233_;
goto v___jp_1162_;
}
}
else
{
lean_object* v___x_1234_; 
lean_dec(v_a_1216_);
lean_dec_ref(v_fn_1199_);
lean_del_object(v___x_1188_);
lean_dec_ref_known(v___x_1178_, 7);
lean_dec(v_decl_1137_);
v___x_1234_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1___redArg(v___x_1197_, v___x_1156_, v___x_1161_, v___y_1140_, v___y_1141_);
lean_dec_ref_known(v___x_1156_, 7);
v___y_1163_ = v___x_1234_;
goto v___jp_1162_;
}
}
else
{
lean_object* v_a_1235_; lean_object* v___x_1237_; uint8_t v_isShared_1238_; uint8_t v_isSharedCheck_1242_; 
lean_dec_ref(v_fn_1199_);
lean_dec_ref_known(v___x_1197_, 2);
lean_del_object(v___x_1188_);
lean_dec_ref_known(v___x_1178_, 7);
lean_dec(v___x_1161_);
lean_dec_ref_known(v___x_1156_, 7);
lean_dec(v_decl_1137_);
v_a_1235_ = lean_ctor_get(v___x_1215_, 0);
v_isSharedCheck_1242_ = !lean_is_exclusive(v___x_1215_);
if (v_isSharedCheck_1242_ == 0)
{
v___x_1237_ = v___x_1215_;
v_isShared_1238_ = v_isSharedCheck_1242_;
goto v_resetjp_1236_;
}
else
{
lean_inc(v_a_1235_);
lean_dec(v___x_1215_);
v___x_1237_ = lean_box(0);
v_isShared_1238_ = v_isSharedCheck_1242_;
goto v_resetjp_1236_;
}
v_resetjp_1236_:
{
lean_object* v___x_1240_; 
if (v_isShared_1238_ == 0)
{
v___x_1240_ = v___x_1237_;
goto v_reusejp_1239_;
}
else
{
lean_object* v_reuseFailAlloc_1241_; 
v_reuseFailAlloc_1241_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1241_, 0, v_a_1235_);
v___x_1240_ = v_reuseFailAlloc_1241_;
goto v_reusejp_1239_;
}
v_reusejp_1239_:
{
return v___x_1240_;
}
}
}
}
else
{
lean_object* v___x_1243_; 
lean_dec_ref(v_fn_1213_);
lean_dec(v___x_1208_);
lean_dec_ref(v___x_1206_);
lean_dec_ref(v_fn_1199_);
lean_del_object(v___x_1188_);
lean_dec_ref_known(v___x_1178_, 7);
lean_dec(v_decl_1137_);
v___x_1243_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1___redArg(v___x_1197_, v___x_1156_, v___x_1161_, v___y_1140_, v___y_1141_);
lean_dec_ref_known(v___x_1156_, 7);
v___y_1163_ = v___x_1243_;
goto v___jp_1162_;
}
}
else
{
lean_object* v___x_1244_; 
lean_dec(v_a_1212_);
lean_dec(v___x_1208_);
lean_dec_ref(v___x_1206_);
lean_dec_ref(v_fn_1199_);
lean_del_object(v___x_1188_);
lean_dec_ref_known(v___x_1178_, 7);
lean_dec(v_decl_1137_);
v___x_1244_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1___redArg(v___x_1197_, v___x_1156_, v___x_1161_, v___y_1140_, v___y_1141_);
lean_dec_ref_known(v___x_1156_, 7);
v___y_1163_ = v___x_1244_;
goto v___jp_1162_;
}
}
else
{
lean_object* v_a_1245_; lean_object* v___x_1247_; uint8_t v_isShared_1248_; uint8_t v_isSharedCheck_1252_; 
lean_dec(v___x_1208_);
lean_dec_ref(v___x_1206_);
lean_dec_ref(v_fn_1199_);
lean_dec_ref_known(v___x_1197_, 2);
lean_del_object(v___x_1188_);
lean_dec_ref_known(v___x_1178_, 7);
lean_dec(v___x_1161_);
lean_dec_ref_known(v___x_1156_, 7);
lean_dec(v_decl_1137_);
v_a_1245_ = lean_ctor_get(v___x_1211_, 0);
v_isSharedCheck_1252_ = !lean_is_exclusive(v___x_1211_);
if (v_isSharedCheck_1252_ == 0)
{
v___x_1247_ = v___x_1211_;
v_isShared_1248_ = v_isSharedCheck_1252_;
goto v_resetjp_1246_;
}
else
{
lean_inc(v_a_1245_);
lean_dec(v___x_1211_);
v___x_1247_ = lean_box(0);
v_isShared_1248_ = v_isSharedCheck_1252_;
goto v_resetjp_1246_;
}
v_resetjp_1246_:
{
lean_object* v___x_1250_; 
if (v_isShared_1248_ == 0)
{
v___x_1250_ = v___x_1247_;
goto v_reusejp_1249_;
}
else
{
lean_object* v_reuseFailAlloc_1251_; 
v_reuseFailAlloc_1251_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1251_, 0, v_a_1245_);
v___x_1250_ = v_reuseFailAlloc_1251_;
goto v_reusejp_1249_;
}
v_reusejp_1249_:
{
return v___x_1250_;
}
}
}
}
}
}
else
{
lean_object* v___x_1253_; 
lean_dec_ref(v_fn_1198_);
lean_del_object(v___x_1188_);
lean_dec(v_fst_1182_);
lean_dec_ref_known(v___x_1178_, 7);
lean_dec(v_decl_1137_);
v___x_1253_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1___redArg(v___x_1197_, v___x_1156_, v___x_1161_, v___y_1140_, v___y_1141_);
lean_dec_ref_known(v___x_1156_, 7);
v___y_1163_ = v___x_1253_;
goto v___jp_1162_;
}
}
else
{
lean_object* v___x_1254_; 
lean_del_object(v___x_1188_);
lean_dec(v_snd_1186_);
lean_dec(v_fst_1182_);
lean_dec_ref_known(v___x_1178_, 7);
lean_dec(v_decl_1137_);
v___x_1254_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1___redArg(v___x_1197_, v___x_1156_, v___x_1161_, v___y_1140_, v___y_1141_);
lean_dec_ref_known(v___x_1156_, 7);
v___y_1163_ = v___x_1254_;
goto v___jp_1162_;
}
}
}
}
}
else
{
lean_object* v_a_1259_; lean_object* v___x_1261_; uint8_t v_isShared_1262_; uint8_t v_isSharedCheck_1266_; 
lean_dec_ref_known(v___x_1178_, 7);
lean_dec_ref(v___x_1175_);
lean_dec(v___x_1161_);
lean_dec_ref_known(v___x_1156_, 7);
lean_dec(v_decl_1137_);
v_a_1259_ = lean_ctor_get(v___x_1179_, 0);
v_isSharedCheck_1266_ = !lean_is_exclusive(v___x_1179_);
if (v_isSharedCheck_1266_ == 0)
{
v___x_1261_ = v___x_1179_;
v_isShared_1262_ = v_isSharedCheck_1266_;
goto v_resetjp_1260_;
}
else
{
lean_inc(v_a_1259_);
lean_dec(v___x_1179_);
v___x_1261_ = lean_box(0);
v_isShared_1262_ = v_isSharedCheck_1266_;
goto v_resetjp_1260_;
}
v_resetjp_1260_:
{
lean_object* v___x_1264_; 
if (v_isShared_1262_ == 0)
{
v___x_1264_ = v___x_1261_;
goto v_reusejp_1263_;
}
else
{
lean_object* v_reuseFailAlloc_1265_; 
v_reuseFailAlloc_1265_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1265_, 0, v_a_1259_);
v___x_1264_ = v_reuseFailAlloc_1265_;
goto v_reusejp_1263_;
}
v_reusejp_1263_:
{
return v___x_1264_;
}
}
}
}
else
{
lean_object* v_a_1267_; lean_object* v___x_1269_; uint8_t v_isShared_1270_; uint8_t v_isSharedCheck_1274_; 
lean_dec(v___x_1161_);
lean_dec_ref_known(v___x_1156_, 7);
lean_dec_ref(v___x_1154_);
lean_dec_ref_known(v___x_1153_, 3);
lean_dec(v_decl_1137_);
lean_dec(v___x_1136_);
lean_dec(v___x_1135_);
v_a_1267_ = lean_ctor_get(v___x_1173_, 0);
v_isSharedCheck_1274_ = !lean_is_exclusive(v___x_1173_);
if (v_isSharedCheck_1274_ == 0)
{
v___x_1269_ = v___x_1173_;
v_isShared_1270_ = v_isSharedCheck_1274_;
goto v_resetjp_1268_;
}
else
{
lean_inc(v_a_1267_);
lean_dec(v___x_1173_);
v___x_1269_ = lean_box(0);
v_isShared_1270_ = v_isSharedCheck_1274_;
goto v_resetjp_1268_;
}
v_resetjp_1268_:
{
lean_object* v___x_1272_; 
if (v_isShared_1270_ == 0)
{
v___x_1272_ = v___x_1269_;
goto v_reusejp_1271_;
}
else
{
lean_object* v_reuseFailAlloc_1273_; 
v_reuseFailAlloc_1273_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1273_, 0, v_a_1267_);
v___x_1272_ = v_reuseFailAlloc_1273_;
goto v_reusejp_1271_;
}
v_reusejp_1271_:
{
return v___x_1272_;
}
}
}
v___jp_1162_:
{
if (lean_obj_tag(v___y_1163_) == 0)
{
lean_object* v_a_1164_; lean_object* v___x_1166_; uint8_t v_isShared_1167_; uint8_t v_isSharedCheck_1172_; 
v_a_1164_ = lean_ctor_get(v___y_1163_, 0);
v_isSharedCheck_1172_ = !lean_is_exclusive(v___y_1163_);
if (v_isSharedCheck_1172_ == 0)
{
v___x_1166_ = v___y_1163_;
v_isShared_1167_ = v_isSharedCheck_1172_;
goto v_resetjp_1165_;
}
else
{
lean_inc(v_a_1164_);
lean_dec(v___y_1163_);
v___x_1166_ = lean_box(0);
v_isShared_1167_ = v_isSharedCheck_1172_;
goto v_resetjp_1165_;
}
v_resetjp_1165_:
{
lean_object* v___x_1168_; lean_object* v___x_1170_; 
v___x_1168_ = lean_st_ref_get(v___x_1161_);
lean_dec(v___x_1161_);
lean_dec(v___x_1168_);
if (v_isShared_1167_ == 0)
{
v___x_1170_ = v___x_1166_;
goto v_reusejp_1169_;
}
else
{
lean_object* v_reuseFailAlloc_1171_; 
v_reuseFailAlloc_1171_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1171_, 0, v_a_1164_);
v___x_1170_ = v_reuseFailAlloc_1171_;
goto v_reusejp_1169_;
}
v_reusejp_1169_:
{
return v___x_1170_;
}
}
}
else
{
lean_dec(v___x_1161_);
return v___y_1163_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2____boxed(lean_object* v___x_1275_, lean_object* v___x_1276_, lean_object* v_decl_1277_, lean_object* v_x_1278_, lean_object* v_kind_1279_, lean_object* v___y_1280_, lean_object* v___y_1281_, lean_object* v___y_1282_){
_start:
{
uint8_t v_kind_boxed_1283_; lean_object* v_res_1284_; 
v_kind_boxed_1283_ = lean_unbox(v_kind_1279_);
v_res_1284_ = lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__0_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_(v___x_1275_, v___x_1276_, v_decl_1277_, v_x_1278_, v_kind_boxed_1283_, v___y_1280_, v___y_1281_);
lean_dec(v___y_1281_);
lean_dec_ref(v___y_1280_);
lean_dec(v_x_1278_);
return v_res_1284_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__3_spec__5(lean_object* v_msgData_1285_, lean_object* v___y_1286_, lean_object* v___y_1287_){
_start:
{
lean_object* v___x_1289_; lean_object* v_env_1290_; lean_object* v_options_1291_; lean_object* v___x_1292_; lean_object* v___x_1293_; lean_object* v___x_1294_; lean_object* v___x_1295_; lean_object* v___x_1296_; lean_object* v___x_1297_; lean_object* v___x_1298_; 
v___x_1289_ = lean_st_ref_get(v___y_1287_);
v_env_1290_ = lean_ctor_get(v___x_1289_, 0);
lean_inc_ref(v_env_1290_);
lean_dec(v___x_1289_);
v_options_1291_ = lean_ctor_get(v___y_1286_, 2);
v___x_1292_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__2, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__2_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__2);
v___x_1293_ = lean_unsigned_to_nat(32u);
v___x_1294_ = lean_mk_empty_array_with_capacity(v___x_1293_);
lean_dec_ref(v___x_1294_);
v___x_1295_ = lean_obj_once(&lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__5, &lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__5_once, _init_lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg___closed__5);
lean_inc_ref(v_options_1291_);
v___x_1296_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1296_, 0, v_env_1290_);
lean_ctor_set(v___x_1296_, 1, v___x_1292_);
lean_ctor_set(v___x_1296_, 2, v___x_1295_);
lean_ctor_set(v___x_1296_, 3, v_options_1291_);
v___x_1297_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1297_, 0, v___x_1296_);
lean_ctor_set(v___x_1297_, 1, v_msgData_1285_);
v___x_1298_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1298_, 0, v___x_1297_);
return v___x_1298_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__3_spec__5___boxed(lean_object* v_msgData_1299_, lean_object* v___y_1300_, lean_object* v___y_1301_, lean_object* v___y_1302_){
_start:
{
lean_object* v_res_1303_; 
v_res_1303_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__3_spec__5(v_msgData_1299_, v___y_1300_, v___y_1301_);
lean_dec(v___y_1301_);
lean_dec_ref(v___y_1300_);
return v_res_1303_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__3___redArg(lean_object* v_msg_1304_, lean_object* v___y_1305_, lean_object* v___y_1306_){
_start:
{
lean_object* v_ref_1308_; lean_object* v___x_1309_; lean_object* v_a_1310_; lean_object* v___x_1312_; uint8_t v_isShared_1313_; uint8_t v_isSharedCheck_1318_; 
v_ref_1308_ = lean_ctor_get(v___y_1305_, 5);
v___x_1309_ = lp_batteries_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__3_spec__5(v_msg_1304_, v___y_1305_, v___y_1306_);
v_a_1310_ = lean_ctor_get(v___x_1309_, 0);
v_isSharedCheck_1318_ = !lean_is_exclusive(v___x_1309_);
if (v_isSharedCheck_1318_ == 0)
{
v___x_1312_ = v___x_1309_;
v_isShared_1313_ = v_isSharedCheck_1318_;
goto v_resetjp_1311_;
}
else
{
lean_inc(v_a_1310_);
lean_dec(v___x_1309_);
v___x_1312_ = lean_box(0);
v_isShared_1313_ = v_isSharedCheck_1318_;
goto v_resetjp_1311_;
}
v_resetjp_1311_:
{
lean_object* v___x_1314_; lean_object* v___x_1316_; 
lean_inc(v_ref_1308_);
v___x_1314_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1314_, 0, v_ref_1308_);
lean_ctor_set(v___x_1314_, 1, v_a_1310_);
if (v_isShared_1313_ == 0)
{
lean_ctor_set_tag(v___x_1312_, 1);
lean_ctor_set(v___x_1312_, 0, v___x_1314_);
v___x_1316_ = v___x_1312_;
goto v_reusejp_1315_;
}
else
{
lean_object* v_reuseFailAlloc_1317_; 
v_reuseFailAlloc_1317_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1317_, 0, v___x_1314_);
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
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__3___redArg___boxed(lean_object* v_msg_1319_, lean_object* v___y_1320_, lean_object* v___y_1321_, lean_object* v___y_1322_){
_start:
{
lean_object* v_res_1323_; 
v_res_1323_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__3___redArg(v_msg_1319_, v___y_1320_, v___y_1321_);
lean_dec(v___y_1321_);
lean_dec_ref(v___y_1320_);
return v_res_1323_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1___closed__1_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1325_; lean_object* v___x_1326_; 
v___x_1325_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1___closed__0_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_));
v___x_1326_ = l_Lean_stringToMessageData(v___x_1325_);
return v___x_1326_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1___closed__3_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1328_; lean_object* v___x_1329_; 
v___x_1328_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1___closed__2_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_));
v___x_1329_ = l_Lean_stringToMessageData(v___x_1328_);
return v___x_1329_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_(lean_object* v___x_1330_, lean_object* v_decl_1331_, lean_object* v___y_1332_, lean_object* v___y_1333_){
_start:
{
lean_object* v___x_1335_; lean_object* v___x_1336_; lean_object* v___x_1337_; lean_object* v___x_1338_; lean_object* v___x_1339_; lean_object* v___x_1340_; 
v___x_1335_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1___closed__1_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1___closed__1_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1___closed__1_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_);
v___x_1336_ = l_Lean_MessageData_ofName(v___x_1330_);
v___x_1337_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1337_, 0, v___x_1335_);
lean_ctor_set(v___x_1337_, 1, v___x_1336_);
v___x_1338_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1___closed__3_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1___closed__3_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1___closed__3_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_);
v___x_1339_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1339_, 0, v___x_1337_);
lean_ctor_set(v___x_1339_, 1, v___x_1338_);
v___x_1340_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__3___redArg(v___x_1339_, v___y_1332_, v___y_1333_);
return v___x_1340_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2____boxed(lean_object* v___x_1341_, lean_object* v_decl_1342_, lean_object* v___y_1343_, lean_object* v___y_1344_, lean_object* v___y_1345_){
_start:
{
lean_object* v_res_1346_; 
v_res_1346_ = lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___lam__1_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_(v___x_1341_, v_decl_1342_, v___y_1343_, v___y_1344_);
lean_dec(v___y_1344_);
lean_dec_ref(v___y_1343_);
lean_dec(v_decl_1342_);
return v_res_1346_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__1_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1350_; lean_object* v___x_1351_; lean_object* v___x_1352_; 
v___x_1350_ = lean_unsigned_to_nat(2247956323u);
v___x_1351_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__19_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_));
v___x_1352_ = l_Lean_Name_num___override(v___x_1351_, v___x_1350_);
return v___x_1352_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__2_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1353_; lean_object* v___x_1354_; lean_object* v___x_1355_; 
v___x_1353_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__21_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_));
v___x_1354_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__1_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__1_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__1_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_);
v___x_1355_ = l_Lean_Name_str___override(v___x_1354_, v___x_1353_);
return v___x_1355_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__3_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1356_; lean_object* v___x_1357_; lean_object* v___x_1358_; 
v___x_1356_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__23_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_));
v___x_1357_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__2_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__2_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__2_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_);
v___x_1358_ = l_Lean_Name_str___override(v___x_1357_, v___x_1356_);
return v___x_1358_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__4_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1359_; lean_object* v___x_1360_; lean_object* v___x_1361_; 
v___x_1359_ = lean_unsigned_to_nat(2u);
v___x_1360_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__3_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__3_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__3_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_);
v___x_1361_ = l_Lean_Name_num___override(v___x_1360_, v___x_1359_);
return v___x_1361_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__8_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_(void){
_start:
{
uint8_t v___x_1367_; lean_object* v___x_1368_; lean_object* v___x_1369_; lean_object* v___x_1370_; lean_object* v___x_1371_; 
v___x_1367_ = 0;
v___x_1368_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__7_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_));
v___x_1369_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__5_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_));
v___x_1370_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__4_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__4_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__4_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_);
v___x_1371_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_1371_, 0, v___x_1370_);
lean_ctor_set(v___x_1371_, 1, v___x_1369_);
lean_ctor_set(v___x_1371_, 2, v___x_1368_);
lean_ctor_set_uint8(v___x_1371_, sizeof(void*)*3, v___x_1367_);
return v___x_1371_;
}
}
static lean_object* _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__9_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___f_1372_; lean_object* v___f_1373_; lean_object* v___x_1374_; lean_object* v___x_1375_; 
v___f_1372_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__6_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_));
v___f_1373_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__0_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_));
v___x_1374_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__8_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__8_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__8_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_);
v___x_1375_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1375_, 0, v___x_1374_);
lean_ctor_set(v___x_1375_, 1, v___f_1373_);
lean_ctor_set(v___x_1375_, 2, v___f_1372_);
return v___x_1375_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_1377_; lean_object* v___x_1378_; 
v___x_1377_ = lean_obj_once(&lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__9_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_, &lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__9_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__once, _init_lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__9_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_);
v___x_1378_ = l_Lean_registerBuiltinAttribute(v___x_1377_);
return v___x_1378_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2____boxed(lean_object* v_a_1379_){
_start:
{
lean_object* v_res_1380_; 
v_res_1380_ = lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_();
return v_res_1380_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1(lean_object* v_00_u03b1_1381_, lean_object* v_msg_1382_, lean_object* v___y_1383_, lean_object* v___y_1384_, lean_object* v___y_1385_, lean_object* v___y_1386_){
_start:
{
lean_object* v___x_1388_; 
v___x_1388_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1___redArg(v_msg_1382_, v___y_1383_, v___y_1384_, v___y_1385_, v___y_1386_);
return v___x_1388_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1___boxed(lean_object* v_00_u03b1_1389_, lean_object* v_msg_1390_, lean_object* v___y_1391_, lean_object* v___y_1392_, lean_object* v___y_1393_, lean_object* v___y_1394_, lean_object* v___y_1395_){
_start:
{
lean_object* v_res_1396_; 
v_res_1396_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1(v_00_u03b1_1389_, v_msg_1390_, v___y_1391_, v___y_1392_, v___y_1393_, v___y_1394_);
lean_dec(v___y_1394_);
lean_dec_ref(v___y_1393_);
lean_dec(v___y_1392_);
lean_dec_ref(v___y_1391_);
return v_res_1396_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__3(lean_object* v_00_u03b1_1397_, lean_object* v_msg_1398_, lean_object* v___y_1399_, lean_object* v___y_1400_){
_start:
{
lean_object* v___x_1402_; 
v___x_1402_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__3___redArg(v_msg_1398_, v___y_1399_, v___y_1400_);
return v___x_1402_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__3___boxed(lean_object* v_00_u03b1_1403_, lean_object* v_msg_1404_, lean_object* v___y_1405_, lean_object* v___y_1406_, lean_object* v___y_1407_){
_start:
{
lean_object* v_res_1408_; 
v_res_1408_ = lp_batteries_Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__3(v_00_u03b1_1403_, v_msg_1404_, v___y_1405_, v___y_1406_);
lean_dec(v___y_1406_);
lean_dec_ref(v___y_1405_);
return v_res_1408_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0(lean_object* v_00_u03b1_1409_, lean_object* v_constName_1410_, lean_object* v___y_1411_, lean_object* v___y_1412_, lean_object* v___y_1413_, lean_object* v___y_1414_){
_start:
{
lean_object* v___x_1416_; 
v___x_1416_ = lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0___redArg(v_constName_1410_, v___y_1411_, v___y_1412_, v___y_1413_, v___y_1414_);
return v___x_1416_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object* v_00_u03b1_1417_, lean_object* v_constName_1418_, lean_object* v___y_1419_, lean_object* v___y_1420_, lean_object* v___y_1421_, lean_object* v___y_1422_, lean_object* v___y_1423_){
_start:
{
lean_object* v_res_1424_; 
v_res_1424_ = lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0(v_00_u03b1_1417_, v_constName_1418_, v___y_1419_, v___y_1420_, v___y_1421_, v___y_1422_);
lean_dec(v___y_1422_);
lean_dec_ref(v___y_1421_);
lean_dec(v___y_1420_);
lean_dec_ref(v___y_1419_);
return v_res_1424_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2(lean_object* v_00_u03b1_1425_, lean_object* v_ref_1426_, lean_object* v_constName_1427_, lean_object* v___y_1428_, lean_object* v___y_1429_, lean_object* v___y_1430_, lean_object* v___y_1431_){
_start:
{
lean_object* v___x_1433_; 
v___x_1433_ = lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___redArg(v_ref_1426_, v_constName_1427_, v___y_1428_, v___y_1429_, v___y_1430_, v___y_1431_);
return v___x_1433_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2___boxed(lean_object* v_00_u03b1_1434_, lean_object* v_ref_1435_, lean_object* v_constName_1436_, lean_object* v___y_1437_, lean_object* v___y_1438_, lean_object* v___y_1439_, lean_object* v___y_1440_, lean_object* v___y_1441_){
_start:
{
lean_object* v_res_1442_; 
v_res_1442_ = lp_batteries_Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2(v_00_u03b1_1434_, v_ref_1435_, v_constName_1436_, v___y_1437_, v___y_1438_, v___y_1439_, v___y_1440_);
lean_dec(v___y_1440_);
lean_dec_ref(v___y_1439_);
lean_dec(v___y_1438_);
lean_dec_ref(v___y_1437_);
lean_dec(v_ref_1435_);
return v_res_1442_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7(lean_object* v_00_u03b1_1443_, lean_object* v_ref_1444_, lean_object* v_msg_1445_, lean_object* v_declHint_1446_, lean_object* v___y_1447_, lean_object* v___y_1448_, lean_object* v___y_1449_, lean_object* v___y_1450_){
_start:
{
lean_object* v___x_1452_; 
v___x_1452_ = lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7___redArg(v_ref_1444_, v_msg_1445_, v_declHint_1446_, v___y_1447_, v___y_1448_, v___y_1449_, v___y_1450_);
return v___x_1452_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7___boxed(lean_object* v_00_u03b1_1453_, lean_object* v_ref_1454_, lean_object* v_msg_1455_, lean_object* v_declHint_1456_, lean_object* v___y_1457_, lean_object* v___y_1458_, lean_object* v___y_1459_, lean_object* v___y_1460_, lean_object* v___y_1461_){
_start:
{
lean_object* v_res_1462_; 
v_res_1462_ = lp_batteries_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7(v_00_u03b1_1453_, v_ref_1454_, v_msg_1455_, v_declHint_1456_, v___y_1457_, v___y_1458_, v___y_1459_, v___y_1460_);
lean_dec(v___y_1460_);
lean_dec_ref(v___y_1459_);
lean_dec(v___y_1458_);
lean_dec_ref(v___y_1457_);
lean_dec(v_ref_1454_);
return v_res_1462_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9(lean_object* v_msg_1463_, lean_object* v_declHint_1464_, lean_object* v___y_1465_, lean_object* v___y_1466_, lean_object* v___y_1467_, lean_object* v___y_1468_){
_start:
{
lean_object* v___x_1470_; 
v___x_1470_ = lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___redArg(v_msg_1463_, v_declHint_1464_, v___y_1468_);
return v___x_1470_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9___boxed(lean_object* v_msg_1471_, lean_object* v_declHint_1472_, lean_object* v___y_1473_, lean_object* v___y_1474_, lean_object* v___y_1475_, lean_object* v___y_1476_, lean_object* v___y_1477_){
_start:
{
lean_object* v_res_1478_; 
v_res_1478_ = lp_batteries_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__8_spec__9(v_msg_1471_, v_declHint_1472_, v___y_1473_, v___y_1474_, v___y_1475_, v___y_1476_);
lean_dec(v___y_1476_);
lean_dec_ref(v___y_1475_);
lean_dec(v___y_1474_);
lean_dec_ref(v___y_1473_);
return v_res_1478_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__9(lean_object* v_00_u03b1_1479_, lean_object* v_ref_1480_, lean_object* v_msg_1481_, lean_object* v___y_1482_, lean_object* v___y_1483_, lean_object* v___y_1484_, lean_object* v___y_1485_){
_start:
{
lean_object* v___x_1487_; 
v___x_1487_ = lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__9___redArg(v_ref_1480_, v_msg_1481_, v___y_1482_, v___y_1483_, v___y_1484_, v___y_1485_);
return v___x_1487_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__9___boxed(lean_object* v_00_u03b1_1488_, lean_object* v_ref_1489_, lean_object* v_msg_1490_, lean_object* v___y_1491_, lean_object* v___y_1492_, lean_object* v___y_1493_, lean_object* v___y_1494_, lean_object* v___y_1495_){
_start:
{
lean_object* v_res_1496_; 
v_res_1496_ = lp_batteries_Lean_throwErrorAt___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0_spec__2_spec__7_spec__9(v_00_u03b1_1488_, v_ref_1489_, v_msg_1490_, v___y_1491_, v___y_1492_, v___y_1493_, v___y_1494_);
lean_dec(v___y_1494_);
lean_dec_ref(v___y_1493_);
lean_dec(v___y_1492_);
lean_dec_ref(v___y_1491_);
lean_dec(v_ref_1489_);
return v_res_1496_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_getExplicitFuncArg_x3f(lean_object* v_e_1497_, lean_object* v_a_1498_, lean_object* v_a_1499_, lean_object* v_a_1500_, lean_object* v_a_1501_){
_start:
{
if (lean_obj_tag(v_e_1497_) == 5)
{
lean_object* v_fn_1503_; lean_object* v_arg_1504_; lean_object* v___x_1505_; lean_object* v___x_1506_; lean_object* v___x_1507_; lean_object* v___x_1508_; 
v_fn_1503_ = lean_ctor_get(v_e_1497_, 0);
lean_inc_ref_n(v_fn_1503_, 2);
v_arg_1504_ = lean_ctor_get(v_e_1497_, 1);
lean_inc_ref_n(v_arg_1504_, 2);
v___x_1505_ = lean_unsigned_to_nat(1u);
v___x_1506_ = lean_mk_empty_array_with_capacity(v___x_1505_);
v___x_1507_ = lean_array_push(v___x_1506_, v_arg_1504_);
v___x_1508_ = l_Lean_Meta_mkAppM_x27(v_fn_1503_, v___x_1507_, v_a_1498_, v_a_1499_, v_a_1500_, v_a_1501_);
if (lean_obj_tag(v___x_1508_) == 0)
{
lean_object* v_a_1509_; lean_object* v___x_1510_; 
v_a_1509_ = lean_ctor_get(v___x_1508_, 0);
lean_inc(v_a_1509_);
lean_dec_ref_known(v___x_1508_, 1);
v___x_1510_ = l_Lean_Meta_isExprDefEq(v_a_1509_, v_e_1497_, v_a_1498_, v_a_1499_, v_a_1500_, v_a_1501_);
if (lean_obj_tag(v___x_1510_) == 0)
{
lean_object* v_a_1511_; lean_object* v___x_1513_; uint8_t v_isShared_1514_; uint8_t v_isSharedCheck_1522_; 
v_a_1511_ = lean_ctor_get(v___x_1510_, 0);
v_isSharedCheck_1522_ = !lean_is_exclusive(v___x_1510_);
if (v_isSharedCheck_1522_ == 0)
{
v___x_1513_ = v___x_1510_;
v_isShared_1514_ = v_isSharedCheck_1522_;
goto v_resetjp_1512_;
}
else
{
lean_inc(v_a_1511_);
lean_dec(v___x_1510_);
v___x_1513_ = lean_box(0);
v_isShared_1514_ = v_isSharedCheck_1522_;
goto v_resetjp_1512_;
}
v_resetjp_1512_:
{
uint8_t v___x_1515_; 
v___x_1515_ = lean_unbox(v_a_1511_);
lean_dec(v_a_1511_);
if (v___x_1515_ == 0)
{
lean_del_object(v___x_1513_);
lean_dec_ref(v_arg_1504_);
v_e_1497_ = v_fn_1503_;
goto _start;
}
else
{
lean_object* v___x_1517_; lean_object* v___x_1518_; lean_object* v___x_1520_; 
v___x_1517_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1517_, 0, v_fn_1503_);
lean_ctor_set(v___x_1517_, 1, v_arg_1504_);
v___x_1518_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1518_, 0, v___x_1517_);
if (v_isShared_1514_ == 0)
{
lean_ctor_set(v___x_1513_, 0, v___x_1518_);
v___x_1520_ = v___x_1513_;
goto v_reusejp_1519_;
}
else
{
lean_object* v_reuseFailAlloc_1521_; 
v_reuseFailAlloc_1521_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1521_, 0, v___x_1518_);
v___x_1520_ = v_reuseFailAlloc_1521_;
goto v_reusejp_1519_;
}
v_reusejp_1519_:
{
return v___x_1520_;
}
}
}
}
else
{
lean_object* v_a_1523_; lean_object* v___x_1525_; uint8_t v_isShared_1526_; uint8_t v_isSharedCheck_1530_; 
lean_dec_ref(v_arg_1504_);
lean_dec_ref(v_fn_1503_);
v_a_1523_ = lean_ctor_get(v___x_1510_, 0);
v_isSharedCheck_1530_ = !lean_is_exclusive(v___x_1510_);
if (v_isSharedCheck_1530_ == 0)
{
v___x_1525_ = v___x_1510_;
v_isShared_1526_ = v_isSharedCheck_1530_;
goto v_resetjp_1524_;
}
else
{
lean_inc(v_a_1523_);
lean_dec(v___x_1510_);
v___x_1525_ = lean_box(0);
v_isShared_1526_ = v_isSharedCheck_1530_;
goto v_resetjp_1524_;
}
v_resetjp_1524_:
{
lean_object* v___x_1528_; 
if (v_isShared_1526_ == 0)
{
v___x_1528_ = v___x_1525_;
goto v_reusejp_1527_;
}
else
{
lean_object* v_reuseFailAlloc_1529_; 
v_reuseFailAlloc_1529_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1529_, 0, v_a_1523_);
v___x_1528_ = v_reuseFailAlloc_1529_;
goto v_reusejp_1527_;
}
v_reusejp_1527_:
{
return v___x_1528_;
}
}
}
}
else
{
lean_object* v_a_1531_; lean_object* v___x_1533_; uint8_t v_isShared_1534_; uint8_t v_isSharedCheck_1538_; 
lean_dec_ref(v_arg_1504_);
lean_dec_ref(v_fn_1503_);
lean_dec_ref_known(v_e_1497_, 2);
v_a_1531_ = lean_ctor_get(v___x_1508_, 0);
v_isSharedCheck_1538_ = !lean_is_exclusive(v___x_1508_);
if (v_isSharedCheck_1538_ == 0)
{
v___x_1533_ = v___x_1508_;
v_isShared_1534_ = v_isSharedCheck_1538_;
goto v_resetjp_1532_;
}
else
{
lean_inc(v_a_1531_);
lean_dec(v___x_1508_);
v___x_1533_ = lean_box(0);
v_isShared_1534_ = v_isSharedCheck_1538_;
goto v_resetjp_1532_;
}
v_resetjp_1532_:
{
lean_object* v___x_1536_; 
if (v_isShared_1534_ == 0)
{
v___x_1536_ = v___x_1533_;
goto v_reusejp_1535_;
}
else
{
lean_object* v_reuseFailAlloc_1537_; 
v_reuseFailAlloc_1537_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1537_, 0, v_a_1531_);
v___x_1536_ = v_reuseFailAlloc_1537_;
goto v_reusejp_1535_;
}
v_reusejp_1535_:
{
return v___x_1536_;
}
}
}
}
else
{
lean_object* v___x_1539_; lean_object* v___x_1540_; 
lean_dec_ref(v_e_1497_);
v___x_1539_ = lean_box(0);
v___x_1540_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1540_, 0, v___x_1539_);
return v___x_1540_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_getExplicitFuncArg_x3f___boxed(lean_object* v_e_1541_, lean_object* v_a_1542_, lean_object* v_a_1543_, lean_object* v_a_1544_, lean_object* v_a_1545_, lean_object* v_a_1546_){
_start:
{
lean_object* v_res_1547_; 
v_res_1547_ = lp_batteries_Batteries_Tactic_getExplicitFuncArg_x3f(v_e_1541_, v_a_1542_, v_a_1543_, v_a_1544_, v_a_1545_);
lean_dec(v_a_1545_);
lean_dec_ref(v_a_1544_);
lean_dec(v_a_1543_);
lean_dec_ref(v_a_1542_);
return v_res_1547_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_getExplicitRelArg_x3f(lean_object* v_tgt_1548_, lean_object* v_f_1549_, lean_object* v_z_1550_, lean_object* v_a_1551_, lean_object* v_a_1552_, lean_object* v_a_1553_, lean_object* v_a_1554_){
_start:
{
if (lean_obj_tag(v_f_1549_) == 5)
{
lean_object* v_fn_1556_; lean_object* v_arg_1557_; lean_object* v___y_1559_; uint8_t v___y_1560_; lean_object* v_a_1564_; lean_object* v___x_1567_; lean_object* v___x_1568_; lean_object* v___x_1569_; lean_object* v___x_1570_; lean_object* v___x_1571_; 
v_fn_1556_ = lean_ctor_get(v_f_1549_, 0);
lean_inc_ref_n(v_fn_1556_, 2);
v_arg_1557_ = lean_ctor_get(v_f_1549_, 1);
lean_inc_ref_n(v_arg_1557_, 2);
lean_dec_ref_known(v_f_1549_, 2);
v___x_1567_ = lean_unsigned_to_nat(2u);
v___x_1568_ = lean_mk_empty_array_with_capacity(v___x_1567_);
v___x_1569_ = lean_array_push(v___x_1568_, v_arg_1557_);
lean_inc_ref(v_z_1550_);
v___x_1570_ = lean_array_push(v___x_1569_, v_z_1550_);
v___x_1571_ = l_Lean_Meta_mkAppM_x27(v_fn_1556_, v___x_1570_, v_a_1551_, v_a_1552_, v_a_1553_, v_a_1554_);
if (lean_obj_tag(v___x_1571_) == 0)
{
lean_object* v_a_1572_; lean_object* v___x_1574_; uint8_t v_isShared_1575_; uint8_t v_isSharedCheck_1592_; 
v_a_1572_ = lean_ctor_get(v___x_1571_, 0);
v_isSharedCheck_1592_ = !lean_is_exclusive(v___x_1571_);
if (v_isSharedCheck_1592_ == 0)
{
v___x_1574_ = v___x_1571_;
v_isShared_1575_ = v_isSharedCheck_1592_;
goto v_resetjp_1573_;
}
else
{
lean_inc(v_a_1572_);
lean_dec(v___x_1571_);
v___x_1574_ = lean_box(0);
v_isShared_1575_ = v_isSharedCheck_1592_;
goto v_resetjp_1573_;
}
v_resetjp_1573_:
{
lean_object* v___x_1576_; 
lean_inc_ref(v_tgt_1548_);
v___x_1576_ = l_Lean_Meta_isExprDefEq(v_a_1572_, v_tgt_1548_, v_a_1551_, v_a_1552_, v_a_1553_, v_a_1554_);
if (lean_obj_tag(v___x_1576_) == 0)
{
lean_object* v_a_1577_; lean_object* v___x_1579_; uint8_t v_isShared_1580_; uint8_t v_isSharedCheck_1590_; 
v_a_1577_ = lean_ctor_get(v___x_1576_, 0);
v_isSharedCheck_1590_ = !lean_is_exclusive(v___x_1576_);
if (v_isSharedCheck_1590_ == 0)
{
v___x_1579_ = v___x_1576_;
v_isShared_1580_ = v_isSharedCheck_1590_;
goto v_resetjp_1578_;
}
else
{
lean_inc(v_a_1577_);
lean_dec(v___x_1576_);
v___x_1579_ = lean_box(0);
v_isShared_1580_ = v_isSharedCheck_1590_;
goto v_resetjp_1578_;
}
v_resetjp_1578_:
{
uint8_t v___x_1581_; 
v___x_1581_ = lean_unbox(v_a_1577_);
lean_dec(v_a_1577_);
if (v___x_1581_ == 0)
{
lean_del_object(v___x_1579_);
lean_del_object(v___x_1574_);
lean_dec_ref(v_arg_1557_);
v_f_1549_ = v_fn_1556_;
goto _start;
}
else
{
lean_object* v___x_1583_; lean_object* v___x_1585_; 
lean_dec_ref(v_z_1550_);
lean_dec_ref(v_tgt_1548_);
v___x_1583_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1583_, 0, v_fn_1556_);
lean_ctor_set(v___x_1583_, 1, v_arg_1557_);
if (v_isShared_1575_ == 0)
{
lean_ctor_set_tag(v___x_1574_, 1);
lean_ctor_set(v___x_1574_, 0, v___x_1583_);
v___x_1585_ = v___x_1574_;
goto v_reusejp_1584_;
}
else
{
lean_object* v_reuseFailAlloc_1589_; 
v_reuseFailAlloc_1589_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1589_, 0, v___x_1583_);
v___x_1585_ = v_reuseFailAlloc_1589_;
goto v_reusejp_1584_;
}
v_reusejp_1584_:
{
lean_object* v___x_1587_; 
if (v_isShared_1580_ == 0)
{
lean_ctor_set(v___x_1579_, 0, v___x_1585_);
v___x_1587_ = v___x_1579_;
goto v_reusejp_1586_;
}
else
{
lean_object* v_reuseFailAlloc_1588_; 
v_reuseFailAlloc_1588_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1588_, 0, v___x_1585_);
v___x_1587_ = v_reuseFailAlloc_1588_;
goto v_reusejp_1586_;
}
v_reusejp_1586_:
{
return v___x_1587_;
}
}
}
}
}
else
{
lean_object* v_a_1591_; 
lean_del_object(v___x_1574_);
lean_dec_ref(v_arg_1557_);
v_a_1591_ = lean_ctor_get(v___x_1576_, 0);
lean_inc(v_a_1591_);
lean_dec_ref_known(v___x_1576_, 1);
v_a_1564_ = v_a_1591_;
goto v___jp_1563_;
}
}
}
else
{
lean_object* v_a_1593_; 
lean_dec_ref(v_arg_1557_);
v_a_1593_ = lean_ctor_get(v___x_1571_, 0);
lean_inc(v_a_1593_);
lean_dec_ref_known(v___x_1571_, 1);
v_a_1564_ = v_a_1593_;
goto v___jp_1563_;
}
v___jp_1558_:
{
if (v___y_1560_ == 0)
{
lean_dec_ref(v___y_1559_);
v_f_1549_ = v_fn_1556_;
goto _start;
}
else
{
lean_object* v___x_1562_; 
lean_dec_ref(v_fn_1556_);
lean_dec_ref(v_z_1550_);
lean_dec_ref(v_tgt_1548_);
v___x_1562_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1562_, 0, v___y_1559_);
return v___x_1562_;
}
}
v___jp_1563_:
{
uint8_t v___x_1565_; 
v___x_1565_ = l_Lean_Exception_isInterrupt(v_a_1564_);
if (v___x_1565_ == 0)
{
uint8_t v___x_1566_; 
lean_inc_ref(v_a_1564_);
v___x_1566_ = l_Lean_Exception_isRuntime(v_a_1564_);
v___y_1559_ = v_a_1564_;
v___y_1560_ = v___x_1566_;
goto v___jp_1558_;
}
else
{
v___y_1559_ = v_a_1564_;
v___y_1560_ = v___x_1565_;
goto v___jp_1558_;
}
}
}
else
{
lean_object* v___x_1594_; lean_object* v___x_1595_; 
lean_dec_ref(v_z_1550_);
lean_dec_ref(v_f_1549_);
lean_dec_ref(v_tgt_1548_);
v___x_1594_ = lean_box(0);
v___x_1595_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1595_, 0, v___x_1594_);
return v___x_1595_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_getExplicitRelArg_x3f___boxed(lean_object* v_tgt_1596_, lean_object* v_f_1597_, lean_object* v_z_1598_, lean_object* v_a_1599_, lean_object* v_a_1600_, lean_object* v_a_1601_, lean_object* v_a_1602_, lean_object* v_a_1603_){
_start:
{
lean_object* v_res_1604_; 
v_res_1604_ = lp_batteries_Batteries_Tactic_getExplicitRelArg_x3f(v_tgt_1596_, v_f_1597_, v_z_1598_, v_a_1599_, v_a_1600_, v_a_1601_, v_a_1602_);
lean_dec(v_a_1602_);
lean_dec_ref(v_a_1601_);
lean_dec(v_a_1600_);
lean_dec_ref(v_a_1599_);
return v_res_1604_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_getExplicitRelArgCore(lean_object* v_tgt_1605_, lean_object* v_rel_1606_, lean_object* v_x_1607_, lean_object* v_z_1608_, lean_object* v_a_1609_, lean_object* v_a_1610_, lean_object* v_a_1611_, lean_object* v_a_1612_){
_start:
{
lean_object* v___y_1618_; uint8_t v___y_1619_; lean_object* v_a_1622_; 
if (lean_obj_tag(v_rel_1606_) == 5)
{
lean_object* v_fn_1625_; lean_object* v___x_1626_; lean_object* v___x_1627_; lean_object* v___x_1628_; lean_object* v___x_1629_; lean_object* v___x_1630_; 
v_fn_1625_ = lean_ctor_get(v_rel_1606_, 0);
v___x_1626_ = lean_unsigned_to_nat(2u);
v___x_1627_ = lean_mk_empty_array_with_capacity(v___x_1626_);
lean_inc_ref(v_x_1607_);
v___x_1628_ = lean_array_push(v___x_1627_, v_x_1607_);
lean_inc_ref(v_z_1608_);
v___x_1629_ = lean_array_push(v___x_1628_, v_z_1608_);
lean_inc_ref(v_fn_1625_);
v___x_1630_ = l_Lean_Meta_mkAppM_x27(v_fn_1625_, v___x_1629_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_);
if (lean_obj_tag(v___x_1630_) == 0)
{
lean_object* v_a_1631_; lean_object* v___x_1632_; 
v_a_1631_ = lean_ctor_get(v___x_1630_, 0);
lean_inc(v_a_1631_);
lean_dec_ref_known(v___x_1630_, 1);
lean_inc_ref(v_tgt_1605_);
v___x_1632_ = l_Lean_Meta_isExprDefEq(v_a_1631_, v_tgt_1605_, v_a_1609_, v_a_1610_, v_a_1611_, v_a_1612_);
if (lean_obj_tag(v___x_1632_) == 0)
{
lean_object* v_a_1633_; uint8_t v___x_1634_; 
v_a_1633_ = lean_ctor_get(v___x_1632_, 0);
lean_inc(v_a_1633_);
lean_dec_ref_known(v___x_1632_, 1);
v___x_1634_ = lean_unbox(v_a_1633_);
lean_dec(v_a_1633_);
if (v___x_1634_ == 0)
{
lean_dec_ref(v_z_1608_);
lean_dec_ref(v_tgt_1605_);
goto v___jp_1614_;
}
else
{
lean_inc_ref(v_fn_1625_);
lean_dec_ref_known(v_rel_1606_, 2);
v_rel_1606_ = v_fn_1625_;
goto _start;
}
}
else
{
lean_object* v_a_1636_; 
lean_dec_ref(v_z_1608_);
lean_dec_ref(v_tgt_1605_);
v_a_1636_ = lean_ctor_get(v___x_1632_, 0);
lean_inc(v_a_1636_);
lean_dec_ref_known(v___x_1632_, 1);
v_a_1622_ = v_a_1636_;
goto v___jp_1621_;
}
}
else
{
lean_object* v_a_1637_; 
lean_dec_ref(v_z_1608_);
lean_dec_ref(v_tgt_1605_);
v_a_1637_ = lean_ctor_get(v___x_1630_, 0);
lean_inc(v_a_1637_);
lean_dec_ref_known(v___x_1630_, 1);
v_a_1622_ = v_a_1637_;
goto v___jp_1621_;
}
}
else
{
lean_object* v___x_1638_; lean_object* v___x_1639_; 
lean_dec_ref(v_z_1608_);
lean_dec_ref(v_tgt_1605_);
v___x_1638_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1638_, 0, v_rel_1606_);
lean_ctor_set(v___x_1638_, 1, v_x_1607_);
v___x_1639_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1639_, 0, v___x_1638_);
return v___x_1639_;
}
v___jp_1614_:
{
lean_object* v___x_1615_; lean_object* v___x_1616_; 
v___x_1615_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1615_, 0, v_rel_1606_);
lean_ctor_set(v___x_1615_, 1, v_x_1607_);
v___x_1616_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1616_, 0, v___x_1615_);
return v___x_1616_;
}
v___jp_1617_:
{
if (v___y_1619_ == 0)
{
lean_dec_ref(v___y_1618_);
goto v___jp_1614_;
}
else
{
lean_object* v___x_1620_; 
lean_dec_ref(v_x_1607_);
lean_dec_ref(v_rel_1606_);
v___x_1620_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1620_, 0, v___y_1618_);
return v___x_1620_;
}
}
v___jp_1621_:
{
uint8_t v___x_1623_; 
v___x_1623_ = l_Lean_Exception_isInterrupt(v_a_1622_);
if (v___x_1623_ == 0)
{
uint8_t v___x_1624_; 
lean_inc_ref(v_a_1622_);
v___x_1624_ = l_Lean_Exception_isRuntime(v_a_1622_);
v___y_1618_ = v_a_1622_;
v___y_1619_ = v___x_1624_;
goto v___jp_1617_;
}
else
{
v___y_1618_ = v_a_1622_;
v___y_1619_ = v___x_1623_;
goto v___jp_1617_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_getExplicitRelArgCore___boxed(lean_object* v_tgt_1640_, lean_object* v_rel_1641_, lean_object* v_x_1642_, lean_object* v_z_1643_, lean_object* v_a_1644_, lean_object* v_a_1645_, lean_object* v_a_1646_, lean_object* v_a_1647_, lean_object* v_a_1648_){
_start:
{
lean_object* v_res_1649_; 
v_res_1649_ = lp_batteries_Batteries_Tactic_getExplicitRelArgCore(v_tgt_1640_, v_rel_1641_, v_x_1642_, v_z_1643_, v_a_1644_, v_a_1645_, v_a_1646_, v_a_1647_);
lean_dec(v_a_1647_);
lean_dec_ref(v_a_1646_);
lean_dec(v_a_1645_);
lean_dec_ref(v_a_1644_);
return v_res_1649_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_TransRelation_ctorIdx(lean_object* v_x_1650_){
_start:
{
if (lean_obj_tag(v_x_1650_) == 0)
{
lean_object* v___x_1651_; 
v___x_1651_ = lean_unsigned_to_nat(0u);
return v___x_1651_;
}
else
{
lean_object* v___x_1652_; 
v___x_1652_ = lean_unsigned_to_nat(1u);
return v___x_1652_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_TransRelation_ctorIdx___boxed(lean_object* v_x_1653_){
_start:
{
lean_object* v_res_1654_; 
v_res_1654_ = lp_batteries_Batteries_Tactic_TransRelation_ctorIdx(v_x_1653_);
lean_dec_ref(v_x_1653_);
return v_res_1654_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_TransRelation_ctorElim___redArg(lean_object* v_t_1655_, lean_object* v_k_1656_){
_start:
{
if (lean_obj_tag(v_t_1655_) == 0)
{
lean_object* v_rel_1657_; lean_object* v___x_1658_; 
v_rel_1657_ = lean_ctor_get(v_t_1655_, 0);
lean_inc_ref(v_rel_1657_);
lean_dec_ref_known(v_t_1655_, 1);
v___x_1658_ = lean_apply_1(v_k_1656_, v_rel_1657_);
return v___x_1658_;
}
else
{
lean_object* v_name_1659_; uint8_t v_bi_1660_; lean_object* v___x_1661_; lean_object* v___x_1662_; 
v_name_1659_ = lean_ctor_get(v_t_1655_, 0);
lean_inc(v_name_1659_);
v_bi_1660_ = lean_ctor_get_uint8(v_t_1655_, sizeof(void*)*1);
lean_dec_ref_known(v_t_1655_, 1);
v___x_1661_ = lean_box(v_bi_1660_);
v___x_1662_ = lean_apply_2(v_k_1656_, v_name_1659_, v___x_1661_);
return v___x_1662_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_TransRelation_ctorElim(lean_object* v_motive_1663_, lean_object* v_ctorIdx_1664_, lean_object* v_t_1665_, lean_object* v_h_1666_, lean_object* v_k_1667_){
_start:
{
lean_object* v___x_1668_; 
v___x_1668_ = lp_batteries_Batteries_Tactic_TransRelation_ctorElim___redArg(v_t_1665_, v_k_1667_);
return v___x_1668_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_TransRelation_ctorElim___boxed(lean_object* v_motive_1669_, lean_object* v_ctorIdx_1670_, lean_object* v_t_1671_, lean_object* v_h_1672_, lean_object* v_k_1673_){
_start:
{
lean_object* v_res_1674_; 
v_res_1674_ = lp_batteries_Batteries_Tactic_TransRelation_ctorElim(v_motive_1669_, v_ctorIdx_1670_, v_t_1671_, v_h_1672_, v_k_1673_);
lean_dec(v_ctorIdx_1670_);
return v_res_1674_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_TransRelation_app_elim___redArg(lean_object* v_t_1675_, lean_object* v_app_1676_){
_start:
{
lean_object* v___x_1677_; 
v___x_1677_ = lp_batteries_Batteries_Tactic_TransRelation_ctorElim___redArg(v_t_1675_, v_app_1676_);
return v___x_1677_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_TransRelation_app_elim(lean_object* v_motive_1678_, lean_object* v_t_1679_, lean_object* v_h_1680_, lean_object* v_app_1681_){
_start:
{
lean_object* v___x_1682_; 
v___x_1682_ = lp_batteries_Batteries_Tactic_TransRelation_ctorElim___redArg(v_t_1679_, v_app_1681_);
return v___x_1682_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_TransRelation_implies_elim___redArg(lean_object* v_t_1683_, lean_object* v_implies_1684_){
_start:
{
lean_object* v___x_1685_; 
v___x_1685_ = lp_batteries_Batteries_Tactic_TransRelation_ctorElim___redArg(v_t_1683_, v_implies_1684_);
return v___x_1685_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_TransRelation_implies_elim(lean_object* v_motive_1686_, lean_object* v_t_1687_, lean_object* v_h_1688_, lean_object* v_implies_1689_){
_start:
{
lean_object* v___x_1690_; 
v___x_1690_ = lp_batteries_Batteries_Tactic_TransRelation_ctorElim___redArg(v_t_1687_, v_implies_1689_);
return v___x_1690_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_getRel(lean_object* v_tgt_1691_, lean_object* v_a_1692_, lean_object* v_a_1693_, lean_object* v_a_1694_, lean_object* v_a_1695_){
_start:
{
switch(lean_obj_tag(v_tgt_1691_))
{
case 7:
{
lean_object* v_binderName_1697_; lean_object* v_binderType_1698_; lean_object* v_body_1699_; uint8_t v_binderInfo_1700_; lean_object* v___x_1701_; lean_object* v___x_1702_; lean_object* v___x_1703_; lean_object* v___x_1704_; lean_object* v___x_1705_; 
v_binderName_1697_ = lean_ctor_get(v_tgt_1691_, 0);
lean_inc(v_binderName_1697_);
v_binderType_1698_ = lean_ctor_get(v_tgt_1691_, 1);
lean_inc_ref(v_binderType_1698_);
v_body_1699_ = lean_ctor_get(v_tgt_1691_, 2);
lean_inc_ref(v_body_1699_);
v_binderInfo_1700_ = lean_ctor_get_uint8(v_tgt_1691_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_tgt_1691_, 3);
v___x_1701_ = lean_alloc_ctor(1, 1, 1);
lean_ctor_set(v___x_1701_, 0, v_binderName_1697_);
lean_ctor_set_uint8(v___x_1701_, sizeof(void*)*1, v_binderInfo_1700_);
v___x_1702_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1702_, 0, v_binderType_1698_);
lean_ctor_set(v___x_1702_, 1, v_body_1699_);
v___x_1703_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1703_, 0, v___x_1701_);
lean_ctor_set(v___x_1703_, 1, v___x_1702_);
v___x_1704_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1704_, 0, v___x_1703_);
v___x_1705_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1705_, 0, v___x_1704_);
return v___x_1705_;
}
case 5:
{
lean_object* v_fn_1706_; lean_object* v_arg_1707_; lean_object* v___x_1708_; 
v_fn_1706_ = lean_ctor_get(v_tgt_1691_, 0);
v_arg_1707_ = lean_ctor_get(v_tgt_1691_, 1);
lean_inc_ref_n(v_arg_1707_, 2);
lean_inc_ref(v_fn_1706_);
lean_inc_ref(v_tgt_1691_);
v___x_1708_ = lp_batteries_Batteries_Tactic_getExplicitRelArg_x3f(v_tgt_1691_, v_fn_1706_, v_arg_1707_, v_a_1692_, v_a_1693_, v_a_1694_, v_a_1695_);
if (lean_obj_tag(v___x_1708_) == 0)
{
lean_object* v_a_1709_; lean_object* v___x_1711_; uint8_t v_isShared_1712_; uint8_t v_isSharedCheck_1761_; 
v_a_1709_ = lean_ctor_get(v___x_1708_, 0);
v_isSharedCheck_1761_ = !lean_is_exclusive(v___x_1708_);
if (v_isSharedCheck_1761_ == 0)
{
v___x_1711_ = v___x_1708_;
v_isShared_1712_ = v_isSharedCheck_1761_;
goto v_resetjp_1710_;
}
else
{
lean_inc(v_a_1709_);
lean_dec(v___x_1708_);
v___x_1711_ = lean_box(0);
v_isShared_1712_ = v_isSharedCheck_1761_;
goto v_resetjp_1710_;
}
v_resetjp_1710_:
{
if (lean_obj_tag(v_a_1709_) == 0)
{
lean_object* v___x_1713_; lean_object* v___x_1715_; 
lean_dec_ref(v_arg_1707_);
lean_dec_ref_known(v_tgt_1691_, 2);
v___x_1713_ = lean_box(0);
if (v_isShared_1712_ == 0)
{
lean_ctor_set(v___x_1711_, 0, v___x_1713_);
v___x_1715_ = v___x_1711_;
goto v_reusejp_1714_;
}
else
{
lean_object* v_reuseFailAlloc_1716_; 
v_reuseFailAlloc_1716_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1716_, 0, v___x_1713_);
v___x_1715_ = v_reuseFailAlloc_1716_;
goto v_reusejp_1714_;
}
v_reusejp_1714_:
{
return v___x_1715_;
}
}
else
{
lean_object* v_val_1717_; lean_object* v___x_1719_; uint8_t v_isShared_1720_; uint8_t v_isSharedCheck_1760_; 
lean_del_object(v___x_1711_);
v_val_1717_ = lean_ctor_get(v_a_1709_, 0);
v_isSharedCheck_1760_ = !lean_is_exclusive(v_a_1709_);
if (v_isSharedCheck_1760_ == 0)
{
v___x_1719_ = v_a_1709_;
v_isShared_1720_ = v_isSharedCheck_1760_;
goto v_resetjp_1718_;
}
else
{
lean_inc(v_val_1717_);
lean_dec(v_a_1709_);
v___x_1719_ = lean_box(0);
v_isShared_1720_ = v_isSharedCheck_1760_;
goto v_resetjp_1718_;
}
v_resetjp_1718_:
{
lean_object* v_fst_1721_; lean_object* v_snd_1722_; lean_object* v___x_1724_; uint8_t v_isShared_1725_; uint8_t v_isSharedCheck_1759_; 
v_fst_1721_ = lean_ctor_get(v_val_1717_, 0);
v_snd_1722_ = lean_ctor_get(v_val_1717_, 1);
v_isSharedCheck_1759_ = !lean_is_exclusive(v_val_1717_);
if (v_isSharedCheck_1759_ == 0)
{
v___x_1724_ = v_val_1717_;
v_isShared_1725_ = v_isSharedCheck_1759_;
goto v_resetjp_1723_;
}
else
{
lean_inc(v_snd_1722_);
lean_inc(v_fst_1721_);
lean_dec(v_val_1717_);
v___x_1724_ = lean_box(0);
v_isShared_1725_ = v_isSharedCheck_1759_;
goto v_resetjp_1723_;
}
v_resetjp_1723_:
{
lean_object* v___x_1726_; 
lean_inc_ref(v_arg_1707_);
v___x_1726_ = lp_batteries_Batteries_Tactic_getExplicitRelArgCore(v_tgt_1691_, v_fst_1721_, v_snd_1722_, v_arg_1707_, v_a_1692_, v_a_1693_, v_a_1694_, v_a_1695_);
if (lean_obj_tag(v___x_1726_) == 0)
{
lean_object* v_a_1727_; lean_object* v___x_1729_; uint8_t v_isShared_1730_; uint8_t v_isSharedCheck_1750_; 
v_a_1727_ = lean_ctor_get(v___x_1726_, 0);
v_isSharedCheck_1750_ = !lean_is_exclusive(v___x_1726_);
if (v_isSharedCheck_1750_ == 0)
{
v___x_1729_ = v___x_1726_;
v_isShared_1730_ = v_isSharedCheck_1750_;
goto v_resetjp_1728_;
}
else
{
lean_inc(v_a_1727_);
lean_dec(v___x_1726_);
v___x_1729_ = lean_box(0);
v_isShared_1730_ = v_isSharedCheck_1750_;
goto v_resetjp_1728_;
}
v_resetjp_1728_:
{
lean_object* v_fst_1731_; lean_object* v_snd_1732_; lean_object* v___x_1734_; uint8_t v_isShared_1735_; uint8_t v_isSharedCheck_1749_; 
v_fst_1731_ = lean_ctor_get(v_a_1727_, 0);
v_snd_1732_ = lean_ctor_get(v_a_1727_, 1);
v_isSharedCheck_1749_ = !lean_is_exclusive(v_a_1727_);
if (v_isSharedCheck_1749_ == 0)
{
v___x_1734_ = v_a_1727_;
v_isShared_1735_ = v_isSharedCheck_1749_;
goto v_resetjp_1733_;
}
else
{
lean_inc(v_snd_1732_);
lean_inc(v_fst_1731_);
lean_dec(v_a_1727_);
v___x_1734_ = lean_box(0);
v_isShared_1735_ = v_isSharedCheck_1749_;
goto v_resetjp_1733_;
}
v_resetjp_1733_:
{
lean_object* v___x_1736_; lean_object* v___x_1738_; 
v___x_1736_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1736_, 0, v_fst_1731_);
if (v_isShared_1735_ == 0)
{
lean_ctor_set(v___x_1734_, 1, v_arg_1707_);
lean_ctor_set(v___x_1734_, 0, v_snd_1732_);
v___x_1738_ = v___x_1734_;
goto v_reusejp_1737_;
}
else
{
lean_object* v_reuseFailAlloc_1748_; 
v_reuseFailAlloc_1748_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1748_, 0, v_snd_1732_);
lean_ctor_set(v_reuseFailAlloc_1748_, 1, v_arg_1707_);
v___x_1738_ = v_reuseFailAlloc_1748_;
goto v_reusejp_1737_;
}
v_reusejp_1737_:
{
lean_object* v___x_1740_; 
if (v_isShared_1725_ == 0)
{
lean_ctor_set(v___x_1724_, 1, v___x_1738_);
lean_ctor_set(v___x_1724_, 0, v___x_1736_);
v___x_1740_ = v___x_1724_;
goto v_reusejp_1739_;
}
else
{
lean_object* v_reuseFailAlloc_1747_; 
v_reuseFailAlloc_1747_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1747_, 0, v___x_1736_);
lean_ctor_set(v_reuseFailAlloc_1747_, 1, v___x_1738_);
v___x_1740_ = v_reuseFailAlloc_1747_;
goto v_reusejp_1739_;
}
v_reusejp_1739_:
{
lean_object* v___x_1742_; 
if (v_isShared_1720_ == 0)
{
lean_ctor_set(v___x_1719_, 0, v___x_1740_);
v___x_1742_ = v___x_1719_;
goto v_reusejp_1741_;
}
else
{
lean_object* v_reuseFailAlloc_1746_; 
v_reuseFailAlloc_1746_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1746_, 0, v___x_1740_);
v___x_1742_ = v_reuseFailAlloc_1746_;
goto v_reusejp_1741_;
}
v_reusejp_1741_:
{
lean_object* v___x_1744_; 
if (v_isShared_1730_ == 0)
{
lean_ctor_set(v___x_1729_, 0, v___x_1742_);
v___x_1744_ = v___x_1729_;
goto v_reusejp_1743_;
}
else
{
lean_object* v_reuseFailAlloc_1745_; 
v_reuseFailAlloc_1745_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1745_, 0, v___x_1742_);
v___x_1744_ = v_reuseFailAlloc_1745_;
goto v_reusejp_1743_;
}
v_reusejp_1743_:
{
return v___x_1744_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_1751_; lean_object* v___x_1753_; uint8_t v_isShared_1754_; uint8_t v_isSharedCheck_1758_; 
lean_del_object(v___x_1724_);
lean_del_object(v___x_1719_);
lean_dec_ref(v_arg_1707_);
v_a_1751_ = lean_ctor_get(v___x_1726_, 0);
v_isSharedCheck_1758_ = !lean_is_exclusive(v___x_1726_);
if (v_isSharedCheck_1758_ == 0)
{
v___x_1753_ = v___x_1726_;
v_isShared_1754_ = v_isSharedCheck_1758_;
goto v_resetjp_1752_;
}
else
{
lean_inc(v_a_1751_);
lean_dec(v___x_1726_);
v___x_1753_ = lean_box(0);
v_isShared_1754_ = v_isSharedCheck_1758_;
goto v_resetjp_1752_;
}
v_resetjp_1752_:
{
lean_object* v___x_1756_; 
if (v_isShared_1754_ == 0)
{
v___x_1756_ = v___x_1753_;
goto v_reusejp_1755_;
}
else
{
lean_object* v_reuseFailAlloc_1757_; 
v_reuseFailAlloc_1757_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1757_, 0, v_a_1751_);
v___x_1756_ = v_reuseFailAlloc_1757_;
goto v_reusejp_1755_;
}
v_reusejp_1755_:
{
return v___x_1756_;
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
lean_object* v_a_1762_; lean_object* v___x_1764_; uint8_t v_isShared_1765_; uint8_t v_isSharedCheck_1769_; 
lean_dec_ref(v_arg_1707_);
lean_dec_ref_known(v_tgt_1691_, 2);
v_a_1762_ = lean_ctor_get(v___x_1708_, 0);
v_isSharedCheck_1769_ = !lean_is_exclusive(v___x_1708_);
if (v_isSharedCheck_1769_ == 0)
{
v___x_1764_ = v___x_1708_;
v_isShared_1765_ = v_isSharedCheck_1769_;
goto v_resetjp_1763_;
}
else
{
lean_inc(v_a_1762_);
lean_dec(v___x_1708_);
v___x_1764_ = lean_box(0);
v_isShared_1765_ = v_isSharedCheck_1769_;
goto v_resetjp_1763_;
}
v_resetjp_1763_:
{
lean_object* v___x_1767_; 
if (v_isShared_1765_ == 0)
{
v___x_1767_ = v___x_1764_;
goto v_reusejp_1766_;
}
else
{
lean_object* v_reuseFailAlloc_1768_; 
v_reuseFailAlloc_1768_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1768_, 0, v_a_1762_);
v___x_1767_ = v_reuseFailAlloc_1768_;
goto v_reusejp_1766_;
}
v_reusejp_1766_:
{
return v___x_1767_;
}
}
}
}
default: 
{
lean_object* v___x_1770_; lean_object* v___x_1771_; 
lean_dec_ref(v_tgt_1691_);
v___x_1770_ = lean_box(0);
v___x_1771_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1771_, 0, v___x_1770_);
return v___x_1771_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_getRel___boxed(lean_object* v_tgt_1772_, lean_object* v_a_1773_, lean_object* v_a_1774_, lean_object* v_a_1775_, lean_object* v_a_1776_, lean_object* v_a_1777_){
_start:
{
lean_object* v_res_1778_; 
v_res_1778_ = lp_batteries_Batteries_Tactic_getRel(v_tgt_1772_, v_a_1773_, v_a_1774_, v_a_1775_, v_a_1776_);
lean_dec(v_a_1776_);
lean_dec_ref(v_a_1775_);
lean_dec(v_a_1774_);
lean_dec_ref(v_a_1773_);
return v_res_1778_;
}
}
static lean_object* _init_lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1829_; lean_object* v___x_1830_; lean_object* v___x_1831_; 
v___x_1829_ = lean_box(0);
v___x_1830_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1831_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1831_, 0, v___x_1830_);
lean_ctor_set(v___x_1831_, 1, v___x_1829_);
return v___x_1831_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__0___redArg(){
_start:
{
lean_object* v___x_1833_; lean_object* v___x_1834_; 
v___x_1833_ = lean_obj_once(&lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__0___redArg___closed__0, &lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__0___redArg___closed__0_once, _init_lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__0___redArg___closed__0);
v___x_1834_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1834_, 0, v___x_1833_);
return v___x_1834_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__0___redArg___boxed(lean_object* v___y_1835_){
_start:
{
lean_object* v_res_1836_; 
v_res_1836_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__0___redArg();
return v_res_1836_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__0(lean_object* v_00_u03b1_1837_, lean_object* v___y_1838_, lean_object* v___y_1839_, lean_object* v___y_1840_, lean_object* v___y_1841_, lean_object* v___y_1842_, lean_object* v___y_1843_, lean_object* v___y_1844_, lean_object* v___y_1845_){
_start:
{
lean_object* v___x_1847_; 
v___x_1847_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__0___redArg();
return v___x_1847_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__0___boxed(lean_object* v_00_u03b1_1848_, lean_object* v___y_1849_, lean_object* v___y_1850_, lean_object* v___y_1851_, lean_object* v___y_1852_, lean_object* v___y_1853_, lean_object* v___y_1854_, lean_object* v___y_1855_, lean_object* v___y_1856_, lean_object* v___y_1857_){
_start:
{
lean_object* v_res_1858_; 
v_res_1858_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__0(v_00_u03b1_1848_, v___y_1849_, v___y_1850_, v___y_1851_, v___y_1852_, v___y_1853_, v___y_1854_, v___y_1855_, v___y_1856_);
lean_dec(v___y_1856_);
lean_dec_ref(v___y_1855_);
lean_dec(v___y_1854_);
lean_dec_ref(v___y_1853_);
lean_dec(v___y_1852_);
lean_dec_ref(v___y_1851_);
lean_dec(v___y_1850_);
lean_dec_ref(v___y_1849_);
return v_res_1858_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instantiateMVars___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__1___redArg(lean_object* v_e_1859_, lean_object* v___y_1860_){
_start:
{
uint8_t v___x_1862_; 
v___x_1862_ = l_Lean_Expr_hasMVar(v_e_1859_);
if (v___x_1862_ == 0)
{
lean_object* v___x_1863_; 
v___x_1863_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1863_, 0, v_e_1859_);
return v___x_1863_;
}
else
{
lean_object* v___x_1864_; lean_object* v_mctx_1865_; lean_object* v___x_1866_; lean_object* v_fst_1867_; lean_object* v_snd_1868_; lean_object* v___x_1869_; lean_object* v_cache_1870_; lean_object* v_zetaDeltaFVarIds_1871_; lean_object* v_postponed_1872_; lean_object* v_diag_1873_; lean_object* v___x_1875_; uint8_t v_isShared_1876_; uint8_t v_isSharedCheck_1882_; 
v___x_1864_ = lean_st_ref_get(v___y_1860_);
v_mctx_1865_ = lean_ctor_get(v___x_1864_, 0);
lean_inc_ref(v_mctx_1865_);
lean_dec(v___x_1864_);
v___x_1866_ = l_Lean_instantiateMVarsCore(v_mctx_1865_, v_e_1859_);
v_fst_1867_ = lean_ctor_get(v___x_1866_, 0);
lean_inc(v_fst_1867_);
v_snd_1868_ = lean_ctor_get(v___x_1866_, 1);
lean_inc(v_snd_1868_);
lean_dec_ref(v___x_1866_);
v___x_1869_ = lean_st_ref_take(v___y_1860_);
v_cache_1870_ = lean_ctor_get(v___x_1869_, 1);
v_zetaDeltaFVarIds_1871_ = lean_ctor_get(v___x_1869_, 2);
v_postponed_1872_ = lean_ctor_get(v___x_1869_, 3);
v_diag_1873_ = lean_ctor_get(v___x_1869_, 4);
v_isSharedCheck_1882_ = !lean_is_exclusive(v___x_1869_);
if (v_isSharedCheck_1882_ == 0)
{
lean_object* v_unused_1883_; 
v_unused_1883_ = lean_ctor_get(v___x_1869_, 0);
lean_dec(v_unused_1883_);
v___x_1875_ = v___x_1869_;
v_isShared_1876_ = v_isSharedCheck_1882_;
goto v_resetjp_1874_;
}
else
{
lean_inc(v_diag_1873_);
lean_inc(v_postponed_1872_);
lean_inc(v_zetaDeltaFVarIds_1871_);
lean_inc(v_cache_1870_);
lean_dec(v___x_1869_);
v___x_1875_ = lean_box(0);
v_isShared_1876_ = v_isSharedCheck_1882_;
goto v_resetjp_1874_;
}
v_resetjp_1874_:
{
lean_object* v___x_1878_; 
if (v_isShared_1876_ == 0)
{
lean_ctor_set(v___x_1875_, 0, v_snd_1868_);
v___x_1878_ = v___x_1875_;
goto v_reusejp_1877_;
}
else
{
lean_object* v_reuseFailAlloc_1881_; 
v_reuseFailAlloc_1881_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1881_, 0, v_snd_1868_);
lean_ctor_set(v_reuseFailAlloc_1881_, 1, v_cache_1870_);
lean_ctor_set(v_reuseFailAlloc_1881_, 2, v_zetaDeltaFVarIds_1871_);
lean_ctor_set(v_reuseFailAlloc_1881_, 3, v_postponed_1872_);
lean_ctor_set(v_reuseFailAlloc_1881_, 4, v_diag_1873_);
v___x_1878_ = v_reuseFailAlloc_1881_;
goto v_reusejp_1877_;
}
v_reusejp_1877_:
{
lean_object* v___x_1879_; lean_object* v___x_1880_; 
v___x_1879_ = lean_st_ref_set(v___y_1860_, v___x_1878_);
v___x_1880_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1880_, 0, v_fst_1867_);
return v___x_1880_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instantiateMVars___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__1___redArg___boxed(lean_object* v_e_1884_, lean_object* v___y_1885_, lean_object* v___y_1886_){
_start:
{
lean_object* v_res_1887_; 
v_res_1887_ = lp_batteries_Lean_instantiateMVars___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__1___redArg(v_e_1884_, v___y_1885_);
lean_dec(v___y_1885_);
return v_res_1887_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instantiateMVars___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__1(lean_object* v_e_1888_, lean_object* v___y_1889_, lean_object* v___y_1890_, lean_object* v___y_1891_, lean_object* v___y_1892_, lean_object* v___y_1893_, lean_object* v___y_1894_, lean_object* v___y_1895_, lean_object* v___y_1896_){
_start:
{
lean_object* v___x_1898_; 
v___x_1898_ = lp_batteries_Lean_instantiateMVars___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__1___redArg(v_e_1888_, v___y_1894_);
return v___x_1898_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_instantiateMVars___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__1___boxed(lean_object* v_e_1899_, lean_object* v___y_1900_, lean_object* v___y_1901_, lean_object* v___y_1902_, lean_object* v___y_1903_, lean_object* v___y_1904_, lean_object* v___y_1905_, lean_object* v___y_1906_, lean_object* v___y_1907_, lean_object* v___y_1908_){
_start:
{
lean_object* v_res_1909_; 
v_res_1909_ = lp_batteries_Lean_instantiateMVars___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__1(v_e_1899_, v___y_1900_, v___y_1901_, v___y_1902_, v___y_1903_, v___y_1904_, v___y_1905_, v___y_1906_, v___y_1907_);
lean_dec(v___y_1907_);
lean_dec_ref(v___y_1906_);
lean_dec(v___y_1905_);
lean_dec_ref(v___y_1904_);
lean_dec(v___y_1903_);
lean_dec_ref(v___y_1902_);
lean_dec(v___y_1901_);
lean_dec_ref(v___y_1900_);
return v_res_1909_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__3___redArg___lam__0(lean_object* v_k_1910_, lean_object* v_b_1911_, lean_object* v_c_1912_, lean_object* v___y_1913_, lean_object* v___y_1914_, lean_object* v___y_1915_, lean_object* v___y_1916_){
_start:
{
lean_object* v___x_1918_; 
lean_inc(v___y_1916_);
lean_inc_ref(v___y_1915_);
lean_inc(v___y_1914_);
lean_inc_ref(v___y_1913_);
v___x_1918_ = lean_apply_7(v_k_1910_, v_b_1911_, v_c_1912_, v___y_1913_, v___y_1914_, v___y_1915_, v___y_1916_, lean_box(0));
return v___x_1918_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__3___redArg___lam__0___boxed(lean_object* v_k_1919_, lean_object* v_b_1920_, lean_object* v_c_1921_, lean_object* v___y_1922_, lean_object* v___y_1923_, lean_object* v___y_1924_, lean_object* v___y_1925_, lean_object* v___y_1926_){
_start:
{
lean_object* v_res_1927_; 
v_res_1927_ = lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__3___redArg___lam__0(v_k_1919_, v_b_1920_, v_c_1921_, v___y_1922_, v___y_1923_, v___y_1924_, v___y_1925_);
lean_dec(v___y_1925_);
lean_dec_ref(v___y_1924_);
lean_dec(v___y_1923_);
lean_dec_ref(v___y_1922_);
return v_res_1927_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__3___redArg(lean_object* v_type_1928_, lean_object* v_k_1929_, uint8_t v_cleanupAnnotations_1930_, uint8_t v_whnfType_1931_, lean_object* v___y_1932_, lean_object* v___y_1933_, lean_object* v___y_1934_, lean_object* v___y_1935_){
_start:
{
lean_object* v___f_1937_; lean_object* v___x_1938_; 
v___f_1937_ = lean_alloc_closure((void*)(lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__3___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_1937_, 0, v_k_1929_);
v___x_1938_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingImp(lean_box(0), v_type_1928_, v___f_1937_, v_cleanupAnnotations_1930_, v_whnfType_1931_, v___y_1932_, v___y_1933_, v___y_1934_, v___y_1935_);
if (lean_obj_tag(v___x_1938_) == 0)
{
lean_object* v_a_1939_; lean_object* v___x_1941_; uint8_t v_isShared_1942_; uint8_t v_isSharedCheck_1946_; 
v_a_1939_ = lean_ctor_get(v___x_1938_, 0);
v_isSharedCheck_1946_ = !lean_is_exclusive(v___x_1938_);
if (v_isSharedCheck_1946_ == 0)
{
v___x_1941_ = v___x_1938_;
v_isShared_1942_ = v_isSharedCheck_1946_;
goto v_resetjp_1940_;
}
else
{
lean_inc(v_a_1939_);
lean_dec(v___x_1938_);
v___x_1941_ = lean_box(0);
v_isShared_1942_ = v_isSharedCheck_1946_;
goto v_resetjp_1940_;
}
v_resetjp_1940_:
{
lean_object* v___x_1944_; 
if (v_isShared_1942_ == 0)
{
v___x_1944_ = v___x_1941_;
goto v_reusejp_1943_;
}
else
{
lean_object* v_reuseFailAlloc_1945_; 
v_reuseFailAlloc_1945_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1945_, 0, v_a_1939_);
v___x_1944_ = v_reuseFailAlloc_1945_;
goto v_reusejp_1943_;
}
v_reusejp_1943_:
{
return v___x_1944_;
}
}
}
else
{
lean_object* v_a_1947_; lean_object* v___x_1949_; uint8_t v_isShared_1950_; uint8_t v_isSharedCheck_1954_; 
v_a_1947_ = lean_ctor_get(v___x_1938_, 0);
v_isSharedCheck_1954_ = !lean_is_exclusive(v___x_1938_);
if (v_isSharedCheck_1954_ == 0)
{
v___x_1949_ = v___x_1938_;
v_isShared_1950_ = v_isSharedCheck_1954_;
goto v_resetjp_1948_;
}
else
{
lean_inc(v_a_1947_);
lean_dec(v___x_1938_);
v___x_1949_ = lean_box(0);
v_isShared_1950_ = v_isSharedCheck_1954_;
goto v_resetjp_1948_;
}
v_resetjp_1948_:
{
lean_object* v___x_1952_; 
if (v_isShared_1950_ == 0)
{
v___x_1952_ = v___x_1949_;
goto v_reusejp_1951_;
}
else
{
lean_object* v_reuseFailAlloc_1953_; 
v_reuseFailAlloc_1953_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1953_, 0, v_a_1947_);
v___x_1952_ = v_reuseFailAlloc_1953_;
goto v_reusejp_1951_;
}
v_reusejp_1951_:
{
return v___x_1952_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__3___redArg___boxed(lean_object* v_type_1955_, lean_object* v_k_1956_, lean_object* v_cleanupAnnotations_1957_, lean_object* v_whnfType_1958_, lean_object* v___y_1959_, lean_object* v___y_1960_, lean_object* v___y_1961_, lean_object* v___y_1962_, lean_object* v___y_1963_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_1964_; uint8_t v_whnfType_boxed_1965_; lean_object* v_res_1966_; 
v_cleanupAnnotations_boxed_1964_ = lean_unbox(v_cleanupAnnotations_1957_);
v_whnfType_boxed_1965_ = lean_unbox(v_whnfType_1958_);
v_res_1966_ = lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__3___redArg(v_type_1955_, v_k_1956_, v_cleanupAnnotations_boxed_1964_, v_whnfType_boxed_1965_, v___y_1959_, v___y_1960_, v___y_1961_, v___y_1962_);
lean_dec(v___y_1962_);
lean_dec_ref(v___y_1961_);
lean_dec(v___y_1960_);
lean_dec_ref(v___y_1959_);
return v_res_1966_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__3(lean_object* v_00_u03b1_1967_, lean_object* v_type_1968_, lean_object* v_k_1969_, uint8_t v_cleanupAnnotations_1970_, uint8_t v_whnfType_1971_, lean_object* v___y_1972_, lean_object* v___y_1973_, lean_object* v___y_1974_, lean_object* v___y_1975_){
_start:
{
lean_object* v___x_1977_; 
v___x_1977_ = lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__3___redArg(v_type_1968_, v_k_1969_, v_cleanupAnnotations_1970_, v_whnfType_1971_, v___y_1972_, v___y_1973_, v___y_1974_, v___y_1975_);
return v___x_1977_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__3___boxed(lean_object* v_00_u03b1_1978_, lean_object* v_type_1979_, lean_object* v_k_1980_, lean_object* v_cleanupAnnotations_1981_, lean_object* v_whnfType_1982_, lean_object* v___y_1983_, lean_object* v___y_1984_, lean_object* v___y_1985_, lean_object* v___y_1986_, lean_object* v___y_1987_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_1988_; uint8_t v_whnfType_boxed_1989_; lean_object* v_res_1990_; 
v_cleanupAnnotations_boxed_1988_ = lean_unbox(v_cleanupAnnotations_1981_);
v_whnfType_boxed_1989_ = lean_unbox(v_whnfType_1982_);
v_res_1990_ = lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__3(v_00_u03b1_1978_, v_type_1979_, v_k_1980_, v_cleanupAnnotations_boxed_1988_, v_whnfType_boxed_1989_, v___y_1983_, v___y_1984_, v___y_1985_, v___y_1986_);
lean_dec(v___y_1986_);
lean_dec_ref(v___y_1985_);
lean_dec(v___y_1984_);
lean_dec_ref(v___y_1983_);
return v_res_1990_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__0(lean_object* v___x_1994_, lean_object* v___y_1995_, lean_object* v___y_1996_, lean_object* v___y_1997_, lean_object* v___y_1998_, lean_object* v___y_1999_, lean_object* v___y_2000_, lean_object* v___y_2001_, lean_object* v___y_2002_){
_start:
{
lean_object* v_options_2004_; uint8_t v_hasTrace_2005_; 
v_options_2004_ = lean_ctor_get(v___y_2001_, 2);
v_hasTrace_2005_ = lean_ctor_get_uint8(v_options_2004_, sizeof(void*)*1);
if (v_hasTrace_2005_ == 0)
{
lean_object* v___x_2006_; lean_object* v___x_2007_; 
lean_dec(v___x_1994_);
v___x_2006_ = lean_box(v_hasTrace_2005_);
v___x_2007_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2007_, 0, v___x_2006_);
return v___x_2007_;
}
else
{
lean_object* v_inheritedTraceOptions_2008_; lean_object* v___x_2009_; lean_object* v___x_2010_; uint8_t v___x_2011_; lean_object* v___x_2012_; lean_object* v___x_2013_; 
v_inheritedTraceOptions_2008_ = lean_ctor_get(v___y_2001_, 13);
v___x_2009_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__0___closed__1));
v___x_2010_ = l_Lean_Name_append(v___x_2009_, v___x_1994_);
v___x_2011_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2008_, v_options_2004_, v___x_2010_);
lean_dec(v___x_2010_);
v___x_2012_ = lean_box(v___x_2011_);
v___x_2013_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2013_, 0, v___x_2012_);
return v___x_2013_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__0___boxed(lean_object* v___x_2014_, lean_object* v___y_2015_, lean_object* v___y_2016_, lean_object* v___y_2017_, lean_object* v___y_2018_, lean_object* v___y_2019_, lean_object* v___y_2020_, lean_object* v___y_2021_, lean_object* v___y_2022_, lean_object* v___y_2023_){
_start:
{
lean_object* v_res_2024_; 
v_res_2024_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__0(v___x_2014_, v___y_2015_, v___y_2016_, v___y_2017_, v___y_2018_, v___y_2019_, v___y_2020_, v___y_2021_, v___y_2022_);
lean_dec(v___y_2022_);
lean_dec_ref(v___y_2021_);
lean_dec(v___y_2020_);
lean_dec_ref(v___y_2019_);
lean_dec(v___y_2018_);
lean_dec_ref(v___y_2017_);
lean_dec(v___y_2016_);
lean_dec_ref(v___y_2015_);
return v_res_2024_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__8___redArg(lean_object* v_msg_2025_, lean_object* v___y_2026_, lean_object* v___y_2027_, lean_object* v___y_2028_, lean_object* v___y_2029_){
_start:
{
lean_object* v_ref_2031_; lean_object* v___x_2032_; lean_object* v_a_2033_; lean_object* v___x_2035_; uint8_t v_isShared_2036_; uint8_t v_isSharedCheck_2041_; 
v_ref_2031_ = lean_ctor_get(v___y_2028_, 5);
v___x_2032_ = lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1_spec__2(v_msg_2025_, v___y_2026_, v___y_2027_, v___y_2028_, v___y_2029_);
v_a_2033_ = lean_ctor_get(v___x_2032_, 0);
v_isSharedCheck_2041_ = !lean_is_exclusive(v___x_2032_);
if (v_isSharedCheck_2041_ == 0)
{
v___x_2035_ = v___x_2032_;
v_isShared_2036_ = v_isSharedCheck_2041_;
goto v_resetjp_2034_;
}
else
{
lean_inc(v_a_2033_);
lean_dec(v___x_2032_);
v___x_2035_ = lean_box(0);
v_isShared_2036_ = v_isSharedCheck_2041_;
goto v_resetjp_2034_;
}
v_resetjp_2034_:
{
lean_object* v___x_2037_; lean_object* v___x_2039_; 
lean_inc(v_ref_2031_);
v___x_2037_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2037_, 0, v_ref_2031_);
lean_ctor_set(v___x_2037_, 1, v_a_2033_);
if (v_isShared_2036_ == 0)
{
lean_ctor_set_tag(v___x_2035_, 1);
lean_ctor_set(v___x_2035_, 0, v___x_2037_);
v___x_2039_ = v___x_2035_;
goto v_reusejp_2038_;
}
else
{
lean_object* v_reuseFailAlloc_2040_; 
v_reuseFailAlloc_2040_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2040_, 0, v___x_2037_);
v___x_2039_ = v_reuseFailAlloc_2040_;
goto v_reusejp_2038_;
}
v_reusejp_2038_:
{
return v___x_2039_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__8___redArg___boxed(lean_object* v_msg_2042_, lean_object* v___y_2043_, lean_object* v___y_2044_, lean_object* v___y_2045_, lean_object* v___y_2046_, lean_object* v___y_2047_){
_start:
{
lean_object* v_res_2048_; 
v_res_2048_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__8___redArg(v_msg_2042_, v___y_2043_, v___y_2044_, v___y_2045_, v___y_2046_);
lean_dec(v___y_2046_);
lean_dec_ref(v___y_2045_);
lean_dec(v___y_2044_);
lean_dec_ref(v___y_2043_);
return v_res_2048_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__4(lean_object* v_a_2049_, uint8_t v___y_2050_, lean_object* v_____r_2051_, lean_object* v___y_2052_, lean_object* v___y_2053_, lean_object* v___y_2054_, lean_object* v___y_2055_, lean_object* v___y_2056_, lean_object* v___y_2057_, lean_object* v___y_2058_, lean_object* v___y_2059_){
_start:
{
lean_object* v___x_2061_; 
v___x_2061_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_2049_, v___y_2050_, v___y_2053_, v___y_2054_, v___y_2055_, v___y_2056_, v___y_2057_, v___y_2058_, v___y_2059_);
if (lean_obj_tag(v___x_2061_) == 0)
{
lean_object* v_a_2062_; lean_object* v___x_2064_; uint8_t v_isShared_2065_; uint8_t v_isSharedCheck_2070_; 
v_a_2062_ = lean_ctor_get(v___x_2061_, 0);
v_isSharedCheck_2070_ = !lean_is_exclusive(v___x_2061_);
if (v_isSharedCheck_2070_ == 0)
{
v___x_2064_ = v___x_2061_;
v_isShared_2065_ = v_isSharedCheck_2070_;
goto v_resetjp_2063_;
}
else
{
lean_inc(v_a_2062_);
lean_dec(v___x_2061_);
v___x_2064_ = lean_box(0);
v_isShared_2065_ = v_isSharedCheck_2070_;
goto v_resetjp_2063_;
}
v_resetjp_2063_:
{
lean_object* v___x_2066_; lean_object* v___x_2068_; 
v___x_2066_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2066_, 0, v_a_2062_);
if (v_isShared_2065_ == 0)
{
lean_ctor_set(v___x_2064_, 0, v___x_2066_);
v___x_2068_ = v___x_2064_;
goto v_reusejp_2067_;
}
else
{
lean_object* v_reuseFailAlloc_2069_; 
v_reuseFailAlloc_2069_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2069_, 0, v___x_2066_);
v___x_2068_ = v_reuseFailAlloc_2069_;
goto v_reusejp_2067_;
}
v_reusejp_2067_:
{
return v___x_2068_;
}
}
}
else
{
lean_object* v_a_2071_; lean_object* v___x_2073_; uint8_t v_isShared_2074_; uint8_t v_isSharedCheck_2078_; 
v_a_2071_ = lean_ctor_get(v___x_2061_, 0);
v_isSharedCheck_2078_ = !lean_is_exclusive(v___x_2061_);
if (v_isSharedCheck_2078_ == 0)
{
v___x_2073_ = v___x_2061_;
v_isShared_2074_ = v_isSharedCheck_2078_;
goto v_resetjp_2072_;
}
else
{
lean_inc(v_a_2071_);
lean_dec(v___x_2061_);
v___x_2073_ = lean_box(0);
v_isShared_2074_ = v_isSharedCheck_2078_;
goto v_resetjp_2072_;
}
v_resetjp_2072_:
{
lean_object* v___x_2076_; 
if (v_isShared_2074_ == 0)
{
v___x_2076_ = v___x_2073_;
goto v_reusejp_2075_;
}
else
{
lean_object* v_reuseFailAlloc_2077_; 
v_reuseFailAlloc_2077_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2077_, 0, v_a_2071_);
v___x_2076_ = v_reuseFailAlloc_2077_;
goto v_reusejp_2075_;
}
v_reusejp_2075_:
{
return v___x_2076_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__4___boxed(lean_object* v_a_2079_, lean_object* v___y_2080_, lean_object* v_____r_2081_, lean_object* v___y_2082_, lean_object* v___y_2083_, lean_object* v___y_2084_, lean_object* v___y_2085_, lean_object* v___y_2086_, lean_object* v___y_2087_, lean_object* v___y_2088_, lean_object* v___y_2089_, lean_object* v___y_2090_){
_start:
{
uint8_t v___y_91768__boxed_2091_; lean_object* v_res_2092_; 
v___y_91768__boxed_2091_ = lean_unbox(v___y_2080_);
v_res_2092_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__4(v_a_2079_, v___y_91768__boxed_2091_, v_____r_2081_, v___y_2082_, v___y_2083_, v___y_2084_, v___y_2085_, v___y_2086_, v___y_2087_, v___y_2088_, v___y_2089_);
lean_dec(v___y_2089_);
lean_dec_ref(v___y_2088_);
lean_dec(v___y_2087_);
lean_dec_ref(v___y_2086_);
lean_dec(v___y_2085_);
lean_dec_ref(v___y_2084_);
lean_dec(v___y_2083_);
lean_dec_ref(v___y_2082_);
return v_res_2092_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__0(lean_object* v_es_2093_, lean_object* v_x_2094_, lean_object* v___y_2095_, lean_object* v___y_2096_, lean_object* v___y_2097_, lean_object* v___y_2098_){
_start:
{
lean_object* v___x_2100_; lean_object* v___x_2101_; 
v___x_2100_ = lean_array_get_size(v_es_2093_);
v___x_2101_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2101_, 0, v___x_2100_);
return v___x_2101_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__0___boxed(lean_object* v_es_2102_, lean_object* v_x_2103_, lean_object* v___y_2104_, lean_object* v___y_2105_, lean_object* v___y_2106_, lean_object* v___y_2107_, lean_object* v___y_2108_){
_start:
{
lean_object* v_res_2109_; 
v_res_2109_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__0(v_es_2102_, v_x_2103_, v___y_2104_, v___y_2105_, v___y_2106_, v___y_2107_);
lean_dec(v___y_2107_);
lean_dec_ref(v___y_2106_);
lean_dec(v___y_2105_);
lean_dec_ref(v___y_2104_);
lean_dec_ref(v_x_2103_);
lean_dec_ref(v_es_2102_);
return v_res_2109_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__1(lean_object* v___x_2110_, lean_object* v___y_2111_, lean_object* v___y_2112_, lean_object* v___y_2113_, lean_object* v___y_2114_){
_start:
{
lean_object* v_options_2116_; uint8_t v_hasTrace_2117_; 
v_options_2116_ = lean_ctor_get(v___y_2113_, 2);
v_hasTrace_2117_ = lean_ctor_get_uint8(v_options_2116_, sizeof(void*)*1);
if (v_hasTrace_2117_ == 0)
{
lean_object* v___x_2118_; lean_object* v___x_2119_; 
lean_dec(v___x_2110_);
v___x_2118_ = lean_box(v_hasTrace_2117_);
v___x_2119_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2119_, 0, v___x_2118_);
return v___x_2119_;
}
else
{
lean_object* v_inheritedTraceOptions_2120_; lean_object* v___x_2121_; lean_object* v___x_2122_; uint8_t v___x_2123_; lean_object* v___x_2124_; lean_object* v___x_2125_; 
v_inheritedTraceOptions_2120_ = lean_ctor_get(v___y_2113_, 13);
v___x_2121_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__0___closed__1));
v___x_2122_ = l_Lean_Name_append(v___x_2121_, v___x_2110_);
v___x_2123_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2120_, v_options_2116_, v___x_2122_);
lean_dec(v___x_2122_);
v___x_2124_ = lean_box(v___x_2123_);
v___x_2125_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2125_, 0, v___x_2124_);
return v___x_2125_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__1___boxed(lean_object* v___x_2126_, lean_object* v___y_2127_, lean_object* v___y_2128_, lean_object* v___y_2129_, lean_object* v___y_2130_, lean_object* v___y_2131_){
_start:
{
lean_object* v_res_2132_; 
v_res_2132_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__1(v___x_2126_, v___y_2127_, v___y_2128_, v___y_2129_, v___y_2130_);
lean_dec(v___y_2130_);
lean_dec_ref(v___y_2129_);
lean_dec(v___y_2128_);
lean_dec_ref(v___y_2127_);
return v_res_2132_;
}
}
static double _init_lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg___closed__0(void){
_start:
{
lean_object* v___x_2133_; double v___x_2134_; 
v___x_2133_ = lean_unsigned_to_nat(0u);
v___x_2134_ = lean_float_of_nat(v___x_2133_);
return v___x_2134_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg(lean_object* v_cls_2138_, lean_object* v_msg_2139_, lean_object* v___y_2140_, lean_object* v___y_2141_, lean_object* v___y_2142_, lean_object* v___y_2143_){
_start:
{
lean_object* v_ref_2145_; lean_object* v___x_2146_; lean_object* v_a_2147_; lean_object* v___x_2149_; uint8_t v_isShared_2150_; uint8_t v_isSharedCheck_2191_; 
v_ref_2145_ = lean_ctor_get(v___y_2142_, 5);
v___x_2146_ = lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1_spec__2(v_msg_2139_, v___y_2140_, v___y_2141_, v___y_2142_, v___y_2143_);
v_a_2147_ = lean_ctor_get(v___x_2146_, 0);
v_isSharedCheck_2191_ = !lean_is_exclusive(v___x_2146_);
if (v_isSharedCheck_2191_ == 0)
{
v___x_2149_ = v___x_2146_;
v_isShared_2150_ = v_isSharedCheck_2191_;
goto v_resetjp_2148_;
}
else
{
lean_inc(v_a_2147_);
lean_dec(v___x_2146_);
v___x_2149_ = lean_box(0);
v_isShared_2150_ = v_isSharedCheck_2191_;
goto v_resetjp_2148_;
}
v_resetjp_2148_:
{
lean_object* v___x_2151_; lean_object* v_traceState_2152_; lean_object* v_env_2153_; lean_object* v_nextMacroScope_2154_; lean_object* v_ngen_2155_; lean_object* v_auxDeclNGen_2156_; lean_object* v_cache_2157_; lean_object* v_messages_2158_; lean_object* v_infoState_2159_; lean_object* v_snapshotTasks_2160_; lean_object* v___x_2162_; uint8_t v_isShared_2163_; uint8_t v_isSharedCheck_2190_; 
v___x_2151_ = lean_st_ref_take(v___y_2143_);
v_traceState_2152_ = lean_ctor_get(v___x_2151_, 4);
v_env_2153_ = lean_ctor_get(v___x_2151_, 0);
v_nextMacroScope_2154_ = lean_ctor_get(v___x_2151_, 1);
v_ngen_2155_ = lean_ctor_get(v___x_2151_, 2);
v_auxDeclNGen_2156_ = lean_ctor_get(v___x_2151_, 3);
v_cache_2157_ = lean_ctor_get(v___x_2151_, 5);
v_messages_2158_ = lean_ctor_get(v___x_2151_, 6);
v_infoState_2159_ = lean_ctor_get(v___x_2151_, 7);
v_snapshotTasks_2160_ = lean_ctor_get(v___x_2151_, 8);
v_isSharedCheck_2190_ = !lean_is_exclusive(v___x_2151_);
if (v_isSharedCheck_2190_ == 0)
{
v___x_2162_ = v___x_2151_;
v_isShared_2163_ = v_isSharedCheck_2190_;
goto v_resetjp_2161_;
}
else
{
lean_inc(v_snapshotTasks_2160_);
lean_inc(v_infoState_2159_);
lean_inc(v_messages_2158_);
lean_inc(v_cache_2157_);
lean_inc(v_traceState_2152_);
lean_inc(v_auxDeclNGen_2156_);
lean_inc(v_ngen_2155_);
lean_inc(v_nextMacroScope_2154_);
lean_inc(v_env_2153_);
lean_dec(v___x_2151_);
v___x_2162_ = lean_box(0);
v_isShared_2163_ = v_isSharedCheck_2190_;
goto v_resetjp_2161_;
}
v_resetjp_2161_:
{
uint64_t v_tid_2164_; lean_object* v_traces_2165_; lean_object* v___x_2167_; uint8_t v_isShared_2168_; uint8_t v_isSharedCheck_2189_; 
v_tid_2164_ = lean_ctor_get_uint64(v_traceState_2152_, sizeof(void*)*1);
v_traces_2165_ = lean_ctor_get(v_traceState_2152_, 0);
v_isSharedCheck_2189_ = !lean_is_exclusive(v_traceState_2152_);
if (v_isSharedCheck_2189_ == 0)
{
v___x_2167_ = v_traceState_2152_;
v_isShared_2168_ = v_isSharedCheck_2189_;
goto v_resetjp_2166_;
}
else
{
lean_inc(v_traces_2165_);
lean_dec(v_traceState_2152_);
v___x_2167_ = lean_box(0);
v_isShared_2168_ = v_isSharedCheck_2189_;
goto v_resetjp_2166_;
}
v_resetjp_2166_:
{
lean_object* v___x_2169_; double v___x_2170_; uint8_t v___x_2171_; lean_object* v___x_2172_; lean_object* v___x_2173_; lean_object* v___x_2174_; lean_object* v___x_2175_; lean_object* v___x_2176_; lean_object* v___x_2177_; lean_object* v___x_2179_; 
v___x_2169_ = lean_box(0);
v___x_2170_ = lean_float_once(&lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg___closed__0, &lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg___closed__0_once, _init_lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg___closed__0);
v___x_2171_ = 0;
v___x_2172_ = ((lean_object*)(lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg___closed__1));
v___x_2173_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_2173_, 0, v_cls_2138_);
lean_ctor_set(v___x_2173_, 1, v___x_2169_);
lean_ctor_set(v___x_2173_, 2, v___x_2172_);
lean_ctor_set_float(v___x_2173_, sizeof(void*)*3, v___x_2170_);
lean_ctor_set_float(v___x_2173_, sizeof(void*)*3 + 8, v___x_2170_);
lean_ctor_set_uint8(v___x_2173_, sizeof(void*)*3 + 16, v___x_2171_);
v___x_2174_ = ((lean_object*)(lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg___closed__2));
v___x_2175_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_2175_, 0, v___x_2173_);
lean_ctor_set(v___x_2175_, 1, v_a_2147_);
lean_ctor_set(v___x_2175_, 2, v___x_2174_);
lean_inc(v_ref_2145_);
v___x_2176_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2176_, 0, v_ref_2145_);
lean_ctor_set(v___x_2176_, 1, v___x_2175_);
v___x_2177_ = l_Lean_PersistentArray_push___redArg(v_traces_2165_, v___x_2176_);
if (v_isShared_2168_ == 0)
{
lean_ctor_set(v___x_2167_, 0, v___x_2177_);
v___x_2179_ = v___x_2167_;
goto v_reusejp_2178_;
}
else
{
lean_object* v_reuseFailAlloc_2188_; 
v_reuseFailAlloc_2188_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2188_, 0, v___x_2177_);
lean_ctor_set_uint64(v_reuseFailAlloc_2188_, sizeof(void*)*1, v_tid_2164_);
v___x_2179_ = v_reuseFailAlloc_2188_;
goto v_reusejp_2178_;
}
v_reusejp_2178_:
{
lean_object* v___x_2181_; 
if (v_isShared_2163_ == 0)
{
lean_ctor_set(v___x_2162_, 4, v___x_2179_);
v___x_2181_ = v___x_2162_;
goto v_reusejp_2180_;
}
else
{
lean_object* v_reuseFailAlloc_2187_; 
v_reuseFailAlloc_2187_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2187_, 0, v_env_2153_);
lean_ctor_set(v_reuseFailAlloc_2187_, 1, v_nextMacroScope_2154_);
lean_ctor_set(v_reuseFailAlloc_2187_, 2, v_ngen_2155_);
lean_ctor_set(v_reuseFailAlloc_2187_, 3, v_auxDeclNGen_2156_);
lean_ctor_set(v_reuseFailAlloc_2187_, 4, v___x_2179_);
lean_ctor_set(v_reuseFailAlloc_2187_, 5, v_cache_2157_);
lean_ctor_set(v_reuseFailAlloc_2187_, 6, v_messages_2158_);
lean_ctor_set(v_reuseFailAlloc_2187_, 7, v_infoState_2159_);
lean_ctor_set(v_reuseFailAlloc_2187_, 8, v_snapshotTasks_2160_);
v___x_2181_ = v_reuseFailAlloc_2187_;
goto v_reusejp_2180_;
}
v_reusejp_2180_:
{
lean_object* v___x_2182_; lean_object* v___x_2183_; lean_object* v___x_2185_; 
v___x_2182_ = lean_st_ref_set(v___y_2143_, v___x_2181_);
v___x_2183_ = lean_box(0);
if (v_isShared_2150_ == 0)
{
lean_ctor_set(v___x_2149_, 0, v___x_2183_);
v___x_2185_ = v___x_2149_;
goto v_reusejp_2184_;
}
else
{
lean_object* v_reuseFailAlloc_2186_; 
v_reuseFailAlloc_2186_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2186_, 0, v___x_2183_);
v___x_2185_ = v_reuseFailAlloc_2186_;
goto v_reusejp_2184_;
}
v_reusejp_2184_:
{
return v___x_2185_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg___boxed(lean_object* v_cls_2192_, lean_object* v_msg_2193_, lean_object* v___y_2194_, lean_object* v___y_2195_, lean_object* v___y_2196_, lean_object* v___y_2197_, lean_object* v___y_2198_){
_start:
{
lean_object* v_res_2199_; 
v_res_2199_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg(v_cls_2192_, v_msg_2193_, v___y_2194_, v___y_2195_, v___y_2196_, v___y_2197_);
lean_dec(v___y_2197_);
lean_dec_ref(v___y_2196_);
lean_dec(v___y_2195_);
lean_dec_ref(v___y_2194_);
return v_res_2199_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__5(lean_object* v_cls_2200_, lean_object* v_msg_2201_, lean_object* v___y_2202_, lean_object* v___y_2203_, lean_object* v___y_2204_, lean_object* v___y_2205_){
_start:
{
lean_object* v_ref_2207_; lean_object* v___x_2208_; lean_object* v_a_2209_; lean_object* v___x_2211_; uint8_t v_isShared_2212_; uint8_t v_isSharedCheck_2253_; 
v_ref_2207_ = lean_ctor_get(v___y_2204_, 5);
v___x_2208_ = lp_batteries_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__1_spec__2(v_msg_2201_, v___y_2202_, v___y_2203_, v___y_2204_, v___y_2205_);
v_a_2209_ = lean_ctor_get(v___x_2208_, 0);
v_isSharedCheck_2253_ = !lean_is_exclusive(v___x_2208_);
if (v_isSharedCheck_2253_ == 0)
{
v___x_2211_ = v___x_2208_;
v_isShared_2212_ = v_isSharedCheck_2253_;
goto v_resetjp_2210_;
}
else
{
lean_inc(v_a_2209_);
lean_dec(v___x_2208_);
v___x_2211_ = lean_box(0);
v_isShared_2212_ = v_isSharedCheck_2253_;
goto v_resetjp_2210_;
}
v_resetjp_2210_:
{
lean_object* v___x_2213_; lean_object* v_traceState_2214_; lean_object* v_env_2215_; lean_object* v_nextMacroScope_2216_; lean_object* v_ngen_2217_; lean_object* v_auxDeclNGen_2218_; lean_object* v_cache_2219_; lean_object* v_messages_2220_; lean_object* v_infoState_2221_; lean_object* v_snapshotTasks_2222_; lean_object* v___x_2224_; uint8_t v_isShared_2225_; uint8_t v_isSharedCheck_2252_; 
v___x_2213_ = lean_st_ref_take(v___y_2205_);
v_traceState_2214_ = lean_ctor_get(v___x_2213_, 4);
v_env_2215_ = lean_ctor_get(v___x_2213_, 0);
v_nextMacroScope_2216_ = lean_ctor_get(v___x_2213_, 1);
v_ngen_2217_ = lean_ctor_get(v___x_2213_, 2);
v_auxDeclNGen_2218_ = lean_ctor_get(v___x_2213_, 3);
v_cache_2219_ = lean_ctor_get(v___x_2213_, 5);
v_messages_2220_ = lean_ctor_get(v___x_2213_, 6);
v_infoState_2221_ = lean_ctor_get(v___x_2213_, 7);
v_snapshotTasks_2222_ = lean_ctor_get(v___x_2213_, 8);
v_isSharedCheck_2252_ = !lean_is_exclusive(v___x_2213_);
if (v_isSharedCheck_2252_ == 0)
{
v___x_2224_ = v___x_2213_;
v_isShared_2225_ = v_isSharedCheck_2252_;
goto v_resetjp_2223_;
}
else
{
lean_inc(v_snapshotTasks_2222_);
lean_inc(v_infoState_2221_);
lean_inc(v_messages_2220_);
lean_inc(v_cache_2219_);
lean_inc(v_traceState_2214_);
lean_inc(v_auxDeclNGen_2218_);
lean_inc(v_ngen_2217_);
lean_inc(v_nextMacroScope_2216_);
lean_inc(v_env_2215_);
lean_dec(v___x_2213_);
v___x_2224_ = lean_box(0);
v_isShared_2225_ = v_isSharedCheck_2252_;
goto v_resetjp_2223_;
}
v_resetjp_2223_:
{
uint64_t v_tid_2226_; lean_object* v_traces_2227_; lean_object* v___x_2229_; uint8_t v_isShared_2230_; uint8_t v_isSharedCheck_2251_; 
v_tid_2226_ = lean_ctor_get_uint64(v_traceState_2214_, sizeof(void*)*1);
v_traces_2227_ = lean_ctor_get(v_traceState_2214_, 0);
v_isSharedCheck_2251_ = !lean_is_exclusive(v_traceState_2214_);
if (v_isSharedCheck_2251_ == 0)
{
v___x_2229_ = v_traceState_2214_;
v_isShared_2230_ = v_isSharedCheck_2251_;
goto v_resetjp_2228_;
}
else
{
lean_inc(v_traces_2227_);
lean_dec(v_traceState_2214_);
v___x_2229_ = lean_box(0);
v_isShared_2230_ = v_isSharedCheck_2251_;
goto v_resetjp_2228_;
}
v_resetjp_2228_:
{
lean_object* v___x_2231_; double v___x_2232_; uint8_t v___x_2233_; lean_object* v___x_2234_; lean_object* v___x_2235_; lean_object* v___x_2236_; lean_object* v___x_2237_; lean_object* v___x_2238_; lean_object* v___x_2239_; lean_object* v___x_2241_; 
v___x_2231_ = lean_box(0);
v___x_2232_ = lean_float_once(&lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg___closed__0, &lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg___closed__0_once, _init_lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg___closed__0);
v___x_2233_ = 0;
v___x_2234_ = ((lean_object*)(lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg___closed__1));
v___x_2235_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_2235_, 0, v_cls_2200_);
lean_ctor_set(v___x_2235_, 1, v___x_2231_);
lean_ctor_set(v___x_2235_, 2, v___x_2234_);
lean_ctor_set_float(v___x_2235_, sizeof(void*)*3, v___x_2232_);
lean_ctor_set_float(v___x_2235_, sizeof(void*)*3 + 8, v___x_2232_);
lean_ctor_set_uint8(v___x_2235_, sizeof(void*)*3 + 16, v___x_2233_);
v___x_2236_ = ((lean_object*)(lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg___closed__2));
v___x_2237_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_2237_, 0, v___x_2235_);
lean_ctor_set(v___x_2237_, 1, v_a_2209_);
lean_ctor_set(v___x_2237_, 2, v___x_2236_);
lean_inc(v_ref_2207_);
v___x_2238_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2238_, 0, v_ref_2207_);
lean_ctor_set(v___x_2238_, 1, v___x_2237_);
v___x_2239_ = l_Lean_PersistentArray_push___redArg(v_traces_2227_, v___x_2238_);
if (v_isShared_2230_ == 0)
{
lean_ctor_set(v___x_2229_, 0, v___x_2239_);
v___x_2241_ = v___x_2229_;
goto v_reusejp_2240_;
}
else
{
lean_object* v_reuseFailAlloc_2250_; 
v_reuseFailAlloc_2250_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2250_, 0, v___x_2239_);
lean_ctor_set_uint64(v_reuseFailAlloc_2250_, sizeof(void*)*1, v_tid_2226_);
v___x_2241_ = v_reuseFailAlloc_2250_;
goto v_reusejp_2240_;
}
v_reusejp_2240_:
{
lean_object* v___x_2243_; 
if (v_isShared_2225_ == 0)
{
lean_ctor_set(v___x_2224_, 4, v___x_2241_);
v___x_2243_ = v___x_2224_;
goto v_reusejp_2242_;
}
else
{
lean_object* v_reuseFailAlloc_2249_; 
v_reuseFailAlloc_2249_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2249_, 0, v_env_2215_);
lean_ctor_set(v_reuseFailAlloc_2249_, 1, v_nextMacroScope_2216_);
lean_ctor_set(v_reuseFailAlloc_2249_, 2, v_ngen_2217_);
lean_ctor_set(v_reuseFailAlloc_2249_, 3, v_auxDeclNGen_2218_);
lean_ctor_set(v_reuseFailAlloc_2249_, 4, v___x_2241_);
lean_ctor_set(v_reuseFailAlloc_2249_, 5, v_cache_2219_);
lean_ctor_set(v_reuseFailAlloc_2249_, 6, v_messages_2220_);
lean_ctor_set(v_reuseFailAlloc_2249_, 7, v_infoState_2221_);
lean_ctor_set(v_reuseFailAlloc_2249_, 8, v_snapshotTasks_2222_);
v___x_2243_ = v_reuseFailAlloc_2249_;
goto v_reusejp_2242_;
}
v_reusejp_2242_:
{
lean_object* v___x_2244_; lean_object* v___x_2245_; lean_object* v___x_2247_; 
v___x_2244_ = lean_st_ref_set(v___y_2205_, v___x_2243_);
v___x_2245_ = lean_box(0);
if (v_isShared_2212_ == 0)
{
lean_ctor_set(v___x_2211_, 0, v___x_2245_);
v___x_2247_ = v___x_2211_;
goto v_reusejp_2246_;
}
else
{
lean_object* v_reuseFailAlloc_2248_; 
v_reuseFailAlloc_2248_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2248_, 0, v___x_2245_);
v___x_2247_ = v_reuseFailAlloc_2248_;
goto v_reusejp_2246_;
}
v_reusejp_2246_:
{
return v___x_2247_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__5___boxed(lean_object* v_cls_2254_, lean_object* v_msg_2255_, lean_object* v___y_2256_, lean_object* v___y_2257_, lean_object* v___y_2258_, lean_object* v___y_2259_, lean_object* v___y_2260_){
_start:
{
lean_object* v_res_2261_; 
v_res_2261_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__5(v_cls_2254_, v_msg_2255_, v___y_2256_, v___y_2257_, v___y_2258_, v___y_2259_);
lean_dec(v___y_2259_);
lean_dec_ref(v___y_2258_);
lean_dec(v___y_2257_);
lean_dec_ref(v___y_2256_);
return v_res_2261_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7_spec__13_spec__15___redArg(lean_object* v_x_2262_, lean_object* v_x_2263_, lean_object* v_x_2264_, lean_object* v_x_2265_){
_start:
{
lean_object* v_ks_2266_; lean_object* v_vs_2267_; lean_object* v___x_2269_; uint8_t v_isShared_2270_; uint8_t v_isSharedCheck_2291_; 
v_ks_2266_ = lean_ctor_get(v_x_2262_, 0);
v_vs_2267_ = lean_ctor_get(v_x_2262_, 1);
v_isSharedCheck_2291_ = !lean_is_exclusive(v_x_2262_);
if (v_isSharedCheck_2291_ == 0)
{
v___x_2269_ = v_x_2262_;
v_isShared_2270_ = v_isSharedCheck_2291_;
goto v_resetjp_2268_;
}
else
{
lean_inc(v_vs_2267_);
lean_inc(v_ks_2266_);
lean_dec(v_x_2262_);
v___x_2269_ = lean_box(0);
v_isShared_2270_ = v_isSharedCheck_2291_;
goto v_resetjp_2268_;
}
v_resetjp_2268_:
{
lean_object* v___x_2271_; uint8_t v___x_2272_; 
v___x_2271_ = lean_array_get_size(v_ks_2266_);
v___x_2272_ = lean_nat_dec_lt(v_x_2263_, v___x_2271_);
if (v___x_2272_ == 0)
{
lean_object* v___x_2273_; lean_object* v___x_2274_; lean_object* v___x_2276_; 
lean_dec(v_x_2263_);
v___x_2273_ = lean_array_push(v_ks_2266_, v_x_2264_);
v___x_2274_ = lean_array_push(v_vs_2267_, v_x_2265_);
if (v_isShared_2270_ == 0)
{
lean_ctor_set(v___x_2269_, 1, v___x_2274_);
lean_ctor_set(v___x_2269_, 0, v___x_2273_);
v___x_2276_ = v___x_2269_;
goto v_reusejp_2275_;
}
else
{
lean_object* v_reuseFailAlloc_2277_; 
v_reuseFailAlloc_2277_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2277_, 0, v___x_2273_);
lean_ctor_set(v_reuseFailAlloc_2277_, 1, v___x_2274_);
v___x_2276_ = v_reuseFailAlloc_2277_;
goto v_reusejp_2275_;
}
v_reusejp_2275_:
{
return v___x_2276_;
}
}
else
{
lean_object* v_k_x27_2278_; uint8_t v___x_2279_; 
v_k_x27_2278_ = lean_array_fget_borrowed(v_ks_2266_, v_x_2263_);
v___x_2279_ = l_Lean_instBEqMVarId_beq(v_x_2264_, v_k_x27_2278_);
if (v___x_2279_ == 0)
{
lean_object* v___x_2281_; 
if (v_isShared_2270_ == 0)
{
v___x_2281_ = v___x_2269_;
goto v_reusejp_2280_;
}
else
{
lean_object* v_reuseFailAlloc_2285_; 
v_reuseFailAlloc_2285_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2285_, 0, v_ks_2266_);
lean_ctor_set(v_reuseFailAlloc_2285_, 1, v_vs_2267_);
v___x_2281_ = v_reuseFailAlloc_2285_;
goto v_reusejp_2280_;
}
v_reusejp_2280_:
{
lean_object* v___x_2282_; lean_object* v___x_2283_; 
v___x_2282_ = lean_unsigned_to_nat(1u);
v___x_2283_ = lean_nat_add(v_x_2263_, v___x_2282_);
lean_dec(v_x_2263_);
v_x_2262_ = v___x_2281_;
v_x_2263_ = v___x_2283_;
goto _start;
}
}
else
{
lean_object* v___x_2286_; lean_object* v___x_2287_; lean_object* v___x_2289_; 
v___x_2286_ = lean_array_fset(v_ks_2266_, v_x_2263_, v_x_2264_);
v___x_2287_ = lean_array_fset(v_vs_2267_, v_x_2263_, v_x_2265_);
lean_dec(v_x_2263_);
if (v_isShared_2270_ == 0)
{
lean_ctor_set(v___x_2269_, 1, v___x_2287_);
lean_ctor_set(v___x_2269_, 0, v___x_2286_);
v___x_2289_ = v___x_2269_;
goto v_reusejp_2288_;
}
else
{
lean_object* v_reuseFailAlloc_2290_; 
v_reuseFailAlloc_2290_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2290_, 0, v___x_2286_);
lean_ctor_set(v_reuseFailAlloc_2290_, 1, v___x_2287_);
v___x_2289_ = v_reuseFailAlloc_2290_;
goto v_reusejp_2288_;
}
v_reusejp_2288_:
{
return v___x_2289_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7_spec__13___redArg(lean_object* v_n_2292_, lean_object* v_k_2293_, lean_object* v_v_2294_){
_start:
{
lean_object* v___x_2295_; lean_object* v___x_2296_; 
v___x_2295_ = lean_unsigned_to_nat(0u);
v___x_2296_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7_spec__13_spec__15___redArg(v_n_2292_, v___x_2295_, v_k_2293_, v_v_2294_);
return v___x_2296_;
}
}
static lean_object* _init_lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7___redArg___closed__0(void){
_start:
{
lean_object* v___x_2297_; 
v___x_2297_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_2297_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7___redArg(lean_object* v_x_2298_, size_t v_x_2299_, size_t v_x_2300_, lean_object* v_x_2301_, lean_object* v_x_2302_){
_start:
{
if (lean_obj_tag(v_x_2298_) == 0)
{
lean_object* v_es_2303_; size_t v___x_2304_; size_t v___x_2305_; lean_object* v_j_2306_; lean_object* v___x_2307_; uint8_t v___x_2308_; 
v_es_2303_ = lean_ctor_get(v_x_2298_, 0);
v___x_2304_ = ((size_t)31ULL);
v___x_2305_ = lean_usize_land(v_x_2299_, v___x_2304_);
v_j_2306_ = lean_usize_to_nat(v___x_2305_);
v___x_2307_ = lean_array_get_size(v_es_2303_);
v___x_2308_ = lean_nat_dec_lt(v_j_2306_, v___x_2307_);
if (v___x_2308_ == 0)
{
lean_dec(v_j_2306_);
lean_dec(v_x_2302_);
lean_dec(v_x_2301_);
return v_x_2298_;
}
else
{
lean_object* v___x_2310_; uint8_t v_isShared_2311_; uint8_t v_isSharedCheck_2347_; 
lean_inc_ref(v_es_2303_);
v_isSharedCheck_2347_ = !lean_is_exclusive(v_x_2298_);
if (v_isSharedCheck_2347_ == 0)
{
lean_object* v_unused_2348_; 
v_unused_2348_ = lean_ctor_get(v_x_2298_, 0);
lean_dec(v_unused_2348_);
v___x_2310_ = v_x_2298_;
v_isShared_2311_ = v_isSharedCheck_2347_;
goto v_resetjp_2309_;
}
else
{
lean_dec(v_x_2298_);
v___x_2310_ = lean_box(0);
v_isShared_2311_ = v_isSharedCheck_2347_;
goto v_resetjp_2309_;
}
v_resetjp_2309_:
{
lean_object* v_v_2312_; lean_object* v___x_2313_; lean_object* v_xs_x27_2314_; lean_object* v___y_2316_; 
v_v_2312_ = lean_array_fget(v_es_2303_, v_j_2306_);
v___x_2313_ = lean_box(0);
v_xs_x27_2314_ = lean_array_fset(v_es_2303_, v_j_2306_, v___x_2313_);
switch(lean_obj_tag(v_v_2312_))
{
case 0:
{
lean_object* v_key_2321_; lean_object* v_val_2322_; lean_object* v___x_2324_; uint8_t v_isShared_2325_; uint8_t v_isSharedCheck_2332_; 
v_key_2321_ = lean_ctor_get(v_v_2312_, 0);
v_val_2322_ = lean_ctor_get(v_v_2312_, 1);
v_isSharedCheck_2332_ = !lean_is_exclusive(v_v_2312_);
if (v_isSharedCheck_2332_ == 0)
{
v___x_2324_ = v_v_2312_;
v_isShared_2325_ = v_isSharedCheck_2332_;
goto v_resetjp_2323_;
}
else
{
lean_inc(v_val_2322_);
lean_inc(v_key_2321_);
lean_dec(v_v_2312_);
v___x_2324_ = lean_box(0);
v_isShared_2325_ = v_isSharedCheck_2332_;
goto v_resetjp_2323_;
}
v_resetjp_2323_:
{
uint8_t v___x_2326_; 
v___x_2326_ = l_Lean_instBEqMVarId_beq(v_x_2301_, v_key_2321_);
if (v___x_2326_ == 0)
{
lean_object* v___x_2327_; lean_object* v___x_2328_; 
lean_del_object(v___x_2324_);
v___x_2327_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_2321_, v_val_2322_, v_x_2301_, v_x_2302_);
v___x_2328_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2328_, 0, v___x_2327_);
v___y_2316_ = v___x_2328_;
goto v___jp_2315_;
}
else
{
lean_object* v___x_2330_; 
lean_dec(v_val_2322_);
lean_dec(v_key_2321_);
if (v_isShared_2325_ == 0)
{
lean_ctor_set(v___x_2324_, 1, v_x_2302_);
lean_ctor_set(v___x_2324_, 0, v_x_2301_);
v___x_2330_ = v___x_2324_;
goto v_reusejp_2329_;
}
else
{
lean_object* v_reuseFailAlloc_2331_; 
v_reuseFailAlloc_2331_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2331_, 0, v_x_2301_);
lean_ctor_set(v_reuseFailAlloc_2331_, 1, v_x_2302_);
v___x_2330_ = v_reuseFailAlloc_2331_;
goto v_reusejp_2329_;
}
v_reusejp_2329_:
{
v___y_2316_ = v___x_2330_;
goto v___jp_2315_;
}
}
}
}
case 1:
{
lean_object* v_node_2333_; lean_object* v___x_2335_; uint8_t v_isShared_2336_; uint8_t v_isSharedCheck_2345_; 
v_node_2333_ = lean_ctor_get(v_v_2312_, 0);
v_isSharedCheck_2345_ = !lean_is_exclusive(v_v_2312_);
if (v_isSharedCheck_2345_ == 0)
{
v___x_2335_ = v_v_2312_;
v_isShared_2336_ = v_isSharedCheck_2345_;
goto v_resetjp_2334_;
}
else
{
lean_inc(v_node_2333_);
lean_dec(v_v_2312_);
v___x_2335_ = lean_box(0);
v_isShared_2336_ = v_isSharedCheck_2345_;
goto v_resetjp_2334_;
}
v_resetjp_2334_:
{
size_t v___x_2337_; size_t v___x_2338_; size_t v___x_2339_; size_t v___x_2340_; lean_object* v___x_2341_; lean_object* v___x_2343_; 
v___x_2337_ = ((size_t)5ULL);
v___x_2338_ = lean_usize_shift_right(v_x_2299_, v___x_2337_);
v___x_2339_ = ((size_t)1ULL);
v___x_2340_ = lean_usize_add(v_x_2300_, v___x_2339_);
v___x_2341_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7___redArg(v_node_2333_, v___x_2338_, v___x_2340_, v_x_2301_, v_x_2302_);
if (v_isShared_2336_ == 0)
{
lean_ctor_set(v___x_2335_, 0, v___x_2341_);
v___x_2343_ = v___x_2335_;
goto v_reusejp_2342_;
}
else
{
lean_object* v_reuseFailAlloc_2344_; 
v_reuseFailAlloc_2344_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2344_, 0, v___x_2341_);
v___x_2343_ = v_reuseFailAlloc_2344_;
goto v_reusejp_2342_;
}
v_reusejp_2342_:
{
v___y_2316_ = v___x_2343_;
goto v___jp_2315_;
}
}
}
default: 
{
lean_object* v___x_2346_; 
v___x_2346_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2346_, 0, v_x_2301_);
lean_ctor_set(v___x_2346_, 1, v_x_2302_);
v___y_2316_ = v___x_2346_;
goto v___jp_2315_;
}
}
v___jp_2315_:
{
lean_object* v___x_2317_; lean_object* v___x_2319_; 
v___x_2317_ = lean_array_fset(v_xs_x27_2314_, v_j_2306_, v___y_2316_);
lean_dec(v_j_2306_);
if (v_isShared_2311_ == 0)
{
lean_ctor_set(v___x_2310_, 0, v___x_2317_);
v___x_2319_ = v___x_2310_;
goto v_reusejp_2318_;
}
else
{
lean_object* v_reuseFailAlloc_2320_; 
v_reuseFailAlloc_2320_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2320_, 0, v___x_2317_);
v___x_2319_ = v_reuseFailAlloc_2320_;
goto v_reusejp_2318_;
}
v_reusejp_2318_:
{
return v___x_2319_;
}
}
}
}
}
else
{
lean_object* v_ks_2349_; lean_object* v_vs_2350_; lean_object* v___x_2352_; uint8_t v_isShared_2353_; uint8_t v_isSharedCheck_2370_; 
v_ks_2349_ = lean_ctor_get(v_x_2298_, 0);
v_vs_2350_ = lean_ctor_get(v_x_2298_, 1);
v_isSharedCheck_2370_ = !lean_is_exclusive(v_x_2298_);
if (v_isSharedCheck_2370_ == 0)
{
v___x_2352_ = v_x_2298_;
v_isShared_2353_ = v_isSharedCheck_2370_;
goto v_resetjp_2351_;
}
else
{
lean_inc(v_vs_2350_);
lean_inc(v_ks_2349_);
lean_dec(v_x_2298_);
v___x_2352_ = lean_box(0);
v_isShared_2353_ = v_isSharedCheck_2370_;
goto v_resetjp_2351_;
}
v_resetjp_2351_:
{
lean_object* v___x_2355_; 
if (v_isShared_2353_ == 0)
{
v___x_2355_ = v___x_2352_;
goto v_reusejp_2354_;
}
else
{
lean_object* v_reuseFailAlloc_2369_; 
v_reuseFailAlloc_2369_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2369_, 0, v_ks_2349_);
lean_ctor_set(v_reuseFailAlloc_2369_, 1, v_vs_2350_);
v___x_2355_ = v_reuseFailAlloc_2369_;
goto v_reusejp_2354_;
}
v_reusejp_2354_:
{
lean_object* v_newNode_2356_; uint8_t v___y_2358_; size_t v___x_2364_; uint8_t v___x_2365_; 
v_newNode_2356_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7_spec__13___redArg(v___x_2355_, v_x_2301_, v_x_2302_);
v___x_2364_ = ((size_t)7ULL);
v___x_2365_ = lean_usize_dec_le(v___x_2364_, v_x_2300_);
if (v___x_2365_ == 0)
{
lean_object* v___x_2366_; lean_object* v___x_2367_; uint8_t v___x_2368_; 
v___x_2366_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_2356_);
v___x_2367_ = lean_unsigned_to_nat(4u);
v___x_2368_ = lean_nat_dec_lt(v___x_2366_, v___x_2367_);
lean_dec(v___x_2366_);
v___y_2358_ = v___x_2368_;
goto v___jp_2357_;
}
else
{
v___y_2358_ = v___x_2365_;
goto v___jp_2357_;
}
v___jp_2357_:
{
if (v___y_2358_ == 0)
{
lean_object* v_ks_2359_; lean_object* v_vs_2360_; lean_object* v___x_2361_; lean_object* v___x_2362_; lean_object* v___x_2363_; 
v_ks_2359_ = lean_ctor_get(v_newNode_2356_, 0);
lean_inc_ref(v_ks_2359_);
v_vs_2360_ = lean_ctor_get(v_newNode_2356_, 1);
lean_inc_ref(v_vs_2360_);
lean_dec_ref(v_newNode_2356_);
v___x_2361_ = lean_unsigned_to_nat(0u);
v___x_2362_ = lean_obj_once(&lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7___redArg___closed__0, &lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7___redArg___closed__0_once, _init_lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7___redArg___closed__0);
v___x_2363_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7_spec__14___redArg(v_x_2300_, v_ks_2359_, v_vs_2360_, v___x_2361_, v___x_2362_);
lean_dec_ref(v_vs_2360_);
lean_dec_ref(v_ks_2359_);
return v___x_2363_;
}
else
{
return v_newNode_2356_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7_spec__14___redArg(size_t v_depth_2371_, lean_object* v_keys_2372_, lean_object* v_vals_2373_, lean_object* v_i_2374_, lean_object* v_entries_2375_){
_start:
{
lean_object* v___x_2376_; uint8_t v___x_2377_; 
v___x_2376_ = lean_array_get_size(v_keys_2372_);
v___x_2377_ = lean_nat_dec_lt(v_i_2374_, v___x_2376_);
if (v___x_2377_ == 0)
{
lean_dec(v_i_2374_);
return v_entries_2375_;
}
else
{
lean_object* v_k_2378_; lean_object* v_v_2379_; uint64_t v___x_2380_; size_t v_h_2381_; size_t v___x_2382_; lean_object* v___x_2383_; size_t v___x_2384_; size_t v___x_2385_; size_t v___x_2386_; size_t v_h_2387_; lean_object* v___x_2388_; lean_object* v___x_2389_; 
v_k_2378_ = lean_array_fget_borrowed(v_keys_2372_, v_i_2374_);
v_v_2379_ = lean_array_fget_borrowed(v_vals_2373_, v_i_2374_);
v___x_2380_ = l_Lean_instHashableMVarId_hash(v_k_2378_);
v_h_2381_ = lean_uint64_to_usize(v___x_2380_);
v___x_2382_ = ((size_t)5ULL);
v___x_2383_ = lean_unsigned_to_nat(1u);
v___x_2384_ = ((size_t)1ULL);
v___x_2385_ = lean_usize_sub(v_depth_2371_, v___x_2384_);
v___x_2386_ = lean_usize_mul(v___x_2382_, v___x_2385_);
v_h_2387_ = lean_usize_shift_right(v_h_2381_, v___x_2386_);
v___x_2388_ = lean_nat_add(v_i_2374_, v___x_2383_);
lean_dec(v_i_2374_);
lean_inc(v_v_2379_);
lean_inc(v_k_2378_);
v___x_2389_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7___redArg(v_entries_2375_, v_h_2387_, v_depth_2371_, v_k_2378_, v_v_2379_);
v_i_2374_ = v___x_2388_;
v_entries_2375_ = v___x_2389_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7_spec__14___redArg___boxed(lean_object* v_depth_2391_, lean_object* v_keys_2392_, lean_object* v_vals_2393_, lean_object* v_i_2394_, lean_object* v_entries_2395_){
_start:
{
size_t v_depth_boxed_2396_; lean_object* v_res_2397_; 
v_depth_boxed_2396_ = lean_unbox_usize(v_depth_2391_);
lean_dec(v_depth_2391_);
v_res_2397_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7_spec__14___redArg(v_depth_boxed_2396_, v_keys_2392_, v_vals_2393_, v_i_2394_, v_entries_2395_);
lean_dec_ref(v_vals_2393_);
lean_dec_ref(v_keys_2392_);
return v_res_2397_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7___redArg___boxed(lean_object* v_x_2398_, lean_object* v_x_2399_, lean_object* v_x_2400_, lean_object* v_x_2401_, lean_object* v_x_2402_){
_start:
{
size_t v_x_92167__boxed_2403_; size_t v_x_92168__boxed_2404_; lean_object* v_res_2405_; 
v_x_92167__boxed_2403_ = lean_unbox_usize(v_x_2399_);
lean_dec(v_x_2399_);
v_x_92168__boxed_2404_ = lean_unbox_usize(v_x_2400_);
lean_dec(v_x_2400_);
v_res_2405_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7___redArg(v_x_2398_, v_x_92167__boxed_2403_, v_x_92168__boxed_2404_, v_x_2401_, v_x_2402_);
return v_res_2405_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6___redArg(lean_object* v_x_2406_, lean_object* v_x_2407_, lean_object* v_x_2408_){
_start:
{
uint64_t v___x_2409_; size_t v___x_2410_; size_t v___x_2411_; lean_object* v___x_2412_; 
v___x_2409_ = l_Lean_instHashableMVarId_hash(v_x_2407_);
v___x_2410_ = lean_uint64_to_usize(v___x_2409_);
v___x_2411_ = ((size_t)1ULL);
v___x_2412_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7___redArg(v_x_2406_, v___x_2410_, v___x_2411_, v_x_2407_, v_x_2408_);
return v___x_2412_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4___redArg(lean_object* v_mvarId_2413_, lean_object* v_val_2414_, lean_object* v___y_2415_){
_start:
{
lean_object* v___x_2417_; lean_object* v_mctx_2418_; lean_object* v_cache_2419_; lean_object* v_zetaDeltaFVarIds_2420_; lean_object* v_postponed_2421_; lean_object* v_diag_2422_; lean_object* v___x_2424_; uint8_t v_isShared_2425_; uint8_t v_isSharedCheck_2450_; 
v___x_2417_ = lean_st_ref_take(v___y_2415_);
v_mctx_2418_ = lean_ctor_get(v___x_2417_, 0);
v_cache_2419_ = lean_ctor_get(v___x_2417_, 1);
v_zetaDeltaFVarIds_2420_ = lean_ctor_get(v___x_2417_, 2);
v_postponed_2421_ = lean_ctor_get(v___x_2417_, 3);
v_diag_2422_ = lean_ctor_get(v___x_2417_, 4);
v_isSharedCheck_2450_ = !lean_is_exclusive(v___x_2417_);
if (v_isSharedCheck_2450_ == 0)
{
v___x_2424_ = v___x_2417_;
v_isShared_2425_ = v_isSharedCheck_2450_;
goto v_resetjp_2423_;
}
else
{
lean_inc(v_diag_2422_);
lean_inc(v_postponed_2421_);
lean_inc(v_zetaDeltaFVarIds_2420_);
lean_inc(v_cache_2419_);
lean_inc(v_mctx_2418_);
lean_dec(v___x_2417_);
v___x_2424_ = lean_box(0);
v_isShared_2425_ = v_isSharedCheck_2450_;
goto v_resetjp_2423_;
}
v_resetjp_2423_:
{
lean_object* v_depth_2426_; lean_object* v_levelAssignDepth_2427_; lean_object* v_lmvarCounter_2428_; lean_object* v_mvarCounter_2429_; lean_object* v_lDecls_2430_; lean_object* v_decls_2431_; lean_object* v_userNames_2432_; lean_object* v_lAssignment_2433_; lean_object* v_eAssignment_2434_; lean_object* v_dAssignment_2435_; lean_object* v___x_2437_; uint8_t v_isShared_2438_; uint8_t v_isSharedCheck_2449_; 
v_depth_2426_ = lean_ctor_get(v_mctx_2418_, 0);
v_levelAssignDepth_2427_ = lean_ctor_get(v_mctx_2418_, 1);
v_lmvarCounter_2428_ = lean_ctor_get(v_mctx_2418_, 2);
v_mvarCounter_2429_ = lean_ctor_get(v_mctx_2418_, 3);
v_lDecls_2430_ = lean_ctor_get(v_mctx_2418_, 4);
v_decls_2431_ = lean_ctor_get(v_mctx_2418_, 5);
v_userNames_2432_ = lean_ctor_get(v_mctx_2418_, 6);
v_lAssignment_2433_ = lean_ctor_get(v_mctx_2418_, 7);
v_eAssignment_2434_ = lean_ctor_get(v_mctx_2418_, 8);
v_dAssignment_2435_ = lean_ctor_get(v_mctx_2418_, 9);
v_isSharedCheck_2449_ = !lean_is_exclusive(v_mctx_2418_);
if (v_isSharedCheck_2449_ == 0)
{
v___x_2437_ = v_mctx_2418_;
v_isShared_2438_ = v_isSharedCheck_2449_;
goto v_resetjp_2436_;
}
else
{
lean_inc(v_dAssignment_2435_);
lean_inc(v_eAssignment_2434_);
lean_inc(v_lAssignment_2433_);
lean_inc(v_userNames_2432_);
lean_inc(v_decls_2431_);
lean_inc(v_lDecls_2430_);
lean_inc(v_mvarCounter_2429_);
lean_inc(v_lmvarCounter_2428_);
lean_inc(v_levelAssignDepth_2427_);
lean_inc(v_depth_2426_);
lean_dec(v_mctx_2418_);
v___x_2437_ = lean_box(0);
v_isShared_2438_ = v_isSharedCheck_2449_;
goto v_resetjp_2436_;
}
v_resetjp_2436_:
{
lean_object* v___x_2439_; lean_object* v___x_2441_; 
v___x_2439_ = lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6___redArg(v_eAssignment_2434_, v_mvarId_2413_, v_val_2414_);
if (v_isShared_2438_ == 0)
{
lean_ctor_set(v___x_2437_, 8, v___x_2439_);
v___x_2441_ = v___x_2437_;
goto v_reusejp_2440_;
}
else
{
lean_object* v_reuseFailAlloc_2448_; 
v_reuseFailAlloc_2448_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_2448_, 0, v_depth_2426_);
lean_ctor_set(v_reuseFailAlloc_2448_, 1, v_levelAssignDepth_2427_);
lean_ctor_set(v_reuseFailAlloc_2448_, 2, v_lmvarCounter_2428_);
lean_ctor_set(v_reuseFailAlloc_2448_, 3, v_mvarCounter_2429_);
lean_ctor_set(v_reuseFailAlloc_2448_, 4, v_lDecls_2430_);
lean_ctor_set(v_reuseFailAlloc_2448_, 5, v_decls_2431_);
lean_ctor_set(v_reuseFailAlloc_2448_, 6, v_userNames_2432_);
lean_ctor_set(v_reuseFailAlloc_2448_, 7, v_lAssignment_2433_);
lean_ctor_set(v_reuseFailAlloc_2448_, 8, v___x_2439_);
lean_ctor_set(v_reuseFailAlloc_2448_, 9, v_dAssignment_2435_);
v___x_2441_ = v_reuseFailAlloc_2448_;
goto v_reusejp_2440_;
}
v_reusejp_2440_:
{
lean_object* v___x_2443_; 
if (v_isShared_2425_ == 0)
{
lean_ctor_set(v___x_2424_, 0, v___x_2441_);
v___x_2443_ = v___x_2424_;
goto v_reusejp_2442_;
}
else
{
lean_object* v_reuseFailAlloc_2447_; 
v_reuseFailAlloc_2447_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2447_, 0, v___x_2441_);
lean_ctor_set(v_reuseFailAlloc_2447_, 1, v_cache_2419_);
lean_ctor_set(v_reuseFailAlloc_2447_, 2, v_zetaDeltaFVarIds_2420_);
lean_ctor_set(v_reuseFailAlloc_2447_, 3, v_postponed_2421_);
lean_ctor_set(v_reuseFailAlloc_2447_, 4, v_diag_2422_);
v___x_2443_ = v_reuseFailAlloc_2447_;
goto v_reusejp_2442_;
}
v_reusejp_2442_:
{
lean_object* v___x_2444_; lean_object* v___x_2445_; lean_object* v___x_2446_; 
v___x_2444_ = lean_st_ref_set(v___y_2415_, v___x_2443_);
v___x_2445_ = lean_box(0);
v___x_2446_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2446_, 0, v___x_2445_);
return v___x_2446_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4___redArg___boxed(lean_object* v_mvarId_2451_, lean_object* v_val_2452_, lean_object* v___y_2453_, lean_object* v___y_2454_){
_start:
{
lean_object* v_res_2455_; 
v_res_2455_ = lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4___redArg(v_mvarId_2451_, v_val_2452_, v___y_2453_);
lean_dec(v___y_2453_);
return v_res_2455_;
}
}
LEAN_EXPORT lean_object* lp_batteries_List_mapTR_loop___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__2_spec__3(lean_object* v_a_2456_, lean_object* v_a_2457_){
_start:
{
if (lean_obj_tag(v_a_2456_) == 0)
{
lean_object* v___x_2458_; 
v___x_2458_ = l_List_reverse___redArg(v_a_2457_);
return v___x_2458_;
}
else
{
lean_object* v_head_2459_; lean_object* v_tail_2460_; lean_object* v___x_2462_; uint8_t v_isShared_2463_; uint8_t v_isSharedCheck_2469_; 
v_head_2459_ = lean_ctor_get(v_a_2456_, 0);
v_tail_2460_ = lean_ctor_get(v_a_2456_, 1);
v_isSharedCheck_2469_ = !lean_is_exclusive(v_a_2456_);
if (v_isSharedCheck_2469_ == 0)
{
v___x_2462_ = v_a_2456_;
v_isShared_2463_ = v_isSharedCheck_2469_;
goto v_resetjp_2461_;
}
else
{
lean_inc(v_tail_2460_);
lean_inc(v_head_2459_);
lean_dec(v_a_2456_);
v___x_2462_ = lean_box(0);
v_isShared_2463_ = v_isSharedCheck_2469_;
goto v_resetjp_2461_;
}
v_resetjp_2461_:
{
lean_object* v___x_2464_; lean_object* v___x_2466_; 
v___x_2464_ = l_Lean_mkLevelParam(v_head_2459_);
if (v_isShared_2463_ == 0)
{
lean_ctor_set(v___x_2462_, 1, v_a_2457_);
lean_ctor_set(v___x_2462_, 0, v___x_2464_);
v___x_2466_ = v___x_2462_;
goto v_reusejp_2465_;
}
else
{
lean_object* v_reuseFailAlloc_2468_; 
v_reuseFailAlloc_2468_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2468_, 0, v___x_2464_);
lean_ctor_set(v_reuseFailAlloc_2468_, 1, v_a_2457_);
v___x_2466_ = v_reuseFailAlloc_2468_;
goto v_reusejp_2465_;
}
v_reusejp_2465_:
{
v_a_2456_ = v_tail_2460_;
v_a_2457_ = v___x_2466_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__2_spec__2(lean_object* v_constName_2470_, lean_object* v___y_2471_, lean_object* v___y_2472_, lean_object* v___y_2473_, lean_object* v___y_2474_){
_start:
{
lean_object* v___x_2476_; lean_object* v_env_2477_; uint8_t v___x_2478_; lean_object* v___x_2479_; 
v___x_2476_ = lean_st_ref_get(v___y_2474_);
v_env_2477_ = lean_ctor_get(v___x_2476_, 0);
lean_inc_ref(v_env_2477_);
lean_dec(v___x_2476_);
v___x_2478_ = 0;
lean_inc(v_constName_2470_);
v___x_2479_ = l_Lean_Environment_findConstVal_x3f(v_env_2477_, v_constName_2470_, v___x_2478_);
if (lean_obj_tag(v___x_2479_) == 0)
{
lean_object* v___x_2480_; 
v___x_2480_ = lp_batteries_Lean_throwUnknownConstant___at___00Lean_getConstInfo___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2__spec__0_spec__0___redArg(v_constName_2470_, v___y_2471_, v___y_2472_, v___y_2473_, v___y_2474_);
return v___x_2480_;
}
else
{
lean_object* v_val_2481_; lean_object* v___x_2483_; uint8_t v_isShared_2484_; uint8_t v_isSharedCheck_2488_; 
lean_dec(v_constName_2470_);
v_val_2481_ = lean_ctor_get(v___x_2479_, 0);
v_isSharedCheck_2488_ = !lean_is_exclusive(v___x_2479_);
if (v_isSharedCheck_2488_ == 0)
{
v___x_2483_ = v___x_2479_;
v_isShared_2484_ = v_isSharedCheck_2488_;
goto v_resetjp_2482_;
}
else
{
lean_inc(v_val_2481_);
lean_dec(v___x_2479_);
v___x_2483_ = lean_box(0);
v_isShared_2484_ = v_isSharedCheck_2488_;
goto v_resetjp_2482_;
}
v_resetjp_2482_:
{
lean_object* v___x_2486_; 
if (v_isShared_2484_ == 0)
{
lean_ctor_set_tag(v___x_2483_, 0);
v___x_2486_ = v___x_2483_;
goto v_reusejp_2485_;
}
else
{
lean_object* v_reuseFailAlloc_2487_; 
v_reuseFailAlloc_2487_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2487_, 0, v_val_2481_);
v___x_2486_ = v_reuseFailAlloc_2487_;
goto v_reusejp_2485_;
}
v_reusejp_2485_:
{
return v___x_2486_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__2_spec__2___boxed(lean_object* v_constName_2489_, lean_object* v___y_2490_, lean_object* v___y_2491_, lean_object* v___y_2492_, lean_object* v___y_2493_, lean_object* v___y_2494_){
_start:
{
lean_object* v_res_2495_; 
v_res_2495_ = lp_batteries_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__2_spec__2(v_constName_2489_, v___y_2490_, v___y_2491_, v___y_2492_, v___y_2493_);
lean_dec(v___y_2493_);
lean_dec_ref(v___y_2492_);
lean_dec(v___y_2491_);
lean_dec_ref(v___y_2490_);
return v_res_2495_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkConstWithLevelParams___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__2(lean_object* v_constName_2496_, lean_object* v___y_2497_, lean_object* v___y_2498_, lean_object* v___y_2499_, lean_object* v___y_2500_){
_start:
{
lean_object* v___x_2502_; 
lean_inc(v_constName_2496_);
v___x_2502_ = lp_batteries_Lean_getConstVal___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__2_spec__2(v_constName_2496_, v___y_2497_, v___y_2498_, v___y_2499_, v___y_2500_);
if (lean_obj_tag(v___x_2502_) == 0)
{
lean_object* v_a_2503_; lean_object* v___x_2505_; uint8_t v_isShared_2506_; uint8_t v_isSharedCheck_2514_; 
v_a_2503_ = lean_ctor_get(v___x_2502_, 0);
v_isSharedCheck_2514_ = !lean_is_exclusive(v___x_2502_);
if (v_isSharedCheck_2514_ == 0)
{
v___x_2505_ = v___x_2502_;
v_isShared_2506_ = v_isSharedCheck_2514_;
goto v_resetjp_2504_;
}
else
{
lean_inc(v_a_2503_);
lean_dec(v___x_2502_);
v___x_2505_ = lean_box(0);
v_isShared_2506_ = v_isSharedCheck_2514_;
goto v_resetjp_2504_;
}
v_resetjp_2504_:
{
lean_object* v_levelParams_2507_; lean_object* v___x_2508_; lean_object* v___x_2509_; lean_object* v___x_2510_; lean_object* v___x_2512_; 
v_levelParams_2507_ = lean_ctor_get(v_a_2503_, 1);
lean_inc(v_levelParams_2507_);
lean_dec(v_a_2503_);
v___x_2508_ = lean_box(0);
v___x_2509_ = lp_batteries_List_mapTR_loop___at___00Lean_mkConstWithLevelParams___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__2_spec__3(v_levelParams_2507_, v___x_2508_);
v___x_2510_ = l_Lean_mkConst(v_constName_2496_, v___x_2509_);
if (v_isShared_2506_ == 0)
{
lean_ctor_set(v___x_2505_, 0, v___x_2510_);
v___x_2512_ = v___x_2505_;
goto v_reusejp_2511_;
}
else
{
lean_object* v_reuseFailAlloc_2513_; 
v_reuseFailAlloc_2513_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2513_, 0, v___x_2510_);
v___x_2512_ = v_reuseFailAlloc_2513_;
goto v_reusejp_2511_;
}
v_reusejp_2511_:
{
return v___x_2512_;
}
}
}
else
{
lean_object* v_a_2515_; lean_object* v___x_2517_; uint8_t v_isShared_2518_; uint8_t v_isSharedCheck_2522_; 
lean_dec(v_constName_2496_);
v_a_2515_ = lean_ctor_get(v___x_2502_, 0);
v_isSharedCheck_2522_ = !lean_is_exclusive(v___x_2502_);
if (v_isSharedCheck_2522_ == 0)
{
v___x_2517_ = v___x_2502_;
v_isShared_2518_ = v_isSharedCheck_2522_;
goto v_resetjp_2516_;
}
else
{
lean_inc(v_a_2515_);
lean_dec(v___x_2502_);
v___x_2517_ = lean_box(0);
v_isShared_2518_ = v_isSharedCheck_2522_;
goto v_resetjp_2516_;
}
v_resetjp_2516_:
{
lean_object* v___x_2520_; 
if (v_isShared_2518_ == 0)
{
v___x_2520_ = v___x_2517_;
goto v_reusejp_2519_;
}
else
{
lean_object* v_reuseFailAlloc_2521_; 
v_reuseFailAlloc_2521_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2521_, 0, v_a_2515_);
v___x_2520_ = v_reuseFailAlloc_2521_;
goto v_reusejp_2519_;
}
v_reusejp_2519_:
{
return v___x_2520_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_mkConstWithLevelParams___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__2___boxed(lean_object* v_constName_2523_, lean_object* v___y_2524_, lean_object* v___y_2525_, lean_object* v___y_2526_, lean_object* v___y_2527_, lean_object* v___y_2528_){
_start:
{
lean_object* v_res_2529_; 
v_res_2529_ = lp_batteries_Lean_mkConstWithLevelParams___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__2(v_constName_2523_, v___y_2524_, v___y_2525_, v___y_2526_, v___y_2527_);
lean_dec(v___y_2527_);
lean_dec_ref(v___y_2526_);
lean_dec(v___y_2525_);
lean_dec_ref(v___y_2524_);
return v_res_2529_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__1(void){
_start:
{
lean_object* v___x_2531_; lean_object* v___x_2532_; 
v___x_2531_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__0));
v___x_2532_ = l_Lean_stringToMessageData(v___x_2531_);
return v___x_2532_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__3(void){
_start:
{
lean_object* v___x_2534_; lean_object* v___x_2535_; 
v___x_2534_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__2));
v___x_2535_ = l_Lean_stringToMessageData(v___x_2534_);
return v___x_2535_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__5(void){
_start:
{
lean_object* v___x_2537_; lean_object* v___x_2538_; 
v___x_2537_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__4));
v___x_2538_ = l_Lean_stringToMessageData(v___x_2537_);
return v___x_2538_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__7(void){
_start:
{
lean_object* v___x_2540_; lean_object* v___x_2541_; 
v___x_2540_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__6));
v___x_2541_ = l_Lean_stringToMessageData(v___x_2540_);
return v___x_2541_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__9(void){
_start:
{
lean_object* v___x_2543_; lean_object* v___x_2544_; 
v___x_2543_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__8));
v___x_2544_ = l_Lean_stringToMessageData(v___x_2543_);
return v___x_2544_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__11(void){
_start:
{
lean_object* v___x_2546_; lean_object* v___x_2547_; 
v___x_2546_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__10));
v___x_2547_ = l_Lean_stringToMessageData(v___x_2546_);
return v___x_2547_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__13(void){
_start:
{
lean_object* v___x_2549_; lean_object* v___x_2550_; 
v___x_2549_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__12));
v___x_2550_ = l_Lean_stringToMessageData(v___x_2549_);
return v___x_2550_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__15(void){
_start:
{
lean_object* v___x_2552_; lean_object* v___x_2553_; 
v___x_2552_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__14));
v___x_2553_ = l_Lean_stringToMessageData(v___x_2552_);
return v___x_2553_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2(lean_object* v_a_2554_, lean_object* v___f_2555_, uint8_t v___y_2556_, lean_object* v___f_2557_, lean_object* v_a_2558_, lean_object* v_a_2559_, lean_object* v_fst_2560_, lean_object* v_rel_2561_, lean_object* v___x_2562_, lean_object* v_snd_2563_, lean_object* v_____r_2564_, lean_object* v___y_2565_, lean_object* v___y_2566_, lean_object* v___y_2567_, lean_object* v___y_2568_){
_start:
{
lean_object* v___y_2571_; lean_object* v___y_2572_; lean_object* v___x_2575_; 
lean_inc(v_a_2554_);
v___x_2575_ = lp_batteries_Lean_mkConstWithLevelParams___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__2(v_a_2554_, v___y_2565_, v___y_2566_, v___y_2567_, v___y_2568_);
if (lean_obj_tag(v___x_2575_) == 0)
{
lean_object* v_a_2576_; lean_object* v___x_2577_; 
v_a_2576_ = lean_ctor_get(v___x_2575_, 0);
lean_inc(v_a_2576_);
lean_dec_ref_known(v___x_2575_, 1);
lean_inc(v___y_2568_);
lean_inc_ref(v___y_2567_);
lean_inc(v___y_2566_);
lean_inc_ref(v___y_2565_);
v___x_2577_ = lean_infer_type(v_a_2576_, v___y_2565_, v___y_2566_, v___y_2567_, v___y_2568_);
if (lean_obj_tag(v___x_2577_) == 0)
{
lean_object* v_a_2578_; lean_object* v_keyedConfig_2579_; uint8_t v_trackZetaDelta_2580_; lean_object* v_zetaDeltaSet_2581_; lean_object* v_lctx_2582_; lean_object* v_localInstances_2583_; lean_object* v_defEqCtx_x3f_2584_; lean_object* v_synthPendingDepth_2585_; lean_object* v_customCanUnfoldPredicate_x3f_2586_; uint8_t v_univApprox_2587_; uint8_t v_inTypeClassResolution_2588_; uint8_t v_cacheInferType_2589_; uint8_t v___x_2590_; lean_object* v___x_2591_; lean_object* v___x_2592_; lean_object* v___x_2593_; 
v_a_2578_ = lean_ctor_get(v___x_2577_, 0);
lean_inc_n(v_a_2578_, 2);
lean_dec_ref_known(v___x_2577_, 1);
v_keyedConfig_2579_ = lean_ctor_get(v___y_2565_, 0);
v_trackZetaDelta_2580_ = lean_ctor_get_uint8(v___y_2565_, sizeof(void*)*7);
v_zetaDeltaSet_2581_ = lean_ctor_get(v___y_2565_, 1);
v_lctx_2582_ = lean_ctor_get(v___y_2565_, 2);
v_localInstances_2583_ = lean_ctor_get(v___y_2565_, 3);
v_defEqCtx_x3f_2584_ = lean_ctor_get(v___y_2565_, 4);
v_synthPendingDepth_2585_ = lean_ctor_get(v___y_2565_, 5);
v_customCanUnfoldPredicate_x3f_2586_ = lean_ctor_get(v___y_2565_, 6);
v_univApprox_2587_ = lean_ctor_get_uint8(v___y_2565_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2588_ = lean_ctor_get_uint8(v___y_2565_, sizeof(void*)*7 + 2);
v_cacheInferType_2589_ = lean_ctor_get_uint8(v___y_2565_, sizeof(void*)*7 + 3);
v___x_2590_ = 2;
lean_inc_ref(v_keyedConfig_2579_);
v___x_2591_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2590_, v_keyedConfig_2579_);
lean_inc(v_customCanUnfoldPredicate_x3f_2586_);
lean_inc(v_synthPendingDepth_2585_);
lean_inc(v_defEqCtx_x3f_2584_);
lean_inc_ref(v_localInstances_2583_);
lean_inc_ref(v_lctx_2582_);
lean_inc(v_zetaDeltaSet_2581_);
v___x_2592_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_2592_, 0, v___x_2591_);
lean_ctor_set(v___x_2592_, 1, v_zetaDeltaSet_2581_);
lean_ctor_set(v___x_2592_, 2, v_lctx_2582_);
lean_ctor_set(v___x_2592_, 3, v_localInstances_2583_);
lean_ctor_set(v___x_2592_, 4, v_defEqCtx_x3f_2584_);
lean_ctor_set(v___x_2592_, 5, v_synthPendingDepth_2585_);
lean_ctor_set(v___x_2592_, 6, v_customCanUnfoldPredicate_x3f_2586_);
lean_ctor_set_uint8(v___x_2592_, sizeof(void*)*7, v_trackZetaDelta_2580_);
lean_ctor_set_uint8(v___x_2592_, sizeof(void*)*7 + 1, v_univApprox_2587_);
lean_ctor_set_uint8(v___x_2592_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2588_);
lean_ctor_set_uint8(v___x_2592_, sizeof(void*)*7 + 3, v_cacheInferType_2589_);
v___x_2593_ = lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__3___redArg(v_a_2578_, v___f_2555_, v___y_2556_, v___y_2556_, v___x_2592_, v___y_2566_, v___y_2567_, v___y_2568_);
lean_dec_ref_known(v___x_2592_, 7);
if (lean_obj_tag(v___x_2593_) == 0)
{
lean_object* v_a_2594_; lean_object* v___y_2596_; lean_object* v___y_2597_; lean_object* v___y_2598_; lean_object* v___y_2599_; lean_object* v___y_2600_; lean_object* v___y_2601_; lean_object* v___y_2602_; lean_object* v___y_2603_; lean_object* v___y_2642_; lean_object* v___y_2643_; lean_object* v___y_2644_; uint8_t v___y_2645_; lean_object* v___y_2646_; lean_object* v___y_2647_; lean_object* v___y_2648_; lean_object* v___y_2649_; lean_object* v___y_2650_; lean_object* v___y_2651_; lean_object* v___y_2695_; lean_object* v___y_2696_; lean_object* v___y_2697_; lean_object* v___y_2698_; lean_object* v___y_2699_; lean_object* v___y_2751_; lean_object* v___y_2752_; lean_object* v___y_2753_; lean_object* v___y_2754_; lean_object* v___y_2755_; lean_object* v___y_2780_; lean_object* v___y_2781_; lean_object* v___y_2782_; lean_object* v___y_2783_; lean_object* v___y_2784_; lean_object* v___y_2809_; lean_object* v___y_2810_; lean_object* v___y_2811_; lean_object* v___y_2812_; lean_object* v___y_2813_; lean_object* v___y_2838_; lean_object* v___y_2839_; lean_object* v___y_2840_; lean_object* v___y_2841_; lean_object* v_a_2842_; lean_object* v___y_2867_; lean_object* v___y_2868_; lean_object* v___y_2869_; lean_object* v___y_2870_; lean_object* v___y_2887_; lean_object* v___y_2888_; lean_object* v___y_2889_; lean_object* v___y_2890_; lean_object* v___x_2914_; 
v_a_2594_ = lean_ctor_get(v___x_2593_, 0);
lean_inc(v_a_2594_);
lean_dec_ref_known(v___x_2593_, 1);
lean_inc_ref(v___f_2557_);
lean_inc(v___y_2568_);
lean_inc_ref(v___y_2567_);
lean_inc(v___y_2566_);
lean_inc_ref(v___y_2565_);
v___x_2914_ = lean_apply_5(v___f_2557_, v___y_2565_, v___y_2566_, v___y_2567_, v___y_2568_, lean_box(0));
if (lean_obj_tag(v___x_2914_) == 0)
{
lean_object* v_a_2915_; uint8_t v___x_2916_; 
v_a_2915_ = lean_ctor_get(v___x_2914_, 0);
lean_inc(v_a_2915_);
lean_dec_ref_known(v___x_2914_, 1);
v___x_2916_ = lean_unbox(v_a_2915_);
lean_dec(v_a_2915_);
if (v___x_2916_ == 0)
{
v___y_2887_ = v___y_2565_;
v___y_2888_ = v___y_2566_;
v___y_2889_ = v___y_2567_;
v___y_2890_ = v___y_2568_;
goto v___jp_2886_;
}
else
{
lean_object* v___x_2917_; lean_object* v___x_2918_; lean_object* v___x_2919_; lean_object* v___x_2920_; lean_object* v___x_2921_; lean_object* v___x_2922_; 
v___x_2917_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__15, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__15_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__15);
lean_inc(v_a_2594_);
v___x_2918_ = l_Nat_reprFast(v_a_2594_);
v___x_2919_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2919_, 0, v___x_2918_);
v___x_2920_ = l_Lean_MessageData_ofFormat(v___x_2919_);
v___x_2921_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2921_, 0, v___x_2917_);
lean_ctor_set(v___x_2921_, 1, v___x_2920_);
lean_inc(v___x_2562_);
v___x_2922_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__5(v___x_2562_, v___x_2921_, v___y_2565_, v___y_2566_, v___y_2567_, v___y_2568_);
if (lean_obj_tag(v___x_2922_) == 0)
{
lean_dec_ref_known(v___x_2922_, 1);
v___y_2887_ = v___y_2565_;
v___y_2888_ = v___y_2566_;
v___y_2889_ = v___y_2567_;
v___y_2890_ = v___y_2568_;
goto v___jp_2886_;
}
else
{
lean_object* v_a_2923_; lean_object* v___x_2925_; uint8_t v_isShared_2926_; uint8_t v_isSharedCheck_2930_; 
lean_dec(v_a_2594_);
lean_dec(v_a_2578_);
lean_dec_ref(v_snd_2563_);
lean_dec(v___x_2562_);
lean_dec_ref(v_rel_2561_);
lean_dec_ref(v_fst_2560_);
lean_dec(v_a_2559_);
lean_dec(v_a_2558_);
lean_dec_ref(v___f_2557_);
lean_dec(v_a_2554_);
v_a_2923_ = lean_ctor_get(v___x_2922_, 0);
v_isSharedCheck_2930_ = !lean_is_exclusive(v___x_2922_);
if (v_isSharedCheck_2930_ == 0)
{
v___x_2925_ = v___x_2922_;
v_isShared_2926_ = v_isSharedCheck_2930_;
goto v_resetjp_2924_;
}
else
{
lean_inc(v_a_2923_);
lean_dec(v___x_2922_);
v___x_2925_ = lean_box(0);
v_isShared_2926_ = v_isSharedCheck_2930_;
goto v_resetjp_2924_;
}
v_resetjp_2924_:
{
lean_object* v___x_2928_; 
if (v_isShared_2926_ == 0)
{
v___x_2928_ = v___x_2925_;
goto v_reusejp_2927_;
}
else
{
lean_object* v_reuseFailAlloc_2929_; 
v_reuseFailAlloc_2929_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2929_, 0, v_a_2923_);
v___x_2928_ = v_reuseFailAlloc_2929_;
goto v_reusejp_2927_;
}
v_reusejp_2927_:
{
return v___x_2928_;
}
}
}
}
}
else
{
lean_object* v_a_2931_; lean_object* v___x_2933_; uint8_t v_isShared_2934_; uint8_t v_isSharedCheck_2938_; 
lean_dec(v_a_2594_);
lean_dec(v_a_2578_);
lean_dec_ref(v_snd_2563_);
lean_dec(v___x_2562_);
lean_dec_ref(v_rel_2561_);
lean_dec_ref(v_fst_2560_);
lean_dec(v_a_2559_);
lean_dec(v_a_2558_);
lean_dec_ref(v___f_2557_);
lean_dec(v_a_2554_);
v_a_2931_ = lean_ctor_get(v___x_2914_, 0);
v_isSharedCheck_2938_ = !lean_is_exclusive(v___x_2914_);
if (v_isSharedCheck_2938_ == 0)
{
v___x_2933_ = v___x_2914_;
v_isShared_2934_ = v_isSharedCheck_2938_;
goto v_resetjp_2932_;
}
else
{
lean_inc(v_a_2931_);
lean_dec(v___x_2914_);
v___x_2933_ = lean_box(0);
v_isShared_2934_ = v_isSharedCheck_2938_;
goto v_resetjp_2932_;
}
v_resetjp_2932_:
{
lean_object* v___x_2936_; 
if (v_isShared_2934_ == 0)
{
v___x_2936_ = v___x_2933_;
goto v_reusejp_2935_;
}
else
{
lean_object* v_reuseFailAlloc_2937_; 
v_reuseFailAlloc_2937_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2937_, 0, v_a_2931_);
v___x_2936_ = v_reuseFailAlloc_2937_;
goto v_reusejp_2935_;
}
v_reusejp_2935_:
{
return v___x_2936_;
}
}
}
v___jp_2595_:
{
lean_object* v___x_2604_; lean_object* v___x_2605_; lean_object* v___x_2606_; lean_object* v___x_2607_; lean_object* v___x_2608_; lean_object* v___x_2609_; lean_object* v___x_2610_; lean_object* v___x_2611_; lean_object* v___x_2612_; lean_object* v___x_2613_; 
v___x_2604_ = lean_nat_sub(v_a_2594_, v___y_2599_);
lean_dec(v_a_2594_);
v___x_2605_ = lean_box(0);
v___x_2606_ = lean_mk_array(v___x_2604_, v___x_2605_);
lean_inc_ref(v___y_2598_);
v___x_2607_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2607_, 0, v___y_2598_);
lean_inc_ref(v___y_2597_);
v___x_2608_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2608_, 0, v___y_2597_);
v___x_2609_ = lean_mk_empty_array_with_capacity(v___y_2599_);
v___x_2610_ = lean_array_push(v___x_2609_, v___x_2607_);
v___x_2611_ = lean_array_push(v___x_2610_, v___x_2608_);
v___x_2612_ = l_Array_append___redArg(v___x_2606_, v___x_2611_);
lean_dec_ref(v___x_2611_);
v___x_2613_ = l_Lean_Meta_mkAppOptM(v_a_2554_, v___x_2612_, v___y_2600_, v___y_2601_, v___y_2602_, v___y_2603_);
if (lean_obj_tag(v___x_2613_) == 0)
{
lean_object* v_a_2614_; lean_object* v___x_2615_; 
v_a_2614_ = lean_ctor_get(v___x_2613_, 0);
lean_inc(v_a_2614_);
lean_dec_ref_known(v___x_2613_, 1);
v___x_2615_ = lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4___redArg(v_a_2558_, v_a_2614_, v___y_2601_);
if (lean_obj_tag(v___x_2615_) == 0)
{
lean_object* v___x_2616_; lean_object* v___x_2617_; lean_object* v___x_2618_; lean_object* v___x_2619_; lean_object* v___x_2620_; 
lean_dec_ref_known(v___x_2615_, 1);
v___x_2616_ = l_Lean_Expr_mvarId_x21(v___y_2598_);
lean_dec_ref(v___y_2598_);
v___x_2617_ = l_Lean_Expr_mvarId_x21(v___y_2597_);
lean_dec_ref(v___y_2597_);
v___x_2618_ = lean_box(0);
v___x_2619_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2619_, 0, v___x_2617_);
lean_ctor_set(v___x_2619_, 1, v___x_2618_);
v___x_2620_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2620_, 0, v___x_2616_);
lean_ctor_set(v___x_2620_, 1, v___x_2619_);
if (lean_obj_tag(v_a_2559_) == 1)
{
lean_object* v_val_2621_; lean_object* v_snd_2622_; 
lean_dec_ref(v___y_2596_);
v_val_2621_ = lean_ctor_get(v_a_2559_, 0);
lean_inc(v_val_2621_);
lean_dec_ref_known(v_a_2559_, 1);
v_snd_2622_ = lean_ctor_get(v_val_2621_, 1);
lean_inc(v_snd_2622_);
lean_dec(v_val_2621_);
v___y_2571_ = v___x_2620_;
v___y_2572_ = v_snd_2622_;
goto v___jp_2570_;
}
else
{
lean_object* v___x_2623_; lean_object* v___x_2624_; 
lean_dec(v_a_2559_);
v___x_2623_ = l_Lean_Expr_mvarId_x21(v___y_2596_);
lean_dec_ref(v___y_2596_);
v___x_2624_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2624_, 0, v___x_2623_);
lean_ctor_set(v___x_2624_, 1, v___x_2618_);
v___y_2571_ = v___x_2620_;
v___y_2572_ = v___x_2624_;
goto v___jp_2570_;
}
}
else
{
lean_object* v_a_2625_; lean_object* v___x_2627_; uint8_t v_isShared_2628_; uint8_t v_isSharedCheck_2632_; 
lean_dec_ref(v___y_2598_);
lean_dec_ref(v___y_2597_);
lean_dec_ref(v___y_2596_);
lean_dec(v_a_2559_);
v_a_2625_ = lean_ctor_get(v___x_2615_, 0);
v_isSharedCheck_2632_ = !lean_is_exclusive(v___x_2615_);
if (v_isSharedCheck_2632_ == 0)
{
v___x_2627_ = v___x_2615_;
v_isShared_2628_ = v_isSharedCheck_2632_;
goto v_resetjp_2626_;
}
else
{
lean_inc(v_a_2625_);
lean_dec(v___x_2615_);
v___x_2627_ = lean_box(0);
v_isShared_2628_ = v_isSharedCheck_2632_;
goto v_resetjp_2626_;
}
v_resetjp_2626_:
{
lean_object* v___x_2630_; 
if (v_isShared_2628_ == 0)
{
v___x_2630_ = v___x_2627_;
goto v_reusejp_2629_;
}
else
{
lean_object* v_reuseFailAlloc_2631_; 
v_reuseFailAlloc_2631_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2631_, 0, v_a_2625_);
v___x_2630_ = v_reuseFailAlloc_2631_;
goto v_reusejp_2629_;
}
v_reusejp_2629_:
{
return v___x_2630_;
}
}
}
}
else
{
lean_object* v_a_2633_; lean_object* v___x_2635_; uint8_t v_isShared_2636_; uint8_t v_isSharedCheck_2640_; 
lean_dec_ref(v___y_2598_);
lean_dec_ref(v___y_2597_);
lean_dec_ref(v___y_2596_);
lean_dec(v_a_2559_);
lean_dec(v_a_2558_);
v_a_2633_ = lean_ctor_get(v___x_2613_, 0);
v_isSharedCheck_2640_ = !lean_is_exclusive(v___x_2613_);
if (v_isSharedCheck_2640_ == 0)
{
v___x_2635_ = v___x_2613_;
v_isShared_2636_ = v_isSharedCheck_2640_;
goto v_resetjp_2634_;
}
else
{
lean_inc(v_a_2633_);
lean_dec(v___x_2613_);
v___x_2635_ = lean_box(0);
v_isShared_2636_ = v_isSharedCheck_2640_;
goto v_resetjp_2634_;
}
v_resetjp_2634_:
{
lean_object* v___x_2638_; 
if (v_isShared_2636_ == 0)
{
v___x_2638_ = v___x_2635_;
goto v_reusejp_2637_;
}
else
{
lean_object* v_reuseFailAlloc_2639_; 
v_reuseFailAlloc_2639_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2639_, 0, v_a_2633_);
v___x_2638_ = v_reuseFailAlloc_2639_;
goto v_reusejp_2637_;
}
v_reusejp_2637_:
{
return v___x_2638_;
}
}
}
}
v___jp_2641_:
{
lean_object* v___x_2652_; lean_object* v___x_2653_; lean_object* v___x_2654_; 
v___x_2652_ = lean_array_push(v___y_2646_, v_fst_2560_);
lean_inc_ref(v___y_2643_);
v___x_2653_ = lean_array_push(v___x_2652_, v___y_2643_);
v___x_2654_ = l_Lean_Meta_mkAppM_x27(v_rel_2561_, v___x_2653_, v___y_2648_, v___y_2649_, v___y_2650_, v___y_2651_);
if (lean_obj_tag(v___x_2654_) == 0)
{
lean_object* v_a_2655_; lean_object* v___x_2656_; lean_object* v___x_2657_; 
v_a_2655_ = lean_ctor_get(v___x_2654_, 0);
lean_inc(v_a_2655_);
lean_dec_ref_known(v___x_2654_, 1);
v___x_2656_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2656_, 0, v_a_2655_);
v___x_2657_ = l_Lean_Meta_mkFreshExprMVar(v___x_2656_, v___y_2645_, v___y_2644_, v___y_2648_, v___y_2649_, v___y_2650_, v___y_2651_);
if (lean_obj_tag(v___x_2657_) == 0)
{
lean_object* v_options_2658_; uint8_t v_hasTrace_2659_; 
v_options_2658_ = lean_ctor_get(v___y_2650_, 2);
v_hasTrace_2659_ = lean_ctor_get_uint8(v_options_2658_, sizeof(void*)*1);
if (v_hasTrace_2659_ == 0)
{
lean_object* v_a_2660_; 
lean_dec(v___x_2562_);
v_a_2660_ = lean_ctor_get(v___x_2657_, 0);
lean_inc(v_a_2660_);
lean_dec_ref_known(v___x_2657_, 1);
v___y_2596_ = v___y_2643_;
v___y_2597_ = v___y_2642_;
v___y_2598_ = v_a_2660_;
v___y_2599_ = v___y_2647_;
v___y_2600_ = v___y_2648_;
v___y_2601_ = v___y_2649_;
v___y_2602_ = v___y_2650_;
v___y_2603_ = v___y_2651_;
goto v___jp_2595_;
}
else
{
lean_object* v_a_2661_; lean_object* v_inheritedTraceOptions_2662_; lean_object* v___x_2663_; lean_object* v___x_2664_; uint8_t v___x_2665_; 
v_a_2661_ = lean_ctor_get(v___x_2657_, 0);
lean_inc(v_a_2661_);
lean_dec_ref_known(v___x_2657_, 1);
v_inheritedTraceOptions_2662_ = lean_ctor_get(v___y_2650_, 13);
v___x_2663_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__0___closed__1));
lean_inc(v___x_2562_);
v___x_2664_ = l_Lean_Name_append(v___x_2663_, v___x_2562_);
v___x_2665_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2662_, v_options_2658_, v___x_2664_);
lean_dec(v___x_2664_);
if (v___x_2665_ == 0)
{
lean_dec(v___x_2562_);
v___y_2596_ = v___y_2643_;
v___y_2597_ = v___y_2642_;
v___y_2598_ = v_a_2661_;
v___y_2599_ = v___y_2647_;
v___y_2600_ = v___y_2648_;
v___y_2601_ = v___y_2649_;
v___y_2602_ = v___y_2650_;
v___y_2603_ = v___y_2651_;
goto v___jp_2595_;
}
else
{
lean_object* v___x_2666_; lean_object* v___x_2667_; lean_object* v___x_2668_; lean_object* v___x_2669_; 
v___x_2666_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__1, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__1_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__1);
lean_inc(v_a_2661_);
v___x_2667_ = l_Lean_MessageData_ofExpr(v_a_2661_);
v___x_2668_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2668_, 0, v___x_2666_);
lean_ctor_set(v___x_2668_, 1, v___x_2667_);
v___x_2669_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__5(v___x_2562_, v___x_2668_, v___y_2648_, v___y_2649_, v___y_2650_, v___y_2651_);
if (lean_obj_tag(v___x_2669_) == 0)
{
lean_dec_ref_known(v___x_2669_, 1);
v___y_2596_ = v___y_2643_;
v___y_2597_ = v___y_2642_;
v___y_2598_ = v_a_2661_;
v___y_2599_ = v___y_2647_;
v___y_2600_ = v___y_2648_;
v___y_2601_ = v___y_2649_;
v___y_2602_ = v___y_2650_;
v___y_2603_ = v___y_2651_;
goto v___jp_2595_;
}
else
{
lean_object* v_a_2670_; lean_object* v___x_2672_; uint8_t v_isShared_2673_; uint8_t v_isSharedCheck_2677_; 
lean_dec(v_a_2661_);
lean_dec_ref(v___y_2643_);
lean_dec_ref(v___y_2642_);
lean_dec(v_a_2594_);
lean_dec(v_a_2559_);
lean_dec(v_a_2558_);
lean_dec(v_a_2554_);
v_a_2670_ = lean_ctor_get(v___x_2669_, 0);
v_isSharedCheck_2677_ = !lean_is_exclusive(v___x_2669_);
if (v_isSharedCheck_2677_ == 0)
{
v___x_2672_ = v___x_2669_;
v_isShared_2673_ = v_isSharedCheck_2677_;
goto v_resetjp_2671_;
}
else
{
lean_inc(v_a_2670_);
lean_dec(v___x_2669_);
v___x_2672_ = lean_box(0);
v_isShared_2673_ = v_isSharedCheck_2677_;
goto v_resetjp_2671_;
}
v_resetjp_2671_:
{
lean_object* v___x_2675_; 
if (v_isShared_2673_ == 0)
{
v___x_2675_ = v___x_2672_;
goto v_reusejp_2674_;
}
else
{
lean_object* v_reuseFailAlloc_2676_; 
v_reuseFailAlloc_2676_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2676_, 0, v_a_2670_);
v___x_2675_ = v_reuseFailAlloc_2676_;
goto v_reusejp_2674_;
}
v_reusejp_2674_:
{
return v___x_2675_;
}
}
}
}
}
}
else
{
lean_object* v_a_2678_; lean_object* v___x_2680_; uint8_t v_isShared_2681_; uint8_t v_isSharedCheck_2685_; 
lean_dec_ref(v___y_2643_);
lean_dec_ref(v___y_2642_);
lean_dec(v_a_2594_);
lean_dec(v___x_2562_);
lean_dec(v_a_2559_);
lean_dec(v_a_2558_);
lean_dec(v_a_2554_);
v_a_2678_ = lean_ctor_get(v___x_2657_, 0);
v_isSharedCheck_2685_ = !lean_is_exclusive(v___x_2657_);
if (v_isSharedCheck_2685_ == 0)
{
v___x_2680_ = v___x_2657_;
v_isShared_2681_ = v_isSharedCheck_2685_;
goto v_resetjp_2679_;
}
else
{
lean_inc(v_a_2678_);
lean_dec(v___x_2657_);
v___x_2680_ = lean_box(0);
v_isShared_2681_ = v_isSharedCheck_2685_;
goto v_resetjp_2679_;
}
v_resetjp_2679_:
{
lean_object* v___x_2683_; 
if (v_isShared_2681_ == 0)
{
v___x_2683_ = v___x_2680_;
goto v_reusejp_2682_;
}
else
{
lean_object* v_reuseFailAlloc_2684_; 
v_reuseFailAlloc_2684_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2684_, 0, v_a_2678_);
v___x_2683_ = v_reuseFailAlloc_2684_;
goto v_reusejp_2682_;
}
v_reusejp_2682_:
{
return v___x_2683_;
}
}
}
}
else
{
lean_object* v_a_2686_; lean_object* v___x_2688_; uint8_t v_isShared_2689_; uint8_t v_isSharedCheck_2693_; 
lean_dec(v___y_2644_);
lean_dec_ref(v___y_2643_);
lean_dec_ref(v___y_2642_);
lean_dec(v_a_2594_);
lean_dec(v___x_2562_);
lean_dec(v_a_2559_);
lean_dec(v_a_2558_);
lean_dec(v_a_2554_);
v_a_2686_ = lean_ctor_get(v___x_2654_, 0);
v_isSharedCheck_2693_ = !lean_is_exclusive(v___x_2654_);
if (v_isSharedCheck_2693_ == 0)
{
v___x_2688_ = v___x_2654_;
v_isShared_2689_ = v_isSharedCheck_2693_;
goto v_resetjp_2687_;
}
else
{
lean_inc(v_a_2686_);
lean_dec(v___x_2654_);
v___x_2688_ = lean_box(0);
v_isShared_2689_ = v_isSharedCheck_2693_;
goto v_resetjp_2687_;
}
v_resetjp_2687_:
{
lean_object* v___x_2691_; 
if (v_isShared_2689_ == 0)
{
v___x_2691_ = v___x_2688_;
goto v_reusejp_2690_;
}
else
{
lean_object* v_reuseFailAlloc_2692_; 
v_reuseFailAlloc_2692_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2692_, 0, v_a_2686_);
v___x_2691_ = v_reuseFailAlloc_2692_;
goto v_reusejp_2690_;
}
v_reusejp_2690_:
{
return v___x_2691_;
}
}
}
}
v___jp_2694_:
{
lean_object* v___x_2700_; lean_object* v___x_2701_; lean_object* v___x_2702_; lean_object* v___x_2703_; lean_object* v___x_2704_; 
v___x_2700_ = lean_unsigned_to_nat(2u);
v___x_2701_ = lean_mk_empty_array_with_capacity(v___x_2700_);
lean_inc_ref(v___y_2695_);
lean_inc_ref(v___x_2701_);
v___x_2702_ = lean_array_push(v___x_2701_, v___y_2695_);
v___x_2703_ = lean_array_push(v___x_2702_, v_snd_2563_);
lean_inc_ref(v_rel_2561_);
v___x_2704_ = l_Lean_Meta_mkAppM_x27(v_rel_2561_, v___x_2703_, v___y_2696_, v___y_2697_, v___y_2698_, v___y_2699_);
if (lean_obj_tag(v___x_2704_) == 0)
{
lean_object* v_a_2705_; lean_object* v___x_2706_; uint8_t v___x_2707_; lean_object* v___x_2708_; lean_object* v___x_2709_; 
v_a_2705_ = lean_ctor_get(v___x_2704_, 0);
lean_inc(v_a_2705_);
lean_dec_ref_known(v___x_2704_, 1);
v___x_2706_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2706_, 0, v_a_2705_);
v___x_2707_ = 1;
v___x_2708_ = lean_box(0);
v___x_2709_ = l_Lean_Meta_mkFreshExprMVar(v___x_2706_, v___x_2707_, v___x_2708_, v___y_2696_, v___y_2697_, v___y_2698_, v___y_2699_);
if (lean_obj_tag(v___x_2709_) == 0)
{
lean_object* v_a_2710_; lean_object* v___x_2711_; 
v_a_2710_ = lean_ctor_get(v___x_2709_, 0);
lean_inc(v_a_2710_);
lean_dec_ref_known(v___x_2709_, 1);
lean_inc(v___y_2699_);
lean_inc_ref(v___y_2698_);
lean_inc(v___y_2697_);
lean_inc_ref(v___y_2696_);
v___x_2711_ = lean_apply_5(v___f_2557_, v___y_2696_, v___y_2697_, v___y_2698_, v___y_2699_, lean_box(0));
if (lean_obj_tag(v___x_2711_) == 0)
{
lean_object* v_a_2712_; uint8_t v___x_2713_; 
v_a_2712_ = lean_ctor_get(v___x_2711_, 0);
lean_inc(v_a_2712_);
lean_dec_ref_known(v___x_2711_, 1);
v___x_2713_ = lean_unbox(v_a_2712_);
lean_dec(v_a_2712_);
if (v___x_2713_ == 0)
{
v___y_2642_ = v_a_2710_;
v___y_2643_ = v___y_2695_;
v___y_2644_ = v___x_2708_;
v___y_2645_ = v___x_2707_;
v___y_2646_ = v___x_2701_;
v___y_2647_ = v___x_2700_;
v___y_2648_ = v___y_2696_;
v___y_2649_ = v___y_2697_;
v___y_2650_ = v___y_2698_;
v___y_2651_ = v___y_2699_;
goto v___jp_2641_;
}
else
{
lean_object* v___x_2714_; lean_object* v___x_2715_; lean_object* v___x_2716_; lean_object* v___x_2717_; 
v___x_2714_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__3, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__3_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__3);
lean_inc(v_a_2710_);
v___x_2715_ = l_Lean_MessageData_ofExpr(v_a_2710_);
v___x_2716_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2716_, 0, v___x_2714_);
lean_ctor_set(v___x_2716_, 1, v___x_2715_);
lean_inc(v___x_2562_);
v___x_2717_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__5(v___x_2562_, v___x_2716_, v___y_2696_, v___y_2697_, v___y_2698_, v___y_2699_);
if (lean_obj_tag(v___x_2717_) == 0)
{
lean_dec_ref_known(v___x_2717_, 1);
v___y_2642_ = v_a_2710_;
v___y_2643_ = v___y_2695_;
v___y_2644_ = v___x_2708_;
v___y_2645_ = v___x_2707_;
v___y_2646_ = v___x_2701_;
v___y_2647_ = v___x_2700_;
v___y_2648_ = v___y_2696_;
v___y_2649_ = v___y_2697_;
v___y_2650_ = v___y_2698_;
v___y_2651_ = v___y_2699_;
goto v___jp_2641_;
}
else
{
lean_object* v_a_2718_; lean_object* v___x_2720_; uint8_t v_isShared_2721_; uint8_t v_isSharedCheck_2725_; 
lean_dec(v_a_2710_);
lean_dec_ref(v___x_2701_);
lean_dec_ref(v___y_2695_);
lean_dec(v_a_2594_);
lean_dec(v___x_2562_);
lean_dec_ref(v_rel_2561_);
lean_dec_ref(v_fst_2560_);
lean_dec(v_a_2559_);
lean_dec(v_a_2558_);
lean_dec(v_a_2554_);
v_a_2718_ = lean_ctor_get(v___x_2717_, 0);
v_isSharedCheck_2725_ = !lean_is_exclusive(v___x_2717_);
if (v_isSharedCheck_2725_ == 0)
{
v___x_2720_ = v___x_2717_;
v_isShared_2721_ = v_isSharedCheck_2725_;
goto v_resetjp_2719_;
}
else
{
lean_inc(v_a_2718_);
lean_dec(v___x_2717_);
v___x_2720_ = lean_box(0);
v_isShared_2721_ = v_isSharedCheck_2725_;
goto v_resetjp_2719_;
}
v_resetjp_2719_:
{
lean_object* v___x_2723_; 
if (v_isShared_2721_ == 0)
{
v___x_2723_ = v___x_2720_;
goto v_reusejp_2722_;
}
else
{
lean_object* v_reuseFailAlloc_2724_; 
v_reuseFailAlloc_2724_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2724_, 0, v_a_2718_);
v___x_2723_ = v_reuseFailAlloc_2724_;
goto v_reusejp_2722_;
}
v_reusejp_2722_:
{
return v___x_2723_;
}
}
}
}
}
else
{
lean_object* v_a_2726_; lean_object* v___x_2728_; uint8_t v_isShared_2729_; uint8_t v_isSharedCheck_2733_; 
lean_dec(v_a_2710_);
lean_dec_ref(v___x_2701_);
lean_dec_ref(v___y_2695_);
lean_dec(v_a_2594_);
lean_dec(v___x_2562_);
lean_dec_ref(v_rel_2561_);
lean_dec_ref(v_fst_2560_);
lean_dec(v_a_2559_);
lean_dec(v_a_2558_);
lean_dec(v_a_2554_);
v_a_2726_ = lean_ctor_get(v___x_2711_, 0);
v_isSharedCheck_2733_ = !lean_is_exclusive(v___x_2711_);
if (v_isSharedCheck_2733_ == 0)
{
v___x_2728_ = v___x_2711_;
v_isShared_2729_ = v_isSharedCheck_2733_;
goto v_resetjp_2727_;
}
else
{
lean_inc(v_a_2726_);
lean_dec(v___x_2711_);
v___x_2728_ = lean_box(0);
v_isShared_2729_ = v_isSharedCheck_2733_;
goto v_resetjp_2727_;
}
v_resetjp_2727_:
{
lean_object* v___x_2731_; 
if (v_isShared_2729_ == 0)
{
v___x_2731_ = v___x_2728_;
goto v_reusejp_2730_;
}
else
{
lean_object* v_reuseFailAlloc_2732_; 
v_reuseFailAlloc_2732_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2732_, 0, v_a_2726_);
v___x_2731_ = v_reuseFailAlloc_2732_;
goto v_reusejp_2730_;
}
v_reusejp_2730_:
{
return v___x_2731_;
}
}
}
}
else
{
lean_object* v_a_2734_; lean_object* v___x_2736_; uint8_t v_isShared_2737_; uint8_t v_isSharedCheck_2741_; 
lean_dec_ref(v___x_2701_);
lean_dec_ref(v___y_2695_);
lean_dec(v_a_2594_);
lean_dec(v___x_2562_);
lean_dec_ref(v_rel_2561_);
lean_dec_ref(v_fst_2560_);
lean_dec(v_a_2559_);
lean_dec(v_a_2558_);
lean_dec_ref(v___f_2557_);
lean_dec(v_a_2554_);
v_a_2734_ = lean_ctor_get(v___x_2709_, 0);
v_isSharedCheck_2741_ = !lean_is_exclusive(v___x_2709_);
if (v_isSharedCheck_2741_ == 0)
{
v___x_2736_ = v___x_2709_;
v_isShared_2737_ = v_isSharedCheck_2741_;
goto v_resetjp_2735_;
}
else
{
lean_inc(v_a_2734_);
lean_dec(v___x_2709_);
v___x_2736_ = lean_box(0);
v_isShared_2737_ = v_isSharedCheck_2741_;
goto v_resetjp_2735_;
}
v_resetjp_2735_:
{
lean_object* v___x_2739_; 
if (v_isShared_2737_ == 0)
{
v___x_2739_ = v___x_2736_;
goto v_reusejp_2738_;
}
else
{
lean_object* v_reuseFailAlloc_2740_; 
v_reuseFailAlloc_2740_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2740_, 0, v_a_2734_);
v___x_2739_ = v_reuseFailAlloc_2740_;
goto v_reusejp_2738_;
}
v_reusejp_2738_:
{
return v___x_2739_;
}
}
}
}
else
{
lean_object* v_a_2742_; lean_object* v___x_2744_; uint8_t v_isShared_2745_; uint8_t v_isSharedCheck_2749_; 
lean_dec_ref(v___x_2701_);
lean_dec_ref(v___y_2695_);
lean_dec(v_a_2594_);
lean_dec(v___x_2562_);
lean_dec_ref(v_rel_2561_);
lean_dec_ref(v_fst_2560_);
lean_dec(v_a_2559_);
lean_dec(v_a_2558_);
lean_dec_ref(v___f_2557_);
lean_dec(v_a_2554_);
v_a_2742_ = lean_ctor_get(v___x_2704_, 0);
v_isSharedCheck_2749_ = !lean_is_exclusive(v___x_2704_);
if (v_isSharedCheck_2749_ == 0)
{
v___x_2744_ = v___x_2704_;
v_isShared_2745_ = v_isSharedCheck_2749_;
goto v_resetjp_2743_;
}
else
{
lean_inc(v_a_2742_);
lean_dec(v___x_2704_);
v___x_2744_ = lean_box(0);
v_isShared_2745_ = v_isSharedCheck_2749_;
goto v_resetjp_2743_;
}
v_resetjp_2743_:
{
lean_object* v___x_2747_; 
if (v_isShared_2745_ == 0)
{
v___x_2747_ = v___x_2744_;
goto v_reusejp_2746_;
}
else
{
lean_object* v_reuseFailAlloc_2748_; 
v_reuseFailAlloc_2748_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2748_, 0, v_a_2742_);
v___x_2747_ = v_reuseFailAlloc_2748_;
goto v_reusejp_2746_;
}
v_reusejp_2746_:
{
return v___x_2747_;
}
}
}
}
v___jp_2750_:
{
lean_object* v___x_2756_; 
lean_inc_ref(v___f_2557_);
lean_inc(v___y_2755_);
lean_inc_ref(v___y_2754_);
lean_inc(v___y_2753_);
lean_inc_ref(v___y_2752_);
v___x_2756_ = lean_apply_5(v___f_2557_, v___y_2752_, v___y_2753_, v___y_2754_, v___y_2755_, lean_box(0));
if (lean_obj_tag(v___x_2756_) == 0)
{
lean_object* v_a_2757_; uint8_t v___x_2758_; 
v_a_2757_ = lean_ctor_get(v___x_2756_, 0);
lean_inc(v_a_2757_);
lean_dec_ref_known(v___x_2756_, 1);
v___x_2758_ = lean_unbox(v_a_2757_);
lean_dec(v_a_2757_);
if (v___x_2758_ == 0)
{
v___y_2695_ = v___y_2751_;
v___y_2696_ = v___y_2752_;
v___y_2697_ = v___y_2753_;
v___y_2698_ = v___y_2754_;
v___y_2699_ = v___y_2755_;
goto v___jp_2694_;
}
else
{
lean_object* v___x_2759_; lean_object* v___x_2760_; lean_object* v___x_2761_; lean_object* v___x_2762_; 
v___x_2759_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__5, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__5_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__5);
lean_inc_ref(v_snd_2563_);
v___x_2760_ = l_Lean_indentExpr(v_snd_2563_);
v___x_2761_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2761_, 0, v___x_2759_);
lean_ctor_set(v___x_2761_, 1, v___x_2760_);
lean_inc(v___x_2562_);
v___x_2762_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__5(v___x_2562_, v___x_2761_, v___y_2752_, v___y_2753_, v___y_2754_, v___y_2755_);
if (lean_obj_tag(v___x_2762_) == 0)
{
lean_dec_ref_known(v___x_2762_, 1);
v___y_2695_ = v___y_2751_;
v___y_2696_ = v___y_2752_;
v___y_2697_ = v___y_2753_;
v___y_2698_ = v___y_2754_;
v___y_2699_ = v___y_2755_;
goto v___jp_2694_;
}
else
{
lean_object* v_a_2763_; lean_object* v___x_2765_; uint8_t v_isShared_2766_; uint8_t v_isSharedCheck_2770_; 
lean_dec_ref(v___y_2751_);
lean_dec(v_a_2594_);
lean_dec_ref(v_snd_2563_);
lean_dec(v___x_2562_);
lean_dec_ref(v_rel_2561_);
lean_dec_ref(v_fst_2560_);
lean_dec(v_a_2559_);
lean_dec(v_a_2558_);
lean_dec_ref(v___f_2557_);
lean_dec(v_a_2554_);
v_a_2763_ = lean_ctor_get(v___x_2762_, 0);
v_isSharedCheck_2770_ = !lean_is_exclusive(v___x_2762_);
if (v_isSharedCheck_2770_ == 0)
{
v___x_2765_ = v___x_2762_;
v_isShared_2766_ = v_isSharedCheck_2770_;
goto v_resetjp_2764_;
}
else
{
lean_inc(v_a_2763_);
lean_dec(v___x_2762_);
v___x_2765_ = lean_box(0);
v_isShared_2766_ = v_isSharedCheck_2770_;
goto v_resetjp_2764_;
}
v_resetjp_2764_:
{
lean_object* v___x_2768_; 
if (v_isShared_2766_ == 0)
{
v___x_2768_ = v___x_2765_;
goto v_reusejp_2767_;
}
else
{
lean_object* v_reuseFailAlloc_2769_; 
v_reuseFailAlloc_2769_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2769_, 0, v_a_2763_);
v___x_2768_ = v_reuseFailAlloc_2769_;
goto v_reusejp_2767_;
}
v_reusejp_2767_:
{
return v___x_2768_;
}
}
}
}
}
else
{
lean_object* v_a_2771_; lean_object* v___x_2773_; uint8_t v_isShared_2774_; uint8_t v_isSharedCheck_2778_; 
lean_dec_ref(v___y_2751_);
lean_dec(v_a_2594_);
lean_dec_ref(v_snd_2563_);
lean_dec(v___x_2562_);
lean_dec_ref(v_rel_2561_);
lean_dec_ref(v_fst_2560_);
lean_dec(v_a_2559_);
lean_dec(v_a_2558_);
lean_dec_ref(v___f_2557_);
lean_dec(v_a_2554_);
v_a_2771_ = lean_ctor_get(v___x_2756_, 0);
v_isSharedCheck_2778_ = !lean_is_exclusive(v___x_2756_);
if (v_isSharedCheck_2778_ == 0)
{
v___x_2773_ = v___x_2756_;
v_isShared_2774_ = v_isSharedCheck_2778_;
goto v_resetjp_2772_;
}
else
{
lean_inc(v_a_2771_);
lean_dec(v___x_2756_);
v___x_2773_ = lean_box(0);
v_isShared_2774_ = v_isSharedCheck_2778_;
goto v_resetjp_2772_;
}
v_resetjp_2772_:
{
lean_object* v___x_2776_; 
if (v_isShared_2774_ == 0)
{
v___x_2776_ = v___x_2773_;
goto v_reusejp_2775_;
}
else
{
lean_object* v_reuseFailAlloc_2777_; 
v_reuseFailAlloc_2777_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2777_, 0, v_a_2771_);
v___x_2776_ = v_reuseFailAlloc_2777_;
goto v_reusejp_2775_;
}
v_reusejp_2775_:
{
return v___x_2776_;
}
}
}
}
v___jp_2779_:
{
lean_object* v___x_2785_; 
lean_inc_ref(v___f_2557_);
lean_inc(v___y_2784_);
lean_inc_ref(v___y_2783_);
lean_inc(v___y_2782_);
lean_inc_ref(v___y_2781_);
v___x_2785_ = lean_apply_5(v___f_2557_, v___y_2781_, v___y_2782_, v___y_2783_, v___y_2784_, lean_box(0));
if (lean_obj_tag(v___x_2785_) == 0)
{
lean_object* v_a_2786_; uint8_t v___x_2787_; 
v_a_2786_ = lean_ctor_get(v___x_2785_, 0);
lean_inc(v_a_2786_);
lean_dec_ref_known(v___x_2785_, 1);
v___x_2787_ = lean_unbox(v_a_2786_);
lean_dec(v_a_2786_);
if (v___x_2787_ == 0)
{
v___y_2751_ = v___y_2780_;
v___y_2752_ = v___y_2781_;
v___y_2753_ = v___y_2782_;
v___y_2754_ = v___y_2783_;
v___y_2755_ = v___y_2784_;
goto v___jp_2750_;
}
else
{
lean_object* v___x_2788_; lean_object* v___x_2789_; lean_object* v___x_2790_; lean_object* v___x_2791_; 
v___x_2788_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__7, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__7_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__7);
lean_inc_ref(v_fst_2560_);
v___x_2789_ = l_Lean_indentExpr(v_fst_2560_);
v___x_2790_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2790_, 0, v___x_2788_);
lean_ctor_set(v___x_2790_, 1, v___x_2789_);
lean_inc(v___x_2562_);
v___x_2791_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__5(v___x_2562_, v___x_2790_, v___y_2781_, v___y_2782_, v___y_2783_, v___y_2784_);
if (lean_obj_tag(v___x_2791_) == 0)
{
lean_dec_ref_known(v___x_2791_, 1);
v___y_2751_ = v___y_2780_;
v___y_2752_ = v___y_2781_;
v___y_2753_ = v___y_2782_;
v___y_2754_ = v___y_2783_;
v___y_2755_ = v___y_2784_;
goto v___jp_2750_;
}
else
{
lean_object* v_a_2792_; lean_object* v___x_2794_; uint8_t v_isShared_2795_; uint8_t v_isSharedCheck_2799_; 
lean_dec_ref(v___y_2780_);
lean_dec(v_a_2594_);
lean_dec_ref(v_snd_2563_);
lean_dec(v___x_2562_);
lean_dec_ref(v_rel_2561_);
lean_dec_ref(v_fst_2560_);
lean_dec(v_a_2559_);
lean_dec(v_a_2558_);
lean_dec_ref(v___f_2557_);
lean_dec(v_a_2554_);
v_a_2792_ = lean_ctor_get(v___x_2791_, 0);
v_isSharedCheck_2799_ = !lean_is_exclusive(v___x_2791_);
if (v_isSharedCheck_2799_ == 0)
{
v___x_2794_ = v___x_2791_;
v_isShared_2795_ = v_isSharedCheck_2799_;
goto v_resetjp_2793_;
}
else
{
lean_inc(v_a_2792_);
lean_dec(v___x_2791_);
v___x_2794_ = lean_box(0);
v_isShared_2795_ = v_isSharedCheck_2799_;
goto v_resetjp_2793_;
}
v_resetjp_2793_:
{
lean_object* v___x_2797_; 
if (v_isShared_2795_ == 0)
{
v___x_2797_ = v___x_2794_;
goto v_reusejp_2796_;
}
else
{
lean_object* v_reuseFailAlloc_2798_; 
v_reuseFailAlloc_2798_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2798_, 0, v_a_2792_);
v___x_2797_ = v_reuseFailAlloc_2798_;
goto v_reusejp_2796_;
}
v_reusejp_2796_:
{
return v___x_2797_;
}
}
}
}
}
else
{
lean_object* v_a_2800_; lean_object* v___x_2802_; uint8_t v_isShared_2803_; uint8_t v_isSharedCheck_2807_; 
lean_dec_ref(v___y_2780_);
lean_dec(v_a_2594_);
lean_dec_ref(v_snd_2563_);
lean_dec(v___x_2562_);
lean_dec_ref(v_rel_2561_);
lean_dec_ref(v_fst_2560_);
lean_dec(v_a_2559_);
lean_dec(v_a_2558_);
lean_dec_ref(v___f_2557_);
lean_dec(v_a_2554_);
v_a_2800_ = lean_ctor_get(v___x_2785_, 0);
v_isSharedCheck_2807_ = !lean_is_exclusive(v___x_2785_);
if (v_isSharedCheck_2807_ == 0)
{
v___x_2802_ = v___x_2785_;
v_isShared_2803_ = v_isSharedCheck_2807_;
goto v_resetjp_2801_;
}
else
{
lean_inc(v_a_2800_);
lean_dec(v___x_2785_);
v___x_2802_ = lean_box(0);
v_isShared_2803_ = v_isSharedCheck_2807_;
goto v_resetjp_2801_;
}
v_resetjp_2801_:
{
lean_object* v___x_2805_; 
if (v_isShared_2803_ == 0)
{
v___x_2805_ = v___x_2802_;
goto v_reusejp_2804_;
}
else
{
lean_object* v_reuseFailAlloc_2806_; 
v_reuseFailAlloc_2806_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2806_, 0, v_a_2800_);
v___x_2805_ = v_reuseFailAlloc_2806_;
goto v_reusejp_2804_;
}
v_reusejp_2804_:
{
return v___x_2805_;
}
}
}
}
v___jp_2808_:
{
lean_object* v___x_2814_; 
lean_inc_ref(v___f_2557_);
lean_inc(v___y_2813_);
lean_inc_ref(v___y_2812_);
lean_inc(v___y_2811_);
lean_inc_ref(v___y_2810_);
v___x_2814_ = lean_apply_5(v___f_2557_, v___y_2810_, v___y_2811_, v___y_2812_, v___y_2813_, lean_box(0));
if (lean_obj_tag(v___x_2814_) == 0)
{
lean_object* v_a_2815_; uint8_t v___x_2816_; 
v_a_2815_ = lean_ctor_get(v___x_2814_, 0);
lean_inc(v_a_2815_);
lean_dec_ref_known(v___x_2814_, 1);
v___x_2816_ = lean_unbox(v_a_2815_);
lean_dec(v_a_2815_);
if (v___x_2816_ == 0)
{
v___y_2780_ = v___y_2809_;
v___y_2781_ = v___y_2810_;
v___y_2782_ = v___y_2811_;
v___y_2783_ = v___y_2812_;
v___y_2784_ = v___y_2813_;
goto v___jp_2779_;
}
else
{
lean_object* v___x_2817_; lean_object* v___x_2818_; lean_object* v___x_2819_; lean_object* v___x_2820_; 
v___x_2817_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__9, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__9_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__9);
lean_inc_ref(v_rel_2561_);
v___x_2818_ = l_Lean_indentExpr(v_rel_2561_);
v___x_2819_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2819_, 0, v___x_2817_);
lean_ctor_set(v___x_2819_, 1, v___x_2818_);
lean_inc(v___x_2562_);
v___x_2820_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__5(v___x_2562_, v___x_2819_, v___y_2810_, v___y_2811_, v___y_2812_, v___y_2813_);
if (lean_obj_tag(v___x_2820_) == 0)
{
lean_dec_ref_known(v___x_2820_, 1);
v___y_2780_ = v___y_2809_;
v___y_2781_ = v___y_2810_;
v___y_2782_ = v___y_2811_;
v___y_2783_ = v___y_2812_;
v___y_2784_ = v___y_2813_;
goto v___jp_2779_;
}
else
{
lean_object* v_a_2821_; lean_object* v___x_2823_; uint8_t v_isShared_2824_; uint8_t v_isSharedCheck_2828_; 
lean_dec_ref(v___y_2809_);
lean_dec(v_a_2594_);
lean_dec_ref(v_snd_2563_);
lean_dec(v___x_2562_);
lean_dec_ref(v_rel_2561_);
lean_dec_ref(v_fst_2560_);
lean_dec(v_a_2559_);
lean_dec(v_a_2558_);
lean_dec_ref(v___f_2557_);
lean_dec(v_a_2554_);
v_a_2821_ = lean_ctor_get(v___x_2820_, 0);
v_isSharedCheck_2828_ = !lean_is_exclusive(v___x_2820_);
if (v_isSharedCheck_2828_ == 0)
{
v___x_2823_ = v___x_2820_;
v_isShared_2824_ = v_isSharedCheck_2828_;
goto v_resetjp_2822_;
}
else
{
lean_inc(v_a_2821_);
lean_dec(v___x_2820_);
v___x_2823_ = lean_box(0);
v_isShared_2824_ = v_isSharedCheck_2828_;
goto v_resetjp_2822_;
}
v_resetjp_2822_:
{
lean_object* v___x_2826_; 
if (v_isShared_2824_ == 0)
{
v___x_2826_ = v___x_2823_;
goto v_reusejp_2825_;
}
else
{
lean_object* v_reuseFailAlloc_2827_; 
v_reuseFailAlloc_2827_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2827_, 0, v_a_2821_);
v___x_2826_ = v_reuseFailAlloc_2827_;
goto v_reusejp_2825_;
}
v_reusejp_2825_:
{
return v___x_2826_;
}
}
}
}
}
else
{
lean_object* v_a_2829_; lean_object* v___x_2831_; uint8_t v_isShared_2832_; uint8_t v_isSharedCheck_2836_; 
lean_dec_ref(v___y_2809_);
lean_dec(v_a_2594_);
lean_dec_ref(v_snd_2563_);
lean_dec(v___x_2562_);
lean_dec_ref(v_rel_2561_);
lean_dec_ref(v_fst_2560_);
lean_dec(v_a_2559_);
lean_dec(v_a_2558_);
lean_dec_ref(v___f_2557_);
lean_dec(v_a_2554_);
v_a_2829_ = lean_ctor_get(v___x_2814_, 0);
v_isSharedCheck_2836_ = !lean_is_exclusive(v___x_2814_);
if (v_isSharedCheck_2836_ == 0)
{
v___x_2831_ = v___x_2814_;
v_isShared_2832_ = v_isSharedCheck_2836_;
goto v_resetjp_2830_;
}
else
{
lean_inc(v_a_2829_);
lean_dec(v___x_2814_);
v___x_2831_ = lean_box(0);
v_isShared_2832_ = v_isSharedCheck_2836_;
goto v_resetjp_2830_;
}
v_resetjp_2830_:
{
lean_object* v___x_2834_; 
if (v_isShared_2832_ == 0)
{
v___x_2834_ = v___x_2831_;
goto v_reusejp_2833_;
}
else
{
lean_object* v_reuseFailAlloc_2835_; 
v_reuseFailAlloc_2835_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2835_, 0, v_a_2829_);
v___x_2834_ = v_reuseFailAlloc_2835_;
goto v_reusejp_2833_;
}
v_reusejp_2833_:
{
return v___x_2834_;
}
}
}
}
v___jp_2837_:
{
lean_object* v___x_2843_; 
lean_inc_ref(v___f_2557_);
lean_inc(v___y_2839_);
lean_inc_ref(v___y_2841_);
lean_inc(v___y_2840_);
lean_inc_ref(v___y_2838_);
v___x_2843_ = lean_apply_5(v___f_2557_, v___y_2838_, v___y_2840_, v___y_2841_, v___y_2839_, lean_box(0));
if (lean_obj_tag(v___x_2843_) == 0)
{
lean_object* v_a_2844_; uint8_t v___x_2845_; 
v_a_2844_ = lean_ctor_get(v___x_2843_, 0);
lean_inc(v_a_2844_);
lean_dec_ref_known(v___x_2843_, 1);
v___x_2845_ = lean_unbox(v_a_2844_);
lean_dec(v_a_2844_);
if (v___x_2845_ == 0)
{
v___y_2809_ = v_a_2842_;
v___y_2810_ = v___y_2838_;
v___y_2811_ = v___y_2840_;
v___y_2812_ = v___y_2841_;
v___y_2813_ = v___y_2839_;
goto v___jp_2808_;
}
else
{
lean_object* v___x_2846_; lean_object* v___x_2847_; lean_object* v___x_2848_; lean_object* v___x_2849_; 
v___x_2846_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__11, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__11_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__11);
lean_inc_ref(v_a_2842_);
v___x_2847_ = l_Lean_MessageData_ofExpr(v_a_2842_);
v___x_2848_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2848_, 0, v___x_2846_);
lean_ctor_set(v___x_2848_, 1, v___x_2847_);
lean_inc(v___x_2562_);
v___x_2849_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__5(v___x_2562_, v___x_2848_, v___y_2838_, v___y_2840_, v___y_2841_, v___y_2839_);
if (lean_obj_tag(v___x_2849_) == 0)
{
lean_dec_ref_known(v___x_2849_, 1);
v___y_2809_ = v_a_2842_;
v___y_2810_ = v___y_2838_;
v___y_2811_ = v___y_2840_;
v___y_2812_ = v___y_2841_;
v___y_2813_ = v___y_2839_;
goto v___jp_2808_;
}
else
{
lean_object* v_a_2850_; lean_object* v___x_2852_; uint8_t v_isShared_2853_; uint8_t v_isSharedCheck_2857_; 
lean_dec_ref(v_a_2842_);
lean_dec(v_a_2594_);
lean_dec_ref(v_snd_2563_);
lean_dec(v___x_2562_);
lean_dec_ref(v_rel_2561_);
lean_dec_ref(v_fst_2560_);
lean_dec(v_a_2559_);
lean_dec(v_a_2558_);
lean_dec_ref(v___f_2557_);
lean_dec(v_a_2554_);
v_a_2850_ = lean_ctor_get(v___x_2849_, 0);
v_isSharedCheck_2857_ = !lean_is_exclusive(v___x_2849_);
if (v_isSharedCheck_2857_ == 0)
{
v___x_2852_ = v___x_2849_;
v_isShared_2853_ = v_isSharedCheck_2857_;
goto v_resetjp_2851_;
}
else
{
lean_inc(v_a_2850_);
lean_dec(v___x_2849_);
v___x_2852_ = lean_box(0);
v_isShared_2853_ = v_isSharedCheck_2857_;
goto v_resetjp_2851_;
}
v_resetjp_2851_:
{
lean_object* v___x_2855_; 
if (v_isShared_2853_ == 0)
{
v___x_2855_ = v___x_2852_;
goto v_reusejp_2854_;
}
else
{
lean_object* v_reuseFailAlloc_2856_; 
v_reuseFailAlloc_2856_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2856_, 0, v_a_2850_);
v___x_2855_ = v_reuseFailAlloc_2856_;
goto v_reusejp_2854_;
}
v_reusejp_2854_:
{
return v___x_2855_;
}
}
}
}
}
else
{
lean_object* v_a_2858_; lean_object* v___x_2860_; uint8_t v_isShared_2861_; uint8_t v_isSharedCheck_2865_; 
lean_dec_ref(v_a_2842_);
lean_dec(v_a_2594_);
lean_dec_ref(v_snd_2563_);
lean_dec(v___x_2562_);
lean_dec_ref(v_rel_2561_);
lean_dec_ref(v_fst_2560_);
lean_dec(v_a_2559_);
lean_dec(v_a_2558_);
lean_dec_ref(v___f_2557_);
lean_dec(v_a_2554_);
v_a_2858_ = lean_ctor_get(v___x_2843_, 0);
v_isSharedCheck_2865_ = !lean_is_exclusive(v___x_2843_);
if (v_isSharedCheck_2865_ == 0)
{
v___x_2860_ = v___x_2843_;
v_isShared_2861_ = v_isSharedCheck_2865_;
goto v_resetjp_2859_;
}
else
{
lean_inc(v_a_2858_);
lean_dec(v___x_2843_);
v___x_2860_ = lean_box(0);
v_isShared_2861_ = v_isSharedCheck_2865_;
goto v_resetjp_2859_;
}
v_resetjp_2859_:
{
lean_object* v___x_2863_; 
if (v_isShared_2861_ == 0)
{
v___x_2863_ = v___x_2860_;
goto v_reusejp_2862_;
}
else
{
lean_object* v_reuseFailAlloc_2864_; 
v_reuseFailAlloc_2864_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2864_, 0, v_a_2858_);
v___x_2863_ = v_reuseFailAlloc_2864_;
goto v_reusejp_2862_;
}
v_reusejp_2862_:
{
return v___x_2863_;
}
}
}
}
v___jp_2866_:
{
if (lean_obj_tag(v_a_2559_) == 0)
{
lean_object* v___x_2871_; uint8_t v___x_2872_; lean_object* v___x_2873_; lean_object* v___x_2874_; 
v___x_2871_ = lean_box(0);
v___x_2872_ = 0;
v___x_2873_ = lean_box(0);
v___x_2874_ = l_Lean_Meta_mkFreshExprMVar(v___x_2871_, v___x_2872_, v___x_2873_, v___y_2867_, v___y_2868_, v___y_2869_, v___y_2870_);
if (lean_obj_tag(v___x_2874_) == 0)
{
lean_object* v_a_2875_; 
v_a_2875_ = lean_ctor_get(v___x_2874_, 0);
lean_inc(v_a_2875_);
lean_dec_ref_known(v___x_2874_, 1);
v___y_2838_ = v___y_2867_;
v___y_2839_ = v___y_2870_;
v___y_2840_ = v___y_2868_;
v___y_2841_ = v___y_2869_;
v_a_2842_ = v_a_2875_;
goto v___jp_2837_;
}
else
{
lean_object* v_a_2876_; lean_object* v___x_2878_; uint8_t v_isShared_2879_; uint8_t v_isSharedCheck_2883_; 
lean_dec(v_a_2594_);
lean_dec_ref(v_snd_2563_);
lean_dec(v___x_2562_);
lean_dec_ref(v_rel_2561_);
lean_dec_ref(v_fst_2560_);
lean_dec(v_a_2558_);
lean_dec_ref(v___f_2557_);
lean_dec(v_a_2554_);
v_a_2876_ = lean_ctor_get(v___x_2874_, 0);
v_isSharedCheck_2883_ = !lean_is_exclusive(v___x_2874_);
if (v_isSharedCheck_2883_ == 0)
{
v___x_2878_ = v___x_2874_;
v_isShared_2879_ = v_isSharedCheck_2883_;
goto v_resetjp_2877_;
}
else
{
lean_inc(v_a_2876_);
lean_dec(v___x_2874_);
v___x_2878_ = lean_box(0);
v_isShared_2879_ = v_isSharedCheck_2883_;
goto v_resetjp_2877_;
}
v_resetjp_2877_:
{
lean_object* v___x_2881_; 
if (v_isShared_2879_ == 0)
{
v___x_2881_ = v___x_2878_;
goto v_reusejp_2880_;
}
else
{
lean_object* v_reuseFailAlloc_2882_; 
v_reuseFailAlloc_2882_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2882_, 0, v_a_2876_);
v___x_2881_ = v_reuseFailAlloc_2882_;
goto v_reusejp_2880_;
}
v_reusejp_2880_:
{
return v___x_2881_;
}
}
}
}
else
{
lean_object* v_val_2884_; lean_object* v_fst_2885_; 
v_val_2884_ = lean_ctor_get(v_a_2559_, 0);
v_fst_2885_ = lean_ctor_get(v_val_2884_, 0);
lean_inc(v_fst_2885_);
v___y_2838_ = v___y_2867_;
v___y_2839_ = v___y_2870_;
v___y_2840_ = v___y_2868_;
v___y_2841_ = v___y_2869_;
v_a_2842_ = v_fst_2885_;
goto v___jp_2837_;
}
}
v___jp_2886_:
{
lean_object* v___x_2891_; 
lean_inc_ref(v___f_2557_);
lean_inc(v___y_2890_);
lean_inc_ref(v___y_2889_);
lean_inc(v___y_2888_);
lean_inc_ref(v___y_2887_);
v___x_2891_ = lean_apply_5(v___f_2557_, v___y_2887_, v___y_2888_, v___y_2889_, v___y_2890_, lean_box(0));
if (lean_obj_tag(v___x_2891_) == 0)
{
lean_object* v_a_2892_; uint8_t v___x_2893_; 
v_a_2892_ = lean_ctor_get(v___x_2891_, 0);
lean_inc(v_a_2892_);
lean_dec_ref_known(v___x_2891_, 1);
v___x_2893_ = lean_unbox(v_a_2892_);
lean_dec(v_a_2892_);
if (v___x_2893_ == 0)
{
lean_dec(v_a_2578_);
v___y_2867_ = v___y_2887_;
v___y_2868_ = v___y_2888_;
v___y_2869_ = v___y_2889_;
v___y_2870_ = v___y_2890_;
goto v___jp_2866_;
}
else
{
lean_object* v___x_2894_; lean_object* v___x_2895_; lean_object* v___x_2896_; lean_object* v___x_2897_; 
v___x_2894_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__13, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__13_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__13);
v___x_2895_ = l_Lean_MessageData_ofExpr(v_a_2578_);
v___x_2896_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2896_, 0, v___x_2894_);
lean_ctor_set(v___x_2896_, 1, v___x_2895_);
lean_inc(v___x_2562_);
v___x_2897_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__5(v___x_2562_, v___x_2896_, v___y_2887_, v___y_2888_, v___y_2889_, v___y_2890_);
if (lean_obj_tag(v___x_2897_) == 0)
{
lean_dec_ref_known(v___x_2897_, 1);
v___y_2867_ = v___y_2887_;
v___y_2868_ = v___y_2888_;
v___y_2869_ = v___y_2889_;
v___y_2870_ = v___y_2890_;
goto v___jp_2866_;
}
else
{
lean_object* v_a_2898_; lean_object* v___x_2900_; uint8_t v_isShared_2901_; uint8_t v_isSharedCheck_2905_; 
lean_dec(v_a_2594_);
lean_dec_ref(v_snd_2563_);
lean_dec(v___x_2562_);
lean_dec_ref(v_rel_2561_);
lean_dec_ref(v_fst_2560_);
lean_dec(v_a_2559_);
lean_dec(v_a_2558_);
lean_dec_ref(v___f_2557_);
lean_dec(v_a_2554_);
v_a_2898_ = lean_ctor_get(v___x_2897_, 0);
v_isSharedCheck_2905_ = !lean_is_exclusive(v___x_2897_);
if (v_isSharedCheck_2905_ == 0)
{
v___x_2900_ = v___x_2897_;
v_isShared_2901_ = v_isSharedCheck_2905_;
goto v_resetjp_2899_;
}
else
{
lean_inc(v_a_2898_);
lean_dec(v___x_2897_);
v___x_2900_ = lean_box(0);
v_isShared_2901_ = v_isSharedCheck_2905_;
goto v_resetjp_2899_;
}
v_resetjp_2899_:
{
lean_object* v___x_2903_; 
if (v_isShared_2901_ == 0)
{
v___x_2903_ = v___x_2900_;
goto v_reusejp_2902_;
}
else
{
lean_object* v_reuseFailAlloc_2904_; 
v_reuseFailAlloc_2904_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2904_, 0, v_a_2898_);
v___x_2903_ = v_reuseFailAlloc_2904_;
goto v_reusejp_2902_;
}
v_reusejp_2902_:
{
return v___x_2903_;
}
}
}
}
}
else
{
lean_object* v_a_2906_; lean_object* v___x_2908_; uint8_t v_isShared_2909_; uint8_t v_isSharedCheck_2913_; 
lean_dec(v_a_2594_);
lean_dec(v_a_2578_);
lean_dec_ref(v_snd_2563_);
lean_dec(v___x_2562_);
lean_dec_ref(v_rel_2561_);
lean_dec_ref(v_fst_2560_);
lean_dec(v_a_2559_);
lean_dec(v_a_2558_);
lean_dec_ref(v___f_2557_);
lean_dec(v_a_2554_);
v_a_2906_ = lean_ctor_get(v___x_2891_, 0);
v_isSharedCheck_2913_ = !lean_is_exclusive(v___x_2891_);
if (v_isSharedCheck_2913_ == 0)
{
v___x_2908_ = v___x_2891_;
v_isShared_2909_ = v_isSharedCheck_2913_;
goto v_resetjp_2907_;
}
else
{
lean_inc(v_a_2906_);
lean_dec(v___x_2891_);
v___x_2908_ = lean_box(0);
v_isShared_2909_ = v_isSharedCheck_2913_;
goto v_resetjp_2907_;
}
v_resetjp_2907_:
{
lean_object* v___x_2911_; 
if (v_isShared_2909_ == 0)
{
v___x_2911_ = v___x_2908_;
goto v_reusejp_2910_;
}
else
{
lean_object* v_reuseFailAlloc_2912_; 
v_reuseFailAlloc_2912_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2912_, 0, v_a_2906_);
v___x_2911_ = v_reuseFailAlloc_2912_;
goto v_reusejp_2910_;
}
v_reusejp_2910_:
{
return v___x_2911_;
}
}
}
}
}
else
{
lean_object* v_a_2939_; lean_object* v___x_2941_; uint8_t v_isShared_2942_; uint8_t v_isSharedCheck_2946_; 
lean_dec(v_a_2578_);
lean_dec_ref(v_snd_2563_);
lean_dec(v___x_2562_);
lean_dec_ref(v_rel_2561_);
lean_dec_ref(v_fst_2560_);
lean_dec(v_a_2559_);
lean_dec(v_a_2558_);
lean_dec_ref(v___f_2557_);
lean_dec(v_a_2554_);
v_a_2939_ = lean_ctor_get(v___x_2593_, 0);
v_isSharedCheck_2946_ = !lean_is_exclusive(v___x_2593_);
if (v_isSharedCheck_2946_ == 0)
{
v___x_2941_ = v___x_2593_;
v_isShared_2942_ = v_isSharedCheck_2946_;
goto v_resetjp_2940_;
}
else
{
lean_inc(v_a_2939_);
lean_dec(v___x_2593_);
v___x_2941_ = lean_box(0);
v_isShared_2942_ = v_isSharedCheck_2946_;
goto v_resetjp_2940_;
}
v_resetjp_2940_:
{
lean_object* v___x_2944_; 
if (v_isShared_2942_ == 0)
{
v___x_2944_ = v___x_2941_;
goto v_reusejp_2943_;
}
else
{
lean_object* v_reuseFailAlloc_2945_; 
v_reuseFailAlloc_2945_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2945_, 0, v_a_2939_);
v___x_2944_ = v_reuseFailAlloc_2945_;
goto v_reusejp_2943_;
}
v_reusejp_2943_:
{
return v___x_2944_;
}
}
}
}
else
{
lean_object* v_a_2947_; lean_object* v___x_2949_; uint8_t v_isShared_2950_; uint8_t v_isSharedCheck_2954_; 
lean_dec_ref(v_snd_2563_);
lean_dec(v___x_2562_);
lean_dec_ref(v_rel_2561_);
lean_dec_ref(v_fst_2560_);
lean_dec(v_a_2559_);
lean_dec(v_a_2558_);
lean_dec_ref(v___f_2557_);
lean_dec_ref(v___f_2555_);
lean_dec(v_a_2554_);
v_a_2947_ = lean_ctor_get(v___x_2577_, 0);
v_isSharedCheck_2954_ = !lean_is_exclusive(v___x_2577_);
if (v_isSharedCheck_2954_ == 0)
{
v___x_2949_ = v___x_2577_;
v_isShared_2950_ = v_isSharedCheck_2954_;
goto v_resetjp_2948_;
}
else
{
lean_inc(v_a_2947_);
lean_dec(v___x_2577_);
v___x_2949_ = lean_box(0);
v_isShared_2950_ = v_isSharedCheck_2954_;
goto v_resetjp_2948_;
}
v_resetjp_2948_:
{
lean_object* v___x_2952_; 
if (v_isShared_2950_ == 0)
{
v___x_2952_ = v___x_2949_;
goto v_reusejp_2951_;
}
else
{
lean_object* v_reuseFailAlloc_2953_; 
v_reuseFailAlloc_2953_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2953_, 0, v_a_2947_);
v___x_2952_ = v_reuseFailAlloc_2953_;
goto v_reusejp_2951_;
}
v_reusejp_2951_:
{
return v___x_2952_;
}
}
}
}
else
{
lean_object* v_a_2955_; lean_object* v___x_2957_; uint8_t v_isShared_2958_; uint8_t v_isSharedCheck_2962_; 
lean_dec_ref(v_snd_2563_);
lean_dec(v___x_2562_);
lean_dec_ref(v_rel_2561_);
lean_dec_ref(v_fst_2560_);
lean_dec(v_a_2559_);
lean_dec(v_a_2558_);
lean_dec_ref(v___f_2557_);
lean_dec_ref(v___f_2555_);
lean_dec(v_a_2554_);
v_a_2955_ = lean_ctor_get(v___x_2575_, 0);
v_isSharedCheck_2962_ = !lean_is_exclusive(v___x_2575_);
if (v_isSharedCheck_2962_ == 0)
{
v___x_2957_ = v___x_2575_;
v_isShared_2958_ = v_isSharedCheck_2962_;
goto v_resetjp_2956_;
}
else
{
lean_inc(v_a_2955_);
lean_dec(v___x_2575_);
v___x_2957_ = lean_box(0);
v_isShared_2958_ = v_isSharedCheck_2962_;
goto v_resetjp_2956_;
}
v_resetjp_2956_:
{
lean_object* v___x_2960_; 
if (v_isShared_2958_ == 0)
{
v___x_2960_ = v___x_2957_;
goto v_reusejp_2959_;
}
else
{
lean_object* v_reuseFailAlloc_2961_; 
v_reuseFailAlloc_2961_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2961_, 0, v_a_2955_);
v___x_2960_ = v_reuseFailAlloc_2961_;
goto v_reusejp_2959_;
}
v_reusejp_2959_:
{
return v___x_2960_;
}
}
}
v___jp_2570_:
{
lean_object* v___x_2573_; lean_object* v___x_2574_; 
v___x_2573_ = l_List_appendTR___redArg(v___y_2571_, v___y_2572_);
v___x_2574_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2574_, 0, v___x_2573_);
return v___x_2574_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___boxed(lean_object* v_a_2963_, lean_object* v___f_2964_, lean_object* v___y_2965_, lean_object* v___f_2966_, lean_object* v_a_2967_, lean_object* v_a_2968_, lean_object* v_fst_2969_, lean_object* v_rel_2970_, lean_object* v___x_2971_, lean_object* v_snd_2972_, lean_object* v_____r_2973_, lean_object* v___y_2974_, lean_object* v___y_2975_, lean_object* v___y_2976_, lean_object* v___y_2977_, lean_object* v___y_2978_){
_start:
{
uint8_t v___y_92554__boxed_2979_; lean_object* v_res_2980_; 
v___y_92554__boxed_2979_ = lean_unbox(v___y_2965_);
v_res_2980_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2(v_a_2963_, v___f_2964_, v___y_92554__boxed_2979_, v___f_2966_, v_a_2967_, v_a_2968_, v_fst_2969_, v_rel_2970_, v___x_2971_, v_snd_2972_, v_____r_2973_, v___y_2974_, v___y_2975_, v___y_2976_, v___y_2977_);
lean_dec(v___y_2977_);
lean_dec_ref(v___y_2976_);
lean_dec(v___y_2975_);
lean_dec_ref(v___y_2974_);
return v_res_2980_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__0___closed__1(void){
_start:
{
lean_object* v___x_2982_; lean_object* v___x_2983_; 
v___x_2982_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__0___closed__0));
v___x_2983_ = l_Lean_stringToMessageData(v___x_2982_);
return v___x_2983_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__0(lean_object* v___f_2984_, lean_object* v___x_2985_, lean_object* v_a_2986_, lean_object* v___f_2987_, uint8_t v___y_2988_, lean_object* v_a_2989_, lean_object* v_fst_2990_, lean_object* v_rel_2991_, lean_object* v___x_2992_, lean_object* v_snd_2993_, lean_object* v___y_2994_, lean_object* v___y_2995_, lean_object* v___y_2996_, lean_object* v___y_2997_, lean_object* v___y_2998_, lean_object* v___y_2999_, lean_object* v___y_3000_, lean_object* v___y_3001_){
_start:
{
lean_object* v___y_3004_; lean_object* v___x_3023_; 
v___x_3023_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_2995_, v___y_2998_, v___y_2999_, v___y_3000_, v___y_3001_);
if (lean_obj_tag(v___x_3023_) == 0)
{
lean_object* v_a_3024_; lean_object* v___x_3025_; 
v_a_3024_ = lean_ctor_get(v___x_3023_, 0);
lean_inc(v_a_3024_);
lean_dec_ref_known(v___x_3023_, 1);
lean_inc_ref(v___f_2984_);
lean_inc(v___y_3001_);
lean_inc_ref(v___y_3000_);
lean_inc(v___y_2999_);
lean_inc_ref(v___y_2998_);
v___x_3025_ = lean_apply_5(v___f_2984_, v___y_2998_, v___y_2999_, v___y_3000_, v___y_3001_, lean_box(0));
if (lean_obj_tag(v___x_3025_) == 0)
{
lean_object* v_a_3026_; uint8_t v___x_3027_; 
v_a_3026_ = lean_ctor_get(v___x_3025_, 0);
lean_inc(v_a_3026_);
lean_dec_ref_known(v___x_3025_, 1);
v___x_3027_ = lean_unbox(v_a_3026_);
lean_dec(v_a_3026_);
if (v___x_3027_ == 0)
{
lean_object* v___x_3028_; 
v___x_3028_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2(v_a_2986_, v___f_2987_, v___y_2988_, v___f_2984_, v_a_3024_, v_a_2989_, v_fst_2990_, v_rel_2991_, v___x_2992_, v_snd_2993_, v___x_2985_, v___y_2998_, v___y_2999_, v___y_3000_, v___y_3001_);
v___y_3004_ = v___x_3028_;
goto v___jp_3003_;
}
else
{
lean_object* v___x_3029_; lean_object* v___x_3030_; lean_object* v___x_3031_; lean_object* v___x_3032_; 
v___x_3029_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__0___closed__1, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__0___closed__1_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__0___closed__1);
lean_inc(v_a_2986_);
v___x_3030_ = l_Lean_MessageData_ofName(v_a_2986_);
v___x_3031_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3031_, 0, v___x_3029_);
lean_ctor_set(v___x_3031_, 1, v___x_3030_);
lean_inc(v___x_2992_);
v___x_3032_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__5(v___x_2992_, v___x_3031_, v___y_2998_, v___y_2999_, v___y_3000_, v___y_3001_);
if (lean_obj_tag(v___x_3032_) == 0)
{
lean_object* v_a_3033_; lean_object* v___x_3034_; 
v_a_3033_ = lean_ctor_get(v___x_3032_, 0);
lean_inc(v_a_3033_);
lean_dec_ref_known(v___x_3032_, 1);
v___x_3034_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2(v_a_2986_, v___f_2987_, v___y_2988_, v___f_2984_, v_a_3024_, v_a_2989_, v_fst_2990_, v_rel_2991_, v___x_2992_, v_snd_2993_, v_a_3033_, v___y_2998_, v___y_2999_, v___y_3000_, v___y_3001_);
v___y_3004_ = v___x_3034_;
goto v___jp_3003_;
}
else
{
lean_dec(v_a_3024_);
lean_dec(v___y_3001_);
lean_dec_ref(v___y_3000_);
lean_dec(v___y_2999_);
lean_dec_ref(v___y_2998_);
lean_dec_ref(v_snd_2993_);
lean_dec(v___x_2992_);
lean_dec_ref(v_rel_2991_);
lean_dec_ref(v_fst_2990_);
lean_dec(v_a_2989_);
lean_dec_ref(v___f_2987_);
lean_dec(v_a_2986_);
lean_dec_ref(v___f_2984_);
return v___x_3032_;
}
}
}
else
{
lean_object* v_a_3035_; lean_object* v___x_3037_; uint8_t v_isShared_3038_; uint8_t v_isSharedCheck_3042_; 
lean_dec(v_a_3024_);
lean_dec(v___y_3001_);
lean_dec_ref(v___y_3000_);
lean_dec(v___y_2999_);
lean_dec_ref(v___y_2998_);
lean_dec_ref(v_snd_2993_);
lean_dec(v___x_2992_);
lean_dec_ref(v_rel_2991_);
lean_dec_ref(v_fst_2990_);
lean_dec(v_a_2989_);
lean_dec_ref(v___f_2987_);
lean_dec(v_a_2986_);
lean_dec_ref(v___f_2984_);
v_a_3035_ = lean_ctor_get(v___x_3025_, 0);
v_isSharedCheck_3042_ = !lean_is_exclusive(v___x_3025_);
if (v_isSharedCheck_3042_ == 0)
{
v___x_3037_ = v___x_3025_;
v_isShared_3038_ = v_isSharedCheck_3042_;
goto v_resetjp_3036_;
}
else
{
lean_inc(v_a_3035_);
lean_dec(v___x_3025_);
v___x_3037_ = lean_box(0);
v_isShared_3038_ = v_isSharedCheck_3042_;
goto v_resetjp_3036_;
}
v_resetjp_3036_:
{
lean_object* v___x_3040_; 
if (v_isShared_3038_ == 0)
{
v___x_3040_ = v___x_3037_;
goto v_reusejp_3039_;
}
else
{
lean_object* v_reuseFailAlloc_3041_; 
v_reuseFailAlloc_3041_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3041_, 0, v_a_3035_);
v___x_3040_ = v_reuseFailAlloc_3041_;
goto v_reusejp_3039_;
}
v_reusejp_3039_:
{
return v___x_3040_;
}
}
}
}
else
{
lean_object* v_a_3043_; lean_object* v___x_3045_; uint8_t v_isShared_3046_; uint8_t v_isSharedCheck_3050_; 
lean_dec(v___y_3001_);
lean_dec_ref(v___y_3000_);
lean_dec(v___y_2999_);
lean_dec_ref(v___y_2998_);
lean_dec_ref(v_snd_2993_);
lean_dec(v___x_2992_);
lean_dec_ref(v_rel_2991_);
lean_dec_ref(v_fst_2990_);
lean_dec(v_a_2989_);
lean_dec_ref(v___f_2987_);
lean_dec(v_a_2986_);
lean_dec_ref(v___f_2984_);
v_a_3043_ = lean_ctor_get(v___x_3023_, 0);
v_isSharedCheck_3050_ = !lean_is_exclusive(v___x_3023_);
if (v_isSharedCheck_3050_ == 0)
{
v___x_3045_ = v___x_3023_;
v_isShared_3046_ = v_isSharedCheck_3050_;
goto v_resetjp_3044_;
}
else
{
lean_inc(v_a_3043_);
lean_dec(v___x_3023_);
v___x_3045_ = lean_box(0);
v_isShared_3046_ = v_isSharedCheck_3050_;
goto v_resetjp_3044_;
}
v_resetjp_3044_:
{
lean_object* v___x_3048_; 
if (v_isShared_3046_ == 0)
{
v___x_3048_ = v___x_3045_;
goto v_reusejp_3047_;
}
else
{
lean_object* v_reuseFailAlloc_3049_; 
v_reuseFailAlloc_3049_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3049_, 0, v_a_3043_);
v___x_3048_ = v_reuseFailAlloc_3049_;
goto v_reusejp_3047_;
}
v_reusejp_3047_:
{
return v___x_3048_;
}
}
}
v___jp_3003_:
{
if (lean_obj_tag(v___y_3004_) == 0)
{
lean_object* v_a_3005_; lean_object* v___x_3006_; 
v_a_3005_ = lean_ctor_get(v___y_3004_, 0);
lean_inc(v_a_3005_);
lean_dec_ref_known(v___y_3004_, 1);
v___x_3006_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v_a_3005_, v___y_2995_, v___y_2998_, v___y_2999_, v___y_3000_, v___y_3001_);
lean_dec(v___y_3001_);
lean_dec_ref(v___y_3000_);
lean_dec(v___y_2999_);
lean_dec_ref(v___y_2998_);
if (lean_obj_tag(v___x_3006_) == 0)
{
lean_object* v___x_3008_; uint8_t v_isShared_3009_; uint8_t v_isSharedCheck_3013_; 
v_isSharedCheck_3013_ = !lean_is_exclusive(v___x_3006_);
if (v_isSharedCheck_3013_ == 0)
{
lean_object* v_unused_3014_; 
v_unused_3014_ = lean_ctor_get(v___x_3006_, 0);
lean_dec(v_unused_3014_);
v___x_3008_ = v___x_3006_;
v_isShared_3009_ = v_isSharedCheck_3013_;
goto v_resetjp_3007_;
}
else
{
lean_dec(v___x_3006_);
v___x_3008_ = lean_box(0);
v_isShared_3009_ = v_isSharedCheck_3013_;
goto v_resetjp_3007_;
}
v_resetjp_3007_:
{
lean_object* v___x_3011_; 
if (v_isShared_3009_ == 0)
{
lean_ctor_set(v___x_3008_, 0, v___x_2985_);
v___x_3011_ = v___x_3008_;
goto v_reusejp_3010_;
}
else
{
lean_object* v_reuseFailAlloc_3012_; 
v_reuseFailAlloc_3012_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3012_, 0, v___x_2985_);
v___x_3011_ = v_reuseFailAlloc_3012_;
goto v_reusejp_3010_;
}
v_reusejp_3010_:
{
return v___x_3011_;
}
}
}
else
{
return v___x_3006_;
}
}
else
{
lean_object* v_a_3015_; lean_object* v___x_3017_; uint8_t v_isShared_3018_; uint8_t v_isSharedCheck_3022_; 
lean_dec(v___y_3001_);
lean_dec_ref(v___y_3000_);
lean_dec(v___y_2999_);
lean_dec_ref(v___y_2998_);
v_a_3015_ = lean_ctor_get(v___y_3004_, 0);
v_isSharedCheck_3022_ = !lean_is_exclusive(v___y_3004_);
if (v_isSharedCheck_3022_ == 0)
{
v___x_3017_ = v___y_3004_;
v_isShared_3018_ = v_isSharedCheck_3022_;
goto v_resetjp_3016_;
}
else
{
lean_inc(v_a_3015_);
lean_dec(v___y_3004_);
v___x_3017_ = lean_box(0);
v_isShared_3018_ = v_isSharedCheck_3022_;
goto v_resetjp_3016_;
}
v_resetjp_3016_:
{
lean_object* v___x_3020_; 
if (v_isShared_3018_ == 0)
{
v___x_3020_ = v___x_3017_;
goto v_reusejp_3019_;
}
else
{
lean_object* v_reuseFailAlloc_3021_; 
v_reuseFailAlloc_3021_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3021_, 0, v_a_3015_);
v___x_3020_ = v_reuseFailAlloc_3021_;
goto v_reusejp_3019_;
}
v_reusejp_3019_:
{
return v___x_3020_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__0___boxed(lean_object** _args){
lean_object* v___f_3051_ = _args[0];
lean_object* v___x_3052_ = _args[1];
lean_object* v_a_3053_ = _args[2];
lean_object* v___f_3054_ = _args[3];
lean_object* v___y_3055_ = _args[4];
lean_object* v_a_3056_ = _args[5];
lean_object* v_fst_3057_ = _args[6];
lean_object* v_rel_3058_ = _args[7];
lean_object* v___x_3059_ = _args[8];
lean_object* v_snd_3060_ = _args[9];
lean_object* v___y_3061_ = _args[10];
lean_object* v___y_3062_ = _args[11];
lean_object* v___y_3063_ = _args[12];
lean_object* v___y_3064_ = _args[13];
lean_object* v___y_3065_ = _args[14];
lean_object* v___y_3066_ = _args[15];
lean_object* v___y_3067_ = _args[16];
lean_object* v___y_3068_ = _args[17];
lean_object* v___y_3069_ = _args[18];
_start:
{
uint8_t v___y_93397__boxed_3070_; lean_object* v_res_3071_; 
v___y_93397__boxed_3070_ = lean_unbox(v___y_3055_);
v_res_3071_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__0(v___f_3051_, v___x_3052_, v_a_3053_, v___f_3054_, v___y_93397__boxed_3070_, v_a_3056_, v_fst_3057_, v_rel_3058_, v___x_3059_, v_snd_3060_, v___y_3061_, v___y_3062_, v___y_3063_, v___y_3064_, v___y_3065_, v___y_3066_, v___y_3067_, v___y_3068_);
lean_dec(v___y_3064_);
lean_dec_ref(v___y_3063_);
lean_dec(v___y_3062_);
lean_dec_ref(v___y_3061_);
return v_res_3071_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__3(void){
_start:
{
lean_object* v___x_3078_; lean_object* v___x_3079_; lean_object* v___x_3080_; 
v___x_3078_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__2_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_));
v___x_3079_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__0___closed__1));
v___x_3080_ = l_Lean_Name_append(v___x_3079_, v___x_3078_);
return v___x_3080_;
}
}
static lean_object* _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__5(void){
_start:
{
lean_object* v___x_3082_; lean_object* v___x_3083_; 
v___x_3082_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__4));
v___x_3083_ = l_Lean_stringToMessageData(v___x_3082_);
return v___x_3083_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10(uint8_t v___y_3084_, lean_object* v_snd_3085_, lean_object* v_rel_3086_, lean_object* v_fst_3087_, lean_object* v_a_3088_, lean_object* v_a_3089_, lean_object* v_as_3090_, size_t v_sz_3091_, size_t v_i_3092_, lean_object* v_b_3093_, lean_object* v___y_3094_, lean_object* v___y_3095_, lean_object* v___y_3096_, lean_object* v___y_3097_, lean_object* v___y_3098_, lean_object* v___y_3099_, lean_object* v___y_3100_, lean_object* v___y_3101_){
_start:
{
uint8_t v___x_3103_; 
v___x_3103_ = lean_usize_dec_lt(v_i_3092_, v_sz_3091_);
if (v___x_3103_ == 0)
{
lean_object* v___x_3104_; 
lean_dec_ref(v_a_3089_);
lean_dec(v_a_3088_);
lean_dec_ref(v_fst_3087_);
lean_dec_ref(v_rel_3086_);
lean_dec_ref(v_snd_3085_);
v___x_3104_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3104_, 0, v_b_3093_);
return v___x_3104_;
}
else
{
lean_object* v___x_3105_; 
lean_dec_ref(v_b_3093_);
v___x_3105_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_3095_, v___y_3097_, v___y_3099_, v___y_3101_);
if (lean_obj_tag(v___x_3105_) == 0)
{
lean_object* v_a_3106_; lean_object* v___x_3108_; uint8_t v_isShared_3109_; uint8_t v_isSharedCheck_3186_; 
v_a_3106_ = lean_ctor_get(v___x_3105_, 0);
v_isSharedCheck_3186_ = !lean_is_exclusive(v___x_3105_);
if (v_isSharedCheck_3186_ == 0)
{
v___x_3108_ = v___x_3105_;
v_isShared_3109_ = v_isSharedCheck_3186_;
goto v_resetjp_3107_;
}
else
{
lean_inc(v_a_3106_);
lean_dec(v___x_3105_);
v___x_3108_ = lean_box(0);
v_isShared_3109_ = v_isSharedCheck_3186_;
goto v_resetjp_3107_;
}
v_resetjp_3107_:
{
lean_object* v___f_3110_; lean_object* v___x_3111_; lean_object* v_a_3113_; lean_object* v___x_3119_; lean_object* v___f_3120_; lean_object* v_a_3121_; lean_object* v___x_3122_; lean_object* v___f_3123_; lean_object* v___x_3124_; 
v___f_3110_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__0));
v___x_3111_ = lean_box(0);
v___x_3119_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__2_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_));
v___f_3120_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__1));
v_a_3121_ = lean_array_uget_borrowed(v_as_3090_, v_i_3092_);
v___x_3122_ = lean_box(v___y_3084_);
lean_inc_ref(v_snd_3085_);
lean_inc_ref(v_rel_3086_);
lean_inc_ref(v_fst_3087_);
lean_inc(v_a_3088_);
lean_inc(v_a_3121_);
v___f_3123_ = lean_alloc_closure((void*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__0___boxed), 19, 10);
lean_closure_set(v___f_3123_, 0, v___f_3120_);
lean_closure_set(v___f_3123_, 1, v___x_3111_);
lean_closure_set(v___f_3123_, 2, v_a_3121_);
lean_closure_set(v___f_3123_, 3, v___f_3110_);
lean_closure_set(v___f_3123_, 4, v___x_3122_);
lean_closure_set(v___f_3123_, 5, v_a_3088_);
lean_closure_set(v___f_3123_, 6, v_fst_3087_);
lean_closure_set(v___f_3123_, 7, v_rel_3086_);
lean_closure_set(v___f_3123_, 8, v___x_3119_);
lean_closure_set(v___f_3123_, 9, v_snd_3085_);
v___x_3124_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_3123_, v___y_3094_, v___y_3095_, v___y_3096_, v___y_3097_, v___y_3098_, v___y_3099_, v___y_3100_, v___y_3101_);
if (lean_obj_tag(v___x_3124_) == 0)
{
lean_dec_ref_known(v___x_3124_, 1);
lean_dec(v_a_3106_);
lean_dec_ref(v_a_3089_);
lean_dec(v_a_3088_);
lean_dec_ref(v_fst_3087_);
lean_dec_ref(v_rel_3086_);
lean_dec_ref(v_snd_3085_);
v_a_3113_ = v___x_3111_;
goto v___jp_3112_;
}
else
{
lean_object* v_a_3125_; lean_object* v___x_3127_; uint8_t v_isShared_3128_; uint8_t v_isSharedCheck_3185_; 
v_a_3125_ = lean_ctor_get(v___x_3124_, 0);
v_isSharedCheck_3185_ = !lean_is_exclusive(v___x_3124_);
if (v_isSharedCheck_3185_ == 0)
{
v___x_3127_ = v___x_3124_;
v_isShared_3128_ = v_isSharedCheck_3185_;
goto v_resetjp_3126_;
}
else
{
lean_inc(v_a_3125_);
lean_dec(v___x_3124_);
v___x_3127_ = lean_box(0);
v_isShared_3128_ = v_isSharedCheck_3185_;
goto v_resetjp_3126_;
}
v_resetjp_3126_:
{
lean_object* v___x_3129_; lean_object* v___y_3131_; lean_object* v___y_3146_; uint8_t v___y_3149_; uint8_t v___x_3183_; 
v___x_3129_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__2));
v___x_3183_ = l_Lean_Exception_isInterrupt(v_a_3125_);
if (v___x_3183_ == 0)
{
uint8_t v___x_3184_; 
lean_inc(v_a_3125_);
v___x_3184_ = l_Lean_Exception_isRuntime(v_a_3125_);
v___y_3149_ = v___x_3184_;
goto v___jp_3148_;
}
else
{
v___y_3149_ = v___x_3183_;
goto v___jp_3148_;
}
v___jp_3130_:
{
if (lean_obj_tag(v___y_3131_) == 0)
{
lean_object* v_a_3132_; 
v_a_3132_ = lean_ctor_get(v___y_3131_, 0);
lean_inc(v_a_3132_);
lean_dec_ref_known(v___y_3131_, 1);
if (lean_obj_tag(v_a_3132_) == 0)
{
lean_object* v_a_3133_; 
lean_dec_ref(v_a_3089_);
lean_dec(v_a_3088_);
lean_dec_ref(v_fst_3087_);
lean_dec_ref(v_rel_3086_);
lean_dec_ref(v_snd_3085_);
v_a_3133_ = lean_ctor_get(v_a_3132_, 0);
lean_inc(v_a_3133_);
lean_dec_ref_known(v_a_3132_, 1);
v_a_3113_ = v_a_3133_;
goto v___jp_3112_;
}
else
{
size_t v___x_3134_; size_t v___x_3135_; 
lean_dec_ref_known(v_a_3132_, 1);
lean_del_object(v___x_3108_);
v___x_3134_ = ((size_t)1ULL);
v___x_3135_ = lean_usize_add(v_i_3092_, v___x_3134_);
v_i_3092_ = v___x_3135_;
v_b_3093_ = v___x_3129_;
goto _start;
}
}
else
{
lean_object* v_a_3137_; lean_object* v___x_3139_; uint8_t v_isShared_3140_; uint8_t v_isSharedCheck_3144_; 
lean_del_object(v___x_3108_);
lean_dec_ref(v_a_3089_);
lean_dec(v_a_3088_);
lean_dec_ref(v_fst_3087_);
lean_dec_ref(v_rel_3086_);
lean_dec_ref(v_snd_3085_);
v_a_3137_ = lean_ctor_get(v___y_3131_, 0);
v_isSharedCheck_3144_ = !lean_is_exclusive(v___y_3131_);
if (v_isSharedCheck_3144_ == 0)
{
v___x_3139_ = v___y_3131_;
v_isShared_3140_ = v_isSharedCheck_3144_;
goto v_resetjp_3138_;
}
else
{
lean_inc(v_a_3137_);
lean_dec(v___y_3131_);
v___x_3139_ = lean_box(0);
v_isShared_3140_ = v_isSharedCheck_3144_;
goto v_resetjp_3138_;
}
v_resetjp_3138_:
{
lean_object* v___x_3142_; 
if (v_isShared_3140_ == 0)
{
v___x_3142_ = v___x_3139_;
goto v_reusejp_3141_;
}
else
{
lean_object* v_reuseFailAlloc_3143_; 
v_reuseFailAlloc_3143_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3143_, 0, v_a_3137_);
v___x_3142_ = v_reuseFailAlloc_3143_;
goto v_reusejp_3141_;
}
v_reusejp_3141_:
{
return v___x_3142_;
}
}
}
}
v___jp_3145_:
{
lean_object* v___x_3147_; 
lean_inc(v___y_3101_);
lean_inc_ref(v___y_3100_);
lean_inc(v___y_3099_);
lean_inc_ref(v___y_3098_);
lean_inc(v___y_3097_);
lean_inc_ref(v___y_3096_);
lean_inc(v___y_3095_);
lean_inc_ref(v___y_3094_);
v___x_3147_ = lean_apply_10(v___y_3146_, v___x_3111_, v___y_3094_, v___y_3095_, v___y_3096_, v___y_3097_, v___y_3098_, v___y_3099_, v___y_3100_, v___y_3101_, lean_box(0));
v___y_3131_ = v___x_3147_;
goto v___jp_3130_;
}
v___jp_3148_:
{
if (v___y_3149_ == 0)
{
lean_object* v___x_3150_; 
lean_del_object(v___x_3127_);
v___x_3150_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_3106_, v___y_3149_, v___y_3095_, v___y_3096_, v___y_3097_, v___y_3098_, v___y_3099_, v___y_3100_, v___y_3101_);
if (lean_obj_tag(v___x_3150_) == 0)
{
lean_object* v_options_3151_; lean_object* v_inheritedTraceOptions_3152_; uint8_t v_hasTrace_3153_; lean_object* v___x_3154_; lean_object* v___f_3155_; 
lean_dec_ref_known(v___x_3150_, 1);
v_options_3151_ = lean_ctor_get(v___y_3100_, 2);
v_inheritedTraceOptions_3152_ = lean_ctor_get(v___y_3100_, 13);
v_hasTrace_3153_ = lean_ctor_get_uint8(v_options_3151_, sizeof(void*)*1);
v___x_3154_ = lean_box(v___y_3149_);
lean_inc_ref(v_a_3089_);
v___f_3155_ = lean_alloc_closure((void*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__4___boxed), 12, 2);
lean_closure_set(v___f_3155_, 0, v_a_3089_);
lean_closure_set(v___f_3155_, 1, v___x_3154_);
if (v_hasTrace_3153_ == 0)
{
lean_dec(v_a_3125_);
v___y_3146_ = v___f_3155_;
goto v___jp_3145_;
}
else
{
lean_object* v___x_3156_; uint8_t v___x_3157_; 
v___x_3156_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__3, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__3_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__3);
v___x_3157_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3152_, v_options_3151_, v___x_3156_);
if (v___x_3157_ == 0)
{
lean_dec(v_a_3125_);
v___y_3146_ = v___f_3155_;
goto v___jp_3145_;
}
else
{
lean_object* v___x_3158_; lean_object* v___x_3159_; lean_object* v___x_3160_; lean_object* v___x_3161_; 
lean_dec_ref(v___f_3155_);
v___x_3158_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__5, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__5_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__5);
v___x_3159_ = l_Lean_Exception_toMessageData(v_a_3125_);
v___x_3160_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3160_, 0, v___x_3158_);
lean_ctor_set(v___x_3160_, 1, v___x_3159_);
v___x_3161_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg(v___x_3119_, v___x_3160_, v___y_3098_, v___y_3099_, v___y_3100_, v___y_3101_);
if (lean_obj_tag(v___x_3161_) == 0)
{
lean_object* v_a_3162_; lean_object* v___x_3163_; 
v_a_3162_ = lean_ctor_get(v___x_3161_, 0);
lean_inc(v_a_3162_);
lean_dec_ref_known(v___x_3161_, 1);
lean_inc_ref(v_a_3089_);
v___x_3163_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__4(v_a_3089_, v___y_3149_, v_a_3162_, v___y_3094_, v___y_3095_, v___y_3096_, v___y_3097_, v___y_3098_, v___y_3099_, v___y_3100_, v___y_3101_);
v___y_3131_ = v___x_3163_;
goto v___jp_3130_;
}
else
{
lean_object* v_a_3164_; lean_object* v___x_3166_; uint8_t v_isShared_3167_; uint8_t v_isSharedCheck_3171_; 
lean_del_object(v___x_3108_);
lean_dec_ref(v_a_3089_);
lean_dec(v_a_3088_);
lean_dec_ref(v_fst_3087_);
lean_dec_ref(v_rel_3086_);
lean_dec_ref(v_snd_3085_);
v_a_3164_ = lean_ctor_get(v___x_3161_, 0);
v_isSharedCheck_3171_ = !lean_is_exclusive(v___x_3161_);
if (v_isSharedCheck_3171_ == 0)
{
v___x_3166_ = v___x_3161_;
v_isShared_3167_ = v_isSharedCheck_3171_;
goto v_resetjp_3165_;
}
else
{
lean_inc(v_a_3164_);
lean_dec(v___x_3161_);
v___x_3166_ = lean_box(0);
v_isShared_3167_ = v_isSharedCheck_3171_;
goto v_resetjp_3165_;
}
v_resetjp_3165_:
{
lean_object* v___x_3169_; 
if (v_isShared_3167_ == 0)
{
v___x_3169_ = v___x_3166_;
goto v_reusejp_3168_;
}
else
{
lean_object* v_reuseFailAlloc_3170_; 
v_reuseFailAlloc_3170_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3170_, 0, v_a_3164_);
v___x_3169_ = v_reuseFailAlloc_3170_;
goto v_reusejp_3168_;
}
v_reusejp_3168_:
{
return v___x_3169_;
}
}
}
}
}
}
else
{
lean_object* v_a_3172_; lean_object* v___x_3174_; uint8_t v_isShared_3175_; uint8_t v_isSharedCheck_3179_; 
lean_dec(v_a_3125_);
lean_del_object(v___x_3108_);
lean_dec_ref(v_a_3089_);
lean_dec(v_a_3088_);
lean_dec_ref(v_fst_3087_);
lean_dec_ref(v_rel_3086_);
lean_dec_ref(v_snd_3085_);
v_a_3172_ = lean_ctor_get(v___x_3150_, 0);
v_isSharedCheck_3179_ = !lean_is_exclusive(v___x_3150_);
if (v_isSharedCheck_3179_ == 0)
{
v___x_3174_ = v___x_3150_;
v_isShared_3175_ = v_isSharedCheck_3179_;
goto v_resetjp_3173_;
}
else
{
lean_inc(v_a_3172_);
lean_dec(v___x_3150_);
v___x_3174_ = lean_box(0);
v_isShared_3175_ = v_isSharedCheck_3179_;
goto v_resetjp_3173_;
}
v_resetjp_3173_:
{
lean_object* v___x_3177_; 
if (v_isShared_3175_ == 0)
{
v___x_3177_ = v___x_3174_;
goto v_reusejp_3176_;
}
else
{
lean_object* v_reuseFailAlloc_3178_; 
v_reuseFailAlloc_3178_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3178_, 0, v_a_3172_);
v___x_3177_ = v_reuseFailAlloc_3178_;
goto v_reusejp_3176_;
}
v_reusejp_3176_:
{
return v___x_3177_;
}
}
}
}
else
{
lean_object* v___x_3181_; 
lean_del_object(v___x_3108_);
lean_dec(v_a_3106_);
lean_dec_ref(v_a_3089_);
lean_dec(v_a_3088_);
lean_dec_ref(v_fst_3087_);
lean_dec_ref(v_rel_3086_);
lean_dec_ref(v_snd_3085_);
if (v_isShared_3128_ == 0)
{
v___x_3181_ = v___x_3127_;
goto v_reusejp_3180_;
}
else
{
lean_object* v_reuseFailAlloc_3182_; 
v_reuseFailAlloc_3182_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3182_, 0, v_a_3125_);
v___x_3181_ = v_reuseFailAlloc_3182_;
goto v_reusejp_3180_;
}
v_reusejp_3180_:
{
return v___x_3181_;
}
}
}
}
}
v___jp_3112_:
{
lean_object* v___x_3114_; lean_object* v___x_3115_; lean_object* v___x_3117_; 
v___x_3114_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3114_, 0, v_a_3113_);
v___x_3115_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3115_, 0, v___x_3114_);
lean_ctor_set(v___x_3115_, 1, v___x_3111_);
if (v_isShared_3109_ == 0)
{
lean_ctor_set(v___x_3108_, 0, v___x_3115_);
v___x_3117_ = v___x_3108_;
goto v_reusejp_3116_;
}
else
{
lean_object* v_reuseFailAlloc_3118_; 
v_reuseFailAlloc_3118_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3118_, 0, v___x_3115_);
v___x_3117_ = v_reuseFailAlloc_3118_;
goto v_reusejp_3116_;
}
v_reusejp_3116_:
{
return v___x_3117_;
}
}
}
}
else
{
lean_object* v_a_3187_; lean_object* v___x_3189_; uint8_t v_isShared_3190_; uint8_t v_isSharedCheck_3194_; 
lean_dec_ref(v_a_3089_);
lean_dec(v_a_3088_);
lean_dec_ref(v_fst_3087_);
lean_dec_ref(v_rel_3086_);
lean_dec_ref(v_snd_3085_);
v_a_3187_ = lean_ctor_get(v___x_3105_, 0);
v_isSharedCheck_3194_ = !lean_is_exclusive(v___x_3105_);
if (v_isSharedCheck_3194_ == 0)
{
v___x_3189_ = v___x_3105_;
v_isShared_3190_ = v_isSharedCheck_3194_;
goto v_resetjp_3188_;
}
else
{
lean_inc(v_a_3187_);
lean_dec(v___x_3105_);
v___x_3189_ = lean_box(0);
v_isShared_3190_ = v_isSharedCheck_3194_;
goto v_resetjp_3188_;
}
v_resetjp_3188_:
{
lean_object* v___x_3192_; 
if (v_isShared_3190_ == 0)
{
v___x_3192_ = v___x_3189_;
goto v_reusejp_3191_;
}
else
{
lean_object* v_reuseFailAlloc_3193_; 
v_reuseFailAlloc_3193_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3193_, 0, v_a_3187_);
v___x_3192_ = v_reuseFailAlloc_3193_;
goto v_reusejp_3191_;
}
v_reusejp_3191_:
{
return v___x_3192_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___boxed(lean_object** _args){
lean_object* v___y_3195_ = _args[0];
lean_object* v_snd_3196_ = _args[1];
lean_object* v_rel_3197_ = _args[2];
lean_object* v_fst_3198_ = _args[3];
lean_object* v_a_3199_ = _args[4];
lean_object* v_a_3200_ = _args[5];
lean_object* v_as_3201_ = _args[6];
lean_object* v_sz_3202_ = _args[7];
lean_object* v_i_3203_ = _args[8];
lean_object* v_b_3204_ = _args[9];
lean_object* v___y_3205_ = _args[10];
lean_object* v___y_3206_ = _args[11];
lean_object* v___y_3207_ = _args[12];
lean_object* v___y_3208_ = _args[13];
lean_object* v___y_3209_ = _args[14];
lean_object* v___y_3210_ = _args[15];
lean_object* v___y_3211_ = _args[16];
lean_object* v___y_3212_ = _args[17];
lean_object* v___y_3213_ = _args[18];
_start:
{
uint8_t v___y_93582__boxed_3214_; size_t v_sz_boxed_3215_; size_t v_i_boxed_3216_; lean_object* v_res_3217_; 
v___y_93582__boxed_3214_ = lean_unbox(v___y_3195_);
v_sz_boxed_3215_ = lean_unbox_usize(v_sz_3202_);
lean_dec(v_sz_3202_);
v_i_boxed_3216_ = lean_unbox_usize(v_i_3203_);
lean_dec(v_i_3203_);
v_res_3217_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10(v___y_93582__boxed_3214_, v_snd_3196_, v_rel_3197_, v_fst_3198_, v_a_3199_, v_a_3200_, v_as_3201_, v_sz_boxed_3215_, v_i_boxed_3216_, v_b_3204_, v___y_3205_, v___y_3206_, v___y_3207_, v___y_3208_, v___y_3209_, v___y_3210_, v___y_3211_, v___y_3212_);
lean_dec(v___y_3212_);
lean_dec_ref(v___y_3211_);
lean_dec(v___y_3210_);
lean_dec_ref(v___y_3209_);
lean_dec(v___y_3208_);
lean_dec_ref(v___y_3207_);
lean_dec(v___y_3206_);
lean_dec_ref(v___y_3205_);
lean_dec_ref(v_as_3201_);
return v_res_3217_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__2(lean_object* v_a_3218_, lean_object* v___f_3219_, uint8_t v___y_3220_, lean_object* v___f_3221_, lean_object* v_a_3222_, lean_object* v_a_3223_, lean_object* v_fst_3224_, lean_object* v_rel_3225_, lean_object* v___x_3226_, lean_object* v_snd_3227_, lean_object* v_____r_3228_, lean_object* v___y_3229_, lean_object* v___y_3230_, lean_object* v___y_3231_, lean_object* v___y_3232_){
_start:
{
lean_object* v___y_3235_; lean_object* v___y_3236_; lean_object* v___x_3239_; 
lean_inc(v_a_3218_);
v___x_3239_ = lp_batteries_Lean_mkConstWithLevelParams___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__2(v_a_3218_, v___y_3229_, v___y_3230_, v___y_3231_, v___y_3232_);
if (lean_obj_tag(v___x_3239_) == 0)
{
lean_object* v_a_3240_; lean_object* v___x_3241_; 
v_a_3240_ = lean_ctor_get(v___x_3239_, 0);
lean_inc(v_a_3240_);
lean_dec_ref_known(v___x_3239_, 1);
lean_inc(v___y_3232_);
lean_inc_ref(v___y_3231_);
lean_inc(v___y_3230_);
lean_inc_ref(v___y_3229_);
v___x_3241_ = lean_infer_type(v_a_3240_, v___y_3229_, v___y_3230_, v___y_3231_, v___y_3232_);
if (lean_obj_tag(v___x_3241_) == 0)
{
lean_object* v_a_3242_; lean_object* v_keyedConfig_3243_; uint8_t v_trackZetaDelta_3244_; lean_object* v_zetaDeltaSet_3245_; lean_object* v_lctx_3246_; lean_object* v_localInstances_3247_; lean_object* v_defEqCtx_x3f_3248_; lean_object* v_synthPendingDepth_3249_; lean_object* v_customCanUnfoldPredicate_x3f_3250_; uint8_t v_univApprox_3251_; uint8_t v_inTypeClassResolution_3252_; uint8_t v_cacheInferType_3253_; uint8_t v___x_3254_; lean_object* v___x_3255_; lean_object* v___x_3256_; lean_object* v___x_3257_; 
v_a_3242_ = lean_ctor_get(v___x_3241_, 0);
lean_inc_n(v_a_3242_, 2);
lean_dec_ref_known(v___x_3241_, 1);
v_keyedConfig_3243_ = lean_ctor_get(v___y_3229_, 0);
v_trackZetaDelta_3244_ = lean_ctor_get_uint8(v___y_3229_, sizeof(void*)*7);
v_zetaDeltaSet_3245_ = lean_ctor_get(v___y_3229_, 1);
v_lctx_3246_ = lean_ctor_get(v___y_3229_, 2);
v_localInstances_3247_ = lean_ctor_get(v___y_3229_, 3);
v_defEqCtx_x3f_3248_ = lean_ctor_get(v___y_3229_, 4);
v_synthPendingDepth_3249_ = lean_ctor_get(v___y_3229_, 5);
v_customCanUnfoldPredicate_x3f_3250_ = lean_ctor_get(v___y_3229_, 6);
v_univApprox_3251_ = lean_ctor_get_uint8(v___y_3229_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_3252_ = lean_ctor_get_uint8(v___y_3229_, sizeof(void*)*7 + 2);
v_cacheInferType_3253_ = lean_ctor_get_uint8(v___y_3229_, sizeof(void*)*7 + 3);
v___x_3254_ = 2;
lean_inc_ref(v_keyedConfig_3243_);
v___x_3255_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_3254_, v_keyedConfig_3243_);
lean_inc(v_customCanUnfoldPredicate_x3f_3250_);
lean_inc(v_synthPendingDepth_3249_);
lean_inc(v_defEqCtx_x3f_3248_);
lean_inc_ref(v_localInstances_3247_);
lean_inc_ref(v_lctx_3246_);
lean_inc(v_zetaDeltaSet_3245_);
v___x_3256_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_3256_, 0, v___x_3255_);
lean_ctor_set(v___x_3256_, 1, v_zetaDeltaSet_3245_);
lean_ctor_set(v___x_3256_, 2, v_lctx_3246_);
lean_ctor_set(v___x_3256_, 3, v_localInstances_3247_);
lean_ctor_set(v___x_3256_, 4, v_defEqCtx_x3f_3248_);
lean_ctor_set(v___x_3256_, 5, v_synthPendingDepth_3249_);
lean_ctor_set(v___x_3256_, 6, v_customCanUnfoldPredicate_x3f_3250_);
lean_ctor_set_uint8(v___x_3256_, sizeof(void*)*7, v_trackZetaDelta_3244_);
lean_ctor_set_uint8(v___x_3256_, sizeof(void*)*7 + 1, v_univApprox_3251_);
lean_ctor_set_uint8(v___x_3256_, sizeof(void*)*7 + 2, v_inTypeClassResolution_3252_);
lean_ctor_set_uint8(v___x_3256_, sizeof(void*)*7 + 3, v_cacheInferType_3253_);
v___x_3257_ = lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__3___redArg(v_a_3242_, v___f_3219_, v___y_3220_, v___y_3220_, v___x_3256_, v___y_3230_, v___y_3231_, v___y_3232_);
lean_dec_ref_known(v___x_3256_, 7);
if (lean_obj_tag(v___x_3257_) == 0)
{
lean_object* v_a_3258_; lean_object* v___y_3260_; lean_object* v___y_3261_; lean_object* v___y_3262_; lean_object* v___y_3263_; lean_object* v___y_3264_; lean_object* v___y_3265_; lean_object* v___y_3266_; lean_object* v___y_3267_; uint8_t v___y_3306_; lean_object* v___y_3307_; lean_object* v___y_3308_; lean_object* v___y_3309_; lean_object* v___y_3310_; lean_object* v___y_3311_; lean_object* v___y_3312_; lean_object* v___y_3313_; lean_object* v___y_3314_; lean_object* v___y_3315_; lean_object* v___y_3359_; lean_object* v___y_3360_; lean_object* v___y_3361_; lean_object* v___y_3362_; lean_object* v___y_3363_; lean_object* v___y_3415_; lean_object* v___y_3416_; lean_object* v___y_3417_; lean_object* v___y_3418_; lean_object* v___y_3419_; lean_object* v___y_3444_; lean_object* v___y_3445_; lean_object* v___y_3446_; lean_object* v___y_3447_; lean_object* v___y_3448_; lean_object* v___y_3473_; lean_object* v___y_3474_; lean_object* v___y_3475_; lean_object* v___y_3476_; lean_object* v___y_3477_; lean_object* v___y_3502_; lean_object* v___y_3503_; lean_object* v___y_3504_; lean_object* v___y_3505_; lean_object* v_a_3506_; lean_object* v___y_3531_; lean_object* v___y_3532_; lean_object* v___y_3533_; lean_object* v___y_3534_; lean_object* v___y_3551_; lean_object* v___y_3552_; lean_object* v___y_3553_; lean_object* v___y_3554_; lean_object* v___x_3578_; 
v_a_3258_ = lean_ctor_get(v___x_3257_, 0);
lean_inc(v_a_3258_);
lean_dec_ref_known(v___x_3257_, 1);
lean_inc_ref(v___f_3221_);
lean_inc(v___y_3232_);
lean_inc_ref(v___y_3231_);
lean_inc(v___y_3230_);
lean_inc_ref(v___y_3229_);
v___x_3578_ = lean_apply_5(v___f_3221_, v___y_3229_, v___y_3230_, v___y_3231_, v___y_3232_, lean_box(0));
if (lean_obj_tag(v___x_3578_) == 0)
{
lean_object* v_a_3579_; uint8_t v___x_3580_; 
v_a_3579_ = lean_ctor_get(v___x_3578_, 0);
lean_inc(v_a_3579_);
lean_dec_ref_known(v___x_3578_, 1);
v___x_3580_ = lean_unbox(v_a_3579_);
lean_dec(v_a_3579_);
if (v___x_3580_ == 0)
{
v___y_3551_ = v___y_3229_;
v___y_3552_ = v___y_3230_;
v___y_3553_ = v___y_3231_;
v___y_3554_ = v___y_3232_;
goto v___jp_3550_;
}
else
{
lean_object* v___x_3581_; lean_object* v___x_3582_; lean_object* v___x_3583_; lean_object* v___x_3584_; lean_object* v___x_3585_; lean_object* v___x_3586_; 
v___x_3581_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__15, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__15_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__15);
lean_inc(v_a_3258_);
v___x_3582_ = l_Nat_reprFast(v_a_3258_);
v___x_3583_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3583_, 0, v___x_3582_);
v___x_3584_ = l_Lean_MessageData_ofFormat(v___x_3583_);
v___x_3585_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3585_, 0, v___x_3581_);
lean_ctor_set(v___x_3585_, 1, v___x_3584_);
lean_inc(v___x_3226_);
v___x_3586_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__5(v___x_3226_, v___x_3585_, v___y_3229_, v___y_3230_, v___y_3231_, v___y_3232_);
if (lean_obj_tag(v___x_3586_) == 0)
{
lean_dec_ref_known(v___x_3586_, 1);
v___y_3551_ = v___y_3229_;
v___y_3552_ = v___y_3230_;
v___y_3553_ = v___y_3231_;
v___y_3554_ = v___y_3232_;
goto v___jp_3550_;
}
else
{
lean_object* v_a_3587_; lean_object* v___x_3589_; uint8_t v_isShared_3590_; uint8_t v_isSharedCheck_3594_; 
lean_dec(v_a_3258_);
lean_dec(v_a_3242_);
lean_dec_ref(v_snd_3227_);
lean_dec(v___x_3226_);
lean_dec_ref(v_rel_3225_);
lean_dec_ref(v_fst_3224_);
lean_dec(v_a_3223_);
lean_dec(v_a_3222_);
lean_dec_ref(v___f_3221_);
lean_dec(v_a_3218_);
v_a_3587_ = lean_ctor_get(v___x_3586_, 0);
v_isSharedCheck_3594_ = !lean_is_exclusive(v___x_3586_);
if (v_isSharedCheck_3594_ == 0)
{
v___x_3589_ = v___x_3586_;
v_isShared_3590_ = v_isSharedCheck_3594_;
goto v_resetjp_3588_;
}
else
{
lean_inc(v_a_3587_);
lean_dec(v___x_3586_);
v___x_3589_ = lean_box(0);
v_isShared_3590_ = v_isSharedCheck_3594_;
goto v_resetjp_3588_;
}
v_resetjp_3588_:
{
lean_object* v___x_3592_; 
if (v_isShared_3590_ == 0)
{
v___x_3592_ = v___x_3589_;
goto v_reusejp_3591_;
}
else
{
lean_object* v_reuseFailAlloc_3593_; 
v_reuseFailAlloc_3593_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3593_, 0, v_a_3587_);
v___x_3592_ = v_reuseFailAlloc_3593_;
goto v_reusejp_3591_;
}
v_reusejp_3591_:
{
return v___x_3592_;
}
}
}
}
}
else
{
lean_object* v_a_3595_; lean_object* v___x_3597_; uint8_t v_isShared_3598_; uint8_t v_isSharedCheck_3602_; 
lean_dec(v_a_3258_);
lean_dec(v_a_3242_);
lean_dec_ref(v_snd_3227_);
lean_dec(v___x_3226_);
lean_dec_ref(v_rel_3225_);
lean_dec_ref(v_fst_3224_);
lean_dec(v_a_3223_);
lean_dec(v_a_3222_);
lean_dec_ref(v___f_3221_);
lean_dec(v_a_3218_);
v_a_3595_ = lean_ctor_get(v___x_3578_, 0);
v_isSharedCheck_3602_ = !lean_is_exclusive(v___x_3578_);
if (v_isSharedCheck_3602_ == 0)
{
v___x_3597_ = v___x_3578_;
v_isShared_3598_ = v_isSharedCheck_3602_;
goto v_resetjp_3596_;
}
else
{
lean_inc(v_a_3595_);
lean_dec(v___x_3578_);
v___x_3597_ = lean_box(0);
v_isShared_3598_ = v_isSharedCheck_3602_;
goto v_resetjp_3596_;
}
v_resetjp_3596_:
{
lean_object* v___x_3600_; 
if (v_isShared_3598_ == 0)
{
v___x_3600_ = v___x_3597_;
goto v_reusejp_3599_;
}
else
{
lean_object* v_reuseFailAlloc_3601_; 
v_reuseFailAlloc_3601_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3601_, 0, v_a_3595_);
v___x_3600_ = v_reuseFailAlloc_3601_;
goto v_reusejp_3599_;
}
v_reusejp_3599_:
{
return v___x_3600_;
}
}
}
v___jp_3259_:
{
lean_object* v___x_3268_; lean_object* v___x_3269_; lean_object* v___x_3270_; lean_object* v___x_3271_; lean_object* v___x_3272_; lean_object* v___x_3273_; lean_object* v___x_3274_; lean_object* v___x_3275_; lean_object* v___x_3276_; lean_object* v___x_3277_; 
v___x_3268_ = lean_nat_sub(v_a_3258_, v___y_3260_);
lean_dec(v_a_3258_);
v___x_3269_ = lean_box(0);
v___x_3270_ = lean_mk_array(v___x_3268_, v___x_3269_);
lean_inc_ref(v___y_3262_);
v___x_3271_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3271_, 0, v___y_3262_);
lean_inc_ref(v___y_3261_);
v___x_3272_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3272_, 0, v___y_3261_);
v___x_3273_ = lean_mk_empty_array_with_capacity(v___y_3260_);
v___x_3274_ = lean_array_push(v___x_3273_, v___x_3271_);
v___x_3275_ = lean_array_push(v___x_3274_, v___x_3272_);
v___x_3276_ = l_Array_append___redArg(v___x_3270_, v___x_3275_);
lean_dec_ref(v___x_3275_);
v___x_3277_ = l_Lean_Meta_mkAppOptM(v_a_3218_, v___x_3276_, v___y_3264_, v___y_3265_, v___y_3266_, v___y_3267_);
if (lean_obj_tag(v___x_3277_) == 0)
{
lean_object* v_a_3278_; lean_object* v___x_3279_; 
v_a_3278_ = lean_ctor_get(v___x_3277_, 0);
lean_inc(v_a_3278_);
lean_dec_ref_known(v___x_3277_, 1);
v___x_3279_ = lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4___redArg(v_a_3222_, v_a_3278_, v___y_3265_);
if (lean_obj_tag(v___x_3279_) == 0)
{
lean_object* v___x_3280_; lean_object* v___x_3281_; lean_object* v___x_3282_; lean_object* v___x_3283_; lean_object* v___x_3284_; 
lean_dec_ref_known(v___x_3279_, 1);
v___x_3280_ = l_Lean_Expr_mvarId_x21(v___y_3262_);
lean_dec_ref(v___y_3262_);
v___x_3281_ = l_Lean_Expr_mvarId_x21(v___y_3261_);
lean_dec_ref(v___y_3261_);
v___x_3282_ = lean_box(0);
v___x_3283_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3283_, 0, v___x_3281_);
lean_ctor_set(v___x_3283_, 1, v___x_3282_);
v___x_3284_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3284_, 0, v___x_3280_);
lean_ctor_set(v___x_3284_, 1, v___x_3283_);
if (lean_obj_tag(v_a_3223_) == 1)
{
lean_object* v_val_3285_; lean_object* v_snd_3286_; 
lean_dec_ref(v___y_3263_);
v_val_3285_ = lean_ctor_get(v_a_3223_, 0);
lean_inc(v_val_3285_);
lean_dec_ref_known(v_a_3223_, 1);
v_snd_3286_ = lean_ctor_get(v_val_3285_, 1);
lean_inc(v_snd_3286_);
lean_dec(v_val_3285_);
v___y_3235_ = v___x_3284_;
v___y_3236_ = v_snd_3286_;
goto v___jp_3234_;
}
else
{
lean_object* v___x_3287_; lean_object* v___x_3288_; 
lean_dec(v_a_3223_);
v___x_3287_ = l_Lean_Expr_mvarId_x21(v___y_3263_);
lean_dec_ref(v___y_3263_);
v___x_3288_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3288_, 0, v___x_3287_);
lean_ctor_set(v___x_3288_, 1, v___x_3282_);
v___y_3235_ = v___x_3284_;
v___y_3236_ = v___x_3288_;
goto v___jp_3234_;
}
}
else
{
lean_object* v_a_3289_; lean_object* v___x_3291_; uint8_t v_isShared_3292_; uint8_t v_isSharedCheck_3296_; 
lean_dec_ref(v___y_3263_);
lean_dec_ref(v___y_3262_);
lean_dec_ref(v___y_3261_);
lean_dec(v_a_3223_);
v_a_3289_ = lean_ctor_get(v___x_3279_, 0);
v_isSharedCheck_3296_ = !lean_is_exclusive(v___x_3279_);
if (v_isSharedCheck_3296_ == 0)
{
v___x_3291_ = v___x_3279_;
v_isShared_3292_ = v_isSharedCheck_3296_;
goto v_resetjp_3290_;
}
else
{
lean_inc(v_a_3289_);
lean_dec(v___x_3279_);
v___x_3291_ = lean_box(0);
v_isShared_3292_ = v_isSharedCheck_3296_;
goto v_resetjp_3290_;
}
v_resetjp_3290_:
{
lean_object* v___x_3294_; 
if (v_isShared_3292_ == 0)
{
v___x_3294_ = v___x_3291_;
goto v_reusejp_3293_;
}
else
{
lean_object* v_reuseFailAlloc_3295_; 
v_reuseFailAlloc_3295_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3295_, 0, v_a_3289_);
v___x_3294_ = v_reuseFailAlloc_3295_;
goto v_reusejp_3293_;
}
v_reusejp_3293_:
{
return v___x_3294_;
}
}
}
}
else
{
lean_object* v_a_3297_; lean_object* v___x_3299_; uint8_t v_isShared_3300_; uint8_t v_isSharedCheck_3304_; 
lean_dec_ref(v___y_3263_);
lean_dec_ref(v___y_3262_);
lean_dec_ref(v___y_3261_);
lean_dec(v_a_3223_);
lean_dec(v_a_3222_);
v_a_3297_ = lean_ctor_get(v___x_3277_, 0);
v_isSharedCheck_3304_ = !lean_is_exclusive(v___x_3277_);
if (v_isSharedCheck_3304_ == 0)
{
v___x_3299_ = v___x_3277_;
v_isShared_3300_ = v_isSharedCheck_3304_;
goto v_resetjp_3298_;
}
else
{
lean_inc(v_a_3297_);
lean_dec(v___x_3277_);
v___x_3299_ = lean_box(0);
v_isShared_3300_ = v_isSharedCheck_3304_;
goto v_resetjp_3298_;
}
v_resetjp_3298_:
{
lean_object* v___x_3302_; 
if (v_isShared_3300_ == 0)
{
v___x_3302_ = v___x_3299_;
goto v_reusejp_3301_;
}
else
{
lean_object* v_reuseFailAlloc_3303_; 
v_reuseFailAlloc_3303_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3303_, 0, v_a_3297_);
v___x_3302_ = v_reuseFailAlloc_3303_;
goto v_reusejp_3301_;
}
v_reusejp_3301_:
{
return v___x_3302_;
}
}
}
}
v___jp_3305_:
{
lean_object* v___x_3316_; lean_object* v___x_3317_; lean_object* v___x_3318_; 
v___x_3316_ = lean_array_push(v___y_3311_, v_fst_3224_);
lean_inc_ref(v___y_3309_);
v___x_3317_ = lean_array_push(v___x_3316_, v___y_3309_);
v___x_3318_ = l_Lean_Meta_mkAppM_x27(v_rel_3225_, v___x_3317_, v___y_3312_, v___y_3313_, v___y_3314_, v___y_3315_);
if (lean_obj_tag(v___x_3318_) == 0)
{
lean_object* v_a_3319_; lean_object* v___x_3320_; lean_object* v___x_3321_; 
v_a_3319_ = lean_ctor_get(v___x_3318_, 0);
lean_inc(v_a_3319_);
lean_dec_ref_known(v___x_3318_, 1);
v___x_3320_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3320_, 0, v_a_3319_);
v___x_3321_ = l_Lean_Meta_mkFreshExprMVar(v___x_3320_, v___y_3306_, v___y_3310_, v___y_3312_, v___y_3313_, v___y_3314_, v___y_3315_);
if (lean_obj_tag(v___x_3321_) == 0)
{
lean_object* v_options_3322_; uint8_t v_hasTrace_3323_; 
v_options_3322_ = lean_ctor_get(v___y_3314_, 2);
v_hasTrace_3323_ = lean_ctor_get_uint8(v_options_3322_, sizeof(void*)*1);
if (v_hasTrace_3323_ == 0)
{
lean_object* v_a_3324_; 
lean_dec(v___x_3226_);
v_a_3324_ = lean_ctor_get(v___x_3321_, 0);
lean_inc(v_a_3324_);
lean_dec_ref_known(v___x_3321_, 1);
v___y_3260_ = v___y_3307_;
v___y_3261_ = v___y_3308_;
v___y_3262_ = v_a_3324_;
v___y_3263_ = v___y_3309_;
v___y_3264_ = v___y_3312_;
v___y_3265_ = v___y_3313_;
v___y_3266_ = v___y_3314_;
v___y_3267_ = v___y_3315_;
goto v___jp_3259_;
}
else
{
lean_object* v_a_3325_; lean_object* v_inheritedTraceOptions_3326_; lean_object* v___x_3327_; lean_object* v___x_3328_; uint8_t v___x_3329_; 
v_a_3325_ = lean_ctor_get(v___x_3321_, 0);
lean_inc(v_a_3325_);
lean_dec_ref_known(v___x_3321_, 1);
v_inheritedTraceOptions_3326_ = lean_ctor_get(v___y_3314_, 13);
v___x_3327_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__0___closed__1));
lean_inc(v___x_3226_);
v___x_3328_ = l_Lean_Name_append(v___x_3327_, v___x_3226_);
v___x_3329_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3326_, v_options_3322_, v___x_3328_);
lean_dec(v___x_3328_);
if (v___x_3329_ == 0)
{
lean_dec(v___x_3226_);
v___y_3260_ = v___y_3307_;
v___y_3261_ = v___y_3308_;
v___y_3262_ = v_a_3325_;
v___y_3263_ = v___y_3309_;
v___y_3264_ = v___y_3312_;
v___y_3265_ = v___y_3313_;
v___y_3266_ = v___y_3314_;
v___y_3267_ = v___y_3315_;
goto v___jp_3259_;
}
else
{
lean_object* v___x_3330_; lean_object* v___x_3331_; lean_object* v___x_3332_; lean_object* v___x_3333_; 
v___x_3330_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__1, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__1_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__1);
lean_inc(v_a_3325_);
v___x_3331_ = l_Lean_MessageData_ofExpr(v_a_3325_);
v___x_3332_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3332_, 0, v___x_3330_);
lean_ctor_set(v___x_3332_, 1, v___x_3331_);
v___x_3333_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__5(v___x_3226_, v___x_3332_, v___y_3312_, v___y_3313_, v___y_3314_, v___y_3315_);
if (lean_obj_tag(v___x_3333_) == 0)
{
lean_dec_ref_known(v___x_3333_, 1);
v___y_3260_ = v___y_3307_;
v___y_3261_ = v___y_3308_;
v___y_3262_ = v_a_3325_;
v___y_3263_ = v___y_3309_;
v___y_3264_ = v___y_3312_;
v___y_3265_ = v___y_3313_;
v___y_3266_ = v___y_3314_;
v___y_3267_ = v___y_3315_;
goto v___jp_3259_;
}
else
{
lean_object* v_a_3334_; lean_object* v___x_3336_; uint8_t v_isShared_3337_; uint8_t v_isSharedCheck_3341_; 
lean_dec(v_a_3325_);
lean_dec_ref(v___y_3309_);
lean_dec_ref(v___y_3308_);
lean_dec(v_a_3258_);
lean_dec(v_a_3223_);
lean_dec(v_a_3222_);
lean_dec(v_a_3218_);
v_a_3334_ = lean_ctor_get(v___x_3333_, 0);
v_isSharedCheck_3341_ = !lean_is_exclusive(v___x_3333_);
if (v_isSharedCheck_3341_ == 0)
{
v___x_3336_ = v___x_3333_;
v_isShared_3337_ = v_isSharedCheck_3341_;
goto v_resetjp_3335_;
}
else
{
lean_inc(v_a_3334_);
lean_dec(v___x_3333_);
v___x_3336_ = lean_box(0);
v_isShared_3337_ = v_isSharedCheck_3341_;
goto v_resetjp_3335_;
}
v_resetjp_3335_:
{
lean_object* v___x_3339_; 
if (v_isShared_3337_ == 0)
{
v___x_3339_ = v___x_3336_;
goto v_reusejp_3338_;
}
else
{
lean_object* v_reuseFailAlloc_3340_; 
v_reuseFailAlloc_3340_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3340_, 0, v_a_3334_);
v___x_3339_ = v_reuseFailAlloc_3340_;
goto v_reusejp_3338_;
}
v_reusejp_3338_:
{
return v___x_3339_;
}
}
}
}
}
}
else
{
lean_object* v_a_3342_; lean_object* v___x_3344_; uint8_t v_isShared_3345_; uint8_t v_isSharedCheck_3349_; 
lean_dec_ref(v___y_3309_);
lean_dec_ref(v___y_3308_);
lean_dec(v_a_3258_);
lean_dec(v___x_3226_);
lean_dec(v_a_3223_);
lean_dec(v_a_3222_);
lean_dec(v_a_3218_);
v_a_3342_ = lean_ctor_get(v___x_3321_, 0);
v_isSharedCheck_3349_ = !lean_is_exclusive(v___x_3321_);
if (v_isSharedCheck_3349_ == 0)
{
v___x_3344_ = v___x_3321_;
v_isShared_3345_ = v_isSharedCheck_3349_;
goto v_resetjp_3343_;
}
else
{
lean_inc(v_a_3342_);
lean_dec(v___x_3321_);
v___x_3344_ = lean_box(0);
v_isShared_3345_ = v_isSharedCheck_3349_;
goto v_resetjp_3343_;
}
v_resetjp_3343_:
{
lean_object* v___x_3347_; 
if (v_isShared_3345_ == 0)
{
v___x_3347_ = v___x_3344_;
goto v_reusejp_3346_;
}
else
{
lean_object* v_reuseFailAlloc_3348_; 
v_reuseFailAlloc_3348_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3348_, 0, v_a_3342_);
v___x_3347_ = v_reuseFailAlloc_3348_;
goto v_reusejp_3346_;
}
v_reusejp_3346_:
{
return v___x_3347_;
}
}
}
}
else
{
lean_object* v_a_3350_; lean_object* v___x_3352_; uint8_t v_isShared_3353_; uint8_t v_isSharedCheck_3357_; 
lean_dec(v___y_3310_);
lean_dec_ref(v___y_3309_);
lean_dec_ref(v___y_3308_);
lean_dec(v_a_3258_);
lean_dec(v___x_3226_);
lean_dec(v_a_3223_);
lean_dec(v_a_3222_);
lean_dec(v_a_3218_);
v_a_3350_ = lean_ctor_get(v___x_3318_, 0);
v_isSharedCheck_3357_ = !lean_is_exclusive(v___x_3318_);
if (v_isSharedCheck_3357_ == 0)
{
v___x_3352_ = v___x_3318_;
v_isShared_3353_ = v_isSharedCheck_3357_;
goto v_resetjp_3351_;
}
else
{
lean_inc(v_a_3350_);
lean_dec(v___x_3318_);
v___x_3352_ = lean_box(0);
v_isShared_3353_ = v_isSharedCheck_3357_;
goto v_resetjp_3351_;
}
v_resetjp_3351_:
{
lean_object* v___x_3355_; 
if (v_isShared_3353_ == 0)
{
v___x_3355_ = v___x_3352_;
goto v_reusejp_3354_;
}
else
{
lean_object* v_reuseFailAlloc_3356_; 
v_reuseFailAlloc_3356_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3356_, 0, v_a_3350_);
v___x_3355_ = v_reuseFailAlloc_3356_;
goto v_reusejp_3354_;
}
v_reusejp_3354_:
{
return v___x_3355_;
}
}
}
}
v___jp_3358_:
{
lean_object* v___x_3364_; lean_object* v___x_3365_; lean_object* v___x_3366_; lean_object* v___x_3367_; lean_object* v___x_3368_; 
v___x_3364_ = lean_unsigned_to_nat(2u);
v___x_3365_ = lean_mk_empty_array_with_capacity(v___x_3364_);
lean_inc_ref(v___y_3359_);
lean_inc_ref(v___x_3365_);
v___x_3366_ = lean_array_push(v___x_3365_, v___y_3359_);
v___x_3367_ = lean_array_push(v___x_3366_, v_snd_3227_);
lean_inc_ref(v_rel_3225_);
v___x_3368_ = l_Lean_Meta_mkAppM_x27(v_rel_3225_, v___x_3367_, v___y_3360_, v___y_3361_, v___y_3362_, v___y_3363_);
if (lean_obj_tag(v___x_3368_) == 0)
{
lean_object* v_a_3369_; lean_object* v___x_3370_; uint8_t v___x_3371_; lean_object* v___x_3372_; lean_object* v___x_3373_; 
v_a_3369_ = lean_ctor_get(v___x_3368_, 0);
lean_inc(v_a_3369_);
lean_dec_ref_known(v___x_3368_, 1);
v___x_3370_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3370_, 0, v_a_3369_);
v___x_3371_ = 1;
v___x_3372_ = lean_box(0);
v___x_3373_ = l_Lean_Meta_mkFreshExprMVar(v___x_3370_, v___x_3371_, v___x_3372_, v___y_3360_, v___y_3361_, v___y_3362_, v___y_3363_);
if (lean_obj_tag(v___x_3373_) == 0)
{
lean_object* v_a_3374_; lean_object* v___x_3375_; 
v_a_3374_ = lean_ctor_get(v___x_3373_, 0);
lean_inc(v_a_3374_);
lean_dec_ref_known(v___x_3373_, 1);
lean_inc(v___y_3363_);
lean_inc_ref(v___y_3362_);
lean_inc(v___y_3361_);
lean_inc_ref(v___y_3360_);
v___x_3375_ = lean_apply_5(v___f_3221_, v___y_3360_, v___y_3361_, v___y_3362_, v___y_3363_, lean_box(0));
if (lean_obj_tag(v___x_3375_) == 0)
{
lean_object* v_a_3376_; uint8_t v___x_3377_; 
v_a_3376_ = lean_ctor_get(v___x_3375_, 0);
lean_inc(v_a_3376_);
lean_dec_ref_known(v___x_3375_, 1);
v___x_3377_ = lean_unbox(v_a_3376_);
lean_dec(v_a_3376_);
if (v___x_3377_ == 0)
{
v___y_3306_ = v___x_3371_;
v___y_3307_ = v___x_3364_;
v___y_3308_ = v_a_3374_;
v___y_3309_ = v___y_3359_;
v___y_3310_ = v___x_3372_;
v___y_3311_ = v___x_3365_;
v___y_3312_ = v___y_3360_;
v___y_3313_ = v___y_3361_;
v___y_3314_ = v___y_3362_;
v___y_3315_ = v___y_3363_;
goto v___jp_3305_;
}
else
{
lean_object* v___x_3378_; lean_object* v___x_3379_; lean_object* v___x_3380_; lean_object* v___x_3381_; 
v___x_3378_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__3, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__3_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__3);
lean_inc(v_a_3374_);
v___x_3379_ = l_Lean_MessageData_ofExpr(v_a_3374_);
v___x_3380_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3380_, 0, v___x_3378_);
lean_ctor_set(v___x_3380_, 1, v___x_3379_);
lean_inc(v___x_3226_);
v___x_3381_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__5(v___x_3226_, v___x_3380_, v___y_3360_, v___y_3361_, v___y_3362_, v___y_3363_);
if (lean_obj_tag(v___x_3381_) == 0)
{
lean_dec_ref_known(v___x_3381_, 1);
v___y_3306_ = v___x_3371_;
v___y_3307_ = v___x_3364_;
v___y_3308_ = v_a_3374_;
v___y_3309_ = v___y_3359_;
v___y_3310_ = v___x_3372_;
v___y_3311_ = v___x_3365_;
v___y_3312_ = v___y_3360_;
v___y_3313_ = v___y_3361_;
v___y_3314_ = v___y_3362_;
v___y_3315_ = v___y_3363_;
goto v___jp_3305_;
}
else
{
lean_object* v_a_3382_; lean_object* v___x_3384_; uint8_t v_isShared_3385_; uint8_t v_isSharedCheck_3389_; 
lean_dec(v_a_3374_);
lean_dec_ref(v___x_3365_);
lean_dec_ref(v___y_3359_);
lean_dec(v_a_3258_);
lean_dec(v___x_3226_);
lean_dec_ref(v_rel_3225_);
lean_dec_ref(v_fst_3224_);
lean_dec(v_a_3223_);
lean_dec(v_a_3222_);
lean_dec(v_a_3218_);
v_a_3382_ = lean_ctor_get(v___x_3381_, 0);
v_isSharedCheck_3389_ = !lean_is_exclusive(v___x_3381_);
if (v_isSharedCheck_3389_ == 0)
{
v___x_3384_ = v___x_3381_;
v_isShared_3385_ = v_isSharedCheck_3389_;
goto v_resetjp_3383_;
}
else
{
lean_inc(v_a_3382_);
lean_dec(v___x_3381_);
v___x_3384_ = lean_box(0);
v_isShared_3385_ = v_isSharedCheck_3389_;
goto v_resetjp_3383_;
}
v_resetjp_3383_:
{
lean_object* v___x_3387_; 
if (v_isShared_3385_ == 0)
{
v___x_3387_ = v___x_3384_;
goto v_reusejp_3386_;
}
else
{
lean_object* v_reuseFailAlloc_3388_; 
v_reuseFailAlloc_3388_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3388_, 0, v_a_3382_);
v___x_3387_ = v_reuseFailAlloc_3388_;
goto v_reusejp_3386_;
}
v_reusejp_3386_:
{
return v___x_3387_;
}
}
}
}
}
else
{
lean_object* v_a_3390_; lean_object* v___x_3392_; uint8_t v_isShared_3393_; uint8_t v_isSharedCheck_3397_; 
lean_dec(v_a_3374_);
lean_dec_ref(v___x_3365_);
lean_dec_ref(v___y_3359_);
lean_dec(v_a_3258_);
lean_dec(v___x_3226_);
lean_dec_ref(v_rel_3225_);
lean_dec_ref(v_fst_3224_);
lean_dec(v_a_3223_);
lean_dec(v_a_3222_);
lean_dec(v_a_3218_);
v_a_3390_ = lean_ctor_get(v___x_3375_, 0);
v_isSharedCheck_3397_ = !lean_is_exclusive(v___x_3375_);
if (v_isSharedCheck_3397_ == 0)
{
v___x_3392_ = v___x_3375_;
v_isShared_3393_ = v_isSharedCheck_3397_;
goto v_resetjp_3391_;
}
else
{
lean_inc(v_a_3390_);
lean_dec(v___x_3375_);
v___x_3392_ = lean_box(0);
v_isShared_3393_ = v_isSharedCheck_3397_;
goto v_resetjp_3391_;
}
v_resetjp_3391_:
{
lean_object* v___x_3395_; 
if (v_isShared_3393_ == 0)
{
v___x_3395_ = v___x_3392_;
goto v_reusejp_3394_;
}
else
{
lean_object* v_reuseFailAlloc_3396_; 
v_reuseFailAlloc_3396_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3396_, 0, v_a_3390_);
v___x_3395_ = v_reuseFailAlloc_3396_;
goto v_reusejp_3394_;
}
v_reusejp_3394_:
{
return v___x_3395_;
}
}
}
}
else
{
lean_object* v_a_3398_; lean_object* v___x_3400_; uint8_t v_isShared_3401_; uint8_t v_isSharedCheck_3405_; 
lean_dec_ref(v___x_3365_);
lean_dec_ref(v___y_3359_);
lean_dec(v_a_3258_);
lean_dec(v___x_3226_);
lean_dec_ref(v_rel_3225_);
lean_dec_ref(v_fst_3224_);
lean_dec(v_a_3223_);
lean_dec(v_a_3222_);
lean_dec_ref(v___f_3221_);
lean_dec(v_a_3218_);
v_a_3398_ = lean_ctor_get(v___x_3373_, 0);
v_isSharedCheck_3405_ = !lean_is_exclusive(v___x_3373_);
if (v_isSharedCheck_3405_ == 0)
{
v___x_3400_ = v___x_3373_;
v_isShared_3401_ = v_isSharedCheck_3405_;
goto v_resetjp_3399_;
}
else
{
lean_inc(v_a_3398_);
lean_dec(v___x_3373_);
v___x_3400_ = lean_box(0);
v_isShared_3401_ = v_isSharedCheck_3405_;
goto v_resetjp_3399_;
}
v_resetjp_3399_:
{
lean_object* v___x_3403_; 
if (v_isShared_3401_ == 0)
{
v___x_3403_ = v___x_3400_;
goto v_reusejp_3402_;
}
else
{
lean_object* v_reuseFailAlloc_3404_; 
v_reuseFailAlloc_3404_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3404_, 0, v_a_3398_);
v___x_3403_ = v_reuseFailAlloc_3404_;
goto v_reusejp_3402_;
}
v_reusejp_3402_:
{
return v___x_3403_;
}
}
}
}
else
{
lean_object* v_a_3406_; lean_object* v___x_3408_; uint8_t v_isShared_3409_; uint8_t v_isSharedCheck_3413_; 
lean_dec_ref(v___x_3365_);
lean_dec_ref(v___y_3359_);
lean_dec(v_a_3258_);
lean_dec(v___x_3226_);
lean_dec_ref(v_rel_3225_);
lean_dec_ref(v_fst_3224_);
lean_dec(v_a_3223_);
lean_dec(v_a_3222_);
lean_dec_ref(v___f_3221_);
lean_dec(v_a_3218_);
v_a_3406_ = lean_ctor_get(v___x_3368_, 0);
v_isSharedCheck_3413_ = !lean_is_exclusive(v___x_3368_);
if (v_isSharedCheck_3413_ == 0)
{
v___x_3408_ = v___x_3368_;
v_isShared_3409_ = v_isSharedCheck_3413_;
goto v_resetjp_3407_;
}
else
{
lean_inc(v_a_3406_);
lean_dec(v___x_3368_);
v___x_3408_ = lean_box(0);
v_isShared_3409_ = v_isSharedCheck_3413_;
goto v_resetjp_3407_;
}
v_resetjp_3407_:
{
lean_object* v___x_3411_; 
if (v_isShared_3409_ == 0)
{
v___x_3411_ = v___x_3408_;
goto v_reusejp_3410_;
}
else
{
lean_object* v_reuseFailAlloc_3412_; 
v_reuseFailAlloc_3412_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3412_, 0, v_a_3406_);
v___x_3411_ = v_reuseFailAlloc_3412_;
goto v_reusejp_3410_;
}
v_reusejp_3410_:
{
return v___x_3411_;
}
}
}
}
v___jp_3414_:
{
lean_object* v___x_3420_; 
lean_inc_ref(v___f_3221_);
lean_inc(v___y_3419_);
lean_inc_ref(v___y_3418_);
lean_inc(v___y_3417_);
lean_inc_ref(v___y_3416_);
v___x_3420_ = lean_apply_5(v___f_3221_, v___y_3416_, v___y_3417_, v___y_3418_, v___y_3419_, lean_box(0));
if (lean_obj_tag(v___x_3420_) == 0)
{
lean_object* v_a_3421_; uint8_t v___x_3422_; 
v_a_3421_ = lean_ctor_get(v___x_3420_, 0);
lean_inc(v_a_3421_);
lean_dec_ref_known(v___x_3420_, 1);
v___x_3422_ = lean_unbox(v_a_3421_);
lean_dec(v_a_3421_);
if (v___x_3422_ == 0)
{
v___y_3359_ = v___y_3415_;
v___y_3360_ = v___y_3416_;
v___y_3361_ = v___y_3417_;
v___y_3362_ = v___y_3418_;
v___y_3363_ = v___y_3419_;
goto v___jp_3358_;
}
else
{
lean_object* v___x_3423_; lean_object* v___x_3424_; lean_object* v___x_3425_; lean_object* v___x_3426_; 
v___x_3423_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__5, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__5_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__5);
lean_inc_ref(v_snd_3227_);
v___x_3424_ = l_Lean_indentExpr(v_snd_3227_);
v___x_3425_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3425_, 0, v___x_3423_);
lean_ctor_set(v___x_3425_, 1, v___x_3424_);
lean_inc(v___x_3226_);
v___x_3426_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__5(v___x_3226_, v___x_3425_, v___y_3416_, v___y_3417_, v___y_3418_, v___y_3419_);
if (lean_obj_tag(v___x_3426_) == 0)
{
lean_dec_ref_known(v___x_3426_, 1);
v___y_3359_ = v___y_3415_;
v___y_3360_ = v___y_3416_;
v___y_3361_ = v___y_3417_;
v___y_3362_ = v___y_3418_;
v___y_3363_ = v___y_3419_;
goto v___jp_3358_;
}
else
{
lean_object* v_a_3427_; lean_object* v___x_3429_; uint8_t v_isShared_3430_; uint8_t v_isSharedCheck_3434_; 
lean_dec_ref(v___y_3415_);
lean_dec(v_a_3258_);
lean_dec_ref(v_snd_3227_);
lean_dec(v___x_3226_);
lean_dec_ref(v_rel_3225_);
lean_dec_ref(v_fst_3224_);
lean_dec(v_a_3223_);
lean_dec(v_a_3222_);
lean_dec_ref(v___f_3221_);
lean_dec(v_a_3218_);
v_a_3427_ = lean_ctor_get(v___x_3426_, 0);
v_isSharedCheck_3434_ = !lean_is_exclusive(v___x_3426_);
if (v_isSharedCheck_3434_ == 0)
{
v___x_3429_ = v___x_3426_;
v_isShared_3430_ = v_isSharedCheck_3434_;
goto v_resetjp_3428_;
}
else
{
lean_inc(v_a_3427_);
lean_dec(v___x_3426_);
v___x_3429_ = lean_box(0);
v_isShared_3430_ = v_isSharedCheck_3434_;
goto v_resetjp_3428_;
}
v_resetjp_3428_:
{
lean_object* v___x_3432_; 
if (v_isShared_3430_ == 0)
{
v___x_3432_ = v___x_3429_;
goto v_reusejp_3431_;
}
else
{
lean_object* v_reuseFailAlloc_3433_; 
v_reuseFailAlloc_3433_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3433_, 0, v_a_3427_);
v___x_3432_ = v_reuseFailAlloc_3433_;
goto v_reusejp_3431_;
}
v_reusejp_3431_:
{
return v___x_3432_;
}
}
}
}
}
else
{
lean_object* v_a_3435_; lean_object* v___x_3437_; uint8_t v_isShared_3438_; uint8_t v_isSharedCheck_3442_; 
lean_dec_ref(v___y_3415_);
lean_dec(v_a_3258_);
lean_dec_ref(v_snd_3227_);
lean_dec(v___x_3226_);
lean_dec_ref(v_rel_3225_);
lean_dec_ref(v_fst_3224_);
lean_dec(v_a_3223_);
lean_dec(v_a_3222_);
lean_dec_ref(v___f_3221_);
lean_dec(v_a_3218_);
v_a_3435_ = lean_ctor_get(v___x_3420_, 0);
v_isSharedCheck_3442_ = !lean_is_exclusive(v___x_3420_);
if (v_isSharedCheck_3442_ == 0)
{
v___x_3437_ = v___x_3420_;
v_isShared_3438_ = v_isSharedCheck_3442_;
goto v_resetjp_3436_;
}
else
{
lean_inc(v_a_3435_);
lean_dec(v___x_3420_);
v___x_3437_ = lean_box(0);
v_isShared_3438_ = v_isSharedCheck_3442_;
goto v_resetjp_3436_;
}
v_resetjp_3436_:
{
lean_object* v___x_3440_; 
if (v_isShared_3438_ == 0)
{
v___x_3440_ = v___x_3437_;
goto v_reusejp_3439_;
}
else
{
lean_object* v_reuseFailAlloc_3441_; 
v_reuseFailAlloc_3441_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3441_, 0, v_a_3435_);
v___x_3440_ = v_reuseFailAlloc_3441_;
goto v_reusejp_3439_;
}
v_reusejp_3439_:
{
return v___x_3440_;
}
}
}
}
v___jp_3443_:
{
lean_object* v___x_3449_; 
lean_inc_ref(v___f_3221_);
lean_inc(v___y_3448_);
lean_inc_ref(v___y_3447_);
lean_inc(v___y_3446_);
lean_inc_ref(v___y_3445_);
v___x_3449_ = lean_apply_5(v___f_3221_, v___y_3445_, v___y_3446_, v___y_3447_, v___y_3448_, lean_box(0));
if (lean_obj_tag(v___x_3449_) == 0)
{
lean_object* v_a_3450_; uint8_t v___x_3451_; 
v_a_3450_ = lean_ctor_get(v___x_3449_, 0);
lean_inc(v_a_3450_);
lean_dec_ref_known(v___x_3449_, 1);
v___x_3451_ = lean_unbox(v_a_3450_);
lean_dec(v_a_3450_);
if (v___x_3451_ == 0)
{
v___y_3415_ = v___y_3444_;
v___y_3416_ = v___y_3445_;
v___y_3417_ = v___y_3446_;
v___y_3418_ = v___y_3447_;
v___y_3419_ = v___y_3448_;
goto v___jp_3414_;
}
else
{
lean_object* v___x_3452_; lean_object* v___x_3453_; lean_object* v___x_3454_; lean_object* v___x_3455_; 
v___x_3452_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__7, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__7_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__7);
lean_inc_ref(v_fst_3224_);
v___x_3453_ = l_Lean_indentExpr(v_fst_3224_);
v___x_3454_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3454_, 0, v___x_3452_);
lean_ctor_set(v___x_3454_, 1, v___x_3453_);
lean_inc(v___x_3226_);
v___x_3455_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__5(v___x_3226_, v___x_3454_, v___y_3445_, v___y_3446_, v___y_3447_, v___y_3448_);
if (lean_obj_tag(v___x_3455_) == 0)
{
lean_dec_ref_known(v___x_3455_, 1);
v___y_3415_ = v___y_3444_;
v___y_3416_ = v___y_3445_;
v___y_3417_ = v___y_3446_;
v___y_3418_ = v___y_3447_;
v___y_3419_ = v___y_3448_;
goto v___jp_3414_;
}
else
{
lean_object* v_a_3456_; lean_object* v___x_3458_; uint8_t v_isShared_3459_; uint8_t v_isSharedCheck_3463_; 
lean_dec_ref(v___y_3444_);
lean_dec(v_a_3258_);
lean_dec_ref(v_snd_3227_);
lean_dec(v___x_3226_);
lean_dec_ref(v_rel_3225_);
lean_dec_ref(v_fst_3224_);
lean_dec(v_a_3223_);
lean_dec(v_a_3222_);
lean_dec_ref(v___f_3221_);
lean_dec(v_a_3218_);
v_a_3456_ = lean_ctor_get(v___x_3455_, 0);
v_isSharedCheck_3463_ = !lean_is_exclusive(v___x_3455_);
if (v_isSharedCheck_3463_ == 0)
{
v___x_3458_ = v___x_3455_;
v_isShared_3459_ = v_isSharedCheck_3463_;
goto v_resetjp_3457_;
}
else
{
lean_inc(v_a_3456_);
lean_dec(v___x_3455_);
v___x_3458_ = lean_box(0);
v_isShared_3459_ = v_isSharedCheck_3463_;
goto v_resetjp_3457_;
}
v_resetjp_3457_:
{
lean_object* v___x_3461_; 
if (v_isShared_3459_ == 0)
{
v___x_3461_ = v___x_3458_;
goto v_reusejp_3460_;
}
else
{
lean_object* v_reuseFailAlloc_3462_; 
v_reuseFailAlloc_3462_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3462_, 0, v_a_3456_);
v___x_3461_ = v_reuseFailAlloc_3462_;
goto v_reusejp_3460_;
}
v_reusejp_3460_:
{
return v___x_3461_;
}
}
}
}
}
else
{
lean_object* v_a_3464_; lean_object* v___x_3466_; uint8_t v_isShared_3467_; uint8_t v_isSharedCheck_3471_; 
lean_dec_ref(v___y_3444_);
lean_dec(v_a_3258_);
lean_dec_ref(v_snd_3227_);
lean_dec(v___x_3226_);
lean_dec_ref(v_rel_3225_);
lean_dec_ref(v_fst_3224_);
lean_dec(v_a_3223_);
lean_dec(v_a_3222_);
lean_dec_ref(v___f_3221_);
lean_dec(v_a_3218_);
v_a_3464_ = lean_ctor_get(v___x_3449_, 0);
v_isSharedCheck_3471_ = !lean_is_exclusive(v___x_3449_);
if (v_isSharedCheck_3471_ == 0)
{
v___x_3466_ = v___x_3449_;
v_isShared_3467_ = v_isSharedCheck_3471_;
goto v_resetjp_3465_;
}
else
{
lean_inc(v_a_3464_);
lean_dec(v___x_3449_);
v___x_3466_ = lean_box(0);
v_isShared_3467_ = v_isSharedCheck_3471_;
goto v_resetjp_3465_;
}
v_resetjp_3465_:
{
lean_object* v___x_3469_; 
if (v_isShared_3467_ == 0)
{
v___x_3469_ = v___x_3466_;
goto v_reusejp_3468_;
}
else
{
lean_object* v_reuseFailAlloc_3470_; 
v_reuseFailAlloc_3470_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3470_, 0, v_a_3464_);
v___x_3469_ = v_reuseFailAlloc_3470_;
goto v_reusejp_3468_;
}
v_reusejp_3468_:
{
return v___x_3469_;
}
}
}
}
v___jp_3472_:
{
lean_object* v___x_3478_; 
lean_inc_ref(v___f_3221_);
lean_inc(v___y_3477_);
lean_inc_ref(v___y_3476_);
lean_inc(v___y_3475_);
lean_inc_ref(v___y_3474_);
v___x_3478_ = lean_apply_5(v___f_3221_, v___y_3474_, v___y_3475_, v___y_3476_, v___y_3477_, lean_box(0));
if (lean_obj_tag(v___x_3478_) == 0)
{
lean_object* v_a_3479_; uint8_t v___x_3480_; 
v_a_3479_ = lean_ctor_get(v___x_3478_, 0);
lean_inc(v_a_3479_);
lean_dec_ref_known(v___x_3478_, 1);
v___x_3480_ = lean_unbox(v_a_3479_);
lean_dec(v_a_3479_);
if (v___x_3480_ == 0)
{
v___y_3444_ = v___y_3473_;
v___y_3445_ = v___y_3474_;
v___y_3446_ = v___y_3475_;
v___y_3447_ = v___y_3476_;
v___y_3448_ = v___y_3477_;
goto v___jp_3443_;
}
else
{
lean_object* v___x_3481_; lean_object* v___x_3482_; lean_object* v___x_3483_; lean_object* v___x_3484_; 
v___x_3481_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__9, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__9_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__9);
lean_inc_ref(v_rel_3225_);
v___x_3482_ = l_Lean_indentExpr(v_rel_3225_);
v___x_3483_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3483_, 0, v___x_3481_);
lean_ctor_set(v___x_3483_, 1, v___x_3482_);
lean_inc(v___x_3226_);
v___x_3484_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__5(v___x_3226_, v___x_3483_, v___y_3474_, v___y_3475_, v___y_3476_, v___y_3477_);
if (lean_obj_tag(v___x_3484_) == 0)
{
lean_dec_ref_known(v___x_3484_, 1);
v___y_3444_ = v___y_3473_;
v___y_3445_ = v___y_3474_;
v___y_3446_ = v___y_3475_;
v___y_3447_ = v___y_3476_;
v___y_3448_ = v___y_3477_;
goto v___jp_3443_;
}
else
{
lean_object* v_a_3485_; lean_object* v___x_3487_; uint8_t v_isShared_3488_; uint8_t v_isSharedCheck_3492_; 
lean_dec_ref(v___y_3473_);
lean_dec(v_a_3258_);
lean_dec_ref(v_snd_3227_);
lean_dec(v___x_3226_);
lean_dec_ref(v_rel_3225_);
lean_dec_ref(v_fst_3224_);
lean_dec(v_a_3223_);
lean_dec(v_a_3222_);
lean_dec_ref(v___f_3221_);
lean_dec(v_a_3218_);
v_a_3485_ = lean_ctor_get(v___x_3484_, 0);
v_isSharedCheck_3492_ = !lean_is_exclusive(v___x_3484_);
if (v_isSharedCheck_3492_ == 0)
{
v___x_3487_ = v___x_3484_;
v_isShared_3488_ = v_isSharedCheck_3492_;
goto v_resetjp_3486_;
}
else
{
lean_inc(v_a_3485_);
lean_dec(v___x_3484_);
v___x_3487_ = lean_box(0);
v_isShared_3488_ = v_isSharedCheck_3492_;
goto v_resetjp_3486_;
}
v_resetjp_3486_:
{
lean_object* v___x_3490_; 
if (v_isShared_3488_ == 0)
{
v___x_3490_ = v___x_3487_;
goto v_reusejp_3489_;
}
else
{
lean_object* v_reuseFailAlloc_3491_; 
v_reuseFailAlloc_3491_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3491_, 0, v_a_3485_);
v___x_3490_ = v_reuseFailAlloc_3491_;
goto v_reusejp_3489_;
}
v_reusejp_3489_:
{
return v___x_3490_;
}
}
}
}
}
else
{
lean_object* v_a_3493_; lean_object* v___x_3495_; uint8_t v_isShared_3496_; uint8_t v_isSharedCheck_3500_; 
lean_dec_ref(v___y_3473_);
lean_dec(v_a_3258_);
lean_dec_ref(v_snd_3227_);
lean_dec(v___x_3226_);
lean_dec_ref(v_rel_3225_);
lean_dec_ref(v_fst_3224_);
lean_dec(v_a_3223_);
lean_dec(v_a_3222_);
lean_dec_ref(v___f_3221_);
lean_dec(v_a_3218_);
v_a_3493_ = lean_ctor_get(v___x_3478_, 0);
v_isSharedCheck_3500_ = !lean_is_exclusive(v___x_3478_);
if (v_isSharedCheck_3500_ == 0)
{
v___x_3495_ = v___x_3478_;
v_isShared_3496_ = v_isSharedCheck_3500_;
goto v_resetjp_3494_;
}
else
{
lean_inc(v_a_3493_);
lean_dec(v___x_3478_);
v___x_3495_ = lean_box(0);
v_isShared_3496_ = v_isSharedCheck_3500_;
goto v_resetjp_3494_;
}
v_resetjp_3494_:
{
lean_object* v___x_3498_; 
if (v_isShared_3496_ == 0)
{
v___x_3498_ = v___x_3495_;
goto v_reusejp_3497_;
}
else
{
lean_object* v_reuseFailAlloc_3499_; 
v_reuseFailAlloc_3499_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3499_, 0, v_a_3493_);
v___x_3498_ = v_reuseFailAlloc_3499_;
goto v_reusejp_3497_;
}
v_reusejp_3497_:
{
return v___x_3498_;
}
}
}
}
v___jp_3501_:
{
lean_object* v___x_3507_; 
lean_inc_ref(v___f_3221_);
lean_inc(v___y_3505_);
lean_inc_ref(v___y_3504_);
lean_inc(v___y_3503_);
lean_inc_ref(v___y_3502_);
v___x_3507_ = lean_apply_5(v___f_3221_, v___y_3502_, v___y_3503_, v___y_3504_, v___y_3505_, lean_box(0));
if (lean_obj_tag(v___x_3507_) == 0)
{
lean_object* v_a_3508_; uint8_t v___x_3509_; 
v_a_3508_ = lean_ctor_get(v___x_3507_, 0);
lean_inc(v_a_3508_);
lean_dec_ref_known(v___x_3507_, 1);
v___x_3509_ = lean_unbox(v_a_3508_);
lean_dec(v_a_3508_);
if (v___x_3509_ == 0)
{
v___y_3473_ = v_a_3506_;
v___y_3474_ = v___y_3502_;
v___y_3475_ = v___y_3503_;
v___y_3476_ = v___y_3504_;
v___y_3477_ = v___y_3505_;
goto v___jp_3472_;
}
else
{
lean_object* v___x_3510_; lean_object* v___x_3511_; lean_object* v___x_3512_; lean_object* v___x_3513_; 
v___x_3510_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__11, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__11_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__11);
lean_inc_ref(v_a_3506_);
v___x_3511_ = l_Lean_MessageData_ofExpr(v_a_3506_);
v___x_3512_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3512_, 0, v___x_3510_);
lean_ctor_set(v___x_3512_, 1, v___x_3511_);
lean_inc(v___x_3226_);
v___x_3513_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__5(v___x_3226_, v___x_3512_, v___y_3502_, v___y_3503_, v___y_3504_, v___y_3505_);
if (lean_obj_tag(v___x_3513_) == 0)
{
lean_dec_ref_known(v___x_3513_, 1);
v___y_3473_ = v_a_3506_;
v___y_3474_ = v___y_3502_;
v___y_3475_ = v___y_3503_;
v___y_3476_ = v___y_3504_;
v___y_3477_ = v___y_3505_;
goto v___jp_3472_;
}
else
{
lean_object* v_a_3514_; lean_object* v___x_3516_; uint8_t v_isShared_3517_; uint8_t v_isSharedCheck_3521_; 
lean_dec_ref(v_a_3506_);
lean_dec(v_a_3258_);
lean_dec_ref(v_snd_3227_);
lean_dec(v___x_3226_);
lean_dec_ref(v_rel_3225_);
lean_dec_ref(v_fst_3224_);
lean_dec(v_a_3223_);
lean_dec(v_a_3222_);
lean_dec_ref(v___f_3221_);
lean_dec(v_a_3218_);
v_a_3514_ = lean_ctor_get(v___x_3513_, 0);
v_isSharedCheck_3521_ = !lean_is_exclusive(v___x_3513_);
if (v_isSharedCheck_3521_ == 0)
{
v___x_3516_ = v___x_3513_;
v_isShared_3517_ = v_isSharedCheck_3521_;
goto v_resetjp_3515_;
}
else
{
lean_inc(v_a_3514_);
lean_dec(v___x_3513_);
v___x_3516_ = lean_box(0);
v_isShared_3517_ = v_isSharedCheck_3521_;
goto v_resetjp_3515_;
}
v_resetjp_3515_:
{
lean_object* v___x_3519_; 
if (v_isShared_3517_ == 0)
{
v___x_3519_ = v___x_3516_;
goto v_reusejp_3518_;
}
else
{
lean_object* v_reuseFailAlloc_3520_; 
v_reuseFailAlloc_3520_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3520_, 0, v_a_3514_);
v___x_3519_ = v_reuseFailAlloc_3520_;
goto v_reusejp_3518_;
}
v_reusejp_3518_:
{
return v___x_3519_;
}
}
}
}
}
else
{
lean_object* v_a_3522_; lean_object* v___x_3524_; uint8_t v_isShared_3525_; uint8_t v_isSharedCheck_3529_; 
lean_dec_ref(v_a_3506_);
lean_dec(v_a_3258_);
lean_dec_ref(v_snd_3227_);
lean_dec(v___x_3226_);
lean_dec_ref(v_rel_3225_);
lean_dec_ref(v_fst_3224_);
lean_dec(v_a_3223_);
lean_dec(v_a_3222_);
lean_dec_ref(v___f_3221_);
lean_dec(v_a_3218_);
v_a_3522_ = lean_ctor_get(v___x_3507_, 0);
v_isSharedCheck_3529_ = !lean_is_exclusive(v___x_3507_);
if (v_isSharedCheck_3529_ == 0)
{
v___x_3524_ = v___x_3507_;
v_isShared_3525_ = v_isSharedCheck_3529_;
goto v_resetjp_3523_;
}
else
{
lean_inc(v_a_3522_);
lean_dec(v___x_3507_);
v___x_3524_ = lean_box(0);
v_isShared_3525_ = v_isSharedCheck_3529_;
goto v_resetjp_3523_;
}
v_resetjp_3523_:
{
lean_object* v___x_3527_; 
if (v_isShared_3525_ == 0)
{
v___x_3527_ = v___x_3524_;
goto v_reusejp_3526_;
}
else
{
lean_object* v_reuseFailAlloc_3528_; 
v_reuseFailAlloc_3528_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3528_, 0, v_a_3522_);
v___x_3527_ = v_reuseFailAlloc_3528_;
goto v_reusejp_3526_;
}
v_reusejp_3526_:
{
return v___x_3527_;
}
}
}
}
v___jp_3530_:
{
if (lean_obj_tag(v_a_3223_) == 0)
{
lean_object* v___x_3535_; uint8_t v___x_3536_; lean_object* v___x_3537_; lean_object* v___x_3538_; 
v___x_3535_ = lean_box(0);
v___x_3536_ = 0;
v___x_3537_ = lean_box(0);
v___x_3538_ = l_Lean_Meta_mkFreshExprMVar(v___x_3535_, v___x_3536_, v___x_3537_, v___y_3531_, v___y_3532_, v___y_3533_, v___y_3534_);
if (lean_obj_tag(v___x_3538_) == 0)
{
lean_object* v_a_3539_; 
v_a_3539_ = lean_ctor_get(v___x_3538_, 0);
lean_inc(v_a_3539_);
lean_dec_ref_known(v___x_3538_, 1);
v___y_3502_ = v___y_3531_;
v___y_3503_ = v___y_3532_;
v___y_3504_ = v___y_3533_;
v___y_3505_ = v___y_3534_;
v_a_3506_ = v_a_3539_;
goto v___jp_3501_;
}
else
{
lean_object* v_a_3540_; lean_object* v___x_3542_; uint8_t v_isShared_3543_; uint8_t v_isSharedCheck_3547_; 
lean_dec(v_a_3258_);
lean_dec_ref(v_snd_3227_);
lean_dec(v___x_3226_);
lean_dec_ref(v_rel_3225_);
lean_dec_ref(v_fst_3224_);
lean_dec(v_a_3222_);
lean_dec_ref(v___f_3221_);
lean_dec(v_a_3218_);
v_a_3540_ = lean_ctor_get(v___x_3538_, 0);
v_isSharedCheck_3547_ = !lean_is_exclusive(v___x_3538_);
if (v_isSharedCheck_3547_ == 0)
{
v___x_3542_ = v___x_3538_;
v_isShared_3543_ = v_isSharedCheck_3547_;
goto v_resetjp_3541_;
}
else
{
lean_inc(v_a_3540_);
lean_dec(v___x_3538_);
v___x_3542_ = lean_box(0);
v_isShared_3543_ = v_isSharedCheck_3547_;
goto v_resetjp_3541_;
}
v_resetjp_3541_:
{
lean_object* v___x_3545_; 
if (v_isShared_3543_ == 0)
{
v___x_3545_ = v___x_3542_;
goto v_reusejp_3544_;
}
else
{
lean_object* v_reuseFailAlloc_3546_; 
v_reuseFailAlloc_3546_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3546_, 0, v_a_3540_);
v___x_3545_ = v_reuseFailAlloc_3546_;
goto v_reusejp_3544_;
}
v_reusejp_3544_:
{
return v___x_3545_;
}
}
}
}
else
{
lean_object* v_val_3548_; lean_object* v_fst_3549_; 
v_val_3548_ = lean_ctor_get(v_a_3223_, 0);
v_fst_3549_ = lean_ctor_get(v_val_3548_, 0);
lean_inc(v_fst_3549_);
v___y_3502_ = v___y_3531_;
v___y_3503_ = v___y_3532_;
v___y_3504_ = v___y_3533_;
v___y_3505_ = v___y_3534_;
v_a_3506_ = v_fst_3549_;
goto v___jp_3501_;
}
}
v___jp_3550_:
{
lean_object* v___x_3555_; 
lean_inc_ref(v___f_3221_);
lean_inc(v___y_3554_);
lean_inc_ref(v___y_3553_);
lean_inc(v___y_3552_);
lean_inc_ref(v___y_3551_);
v___x_3555_ = lean_apply_5(v___f_3221_, v___y_3551_, v___y_3552_, v___y_3553_, v___y_3554_, lean_box(0));
if (lean_obj_tag(v___x_3555_) == 0)
{
lean_object* v_a_3556_; uint8_t v___x_3557_; 
v_a_3556_ = lean_ctor_get(v___x_3555_, 0);
lean_inc(v_a_3556_);
lean_dec_ref_known(v___x_3555_, 1);
v___x_3557_ = lean_unbox(v_a_3556_);
lean_dec(v_a_3556_);
if (v___x_3557_ == 0)
{
lean_dec(v_a_3242_);
v___y_3531_ = v___y_3551_;
v___y_3532_ = v___y_3552_;
v___y_3533_ = v___y_3553_;
v___y_3534_ = v___y_3554_;
goto v___jp_3530_;
}
else
{
lean_object* v___x_3558_; lean_object* v___x_3559_; lean_object* v___x_3560_; lean_object* v___x_3561_; 
v___x_3558_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__13, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__13_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__13);
v___x_3559_ = l_Lean_MessageData_ofExpr(v_a_3242_);
v___x_3560_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3560_, 0, v___x_3558_);
lean_ctor_set(v___x_3560_, 1, v___x_3559_);
lean_inc(v___x_3226_);
v___x_3561_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__5(v___x_3226_, v___x_3560_, v___y_3551_, v___y_3552_, v___y_3553_, v___y_3554_);
if (lean_obj_tag(v___x_3561_) == 0)
{
lean_dec_ref_known(v___x_3561_, 1);
v___y_3531_ = v___y_3551_;
v___y_3532_ = v___y_3552_;
v___y_3533_ = v___y_3553_;
v___y_3534_ = v___y_3554_;
goto v___jp_3530_;
}
else
{
lean_object* v_a_3562_; lean_object* v___x_3564_; uint8_t v_isShared_3565_; uint8_t v_isSharedCheck_3569_; 
lean_dec(v_a_3258_);
lean_dec_ref(v_snd_3227_);
lean_dec(v___x_3226_);
lean_dec_ref(v_rel_3225_);
lean_dec_ref(v_fst_3224_);
lean_dec(v_a_3223_);
lean_dec(v_a_3222_);
lean_dec_ref(v___f_3221_);
lean_dec(v_a_3218_);
v_a_3562_ = lean_ctor_get(v___x_3561_, 0);
v_isSharedCheck_3569_ = !lean_is_exclusive(v___x_3561_);
if (v_isSharedCheck_3569_ == 0)
{
v___x_3564_ = v___x_3561_;
v_isShared_3565_ = v_isSharedCheck_3569_;
goto v_resetjp_3563_;
}
else
{
lean_inc(v_a_3562_);
lean_dec(v___x_3561_);
v___x_3564_ = lean_box(0);
v_isShared_3565_ = v_isSharedCheck_3569_;
goto v_resetjp_3563_;
}
v_resetjp_3563_:
{
lean_object* v___x_3567_; 
if (v_isShared_3565_ == 0)
{
v___x_3567_ = v___x_3564_;
goto v_reusejp_3566_;
}
else
{
lean_object* v_reuseFailAlloc_3568_; 
v_reuseFailAlloc_3568_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3568_, 0, v_a_3562_);
v___x_3567_ = v_reuseFailAlloc_3568_;
goto v_reusejp_3566_;
}
v_reusejp_3566_:
{
return v___x_3567_;
}
}
}
}
}
else
{
lean_object* v_a_3570_; lean_object* v___x_3572_; uint8_t v_isShared_3573_; uint8_t v_isSharedCheck_3577_; 
lean_dec(v_a_3258_);
lean_dec(v_a_3242_);
lean_dec_ref(v_snd_3227_);
lean_dec(v___x_3226_);
lean_dec_ref(v_rel_3225_);
lean_dec_ref(v_fst_3224_);
lean_dec(v_a_3223_);
lean_dec(v_a_3222_);
lean_dec_ref(v___f_3221_);
lean_dec(v_a_3218_);
v_a_3570_ = lean_ctor_get(v___x_3555_, 0);
v_isSharedCheck_3577_ = !lean_is_exclusive(v___x_3555_);
if (v_isSharedCheck_3577_ == 0)
{
v___x_3572_ = v___x_3555_;
v_isShared_3573_ = v_isSharedCheck_3577_;
goto v_resetjp_3571_;
}
else
{
lean_inc(v_a_3570_);
lean_dec(v___x_3555_);
v___x_3572_ = lean_box(0);
v_isShared_3573_ = v_isSharedCheck_3577_;
goto v_resetjp_3571_;
}
v_resetjp_3571_:
{
lean_object* v___x_3575_; 
if (v_isShared_3573_ == 0)
{
v___x_3575_ = v___x_3572_;
goto v_reusejp_3574_;
}
else
{
lean_object* v_reuseFailAlloc_3576_; 
v_reuseFailAlloc_3576_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3576_, 0, v_a_3570_);
v___x_3575_ = v_reuseFailAlloc_3576_;
goto v_reusejp_3574_;
}
v_reusejp_3574_:
{
return v___x_3575_;
}
}
}
}
}
else
{
lean_object* v_a_3603_; lean_object* v___x_3605_; uint8_t v_isShared_3606_; uint8_t v_isSharedCheck_3610_; 
lean_dec(v_a_3242_);
lean_dec_ref(v_snd_3227_);
lean_dec(v___x_3226_);
lean_dec_ref(v_rel_3225_);
lean_dec_ref(v_fst_3224_);
lean_dec(v_a_3223_);
lean_dec(v_a_3222_);
lean_dec_ref(v___f_3221_);
lean_dec(v_a_3218_);
v_a_3603_ = lean_ctor_get(v___x_3257_, 0);
v_isSharedCheck_3610_ = !lean_is_exclusive(v___x_3257_);
if (v_isSharedCheck_3610_ == 0)
{
v___x_3605_ = v___x_3257_;
v_isShared_3606_ = v_isSharedCheck_3610_;
goto v_resetjp_3604_;
}
else
{
lean_inc(v_a_3603_);
lean_dec(v___x_3257_);
v___x_3605_ = lean_box(0);
v_isShared_3606_ = v_isSharedCheck_3610_;
goto v_resetjp_3604_;
}
v_resetjp_3604_:
{
lean_object* v___x_3608_; 
if (v_isShared_3606_ == 0)
{
v___x_3608_ = v___x_3605_;
goto v_reusejp_3607_;
}
else
{
lean_object* v_reuseFailAlloc_3609_; 
v_reuseFailAlloc_3609_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3609_, 0, v_a_3603_);
v___x_3608_ = v_reuseFailAlloc_3609_;
goto v_reusejp_3607_;
}
v_reusejp_3607_:
{
return v___x_3608_;
}
}
}
}
else
{
lean_object* v_a_3611_; lean_object* v___x_3613_; uint8_t v_isShared_3614_; uint8_t v_isSharedCheck_3618_; 
lean_dec_ref(v_snd_3227_);
lean_dec(v___x_3226_);
lean_dec_ref(v_rel_3225_);
lean_dec_ref(v_fst_3224_);
lean_dec(v_a_3223_);
lean_dec(v_a_3222_);
lean_dec_ref(v___f_3221_);
lean_dec_ref(v___f_3219_);
lean_dec(v_a_3218_);
v_a_3611_ = lean_ctor_get(v___x_3241_, 0);
v_isSharedCheck_3618_ = !lean_is_exclusive(v___x_3241_);
if (v_isSharedCheck_3618_ == 0)
{
v___x_3613_ = v___x_3241_;
v_isShared_3614_ = v_isSharedCheck_3618_;
goto v_resetjp_3612_;
}
else
{
lean_inc(v_a_3611_);
lean_dec(v___x_3241_);
v___x_3613_ = lean_box(0);
v_isShared_3614_ = v_isSharedCheck_3618_;
goto v_resetjp_3612_;
}
v_resetjp_3612_:
{
lean_object* v___x_3616_; 
if (v_isShared_3614_ == 0)
{
v___x_3616_ = v___x_3613_;
goto v_reusejp_3615_;
}
else
{
lean_object* v_reuseFailAlloc_3617_; 
v_reuseFailAlloc_3617_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3617_, 0, v_a_3611_);
v___x_3616_ = v_reuseFailAlloc_3617_;
goto v_reusejp_3615_;
}
v_reusejp_3615_:
{
return v___x_3616_;
}
}
}
}
else
{
lean_object* v_a_3619_; lean_object* v___x_3621_; uint8_t v_isShared_3622_; uint8_t v_isSharedCheck_3626_; 
lean_dec_ref(v_snd_3227_);
lean_dec(v___x_3226_);
lean_dec_ref(v_rel_3225_);
lean_dec_ref(v_fst_3224_);
lean_dec(v_a_3223_);
lean_dec(v_a_3222_);
lean_dec_ref(v___f_3221_);
lean_dec_ref(v___f_3219_);
lean_dec(v_a_3218_);
v_a_3619_ = lean_ctor_get(v___x_3239_, 0);
v_isSharedCheck_3626_ = !lean_is_exclusive(v___x_3239_);
if (v_isSharedCheck_3626_ == 0)
{
v___x_3621_ = v___x_3239_;
v_isShared_3622_ = v_isSharedCheck_3626_;
goto v_resetjp_3620_;
}
else
{
lean_inc(v_a_3619_);
lean_dec(v___x_3239_);
v___x_3621_ = lean_box(0);
v_isShared_3622_ = v_isSharedCheck_3626_;
goto v_resetjp_3620_;
}
v_resetjp_3620_:
{
lean_object* v___x_3624_; 
if (v_isShared_3622_ == 0)
{
v___x_3624_ = v___x_3621_;
goto v_reusejp_3623_;
}
else
{
lean_object* v_reuseFailAlloc_3625_; 
v_reuseFailAlloc_3625_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3625_, 0, v_a_3619_);
v___x_3624_ = v_reuseFailAlloc_3625_;
goto v_reusejp_3623_;
}
v_reusejp_3623_:
{
return v___x_3624_;
}
}
}
v___jp_3234_:
{
lean_object* v___x_3237_; lean_object* v___x_3238_; 
v___x_3237_ = l_List_appendTR___redArg(v___y_3235_, v___y_3236_);
v___x_3238_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3238_, 0, v___x_3237_);
return v___x_3238_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__2___boxed(lean_object* v_a_3627_, lean_object* v___f_3628_, lean_object* v___y_3629_, lean_object* v___f_3630_, lean_object* v_a_3631_, lean_object* v_a_3632_, lean_object* v_fst_3633_, lean_object* v_rel_3634_, lean_object* v___x_3635_, lean_object* v_snd_3636_, lean_object* v_____r_3637_, lean_object* v___y_3638_, lean_object* v___y_3639_, lean_object* v___y_3640_, lean_object* v___y_3641_, lean_object* v___y_3642_){
_start:
{
uint8_t v___y_93845__boxed_3643_; lean_object* v_res_3644_; 
v___y_93845__boxed_3643_ = lean_unbox(v___y_3629_);
v_res_3644_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__2(v_a_3627_, v___f_3628_, v___y_93845__boxed_3643_, v___f_3630_, v_a_3631_, v_a_3632_, v_fst_3633_, v_rel_3634_, v___x_3635_, v_snd_3636_, v_____r_3637_, v___y_3638_, v___y_3639_, v___y_3640_, v___y_3641_);
lean_dec(v___y_3641_);
lean_dec_ref(v___y_3640_);
lean_dec(v___y_3639_);
lean_dec_ref(v___y_3638_);
return v_res_3644_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__3(lean_object* v___f_3645_, lean_object* v___x_3646_, lean_object* v_a_3647_, lean_object* v___f_3648_, uint8_t v___y_3649_, lean_object* v_a_3650_, lean_object* v_fst_3651_, lean_object* v_rel_3652_, lean_object* v___x_3653_, lean_object* v_snd_3654_, lean_object* v___y_3655_, lean_object* v___y_3656_, lean_object* v___y_3657_, lean_object* v___y_3658_, lean_object* v___y_3659_, lean_object* v___y_3660_, lean_object* v___y_3661_, lean_object* v___y_3662_){
_start:
{
lean_object* v___y_3665_; lean_object* v___x_3684_; 
v___x_3684_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_3656_, v___y_3659_, v___y_3660_, v___y_3661_, v___y_3662_);
if (lean_obj_tag(v___x_3684_) == 0)
{
lean_object* v_a_3685_; lean_object* v___x_3686_; 
v_a_3685_ = lean_ctor_get(v___x_3684_, 0);
lean_inc(v_a_3685_);
lean_dec_ref_known(v___x_3684_, 1);
lean_inc_ref(v___f_3645_);
lean_inc(v___y_3662_);
lean_inc_ref(v___y_3661_);
lean_inc(v___y_3660_);
lean_inc_ref(v___y_3659_);
v___x_3686_ = lean_apply_5(v___f_3645_, v___y_3659_, v___y_3660_, v___y_3661_, v___y_3662_, lean_box(0));
if (lean_obj_tag(v___x_3686_) == 0)
{
lean_object* v_a_3687_; uint8_t v___x_3688_; 
v_a_3687_ = lean_ctor_get(v___x_3686_, 0);
lean_inc(v_a_3687_);
lean_dec_ref_known(v___x_3686_, 1);
v___x_3688_ = lean_unbox(v_a_3687_);
lean_dec(v_a_3687_);
if (v___x_3688_ == 0)
{
lean_object* v___x_3689_; 
v___x_3689_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__2(v_a_3647_, v___f_3648_, v___y_3649_, v___f_3645_, v_a_3685_, v_a_3650_, v_fst_3651_, v_rel_3652_, v___x_3653_, v_snd_3654_, v___x_3646_, v___y_3659_, v___y_3660_, v___y_3661_, v___y_3662_);
v___y_3665_ = v___x_3689_;
goto v___jp_3664_;
}
else
{
lean_object* v___x_3690_; lean_object* v___x_3691_; lean_object* v___x_3692_; lean_object* v___x_3693_; 
v___x_3690_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__0___closed__1, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__0___closed__1_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__0___closed__1);
lean_inc(v_a_3647_);
v___x_3691_ = l_Lean_MessageData_ofName(v_a_3647_);
v___x_3692_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3692_, 0, v___x_3690_);
lean_ctor_set(v___x_3692_, 1, v___x_3691_);
lean_inc(v___x_3653_);
v___x_3693_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__5(v___x_3653_, v___x_3692_, v___y_3659_, v___y_3660_, v___y_3661_, v___y_3662_);
if (lean_obj_tag(v___x_3693_) == 0)
{
lean_object* v_a_3694_; lean_object* v___x_3695_; 
v_a_3694_ = lean_ctor_get(v___x_3693_, 0);
lean_inc(v_a_3694_);
lean_dec_ref_known(v___x_3693_, 1);
v___x_3695_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__2(v_a_3647_, v___f_3648_, v___y_3649_, v___f_3645_, v_a_3685_, v_a_3650_, v_fst_3651_, v_rel_3652_, v___x_3653_, v_snd_3654_, v_a_3694_, v___y_3659_, v___y_3660_, v___y_3661_, v___y_3662_);
v___y_3665_ = v___x_3695_;
goto v___jp_3664_;
}
else
{
lean_dec(v_a_3685_);
lean_dec(v___y_3662_);
lean_dec_ref(v___y_3661_);
lean_dec(v___y_3660_);
lean_dec_ref(v___y_3659_);
lean_dec_ref(v_snd_3654_);
lean_dec(v___x_3653_);
lean_dec_ref(v_rel_3652_);
lean_dec_ref(v_fst_3651_);
lean_dec(v_a_3650_);
lean_dec_ref(v___f_3648_);
lean_dec(v_a_3647_);
lean_dec_ref(v___f_3645_);
return v___x_3693_;
}
}
}
else
{
lean_object* v_a_3696_; lean_object* v___x_3698_; uint8_t v_isShared_3699_; uint8_t v_isSharedCheck_3703_; 
lean_dec(v_a_3685_);
lean_dec(v___y_3662_);
lean_dec_ref(v___y_3661_);
lean_dec(v___y_3660_);
lean_dec_ref(v___y_3659_);
lean_dec_ref(v_snd_3654_);
lean_dec(v___x_3653_);
lean_dec_ref(v_rel_3652_);
lean_dec_ref(v_fst_3651_);
lean_dec(v_a_3650_);
lean_dec_ref(v___f_3648_);
lean_dec(v_a_3647_);
lean_dec_ref(v___f_3645_);
v_a_3696_ = lean_ctor_get(v___x_3686_, 0);
v_isSharedCheck_3703_ = !lean_is_exclusive(v___x_3686_);
if (v_isSharedCheck_3703_ == 0)
{
v___x_3698_ = v___x_3686_;
v_isShared_3699_ = v_isSharedCheck_3703_;
goto v_resetjp_3697_;
}
else
{
lean_inc(v_a_3696_);
lean_dec(v___x_3686_);
v___x_3698_ = lean_box(0);
v_isShared_3699_ = v_isSharedCheck_3703_;
goto v_resetjp_3697_;
}
v_resetjp_3697_:
{
lean_object* v___x_3701_; 
if (v_isShared_3699_ == 0)
{
v___x_3701_ = v___x_3698_;
goto v_reusejp_3700_;
}
else
{
lean_object* v_reuseFailAlloc_3702_; 
v_reuseFailAlloc_3702_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3702_, 0, v_a_3696_);
v___x_3701_ = v_reuseFailAlloc_3702_;
goto v_reusejp_3700_;
}
v_reusejp_3700_:
{
return v___x_3701_;
}
}
}
}
else
{
lean_object* v_a_3704_; lean_object* v___x_3706_; uint8_t v_isShared_3707_; uint8_t v_isSharedCheck_3711_; 
lean_dec(v___y_3662_);
lean_dec_ref(v___y_3661_);
lean_dec(v___y_3660_);
lean_dec_ref(v___y_3659_);
lean_dec_ref(v_snd_3654_);
lean_dec(v___x_3653_);
lean_dec_ref(v_rel_3652_);
lean_dec_ref(v_fst_3651_);
lean_dec(v_a_3650_);
lean_dec_ref(v___f_3648_);
lean_dec(v_a_3647_);
lean_dec_ref(v___f_3645_);
v_a_3704_ = lean_ctor_get(v___x_3684_, 0);
v_isSharedCheck_3711_ = !lean_is_exclusive(v___x_3684_);
if (v_isSharedCheck_3711_ == 0)
{
v___x_3706_ = v___x_3684_;
v_isShared_3707_ = v_isSharedCheck_3711_;
goto v_resetjp_3705_;
}
else
{
lean_inc(v_a_3704_);
lean_dec(v___x_3684_);
v___x_3706_ = lean_box(0);
v_isShared_3707_ = v_isSharedCheck_3711_;
goto v_resetjp_3705_;
}
v_resetjp_3705_:
{
lean_object* v___x_3709_; 
if (v_isShared_3707_ == 0)
{
v___x_3709_ = v___x_3706_;
goto v_reusejp_3708_;
}
else
{
lean_object* v_reuseFailAlloc_3710_; 
v_reuseFailAlloc_3710_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3710_, 0, v_a_3704_);
v___x_3709_ = v_reuseFailAlloc_3710_;
goto v_reusejp_3708_;
}
v_reusejp_3708_:
{
return v___x_3709_;
}
}
}
v___jp_3664_:
{
if (lean_obj_tag(v___y_3665_) == 0)
{
lean_object* v_a_3666_; lean_object* v___x_3667_; 
v_a_3666_ = lean_ctor_get(v___y_3665_, 0);
lean_inc(v_a_3666_);
lean_dec_ref_known(v___y_3665_, 1);
v___x_3667_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v_a_3666_, v___y_3656_, v___y_3659_, v___y_3660_, v___y_3661_, v___y_3662_);
lean_dec(v___y_3662_);
lean_dec_ref(v___y_3661_);
lean_dec(v___y_3660_);
lean_dec_ref(v___y_3659_);
if (lean_obj_tag(v___x_3667_) == 0)
{
lean_object* v___x_3669_; uint8_t v_isShared_3670_; uint8_t v_isSharedCheck_3674_; 
v_isSharedCheck_3674_ = !lean_is_exclusive(v___x_3667_);
if (v_isSharedCheck_3674_ == 0)
{
lean_object* v_unused_3675_; 
v_unused_3675_ = lean_ctor_get(v___x_3667_, 0);
lean_dec(v_unused_3675_);
v___x_3669_ = v___x_3667_;
v_isShared_3670_ = v_isSharedCheck_3674_;
goto v_resetjp_3668_;
}
else
{
lean_dec(v___x_3667_);
v___x_3669_ = lean_box(0);
v_isShared_3670_ = v_isSharedCheck_3674_;
goto v_resetjp_3668_;
}
v_resetjp_3668_:
{
lean_object* v___x_3672_; 
if (v_isShared_3670_ == 0)
{
lean_ctor_set(v___x_3669_, 0, v___x_3646_);
v___x_3672_ = v___x_3669_;
goto v_reusejp_3671_;
}
else
{
lean_object* v_reuseFailAlloc_3673_; 
v_reuseFailAlloc_3673_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3673_, 0, v___x_3646_);
v___x_3672_ = v_reuseFailAlloc_3673_;
goto v_reusejp_3671_;
}
v_reusejp_3671_:
{
return v___x_3672_;
}
}
}
else
{
return v___x_3667_;
}
}
else
{
lean_object* v_a_3676_; lean_object* v___x_3678_; uint8_t v_isShared_3679_; uint8_t v_isSharedCheck_3683_; 
lean_dec(v___y_3662_);
lean_dec_ref(v___y_3661_);
lean_dec(v___y_3660_);
lean_dec_ref(v___y_3659_);
v_a_3676_ = lean_ctor_get(v___y_3665_, 0);
v_isSharedCheck_3683_ = !lean_is_exclusive(v___y_3665_);
if (v_isSharedCheck_3683_ == 0)
{
v___x_3678_ = v___y_3665_;
v_isShared_3679_ = v_isSharedCheck_3683_;
goto v_resetjp_3677_;
}
else
{
lean_inc(v_a_3676_);
lean_dec(v___y_3665_);
v___x_3678_ = lean_box(0);
v_isShared_3679_ = v_isSharedCheck_3683_;
goto v_resetjp_3677_;
}
v_resetjp_3677_:
{
lean_object* v___x_3681_; 
if (v_isShared_3679_ == 0)
{
v___x_3681_ = v___x_3678_;
goto v_reusejp_3680_;
}
else
{
lean_object* v_reuseFailAlloc_3682_; 
v_reuseFailAlloc_3682_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3682_, 0, v_a_3676_);
v___x_3681_ = v_reuseFailAlloc_3682_;
goto v_reusejp_3680_;
}
v_reusejp_3680_:
{
return v___x_3681_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__3___boxed(lean_object** _args){
lean_object* v___f_3712_ = _args[0];
lean_object* v___x_3713_ = _args[1];
lean_object* v_a_3714_ = _args[2];
lean_object* v___f_3715_ = _args[3];
lean_object* v___y_3716_ = _args[4];
lean_object* v_a_3717_ = _args[5];
lean_object* v_fst_3718_ = _args[6];
lean_object* v_rel_3719_ = _args[7];
lean_object* v___x_3720_ = _args[8];
lean_object* v_snd_3721_ = _args[9];
lean_object* v___y_3722_ = _args[10];
lean_object* v___y_3723_ = _args[11];
lean_object* v___y_3724_ = _args[12];
lean_object* v___y_3725_ = _args[13];
lean_object* v___y_3726_ = _args[14];
lean_object* v___y_3727_ = _args[15];
lean_object* v___y_3728_ = _args[16];
lean_object* v___y_3729_ = _args[17];
lean_object* v___y_3730_ = _args[18];
_start:
{
uint8_t v___y_94670__boxed_3731_; lean_object* v_res_3732_; 
v___y_94670__boxed_3731_ = lean_unbox(v___y_3716_);
v_res_3732_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__3(v___f_3712_, v___x_3713_, v_a_3714_, v___f_3715_, v___y_94670__boxed_3731_, v_a_3717_, v_fst_3718_, v_rel_3719_, v___x_3720_, v_snd_3721_, v___y_3722_, v___y_3723_, v___y_3724_, v___y_3725_, v___y_3726_, v___y_3727_, v___y_3728_, v___y_3729_);
lean_dec(v___y_3725_);
lean_dec_ref(v___y_3724_);
lean_dec(v___y_3723_);
lean_dec_ref(v___y_3722_);
return v_res_3732_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7(uint8_t v___y_3733_, lean_object* v_snd_3734_, lean_object* v_rel_3735_, lean_object* v_fst_3736_, lean_object* v_a_3737_, lean_object* v_a_3738_, lean_object* v_as_3739_, size_t v_sz_3740_, size_t v_i_3741_, lean_object* v_b_3742_, lean_object* v___y_3743_, lean_object* v___y_3744_, lean_object* v___y_3745_, lean_object* v___y_3746_, lean_object* v___y_3747_, lean_object* v___y_3748_, lean_object* v___y_3749_, lean_object* v___y_3750_){
_start:
{
uint8_t v___x_3752_; 
v___x_3752_ = lean_usize_dec_lt(v_i_3741_, v_sz_3740_);
if (v___x_3752_ == 0)
{
lean_object* v___x_3753_; 
lean_dec_ref(v_a_3738_);
lean_dec(v_a_3737_);
lean_dec_ref(v_fst_3736_);
lean_dec_ref(v_rel_3735_);
lean_dec_ref(v_snd_3734_);
v___x_3753_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3753_, 0, v_b_3742_);
return v___x_3753_;
}
else
{
lean_object* v___x_3754_; 
lean_dec_ref(v_b_3742_);
v___x_3754_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_3744_, v___y_3746_, v___y_3748_, v___y_3750_);
if (lean_obj_tag(v___x_3754_) == 0)
{
lean_object* v_a_3755_; lean_object* v___x_3757_; uint8_t v_isShared_3758_; uint8_t v_isSharedCheck_3835_; 
v_a_3755_ = lean_ctor_get(v___x_3754_, 0);
v_isSharedCheck_3835_ = !lean_is_exclusive(v___x_3754_);
if (v_isSharedCheck_3835_ == 0)
{
v___x_3757_ = v___x_3754_;
v_isShared_3758_ = v_isSharedCheck_3835_;
goto v_resetjp_3756_;
}
else
{
lean_inc(v_a_3755_);
lean_dec(v___x_3754_);
v___x_3757_ = lean_box(0);
v_isShared_3758_ = v_isSharedCheck_3835_;
goto v_resetjp_3756_;
}
v_resetjp_3756_:
{
lean_object* v___f_3759_; lean_object* v___x_3760_; lean_object* v_a_3762_; lean_object* v___x_3768_; lean_object* v___f_3769_; lean_object* v_a_3770_; lean_object* v___x_3771_; lean_object* v___f_3772_; lean_object* v___x_3773_; 
v___f_3759_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__0));
v___x_3760_ = lean_box(0);
v___x_3768_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__2_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_));
v___f_3769_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__1));
v_a_3770_ = lean_array_uget_borrowed(v_as_3739_, v_i_3741_);
v___x_3771_ = lean_box(v___y_3733_);
lean_inc_ref(v_snd_3734_);
lean_inc_ref(v_rel_3735_);
lean_inc_ref(v_fst_3736_);
lean_inc(v_a_3737_);
lean_inc(v_a_3770_);
v___f_3772_ = lean_alloc_closure((void*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__3___boxed), 19, 10);
lean_closure_set(v___f_3772_, 0, v___f_3769_);
lean_closure_set(v___f_3772_, 1, v___x_3760_);
lean_closure_set(v___f_3772_, 2, v_a_3770_);
lean_closure_set(v___f_3772_, 3, v___f_3759_);
lean_closure_set(v___f_3772_, 4, v___x_3771_);
lean_closure_set(v___f_3772_, 5, v_a_3737_);
lean_closure_set(v___f_3772_, 6, v_fst_3736_);
lean_closure_set(v___f_3772_, 7, v_rel_3735_);
lean_closure_set(v___f_3772_, 8, v___x_3768_);
lean_closure_set(v___f_3772_, 9, v_snd_3734_);
v___x_3773_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_3772_, v___y_3743_, v___y_3744_, v___y_3745_, v___y_3746_, v___y_3747_, v___y_3748_, v___y_3749_, v___y_3750_);
if (lean_obj_tag(v___x_3773_) == 0)
{
lean_dec_ref_known(v___x_3773_, 1);
lean_dec(v_a_3755_);
lean_dec_ref(v_a_3738_);
lean_dec(v_a_3737_);
lean_dec_ref(v_fst_3736_);
lean_dec_ref(v_rel_3735_);
lean_dec_ref(v_snd_3734_);
v_a_3762_ = v___x_3760_;
goto v___jp_3761_;
}
else
{
lean_object* v_a_3774_; lean_object* v___x_3776_; uint8_t v_isShared_3777_; uint8_t v_isSharedCheck_3834_; 
v_a_3774_ = lean_ctor_get(v___x_3773_, 0);
v_isSharedCheck_3834_ = !lean_is_exclusive(v___x_3773_);
if (v_isSharedCheck_3834_ == 0)
{
v___x_3776_ = v___x_3773_;
v_isShared_3777_ = v_isSharedCheck_3834_;
goto v_resetjp_3775_;
}
else
{
lean_inc(v_a_3774_);
lean_dec(v___x_3773_);
v___x_3776_ = lean_box(0);
v_isShared_3777_ = v_isSharedCheck_3834_;
goto v_resetjp_3775_;
}
v_resetjp_3775_:
{
lean_object* v___x_3778_; lean_object* v___y_3780_; lean_object* v___y_3795_; uint8_t v___y_3798_; uint8_t v___x_3832_; 
v___x_3778_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__2));
v___x_3832_ = l_Lean_Exception_isInterrupt(v_a_3774_);
if (v___x_3832_ == 0)
{
uint8_t v___x_3833_; 
lean_inc(v_a_3774_);
v___x_3833_ = l_Lean_Exception_isRuntime(v_a_3774_);
v___y_3798_ = v___x_3833_;
goto v___jp_3797_;
}
else
{
v___y_3798_ = v___x_3832_;
goto v___jp_3797_;
}
v___jp_3779_:
{
if (lean_obj_tag(v___y_3780_) == 0)
{
lean_object* v_a_3781_; 
v_a_3781_ = lean_ctor_get(v___y_3780_, 0);
lean_inc(v_a_3781_);
lean_dec_ref_known(v___y_3780_, 1);
if (lean_obj_tag(v_a_3781_) == 0)
{
lean_object* v_a_3782_; 
lean_dec_ref(v_a_3738_);
lean_dec(v_a_3737_);
lean_dec_ref(v_fst_3736_);
lean_dec_ref(v_rel_3735_);
lean_dec_ref(v_snd_3734_);
v_a_3782_ = lean_ctor_get(v_a_3781_, 0);
lean_inc(v_a_3782_);
lean_dec_ref_known(v_a_3781_, 1);
v_a_3762_ = v_a_3782_;
goto v___jp_3761_;
}
else
{
size_t v___x_3783_; size_t v___x_3784_; lean_object* v___x_3785_; 
lean_dec_ref_known(v_a_3781_, 1);
lean_del_object(v___x_3757_);
v___x_3783_ = ((size_t)1ULL);
v___x_3784_ = lean_usize_add(v_i_3741_, v___x_3783_);
v___x_3785_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10(v___y_3733_, v_snd_3734_, v_rel_3735_, v_fst_3736_, v_a_3737_, v_a_3738_, v_as_3739_, v_sz_3740_, v___x_3784_, v___x_3778_, v___y_3743_, v___y_3744_, v___y_3745_, v___y_3746_, v___y_3747_, v___y_3748_, v___y_3749_, v___y_3750_);
return v___x_3785_;
}
}
else
{
lean_object* v_a_3786_; lean_object* v___x_3788_; uint8_t v_isShared_3789_; uint8_t v_isSharedCheck_3793_; 
lean_del_object(v___x_3757_);
lean_dec_ref(v_a_3738_);
lean_dec(v_a_3737_);
lean_dec_ref(v_fst_3736_);
lean_dec_ref(v_rel_3735_);
lean_dec_ref(v_snd_3734_);
v_a_3786_ = lean_ctor_get(v___y_3780_, 0);
v_isSharedCheck_3793_ = !lean_is_exclusive(v___y_3780_);
if (v_isSharedCheck_3793_ == 0)
{
v___x_3788_ = v___y_3780_;
v_isShared_3789_ = v_isSharedCheck_3793_;
goto v_resetjp_3787_;
}
else
{
lean_inc(v_a_3786_);
lean_dec(v___y_3780_);
v___x_3788_ = lean_box(0);
v_isShared_3789_ = v_isSharedCheck_3793_;
goto v_resetjp_3787_;
}
v_resetjp_3787_:
{
lean_object* v___x_3791_; 
if (v_isShared_3789_ == 0)
{
v___x_3791_ = v___x_3788_;
goto v_reusejp_3790_;
}
else
{
lean_object* v_reuseFailAlloc_3792_; 
v_reuseFailAlloc_3792_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3792_, 0, v_a_3786_);
v___x_3791_ = v_reuseFailAlloc_3792_;
goto v_reusejp_3790_;
}
v_reusejp_3790_:
{
return v___x_3791_;
}
}
}
}
v___jp_3794_:
{
lean_object* v___x_3796_; 
lean_inc(v___y_3750_);
lean_inc_ref(v___y_3749_);
lean_inc(v___y_3748_);
lean_inc_ref(v___y_3747_);
lean_inc(v___y_3746_);
lean_inc_ref(v___y_3745_);
lean_inc(v___y_3744_);
lean_inc_ref(v___y_3743_);
v___x_3796_ = lean_apply_10(v___y_3795_, v___x_3760_, v___y_3743_, v___y_3744_, v___y_3745_, v___y_3746_, v___y_3747_, v___y_3748_, v___y_3749_, v___y_3750_, lean_box(0));
v___y_3780_ = v___x_3796_;
goto v___jp_3779_;
}
v___jp_3797_:
{
if (v___y_3798_ == 0)
{
lean_object* v___x_3799_; 
lean_del_object(v___x_3776_);
v___x_3799_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_3755_, v___y_3798_, v___y_3744_, v___y_3745_, v___y_3746_, v___y_3747_, v___y_3748_, v___y_3749_, v___y_3750_);
if (lean_obj_tag(v___x_3799_) == 0)
{
lean_object* v_options_3800_; lean_object* v_inheritedTraceOptions_3801_; uint8_t v_hasTrace_3802_; lean_object* v___x_3803_; lean_object* v___f_3804_; 
lean_dec_ref_known(v___x_3799_, 1);
v_options_3800_ = lean_ctor_get(v___y_3749_, 2);
v_inheritedTraceOptions_3801_ = lean_ctor_get(v___y_3749_, 13);
v_hasTrace_3802_ = lean_ctor_get_uint8(v_options_3800_, sizeof(void*)*1);
v___x_3803_ = lean_box(v___y_3798_);
lean_inc_ref(v_a_3738_);
v___f_3804_ = lean_alloc_closure((void*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__4___boxed), 12, 2);
lean_closure_set(v___f_3804_, 0, v_a_3738_);
lean_closure_set(v___f_3804_, 1, v___x_3803_);
if (v_hasTrace_3802_ == 0)
{
lean_dec(v_a_3774_);
v___y_3795_ = v___f_3804_;
goto v___jp_3794_;
}
else
{
lean_object* v___x_3805_; uint8_t v___x_3806_; 
v___x_3805_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__3, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__3_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__3);
v___x_3806_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3801_, v_options_3800_, v___x_3805_);
if (v___x_3806_ == 0)
{
lean_dec(v_a_3774_);
v___y_3795_ = v___f_3804_;
goto v___jp_3794_;
}
else
{
lean_object* v___x_3807_; lean_object* v___x_3808_; lean_object* v___x_3809_; lean_object* v___x_3810_; 
lean_dec_ref(v___f_3804_);
v___x_3807_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__5, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__5_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__5);
v___x_3808_ = l_Lean_Exception_toMessageData(v_a_3774_);
v___x_3809_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3809_, 0, v___x_3807_);
lean_ctor_set(v___x_3809_, 1, v___x_3808_);
v___x_3810_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg(v___x_3768_, v___x_3809_, v___y_3747_, v___y_3748_, v___y_3749_, v___y_3750_);
if (lean_obj_tag(v___x_3810_) == 0)
{
lean_object* v_a_3811_; lean_object* v___x_3812_; 
v_a_3811_ = lean_ctor_get(v___x_3810_, 0);
lean_inc(v_a_3811_);
lean_dec_ref_known(v___x_3810_, 1);
lean_inc_ref(v_a_3738_);
v___x_3812_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___lam__4(v_a_3738_, v___y_3798_, v_a_3811_, v___y_3743_, v___y_3744_, v___y_3745_, v___y_3746_, v___y_3747_, v___y_3748_, v___y_3749_, v___y_3750_);
v___y_3780_ = v___x_3812_;
goto v___jp_3779_;
}
else
{
lean_object* v_a_3813_; lean_object* v___x_3815_; uint8_t v_isShared_3816_; uint8_t v_isSharedCheck_3820_; 
lean_del_object(v___x_3757_);
lean_dec_ref(v_a_3738_);
lean_dec(v_a_3737_);
lean_dec_ref(v_fst_3736_);
lean_dec_ref(v_rel_3735_);
lean_dec_ref(v_snd_3734_);
v_a_3813_ = lean_ctor_get(v___x_3810_, 0);
v_isSharedCheck_3820_ = !lean_is_exclusive(v___x_3810_);
if (v_isSharedCheck_3820_ == 0)
{
v___x_3815_ = v___x_3810_;
v_isShared_3816_ = v_isSharedCheck_3820_;
goto v_resetjp_3814_;
}
else
{
lean_inc(v_a_3813_);
lean_dec(v___x_3810_);
v___x_3815_ = lean_box(0);
v_isShared_3816_ = v_isSharedCheck_3820_;
goto v_resetjp_3814_;
}
v_resetjp_3814_:
{
lean_object* v___x_3818_; 
if (v_isShared_3816_ == 0)
{
v___x_3818_ = v___x_3815_;
goto v_reusejp_3817_;
}
else
{
lean_object* v_reuseFailAlloc_3819_; 
v_reuseFailAlloc_3819_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3819_, 0, v_a_3813_);
v___x_3818_ = v_reuseFailAlloc_3819_;
goto v_reusejp_3817_;
}
v_reusejp_3817_:
{
return v___x_3818_;
}
}
}
}
}
}
else
{
lean_object* v_a_3821_; lean_object* v___x_3823_; uint8_t v_isShared_3824_; uint8_t v_isSharedCheck_3828_; 
lean_dec(v_a_3774_);
lean_del_object(v___x_3757_);
lean_dec_ref(v_a_3738_);
lean_dec(v_a_3737_);
lean_dec_ref(v_fst_3736_);
lean_dec_ref(v_rel_3735_);
lean_dec_ref(v_snd_3734_);
v_a_3821_ = lean_ctor_get(v___x_3799_, 0);
v_isSharedCheck_3828_ = !lean_is_exclusive(v___x_3799_);
if (v_isSharedCheck_3828_ == 0)
{
v___x_3823_ = v___x_3799_;
v_isShared_3824_ = v_isSharedCheck_3828_;
goto v_resetjp_3822_;
}
else
{
lean_inc(v_a_3821_);
lean_dec(v___x_3799_);
v___x_3823_ = lean_box(0);
v_isShared_3824_ = v_isSharedCheck_3828_;
goto v_resetjp_3822_;
}
v_resetjp_3822_:
{
lean_object* v___x_3826_; 
if (v_isShared_3824_ == 0)
{
v___x_3826_ = v___x_3823_;
goto v_reusejp_3825_;
}
else
{
lean_object* v_reuseFailAlloc_3827_; 
v_reuseFailAlloc_3827_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3827_, 0, v_a_3821_);
v___x_3826_ = v_reuseFailAlloc_3827_;
goto v_reusejp_3825_;
}
v_reusejp_3825_:
{
return v___x_3826_;
}
}
}
}
else
{
lean_object* v___x_3830_; 
lean_del_object(v___x_3757_);
lean_dec(v_a_3755_);
lean_dec_ref(v_a_3738_);
lean_dec(v_a_3737_);
lean_dec_ref(v_fst_3736_);
lean_dec_ref(v_rel_3735_);
lean_dec_ref(v_snd_3734_);
if (v_isShared_3777_ == 0)
{
v___x_3830_ = v___x_3776_;
goto v_reusejp_3829_;
}
else
{
lean_object* v_reuseFailAlloc_3831_; 
v_reuseFailAlloc_3831_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3831_, 0, v_a_3774_);
v___x_3830_ = v_reuseFailAlloc_3831_;
goto v_reusejp_3829_;
}
v_reusejp_3829_:
{
return v___x_3830_;
}
}
}
}
}
v___jp_3761_:
{
lean_object* v___x_3763_; lean_object* v___x_3764_; lean_object* v___x_3766_; 
v___x_3763_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3763_, 0, v_a_3762_);
v___x_3764_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3764_, 0, v___x_3763_);
lean_ctor_set(v___x_3764_, 1, v___x_3760_);
if (v_isShared_3758_ == 0)
{
lean_ctor_set(v___x_3757_, 0, v___x_3764_);
v___x_3766_ = v___x_3757_;
goto v_reusejp_3765_;
}
else
{
lean_object* v_reuseFailAlloc_3767_; 
v_reuseFailAlloc_3767_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3767_, 0, v___x_3764_);
v___x_3766_ = v_reuseFailAlloc_3767_;
goto v_reusejp_3765_;
}
v_reusejp_3765_:
{
return v___x_3766_;
}
}
}
}
else
{
lean_object* v_a_3836_; lean_object* v___x_3838_; uint8_t v_isShared_3839_; uint8_t v_isSharedCheck_3843_; 
lean_dec_ref(v_a_3738_);
lean_dec(v_a_3737_);
lean_dec_ref(v_fst_3736_);
lean_dec_ref(v_rel_3735_);
lean_dec_ref(v_snd_3734_);
v_a_3836_ = lean_ctor_get(v___x_3754_, 0);
v_isSharedCheck_3843_ = !lean_is_exclusive(v___x_3754_);
if (v_isSharedCheck_3843_ == 0)
{
v___x_3838_ = v___x_3754_;
v_isShared_3839_ = v_isSharedCheck_3843_;
goto v_resetjp_3837_;
}
else
{
lean_inc(v_a_3836_);
lean_dec(v___x_3754_);
v___x_3838_ = lean_box(0);
v_isShared_3839_ = v_isSharedCheck_3843_;
goto v_resetjp_3837_;
}
v_resetjp_3837_:
{
lean_object* v___x_3841_; 
if (v_isShared_3839_ == 0)
{
v___x_3841_ = v___x_3838_;
goto v_reusejp_3840_;
}
else
{
lean_object* v_reuseFailAlloc_3842_; 
v_reuseFailAlloc_3842_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3842_, 0, v_a_3836_);
v___x_3841_ = v_reuseFailAlloc_3842_;
goto v_reusejp_3840_;
}
v_reusejp_3840_:
{
return v___x_3841_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7___boxed(lean_object** _args){
lean_object* v___y_3844_ = _args[0];
lean_object* v_snd_3845_ = _args[1];
lean_object* v_rel_3846_ = _args[2];
lean_object* v_fst_3847_ = _args[3];
lean_object* v_a_3848_ = _args[4];
lean_object* v_a_3849_ = _args[5];
lean_object* v_as_3850_ = _args[6];
lean_object* v_sz_3851_ = _args[7];
lean_object* v_i_3852_ = _args[8];
lean_object* v_b_3853_ = _args[9];
lean_object* v___y_3854_ = _args[10];
lean_object* v___y_3855_ = _args[11];
lean_object* v___y_3856_ = _args[12];
lean_object* v___y_3857_ = _args[13];
lean_object* v___y_3858_ = _args[14];
lean_object* v___y_3859_ = _args[15];
lean_object* v___y_3860_ = _args[16];
lean_object* v___y_3861_ = _args[17];
lean_object* v___y_3862_ = _args[18];
_start:
{
uint8_t v___y_94841__boxed_3863_; size_t v_sz_boxed_3864_; size_t v_i_boxed_3865_; lean_object* v_res_3866_; 
v___y_94841__boxed_3863_ = lean_unbox(v___y_3844_);
v_sz_boxed_3864_ = lean_unbox_usize(v_sz_3851_);
lean_dec(v_sz_3851_);
v_i_boxed_3865_ = lean_unbox_usize(v_i_3852_);
lean_dec(v_i_3852_);
v_res_3866_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7(v___y_94841__boxed_3863_, v_snd_3845_, v_rel_3846_, v_fst_3847_, v_a_3848_, v_a_3849_, v_as_3850_, v_sz_boxed_3864_, v_i_boxed_3865_, v_b_3853_, v___y_3854_, v___y_3855_, v___y_3856_, v___y_3857_, v___y_3858_, v___y_3859_, v___y_3860_, v___y_3861_);
lean_dec(v___y_3861_);
lean_dec_ref(v___y_3860_);
lean_dec(v___y_3859_);
lean_dec_ref(v___y_3858_);
lean_dec(v___y_3857_);
lean_dec_ref(v___y_3856_);
lean_dec(v___y_3855_);
lean_dec_ref(v___y_3854_);
lean_dec_ref(v_as_3850_);
return v_res_3866_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__1___closed__2(void){
_start:
{
lean_object* v___x_3869_; lean_object* v___x_3870_; 
v___x_3869_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__1___closed__1));
v___x_3870_ = l_Lean_stringToMessageData(v___x_3869_);
return v___x_3870_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__1(lean_object* v_rel_3871_, lean_object* v___x_3872_, uint8_t v___y_3873_, lean_object* v_snd_3874_, lean_object* v_fst_3875_, lean_object* v___x_3876_, lean_object* v___y_3877_, lean_object* v_____r_3878_, lean_object* v___y_3879_, lean_object* v___y_3880_, lean_object* v___y_3881_, lean_object* v___y_3882_, lean_object* v___y_3883_, lean_object* v___y_3884_, lean_object* v___y_3885_, lean_object* v___y_3886_){
_start:
{
lean_object* v_a_3889_; lean_object* v___x_3972_; 
v___x_3972_ = l_Lean_Elab_Tactic_getMainTag___redArg(v___y_3880_, v___y_3883_, v___y_3884_, v___y_3885_, v___y_3886_);
if (lean_obj_tag(v___x_3972_) == 0)
{
if (lean_obj_tag(v___y_3877_) == 0)
{
lean_object* v___x_3973_; 
lean_dec_ref_known(v___x_3972_, 1);
v___x_3973_ = lean_box(0);
v_a_3889_ = v___x_3973_;
goto v___jp_3888_;
}
else
{
lean_object* v_a_3974_; lean_object* v_val_3975_; lean_object* v___x_3977_; uint8_t v_isShared_3978_; uint8_t v_isSharedCheck_3993_; 
v_a_3974_ = lean_ctor_get(v___x_3972_, 0);
lean_inc(v_a_3974_);
lean_dec_ref_known(v___x_3972_, 1);
v_val_3975_ = lean_ctor_get(v___y_3877_, 0);
v_isSharedCheck_3993_ = !lean_is_exclusive(v___y_3877_);
if (v_isSharedCheck_3993_ == 0)
{
v___x_3977_ = v___y_3877_;
v_isShared_3978_ = v_isSharedCheck_3993_;
goto v_resetjp_3976_;
}
else
{
lean_inc(v_val_3975_);
lean_dec(v___y_3877_);
v___x_3977_ = lean_box(0);
v_isShared_3978_ = v_isSharedCheck_3993_;
goto v_resetjp_3976_;
}
v_resetjp_3976_:
{
lean_object* v___x_3979_; lean_object* v___x_3980_; 
v___x_3979_ = lean_box(0);
v___x_3980_ = l_Lean_Elab_Tactic_elabTermWithHoles(v_val_3975_, v___x_3979_, v_a_3974_, v___y_3873_, v___x_3979_, v___y_3879_, v___y_3880_, v___y_3881_, v___y_3882_, v___y_3883_, v___y_3884_, v___y_3885_, v___y_3886_);
if (lean_obj_tag(v___x_3980_) == 0)
{
lean_object* v_a_3981_; lean_object* v___x_3983_; 
v_a_3981_ = lean_ctor_get(v___x_3980_, 0);
lean_inc(v_a_3981_);
lean_dec_ref_known(v___x_3980_, 1);
if (v_isShared_3978_ == 0)
{
lean_ctor_set(v___x_3977_, 0, v_a_3981_);
v___x_3983_ = v___x_3977_;
goto v_reusejp_3982_;
}
else
{
lean_object* v_reuseFailAlloc_3984_; 
v_reuseFailAlloc_3984_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3984_, 0, v_a_3981_);
v___x_3983_ = v_reuseFailAlloc_3984_;
goto v_reusejp_3982_;
}
v_reusejp_3982_:
{
v_a_3889_ = v___x_3983_;
goto v___jp_3888_;
}
}
else
{
lean_object* v_a_3985_; lean_object* v___x_3987_; uint8_t v_isShared_3988_; uint8_t v_isSharedCheck_3992_; 
lean_del_object(v___x_3977_);
lean_dec_ref(v___x_3876_);
lean_dec_ref(v_fst_3875_);
lean_dec_ref(v_snd_3874_);
lean_dec_ref(v___x_3872_);
lean_dec_ref(v_rel_3871_);
v_a_3985_ = lean_ctor_get(v___x_3980_, 0);
v_isSharedCheck_3992_ = !lean_is_exclusive(v___x_3980_);
if (v_isSharedCheck_3992_ == 0)
{
v___x_3987_ = v___x_3980_;
v_isShared_3988_ = v_isSharedCheck_3992_;
goto v_resetjp_3986_;
}
else
{
lean_inc(v_a_3985_);
lean_dec(v___x_3980_);
v___x_3987_ = lean_box(0);
v_isShared_3988_ = v_isSharedCheck_3992_;
goto v_resetjp_3986_;
}
v_resetjp_3986_:
{
lean_object* v___x_3990_; 
if (v_isShared_3988_ == 0)
{
v___x_3990_ = v___x_3987_;
goto v_reusejp_3989_;
}
else
{
lean_object* v_reuseFailAlloc_3991_; 
v_reuseFailAlloc_3991_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3991_, 0, v_a_3985_);
v___x_3990_ = v_reuseFailAlloc_3991_;
goto v_reusejp_3989_;
}
v_reusejp_3989_:
{
return v___x_3990_;
}
}
}
}
}
}
else
{
lean_object* v_a_3994_; lean_object* v___x_3996_; uint8_t v_isShared_3997_; uint8_t v_isSharedCheck_4001_; 
lean_dec(v___y_3877_);
lean_dec_ref(v___x_3876_);
lean_dec_ref(v_fst_3875_);
lean_dec_ref(v_snd_3874_);
lean_dec_ref(v___x_3872_);
lean_dec_ref(v_rel_3871_);
v_a_3994_ = lean_ctor_get(v___x_3972_, 0);
v_isSharedCheck_4001_ = !lean_is_exclusive(v___x_3972_);
if (v_isSharedCheck_4001_ == 0)
{
v___x_3996_ = v___x_3972_;
v_isShared_3997_ = v_isSharedCheck_4001_;
goto v_resetjp_3995_;
}
else
{
lean_inc(v_a_3994_);
lean_dec(v___x_3972_);
v___x_3996_ = lean_box(0);
v_isShared_3997_ = v_isSharedCheck_4001_;
goto v_resetjp_3995_;
}
v_resetjp_3995_:
{
lean_object* v___x_3999_; 
if (v_isShared_3997_ == 0)
{
v___x_3999_ = v___x_3996_;
goto v_reusejp_3998_;
}
else
{
lean_object* v_reuseFailAlloc_4000_; 
v_reuseFailAlloc_4000_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4000_, 0, v_a_3994_);
v___x_3999_ = v_reuseFailAlloc_4000_;
goto v_reusejp_3998_;
}
v_reusejp_3998_:
{
return v___x_3999_;
}
}
}
v___jp_3888_:
{
lean_object* v___x_3890_; 
v___x_3890_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_3880_, v___y_3882_, v___y_3884_, v___y_3886_);
if (lean_obj_tag(v___x_3890_) == 0)
{
lean_object* v_a_3891_; lean_object* v___x_3892_; lean_object* v_env_3893_; lean_object* v___x_3894_; lean_object* v_ext_3895_; lean_object* v_toEnvExtension_3896_; lean_object* v_asyncMode_3897_; lean_object* v___x_3898_; lean_object* v___x_3899_; lean_object* v___x_3900_; 
v_a_3891_ = lean_ctor_get(v___x_3890_, 0);
lean_inc(v_a_3891_);
lean_dec_ref_known(v___x_3890_, 1);
v___x_3892_ = lean_st_ref_get(v___y_3886_);
v_env_3893_ = lean_ctor_get(v___x_3892_, 0);
lean_inc_ref(v_env_3893_);
lean_dec(v___x_3892_);
v___x_3894_ = lp_batteries_Batteries_Tactic_transExt;
v_ext_3895_ = lean_ctor_get(v___x_3894_, 1);
v_toEnvExtension_3896_ = lean_ctor_get(v_ext_3895_, 0);
v_asyncMode_3897_ = lean_ctor_get(v_toEnvExtension_3896_, 2);
v___x_3898_ = lean_obj_once(&lp_batteries_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__2___closed__0, &lp_batteries_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__2___closed__0_once, _init_lp_batteries_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__2___closed__0);
v___x_3899_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_3898_, v___x_3894_, v_env_3893_, v_asyncMode_3897_);
lean_inc_ref(v_rel_3871_);
v___x_3900_ = l_Lean_Meta_DiscrTree_getUnify___redArg(v___x_3899_, v_rel_3871_, v___y_3883_, v___y_3884_, v___y_3885_, v___y_3886_);
if (lean_obj_tag(v___x_3900_) == 0)
{
lean_object* v_a_3901_; lean_object* v___x_3902_; lean_object* v___x_3903_; lean_object* v___x_3904_; lean_object* v___x_3905_; lean_object* v___x_3906_; lean_object* v___x_3907_; lean_object* v___x_3908_; size_t v_sz_3909_; size_t v___x_3910_; lean_object* v___x_3911_; 
v_a_3901_ = lean_ctor_get(v___x_3900_, 0);
lean_inc(v_a_3901_);
lean_dec_ref_known(v___x_3900_, 1);
v___x_3902_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__1___closed__0));
lean_inc_ref(v___x_3872_);
v___x_3903_ = l_Lean_Name_mkStr2(v___x_3902_, v___x_3872_);
v___x_3904_ = lean_array_push(v_a_3901_, v___x_3903_);
v___x_3905_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__8_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_));
v___x_3906_ = l_Lean_Name_mkStr2(v___x_3905_, v___x_3872_);
v___x_3907_ = lean_array_push(v___x_3904_, v___x_3906_);
v___x_3908_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__2));
v_sz_3909_ = lean_array_size(v___x_3907_);
v___x_3910_ = ((size_t)0ULL);
v___x_3911_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7(v___y_3873_, v_snd_3874_, v_rel_3871_, v_fst_3875_, v_a_3889_, v_a_3891_, v___x_3907_, v_sz_3909_, v___x_3910_, v___x_3908_, v___y_3879_, v___y_3880_, v___y_3881_, v___y_3882_, v___y_3883_, v___y_3884_, v___y_3885_, v___y_3886_);
lean_dec_ref(v___x_3907_);
if (lean_obj_tag(v___x_3911_) == 0)
{
lean_object* v_a_3912_; lean_object* v___x_3914_; uint8_t v_isShared_3915_; uint8_t v_isSharedCheck_3947_; 
v_a_3912_ = lean_ctor_get(v___x_3911_, 0);
v_isSharedCheck_3947_ = !lean_is_exclusive(v___x_3911_);
if (v_isSharedCheck_3947_ == 0)
{
v___x_3914_ = v___x_3911_;
v_isShared_3915_ = v_isSharedCheck_3947_;
goto v_resetjp_3913_;
}
else
{
lean_inc(v_a_3912_);
lean_dec(v___x_3911_);
v___x_3914_ = lean_box(0);
v_isShared_3915_ = v_isSharedCheck_3947_;
goto v_resetjp_3913_;
}
v_resetjp_3913_:
{
lean_object* v_fst_3916_; lean_object* v___x_3918_; uint8_t v_isShared_3919_; uint8_t v_isSharedCheck_3945_; 
v_fst_3916_ = lean_ctor_get(v_a_3912_, 0);
v_isSharedCheck_3945_ = !lean_is_exclusive(v_a_3912_);
if (v_isSharedCheck_3945_ == 0)
{
lean_object* v_unused_3946_; 
v_unused_3946_ = lean_ctor_get(v_a_3912_, 1);
lean_dec(v_unused_3946_);
v___x_3918_ = v_a_3912_;
v_isShared_3919_ = v_isSharedCheck_3945_;
goto v_resetjp_3917_;
}
else
{
lean_inc(v_fst_3916_);
lean_dec(v_a_3912_);
v___x_3918_ = lean_box(0);
v_isShared_3919_ = v_isSharedCheck_3945_;
goto v_resetjp_3917_;
}
v_resetjp_3917_:
{
if (lean_obj_tag(v_fst_3916_) == 0)
{
lean_object* v___x_3920_; lean_object* v___x_3921_; lean_object* v___x_3923_; 
lean_del_object(v___x_3914_);
v___x_3920_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__1___closed__2, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__1___closed__2_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__1___closed__2);
v___x_3921_ = l_Lean_indentExpr(v___x_3876_);
if (v_isShared_3919_ == 0)
{
lean_ctor_set_tag(v___x_3918_, 7);
lean_ctor_set(v___x_3918_, 1, v___x_3921_);
lean_ctor_set(v___x_3918_, 0, v___x_3920_);
v___x_3923_ = v___x_3918_;
goto v_reusejp_3922_;
}
else
{
lean_object* v_reuseFailAlloc_3933_; 
v_reuseFailAlloc_3933_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3933_, 0, v___x_3920_);
lean_ctor_set(v_reuseFailAlloc_3933_, 1, v___x_3921_);
v___x_3923_ = v_reuseFailAlloc_3933_;
goto v_reusejp_3922_;
}
v_reusejp_3922_:
{
lean_object* v___x_3924_; lean_object* v_a_3925_; lean_object* v___x_3927_; uint8_t v_isShared_3928_; uint8_t v_isSharedCheck_3932_; 
v___x_3924_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__8___redArg(v___x_3923_, v___y_3883_, v___y_3884_, v___y_3885_, v___y_3886_);
v_a_3925_ = lean_ctor_get(v___x_3924_, 0);
v_isSharedCheck_3932_ = !lean_is_exclusive(v___x_3924_);
if (v_isSharedCheck_3932_ == 0)
{
v___x_3927_ = v___x_3924_;
v_isShared_3928_ = v_isSharedCheck_3932_;
goto v_resetjp_3926_;
}
else
{
lean_inc(v_a_3925_);
lean_dec(v___x_3924_);
v___x_3927_ = lean_box(0);
v_isShared_3928_ = v_isSharedCheck_3932_;
goto v_resetjp_3926_;
}
v_resetjp_3926_:
{
lean_object* v___x_3930_; 
if (v_isShared_3928_ == 0)
{
v___x_3930_ = v___x_3927_;
goto v_reusejp_3929_;
}
else
{
lean_object* v_reuseFailAlloc_3931_; 
v_reuseFailAlloc_3931_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3931_, 0, v_a_3925_);
v___x_3930_ = v_reuseFailAlloc_3931_;
goto v_reusejp_3929_;
}
v_reusejp_3929_:
{
return v___x_3930_;
}
}
}
}
else
{
lean_object* v_val_3934_; lean_object* v___x_3936_; uint8_t v_isShared_3937_; uint8_t v_isSharedCheck_3944_; 
lean_del_object(v___x_3918_);
lean_dec_ref(v___x_3876_);
v_val_3934_ = lean_ctor_get(v_fst_3916_, 0);
v_isSharedCheck_3944_ = !lean_is_exclusive(v_fst_3916_);
if (v_isSharedCheck_3944_ == 0)
{
v___x_3936_ = v_fst_3916_;
v_isShared_3937_ = v_isSharedCheck_3944_;
goto v_resetjp_3935_;
}
else
{
lean_inc(v_val_3934_);
lean_dec(v_fst_3916_);
v___x_3936_ = lean_box(0);
v_isShared_3937_ = v_isSharedCheck_3944_;
goto v_resetjp_3935_;
}
v_resetjp_3935_:
{
lean_object* v___x_3939_; 
if (v_isShared_3937_ == 0)
{
lean_ctor_set_tag(v___x_3936_, 0);
v___x_3939_ = v___x_3936_;
goto v_reusejp_3938_;
}
else
{
lean_object* v_reuseFailAlloc_3943_; 
v_reuseFailAlloc_3943_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3943_, 0, v_val_3934_);
v___x_3939_ = v_reuseFailAlloc_3943_;
goto v_reusejp_3938_;
}
v_reusejp_3938_:
{
lean_object* v___x_3941_; 
if (v_isShared_3915_ == 0)
{
lean_ctor_set(v___x_3914_, 0, v___x_3939_);
v___x_3941_ = v___x_3914_;
goto v_reusejp_3940_;
}
else
{
lean_object* v_reuseFailAlloc_3942_; 
v_reuseFailAlloc_3942_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3942_, 0, v___x_3939_);
v___x_3941_ = v_reuseFailAlloc_3942_;
goto v_reusejp_3940_;
}
v_reusejp_3940_:
{
return v___x_3941_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_3948_; lean_object* v___x_3950_; uint8_t v_isShared_3951_; uint8_t v_isSharedCheck_3955_; 
lean_dec_ref(v___x_3876_);
v_a_3948_ = lean_ctor_get(v___x_3911_, 0);
v_isSharedCheck_3955_ = !lean_is_exclusive(v___x_3911_);
if (v_isSharedCheck_3955_ == 0)
{
v___x_3950_ = v___x_3911_;
v_isShared_3951_ = v_isSharedCheck_3955_;
goto v_resetjp_3949_;
}
else
{
lean_inc(v_a_3948_);
lean_dec(v___x_3911_);
v___x_3950_ = lean_box(0);
v_isShared_3951_ = v_isSharedCheck_3955_;
goto v_resetjp_3949_;
}
v_resetjp_3949_:
{
lean_object* v___x_3953_; 
if (v_isShared_3951_ == 0)
{
v___x_3953_ = v___x_3950_;
goto v_reusejp_3952_;
}
else
{
lean_object* v_reuseFailAlloc_3954_; 
v_reuseFailAlloc_3954_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3954_, 0, v_a_3948_);
v___x_3953_ = v_reuseFailAlloc_3954_;
goto v_reusejp_3952_;
}
v_reusejp_3952_:
{
return v___x_3953_;
}
}
}
}
else
{
lean_object* v_a_3956_; lean_object* v___x_3958_; uint8_t v_isShared_3959_; uint8_t v_isSharedCheck_3963_; 
lean_dec(v_a_3891_);
lean_dec(v_a_3889_);
lean_dec_ref(v___x_3876_);
lean_dec_ref(v_fst_3875_);
lean_dec_ref(v_snd_3874_);
lean_dec_ref(v___x_3872_);
lean_dec_ref(v_rel_3871_);
v_a_3956_ = lean_ctor_get(v___x_3900_, 0);
v_isSharedCheck_3963_ = !lean_is_exclusive(v___x_3900_);
if (v_isSharedCheck_3963_ == 0)
{
v___x_3958_ = v___x_3900_;
v_isShared_3959_ = v_isSharedCheck_3963_;
goto v_resetjp_3957_;
}
else
{
lean_inc(v_a_3956_);
lean_dec(v___x_3900_);
v___x_3958_ = lean_box(0);
v_isShared_3959_ = v_isSharedCheck_3963_;
goto v_resetjp_3957_;
}
v_resetjp_3957_:
{
lean_object* v___x_3961_; 
if (v_isShared_3959_ == 0)
{
v___x_3961_ = v___x_3958_;
goto v_reusejp_3960_;
}
else
{
lean_object* v_reuseFailAlloc_3962_; 
v_reuseFailAlloc_3962_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3962_, 0, v_a_3956_);
v___x_3961_ = v_reuseFailAlloc_3962_;
goto v_reusejp_3960_;
}
v_reusejp_3960_:
{
return v___x_3961_;
}
}
}
}
else
{
lean_object* v_a_3964_; lean_object* v___x_3966_; uint8_t v_isShared_3967_; uint8_t v_isSharedCheck_3971_; 
lean_dec(v_a_3889_);
lean_dec_ref(v___x_3876_);
lean_dec_ref(v_fst_3875_);
lean_dec_ref(v_snd_3874_);
lean_dec_ref(v___x_3872_);
lean_dec_ref(v_rel_3871_);
v_a_3964_ = lean_ctor_get(v___x_3890_, 0);
v_isSharedCheck_3971_ = !lean_is_exclusive(v___x_3890_);
if (v_isSharedCheck_3971_ == 0)
{
v___x_3966_ = v___x_3890_;
v_isShared_3967_ = v_isSharedCheck_3971_;
goto v_resetjp_3965_;
}
else
{
lean_inc(v_a_3964_);
lean_dec(v___x_3890_);
v___x_3966_ = lean_box(0);
v_isShared_3967_ = v_isSharedCheck_3971_;
goto v_resetjp_3965_;
}
v_resetjp_3965_:
{
lean_object* v___x_3969_; 
if (v_isShared_3967_ == 0)
{
v___x_3969_ = v___x_3966_;
goto v_reusejp_3968_;
}
else
{
lean_object* v_reuseFailAlloc_3970_; 
v_reuseFailAlloc_3970_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3970_, 0, v_a_3964_);
v___x_3969_ = v_reuseFailAlloc_3970_;
goto v_reusejp_3968_;
}
v_reusejp_3968_:
{
return v___x_3969_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__1___boxed(lean_object** _args){
lean_object* v_rel_4002_ = _args[0];
lean_object* v___x_4003_ = _args[1];
lean_object* v___y_4004_ = _args[2];
lean_object* v_snd_4005_ = _args[3];
lean_object* v_fst_4006_ = _args[4];
lean_object* v___x_4007_ = _args[5];
lean_object* v___y_4008_ = _args[6];
lean_object* v_____r_4009_ = _args[7];
lean_object* v___y_4010_ = _args[8];
lean_object* v___y_4011_ = _args[9];
lean_object* v___y_4012_ = _args[10];
lean_object* v___y_4013_ = _args[11];
lean_object* v___y_4014_ = _args[12];
lean_object* v___y_4015_ = _args[13];
lean_object* v___y_4016_ = _args[14];
lean_object* v___y_4017_ = _args[15];
lean_object* v___y_4018_ = _args[16];
_start:
{
uint8_t v___y_95085__boxed_4019_; lean_object* v_res_4020_; 
v___y_95085__boxed_4019_ = lean_unbox(v___y_4004_);
v_res_4020_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__1(v_rel_4002_, v___x_4003_, v___y_95085__boxed_4019_, v_snd_4005_, v_fst_4006_, v___x_4007_, v___y_4008_, v_____r_4009_, v___y_4010_, v___y_4011_, v___y_4012_, v___y_4013_, v___y_4014_, v___y_4015_, v___y_4016_, v___y_4017_);
lean_dec(v___y_4017_);
lean_dec_ref(v___y_4016_);
lean_dec(v___y_4015_);
lean_dec_ref(v___y_4014_);
lean_dec(v___y_4013_);
lean_dec_ref(v___y_4012_);
lean_dec(v___y_4011_);
lean_dec_ref(v___y_4010_);
return v_res_4020_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__9___lam__1(lean_object* v_a_4021_, lean_object* v___f_4022_, lean_object* v___x_4023_, lean_object* v_fst_4024_, lean_object* v_rel_4025_, lean_object* v_snd_4026_, lean_object* v_a_4027_, lean_object* v_a_4028_, lean_object* v___y_4029_, lean_object* v___y_4030_, lean_object* v___y_4031_, lean_object* v___y_4032_, lean_object* v___y_4033_, lean_object* v___y_4034_, lean_object* v___y_4035_, lean_object* v___y_4036_){
_start:
{
lean_object* v___y_4039_; lean_object* v___y_4040_; lean_object* v___x_4051_; 
v___x_4051_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_4030_, v___y_4033_, v___y_4034_, v___y_4035_, v___y_4036_);
if (lean_obj_tag(v___x_4051_) == 0)
{
lean_object* v_a_4052_; lean_object* v___x_4053_; 
v_a_4052_ = lean_ctor_get(v___x_4051_, 0);
lean_inc(v_a_4052_);
lean_dec_ref_known(v___x_4051_, 1);
lean_inc(v_a_4021_);
v___x_4053_ = lp_batteries_Lean_mkConstWithLevelParams___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__2(v_a_4021_, v___y_4033_, v___y_4034_, v___y_4035_, v___y_4036_);
if (lean_obj_tag(v___x_4053_) == 0)
{
lean_object* v_a_4054_; lean_object* v___x_4055_; 
v_a_4054_ = lean_ctor_get(v___x_4053_, 0);
lean_inc(v_a_4054_);
lean_dec_ref_known(v___x_4053_, 1);
lean_inc(v___y_4036_);
lean_inc_ref(v___y_4035_);
lean_inc(v___y_4034_);
lean_inc_ref(v___y_4033_);
v___x_4055_ = lean_infer_type(v_a_4054_, v___y_4033_, v___y_4034_, v___y_4035_, v___y_4036_);
if (lean_obj_tag(v___x_4055_) == 0)
{
lean_object* v_a_4056_; lean_object* v_keyedConfig_4057_; uint8_t v_trackZetaDelta_4058_; lean_object* v_zetaDeltaSet_4059_; lean_object* v_lctx_4060_; lean_object* v_localInstances_4061_; lean_object* v_defEqCtx_x3f_4062_; lean_object* v_synthPendingDepth_4063_; lean_object* v_customCanUnfoldPredicate_x3f_4064_; uint8_t v_univApprox_4065_; uint8_t v_inTypeClassResolution_4066_; uint8_t v_cacheInferType_4067_; uint8_t v___x_4068_; uint8_t v___x_4069_; lean_object* v___x_4070_; lean_object* v___x_4071_; lean_object* v___x_4072_; 
v_a_4056_ = lean_ctor_get(v___x_4055_, 0);
lean_inc(v_a_4056_);
lean_dec_ref_known(v___x_4055_, 1);
v_keyedConfig_4057_ = lean_ctor_get(v___y_4033_, 0);
v_trackZetaDelta_4058_ = lean_ctor_get_uint8(v___y_4033_, sizeof(void*)*7);
v_zetaDeltaSet_4059_ = lean_ctor_get(v___y_4033_, 1);
v_lctx_4060_ = lean_ctor_get(v___y_4033_, 2);
v_localInstances_4061_ = lean_ctor_get(v___y_4033_, 3);
v_defEqCtx_x3f_4062_ = lean_ctor_get(v___y_4033_, 4);
v_synthPendingDepth_4063_ = lean_ctor_get(v___y_4033_, 5);
v_customCanUnfoldPredicate_x3f_4064_ = lean_ctor_get(v___y_4033_, 6);
v_univApprox_4065_ = lean_ctor_get_uint8(v___y_4033_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_4066_ = lean_ctor_get_uint8(v___y_4033_, sizeof(void*)*7 + 2);
v_cacheInferType_4067_ = lean_ctor_get_uint8(v___y_4033_, sizeof(void*)*7 + 3);
v___x_4068_ = 0;
v___x_4069_ = 2;
lean_inc_ref(v_keyedConfig_4057_);
v___x_4070_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_4069_, v_keyedConfig_4057_);
lean_inc(v_customCanUnfoldPredicate_x3f_4064_);
lean_inc(v_synthPendingDepth_4063_);
lean_inc(v_defEqCtx_x3f_4062_);
lean_inc_ref(v_localInstances_4061_);
lean_inc_ref(v_lctx_4060_);
lean_inc(v_zetaDeltaSet_4059_);
v___x_4071_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_4071_, 0, v___x_4070_);
lean_ctor_set(v___x_4071_, 1, v_zetaDeltaSet_4059_);
lean_ctor_set(v___x_4071_, 2, v_lctx_4060_);
lean_ctor_set(v___x_4071_, 3, v_localInstances_4061_);
lean_ctor_set(v___x_4071_, 4, v_defEqCtx_x3f_4062_);
lean_ctor_set(v___x_4071_, 5, v_synthPendingDepth_4063_);
lean_ctor_set(v___x_4071_, 6, v_customCanUnfoldPredicate_x3f_4064_);
lean_ctor_set_uint8(v___x_4071_, sizeof(void*)*7, v_trackZetaDelta_4058_);
lean_ctor_set_uint8(v___x_4071_, sizeof(void*)*7 + 1, v_univApprox_4065_);
lean_ctor_set_uint8(v___x_4071_, sizeof(void*)*7 + 2, v_inTypeClassResolution_4066_);
lean_ctor_set_uint8(v___x_4071_, sizeof(void*)*7 + 3, v_cacheInferType_4067_);
v___x_4072_ = lp_batteries_Lean_Meta_forallTelescopeReducing___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__3___redArg(v_a_4056_, v___f_4022_, v___x_4068_, v___x_4068_, v___x_4071_, v___y_4034_, v___y_4035_, v___y_4036_);
lean_dec_ref_known(v___x_4071_, 7);
if (lean_obj_tag(v___x_4072_) == 0)
{
lean_object* v_a_4073_; lean_object* v_a_4075_; 
v_a_4073_ = lean_ctor_get(v___x_4072_, 0);
lean_inc(v_a_4073_);
lean_dec_ref_known(v___x_4072_, 1);
if (lean_obj_tag(v_a_4027_) == 0)
{
lean_object* v___x_4154_; uint8_t v___x_4155_; lean_object* v___x_4156_; lean_object* v___x_4157_; 
v___x_4154_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4154_, 0, v_a_4028_);
v___x_4155_ = 0;
v___x_4156_ = lean_box(0);
v___x_4157_ = l_Lean_Meta_mkFreshExprMVar(v___x_4154_, v___x_4155_, v___x_4156_, v___y_4033_, v___y_4034_, v___y_4035_, v___y_4036_);
if (lean_obj_tag(v___x_4157_) == 0)
{
lean_object* v_a_4158_; 
v_a_4158_ = lean_ctor_get(v___x_4157_, 0);
lean_inc(v_a_4158_);
lean_dec_ref_known(v___x_4157_, 1);
v_a_4075_ = v_a_4158_;
goto v___jp_4074_;
}
else
{
lean_object* v_a_4159_; lean_object* v___x_4161_; uint8_t v_isShared_4162_; uint8_t v_isSharedCheck_4166_; 
lean_dec(v_a_4073_);
lean_dec(v_a_4052_);
lean_dec(v___y_4036_);
lean_dec_ref(v___y_4035_);
lean_dec(v___y_4034_);
lean_dec_ref(v___y_4033_);
lean_dec_ref(v_snd_4026_);
lean_dec_ref(v_rel_4025_);
lean_dec_ref(v_fst_4024_);
lean_dec(v_a_4021_);
v_a_4159_ = lean_ctor_get(v___x_4157_, 0);
v_isSharedCheck_4166_ = !lean_is_exclusive(v___x_4157_);
if (v_isSharedCheck_4166_ == 0)
{
v___x_4161_ = v___x_4157_;
v_isShared_4162_ = v_isSharedCheck_4166_;
goto v_resetjp_4160_;
}
else
{
lean_inc(v_a_4159_);
lean_dec(v___x_4157_);
v___x_4161_ = lean_box(0);
v_isShared_4162_ = v_isSharedCheck_4166_;
goto v_resetjp_4160_;
}
v_resetjp_4160_:
{
lean_object* v___x_4164_; 
if (v_isShared_4162_ == 0)
{
v___x_4164_ = v___x_4161_;
goto v_reusejp_4163_;
}
else
{
lean_object* v_reuseFailAlloc_4165_; 
v_reuseFailAlloc_4165_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4165_, 0, v_a_4159_);
v___x_4164_ = v_reuseFailAlloc_4165_;
goto v_reusejp_4163_;
}
v_reusejp_4163_:
{
return v___x_4164_;
}
}
}
}
else
{
lean_object* v_val_4167_; lean_object* v_fst_4168_; 
lean_dec_ref(v_a_4028_);
v_val_4167_ = lean_ctor_get(v_a_4027_, 0);
v_fst_4168_ = lean_ctor_get(v_val_4167_, 0);
lean_inc(v_fst_4168_);
v_a_4075_ = v_fst_4168_;
goto v___jp_4074_;
}
v___jp_4074_:
{
lean_object* v___x_4076_; lean_object* v___x_4077_; lean_object* v___x_4078_; lean_object* v___x_4079_; lean_object* v___x_4080_; 
v___x_4076_ = lean_unsigned_to_nat(2u);
v___x_4077_ = lean_mk_empty_array_with_capacity(v___x_4076_);
lean_inc_ref(v___x_4077_);
v___x_4078_ = lean_array_push(v___x_4077_, v_fst_4024_);
lean_inc_ref(v_a_4075_);
v___x_4079_ = lean_array_push(v___x_4078_, v_a_4075_);
lean_inc_ref(v_rel_4025_);
v___x_4080_ = l_Lean_Meta_mkAppM_x27(v_rel_4025_, v___x_4079_, v___y_4033_, v___y_4034_, v___y_4035_, v___y_4036_);
if (lean_obj_tag(v___x_4080_) == 0)
{
lean_object* v_a_4081_; lean_object* v___x_4082_; uint8_t v___x_4083_; lean_object* v___x_4084_; lean_object* v___x_4085_; 
v_a_4081_ = lean_ctor_get(v___x_4080_, 0);
lean_inc(v_a_4081_);
lean_dec_ref_known(v___x_4080_, 1);
v___x_4082_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4082_, 0, v_a_4081_);
v___x_4083_ = 1;
v___x_4084_ = lean_box(0);
v___x_4085_ = l_Lean_Meta_mkFreshExprMVar(v___x_4082_, v___x_4083_, v___x_4084_, v___y_4033_, v___y_4034_, v___y_4035_, v___y_4036_);
if (lean_obj_tag(v___x_4085_) == 0)
{
lean_object* v_a_4086_; lean_object* v___x_4087_; lean_object* v___x_4088_; lean_object* v___x_4089_; 
v_a_4086_ = lean_ctor_get(v___x_4085_, 0);
lean_inc(v_a_4086_);
lean_dec_ref_known(v___x_4085_, 1);
lean_inc_ref(v_a_4075_);
lean_inc_ref(v___x_4077_);
v___x_4087_ = lean_array_push(v___x_4077_, v_a_4075_);
v___x_4088_ = lean_array_push(v___x_4087_, v_snd_4026_);
v___x_4089_ = l_Lean_Meta_mkAppM_x27(v_rel_4025_, v___x_4088_, v___y_4033_, v___y_4034_, v___y_4035_, v___y_4036_);
if (lean_obj_tag(v___x_4089_) == 0)
{
lean_object* v_a_4090_; lean_object* v___x_4091_; lean_object* v___x_4092_; 
v_a_4090_ = lean_ctor_get(v___x_4089_, 0);
lean_inc(v_a_4090_);
lean_dec_ref_known(v___x_4089_, 1);
v___x_4091_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4091_, 0, v_a_4090_);
v___x_4092_ = l_Lean_Meta_mkFreshExprMVar(v___x_4091_, v___x_4083_, v___x_4084_, v___y_4033_, v___y_4034_, v___y_4035_, v___y_4036_);
if (lean_obj_tag(v___x_4092_) == 0)
{
lean_object* v_a_4093_; lean_object* v___x_4094_; lean_object* v___x_4095_; lean_object* v___x_4096_; lean_object* v___x_4097_; lean_object* v___x_4098_; lean_object* v___x_4099_; lean_object* v___x_4100_; lean_object* v___x_4101_; lean_object* v___x_4102_; 
v_a_4093_ = lean_ctor_get(v___x_4092_, 0);
lean_inc_n(v_a_4093_, 2);
lean_dec_ref_known(v___x_4092_, 1);
v___x_4094_ = lean_nat_sub(v_a_4073_, v___x_4076_);
lean_dec(v_a_4073_);
v___x_4095_ = lean_box(0);
v___x_4096_ = lean_mk_array(v___x_4094_, v___x_4095_);
lean_inc(v_a_4086_);
v___x_4097_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4097_, 0, v_a_4086_);
v___x_4098_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4098_, 0, v_a_4093_);
v___x_4099_ = lean_array_push(v___x_4077_, v___x_4097_);
v___x_4100_ = lean_array_push(v___x_4099_, v___x_4098_);
v___x_4101_ = l_Array_append___redArg(v___x_4096_, v___x_4100_);
lean_dec_ref(v___x_4100_);
v___x_4102_ = l_Lean_Meta_mkAppOptM(v_a_4021_, v___x_4101_, v___y_4033_, v___y_4034_, v___y_4035_, v___y_4036_);
if (lean_obj_tag(v___x_4102_) == 0)
{
lean_object* v_a_4103_; lean_object* v___x_4104_; 
v_a_4103_ = lean_ctor_get(v___x_4102_, 0);
lean_inc(v_a_4103_);
lean_dec_ref_known(v___x_4102_, 1);
v___x_4104_ = lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4___redArg(v_a_4052_, v_a_4103_, v___y_4034_);
if (lean_obj_tag(v___x_4104_) == 0)
{
lean_object* v___x_4105_; lean_object* v___x_4106_; lean_object* v___x_4107_; lean_object* v___x_4108_; lean_object* v___x_4109_; 
lean_dec_ref_known(v___x_4104_, 1);
v___x_4105_ = l_Lean_Expr_mvarId_x21(v_a_4086_);
lean_dec(v_a_4086_);
v___x_4106_ = l_Lean_Expr_mvarId_x21(v_a_4093_);
lean_dec(v_a_4093_);
v___x_4107_ = lean_box(0);
v___x_4108_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4108_, 0, v___x_4106_);
lean_ctor_set(v___x_4108_, 1, v___x_4107_);
v___x_4109_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4109_, 0, v___x_4105_);
lean_ctor_set(v___x_4109_, 1, v___x_4108_);
if (lean_obj_tag(v_a_4027_) == 1)
{
lean_object* v_val_4110_; lean_object* v_snd_4111_; 
lean_dec_ref(v_a_4075_);
v_val_4110_ = lean_ctor_get(v_a_4027_, 0);
lean_inc(v_val_4110_);
lean_dec_ref_known(v_a_4027_, 1);
v_snd_4111_ = lean_ctor_get(v_val_4110_, 1);
lean_inc(v_snd_4111_);
lean_dec(v_val_4110_);
v___y_4039_ = v___x_4109_;
v___y_4040_ = v_snd_4111_;
goto v___jp_4038_;
}
else
{
lean_object* v___x_4112_; lean_object* v___x_4113_; 
lean_dec(v_a_4027_);
v___x_4112_ = l_Lean_Expr_mvarId_x21(v_a_4075_);
lean_dec_ref(v_a_4075_);
v___x_4113_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4113_, 0, v___x_4112_);
lean_ctor_set(v___x_4113_, 1, v___x_4107_);
v___y_4039_ = v___x_4109_;
v___y_4040_ = v___x_4113_;
goto v___jp_4038_;
}
}
else
{
lean_dec(v_a_4093_);
lean_dec(v_a_4086_);
lean_dec_ref(v_a_4075_);
lean_dec(v___y_4036_);
lean_dec_ref(v___y_4035_);
lean_dec(v___y_4034_);
lean_dec_ref(v___y_4033_);
lean_dec(v_a_4027_);
return v___x_4104_;
}
}
else
{
lean_object* v_a_4114_; lean_object* v___x_4116_; uint8_t v_isShared_4117_; uint8_t v_isSharedCheck_4121_; 
lean_dec(v_a_4093_);
lean_dec(v_a_4086_);
lean_dec_ref(v_a_4075_);
lean_dec(v_a_4052_);
lean_dec(v___y_4036_);
lean_dec_ref(v___y_4035_);
lean_dec(v___y_4034_);
lean_dec_ref(v___y_4033_);
lean_dec(v_a_4027_);
v_a_4114_ = lean_ctor_get(v___x_4102_, 0);
v_isSharedCheck_4121_ = !lean_is_exclusive(v___x_4102_);
if (v_isSharedCheck_4121_ == 0)
{
v___x_4116_ = v___x_4102_;
v_isShared_4117_ = v_isSharedCheck_4121_;
goto v_resetjp_4115_;
}
else
{
lean_inc(v_a_4114_);
lean_dec(v___x_4102_);
v___x_4116_ = lean_box(0);
v_isShared_4117_ = v_isSharedCheck_4121_;
goto v_resetjp_4115_;
}
v_resetjp_4115_:
{
lean_object* v___x_4119_; 
if (v_isShared_4117_ == 0)
{
v___x_4119_ = v___x_4116_;
goto v_reusejp_4118_;
}
else
{
lean_object* v_reuseFailAlloc_4120_; 
v_reuseFailAlloc_4120_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4120_, 0, v_a_4114_);
v___x_4119_ = v_reuseFailAlloc_4120_;
goto v_reusejp_4118_;
}
v_reusejp_4118_:
{
return v___x_4119_;
}
}
}
}
else
{
lean_object* v_a_4122_; lean_object* v___x_4124_; uint8_t v_isShared_4125_; uint8_t v_isSharedCheck_4129_; 
lean_dec(v_a_4086_);
lean_dec_ref(v___x_4077_);
lean_dec_ref(v_a_4075_);
lean_dec(v_a_4073_);
lean_dec(v_a_4052_);
lean_dec(v___y_4036_);
lean_dec_ref(v___y_4035_);
lean_dec(v___y_4034_);
lean_dec_ref(v___y_4033_);
lean_dec(v_a_4027_);
lean_dec(v_a_4021_);
v_a_4122_ = lean_ctor_get(v___x_4092_, 0);
v_isSharedCheck_4129_ = !lean_is_exclusive(v___x_4092_);
if (v_isSharedCheck_4129_ == 0)
{
v___x_4124_ = v___x_4092_;
v_isShared_4125_ = v_isSharedCheck_4129_;
goto v_resetjp_4123_;
}
else
{
lean_inc(v_a_4122_);
lean_dec(v___x_4092_);
v___x_4124_ = lean_box(0);
v_isShared_4125_ = v_isSharedCheck_4129_;
goto v_resetjp_4123_;
}
v_resetjp_4123_:
{
lean_object* v___x_4127_; 
if (v_isShared_4125_ == 0)
{
v___x_4127_ = v___x_4124_;
goto v_reusejp_4126_;
}
else
{
lean_object* v_reuseFailAlloc_4128_; 
v_reuseFailAlloc_4128_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4128_, 0, v_a_4122_);
v___x_4127_ = v_reuseFailAlloc_4128_;
goto v_reusejp_4126_;
}
v_reusejp_4126_:
{
return v___x_4127_;
}
}
}
}
else
{
lean_object* v_a_4130_; lean_object* v___x_4132_; uint8_t v_isShared_4133_; uint8_t v_isSharedCheck_4137_; 
lean_dec(v_a_4086_);
lean_dec_ref(v___x_4077_);
lean_dec_ref(v_a_4075_);
lean_dec(v_a_4073_);
lean_dec(v_a_4052_);
lean_dec(v___y_4036_);
lean_dec_ref(v___y_4035_);
lean_dec(v___y_4034_);
lean_dec_ref(v___y_4033_);
lean_dec(v_a_4027_);
lean_dec(v_a_4021_);
v_a_4130_ = lean_ctor_get(v___x_4089_, 0);
v_isSharedCheck_4137_ = !lean_is_exclusive(v___x_4089_);
if (v_isSharedCheck_4137_ == 0)
{
v___x_4132_ = v___x_4089_;
v_isShared_4133_ = v_isSharedCheck_4137_;
goto v_resetjp_4131_;
}
else
{
lean_inc(v_a_4130_);
lean_dec(v___x_4089_);
v___x_4132_ = lean_box(0);
v_isShared_4133_ = v_isSharedCheck_4137_;
goto v_resetjp_4131_;
}
v_resetjp_4131_:
{
lean_object* v___x_4135_; 
if (v_isShared_4133_ == 0)
{
v___x_4135_ = v___x_4132_;
goto v_reusejp_4134_;
}
else
{
lean_object* v_reuseFailAlloc_4136_; 
v_reuseFailAlloc_4136_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4136_, 0, v_a_4130_);
v___x_4135_ = v_reuseFailAlloc_4136_;
goto v_reusejp_4134_;
}
v_reusejp_4134_:
{
return v___x_4135_;
}
}
}
}
else
{
lean_object* v_a_4138_; lean_object* v___x_4140_; uint8_t v_isShared_4141_; uint8_t v_isSharedCheck_4145_; 
lean_dec_ref(v___x_4077_);
lean_dec_ref(v_a_4075_);
lean_dec(v_a_4073_);
lean_dec(v_a_4052_);
lean_dec(v___y_4036_);
lean_dec_ref(v___y_4035_);
lean_dec(v___y_4034_);
lean_dec_ref(v___y_4033_);
lean_dec(v_a_4027_);
lean_dec_ref(v_snd_4026_);
lean_dec_ref(v_rel_4025_);
lean_dec(v_a_4021_);
v_a_4138_ = lean_ctor_get(v___x_4085_, 0);
v_isSharedCheck_4145_ = !lean_is_exclusive(v___x_4085_);
if (v_isSharedCheck_4145_ == 0)
{
v___x_4140_ = v___x_4085_;
v_isShared_4141_ = v_isSharedCheck_4145_;
goto v_resetjp_4139_;
}
else
{
lean_inc(v_a_4138_);
lean_dec(v___x_4085_);
v___x_4140_ = lean_box(0);
v_isShared_4141_ = v_isSharedCheck_4145_;
goto v_resetjp_4139_;
}
v_resetjp_4139_:
{
lean_object* v___x_4143_; 
if (v_isShared_4141_ == 0)
{
v___x_4143_ = v___x_4140_;
goto v_reusejp_4142_;
}
else
{
lean_object* v_reuseFailAlloc_4144_; 
v_reuseFailAlloc_4144_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4144_, 0, v_a_4138_);
v___x_4143_ = v_reuseFailAlloc_4144_;
goto v_reusejp_4142_;
}
v_reusejp_4142_:
{
return v___x_4143_;
}
}
}
}
else
{
lean_object* v_a_4146_; lean_object* v___x_4148_; uint8_t v_isShared_4149_; uint8_t v_isSharedCheck_4153_; 
lean_dec_ref(v___x_4077_);
lean_dec_ref(v_a_4075_);
lean_dec(v_a_4073_);
lean_dec(v_a_4052_);
lean_dec(v___y_4036_);
lean_dec_ref(v___y_4035_);
lean_dec(v___y_4034_);
lean_dec_ref(v___y_4033_);
lean_dec(v_a_4027_);
lean_dec_ref(v_snd_4026_);
lean_dec_ref(v_rel_4025_);
lean_dec(v_a_4021_);
v_a_4146_ = lean_ctor_get(v___x_4080_, 0);
v_isSharedCheck_4153_ = !lean_is_exclusive(v___x_4080_);
if (v_isSharedCheck_4153_ == 0)
{
v___x_4148_ = v___x_4080_;
v_isShared_4149_ = v_isSharedCheck_4153_;
goto v_resetjp_4147_;
}
else
{
lean_inc(v_a_4146_);
lean_dec(v___x_4080_);
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
}
else
{
lean_object* v_a_4169_; lean_object* v___x_4171_; uint8_t v_isShared_4172_; uint8_t v_isSharedCheck_4176_; 
lean_dec(v_a_4052_);
lean_dec(v___y_4036_);
lean_dec_ref(v___y_4035_);
lean_dec(v___y_4034_);
lean_dec_ref(v___y_4033_);
lean_dec_ref(v_a_4028_);
lean_dec(v_a_4027_);
lean_dec_ref(v_snd_4026_);
lean_dec_ref(v_rel_4025_);
lean_dec_ref(v_fst_4024_);
lean_dec(v_a_4021_);
v_a_4169_ = lean_ctor_get(v___x_4072_, 0);
v_isSharedCheck_4176_ = !lean_is_exclusive(v___x_4072_);
if (v_isSharedCheck_4176_ == 0)
{
v___x_4171_ = v___x_4072_;
v_isShared_4172_ = v_isSharedCheck_4176_;
goto v_resetjp_4170_;
}
else
{
lean_inc(v_a_4169_);
lean_dec(v___x_4072_);
v___x_4171_ = lean_box(0);
v_isShared_4172_ = v_isSharedCheck_4176_;
goto v_resetjp_4170_;
}
v_resetjp_4170_:
{
lean_object* v___x_4174_; 
if (v_isShared_4172_ == 0)
{
v___x_4174_ = v___x_4171_;
goto v_reusejp_4173_;
}
else
{
lean_object* v_reuseFailAlloc_4175_; 
v_reuseFailAlloc_4175_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4175_, 0, v_a_4169_);
v___x_4174_ = v_reuseFailAlloc_4175_;
goto v_reusejp_4173_;
}
v_reusejp_4173_:
{
return v___x_4174_;
}
}
}
}
else
{
lean_object* v_a_4177_; lean_object* v___x_4179_; uint8_t v_isShared_4180_; uint8_t v_isSharedCheck_4184_; 
lean_dec(v_a_4052_);
lean_dec(v___y_4036_);
lean_dec_ref(v___y_4035_);
lean_dec(v___y_4034_);
lean_dec_ref(v___y_4033_);
lean_dec_ref(v_a_4028_);
lean_dec(v_a_4027_);
lean_dec_ref(v_snd_4026_);
lean_dec_ref(v_rel_4025_);
lean_dec_ref(v_fst_4024_);
lean_dec_ref(v___f_4022_);
lean_dec(v_a_4021_);
v_a_4177_ = lean_ctor_get(v___x_4055_, 0);
v_isSharedCheck_4184_ = !lean_is_exclusive(v___x_4055_);
if (v_isSharedCheck_4184_ == 0)
{
v___x_4179_ = v___x_4055_;
v_isShared_4180_ = v_isSharedCheck_4184_;
goto v_resetjp_4178_;
}
else
{
lean_inc(v_a_4177_);
lean_dec(v___x_4055_);
v___x_4179_ = lean_box(0);
v_isShared_4180_ = v_isSharedCheck_4184_;
goto v_resetjp_4178_;
}
v_resetjp_4178_:
{
lean_object* v___x_4182_; 
if (v_isShared_4180_ == 0)
{
v___x_4182_ = v___x_4179_;
goto v_reusejp_4181_;
}
else
{
lean_object* v_reuseFailAlloc_4183_; 
v_reuseFailAlloc_4183_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4183_, 0, v_a_4177_);
v___x_4182_ = v_reuseFailAlloc_4183_;
goto v_reusejp_4181_;
}
v_reusejp_4181_:
{
return v___x_4182_;
}
}
}
}
else
{
lean_object* v_a_4185_; lean_object* v___x_4187_; uint8_t v_isShared_4188_; uint8_t v_isSharedCheck_4192_; 
lean_dec(v_a_4052_);
lean_dec(v___y_4036_);
lean_dec_ref(v___y_4035_);
lean_dec(v___y_4034_);
lean_dec_ref(v___y_4033_);
lean_dec_ref(v_a_4028_);
lean_dec(v_a_4027_);
lean_dec_ref(v_snd_4026_);
lean_dec_ref(v_rel_4025_);
lean_dec_ref(v_fst_4024_);
lean_dec_ref(v___f_4022_);
lean_dec(v_a_4021_);
v_a_4185_ = lean_ctor_get(v___x_4053_, 0);
v_isSharedCheck_4192_ = !lean_is_exclusive(v___x_4053_);
if (v_isSharedCheck_4192_ == 0)
{
v___x_4187_ = v___x_4053_;
v_isShared_4188_ = v_isSharedCheck_4192_;
goto v_resetjp_4186_;
}
else
{
lean_inc(v_a_4185_);
lean_dec(v___x_4053_);
v___x_4187_ = lean_box(0);
v_isShared_4188_ = v_isSharedCheck_4192_;
goto v_resetjp_4186_;
}
v_resetjp_4186_:
{
lean_object* v___x_4190_; 
if (v_isShared_4188_ == 0)
{
v___x_4190_ = v___x_4187_;
goto v_reusejp_4189_;
}
else
{
lean_object* v_reuseFailAlloc_4191_; 
v_reuseFailAlloc_4191_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4191_, 0, v_a_4185_);
v___x_4190_ = v_reuseFailAlloc_4191_;
goto v_reusejp_4189_;
}
v_reusejp_4189_:
{
return v___x_4190_;
}
}
}
}
else
{
lean_object* v_a_4193_; lean_object* v___x_4195_; uint8_t v_isShared_4196_; uint8_t v_isSharedCheck_4200_; 
lean_dec(v___y_4036_);
lean_dec_ref(v___y_4035_);
lean_dec(v___y_4034_);
lean_dec_ref(v___y_4033_);
lean_dec_ref(v_a_4028_);
lean_dec(v_a_4027_);
lean_dec_ref(v_snd_4026_);
lean_dec_ref(v_rel_4025_);
lean_dec_ref(v_fst_4024_);
lean_dec_ref(v___f_4022_);
lean_dec(v_a_4021_);
v_a_4193_ = lean_ctor_get(v___x_4051_, 0);
v_isSharedCheck_4200_ = !lean_is_exclusive(v___x_4051_);
if (v_isSharedCheck_4200_ == 0)
{
v___x_4195_ = v___x_4051_;
v_isShared_4196_ = v_isSharedCheck_4200_;
goto v_resetjp_4194_;
}
else
{
lean_inc(v_a_4193_);
lean_dec(v___x_4051_);
v___x_4195_ = lean_box(0);
v_isShared_4196_ = v_isSharedCheck_4200_;
goto v_resetjp_4194_;
}
v_resetjp_4194_:
{
lean_object* v___x_4198_; 
if (v_isShared_4196_ == 0)
{
v___x_4198_ = v___x_4195_;
goto v_reusejp_4197_;
}
else
{
lean_object* v_reuseFailAlloc_4199_; 
v_reuseFailAlloc_4199_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4199_, 0, v_a_4193_);
v___x_4198_ = v_reuseFailAlloc_4199_;
goto v_reusejp_4197_;
}
v_reusejp_4197_:
{
return v___x_4198_;
}
}
}
v___jp_4038_:
{
lean_object* v___x_4041_; lean_object* v___x_4042_; 
v___x_4041_ = l_List_appendTR___redArg(v___y_4039_, v___y_4040_);
v___x_4042_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_4041_, v___y_4030_, v___y_4033_, v___y_4034_, v___y_4035_, v___y_4036_);
lean_dec(v___y_4036_);
lean_dec_ref(v___y_4035_);
lean_dec(v___y_4034_);
lean_dec_ref(v___y_4033_);
if (lean_obj_tag(v___x_4042_) == 0)
{
lean_object* v___x_4044_; uint8_t v_isShared_4045_; uint8_t v_isSharedCheck_4049_; 
v_isSharedCheck_4049_ = !lean_is_exclusive(v___x_4042_);
if (v_isSharedCheck_4049_ == 0)
{
lean_object* v_unused_4050_; 
v_unused_4050_ = lean_ctor_get(v___x_4042_, 0);
lean_dec(v_unused_4050_);
v___x_4044_ = v___x_4042_;
v_isShared_4045_ = v_isSharedCheck_4049_;
goto v_resetjp_4043_;
}
else
{
lean_dec(v___x_4042_);
v___x_4044_ = lean_box(0);
v_isShared_4045_ = v_isSharedCheck_4049_;
goto v_resetjp_4043_;
}
v_resetjp_4043_:
{
lean_object* v___x_4047_; 
if (v_isShared_4045_ == 0)
{
lean_ctor_set(v___x_4044_, 0, v___x_4023_);
v___x_4047_ = v___x_4044_;
goto v_reusejp_4046_;
}
else
{
lean_object* v_reuseFailAlloc_4048_; 
v_reuseFailAlloc_4048_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4048_, 0, v___x_4023_);
v___x_4047_ = v_reuseFailAlloc_4048_;
goto v_reusejp_4046_;
}
v_reusejp_4046_:
{
return v___x_4047_;
}
}
}
else
{
return v___x_4042_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__9___lam__1___boxed(lean_object** _args){
lean_object* v_a_4201_ = _args[0];
lean_object* v___f_4202_ = _args[1];
lean_object* v___x_4203_ = _args[2];
lean_object* v_fst_4204_ = _args[3];
lean_object* v_rel_4205_ = _args[4];
lean_object* v_snd_4206_ = _args[5];
lean_object* v_a_4207_ = _args[6];
lean_object* v_a_4208_ = _args[7];
lean_object* v___y_4209_ = _args[8];
lean_object* v___y_4210_ = _args[9];
lean_object* v___y_4211_ = _args[10];
lean_object* v___y_4212_ = _args[11];
lean_object* v___y_4213_ = _args[12];
lean_object* v___y_4214_ = _args[13];
lean_object* v___y_4215_ = _args[14];
lean_object* v___y_4216_ = _args[15];
lean_object* v___y_4217_ = _args[16];
_start:
{
lean_object* v_res_4218_; 
v_res_4218_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__9___lam__1(v_a_4201_, v___f_4202_, v___x_4203_, v_fst_4204_, v_rel_4205_, v_snd_4206_, v_a_4207_, v_a_4208_, v___y_4209_, v___y_4210_, v___y_4211_, v___y_4212_, v___y_4213_, v___y_4214_, v___y_4215_, v___y_4216_);
lean_dec(v___y_4212_);
lean_dec_ref(v___y_4211_);
lean_dec(v___y_4210_);
lean_dec_ref(v___y_4209_);
return v_res_4218_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__9(lean_object* v_fst_4224_, lean_object* v_rel_4225_, lean_object* v_snd_4226_, lean_object* v_a_4227_, lean_object* v_a_4228_, lean_object* v_a_4229_, lean_object* v_as_4230_, size_t v_sz_4231_, size_t v_i_4232_, lean_object* v_b_4233_, lean_object* v___y_4234_, lean_object* v___y_4235_, lean_object* v___y_4236_, lean_object* v___y_4237_, lean_object* v___y_4238_, lean_object* v___y_4239_, lean_object* v___y_4240_, lean_object* v___y_4241_){
_start:
{
uint8_t v___x_4243_; 
v___x_4243_ = lean_usize_dec_lt(v_i_4232_, v_sz_4231_);
if (v___x_4243_ == 0)
{
lean_object* v___x_4244_; 
lean_dec_ref(v_a_4229_);
lean_dec_ref(v_a_4228_);
lean_dec(v_a_4227_);
lean_dec_ref(v_snd_4226_);
lean_dec_ref(v_rel_4225_);
lean_dec_ref(v_fst_4224_);
v___x_4244_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4244_, 0, v_b_4233_);
return v___x_4244_;
}
else
{
lean_object* v_options_4245_; lean_object* v_inheritedTraceOptions_4246_; uint8_t v_hasTrace_4247_; lean_object* v___f_4248_; lean_object* v___x_4249_; lean_object* v___x_4250_; lean_object* v___y_4252_; lean_object* v___y_4253_; lean_object* v___y_4254_; lean_object* v___y_4255_; lean_object* v___y_4256_; lean_object* v___y_4257_; lean_object* v___y_4258_; lean_object* v___y_4259_; lean_object* v___y_4260_; uint8_t v___y_4261_; lean_object* v_a_4284_; lean_object* v___f_4285_; lean_object* v___y_4287_; lean_object* v___y_4288_; lean_object* v___y_4289_; lean_object* v___y_4290_; lean_object* v___y_4291_; lean_object* v___y_4292_; lean_object* v___y_4293_; lean_object* v___y_4294_; 
lean_dec_ref(v_b_4233_);
v_options_4245_ = lean_ctor_get(v___y_4240_, 2);
v_inheritedTraceOptions_4246_ = lean_ctor_get(v___y_4240_, 13);
v_hasTrace_4247_ = lean_ctor_get_uint8(v_options_4245_, sizeof(void*)*1);
v___f_4248_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__0));
v___x_4249_ = lean_box(0);
v___x_4250_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__2));
v_a_4284_ = lean_array_uget_borrowed(v_as_4230_, v_i_4232_);
lean_inc_ref(v_a_4228_);
lean_inc(v_a_4227_);
lean_inc_ref(v_snd_4226_);
lean_inc_ref(v_rel_4225_);
lean_inc_ref(v_fst_4224_);
lean_inc(v_a_4284_);
v___f_4285_ = lean_alloc_closure((void*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__9___lam__1___boxed), 17, 8);
lean_closure_set(v___f_4285_, 0, v_a_4284_);
lean_closure_set(v___f_4285_, 1, v___f_4248_);
lean_closure_set(v___f_4285_, 2, v___x_4249_);
lean_closure_set(v___f_4285_, 3, v_fst_4224_);
lean_closure_set(v___f_4285_, 4, v_rel_4225_);
lean_closure_set(v___f_4285_, 5, v_snd_4226_);
lean_closure_set(v___f_4285_, 6, v_a_4227_);
lean_closure_set(v___f_4285_, 7, v_a_4228_);
if (v_hasTrace_4247_ == 0)
{
v___y_4287_ = v___y_4234_;
v___y_4288_ = v___y_4235_;
v___y_4289_ = v___y_4236_;
v___y_4290_ = v___y_4237_;
v___y_4291_ = v___y_4238_;
v___y_4292_ = v___y_4239_;
v___y_4293_ = v___y_4240_;
v___y_4294_ = v___y_4241_;
goto v___jp_4286_;
}
else
{
lean_object* v___x_4318_; lean_object* v___x_4319_; uint8_t v___x_4320_; 
v___x_4318_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__2_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_));
v___x_4319_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__3, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__3_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__3);
v___x_4320_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_4246_, v_options_4245_, v___x_4319_);
if (v___x_4320_ == 0)
{
v___y_4287_ = v___y_4234_;
v___y_4288_ = v___y_4235_;
v___y_4289_ = v___y_4236_;
v___y_4290_ = v___y_4237_;
v___y_4291_ = v___y_4238_;
v___y_4292_ = v___y_4239_;
v___y_4293_ = v___y_4240_;
v___y_4294_ = v___y_4241_;
goto v___jp_4286_;
}
else
{
lean_object* v___x_4321_; lean_object* v___x_4322_; lean_object* v___x_4323_; lean_object* v___x_4324_; 
v___x_4321_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__0___closed__1, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__0___closed__1_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__0___closed__1);
lean_inc(v_a_4284_);
v___x_4322_ = l_Lean_MessageData_ofName(v_a_4284_);
v___x_4323_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4323_, 0, v___x_4321_);
lean_ctor_set(v___x_4323_, 1, v___x_4322_);
v___x_4324_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg(v___x_4318_, v___x_4323_, v___y_4238_, v___y_4239_, v___y_4240_, v___y_4241_);
if (lean_obj_tag(v___x_4324_) == 0)
{
lean_dec_ref_known(v___x_4324_, 1);
v___y_4287_ = v___y_4234_;
v___y_4288_ = v___y_4235_;
v___y_4289_ = v___y_4236_;
v___y_4290_ = v___y_4237_;
v___y_4291_ = v___y_4238_;
v___y_4292_ = v___y_4239_;
v___y_4293_ = v___y_4240_;
v___y_4294_ = v___y_4241_;
goto v___jp_4286_;
}
else
{
lean_object* v_a_4325_; lean_object* v___x_4327_; uint8_t v_isShared_4328_; uint8_t v_isSharedCheck_4332_; 
lean_dec_ref(v___f_4285_);
lean_dec_ref(v_a_4229_);
lean_dec_ref(v_a_4228_);
lean_dec(v_a_4227_);
lean_dec_ref(v_snd_4226_);
lean_dec_ref(v_rel_4225_);
lean_dec_ref(v_fst_4224_);
v_a_4325_ = lean_ctor_get(v___x_4324_, 0);
v_isSharedCheck_4332_ = !lean_is_exclusive(v___x_4324_);
if (v_isSharedCheck_4332_ == 0)
{
v___x_4327_ = v___x_4324_;
v_isShared_4328_ = v_isSharedCheck_4332_;
goto v_resetjp_4326_;
}
else
{
lean_inc(v_a_4325_);
lean_dec(v___x_4324_);
v___x_4327_ = lean_box(0);
v_isShared_4328_ = v_isSharedCheck_4332_;
goto v_resetjp_4326_;
}
v_resetjp_4326_:
{
lean_object* v___x_4330_; 
if (v_isShared_4328_ == 0)
{
v___x_4330_ = v___x_4327_;
goto v_reusejp_4329_;
}
else
{
lean_object* v_reuseFailAlloc_4331_; 
v_reuseFailAlloc_4331_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4331_, 0, v_a_4325_);
v___x_4330_ = v_reuseFailAlloc_4331_;
goto v_reusejp_4329_;
}
v_reusejp_4329_:
{
return v___x_4330_;
}
}
}
}
}
v___jp_4251_:
{
if (v___y_4261_ == 0)
{
lean_object* v___x_4262_; 
lean_dec_ref(v___y_4254_);
v___x_4262_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v___y_4256_, v___y_4261_, v___y_4259_, v___y_4253_, v___y_4258_, v___y_4260_, v___y_4257_, v___y_4255_, v___y_4252_);
if (lean_obj_tag(v___x_4262_) == 0)
{
lean_object* v___x_4263_; 
lean_dec_ref_known(v___x_4262_, 1);
lean_inc_ref(v_a_4229_);
v___x_4263_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_4229_, v___y_4261_, v___y_4259_, v___y_4253_, v___y_4258_, v___y_4260_, v___y_4257_, v___y_4255_, v___y_4252_);
if (lean_obj_tag(v___x_4263_) == 0)
{
size_t v___x_4264_; size_t v___x_4265_; 
lean_dec_ref_known(v___x_4263_, 1);
v___x_4264_ = ((size_t)1ULL);
v___x_4265_ = lean_usize_add(v_i_4232_, v___x_4264_);
v_i_4232_ = v___x_4265_;
v_b_4233_ = v___x_4250_;
goto _start;
}
else
{
lean_object* v_a_4267_; lean_object* v___x_4269_; uint8_t v_isShared_4270_; uint8_t v_isSharedCheck_4274_; 
lean_dec_ref(v_a_4229_);
lean_dec_ref(v_a_4228_);
lean_dec(v_a_4227_);
lean_dec_ref(v_snd_4226_);
lean_dec_ref(v_rel_4225_);
lean_dec_ref(v_fst_4224_);
v_a_4267_ = lean_ctor_get(v___x_4263_, 0);
v_isSharedCheck_4274_ = !lean_is_exclusive(v___x_4263_);
if (v_isSharedCheck_4274_ == 0)
{
v___x_4269_ = v___x_4263_;
v_isShared_4270_ = v_isSharedCheck_4274_;
goto v_resetjp_4268_;
}
else
{
lean_inc(v_a_4267_);
lean_dec(v___x_4263_);
v___x_4269_ = lean_box(0);
v_isShared_4270_ = v_isSharedCheck_4274_;
goto v_resetjp_4268_;
}
v_resetjp_4268_:
{
lean_object* v___x_4272_; 
if (v_isShared_4270_ == 0)
{
v___x_4272_ = v___x_4269_;
goto v_reusejp_4271_;
}
else
{
lean_object* v_reuseFailAlloc_4273_; 
v_reuseFailAlloc_4273_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4273_, 0, v_a_4267_);
v___x_4272_ = v_reuseFailAlloc_4273_;
goto v_reusejp_4271_;
}
v_reusejp_4271_:
{
return v___x_4272_;
}
}
}
}
else
{
lean_object* v_a_4275_; lean_object* v___x_4277_; uint8_t v_isShared_4278_; uint8_t v_isSharedCheck_4282_; 
lean_dec_ref(v_a_4229_);
lean_dec_ref(v_a_4228_);
lean_dec(v_a_4227_);
lean_dec_ref(v_snd_4226_);
lean_dec_ref(v_rel_4225_);
lean_dec_ref(v_fst_4224_);
v_a_4275_ = lean_ctor_get(v___x_4262_, 0);
v_isSharedCheck_4282_ = !lean_is_exclusive(v___x_4262_);
if (v_isSharedCheck_4282_ == 0)
{
v___x_4277_ = v___x_4262_;
v_isShared_4278_ = v_isSharedCheck_4282_;
goto v_resetjp_4276_;
}
else
{
lean_inc(v_a_4275_);
lean_dec(v___x_4262_);
v___x_4277_ = lean_box(0);
v_isShared_4278_ = v_isSharedCheck_4282_;
goto v_resetjp_4276_;
}
v_resetjp_4276_:
{
lean_object* v___x_4280_; 
if (v_isShared_4278_ == 0)
{
v___x_4280_ = v___x_4277_;
goto v_reusejp_4279_;
}
else
{
lean_object* v_reuseFailAlloc_4281_; 
v_reuseFailAlloc_4281_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4281_, 0, v_a_4275_);
v___x_4280_ = v_reuseFailAlloc_4281_;
goto v_reusejp_4279_;
}
v_reusejp_4279_:
{
return v___x_4280_;
}
}
}
}
else
{
lean_object* v___x_4283_; 
lean_dec_ref(v___y_4256_);
lean_dec_ref(v_a_4229_);
lean_dec_ref(v_a_4228_);
lean_dec(v_a_4227_);
lean_dec_ref(v_snd_4226_);
lean_dec_ref(v_rel_4225_);
lean_dec_ref(v_fst_4224_);
v___x_4283_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4283_, 0, v___y_4254_);
return v___x_4283_;
}
}
v___jp_4286_:
{
lean_object* v___x_4295_; 
v___x_4295_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_4288_, v___y_4290_, v___y_4292_, v___y_4294_);
if (lean_obj_tag(v___x_4295_) == 0)
{
lean_object* v_a_4296_; lean_object* v___x_4297_; 
v_a_4296_ = lean_ctor_get(v___x_4295_, 0);
lean_inc(v_a_4296_);
lean_dec_ref_known(v___x_4295_, 1);
v___x_4297_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_4285_, v___y_4287_, v___y_4288_, v___y_4289_, v___y_4290_, v___y_4291_, v___y_4292_, v___y_4293_, v___y_4294_);
if (lean_obj_tag(v___x_4297_) == 0)
{
lean_object* v___x_4299_; uint8_t v_isShared_4300_; uint8_t v_isSharedCheck_4305_; 
lean_dec(v_a_4296_);
lean_dec_ref(v_a_4229_);
lean_dec_ref(v_a_4228_);
lean_dec(v_a_4227_);
lean_dec_ref(v_snd_4226_);
lean_dec_ref(v_rel_4225_);
lean_dec_ref(v_fst_4224_);
v_isSharedCheck_4305_ = !lean_is_exclusive(v___x_4297_);
if (v_isSharedCheck_4305_ == 0)
{
lean_object* v_unused_4306_; 
v_unused_4306_ = lean_ctor_get(v___x_4297_, 0);
lean_dec(v_unused_4306_);
v___x_4299_ = v___x_4297_;
v_isShared_4300_ = v_isSharedCheck_4305_;
goto v_resetjp_4298_;
}
else
{
lean_dec(v___x_4297_);
v___x_4299_ = lean_box(0);
v_isShared_4300_ = v_isSharedCheck_4305_;
goto v_resetjp_4298_;
}
v_resetjp_4298_:
{
lean_object* v___x_4301_; lean_object* v___x_4303_; 
v___x_4301_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__9___closed__1));
if (v_isShared_4300_ == 0)
{
lean_ctor_set(v___x_4299_, 0, v___x_4301_);
v___x_4303_ = v___x_4299_;
goto v_reusejp_4302_;
}
else
{
lean_object* v_reuseFailAlloc_4304_; 
v_reuseFailAlloc_4304_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4304_, 0, v___x_4301_);
v___x_4303_ = v_reuseFailAlloc_4304_;
goto v_reusejp_4302_;
}
v_reusejp_4302_:
{
return v___x_4303_;
}
}
}
else
{
lean_object* v_a_4307_; uint8_t v___x_4308_; 
v_a_4307_ = lean_ctor_get(v___x_4297_, 0);
lean_inc(v_a_4307_);
lean_dec_ref_known(v___x_4297_, 1);
v___x_4308_ = l_Lean_Exception_isInterrupt(v_a_4307_);
if (v___x_4308_ == 0)
{
uint8_t v___x_4309_; 
lean_inc(v_a_4307_);
v___x_4309_ = l_Lean_Exception_isRuntime(v_a_4307_);
v___y_4252_ = v___y_4294_;
v___y_4253_ = v___y_4289_;
v___y_4254_ = v_a_4307_;
v___y_4255_ = v___y_4293_;
v___y_4256_ = v_a_4296_;
v___y_4257_ = v___y_4292_;
v___y_4258_ = v___y_4290_;
v___y_4259_ = v___y_4288_;
v___y_4260_ = v___y_4291_;
v___y_4261_ = v___x_4309_;
goto v___jp_4251_;
}
else
{
v___y_4252_ = v___y_4294_;
v___y_4253_ = v___y_4289_;
v___y_4254_ = v_a_4307_;
v___y_4255_ = v___y_4293_;
v___y_4256_ = v_a_4296_;
v___y_4257_ = v___y_4292_;
v___y_4258_ = v___y_4290_;
v___y_4259_ = v___y_4288_;
v___y_4260_ = v___y_4291_;
v___y_4261_ = v___x_4308_;
goto v___jp_4251_;
}
}
}
else
{
lean_object* v_a_4310_; lean_object* v___x_4312_; uint8_t v_isShared_4313_; uint8_t v_isSharedCheck_4317_; 
lean_dec_ref(v___f_4285_);
lean_dec_ref(v_a_4229_);
lean_dec_ref(v_a_4228_);
lean_dec(v_a_4227_);
lean_dec_ref(v_snd_4226_);
lean_dec_ref(v_rel_4225_);
lean_dec_ref(v_fst_4224_);
v_a_4310_ = lean_ctor_get(v___x_4295_, 0);
v_isSharedCheck_4317_ = !lean_is_exclusive(v___x_4295_);
if (v_isSharedCheck_4317_ == 0)
{
v___x_4312_ = v___x_4295_;
v_isShared_4313_ = v_isSharedCheck_4317_;
goto v_resetjp_4311_;
}
else
{
lean_inc(v_a_4310_);
lean_dec(v___x_4295_);
v___x_4312_ = lean_box(0);
v_isShared_4313_ = v_isSharedCheck_4317_;
goto v_resetjp_4311_;
}
v_resetjp_4311_:
{
lean_object* v___x_4315_; 
if (v_isShared_4313_ == 0)
{
v___x_4315_ = v___x_4312_;
goto v_reusejp_4314_;
}
else
{
lean_object* v_reuseFailAlloc_4316_; 
v_reuseFailAlloc_4316_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4316_, 0, v_a_4310_);
v___x_4315_ = v_reuseFailAlloc_4316_;
goto v_reusejp_4314_;
}
v_reusejp_4314_:
{
return v___x_4315_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__9___boxed(lean_object** _args){
lean_object* v_fst_4333_ = _args[0];
lean_object* v_rel_4334_ = _args[1];
lean_object* v_snd_4335_ = _args[2];
lean_object* v_a_4336_ = _args[3];
lean_object* v_a_4337_ = _args[4];
lean_object* v_a_4338_ = _args[5];
lean_object* v_as_4339_ = _args[6];
lean_object* v_sz_4340_ = _args[7];
lean_object* v_i_4341_ = _args[8];
lean_object* v_b_4342_ = _args[9];
lean_object* v___y_4343_ = _args[10];
lean_object* v___y_4344_ = _args[11];
lean_object* v___y_4345_ = _args[12];
lean_object* v___y_4346_ = _args[13];
lean_object* v___y_4347_ = _args[14];
lean_object* v___y_4348_ = _args[15];
lean_object* v___y_4349_ = _args[16];
lean_object* v___y_4350_ = _args[17];
lean_object* v___y_4351_ = _args[18];
_start:
{
size_t v_sz_boxed_4352_; size_t v_i_boxed_4353_; lean_object* v_res_4354_; 
v_sz_boxed_4352_ = lean_unbox_usize(v_sz_4340_);
lean_dec(v_sz_4340_);
v_i_boxed_4353_ = lean_unbox_usize(v_i_4341_);
lean_dec(v_i_4341_);
v_res_4354_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__9(v_fst_4333_, v_rel_4334_, v_snd_4335_, v_a_4336_, v_a_4337_, v_a_4338_, v_as_4339_, v_sz_boxed_4352_, v_i_boxed_4353_, v_b_4342_, v___y_4343_, v___y_4344_, v___y_4345_, v___y_4346_, v___y_4347_, v___y_4348_, v___y_4349_, v___y_4350_);
lean_dec(v___y_4350_);
lean_dec_ref(v___y_4349_);
lean_dec(v___y_4348_);
lean_dec_ref(v___y_4347_);
lean_dec(v___y_4346_);
lean_dec_ref(v___y_4345_);
lean_dec(v___y_4344_);
lean_dec_ref(v___y_4343_);
lean_dec_ref(v_as_4339_);
return v_res_4354_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__2(lean_object* v_rel_4361_, lean_object* v_fst_4362_, lean_object* v_snd_4363_, lean_object* v_a_4364_, lean_object* v_a_4365_, lean_object* v_a_4366_, lean_object* v_____r_4367_, lean_object* v___y_4368_, lean_object* v___y_4369_, lean_object* v___y_4370_, lean_object* v___y_4371_, lean_object* v___y_4372_, lean_object* v___y_4373_, lean_object* v___y_4374_, lean_object* v___y_4375_){
_start:
{
lean_object* v___x_4377_; lean_object* v_env_4378_; lean_object* v___x_4379_; lean_object* v_ext_4380_; lean_object* v_toEnvExtension_4381_; lean_object* v_asyncMode_4382_; lean_object* v___x_4383_; lean_object* v___x_4384_; lean_object* v___x_4385_; 
v___x_4377_ = lean_st_ref_get(v___y_4375_);
v_env_4378_ = lean_ctor_get(v___x_4377_, 0);
lean_inc_ref(v_env_4378_);
lean_dec(v___x_4377_);
v___x_4379_ = lp_batteries_Batteries_Tactic_transExt;
v_ext_4380_ = lean_ctor_get(v___x_4379_, 1);
v_toEnvExtension_4381_ = lean_ctor_get(v_ext_4380_, 0);
v_asyncMode_4382_ = lean_ctor_get(v_toEnvExtension_4381_, 2);
v___x_4383_ = lean_obj_once(&lp_batteries_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__2___closed__0, &lp_batteries_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__2___closed__0_once, _init_lp_batteries_panic___at___00Lean_Meta_DiscrTree_insertKeyValue___at___00__private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2__spec__0_spec__2___closed__0);
v___x_4384_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_4383_, v___x_4379_, v_env_4378_, v_asyncMode_4382_);
lean_inc_ref(v_rel_4361_);
v___x_4385_ = l_Lean_Meta_DiscrTree_getUnify___redArg(v___x_4384_, v_rel_4361_, v___y_4372_, v___y_4373_, v___y_4374_, v___y_4375_);
if (lean_obj_tag(v___x_4385_) == 0)
{
lean_object* v_a_4386_; lean_object* v___x_4387_; lean_object* v___x_4388_; lean_object* v___x_4389_; size_t v_sz_4390_; size_t v___x_4391_; lean_object* v___x_4392_; 
v_a_4386_ = lean_ctor_get(v___x_4385_, 0);
lean_inc(v_a_4386_);
lean_dec_ref_known(v___x_4385_, 1);
v___x_4387_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__2___closed__1));
v___x_4388_ = lean_array_push(v_a_4386_, v___x_4387_);
v___x_4389_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___closed__2));
v_sz_4390_ = lean_array_size(v___x_4388_);
v___x_4391_ = ((size_t)0ULL);
v___x_4392_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__9(v_fst_4362_, v_rel_4361_, v_snd_4363_, v_a_4364_, v_a_4365_, v_a_4366_, v___x_4388_, v_sz_4390_, v___x_4391_, v___x_4389_, v___y_4368_, v___y_4369_, v___y_4370_, v___y_4371_, v___y_4372_, v___y_4373_, v___y_4374_, v___y_4375_);
lean_dec_ref(v___x_4388_);
if (lean_obj_tag(v___x_4392_) == 0)
{
lean_object* v_a_4393_; lean_object* v___x_4395_; uint8_t v_isShared_4396_; uint8_t v_isSharedCheck_4413_; 
v_a_4393_ = lean_ctor_get(v___x_4392_, 0);
v_isSharedCheck_4413_ = !lean_is_exclusive(v___x_4392_);
if (v_isSharedCheck_4413_ == 0)
{
v___x_4395_ = v___x_4392_;
v_isShared_4396_ = v_isSharedCheck_4413_;
goto v_resetjp_4394_;
}
else
{
lean_inc(v_a_4393_);
lean_dec(v___x_4392_);
v___x_4395_ = lean_box(0);
v_isShared_4396_ = v_isSharedCheck_4413_;
goto v_resetjp_4394_;
}
v_resetjp_4394_:
{
lean_object* v_fst_4397_; 
v_fst_4397_ = lean_ctor_get(v_a_4393_, 0);
lean_inc(v_fst_4397_);
lean_dec(v_a_4393_);
if (lean_obj_tag(v_fst_4397_) == 0)
{
lean_object* v___x_4398_; lean_object* v___x_4400_; 
v___x_4398_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__2___closed__2));
if (v_isShared_4396_ == 0)
{
lean_ctor_set(v___x_4395_, 0, v___x_4398_);
v___x_4400_ = v___x_4395_;
goto v_reusejp_4399_;
}
else
{
lean_object* v_reuseFailAlloc_4401_; 
v_reuseFailAlloc_4401_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4401_, 0, v___x_4398_);
v___x_4400_ = v_reuseFailAlloc_4401_;
goto v_reusejp_4399_;
}
v_reusejp_4399_:
{
return v___x_4400_;
}
}
else
{
lean_object* v_val_4402_; lean_object* v___x_4404_; uint8_t v_isShared_4405_; uint8_t v_isSharedCheck_4412_; 
v_val_4402_ = lean_ctor_get(v_fst_4397_, 0);
v_isSharedCheck_4412_ = !lean_is_exclusive(v_fst_4397_);
if (v_isSharedCheck_4412_ == 0)
{
v___x_4404_ = v_fst_4397_;
v_isShared_4405_ = v_isSharedCheck_4412_;
goto v_resetjp_4403_;
}
else
{
lean_inc(v_val_4402_);
lean_dec(v_fst_4397_);
v___x_4404_ = lean_box(0);
v_isShared_4405_ = v_isSharedCheck_4412_;
goto v_resetjp_4403_;
}
v_resetjp_4403_:
{
lean_object* v___x_4407_; 
if (v_isShared_4405_ == 0)
{
lean_ctor_set_tag(v___x_4404_, 0);
v___x_4407_ = v___x_4404_;
goto v_reusejp_4406_;
}
else
{
lean_object* v_reuseFailAlloc_4411_; 
v_reuseFailAlloc_4411_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4411_, 0, v_val_4402_);
v___x_4407_ = v_reuseFailAlloc_4411_;
goto v_reusejp_4406_;
}
v_reusejp_4406_:
{
lean_object* v___x_4409_; 
if (v_isShared_4396_ == 0)
{
lean_ctor_set(v___x_4395_, 0, v___x_4407_);
v___x_4409_ = v___x_4395_;
goto v_reusejp_4408_;
}
else
{
lean_object* v_reuseFailAlloc_4410_; 
v_reuseFailAlloc_4410_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4410_, 0, v___x_4407_);
v___x_4409_ = v_reuseFailAlloc_4410_;
goto v_reusejp_4408_;
}
v_reusejp_4408_:
{
return v___x_4409_;
}
}
}
}
}
}
else
{
lean_object* v_a_4414_; lean_object* v___x_4416_; uint8_t v_isShared_4417_; uint8_t v_isSharedCheck_4421_; 
v_a_4414_ = lean_ctor_get(v___x_4392_, 0);
v_isSharedCheck_4421_ = !lean_is_exclusive(v___x_4392_);
if (v_isSharedCheck_4421_ == 0)
{
v___x_4416_ = v___x_4392_;
v_isShared_4417_ = v_isSharedCheck_4421_;
goto v_resetjp_4415_;
}
else
{
lean_inc(v_a_4414_);
lean_dec(v___x_4392_);
v___x_4416_ = lean_box(0);
v_isShared_4417_ = v_isSharedCheck_4421_;
goto v_resetjp_4415_;
}
v_resetjp_4415_:
{
lean_object* v___x_4419_; 
if (v_isShared_4417_ == 0)
{
v___x_4419_ = v___x_4416_;
goto v_reusejp_4418_;
}
else
{
lean_object* v_reuseFailAlloc_4420_; 
v_reuseFailAlloc_4420_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4420_, 0, v_a_4414_);
v___x_4419_ = v_reuseFailAlloc_4420_;
goto v_reusejp_4418_;
}
v_reusejp_4418_:
{
return v___x_4419_;
}
}
}
}
else
{
lean_object* v_a_4422_; lean_object* v___x_4424_; uint8_t v_isShared_4425_; uint8_t v_isSharedCheck_4429_; 
lean_dec_ref(v_a_4366_);
lean_dec_ref(v_a_4365_);
lean_dec(v_a_4364_);
lean_dec_ref(v_snd_4363_);
lean_dec_ref(v_fst_4362_);
lean_dec_ref(v_rel_4361_);
v_a_4422_ = lean_ctor_get(v___x_4385_, 0);
v_isSharedCheck_4429_ = !lean_is_exclusive(v___x_4385_);
if (v_isSharedCheck_4429_ == 0)
{
v___x_4424_ = v___x_4385_;
v_isShared_4425_ = v_isSharedCheck_4429_;
goto v_resetjp_4423_;
}
else
{
lean_inc(v_a_4422_);
lean_dec(v___x_4385_);
v___x_4424_ = lean_box(0);
v_isShared_4425_ = v_isSharedCheck_4429_;
goto v_resetjp_4423_;
}
v_resetjp_4423_:
{
lean_object* v___x_4427_; 
if (v_isShared_4425_ == 0)
{
v___x_4427_ = v___x_4424_;
goto v_reusejp_4426_;
}
else
{
lean_object* v_reuseFailAlloc_4428_; 
v_reuseFailAlloc_4428_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4428_, 0, v_a_4422_);
v___x_4427_ = v_reuseFailAlloc_4428_;
goto v_reusejp_4426_;
}
v_reusejp_4426_:
{
return v___x_4427_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__2___boxed(lean_object* v_rel_4430_, lean_object* v_fst_4431_, lean_object* v_snd_4432_, lean_object* v_a_4433_, lean_object* v_a_4434_, lean_object* v_a_4435_, lean_object* v_____r_4436_, lean_object* v___y_4437_, lean_object* v___y_4438_, lean_object* v___y_4439_, lean_object* v___y_4440_, lean_object* v___y_4441_, lean_object* v___y_4442_, lean_object* v___y_4443_, lean_object* v___y_4444_, lean_object* v___y_4445_){
_start:
{
lean_object* v_res_4446_; 
v_res_4446_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__2(v_rel_4430_, v_fst_4431_, v_snd_4432_, v_a_4433_, v_a_4434_, v_a_4435_, v_____r_4436_, v___y_4437_, v___y_4438_, v___y_4439_, v___y_4440_, v___y_4441_, v___y_4442_, v___y_4443_, v___y_4444_);
lean_dec(v___y_4444_);
lean_dec_ref(v___y_4443_);
lean_dec(v___y_4442_);
lean_dec_ref(v___y_4441_);
lean_dec(v___y_4440_);
lean_dec_ref(v___y_4439_);
lean_dec(v___y_4438_);
lean_dec_ref(v___y_4437_);
return v_res_4446_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__3(lean_object* v_name_4447_, lean_object* v_fst_4448_, lean_object* v_a_4449_, uint8_t v_bi_4450_, lean_object* v___x_4451_, lean_object* v_snd_4452_, lean_object* v___x_4453_, lean_object* v_a_4454_, lean_object* v___y_4455_, lean_object* v___y_4456_, lean_object* v___y_4457_, lean_object* v___y_4458_, lean_object* v___y_4459_, lean_object* v___y_4460_, lean_object* v___y_4461_, lean_object* v___y_4462_){
_start:
{
lean_object* v___x_4464_; 
v___x_4464_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_4456_, v___y_4459_, v___y_4460_, v___y_4461_, v___y_4462_);
if (lean_obj_tag(v___x_4464_) == 0)
{
lean_object* v_a_4465_; lean_object* v___x_4466_; lean_object* v___x_4467_; uint8_t v___x_4468_; lean_object* v___x_4469_; 
v_a_4465_ = lean_ctor_get(v___x_4464_, 0);
lean_inc(v_a_4465_);
lean_dec_ref_known(v___x_4464_, 1);
lean_inc_ref(v_a_4449_);
lean_inc_ref(v_fst_4448_);
lean_inc(v_name_4447_);
v___x_4466_ = l_Lean_Expr_forallE___override(v_name_4447_, v_fst_4448_, v_a_4449_, v_bi_4450_);
v___x_4467_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4467_, 0, v___x_4466_);
v___x_4468_ = 1;
lean_inc(v___x_4451_);
v___x_4469_ = l_Lean_Meta_mkFreshExprMVar(v___x_4467_, v___x_4468_, v___x_4451_, v___y_4459_, v___y_4460_, v___y_4461_, v___y_4462_);
if (lean_obj_tag(v___x_4469_) == 0)
{
lean_object* v_a_4470_; lean_object* v___x_4471_; lean_object* v___x_4472_; lean_object* v___x_4473_; 
v_a_4470_ = lean_ctor_get(v___x_4469_, 0);
lean_inc(v_a_4470_);
lean_dec_ref_known(v___x_4469_, 1);
lean_inc_ref(v_a_4449_);
lean_inc(v_name_4447_);
v___x_4471_ = l_Lean_Expr_forallE___override(v_name_4447_, v_a_4449_, v_snd_4452_, v_bi_4450_);
v___x_4472_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4472_, 0, v___x_4471_);
v___x_4473_ = l_Lean_Meta_mkFreshExprMVar(v___x_4472_, v___x_4468_, v___x_4451_, v___y_4459_, v___y_4460_, v___y_4461_, v___y_4462_);
if (lean_obj_tag(v___x_4473_) == 0)
{
lean_object* v_a_4474_; lean_object* v___x_4475_; lean_object* v___x_4476_; lean_object* v___x_4477_; uint8_t v___x_4478_; lean_object* v___x_4479_; lean_object* v___x_4480_; lean_object* v___x_4481_; lean_object* v___x_4482_; lean_object* v___x_4483_; lean_object* v___x_4484_; lean_object* v___x_4485_; lean_object* v___y_4487_; 
v_a_4474_ = lean_ctor_get(v___x_4473_, 0);
lean_inc_n(v_a_4474_, 2);
lean_dec_ref_known(v___x_4473_, 1);
v___x_4475_ = l_Lean_Expr_bvar___override(v___x_4453_);
lean_inc(v_a_4470_);
v___x_4476_ = l_Lean_Expr_app___override(v_a_4470_, v___x_4475_);
v___x_4477_ = l_Lean_Expr_app___override(v_a_4474_, v___x_4476_);
v___x_4478_ = 0;
v___x_4479_ = l_Lean_Expr_lam___override(v_name_4447_, v_fst_4448_, v___x_4477_, v___x_4478_);
v___x_4480_ = lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4___redArg(v_a_4465_, v___x_4479_, v___y_4460_);
lean_dec_ref(v___x_4480_);
v___x_4481_ = l_Lean_Expr_mvarId_x21(v_a_4470_);
lean_dec(v_a_4470_);
v___x_4482_ = l_Lean_Expr_mvarId_x21(v_a_4474_);
lean_dec(v_a_4474_);
v___x_4483_ = lean_box(0);
v___x_4484_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4484_, 0, v___x_4482_);
lean_ctor_set(v___x_4484_, 1, v___x_4483_);
v___x_4485_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4485_, 0, v___x_4481_);
lean_ctor_set(v___x_4485_, 1, v___x_4484_);
if (lean_obj_tag(v_a_4454_) == 1)
{
lean_object* v_val_4499_; lean_object* v_snd_4500_; 
lean_dec_ref(v_a_4449_);
v_val_4499_ = lean_ctor_get(v_a_4454_, 0);
lean_inc(v_val_4499_);
lean_dec_ref_known(v_a_4454_, 1);
v_snd_4500_ = lean_ctor_get(v_val_4499_, 1);
lean_inc(v_snd_4500_);
lean_dec(v_val_4499_);
v___y_4487_ = v_snd_4500_;
goto v___jp_4486_;
}
else
{
lean_object* v___x_4501_; lean_object* v___x_4502_; 
lean_dec(v_a_4454_);
v___x_4501_ = l_Lean_Expr_mvarId_x21(v_a_4449_);
lean_dec_ref(v_a_4449_);
v___x_4502_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4502_, 0, v___x_4501_);
lean_ctor_set(v___x_4502_, 1, v___x_4483_);
v___y_4487_ = v___x_4502_;
goto v___jp_4486_;
}
v___jp_4486_:
{
lean_object* v___x_4488_; lean_object* v___x_4489_; 
v___x_4488_ = l_List_appendTR___redArg(v___x_4485_, v___y_4487_);
v___x_4489_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_4488_, v___y_4456_, v___y_4459_, v___y_4460_, v___y_4461_, v___y_4462_);
if (lean_obj_tag(v___x_4489_) == 0)
{
lean_object* v___x_4491_; uint8_t v_isShared_4492_; uint8_t v_isSharedCheck_4497_; 
v_isSharedCheck_4497_ = !lean_is_exclusive(v___x_4489_);
if (v_isSharedCheck_4497_ == 0)
{
lean_object* v_unused_4498_; 
v_unused_4498_ = lean_ctor_get(v___x_4489_, 0);
lean_dec(v_unused_4498_);
v___x_4491_ = v___x_4489_;
v_isShared_4492_ = v_isSharedCheck_4497_;
goto v_resetjp_4490_;
}
else
{
lean_dec(v___x_4489_);
v___x_4491_ = lean_box(0);
v_isShared_4492_ = v_isSharedCheck_4497_;
goto v_resetjp_4490_;
}
v_resetjp_4490_:
{
lean_object* v___x_4493_; lean_object* v___x_4495_; 
v___x_4493_ = lean_box(0);
if (v_isShared_4492_ == 0)
{
lean_ctor_set(v___x_4491_, 0, v___x_4493_);
v___x_4495_ = v___x_4491_;
goto v_reusejp_4494_;
}
else
{
lean_object* v_reuseFailAlloc_4496_; 
v_reuseFailAlloc_4496_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4496_, 0, v___x_4493_);
v___x_4495_ = v_reuseFailAlloc_4496_;
goto v_reusejp_4494_;
}
v_reusejp_4494_:
{
return v___x_4495_;
}
}
}
else
{
return v___x_4489_;
}
}
}
else
{
lean_object* v_a_4503_; lean_object* v___x_4505_; uint8_t v_isShared_4506_; uint8_t v_isSharedCheck_4510_; 
lean_dec(v_a_4470_);
lean_dec(v_a_4465_);
lean_dec(v_a_4454_);
lean_dec(v___x_4453_);
lean_dec_ref(v_a_4449_);
lean_dec_ref(v_fst_4448_);
lean_dec(v_name_4447_);
v_a_4503_ = lean_ctor_get(v___x_4473_, 0);
v_isSharedCheck_4510_ = !lean_is_exclusive(v___x_4473_);
if (v_isSharedCheck_4510_ == 0)
{
v___x_4505_ = v___x_4473_;
v_isShared_4506_ = v_isSharedCheck_4510_;
goto v_resetjp_4504_;
}
else
{
lean_inc(v_a_4503_);
lean_dec(v___x_4473_);
v___x_4505_ = lean_box(0);
v_isShared_4506_ = v_isSharedCheck_4510_;
goto v_resetjp_4504_;
}
v_resetjp_4504_:
{
lean_object* v___x_4508_; 
if (v_isShared_4506_ == 0)
{
v___x_4508_ = v___x_4505_;
goto v_reusejp_4507_;
}
else
{
lean_object* v_reuseFailAlloc_4509_; 
v_reuseFailAlloc_4509_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4509_, 0, v_a_4503_);
v___x_4508_ = v_reuseFailAlloc_4509_;
goto v_reusejp_4507_;
}
v_reusejp_4507_:
{
return v___x_4508_;
}
}
}
}
else
{
lean_object* v_a_4511_; lean_object* v___x_4513_; uint8_t v_isShared_4514_; uint8_t v_isSharedCheck_4518_; 
lean_dec(v_a_4465_);
lean_dec(v_a_4454_);
lean_dec(v___x_4453_);
lean_dec_ref(v_snd_4452_);
lean_dec(v___x_4451_);
lean_dec_ref(v_a_4449_);
lean_dec_ref(v_fst_4448_);
lean_dec(v_name_4447_);
v_a_4511_ = lean_ctor_get(v___x_4469_, 0);
v_isSharedCheck_4518_ = !lean_is_exclusive(v___x_4469_);
if (v_isSharedCheck_4518_ == 0)
{
v___x_4513_ = v___x_4469_;
v_isShared_4514_ = v_isSharedCheck_4518_;
goto v_resetjp_4512_;
}
else
{
lean_inc(v_a_4511_);
lean_dec(v___x_4469_);
v___x_4513_ = lean_box(0);
v_isShared_4514_ = v_isSharedCheck_4518_;
goto v_resetjp_4512_;
}
v_resetjp_4512_:
{
lean_object* v___x_4516_; 
if (v_isShared_4514_ == 0)
{
v___x_4516_ = v___x_4513_;
goto v_reusejp_4515_;
}
else
{
lean_object* v_reuseFailAlloc_4517_; 
v_reuseFailAlloc_4517_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4517_, 0, v_a_4511_);
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
lean_object* v_a_4519_; lean_object* v___x_4521_; uint8_t v_isShared_4522_; uint8_t v_isSharedCheck_4526_; 
lean_dec(v_a_4454_);
lean_dec(v___x_4453_);
lean_dec_ref(v_snd_4452_);
lean_dec(v___x_4451_);
lean_dec_ref(v_a_4449_);
lean_dec_ref(v_fst_4448_);
lean_dec(v_name_4447_);
v_a_4519_ = lean_ctor_get(v___x_4464_, 0);
v_isSharedCheck_4526_ = !lean_is_exclusive(v___x_4464_);
if (v_isSharedCheck_4526_ == 0)
{
v___x_4521_ = v___x_4464_;
v_isShared_4522_ = v_isSharedCheck_4526_;
goto v_resetjp_4520_;
}
else
{
lean_inc(v_a_4519_);
lean_dec(v___x_4464_);
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
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__3___boxed(lean_object** _args){
lean_object* v_name_4527_ = _args[0];
lean_object* v_fst_4528_ = _args[1];
lean_object* v_a_4529_ = _args[2];
lean_object* v_bi_4530_ = _args[3];
lean_object* v___x_4531_ = _args[4];
lean_object* v_snd_4532_ = _args[5];
lean_object* v___x_4533_ = _args[6];
lean_object* v_a_4534_ = _args[7];
lean_object* v___y_4535_ = _args[8];
lean_object* v___y_4536_ = _args[9];
lean_object* v___y_4537_ = _args[10];
lean_object* v___y_4538_ = _args[11];
lean_object* v___y_4539_ = _args[12];
lean_object* v___y_4540_ = _args[13];
lean_object* v___y_4541_ = _args[14];
lean_object* v___y_4542_ = _args[15];
lean_object* v___y_4543_ = _args[16];
_start:
{
uint8_t v_bi_96132__boxed_4544_; lean_object* v_res_4545_; 
v_bi_96132__boxed_4544_ = lean_unbox(v_bi_4530_);
v_res_4545_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__3(v_name_4527_, v_fst_4528_, v_a_4529_, v_bi_96132__boxed_4544_, v___x_4531_, v_snd_4532_, v___x_4533_, v_a_4534_, v___y_4535_, v___y_4536_, v___y_4537_, v___y_4538_, v___y_4539_, v___y_4540_, v___y_4541_, v___y_4542_);
lean_dec(v___y_4542_);
lean_dec_ref(v___y_4541_);
lean_dec(v___y_4540_);
lean_dec_ref(v___y_4539_);
lean_dec(v___y_4538_);
lean_dec_ref(v___y_4537_);
lean_dec(v___y_4536_);
lean_dec_ref(v___y_4535_);
return v_res_4545_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__1(void){
_start:
{
lean_object* v___x_4547_; lean_object* v___x_4548_; 
v___x_4547_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__0));
v___x_4548_ = l_Lean_stringToMessageData(v___x_4547_);
return v___x_4548_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__3(void){
_start:
{
lean_object* v___x_4550_; lean_object* v___x_4551_; 
v___x_4550_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__2));
v___x_4551_ = l_Lean_stringToMessageData(v___x_4550_);
return v___x_4551_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__5(void){
_start:
{
lean_object* v___x_4553_; lean_object* v___x_4554_; 
v___x_4553_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__4));
v___x_4554_ = l_Lean_stringToMessageData(v___x_4553_);
return v___x_4554_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__7(void){
_start:
{
lean_object* v___x_4556_; lean_object* v___x_4557_; 
v___x_4556_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__6));
v___x_4557_ = l_Lean_stringToMessageData(v___x_4556_);
return v___x_4557_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__9(void){
_start:
{
lean_object* v___x_4559_; lean_object* v___x_4560_; 
v___x_4559_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__8));
v___x_4560_ = l_Lean_stringToMessageData(v___x_4559_);
return v___x_4560_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__11(void){
_start:
{
lean_object* v___x_4562_; lean_object* v___x_4563_; 
v___x_4562_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__10));
v___x_4563_ = l_Lean_stringToMessageData(v___x_4562_);
return v___x_4563_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__13(void){
_start:
{
lean_object* v___x_4565_; lean_object* v___x_4566_; 
v___x_4565_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__12));
v___x_4566_ = l_Lean_stringToMessageData(v___x_4565_);
return v___x_4566_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__15(void){
_start:
{
lean_object* v___x_4568_; lean_object* v___x_4569_; 
v___x_4568_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__14));
v___x_4569_ = l_Lean_stringToMessageData(v___x_4568_);
return v___x_4569_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4(lean_object* v___x_4570_, lean_object* v___y_4571_, lean_object* v___x_4572_, lean_object* v___y_4573_, lean_object* v___y_4574_, lean_object* v___y_4575_, lean_object* v___y_4576_, lean_object* v___y_4577_, lean_object* v___y_4578_, lean_object* v___y_4579_, lean_object* v___y_4580_){
_start:
{
lean_object* v_a_4583_; lean_object* v___y_4601_; lean_object* v___x_4611_; 
v___x_4611_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_4574_, v___y_4577_, v___y_4578_, v___y_4579_, v___y_4580_);
if (lean_obj_tag(v___x_4611_) == 0)
{
lean_object* v_a_4612_; lean_object* v___x_4613_; 
v_a_4612_ = lean_ctor_get(v___x_4611_, 0);
lean_inc(v_a_4612_);
lean_dec_ref_known(v___x_4611_, 1);
v___x_4613_ = l_Lean_MVarId_getType(v_a_4612_, v___y_4577_, v___y_4578_, v___y_4579_, v___y_4580_);
if (lean_obj_tag(v___x_4613_) == 0)
{
lean_object* v_a_4614_; lean_object* v___x_4615_; lean_object* v_a_4616_; lean_object* v___x_4617_; lean_object* v___x_4618_; 
v_a_4614_ = lean_ctor_get(v___x_4613_, 0);
lean_inc(v_a_4614_);
lean_dec_ref_known(v___x_4613_, 1);
v___x_4615_ = lp_batteries_Lean_instantiateMVars___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__1___redArg(v_a_4614_, v___y_4578_);
v_a_4616_ = lean_ctor_get(v___x_4615_, 0);
lean_inc(v_a_4616_);
lean_dec_ref(v___x_4615_);
v___x_4617_ = l_Lean_Expr_cleanupAnnotations(v_a_4616_);
lean_inc_ref(v___x_4617_);
v___x_4618_ = lp_batteries_Batteries_Tactic_getRel(v___x_4617_, v___y_4577_, v___y_4578_, v___y_4579_, v___y_4580_);
if (lean_obj_tag(v___x_4618_) == 0)
{
lean_object* v_a_4619_; lean_object* v___x_4621_; uint8_t v_isShared_4622_; uint8_t v_isSharedCheck_4931_; 
v_a_4619_ = lean_ctor_get(v___x_4618_, 0);
v_isSharedCheck_4931_ = !lean_is_exclusive(v___x_4618_);
if (v_isSharedCheck_4931_ == 0)
{
v___x_4621_ = v___x_4618_;
v_isShared_4622_ = v_isSharedCheck_4931_;
goto v_resetjp_4620_;
}
else
{
lean_inc(v_a_4619_);
lean_dec(v___x_4618_);
v___x_4621_ = lean_box(0);
v_isShared_4622_ = v_isSharedCheck_4931_;
goto v_resetjp_4620_;
}
v_resetjp_4620_:
{
if (lean_obj_tag(v_a_4619_) == 1)
{
lean_object* v_val_4623_; lean_object* v___x_4625_; uint8_t v_isShared_4626_; uint8_t v_isSharedCheck_4924_; 
v_val_4623_ = lean_ctor_get(v_a_4619_, 0);
v_isSharedCheck_4924_ = !lean_is_exclusive(v_a_4619_);
if (v_isSharedCheck_4924_ == 0)
{
v___x_4625_ = v_a_4619_;
v_isShared_4626_ = v_isSharedCheck_4924_;
goto v_resetjp_4624_;
}
else
{
lean_inc(v_val_4623_);
lean_dec(v_a_4619_);
v___x_4625_ = lean_box(0);
v_isShared_4626_ = v_isSharedCheck_4924_;
goto v_resetjp_4624_;
}
v_resetjp_4624_:
{
lean_object* v_snd_4627_; lean_object* v_fst_4628_; lean_object* v___x_4630_; uint8_t v_isShared_4631_; uint8_t v_isSharedCheck_4923_; 
v_snd_4627_ = lean_ctor_get(v_val_4623_, 1);
v_fst_4628_ = lean_ctor_get(v_val_4623_, 0);
v_isSharedCheck_4923_ = !lean_is_exclusive(v_val_4623_);
if (v_isSharedCheck_4923_ == 0)
{
v___x_4630_ = v_val_4623_;
v_isShared_4631_ = v_isSharedCheck_4923_;
goto v_resetjp_4629_;
}
else
{
lean_inc(v_snd_4627_);
lean_inc(v_fst_4628_);
lean_dec(v_val_4623_);
v___x_4630_ = lean_box(0);
v_isShared_4631_ = v_isSharedCheck_4923_;
goto v_resetjp_4629_;
}
v_resetjp_4629_:
{
if (lean_obj_tag(v_fst_4628_) == 0)
{
lean_object* v_fst_4632_; lean_object* v_snd_4633_; lean_object* v___x_4635_; uint8_t v_isShared_4636_; uint8_t v_isSharedCheck_4815_; 
lean_dec(v___x_4572_);
v_fst_4632_ = lean_ctor_get(v_snd_4627_, 0);
v_snd_4633_ = lean_ctor_get(v_snd_4627_, 1);
v_isSharedCheck_4815_ = !lean_is_exclusive(v_snd_4627_);
if (v_isSharedCheck_4815_ == 0)
{
v___x_4635_ = v_snd_4627_;
v_isShared_4636_ = v_isSharedCheck_4815_;
goto v_resetjp_4634_;
}
else
{
lean_inc(v_snd_4633_);
lean_inc(v_fst_4632_);
lean_dec(v_snd_4627_);
v___x_4635_ = lean_box(0);
v_isShared_4636_ = v_isSharedCheck_4815_;
goto v_resetjp_4634_;
}
v_resetjp_4634_:
{
lean_object* v_rel_4637_; lean_object* v___x_4638_; lean_object* v___x_4639_; lean_object* v___y_4641_; lean_object* v___y_4642_; lean_object* v___y_4643_; lean_object* v___y_4644_; lean_object* v___y_4645_; lean_object* v___y_4646_; lean_object* v___y_4647_; lean_object* v___y_4648_; lean_object* v___y_4649_; lean_object* v___y_4650_; uint8_t v___y_4651_; lean_object* v___y_4666_; lean_object* v___y_4667_; lean_object* v___y_4668_; lean_object* v___y_4669_; lean_object* v___y_4670_; lean_object* v___y_4671_; lean_object* v___y_4672_; lean_object* v___y_4673_; lean_object* v___y_4674_; lean_object* v_a_4675_; lean_object* v___y_4679_; lean_object* v___y_4680_; lean_object* v___y_4681_; lean_object* v___y_4682_; lean_object* v___y_4683_; lean_object* v___y_4684_; lean_object* v___y_4685_; lean_object* v___y_4686_; lean_object* v___y_4687_; lean_object* v___y_4688_; lean_object* v___y_4692_; lean_object* v___y_4693_; lean_object* v___y_4694_; lean_object* v___y_4695_; lean_object* v___y_4696_; lean_object* v___y_4697_; lean_object* v___y_4698_; lean_object* v___y_4699_; lean_object* v___y_4700_; lean_object* v___y_4701_; lean_object* v_a_4702_; lean_object* v___y_4717_; lean_object* v___y_4718_; lean_object* v___y_4719_; lean_object* v___y_4720_; lean_object* v___y_4721_; lean_object* v___y_4722_; lean_object* v___y_4723_; lean_object* v___y_4724_; lean_object* v___y_4759_; lean_object* v___y_4760_; lean_object* v___y_4761_; lean_object* v___y_4762_; lean_object* v___y_4763_; lean_object* v___y_4764_; lean_object* v___y_4765_; lean_object* v___y_4766_; lean_object* v___y_4777_; lean_object* v___y_4778_; lean_object* v___y_4779_; lean_object* v___y_4780_; lean_object* v___y_4781_; lean_object* v___y_4782_; lean_object* v___y_4783_; lean_object* v___y_4784_; lean_object* v___y_4795_; lean_object* v___y_4796_; lean_object* v___y_4797_; lean_object* v___y_4798_; lean_object* v___y_4799_; lean_object* v___y_4800_; lean_object* v___y_4801_; lean_object* v___y_4802_; lean_object* v___x_4810_; lean_object* v_a_4811_; uint8_t v___x_4812_; 
v_rel_4637_ = lean_ctor_get(v_fst_4628_, 0);
lean_inc_ref(v_rel_4637_);
lean_dec_ref_known(v_fst_4628_, 1);
v___x_4638_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__1_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_));
v___x_4639_ = l_Lean_Name_mkStr2(v___x_4570_, v___x_4638_);
lean_inc(v___x_4639_);
v___x_4810_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__0(v___x_4639_, v___y_4573_, v___y_4574_, v___y_4575_, v___y_4576_, v___y_4577_, v___y_4578_, v___y_4579_, v___y_4580_);
v_a_4811_ = lean_ctor_get(v___x_4810_, 0);
lean_inc(v_a_4811_);
lean_dec_ref(v___x_4810_);
v___x_4812_ = lean_unbox(v_a_4811_);
lean_dec(v_a_4811_);
if (v___x_4812_ == 0)
{
v___y_4795_ = v___y_4573_;
v___y_4796_ = v___y_4574_;
v___y_4797_ = v___y_4575_;
v___y_4798_ = v___y_4576_;
v___y_4799_ = v___y_4577_;
v___y_4800_ = v___y_4578_;
v___y_4801_ = v___y_4579_;
v___y_4802_ = v___y_4580_;
goto v___jp_4794_;
}
else
{
lean_object* v___x_4813_; lean_object* v___x_4814_; 
v___x_4813_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__9, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__9_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__9);
lean_inc(v___x_4639_);
v___x_4814_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg(v___x_4639_, v___x_4813_, v___y_4577_, v___y_4578_, v___y_4579_, v___y_4580_);
if (lean_obj_tag(v___x_4814_) == 0)
{
lean_dec_ref_known(v___x_4814_, 1);
v___y_4795_ = v___y_4573_;
v___y_4796_ = v___y_4574_;
v___y_4797_ = v___y_4575_;
v___y_4798_ = v___y_4576_;
v___y_4799_ = v___y_4577_;
v___y_4800_ = v___y_4578_;
v___y_4801_ = v___y_4579_;
v___y_4802_ = v___y_4580_;
goto v___jp_4794_;
}
else
{
lean_dec(v___x_4639_);
lean_dec_ref(v_rel_4637_);
lean_del_object(v___x_4635_);
lean_dec(v_snd_4633_);
lean_dec(v_fst_4632_);
lean_del_object(v___x_4630_);
lean_del_object(v___x_4625_);
lean_del_object(v___x_4621_);
lean_dec_ref(v___x_4617_);
lean_dec(v___y_4580_);
lean_dec_ref(v___y_4579_);
lean_dec(v___y_4578_);
lean_dec_ref(v___y_4577_);
lean_dec(v___y_4571_);
return v___x_4814_;
}
}
v___jp_4640_:
{
if (v___y_4651_ == 0)
{
lean_object* v___x_4652_; 
lean_dec_ref(v___y_4641_);
lean_del_object(v___x_4621_);
v___x_4652_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v___y_4645_, v___y_4651_, v___y_4649_, v___y_4647_, v___y_4650_, v___y_4646_, v___y_4642_, v___y_4643_, v___y_4648_);
if (lean_obj_tag(v___x_4652_) == 0)
{
lean_object* v___x_4653_; lean_object* v_a_4654_; uint8_t v___x_4655_; 
lean_dec_ref_known(v___x_4652_, 1);
lean_inc(v___x_4639_);
v___x_4653_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__0(v___x_4639_, v___y_4644_, v___y_4649_, v___y_4647_, v___y_4650_, v___y_4646_, v___y_4642_, v___y_4643_, v___y_4648_);
v_a_4654_ = lean_ctor_get(v___x_4653_, 0);
lean_inc(v_a_4654_);
lean_dec_ref(v___x_4653_);
v___x_4655_ = lean_unbox(v_a_4654_);
lean_dec(v_a_4654_);
if (v___x_4655_ == 0)
{
lean_object* v___x_4656_; lean_object* v___x_4657_; 
lean_dec(v___x_4639_);
v___x_4656_ = lean_box(0);
v___x_4657_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__1(v_rel_4637_, v___x_4638_, v___y_4651_, v_snd_4633_, v_fst_4632_, v___x_4617_, v___y_4571_, v___x_4656_, v___y_4644_, v___y_4649_, v___y_4647_, v___y_4650_, v___y_4646_, v___y_4642_, v___y_4643_, v___y_4648_);
lean_dec(v___y_4648_);
lean_dec_ref(v___y_4643_);
lean_dec(v___y_4642_);
lean_dec_ref(v___y_4646_);
v___y_4601_ = v___x_4657_;
goto v___jp_4600_;
}
else
{
lean_object* v___x_4658_; lean_object* v___x_4659_; 
v___x_4658_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__1, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__1_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__1);
v___x_4659_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg(v___x_4639_, v___x_4658_, v___y_4646_, v___y_4642_, v___y_4643_, v___y_4648_);
if (lean_obj_tag(v___x_4659_) == 0)
{
lean_object* v_a_4660_; lean_object* v___x_4661_; 
v_a_4660_ = lean_ctor_get(v___x_4659_, 0);
lean_inc(v_a_4660_);
lean_dec_ref_known(v___x_4659_, 1);
v___x_4661_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__1(v_rel_4637_, v___x_4638_, v___y_4651_, v_snd_4633_, v_fst_4632_, v___x_4617_, v___y_4571_, v_a_4660_, v___y_4644_, v___y_4649_, v___y_4647_, v___y_4650_, v___y_4646_, v___y_4642_, v___y_4643_, v___y_4648_);
lean_dec(v___y_4648_);
lean_dec_ref(v___y_4643_);
lean_dec(v___y_4642_);
lean_dec_ref(v___y_4646_);
v___y_4601_ = v___x_4661_;
goto v___jp_4600_;
}
else
{
lean_dec(v___y_4648_);
lean_dec_ref(v___y_4646_);
lean_dec_ref(v___y_4643_);
lean_dec(v___y_4642_);
lean_dec_ref(v_rel_4637_);
lean_dec(v_snd_4633_);
lean_dec(v_fst_4632_);
lean_dec_ref(v___x_4617_);
lean_dec(v___y_4571_);
return v___x_4659_;
}
}
}
else
{
lean_dec(v___y_4648_);
lean_dec_ref(v___y_4646_);
lean_dec_ref(v___y_4643_);
lean_dec(v___y_4642_);
lean_dec(v___x_4639_);
lean_dec_ref(v_rel_4637_);
lean_dec(v_snd_4633_);
lean_dec(v_fst_4632_);
lean_dec_ref(v___x_4617_);
lean_dec(v___y_4571_);
return v___x_4652_;
}
}
else
{
lean_object* v___x_4663_; 
lean_dec(v___y_4648_);
lean_dec_ref(v___y_4646_);
lean_dec_ref(v___y_4645_);
lean_dec_ref(v___y_4643_);
lean_dec(v___y_4642_);
lean_dec(v___x_4639_);
lean_dec_ref(v_rel_4637_);
lean_dec(v_snd_4633_);
lean_dec(v_fst_4632_);
lean_dec_ref(v___x_4617_);
lean_dec(v___y_4571_);
if (v_isShared_4622_ == 0)
{
lean_ctor_set_tag(v___x_4621_, 1);
lean_ctor_set(v___x_4621_, 0, v___y_4641_);
v___x_4663_ = v___x_4621_;
goto v_reusejp_4662_;
}
else
{
lean_object* v_reuseFailAlloc_4664_; 
v_reuseFailAlloc_4664_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4664_, 0, v___y_4641_);
v___x_4663_ = v_reuseFailAlloc_4664_;
goto v_reusejp_4662_;
}
v_reusejp_4662_:
{
return v___x_4663_;
}
}
}
v___jp_4665_:
{
uint8_t v___x_4676_; 
v___x_4676_ = l_Lean_Exception_isInterrupt(v_a_4675_);
if (v___x_4676_ == 0)
{
uint8_t v___x_4677_; 
lean_inc_ref(v_a_4675_);
v___x_4677_ = l_Lean_Exception_isRuntime(v_a_4675_);
v___y_4641_ = v_a_4675_;
v___y_4642_ = v___y_4666_;
v___y_4643_ = v___y_4667_;
v___y_4644_ = v___y_4669_;
v___y_4645_ = v___y_4668_;
v___y_4646_ = v___y_4670_;
v___y_4647_ = v___y_4671_;
v___y_4648_ = v___y_4672_;
v___y_4649_ = v___y_4673_;
v___y_4650_ = v___y_4674_;
v___y_4651_ = v___x_4677_;
goto v___jp_4640_;
}
else
{
v___y_4641_ = v_a_4675_;
v___y_4642_ = v___y_4666_;
v___y_4643_ = v___y_4667_;
v___y_4644_ = v___y_4669_;
v___y_4645_ = v___y_4668_;
v___y_4646_ = v___y_4670_;
v___y_4647_ = v___y_4671_;
v___y_4648_ = v___y_4672_;
v___y_4649_ = v___y_4673_;
v___y_4650_ = v___y_4674_;
v___y_4651_ = v___x_4676_;
goto v___jp_4640_;
}
}
v___jp_4678_:
{
if (lean_obj_tag(v___y_4688_) == 0)
{
lean_object* v_a_4689_; 
lean_dec(v___y_4685_);
lean_dec_ref(v___y_4683_);
lean_dec_ref(v___y_4682_);
lean_dec_ref(v___y_4680_);
lean_dec(v___y_4679_);
lean_dec(v___x_4639_);
lean_dec_ref(v_rel_4637_);
lean_dec(v_snd_4633_);
lean_dec(v_fst_4632_);
lean_del_object(v___x_4621_);
lean_dec_ref(v___x_4617_);
lean_dec(v___y_4571_);
v_a_4689_ = lean_ctor_get(v___y_4688_, 0);
lean_inc(v_a_4689_);
lean_dec_ref_known(v___y_4688_, 1);
v_a_4583_ = v_a_4689_;
goto v___jp_4582_;
}
else
{
lean_object* v_a_4690_; 
v_a_4690_ = lean_ctor_get(v___y_4688_, 0);
lean_inc(v_a_4690_);
lean_dec_ref_known(v___y_4688_, 1);
v___y_4666_ = v___y_4679_;
v___y_4667_ = v___y_4680_;
v___y_4668_ = v___y_4682_;
v___y_4669_ = v___y_4681_;
v___y_4670_ = v___y_4683_;
v___y_4671_ = v___y_4684_;
v___y_4672_ = v___y_4685_;
v___y_4673_ = v___y_4686_;
v___y_4674_ = v___y_4687_;
v_a_4675_ = v_a_4690_;
goto v___jp_4665_;
}
}
v___jp_4691_:
{
lean_object* v___x_4703_; 
v___x_4703_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_4700_, v___y_4701_, v___y_4693_, v___y_4699_);
if (lean_obj_tag(v___x_4703_) == 0)
{
lean_object* v_a_4704_; lean_object* v___x_4705_; lean_object* v_a_4706_; uint8_t v___x_4707_; 
v_a_4704_ = lean_ctor_get(v___x_4703_, 0);
lean_inc(v_a_4704_);
lean_dec_ref_known(v___x_4703_, 1);
lean_inc(v___x_4639_);
v___x_4705_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__0(v___x_4639_, v___y_4696_, v___y_4700_, v___y_4698_, v___y_4701_, v___y_4697_, v___y_4693_, v___y_4694_, v___y_4699_);
v_a_4706_ = lean_ctor_get(v___x_4705_, 0);
lean_inc(v_a_4706_);
lean_dec_ref(v___x_4705_);
v___x_4707_ = lean_unbox(v_a_4706_);
lean_dec(v_a_4706_);
if (v___x_4707_ == 0)
{
lean_object* v___x_4708_; lean_object* v___x_4709_; 
v___x_4708_ = lean_box(0);
lean_inc(v_snd_4633_);
lean_inc(v_fst_4632_);
lean_inc_ref(v_rel_4637_);
v___x_4709_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__2(v_rel_4637_, v_fst_4632_, v_snd_4633_, v_a_4702_, v___y_4692_, v_a_4704_, v___x_4708_, v___y_4696_, v___y_4700_, v___y_4698_, v___y_4701_, v___y_4697_, v___y_4693_, v___y_4694_, v___y_4699_);
v___y_4679_ = v___y_4693_;
v___y_4680_ = v___y_4694_;
v___y_4681_ = v___y_4696_;
v___y_4682_ = v___y_4695_;
v___y_4683_ = v___y_4697_;
v___y_4684_ = v___y_4698_;
v___y_4685_ = v___y_4699_;
v___y_4686_ = v___y_4700_;
v___y_4687_ = v___y_4701_;
v___y_4688_ = v___x_4709_;
goto v___jp_4678_;
}
else
{
lean_object* v___x_4710_; lean_object* v___x_4711_; 
v___x_4710_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__3, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__3_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__3);
lean_inc(v___x_4639_);
v___x_4711_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg(v___x_4639_, v___x_4710_, v___y_4697_, v___y_4693_, v___y_4694_, v___y_4699_);
if (lean_obj_tag(v___x_4711_) == 0)
{
lean_object* v_a_4712_; lean_object* v___x_4713_; 
v_a_4712_ = lean_ctor_get(v___x_4711_, 0);
lean_inc(v_a_4712_);
lean_dec_ref_known(v___x_4711_, 1);
lean_inc(v_snd_4633_);
lean_inc(v_fst_4632_);
lean_inc_ref(v_rel_4637_);
v___x_4713_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__2(v_rel_4637_, v_fst_4632_, v_snd_4633_, v_a_4702_, v___y_4692_, v_a_4704_, v_a_4712_, v___y_4696_, v___y_4700_, v___y_4698_, v___y_4701_, v___y_4697_, v___y_4693_, v___y_4694_, v___y_4699_);
v___y_4679_ = v___y_4693_;
v___y_4680_ = v___y_4694_;
v___y_4681_ = v___y_4696_;
v___y_4682_ = v___y_4695_;
v___y_4683_ = v___y_4697_;
v___y_4684_ = v___y_4698_;
v___y_4685_ = v___y_4699_;
v___y_4686_ = v___y_4700_;
v___y_4687_ = v___y_4701_;
v___y_4688_ = v___x_4713_;
goto v___jp_4678_;
}
else
{
lean_object* v_a_4714_; 
lean_dec(v_a_4704_);
lean_dec(v_a_4702_);
lean_dec_ref(v___y_4692_);
v_a_4714_ = lean_ctor_get(v___x_4711_, 0);
lean_inc(v_a_4714_);
lean_dec_ref_known(v___x_4711_, 1);
v___y_4666_ = v___y_4693_;
v___y_4667_ = v___y_4694_;
v___y_4668_ = v___y_4695_;
v___y_4669_ = v___y_4696_;
v___y_4670_ = v___y_4697_;
v___y_4671_ = v___y_4698_;
v___y_4672_ = v___y_4699_;
v___y_4673_ = v___y_4700_;
v___y_4674_ = v___y_4701_;
v_a_4675_ = v_a_4714_;
goto v___jp_4665_;
}
}
}
else
{
lean_object* v_a_4715_; 
lean_dec(v_a_4702_);
lean_dec_ref(v___y_4692_);
v_a_4715_ = lean_ctor_get(v___x_4703_, 0);
lean_inc(v_a_4715_);
lean_dec_ref_known(v___x_4703_, 1);
v___y_4666_ = v___y_4693_;
v___y_4667_ = v___y_4694_;
v___y_4668_ = v___y_4695_;
v___y_4669_ = v___y_4696_;
v___y_4670_ = v___y_4697_;
v___y_4671_ = v___y_4698_;
v___y_4672_ = v___y_4699_;
v___y_4673_ = v___y_4700_;
v___y_4674_ = v___y_4701_;
v_a_4675_ = v_a_4715_;
goto v___jp_4665_;
}
}
v___jp_4716_:
{
lean_object* v___x_4725_; 
v___x_4725_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_4718_, v___y_4720_, v___y_4722_, v___y_4724_);
if (lean_obj_tag(v___x_4725_) == 0)
{
lean_object* v_a_4726_; lean_object* v___x_4727_; 
v_a_4726_ = lean_ctor_get(v___x_4725_, 0);
lean_inc(v_a_4726_);
lean_dec_ref_known(v___x_4725_, 1);
lean_inc(v___y_4724_);
lean_inc_ref(v___y_4723_);
lean_inc(v___y_4722_);
lean_inc_ref(v___y_4721_);
lean_inc(v_fst_4632_);
v___x_4727_ = lean_infer_type(v_fst_4632_, v___y_4721_, v___y_4722_, v___y_4723_, v___y_4724_);
if (lean_obj_tag(v___x_4727_) == 0)
{
lean_object* v_a_4728_; lean_object* v___x_4729_; 
v_a_4728_ = lean_ctor_get(v___x_4727_, 0);
lean_inc(v_a_4728_);
lean_dec_ref_known(v___x_4727_, 1);
v___x_4729_ = l_Lean_Elab_Tactic_getMainTag___redArg(v___y_4718_, v___y_4721_, v___y_4722_, v___y_4723_, v___y_4724_);
if (lean_obj_tag(v___x_4729_) == 0)
{
if (lean_obj_tag(v___y_4571_) == 0)
{
lean_object* v___x_4730_; 
lean_dec_ref_known(v___x_4729_, 1);
lean_del_object(v___x_4625_);
v___x_4730_ = lean_box(0);
v___y_4692_ = v_a_4728_;
v___y_4693_ = v___y_4722_;
v___y_4694_ = v___y_4723_;
v___y_4695_ = v_a_4726_;
v___y_4696_ = v___y_4717_;
v___y_4697_ = v___y_4721_;
v___y_4698_ = v___y_4719_;
v___y_4699_ = v___y_4724_;
v___y_4700_ = v___y_4718_;
v___y_4701_ = v___y_4720_;
v_a_4702_ = v___x_4730_;
goto v___jp_4691_;
}
else
{
lean_object* v_a_4731_; lean_object* v_val_4732_; lean_object* v___x_4734_; 
v_a_4731_ = lean_ctor_get(v___x_4729_, 0);
lean_inc(v_a_4731_);
lean_dec_ref_known(v___x_4729_, 1);
v_val_4732_ = lean_ctor_get(v___y_4571_, 0);
lean_inc(v_a_4728_);
if (v_isShared_4626_ == 0)
{
lean_ctor_set(v___x_4625_, 0, v_a_4728_);
v___x_4734_ = v___x_4625_;
goto v_reusejp_4733_;
}
else
{
lean_object* v_reuseFailAlloc_4747_; 
v_reuseFailAlloc_4747_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4747_, 0, v_a_4728_);
v___x_4734_ = v_reuseFailAlloc_4747_;
goto v_reusejp_4733_;
}
v_reusejp_4733_:
{
uint8_t v___x_4735_; lean_object* v___x_4736_; lean_object* v___x_4737_; 
v___x_4735_ = 0;
v___x_4736_ = lean_box(0);
lean_inc(v_val_4732_);
v___x_4737_ = l_Lean_Elab_Tactic_elabTermWithHoles(v_val_4732_, v___x_4734_, v_a_4731_, v___x_4735_, v___x_4736_, v___y_4717_, v___y_4718_, v___y_4719_, v___y_4720_, v___y_4721_, v___y_4722_, v___y_4723_, v___y_4724_);
if (lean_obj_tag(v___x_4737_) == 0)
{
lean_object* v_a_4738_; lean_object* v___x_4740_; uint8_t v_isShared_4741_; uint8_t v_isSharedCheck_4745_; 
v_a_4738_ = lean_ctor_get(v___x_4737_, 0);
v_isSharedCheck_4745_ = !lean_is_exclusive(v___x_4737_);
if (v_isSharedCheck_4745_ == 0)
{
v___x_4740_ = v___x_4737_;
v_isShared_4741_ = v_isSharedCheck_4745_;
goto v_resetjp_4739_;
}
else
{
lean_inc(v_a_4738_);
lean_dec(v___x_4737_);
v___x_4740_ = lean_box(0);
v_isShared_4741_ = v_isSharedCheck_4745_;
goto v_resetjp_4739_;
}
v_resetjp_4739_:
{
lean_object* v___x_4743_; 
if (v_isShared_4741_ == 0)
{
lean_ctor_set_tag(v___x_4740_, 1);
v___x_4743_ = v___x_4740_;
goto v_reusejp_4742_;
}
else
{
lean_object* v_reuseFailAlloc_4744_; 
v_reuseFailAlloc_4744_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4744_, 0, v_a_4738_);
v___x_4743_ = v_reuseFailAlloc_4744_;
goto v_reusejp_4742_;
}
v_reusejp_4742_:
{
v___y_4692_ = v_a_4728_;
v___y_4693_ = v___y_4722_;
v___y_4694_ = v___y_4723_;
v___y_4695_ = v_a_4726_;
v___y_4696_ = v___y_4717_;
v___y_4697_ = v___y_4721_;
v___y_4698_ = v___y_4719_;
v___y_4699_ = v___y_4724_;
v___y_4700_ = v___y_4718_;
v___y_4701_ = v___y_4720_;
v_a_4702_ = v___x_4743_;
goto v___jp_4691_;
}
}
}
else
{
lean_object* v_a_4746_; 
lean_dec(v_a_4728_);
v_a_4746_ = lean_ctor_get(v___x_4737_, 0);
lean_inc(v_a_4746_);
lean_dec_ref_known(v___x_4737_, 1);
v___y_4666_ = v___y_4722_;
v___y_4667_ = v___y_4723_;
v___y_4668_ = v_a_4726_;
v___y_4669_ = v___y_4717_;
v___y_4670_ = v___y_4721_;
v___y_4671_ = v___y_4719_;
v___y_4672_ = v___y_4724_;
v___y_4673_ = v___y_4718_;
v___y_4674_ = v___y_4720_;
v_a_4675_ = v_a_4746_;
goto v___jp_4665_;
}
}
}
}
else
{
lean_object* v_a_4748_; 
lean_dec(v_a_4728_);
lean_del_object(v___x_4625_);
v_a_4748_ = lean_ctor_get(v___x_4729_, 0);
lean_inc(v_a_4748_);
lean_dec_ref_known(v___x_4729_, 1);
v___y_4666_ = v___y_4722_;
v___y_4667_ = v___y_4723_;
v___y_4668_ = v_a_4726_;
v___y_4669_ = v___y_4717_;
v___y_4670_ = v___y_4721_;
v___y_4671_ = v___y_4719_;
v___y_4672_ = v___y_4724_;
v___y_4673_ = v___y_4718_;
v___y_4674_ = v___y_4720_;
v_a_4675_ = v_a_4748_;
goto v___jp_4665_;
}
}
else
{
lean_object* v_a_4749_; 
lean_del_object(v___x_4625_);
v_a_4749_ = lean_ctor_get(v___x_4727_, 0);
lean_inc(v_a_4749_);
lean_dec_ref_known(v___x_4727_, 1);
v___y_4666_ = v___y_4722_;
v___y_4667_ = v___y_4723_;
v___y_4668_ = v_a_4726_;
v___y_4669_ = v___y_4717_;
v___y_4670_ = v___y_4721_;
v___y_4671_ = v___y_4719_;
v___y_4672_ = v___y_4724_;
v___y_4673_ = v___y_4718_;
v___y_4674_ = v___y_4720_;
v_a_4675_ = v_a_4749_;
goto v___jp_4665_;
}
}
else
{
lean_object* v_a_4750_; lean_object* v___x_4752_; uint8_t v_isShared_4753_; uint8_t v_isSharedCheck_4757_; 
lean_dec(v___y_4724_);
lean_dec_ref(v___y_4723_);
lean_dec(v___y_4722_);
lean_dec_ref(v___y_4721_);
lean_dec(v___x_4639_);
lean_dec_ref(v_rel_4637_);
lean_dec(v_snd_4633_);
lean_dec(v_fst_4632_);
lean_del_object(v___x_4625_);
lean_del_object(v___x_4621_);
lean_dec_ref(v___x_4617_);
lean_dec(v___y_4571_);
v_a_4750_ = lean_ctor_get(v___x_4725_, 0);
v_isSharedCheck_4757_ = !lean_is_exclusive(v___x_4725_);
if (v_isSharedCheck_4757_ == 0)
{
v___x_4752_ = v___x_4725_;
v_isShared_4753_ = v_isSharedCheck_4757_;
goto v_resetjp_4751_;
}
else
{
lean_inc(v_a_4750_);
lean_dec(v___x_4725_);
v___x_4752_ = lean_box(0);
v_isShared_4753_ = v_isSharedCheck_4757_;
goto v_resetjp_4751_;
}
v_resetjp_4751_:
{
lean_object* v___x_4755_; 
if (v_isShared_4753_ == 0)
{
v___x_4755_ = v___x_4752_;
goto v_reusejp_4754_;
}
else
{
lean_object* v_reuseFailAlloc_4756_; 
v_reuseFailAlloc_4756_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4756_, 0, v_a_4750_);
v___x_4755_ = v_reuseFailAlloc_4756_;
goto v_reusejp_4754_;
}
v_reusejp_4754_:
{
return v___x_4755_;
}
}
}
}
v___jp_4758_:
{
lean_object* v___x_4767_; lean_object* v_a_4768_; uint8_t v___x_4769_; 
lean_inc(v___x_4639_);
v___x_4767_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__0(v___x_4639_, v___y_4759_, v___y_4760_, v___y_4761_, v___y_4762_, v___y_4763_, v___y_4764_, v___y_4765_, v___y_4766_);
v_a_4768_ = lean_ctor_get(v___x_4767_, 0);
lean_inc(v_a_4768_);
lean_dec_ref(v___x_4767_);
v___x_4769_ = lean_unbox(v_a_4768_);
lean_dec(v_a_4768_);
if (v___x_4769_ == 0)
{
lean_del_object(v___x_4635_);
v___y_4717_ = v___y_4759_;
v___y_4718_ = v___y_4760_;
v___y_4719_ = v___y_4761_;
v___y_4720_ = v___y_4762_;
v___y_4721_ = v___y_4763_;
v___y_4722_ = v___y_4764_;
v___y_4723_ = v___y_4765_;
v___y_4724_ = v___y_4766_;
goto v___jp_4716_;
}
else
{
lean_object* v___x_4770_; lean_object* v___x_4771_; lean_object* v___x_4773_; 
v___x_4770_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__5, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__5_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__5);
lean_inc(v_snd_4633_);
v___x_4771_ = l_Lean_indentExpr(v_snd_4633_);
if (v_isShared_4636_ == 0)
{
lean_ctor_set_tag(v___x_4635_, 7);
lean_ctor_set(v___x_4635_, 1, v___x_4771_);
lean_ctor_set(v___x_4635_, 0, v___x_4770_);
v___x_4773_ = v___x_4635_;
goto v_reusejp_4772_;
}
else
{
lean_object* v_reuseFailAlloc_4775_; 
v_reuseFailAlloc_4775_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4775_, 0, v___x_4770_);
lean_ctor_set(v_reuseFailAlloc_4775_, 1, v___x_4771_);
v___x_4773_ = v_reuseFailAlloc_4775_;
goto v_reusejp_4772_;
}
v_reusejp_4772_:
{
lean_object* v___x_4774_; 
lean_inc(v___x_4639_);
v___x_4774_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg(v___x_4639_, v___x_4773_, v___y_4763_, v___y_4764_, v___y_4765_, v___y_4766_);
if (lean_obj_tag(v___x_4774_) == 0)
{
lean_dec_ref_known(v___x_4774_, 1);
v___y_4717_ = v___y_4759_;
v___y_4718_ = v___y_4760_;
v___y_4719_ = v___y_4761_;
v___y_4720_ = v___y_4762_;
v___y_4721_ = v___y_4763_;
v___y_4722_ = v___y_4764_;
v___y_4723_ = v___y_4765_;
v___y_4724_ = v___y_4766_;
goto v___jp_4716_;
}
else
{
lean_dec(v___y_4766_);
lean_dec_ref(v___y_4765_);
lean_dec(v___y_4764_);
lean_dec_ref(v___y_4763_);
lean_dec(v___x_4639_);
lean_dec_ref(v_rel_4637_);
lean_dec(v_snd_4633_);
lean_dec(v_fst_4632_);
lean_del_object(v___x_4625_);
lean_del_object(v___x_4621_);
lean_dec_ref(v___x_4617_);
lean_dec(v___y_4571_);
return v___x_4774_;
}
}
}
}
v___jp_4776_:
{
lean_object* v___x_4785_; lean_object* v_a_4786_; uint8_t v___x_4787_; 
lean_inc(v___x_4639_);
v___x_4785_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__0(v___x_4639_, v___y_4777_, v___y_4778_, v___y_4779_, v___y_4780_, v___y_4781_, v___y_4782_, v___y_4783_, v___y_4784_);
v_a_4786_ = lean_ctor_get(v___x_4785_, 0);
lean_inc(v_a_4786_);
lean_dec_ref(v___x_4785_);
v___x_4787_ = lean_unbox(v_a_4786_);
lean_dec(v_a_4786_);
if (v___x_4787_ == 0)
{
lean_del_object(v___x_4630_);
v___y_4759_ = v___y_4777_;
v___y_4760_ = v___y_4778_;
v___y_4761_ = v___y_4779_;
v___y_4762_ = v___y_4780_;
v___y_4763_ = v___y_4781_;
v___y_4764_ = v___y_4782_;
v___y_4765_ = v___y_4783_;
v___y_4766_ = v___y_4784_;
goto v___jp_4758_;
}
else
{
lean_object* v___x_4788_; lean_object* v___x_4789_; lean_object* v___x_4791_; 
v___x_4788_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__7, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__7_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__7);
lean_inc(v_fst_4632_);
v___x_4789_ = l_Lean_indentExpr(v_fst_4632_);
if (v_isShared_4631_ == 0)
{
lean_ctor_set_tag(v___x_4630_, 7);
lean_ctor_set(v___x_4630_, 1, v___x_4789_);
lean_ctor_set(v___x_4630_, 0, v___x_4788_);
v___x_4791_ = v___x_4630_;
goto v_reusejp_4790_;
}
else
{
lean_object* v_reuseFailAlloc_4793_; 
v_reuseFailAlloc_4793_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4793_, 0, v___x_4788_);
lean_ctor_set(v_reuseFailAlloc_4793_, 1, v___x_4789_);
v___x_4791_ = v_reuseFailAlloc_4793_;
goto v_reusejp_4790_;
}
v_reusejp_4790_:
{
lean_object* v___x_4792_; 
lean_inc(v___x_4639_);
v___x_4792_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg(v___x_4639_, v___x_4791_, v___y_4781_, v___y_4782_, v___y_4783_, v___y_4784_);
if (lean_obj_tag(v___x_4792_) == 0)
{
lean_dec_ref_known(v___x_4792_, 1);
v___y_4759_ = v___y_4777_;
v___y_4760_ = v___y_4778_;
v___y_4761_ = v___y_4779_;
v___y_4762_ = v___y_4780_;
v___y_4763_ = v___y_4781_;
v___y_4764_ = v___y_4782_;
v___y_4765_ = v___y_4783_;
v___y_4766_ = v___y_4784_;
goto v___jp_4758_;
}
else
{
lean_dec(v___y_4784_);
lean_dec_ref(v___y_4783_);
lean_dec(v___y_4782_);
lean_dec_ref(v___y_4781_);
lean_dec(v___x_4639_);
lean_dec_ref(v_rel_4637_);
lean_del_object(v___x_4635_);
lean_dec(v_snd_4633_);
lean_dec(v_fst_4632_);
lean_del_object(v___x_4625_);
lean_del_object(v___x_4621_);
lean_dec_ref(v___x_4617_);
lean_dec(v___y_4571_);
return v___x_4792_;
}
}
}
}
v___jp_4794_:
{
lean_object* v___x_4803_; lean_object* v_a_4804_; uint8_t v___x_4805_; 
lean_inc(v___x_4639_);
v___x_4803_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__0(v___x_4639_, v___y_4795_, v___y_4796_, v___y_4797_, v___y_4798_, v___y_4799_, v___y_4800_, v___y_4801_, v___y_4802_);
v_a_4804_ = lean_ctor_get(v___x_4803_, 0);
lean_inc(v_a_4804_);
lean_dec_ref(v___x_4803_);
v___x_4805_ = lean_unbox(v_a_4804_);
lean_dec(v_a_4804_);
if (v___x_4805_ == 0)
{
v___y_4777_ = v___y_4795_;
v___y_4778_ = v___y_4796_;
v___y_4779_ = v___y_4797_;
v___y_4780_ = v___y_4798_;
v___y_4781_ = v___y_4799_;
v___y_4782_ = v___y_4800_;
v___y_4783_ = v___y_4801_;
v___y_4784_ = v___y_4802_;
goto v___jp_4776_;
}
else
{
lean_object* v___x_4806_; lean_object* v___x_4807_; lean_object* v___x_4808_; lean_object* v___x_4809_; 
v___x_4806_ = lean_obj_once(&lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__9, &lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__9_once, _init_lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__7_spec__10___lam__2___closed__9);
lean_inc_ref(v_rel_4637_);
v___x_4807_ = l_Lean_indentExpr(v_rel_4637_);
v___x_4808_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4808_, 0, v___x_4806_);
lean_ctor_set(v___x_4808_, 1, v___x_4807_);
lean_inc(v___x_4639_);
v___x_4809_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg(v___x_4639_, v___x_4808_, v___y_4799_, v___y_4800_, v___y_4801_, v___y_4802_);
if (lean_obj_tag(v___x_4809_) == 0)
{
lean_dec_ref_known(v___x_4809_, 1);
v___y_4777_ = v___y_4795_;
v___y_4778_ = v___y_4796_;
v___y_4779_ = v___y_4797_;
v___y_4780_ = v___y_4798_;
v___y_4781_ = v___y_4799_;
v___y_4782_ = v___y_4800_;
v___y_4783_ = v___y_4801_;
v___y_4784_ = v___y_4802_;
goto v___jp_4776_;
}
else
{
lean_dec(v___y_4802_);
lean_dec_ref(v___y_4801_);
lean_dec(v___y_4800_);
lean_dec_ref(v___y_4799_);
lean_dec(v___x_4639_);
lean_dec_ref(v_rel_4637_);
lean_del_object(v___x_4635_);
lean_dec(v_snd_4633_);
lean_dec(v_fst_4632_);
lean_del_object(v___x_4630_);
lean_del_object(v___x_4625_);
lean_del_object(v___x_4621_);
lean_dec_ref(v___x_4617_);
lean_dec(v___y_4571_);
return v___x_4809_;
}
}
}
}
}
else
{
lean_object* v_fst_4816_; lean_object* v_snd_4817_; lean_object* v___x_4819_; uint8_t v_isShared_4820_; uint8_t v_isSharedCheck_4922_; 
lean_del_object(v___x_4630_);
lean_del_object(v___x_4621_);
lean_dec_ref(v___x_4570_);
v_fst_4816_ = lean_ctor_get(v_snd_4627_, 0);
v_snd_4817_ = lean_ctor_get(v_snd_4627_, 1);
v_isSharedCheck_4922_ = !lean_is_exclusive(v_snd_4627_);
if (v_isSharedCheck_4922_ == 0)
{
v___x_4819_ = v_snd_4627_;
v_isShared_4820_ = v_isSharedCheck_4922_;
goto v_resetjp_4818_;
}
else
{
lean_inc(v_snd_4817_);
lean_inc(v_fst_4816_);
lean_dec(v_snd_4627_);
v___x_4819_ = lean_box(0);
v_isShared_4820_ = v_isSharedCheck_4922_;
goto v_resetjp_4818_;
}
v_resetjp_4818_:
{
lean_object* v_name_4821_; uint8_t v_bi_4822_; lean_object* v___y_4824_; lean_object* v___y_4825_; lean_object* v___y_4826_; lean_object* v___y_4827_; lean_object* v___y_4828_; lean_object* v___y_4829_; lean_object* v___y_4830_; lean_object* v___y_4831_; lean_object* v___y_4832_; lean_object* v___y_4833_; lean_object* v_a_4834_; lean_object* v___y_4848_; lean_object* v___y_4849_; lean_object* v___y_4850_; lean_object* v___y_4851_; lean_object* v___y_4852_; lean_object* v___y_4853_; lean_object* v___y_4854_; lean_object* v___y_4855_; uint8_t v___x_4915_; 
v_name_4821_ = lean_ctor_get(v_fst_4628_, 0);
lean_inc(v_name_4821_);
v_bi_4822_ = lean_ctor_get_uint8(v_fst_4628_, sizeof(void*)*1);
lean_dec_ref_known(v_fst_4628_, 1);
v___x_4915_ = l_Lean_Expr_hasLooseBVars(v_snd_4817_);
if (v___x_4915_ == 0)
{
lean_del_object(v___x_4819_);
lean_dec_ref(v___x_4617_);
v___y_4848_ = v___y_4573_;
v___y_4849_ = v___y_4574_;
v___y_4850_ = v___y_4575_;
v___y_4851_ = v___y_4576_;
v___y_4852_ = v___y_4577_;
v___y_4853_ = v___y_4578_;
v___y_4854_ = v___y_4579_;
v___y_4855_ = v___y_4580_;
goto v___jp_4847_;
}
else
{
lean_object* v___x_4916_; lean_object* v___x_4917_; lean_object* v___x_4919_; 
lean_dec(v_name_4821_);
lean_dec(v_snd_4817_);
lean_dec(v_fst_4816_);
lean_del_object(v___x_4625_);
lean_dec(v___x_4572_);
lean_dec(v___y_4571_);
v___x_4916_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__11, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__11_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__11);
v___x_4917_ = l_Lean_indentExpr(v___x_4617_);
if (v_isShared_4820_ == 0)
{
lean_ctor_set_tag(v___x_4819_, 7);
lean_ctor_set(v___x_4819_, 1, v___x_4917_);
lean_ctor_set(v___x_4819_, 0, v___x_4916_);
v___x_4919_ = v___x_4819_;
goto v_reusejp_4918_;
}
else
{
lean_object* v_reuseFailAlloc_4921_; 
v_reuseFailAlloc_4921_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4921_, 0, v___x_4916_);
lean_ctor_set(v_reuseFailAlloc_4921_, 1, v___x_4917_);
v___x_4919_ = v_reuseFailAlloc_4921_;
goto v_reusejp_4918_;
}
v_reusejp_4918_:
{
lean_object* v___x_4920_; 
v___x_4920_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__8___redArg(v___x_4919_, v___y_4577_, v___y_4578_, v___y_4579_, v___y_4580_);
lean_dec(v___y_4580_);
lean_dec_ref(v___y_4579_);
lean_dec(v___y_4578_);
lean_dec_ref(v___y_4577_);
return v___x_4920_;
}
}
v___jp_4823_:
{
lean_object* v___x_4835_; lean_object* v___f_4836_; lean_object* v___x_4837_; 
v___x_4835_ = lean_box(v_bi_4822_);
v___f_4836_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__3___boxed), 17, 8);
lean_closure_set(v___f_4836_, 0, v_name_4821_);
lean_closure_set(v___f_4836_, 1, v_fst_4816_);
lean_closure_set(v___f_4836_, 2, v_a_4834_);
lean_closure_set(v___f_4836_, 3, v___x_4835_);
lean_closure_set(v___f_4836_, 4, v___y_4827_);
lean_closure_set(v___f_4836_, 5, v_snd_4817_);
lean_closure_set(v___f_4836_, 6, v___x_4572_);
lean_closure_set(v___f_4836_, 7, v___y_4828_);
v___x_4837_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_4836_, v___y_4826_, v___y_4830_, v___y_4833_, v___y_4825_, v___y_4832_, v___y_4829_, v___y_4831_, v___y_4824_);
lean_dec(v___y_4824_);
lean_dec_ref(v___y_4831_);
lean_dec(v___y_4829_);
lean_dec_ref(v___y_4832_);
if (lean_obj_tag(v___x_4837_) == 0)
{
lean_object* v___x_4839_; uint8_t v_isShared_4840_; uint8_t v_isSharedCheck_4845_; 
v_isSharedCheck_4845_ = !lean_is_exclusive(v___x_4837_);
if (v_isSharedCheck_4845_ == 0)
{
lean_object* v_unused_4846_; 
v_unused_4846_ = lean_ctor_get(v___x_4837_, 0);
lean_dec(v_unused_4846_);
v___x_4839_ = v___x_4837_;
v_isShared_4840_ = v_isSharedCheck_4845_;
goto v_resetjp_4838_;
}
else
{
lean_dec(v___x_4837_);
v___x_4839_ = lean_box(0);
v_isShared_4840_ = v_isSharedCheck_4845_;
goto v_resetjp_4838_;
}
v_resetjp_4838_:
{
lean_object* v___x_4841_; lean_object* v___x_4843_; 
v___x_4841_ = lean_box(0);
if (v_isShared_4840_ == 0)
{
lean_ctor_set(v___x_4839_, 0, v___x_4841_);
v___x_4843_ = v___x_4839_;
goto v_reusejp_4842_;
}
else
{
lean_object* v_reuseFailAlloc_4844_; 
v_reuseFailAlloc_4844_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4844_, 0, v___x_4841_);
v___x_4843_ = v_reuseFailAlloc_4844_;
goto v_reusejp_4842_;
}
v_reusejp_4842_:
{
return v___x_4843_;
}
}
}
else
{
return v___x_4837_;
}
}
v___jp_4847_:
{
lean_object* v___x_4856_; uint8_t v___x_4857_; lean_object* v___x_4858_; lean_object* v___x_4859_; 
v___x_4856_ = lean_box(0);
v___x_4857_ = 0;
v___x_4858_ = lean_box(0);
v___x_4859_ = l_Lean_Meta_mkFreshExprMVar(v___x_4856_, v___x_4857_, v___x_4858_, v___y_4852_, v___y_4853_, v___y_4854_, v___y_4855_);
if (lean_obj_tag(v___x_4859_) == 0)
{
lean_object* v_a_4860_; lean_object* v___x_4861_; 
v_a_4860_ = lean_ctor_get(v___x_4859_, 0);
lean_inc(v_a_4860_);
lean_dec_ref_known(v___x_4859_, 1);
v___x_4861_ = l_Lean_Elab_Tactic_getMainTag___redArg(v___y_4849_, v___y_4852_, v___y_4853_, v___y_4854_, v___y_4855_);
if (lean_obj_tag(v___x_4861_) == 0)
{
if (lean_obj_tag(v___y_4571_) == 0)
{
lean_object* v___x_4863_; 
lean_dec_ref_known(v___x_4861_, 1);
if (v_isShared_4626_ == 0)
{
lean_ctor_set(v___x_4625_, 0, v_a_4860_);
v___x_4863_ = v___x_4625_;
goto v_reusejp_4862_;
}
else
{
lean_object* v_reuseFailAlloc_4874_; 
v_reuseFailAlloc_4874_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4874_, 0, v_a_4860_);
v___x_4863_ = v_reuseFailAlloc_4874_;
goto v_reusejp_4862_;
}
v_reusejp_4862_:
{
lean_object* v___x_4864_; 
v___x_4864_ = l_Lean_Meta_mkFreshExprMVar(v___x_4863_, v___x_4857_, v___x_4858_, v___y_4852_, v___y_4853_, v___y_4854_, v___y_4855_);
if (lean_obj_tag(v___x_4864_) == 0)
{
lean_object* v_a_4865_; 
v_a_4865_ = lean_ctor_get(v___x_4864_, 0);
lean_inc(v_a_4865_);
lean_dec_ref_known(v___x_4864_, 1);
v___y_4824_ = v___y_4855_;
v___y_4825_ = v___y_4851_;
v___y_4826_ = v___y_4848_;
v___y_4827_ = v___x_4858_;
v___y_4828_ = v___x_4856_;
v___y_4829_ = v___y_4853_;
v___y_4830_ = v___y_4849_;
v___y_4831_ = v___y_4854_;
v___y_4832_ = v___y_4852_;
v___y_4833_ = v___y_4850_;
v_a_4834_ = v_a_4865_;
goto v___jp_4823_;
}
else
{
lean_object* v_a_4866_; lean_object* v___x_4868_; uint8_t v_isShared_4869_; uint8_t v_isSharedCheck_4873_; 
lean_dec(v___y_4855_);
lean_dec_ref(v___y_4854_);
lean_dec(v___y_4853_);
lean_dec_ref(v___y_4852_);
lean_dec(v_name_4821_);
lean_dec(v_snd_4817_);
lean_dec(v_fst_4816_);
lean_dec(v___x_4572_);
v_a_4866_ = lean_ctor_get(v___x_4864_, 0);
v_isSharedCheck_4873_ = !lean_is_exclusive(v___x_4864_);
if (v_isSharedCheck_4873_ == 0)
{
v___x_4868_ = v___x_4864_;
v_isShared_4869_ = v_isSharedCheck_4873_;
goto v_resetjp_4867_;
}
else
{
lean_inc(v_a_4866_);
lean_dec(v___x_4864_);
v___x_4868_ = lean_box(0);
v_isShared_4869_ = v_isSharedCheck_4873_;
goto v_resetjp_4867_;
}
v_resetjp_4867_:
{
lean_object* v___x_4871_; 
if (v_isShared_4869_ == 0)
{
v___x_4871_ = v___x_4868_;
goto v_reusejp_4870_;
}
else
{
lean_object* v_reuseFailAlloc_4872_; 
v_reuseFailAlloc_4872_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4872_, 0, v_a_4866_);
v___x_4871_ = v_reuseFailAlloc_4872_;
goto v_reusejp_4870_;
}
v_reusejp_4870_:
{
return v___x_4871_;
}
}
}
}
}
else
{
lean_object* v_a_4875_; lean_object* v_val_4876_; lean_object* v___x_4878_; uint8_t v_isShared_4879_; uint8_t v_isSharedCheck_4898_; 
v_a_4875_ = lean_ctor_get(v___x_4861_, 0);
lean_inc(v_a_4875_);
lean_dec_ref_known(v___x_4861_, 1);
v_val_4876_ = lean_ctor_get(v___y_4571_, 0);
v_isSharedCheck_4898_ = !lean_is_exclusive(v___y_4571_);
if (v_isSharedCheck_4898_ == 0)
{
v___x_4878_ = v___y_4571_;
v_isShared_4879_ = v_isSharedCheck_4898_;
goto v_resetjp_4877_;
}
else
{
lean_inc(v_val_4876_);
lean_dec(v___y_4571_);
v___x_4878_ = lean_box(0);
v_isShared_4879_ = v_isSharedCheck_4898_;
goto v_resetjp_4877_;
}
v_resetjp_4877_:
{
lean_object* v___x_4881_; 
if (v_isShared_4879_ == 0)
{
lean_ctor_set(v___x_4878_, 0, v_a_4860_);
v___x_4881_ = v___x_4878_;
goto v_reusejp_4880_;
}
else
{
lean_object* v_reuseFailAlloc_4897_; 
v_reuseFailAlloc_4897_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4897_, 0, v_a_4860_);
v___x_4881_ = v_reuseFailAlloc_4897_;
goto v_reusejp_4880_;
}
v_reusejp_4880_:
{
uint8_t v___x_4882_; lean_object* v___x_4883_; 
v___x_4882_ = 0;
v___x_4883_ = l_Lean_Elab_Tactic_elabTermWithHoles(v_val_4876_, v___x_4881_, v_a_4875_, v___x_4882_, v___x_4856_, v___y_4848_, v___y_4849_, v___y_4850_, v___y_4851_, v___y_4852_, v___y_4853_, v___y_4854_, v___y_4855_);
if (lean_obj_tag(v___x_4883_) == 0)
{
lean_object* v_a_4884_; lean_object* v___x_4886_; 
v_a_4884_ = lean_ctor_get(v___x_4883_, 0);
lean_inc_n(v_a_4884_, 2);
lean_dec_ref_known(v___x_4883_, 1);
if (v_isShared_4626_ == 0)
{
lean_ctor_set(v___x_4625_, 0, v_a_4884_);
v___x_4886_ = v___x_4625_;
goto v_reusejp_4885_;
}
else
{
lean_object* v_reuseFailAlloc_4888_; 
v_reuseFailAlloc_4888_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4888_, 0, v_a_4884_);
v___x_4886_ = v_reuseFailAlloc_4888_;
goto v_reusejp_4885_;
}
v_reusejp_4885_:
{
lean_object* v_fst_4887_; 
v_fst_4887_ = lean_ctor_get(v_a_4884_, 0);
lean_inc(v_fst_4887_);
lean_dec(v_a_4884_);
v___y_4824_ = v___y_4855_;
v___y_4825_ = v___y_4851_;
v___y_4826_ = v___y_4848_;
v___y_4827_ = v___x_4858_;
v___y_4828_ = v___x_4886_;
v___y_4829_ = v___y_4853_;
v___y_4830_ = v___y_4849_;
v___y_4831_ = v___y_4854_;
v___y_4832_ = v___y_4852_;
v___y_4833_ = v___y_4850_;
v_a_4834_ = v_fst_4887_;
goto v___jp_4823_;
}
}
else
{
lean_object* v_a_4889_; lean_object* v___x_4891_; uint8_t v_isShared_4892_; uint8_t v_isSharedCheck_4896_; 
lean_dec(v___y_4855_);
lean_dec_ref(v___y_4854_);
lean_dec(v___y_4853_);
lean_dec_ref(v___y_4852_);
lean_dec(v_name_4821_);
lean_dec(v_snd_4817_);
lean_dec(v_fst_4816_);
lean_del_object(v___x_4625_);
lean_dec(v___x_4572_);
v_a_4889_ = lean_ctor_get(v___x_4883_, 0);
v_isSharedCheck_4896_ = !lean_is_exclusive(v___x_4883_);
if (v_isSharedCheck_4896_ == 0)
{
v___x_4891_ = v___x_4883_;
v_isShared_4892_ = v_isSharedCheck_4896_;
goto v_resetjp_4890_;
}
else
{
lean_inc(v_a_4889_);
lean_dec(v___x_4883_);
v___x_4891_ = lean_box(0);
v_isShared_4892_ = v_isSharedCheck_4896_;
goto v_resetjp_4890_;
}
v_resetjp_4890_:
{
lean_object* v___x_4894_; 
if (v_isShared_4892_ == 0)
{
v___x_4894_ = v___x_4891_;
goto v_reusejp_4893_;
}
else
{
lean_object* v_reuseFailAlloc_4895_; 
v_reuseFailAlloc_4895_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4895_, 0, v_a_4889_);
v___x_4894_ = v_reuseFailAlloc_4895_;
goto v_reusejp_4893_;
}
v_reusejp_4893_:
{
return v___x_4894_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_4899_; lean_object* v___x_4901_; uint8_t v_isShared_4902_; uint8_t v_isSharedCheck_4906_; 
lean_dec(v_a_4860_);
lean_dec(v___y_4855_);
lean_dec_ref(v___y_4854_);
lean_dec(v___y_4853_);
lean_dec_ref(v___y_4852_);
lean_dec(v_name_4821_);
lean_dec(v_snd_4817_);
lean_dec(v_fst_4816_);
lean_del_object(v___x_4625_);
lean_dec(v___x_4572_);
lean_dec(v___y_4571_);
v_a_4899_ = lean_ctor_get(v___x_4861_, 0);
v_isSharedCheck_4906_ = !lean_is_exclusive(v___x_4861_);
if (v_isSharedCheck_4906_ == 0)
{
v___x_4901_ = v___x_4861_;
v_isShared_4902_ = v_isSharedCheck_4906_;
goto v_resetjp_4900_;
}
else
{
lean_inc(v_a_4899_);
lean_dec(v___x_4861_);
v___x_4901_ = lean_box(0);
v_isShared_4902_ = v_isSharedCheck_4906_;
goto v_resetjp_4900_;
}
v_resetjp_4900_:
{
lean_object* v___x_4904_; 
if (v_isShared_4902_ == 0)
{
v___x_4904_ = v___x_4901_;
goto v_reusejp_4903_;
}
else
{
lean_object* v_reuseFailAlloc_4905_; 
v_reuseFailAlloc_4905_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4905_, 0, v_a_4899_);
v___x_4904_ = v_reuseFailAlloc_4905_;
goto v_reusejp_4903_;
}
v_reusejp_4903_:
{
return v___x_4904_;
}
}
}
}
else
{
lean_object* v_a_4907_; lean_object* v___x_4909_; uint8_t v_isShared_4910_; uint8_t v_isSharedCheck_4914_; 
lean_dec(v___y_4855_);
lean_dec_ref(v___y_4854_);
lean_dec(v___y_4853_);
lean_dec_ref(v___y_4852_);
lean_dec(v_name_4821_);
lean_dec(v_snd_4817_);
lean_dec(v_fst_4816_);
lean_del_object(v___x_4625_);
lean_dec(v___x_4572_);
lean_dec(v___y_4571_);
v_a_4907_ = lean_ctor_get(v___x_4859_, 0);
v_isSharedCheck_4914_ = !lean_is_exclusive(v___x_4859_);
if (v_isSharedCheck_4914_ == 0)
{
v___x_4909_ = v___x_4859_;
v_isShared_4910_ = v_isSharedCheck_4914_;
goto v_resetjp_4908_;
}
else
{
lean_inc(v_a_4907_);
lean_dec(v___x_4859_);
v___x_4909_ = lean_box(0);
v_isShared_4910_ = v_isSharedCheck_4914_;
goto v_resetjp_4908_;
}
v_resetjp_4908_:
{
lean_object* v___x_4912_; 
if (v_isShared_4910_ == 0)
{
v___x_4912_ = v___x_4909_;
goto v_reusejp_4911_;
}
else
{
lean_object* v_reuseFailAlloc_4913_; 
v_reuseFailAlloc_4913_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4913_, 0, v_a_4907_);
v___x_4912_ = v_reuseFailAlloc_4913_;
goto v_reusejp_4911_;
}
v_reusejp_4911_:
{
return v___x_4912_;
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
lean_object* v___x_4925_; lean_object* v___x_4926_; lean_object* v___x_4927_; lean_object* v___x_4928_; lean_object* v___x_4929_; lean_object* v___x_4930_; 
lean_del_object(v___x_4621_);
lean_dec(v_a_4619_);
lean_dec(v___x_4572_);
lean_dec(v___y_4571_);
lean_dec_ref(v___x_4570_);
v___x_4925_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__13, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__13_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__13);
v___x_4926_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__15, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__15_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___closed__15);
v___x_4927_ = l_Lean_indentExpr(v___x_4617_);
v___x_4928_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4928_, 0, v___x_4926_);
lean_ctor_set(v___x_4928_, 1, v___x_4927_);
v___x_4929_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4929_, 0, v___x_4925_);
lean_ctor_set(v___x_4929_, 1, v___x_4928_);
v___x_4930_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__8___redArg(v___x_4929_, v___y_4577_, v___y_4578_, v___y_4579_, v___y_4580_);
lean_dec(v___y_4580_);
lean_dec_ref(v___y_4579_);
lean_dec(v___y_4578_);
lean_dec_ref(v___y_4577_);
return v___x_4930_;
}
}
}
else
{
lean_object* v_a_4932_; lean_object* v___x_4934_; uint8_t v_isShared_4935_; uint8_t v_isSharedCheck_4939_; 
lean_dec_ref(v___x_4617_);
lean_dec(v___y_4580_);
lean_dec_ref(v___y_4579_);
lean_dec(v___y_4578_);
lean_dec_ref(v___y_4577_);
lean_dec(v___x_4572_);
lean_dec(v___y_4571_);
lean_dec_ref(v___x_4570_);
v_a_4932_ = lean_ctor_get(v___x_4618_, 0);
v_isSharedCheck_4939_ = !lean_is_exclusive(v___x_4618_);
if (v_isSharedCheck_4939_ == 0)
{
v___x_4934_ = v___x_4618_;
v_isShared_4935_ = v_isSharedCheck_4939_;
goto v_resetjp_4933_;
}
else
{
lean_inc(v_a_4932_);
lean_dec(v___x_4618_);
v___x_4934_ = lean_box(0);
v_isShared_4935_ = v_isSharedCheck_4939_;
goto v_resetjp_4933_;
}
v_resetjp_4933_:
{
lean_object* v___x_4937_; 
if (v_isShared_4935_ == 0)
{
v___x_4937_ = v___x_4934_;
goto v_reusejp_4936_;
}
else
{
lean_object* v_reuseFailAlloc_4938_; 
v_reuseFailAlloc_4938_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4938_, 0, v_a_4932_);
v___x_4937_ = v_reuseFailAlloc_4938_;
goto v_reusejp_4936_;
}
v_reusejp_4936_:
{
return v___x_4937_;
}
}
}
}
else
{
lean_object* v_a_4940_; lean_object* v___x_4942_; uint8_t v_isShared_4943_; uint8_t v_isSharedCheck_4947_; 
lean_dec(v___y_4580_);
lean_dec_ref(v___y_4579_);
lean_dec(v___y_4578_);
lean_dec_ref(v___y_4577_);
lean_dec(v___x_4572_);
lean_dec(v___y_4571_);
lean_dec_ref(v___x_4570_);
v_a_4940_ = lean_ctor_get(v___x_4613_, 0);
v_isSharedCheck_4947_ = !lean_is_exclusive(v___x_4613_);
if (v_isSharedCheck_4947_ == 0)
{
v___x_4942_ = v___x_4613_;
v_isShared_4943_ = v_isSharedCheck_4947_;
goto v_resetjp_4941_;
}
else
{
lean_inc(v_a_4940_);
lean_dec(v___x_4613_);
v___x_4942_ = lean_box(0);
v_isShared_4943_ = v_isSharedCheck_4947_;
goto v_resetjp_4941_;
}
v_resetjp_4941_:
{
lean_object* v___x_4945_; 
if (v_isShared_4943_ == 0)
{
v___x_4945_ = v___x_4942_;
goto v_reusejp_4944_;
}
else
{
lean_object* v_reuseFailAlloc_4946_; 
v_reuseFailAlloc_4946_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4946_, 0, v_a_4940_);
v___x_4945_ = v_reuseFailAlloc_4946_;
goto v_reusejp_4944_;
}
v_reusejp_4944_:
{
return v___x_4945_;
}
}
}
}
else
{
lean_object* v_a_4948_; lean_object* v___x_4950_; uint8_t v_isShared_4951_; uint8_t v_isSharedCheck_4955_; 
lean_dec(v___y_4580_);
lean_dec_ref(v___y_4579_);
lean_dec(v___y_4578_);
lean_dec_ref(v___y_4577_);
lean_dec(v___x_4572_);
lean_dec(v___y_4571_);
lean_dec_ref(v___x_4570_);
v_a_4948_ = lean_ctor_get(v___x_4611_, 0);
v_isSharedCheck_4955_ = !lean_is_exclusive(v___x_4611_);
if (v_isSharedCheck_4955_ == 0)
{
v___x_4950_ = v___x_4611_;
v_isShared_4951_ = v_isSharedCheck_4955_;
goto v_resetjp_4949_;
}
else
{
lean_inc(v_a_4948_);
lean_dec(v___x_4611_);
v___x_4950_ = lean_box(0);
v_isShared_4951_ = v_isSharedCheck_4955_;
goto v_resetjp_4949_;
}
v_resetjp_4949_:
{
lean_object* v___x_4953_; 
if (v_isShared_4951_ == 0)
{
v___x_4953_ = v___x_4950_;
goto v_reusejp_4952_;
}
else
{
lean_object* v_reuseFailAlloc_4954_; 
v_reuseFailAlloc_4954_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4954_, 0, v_a_4948_);
v___x_4953_ = v_reuseFailAlloc_4954_;
goto v_reusejp_4952_;
}
v_reusejp_4952_:
{
return v___x_4953_;
}
}
}
v___jp_4582_:
{
if (lean_obj_tag(v_a_4583_) == 0)
{
lean_object* v_a_4584_; lean_object* v___x_4586_; uint8_t v_isShared_4587_; uint8_t v_isSharedCheck_4591_; 
v_a_4584_ = lean_ctor_get(v_a_4583_, 0);
v_isSharedCheck_4591_ = !lean_is_exclusive(v_a_4583_);
if (v_isSharedCheck_4591_ == 0)
{
v___x_4586_ = v_a_4583_;
v_isShared_4587_ = v_isSharedCheck_4591_;
goto v_resetjp_4585_;
}
else
{
lean_inc(v_a_4584_);
lean_dec(v_a_4583_);
v___x_4586_ = lean_box(0);
v_isShared_4587_ = v_isSharedCheck_4591_;
goto v_resetjp_4585_;
}
v_resetjp_4585_:
{
lean_object* v___x_4589_; 
if (v_isShared_4587_ == 0)
{
v___x_4589_ = v___x_4586_;
goto v_reusejp_4588_;
}
else
{
lean_object* v_reuseFailAlloc_4590_; 
v_reuseFailAlloc_4590_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4590_, 0, v_a_4584_);
v___x_4589_ = v_reuseFailAlloc_4590_;
goto v_reusejp_4588_;
}
v_reusejp_4588_:
{
return v___x_4589_;
}
}
}
else
{
lean_object* v_a_4592_; lean_object* v___x_4594_; uint8_t v_isShared_4595_; uint8_t v_isSharedCheck_4599_; 
v_a_4592_ = lean_ctor_get(v_a_4583_, 0);
v_isSharedCheck_4599_ = !lean_is_exclusive(v_a_4583_);
if (v_isSharedCheck_4599_ == 0)
{
v___x_4594_ = v_a_4583_;
v_isShared_4595_ = v_isSharedCheck_4599_;
goto v_resetjp_4593_;
}
else
{
lean_inc(v_a_4592_);
lean_dec(v_a_4583_);
v___x_4594_ = lean_box(0);
v_isShared_4595_ = v_isSharedCheck_4599_;
goto v_resetjp_4593_;
}
v_resetjp_4593_:
{
lean_object* v___x_4597_; 
if (v_isShared_4595_ == 0)
{
lean_ctor_set_tag(v___x_4594_, 0);
v___x_4597_ = v___x_4594_;
goto v_reusejp_4596_;
}
else
{
lean_object* v_reuseFailAlloc_4598_; 
v_reuseFailAlloc_4598_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4598_, 0, v_a_4592_);
v___x_4597_ = v_reuseFailAlloc_4598_;
goto v_reusejp_4596_;
}
v_reusejp_4596_:
{
return v___x_4597_;
}
}
}
}
v___jp_4600_:
{
if (lean_obj_tag(v___y_4601_) == 0)
{
lean_object* v_a_4602_; 
v_a_4602_ = lean_ctor_get(v___y_4601_, 0);
lean_inc(v_a_4602_);
lean_dec_ref_known(v___y_4601_, 1);
v_a_4583_ = v_a_4602_;
goto v___jp_4582_;
}
else
{
lean_object* v_a_4603_; lean_object* v___x_4605_; uint8_t v_isShared_4606_; uint8_t v_isSharedCheck_4610_; 
v_a_4603_ = lean_ctor_get(v___y_4601_, 0);
v_isSharedCheck_4610_ = !lean_is_exclusive(v___y_4601_);
if (v_isSharedCheck_4610_ == 0)
{
v___x_4605_ = v___y_4601_;
v_isShared_4606_ = v_isSharedCheck_4610_;
goto v_resetjp_4604_;
}
else
{
lean_inc(v_a_4603_);
lean_dec(v___y_4601_);
v___x_4605_ = lean_box(0);
v_isShared_4606_ = v_isSharedCheck_4610_;
goto v_resetjp_4604_;
}
v_resetjp_4604_:
{
lean_object* v___x_4608_; 
if (v_isShared_4606_ == 0)
{
v___x_4608_ = v___x_4605_;
goto v_reusejp_4607_;
}
else
{
lean_object* v_reuseFailAlloc_4609_; 
v_reuseFailAlloc_4609_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4609_, 0, v_a_4603_);
v___x_4608_ = v_reuseFailAlloc_4609_;
goto v_reusejp_4607_;
}
v_reusejp_4607_:
{
return v___x_4608_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___boxed(lean_object* v___x_4956_, lean_object* v___y_4957_, lean_object* v___x_4958_, lean_object* v___y_4959_, lean_object* v___y_4960_, lean_object* v___y_4961_, lean_object* v___y_4962_, lean_object* v___y_4963_, lean_object* v___y_4964_, lean_object* v___y_4965_, lean_object* v___y_4966_, lean_object* v___y_4967_){
_start:
{
lean_object* v_res_4968_; 
v_res_4968_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4(v___x_4956_, v___y_4957_, v___x_4958_, v___y_4959_, v___y_4960_, v___y_4961_, v___y_4962_, v___y_4963_, v___y_4964_, v___y_4965_, v___y_4966_);
lean_dec(v___y_4962_);
lean_dec_ref(v___y_4961_);
lean_dec(v___y_4960_);
lean_dec_ref(v___y_4959_);
return v_res_4968_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1(lean_object* v_x_4969_, lean_object* v_a_4970_, lean_object* v_a_4971_, lean_object* v_a_4972_, lean_object* v_a_4973_, lean_object* v_a_4974_, lean_object* v_a_4975_, lean_object* v_a_4976_, lean_object* v_a_4977_){
_start:
{
lean_object* v___x_4979_; lean_object* v___x_4980_; uint8_t v___x_4981_; 
v___x_4979_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__0_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_));
v___x_4980_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__1));
lean_inc(v_x_4969_);
v___x_4981_ = l_Lean_Syntax_isOfKind(v_x_4969_, v___x_4980_);
if (v___x_4981_ == 0)
{
lean_object* v___x_4982_; 
lean_dec(v_x_4969_);
v___x_4982_ = lp_batteries_Lean_Elab_throwUnsupportedSyntax___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__0___redArg();
return v___x_4982_;
}
else
{
lean_object* v___x_4983_; lean_object* v___y_4985_; lean_object* v___x_4988_; lean_object* v___x_4989_; lean_object* v___x_4990_; 
v___x_4983_ = lean_unsigned_to_nat(0u);
v___x_4988_ = lean_unsigned_to_nat(1u);
v___x_4989_ = l_Lean_Syntax_getArg(v_x_4969_, v___x_4988_);
lean_dec(v_x_4969_);
v___x_4990_ = l_Lean_Syntax_getOptional_x3f(v___x_4989_);
lean_dec(v___x_4989_);
if (lean_obj_tag(v___x_4990_) == 0)
{
lean_object* v___x_4991_; 
v___x_4991_ = lean_box(0);
v___y_4985_ = v___x_4991_;
goto v___jp_4984_;
}
else
{
lean_object* v_val_4992_; lean_object* v___x_4994_; uint8_t v_isShared_4995_; uint8_t v_isSharedCheck_4999_; 
v_val_4992_ = lean_ctor_get(v___x_4990_, 0);
v_isSharedCheck_4999_ = !lean_is_exclusive(v___x_4990_);
if (v_isSharedCheck_4999_ == 0)
{
v___x_4994_ = v___x_4990_;
v_isShared_4995_ = v_isSharedCheck_4999_;
goto v_resetjp_4993_;
}
else
{
lean_inc(v_val_4992_);
lean_dec(v___x_4990_);
v___x_4994_ = lean_box(0);
v_isShared_4995_ = v_isSharedCheck_4999_;
goto v_resetjp_4993_;
}
v_resetjp_4993_:
{
lean_object* v___x_4997_; 
if (v_isShared_4995_ == 0)
{
v___x_4997_ = v___x_4994_;
goto v_reusejp_4996_;
}
else
{
lean_object* v_reuseFailAlloc_4998_; 
v_reuseFailAlloc_4998_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4998_, 0, v_val_4992_);
v___x_4997_ = v_reuseFailAlloc_4998_;
goto v_reusejp_4996_;
}
v_reusejp_4996_:
{
v___y_4985_ = v___x_4997_;
goto v___jp_4984_;
}
}
}
v___jp_4984_:
{
lean_object* v___f_4986_; lean_object* v___x_4987_; 
v___f_4986_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___lam__4___boxed), 12, 3);
lean_closure_set(v___f_4986_, 0, v___x_4979_);
lean_closure_set(v___f_4986_, 1, v___y_4985_);
lean_closure_set(v___f_4986_, 2, v___x_4983_);
v___x_4987_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_4986_, v_a_4970_, v_a_4971_, v_a_4972_, v_a_4973_, v_a_4974_, v_a_4975_, v_a_4976_, v_a_4977_);
return v___x_4987_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1___boxed(lean_object* v_x_5000_, lean_object* v_a_5001_, lean_object* v_a_5002_, lean_object* v_a_5003_, lean_object* v_a_5004_, lean_object* v_a_5005_, lean_object* v_a_5006_, lean_object* v_a_5007_, lean_object* v_a_5008_, lean_object* v_a_5009_){
_start:
{
lean_object* v_res_5010_; 
v_res_5010_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1(v_x_5000_, v_a_5001_, v_a_5002_, v_a_5003_, v_a_5004_, v_a_5005_, v_a_5006_, v_a_5007_, v_a_5008_);
lean_dec(v_a_5008_);
lean_dec_ref(v_a_5007_);
lean_dec(v_a_5006_);
lean_dec_ref(v_a_5005_);
lean_dec(v_a_5004_);
lean_dec_ref(v_a_5003_);
lean_dec(v_a_5002_);
lean_dec_ref(v_a_5001_);
return v_res_5010_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4(lean_object* v_mvarId_5011_, lean_object* v_val_5012_, lean_object* v___y_5013_, lean_object* v___y_5014_, lean_object* v___y_5015_, lean_object* v___y_5016_){
_start:
{
lean_object* v___x_5018_; 
v___x_5018_ = lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4___redArg(v_mvarId_5011_, v_val_5012_, v___y_5014_);
return v___x_5018_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4___boxed(lean_object* v_mvarId_5019_, lean_object* v_val_5020_, lean_object* v___y_5021_, lean_object* v___y_5022_, lean_object* v___y_5023_, lean_object* v___y_5024_, lean_object* v___y_5025_){
_start:
{
lean_object* v_res_5026_; 
v_res_5026_ = lp_batteries_Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4(v_mvarId_5019_, v_val_5020_, v___y_5021_, v___y_5022_, v___y_5023_, v___y_5024_);
lean_dec(v___y_5024_);
lean_dec_ref(v___y_5023_);
lean_dec(v___y_5022_);
lean_dec_ref(v___y_5021_);
return v_res_5026_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6(lean_object* v_cls_5027_, lean_object* v_msg_5028_, lean_object* v___y_5029_, lean_object* v___y_5030_, lean_object* v___y_5031_, lean_object* v___y_5032_, lean_object* v___y_5033_, lean_object* v___y_5034_, lean_object* v___y_5035_, lean_object* v___y_5036_){
_start:
{
lean_object* v___x_5038_; 
v___x_5038_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___redArg(v_cls_5027_, v_msg_5028_, v___y_5033_, v___y_5034_, v___y_5035_, v___y_5036_);
return v___x_5038_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6___boxed(lean_object* v_cls_5039_, lean_object* v_msg_5040_, lean_object* v___y_5041_, lean_object* v___y_5042_, lean_object* v___y_5043_, lean_object* v___y_5044_, lean_object* v___y_5045_, lean_object* v___y_5046_, lean_object* v___y_5047_, lean_object* v___y_5048_, lean_object* v___y_5049_){
_start:
{
lean_object* v_res_5050_; 
v_res_5050_ = lp_batteries_Lean_addTrace___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__6(v_cls_5039_, v_msg_5040_, v___y_5041_, v___y_5042_, v___y_5043_, v___y_5044_, v___y_5045_, v___y_5046_, v___y_5047_, v___y_5048_);
lean_dec(v___y_5048_);
lean_dec_ref(v___y_5047_);
lean_dec(v___y_5046_);
lean_dec_ref(v___y_5045_);
lean_dec(v___y_5044_);
lean_dec_ref(v___y_5043_);
lean_dec(v___y_5042_);
lean_dec_ref(v___y_5041_);
return v_res_5050_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__8(lean_object* v_00_u03b1_5051_, lean_object* v_msg_5052_, lean_object* v___y_5053_, lean_object* v___y_5054_, lean_object* v___y_5055_, lean_object* v___y_5056_, lean_object* v___y_5057_, lean_object* v___y_5058_, lean_object* v___y_5059_, lean_object* v___y_5060_){
_start:
{
lean_object* v___x_5062_; 
v___x_5062_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__8___redArg(v_msg_5052_, v___y_5057_, v___y_5058_, v___y_5059_, v___y_5060_);
return v___x_5062_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__8___boxed(lean_object* v_00_u03b1_5063_, lean_object* v_msg_5064_, lean_object* v___y_5065_, lean_object* v___y_5066_, lean_object* v___y_5067_, lean_object* v___y_5068_, lean_object* v___y_5069_, lean_object* v___y_5070_, lean_object* v___y_5071_, lean_object* v___y_5072_, lean_object* v___y_5073_){
_start:
{
lean_object* v_res_5074_; 
v_res_5074_ = lp_batteries_Lean_throwError___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__8(v_00_u03b1_5063_, v_msg_5064_, v___y_5065_, v___y_5066_, v___y_5067_, v___y_5068_, v___y_5069_, v___y_5070_, v___y_5071_, v___y_5072_);
lean_dec(v___y_5072_);
lean_dec_ref(v___y_5071_);
lean_dec(v___y_5070_);
lean_dec_ref(v___y_5069_);
lean_dec(v___y_5068_);
lean_dec_ref(v___y_5067_);
lean_dec(v___y_5066_);
lean_dec_ref(v___y_5065_);
return v_res_5074_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6(lean_object* v_00_u03b2_5075_, lean_object* v_x_5076_, lean_object* v_x_5077_, lean_object* v_x_5078_){
_start:
{
lean_object* v___x_5079_; 
v___x_5079_ = lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6___redArg(v_x_5076_, v_x_5077_, v_x_5078_);
return v___x_5079_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7(lean_object* v_00_u03b2_5080_, lean_object* v_x_5081_, size_t v_x_5082_, size_t v_x_5083_, lean_object* v_x_5084_, lean_object* v_x_5085_){
_start:
{
lean_object* v___x_5086_; 
v___x_5086_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7___redArg(v_x_5081_, v_x_5082_, v_x_5083_, v_x_5084_, v_x_5085_);
return v___x_5086_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7___boxed(lean_object* v_00_u03b2_5087_, lean_object* v_x_5088_, lean_object* v_x_5089_, lean_object* v_x_5090_, lean_object* v_x_5091_, lean_object* v_x_5092_){
_start:
{
size_t v_x_97389__boxed_5093_; size_t v_x_97390__boxed_5094_; lean_object* v_res_5095_; 
v_x_97389__boxed_5093_ = lean_unbox_usize(v_x_5089_);
lean_dec(v_x_5089_);
v_x_97390__boxed_5094_ = lean_unbox_usize(v_x_5090_);
lean_dec(v_x_5090_);
v_res_5095_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7(v_00_u03b2_5087_, v_x_5088_, v_x_97389__boxed_5093_, v_x_97390__boxed_5094_, v_x_5091_, v_x_5092_);
return v_res_5095_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7_spec__13(lean_object* v_00_u03b2_5096_, lean_object* v_n_5097_, lean_object* v_k_5098_, lean_object* v_v_5099_){
_start:
{
lean_object* v___x_5100_; 
v___x_5100_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7_spec__13___redArg(v_n_5097_, v_k_5098_, v_v_5099_);
return v___x_5100_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7_spec__14(lean_object* v_00_u03b2_5101_, size_t v_depth_5102_, lean_object* v_keys_5103_, lean_object* v_vals_5104_, lean_object* v_heq_5105_, lean_object* v_i_5106_, lean_object* v_entries_5107_){
_start:
{
lean_object* v___x_5108_; 
v___x_5108_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7_spec__14___redArg(v_depth_5102_, v_keys_5103_, v_vals_5104_, v_i_5106_, v_entries_5107_);
return v___x_5108_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7_spec__14___boxed(lean_object* v_00_u03b2_5109_, lean_object* v_depth_5110_, lean_object* v_keys_5111_, lean_object* v_vals_5112_, lean_object* v_heq_5113_, lean_object* v_i_5114_, lean_object* v_entries_5115_){
_start:
{
size_t v_depth_boxed_5116_; lean_object* v_res_5117_; 
v_depth_boxed_5116_ = lean_unbox_usize(v_depth_5110_);
lean_dec(v_depth_5110_);
v_res_5117_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7_spec__14(v_00_u03b2_5109_, v_depth_boxed_5116_, v_keys_5111_, v_vals_5112_, v_heq_5113_, v_i_5114_, v_entries_5115_);
lean_dec_ref(v_vals_5112_);
lean_dec_ref(v_keys_5111_);
return v_res_5117_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7_spec__13_spec__15(lean_object* v_00_u03b2_5118_, lean_object* v_x_5119_, lean_object* v_x_5120_, lean_object* v_x_5121_, lean_object* v_x_5122_){
_start:
{
lean_object* v___x_5123_; 
v___x_5123_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Batteries_Tactic___aux__Batteries__Tactic__Trans______elabRules__Batteries__Tactic__tacticTrans________1_spec__4_spec__6_spec__7_spec__13_spec__15___redArg(v_x_5119_, v_x_5120_, v_x_5121_, v_x_5122_);
return v___x_5123_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______macroRules__Batteries__Tactic__tacticTransitivity________1___closed__2(void){
_start:
{
lean_object* v___x_5145_; 
v___x_5145_ = l_Array_mkArray0(lean_box(0));
return v___x_5145_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______macroRules__Batteries__Tactic__tacticTransitivity________1(lean_object* v_x_5146_, lean_object* v_a_5147_, lean_object* v_a_5148_){
_start:
{
lean_object* v___x_5149_; uint8_t v___x_5150_; 
v___x_5149_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tacticTransitivity_______00__closed__1));
lean_inc(v_x_5146_);
v___x_5150_ = l_Lean_Syntax_isOfKind(v_x_5146_, v___x_5149_);
if (v___x_5150_ == 0)
{
lean_object* v___x_5151_; lean_object* v___x_5152_; 
lean_dec(v_x_5146_);
v___x_5151_ = lean_box(1);
v___x_5152_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5152_, 0, v___x_5151_);
lean_ctor_set(v___x_5152_, 1, v_a_5148_);
return v___x_5152_;
}
else
{
lean_object* v___x_5153_; lean_object* v___x_5154_; lean_object* v___x_5155_; uint8_t v___x_5156_; 
v___x_5153_ = lean_unsigned_to_nat(0u);
v___x_5154_ = lean_unsigned_to_nat(1u);
v___x_5155_ = l_Lean_Syntax_getArg(v_x_5146_, v___x_5154_);
lean_dec(v_x_5146_);
lean_inc(v___x_5155_);
v___x_5156_ = l_Lean_Syntax_matchesNull(v___x_5155_, v___x_5153_);
if (v___x_5156_ == 0)
{
uint8_t v___x_5157_; 
lean_inc(v___x_5155_);
v___x_5157_ = l_Lean_Syntax_matchesNull(v___x_5155_, v___x_5154_);
if (v___x_5157_ == 0)
{
lean_object* v___x_5158_; lean_object* v___x_5159_; 
lean_dec(v___x_5155_);
v___x_5158_ = lean_box(1);
v___x_5159_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5159_, 0, v___x_5158_);
lean_ctor_set(v___x_5159_, 1, v_a_5148_);
return v___x_5159_;
}
else
{
lean_object* v_ref_5160_; lean_object* v___x_5161_; lean_object* v___x_5162_; lean_object* v___x_5163_; lean_object* v___x_5164_; lean_object* v___x_5165_; lean_object* v___x_5166_; lean_object* v___x_5167_; lean_object* v___x_5168_; lean_object* v___x_5169_; 
v_ref_5160_ = lean_ctor_get(v_a_5147_, 5);
v___x_5161_ = l_Lean_Syntax_getArg(v___x_5155_, v___x_5153_);
lean_dec(v___x_5155_);
v___x_5162_ = l_Lean_SourceInfo_fromRef(v_ref_5160_, v___x_5156_);
v___x_5163_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__1));
v___x_5164_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__1_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_));
lean_inc_n(v___x_5162_, 2);
v___x_5165_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5165_, 0, v___x_5162_);
lean_ctor_set(v___x_5165_, 1, v___x_5164_);
v___x_5166_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______macroRules__Batteries__Tactic__tacticTransitivity________1___closed__1));
v___x_5167_ = l_Lean_Syntax_node1(v___x_5162_, v___x_5166_, v___x_5161_);
v___x_5168_ = l_Lean_Syntax_node2(v___x_5162_, v___x_5163_, v___x_5165_, v___x_5167_);
v___x_5169_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5169_, 0, v___x_5168_);
lean_ctor_set(v___x_5169_, 1, v_a_5148_);
return v___x_5169_;
}
}
else
{
lean_object* v_ref_5170_; uint8_t v___x_5171_; lean_object* v___x_5172_; lean_object* v___x_5173_; lean_object* v___x_5174_; lean_object* v___x_5175_; lean_object* v___x_5176_; lean_object* v___x_5177_; lean_object* v___x_5178_; lean_object* v___x_5179_; lean_object* v___x_5180_; 
lean_dec(v___x_5155_);
v_ref_5170_ = lean_ctor_get(v_a_5147_, 5);
v___x_5171_ = 0;
v___x_5172_ = l_Lean_SourceInfo_fromRef(v_ref_5170_, v___x_5171_);
v___x_5173_ = ((lean_object*)(lp_batteries_Batteries_Tactic_tacticTrans_______00__closed__1));
v___x_5174_ = ((lean_object*)(lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn___closed__1_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_));
lean_inc_n(v___x_5172_, 2);
v___x_5175_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5175_, 0, v___x_5172_);
lean_ctor_set(v___x_5175_, 1, v___x_5174_);
v___x_5176_ = ((lean_object*)(lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______macroRules__Batteries__Tactic__tacticTransitivity________1___closed__1));
v___x_5177_ = lean_obj_once(&lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______macroRules__Batteries__Tactic__tacticTransitivity________1___closed__2, &lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______macroRules__Batteries__Tactic__tacticTransitivity________1___closed__2_once, _init_lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______macroRules__Batteries__Tactic__tacticTransitivity________1___closed__2);
v___x_5178_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_5178_, 0, v___x_5172_);
lean_ctor_set(v___x_5178_, 1, v___x_5176_);
lean_ctor_set(v___x_5178_, 2, v___x_5177_);
v___x_5179_ = l_Lean_Syntax_node2(v___x_5172_, v___x_5173_, v___x_5175_, v___x_5178_);
v___x_5180_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5180_, 0, v___x_5179_);
lean_ctor_set(v___x_5180_, 1, v_a_5148_);
return v___x_5180_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______macroRules__Batteries__Tactic__tacticTransitivity________1___boxed(lean_object* v_x_5181_, lean_object* v_a_5182_, lean_object* v_a_5183_){
_start:
{
lean_object* v_res_5184_; 
v_res_5184_ = lp_batteries_Batteries_Tactic___aux__Batteries__Tactic__Trans______macroRules__Batteries__Tactic__tacticTransitivity________1(v_x_5181_, v_a_5182_, v_a_5183_);
lean_dec_ref(v_a_5182_);
return v_res_5184_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Tactic_Trans(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_Tactic_ElabTerm(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Tactic_Trans(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_ElabTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2429574326____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_3788219280____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_batteries_Batteries_Tactic_transExt = lean_io_result_get_value(res);
lean_mark_persistent(lp_batteries_Batteries_Tactic_transExt);
lean_dec_ref(res);
res = lp_batteries___private_Batteries_Tactic_Trans_0__Batteries_Tactic_initFn_00___x40_Batteries_Tactic_Trans_2247956323____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_ElabTerm(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Tactic_Trans(uint8_t builtin) {
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
res = runtime_initialize_batteries_Batteries_Tactic_Trans(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Tactic_Trans(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Tactic_Trans(builtin);
}
#ifdef __cplusplus
}
#endif
