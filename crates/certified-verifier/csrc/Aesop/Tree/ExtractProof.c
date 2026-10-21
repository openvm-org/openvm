// Lean compiler output
// Module: Aesop.Tree.ExtractProof
// Imports: public import Init public meta import Init public import Aesop.Tree.TreeM import Batteries.Lean.Meta.SavedState
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
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint8_t lp_batteries_Lean_MetavarContext_isExprMVarDeclared(lean_object*, lean_object*);
lean_object* l_Lean_MetavarContext_getExprAssignmentCore_x3f(lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l_Lean_MetavarContext_getDelayedMVarAssignmentCore_x3f(lean_object*, lean_object*);
lean_object* lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_getMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
extern lean_object* lp_aesop_Aesop_TraceOption_extraction;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* lean_expr_dbg_to_string(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* l_Lean_mkMVar(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
lean_object* l_Lean_MetavarContext_getDecl(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l_Lean_instInhabitedPersistentArrayNode_default(lean_object*);
size_t lean_usize_shift_left(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
uint8_t l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(lean_object*, lean_object*);
lean_object* l_Lean_LocalContext_mkAuxDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedLocalContext_default;
lean_object* l_instInhabitedOfMonad___redArg(lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_LocalContext_mkLocalDecl(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t);
lean_object* l_Lean_LocalContext_mkLetDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_sharecommon_quick(lean_object*);
lean_object* l_Lean_MVarId_getDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getMVarDependencies(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_indentD(lean_object*);
extern lean_object* lp_aesop_Aesop_treeImpl;
uint8_t lp_aesop_Aesop_NodeState_isProven(uint8_t);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Environment_replayConsts(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lp_aesop_Aesop_Goal_safeRapps(lean_object*);
uint8_t lp_aesop_Aesop_GoalState_isProven(uint8_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lp_aesop_Aesop_getRootGoal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__0_value;
static const lean_ctor_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__1_value;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Aesop"};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__2_value;
static const lean_ctor_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__1_value),((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(213, 96, 250, 13, 195, 1, 48, 100)}};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__3_value;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Tree"};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__4_value;
static const lean_ctor_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__3_value),((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(9, 203, 185, 142, 106, 31, 71, 113)}};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__5 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__5_value;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "ExtractProof"};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__6 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__6_value;
static const lean_ctor_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__5_value),((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(20, 211, 199, 56, 234, 38, 204, 235)}};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__7 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__7_value;
static const lean_ctor_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(21, 118, 128, 133, 218, 97, 143, 12)}};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__8 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__8_value;
static const lean_ctor_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__8_value),((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(175, 186, 249, 124, 212, 216, 144, 97)}};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__9 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__9_value;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "termThrowPRError_"};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__10 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__10_value;
static const lean_ctor_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__9_value),((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__10_value),LEAN_SCALAR_PTR_LITERAL(102, 36, 143, 226, 187, 70, 72, 43)}};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__11 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__11_value;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__12 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__12_value;
static const lean_ctor_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__12_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__13 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__13_value;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "throwPRError "};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__14 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__14_value;
static const lean_ctor_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__14_value)}};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__15 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__15_value;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "interpolatedStr"};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__16 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__16_value;
static const lean_ctor_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__16_value),LEAN_SCALAR_PTR_LITERAL(156, 58, 177, 246, 99, 11, 16, 252)}};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__17 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__17_value;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__18 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__18_value;
static const lean_ctor_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__18_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__19 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__19_value;
static const lean_ctor_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__19_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__20 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__20_value;
static const lean_ctor_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__17_value),((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__20_value)}};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__21 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__21_value;
static const lean_ctor_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__13_value),((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__15_value),((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__21_value)}};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__22 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__22_value;
static const lean_ctor_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__11_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__22_value)}};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__23 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__23_value;
LEAN_EXPORT const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError__ = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__23_value;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "interpolatedStrKind"};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__0_value;
static const lean_ctor_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(239, 118, 32, 248, 73, 51, 110, 198)}};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__1_value;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__2_value;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "termThrowError__"};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__3_value;
static const lean_ctor_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__4_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(225, 45, 105, 121, 242, 5, 105, 46)}};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__4_value;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "throwError"};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__5 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__5_value;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "term_++_"};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__6 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__6_value;
static const lean_ctor_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(90, 69, 86, 178, 149, 48, 216, 23)}};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__7 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__7_value;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "termM!_"};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__8 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__8_value;
static const lean_ctor_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__9_value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(241, 254, 249, 246, 41, 222, 210, 184)}};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__9 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__9_value;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "m!"};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__10 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__10_value;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "interpolatedStrLitKind"};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__11 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__11_value;
static const lean_ctor_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(216, 181, 130, 246, 88, 58, 26, 43)}};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__12 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__12_value;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "\"aesop: internal error during proof reconstruction: \""};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__13 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__13_value;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "++"};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__14 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__14_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___redArg___closed__0;
static lean_once_cell_t lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___redArg___closed__1;
static lean_once_cell_t lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___redArg___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_getExprMVarAssignment_x3f___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_getExprMVarAssignment_x3f___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_getExprMVarAssignment_x3f___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_getExprMVarAssignment_x3f___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_getDelayedMVarAssignment_x3f___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_getDelayedMVarAssignment_x3f___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_getDelayedMVarAssignment_x3f___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_getDelayedMVarAssignment_x3f___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isDeclared___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__10___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isDeclared___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isDeclared___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isDeclared___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__4_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__4_spec__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6___closed__0;
static const lean_string_object lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6___closed__1 = (const lean_object*)&lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6___closed__1_value;
static const lean_array_object lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6___closed__2 = (const lean_object*)&lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__25___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__25___redArg___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26___closed__0;
static const lean_closure_object lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26___closed__1 = (const lean_object*)&lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26___closed__1_value;
static const lean_closure_object lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26___closed__2 = (const lean_object*)&lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26___closed__2_value;
static const lean_closure_object lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26___closed__3 = (const lean_object*)&lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26___closed__3_value;
static const lean_closure_object lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26___closed__4 = (const lean_object*)&lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26___closed__4_value;
LEAN_EXPORT lean_object* lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "Lean.MetavarContext"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34___closed__0_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Lean.instantiateLCtxMVars"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34___closed__1 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34___closed__1_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 55, .m_capacity = 55, .m_length = 54, .m_data = "Invalid auxiliary declaration found in local context: "};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34___closed__2 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34___closed__2_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 40, .m_capacity = 40, .m_length = 39, .m_data = " does not have an associated full name."};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34___closed__3 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34___closed__3_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__35(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__33_spec__35(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__33_spec__35___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__35___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__33___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__33___closed__0;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__33(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__33___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__0;
static lean_once_cell_t lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__1;
static lean_once_cell_t lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__2;
static lean_once_cell_t lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__3;
static lean_once_cell_t lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__4;
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17_spec__22_spec__30___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17_spec__22___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17_spec__23___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17_spec__23___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__17___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__17___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16_spec__21_spec__24___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16_spec__21_spec__24___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16_spec__21___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16_spec__21___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "unknown metavariable '\?"};
static const lean_object* lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14___closed__0 = (const lean_object*)&lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14___closed__0_value;
static lean_once_cell_t lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14___closed__1;
static const lean_string_object lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14___closed__2 = (const lean_object*)&lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14___closed__2_value;
static lean_once_cell_t lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "declare \?"};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1___closed__1;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1___closed__2_value;
static lean_once_cell_t lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1___closed__3;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_assignDelayedMVar___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_assignDelayedMVar___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11_spec__13_spec__18___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11_spec__13_spec__18___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11_spec__13___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11_spec__13___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "assign  \?"};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__1;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__2_value;
static lean_once_cell_t lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__3;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "dassign \?"};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__4_value;
static lean_once_cell_t lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__5;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " => "};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__6 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__6_value;
static lean_once_cell_t lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__7;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__14(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_assignDelayedMVar___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_assignDelayedMVar___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11_spec__13(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__17(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__17___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17_spec__22(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17_spec__23(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17_spec__23___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11_spec__13_spec__18(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11_spec__13_spec__18___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16_spec__21(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16_spec__21___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__25(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__25___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17_spec__22_spec__30(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16_spec__21_spec__24(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16_spec__21_spec__24___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 52, .m_capacity = 52, .m_length = 51, .m_data = "aesop: internal error during proof reconstruction: "};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__1;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "goal "};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__2_value;
static lean_once_cell_t lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__3;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = " was not normalised."};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__4_value;
static lean_once_cell_t lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__5;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "visiting G"};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__6 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__6_value;
static lean_once_cell_t lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__7;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitRapp___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "visiting R"};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitRapp___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitRapp___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitRapp___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitRapp___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitRapp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitRapp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__4___redArg(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__6___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__6___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__6___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__6___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal_spec__0___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 66, .m_capacity = 66, .m_length = 65, .m_data = "an mvar cluster does not contain a proven goal (candidate goals: "};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster___closed__1;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ")."};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster___closed__2_value;
static lean_once_cell_t lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster___closed__3;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = " does not have a proven rapp."};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofRapp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofRapp_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofRapp_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofRapp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__4(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__6(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixMVarCluster(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixRapp_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixRapp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "aesop: internal error: goal "};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal___closed__1;
static const lean_string_object lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = " has multiple safe rapps"};
static const lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal___closed__2_value;
static lean_once_cell_t lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal___closed__3;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixMVarCluster_spec__4(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixMVarCluster_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixRapp_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixMVarCluster___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixRapp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_extractProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_extractProof___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_extractProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_extractProof___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_Goal_extractSafePrefix___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_Goal_extractSafePrefix___closed__0 = (const lean_object*)&lp_aesop_Aesop_Goal_extractSafePrefix___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_extractSafePrefix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_extractSafePrefix___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_extractSafePrefix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_extractSafePrefix___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1(lean_object* v_x_76_, lean_object* v_a_77_, lean_object* v_a_78_){
_start:
{
lean_object* v___x_79_; uint8_t v___x_80_; 
v___x_79_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_termThrowPRError___00__closed__11));
lean_inc(v_x_76_);
v___x_80_ = l_Lean_Syntax_isOfKind(v_x_76_, v___x_79_);
if (v___x_80_ == 0)
{
lean_object* v___x_81_; lean_object* v___x_82_; 
lean_dec(v_x_76_);
v___x_81_ = lean_box(1);
v___x_82_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_82_, 0, v___x_81_);
lean_ctor_set(v___x_82_, 1, v_a_78_);
return v___x_82_;
}
else
{
lean_object* v_ref_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; uint8_t v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; 
v_ref_83_ = lean_ctor_get(v_a_77_, 5);
v___x_84_ = lean_unsigned_to_nat(1u);
v___x_85_ = l_Lean_Syntax_getArg(v_x_76_, v___x_84_);
lean_dec(v_x_76_);
v___x_86_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__1));
v___x_87_ = 0;
v___x_88_ = l_Lean_SourceInfo_fromRef(v_ref_83_, v___x_87_);
v___x_89_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__4));
v___x_90_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__5));
lean_inc_n(v___x_88_, 9);
v___x_91_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_91_, 0, v___x_88_);
lean_ctor_set(v___x_91_, 1, v___x_90_);
v___x_92_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__7));
v___x_93_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__9));
v___x_94_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__10));
v___x_95_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_95_, 0, v___x_88_);
lean_ctor_set(v___x_95_, 1, v___x_94_);
v___x_96_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__12));
v___x_97_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__13));
v___x_98_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_98_, 0, v___x_88_);
lean_ctor_set(v___x_98_, 1, v___x_97_);
v___x_99_ = l_Lean_Syntax_node1(v___x_88_, v___x_96_, v___x_98_);
v___x_100_ = l_Lean_Syntax_node1(v___x_88_, v___x_86_, v___x_99_);
lean_inc_ref(v___x_95_);
v___x_101_ = l_Lean_Syntax_node2(v___x_88_, v___x_93_, v___x_95_, v___x_100_);
v___x_102_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___closed__14));
v___x_103_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_103_, 0, v___x_88_);
lean_ctor_set(v___x_103_, 1, v___x_102_);
v___x_104_ = l_Lean_Syntax_node2(v___x_88_, v___x_93_, v___x_95_, v___x_85_);
v___x_105_ = l_Lean_Syntax_node3(v___x_88_, v___x_92_, v___x_101_, v___x_103_, v___x_104_);
v___x_106_ = l_Lean_Syntax_node2(v___x_88_, v___x_89_, v___x_91_, v___x_105_);
v___x_107_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_107_, 0, v___x_106_);
lean_ctor_set(v___x_107_, 1, v_a_78_);
return v___x_107_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1___boxed(lean_object* v_x_108_, lean_object* v_a_109_, lean_object* v_a_110_){
_start:
{
lean_object* v_res_111_; 
v_res_111_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop___aux__Aesop__Tree__ExtractProof______macroRules____private__Aesop__Tree__ExtractProof__0__Aesop__termThrowPRError____1(v_x_108_, v_a_109_, v_a_110_);
lean_dec_ref(v_a_109_);
return v_res_111_;
}
}
static lean_object* _init_lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_112_; 
v___x_112_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_112_;
}
}
static lean_object* _init_lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___redArg___closed__1(void){
_start:
{
lean_object* v___x_113_; lean_object* v___x_114_; 
v___x_113_ = lean_obj_once(&lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___redArg___closed__0, &lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___redArg___closed__0_once, _init_lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___redArg___closed__0);
v___x_114_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_114_, 0, v___x_113_);
return v___x_114_;
}
}
static lean_object* _init_lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___redArg___closed__2(void){
_start:
{
lean_object* v___x_115_; lean_object* v___x_116_; 
v___x_115_ = lean_obj_once(&lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___redArg___closed__1, &lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___redArg___closed__1_once, _init_lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___redArg___closed__1);
v___x_116_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_116_, 0, v___x_115_);
lean_ctor_set(v___x_116_, 1, v___x_115_);
return v___x_116_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___redArg(lean_object* v_env_117_, lean_object* v___y_118_){
_start:
{
lean_object* v___x_120_; lean_object* v_nextMacroScope_121_; lean_object* v_ngen_122_; lean_object* v_auxDeclNGen_123_; lean_object* v_traceState_124_; lean_object* v_messages_125_; lean_object* v_infoState_126_; lean_object* v_snapshotTasks_127_; lean_object* v___x_129_; uint8_t v_isShared_130_; uint8_t v_isSharedCheck_138_; 
v___x_120_ = lean_st_ref_take(v___y_118_);
v_nextMacroScope_121_ = lean_ctor_get(v___x_120_, 1);
v_ngen_122_ = lean_ctor_get(v___x_120_, 2);
v_auxDeclNGen_123_ = lean_ctor_get(v___x_120_, 3);
v_traceState_124_ = lean_ctor_get(v___x_120_, 4);
v_messages_125_ = lean_ctor_get(v___x_120_, 6);
v_infoState_126_ = lean_ctor_get(v___x_120_, 7);
v_snapshotTasks_127_ = lean_ctor_get(v___x_120_, 8);
v_isSharedCheck_138_ = !lean_is_exclusive(v___x_120_);
if (v_isSharedCheck_138_ == 0)
{
lean_object* v_unused_139_; lean_object* v_unused_140_; 
v_unused_139_ = lean_ctor_get(v___x_120_, 5);
lean_dec(v_unused_139_);
v_unused_140_ = lean_ctor_get(v___x_120_, 0);
lean_dec(v_unused_140_);
v___x_129_ = v___x_120_;
v_isShared_130_ = v_isSharedCheck_138_;
goto v_resetjp_128_;
}
else
{
lean_inc(v_snapshotTasks_127_);
lean_inc(v_infoState_126_);
lean_inc(v_messages_125_);
lean_inc(v_traceState_124_);
lean_inc(v_auxDeclNGen_123_);
lean_inc(v_ngen_122_);
lean_inc(v_nextMacroScope_121_);
lean_dec(v___x_120_);
v___x_129_ = lean_box(0);
v_isShared_130_ = v_isSharedCheck_138_;
goto v_resetjp_128_;
}
v_resetjp_128_:
{
lean_object* v___x_131_; lean_object* v___x_133_; 
v___x_131_ = lean_obj_once(&lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___redArg___closed__2, &lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___redArg___closed__2_once, _init_lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___redArg___closed__2);
if (v_isShared_130_ == 0)
{
lean_ctor_set(v___x_129_, 5, v___x_131_);
lean_ctor_set(v___x_129_, 0, v_env_117_);
v___x_133_ = v___x_129_;
goto v_reusejp_132_;
}
else
{
lean_object* v_reuseFailAlloc_137_; 
v_reuseFailAlloc_137_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_137_, 0, v_env_117_);
lean_ctor_set(v_reuseFailAlloc_137_, 1, v_nextMacroScope_121_);
lean_ctor_set(v_reuseFailAlloc_137_, 2, v_ngen_122_);
lean_ctor_set(v_reuseFailAlloc_137_, 3, v_auxDeclNGen_123_);
lean_ctor_set(v_reuseFailAlloc_137_, 4, v_traceState_124_);
lean_ctor_set(v_reuseFailAlloc_137_, 5, v___x_131_);
lean_ctor_set(v_reuseFailAlloc_137_, 6, v_messages_125_);
lean_ctor_set(v_reuseFailAlloc_137_, 7, v_infoState_126_);
lean_ctor_set(v_reuseFailAlloc_137_, 8, v_snapshotTasks_127_);
v___x_133_ = v_reuseFailAlloc_137_;
goto v_reusejp_132_;
}
v_reusejp_132_:
{
lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; 
v___x_134_ = lean_st_ref_set(v___y_118_, v___x_133_);
v___x_135_ = lean_box(0);
v___x_136_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_136_, 0, v___x_135_);
return v___x_136_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___redArg___boxed(lean_object* v_env_141_, lean_object* v___y_142_, lean_object* v___y_143_){
_start:
{
lean_object* v_res_144_; 
v_res_144_ = lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___redArg(v_env_141_, v___y_142_);
lean_dec(v___y_142_);
return v_res_144_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0(lean_object* v_env_145_, lean_object* v___y_146_, lean_object* v___y_147_){
_start:
{
lean_object* v___x_149_; 
v___x_149_ = lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___redArg(v_env_145_, v___y_147_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___boxed(lean_object* v_env_150_, lean_object* v___y_151_, lean_object* v___y_152_, lean_object* v___y_153_){
_start:
{
lean_object* v_res_154_; 
v_res_154_ = lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0(v_env_150_, v___y_151_, v___y_152_);
lean_dec(v___y_152_);
lean_dec_ref(v___y_151_);
return v_res_154_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications(lean_object* v_oldEnv_155_, lean_object* v_newEnv_156_, lean_object* v_a_157_, lean_object* v_a_158_){
_start:
{
lean_object* v___x_160_; lean_object* v_env_161_; uint8_t v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; 
v___x_160_ = lean_st_ref_get(v_a_158_);
v_env_161_ = lean_ctor_get(v___x_160_, 0);
lean_inc_ref(v_env_161_);
lean_dec(v___x_160_);
v___x_162_ = 1;
v___x_163_ = l_Lean_Environment_replayConsts(v_env_161_, v_oldEnv_155_, v_newEnv_156_, v___x_162_);
v___x_164_ = lp_aesop_Lean_setEnv___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications_spec__0___redArg(v___x_163_, v_a_158_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications___boxed(lean_object* v_oldEnv_165_, lean_object* v_newEnv_166_, lean_object* v_a_167_, lean_object* v_a_168_, lean_object* v_a_169_){
_start:
{
lean_object* v_res_170_; 
v_res_170_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications(v_oldEnv_165_, v_newEnv_166_, v_a_167_, v_a_168_);
lean_dec(v_a_168_);
lean_dec_ref(v_a_167_);
return v_res_170_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_getExprMVarAssignment_x3f___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__0___redArg(lean_object* v_mvarId_171_, lean_object* v___y_172_){
_start:
{
lean_object* v___x_174_; lean_object* v_mctx_175_; lean_object* v___x_176_; lean_object* v___x_177_; 
v___x_174_ = lean_st_ref_get(v___y_172_);
v_mctx_175_ = lean_ctor_get(v___x_174_, 0);
lean_inc_ref(v_mctx_175_);
lean_dec(v___x_174_);
v___x_176_ = l_Lean_MetavarContext_getExprAssignmentCore_x3f(v_mctx_175_, v_mvarId_171_);
lean_dec_ref(v_mctx_175_);
v___x_177_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_177_, 0, v___x_176_);
return v___x_177_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_getExprMVarAssignment_x3f___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__0___redArg___boxed(lean_object* v_mvarId_178_, lean_object* v___y_179_, lean_object* v___y_180_){
_start:
{
lean_object* v_res_181_; 
v_res_181_ = lp_aesop_Lean_getExprMVarAssignment_x3f___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__0___redArg(v_mvarId_178_, v___y_179_);
lean_dec(v___y_179_);
lean_dec(v_mvarId_178_);
return v_res_181_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_getExprMVarAssignment_x3f___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__0(lean_object* v_mvarId_182_, lean_object* v___y_183_, lean_object* v___y_184_, lean_object* v___y_185_, lean_object* v___y_186_){
_start:
{
lean_object* v___x_188_; 
v___x_188_ = lp_aesop_Lean_getExprMVarAssignment_x3f___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__0___redArg(v_mvarId_182_, v___y_184_);
return v___x_188_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_getExprMVarAssignment_x3f___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__0___boxed(lean_object* v_mvarId_189_, lean_object* v___y_190_, lean_object* v___y_191_, lean_object* v___y_192_, lean_object* v___y_193_, lean_object* v___y_194_){
_start:
{
lean_object* v_res_195_; 
v_res_195_ = lp_aesop_Lean_getExprMVarAssignment_x3f___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__0(v_mvarId_189_, v___y_190_, v___y_191_, v___y_192_, v___y_193_);
lean_dec(v___y_193_);
lean_dec_ref(v___y_192_);
lean_dec(v___y_191_);
lean_dec_ref(v___y_190_);
lean_dec(v_mvarId_189_);
return v_res_195_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__1___redArg(lean_object* v_e_196_, lean_object* v___y_197_){
_start:
{
uint8_t v___x_199_; 
v___x_199_ = l_Lean_Expr_hasMVar(v_e_196_);
if (v___x_199_ == 0)
{
lean_object* v___x_200_; 
v___x_200_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_200_, 0, v_e_196_);
return v___x_200_;
}
else
{
lean_object* v___x_201_; lean_object* v_mctx_202_; lean_object* v___x_203_; lean_object* v_fst_204_; lean_object* v_snd_205_; lean_object* v___x_206_; lean_object* v_cache_207_; lean_object* v_zetaDeltaFVarIds_208_; lean_object* v_postponed_209_; lean_object* v_diag_210_; lean_object* v___x_212_; uint8_t v_isShared_213_; uint8_t v_isSharedCheck_219_; 
v___x_201_ = lean_st_ref_get(v___y_197_);
v_mctx_202_ = lean_ctor_get(v___x_201_, 0);
lean_inc_ref(v_mctx_202_);
lean_dec(v___x_201_);
v___x_203_ = l_Lean_instantiateMVarsCore(v_mctx_202_, v_e_196_);
v_fst_204_ = lean_ctor_get(v___x_203_, 0);
lean_inc(v_fst_204_);
v_snd_205_ = lean_ctor_get(v___x_203_, 1);
lean_inc(v_snd_205_);
lean_dec_ref(v___x_203_);
v___x_206_ = lean_st_ref_take(v___y_197_);
v_cache_207_ = lean_ctor_get(v___x_206_, 1);
v_zetaDeltaFVarIds_208_ = lean_ctor_get(v___x_206_, 2);
v_postponed_209_ = lean_ctor_get(v___x_206_, 3);
v_diag_210_ = lean_ctor_get(v___x_206_, 4);
v_isSharedCheck_219_ = !lean_is_exclusive(v___x_206_);
if (v_isSharedCheck_219_ == 0)
{
lean_object* v_unused_220_; 
v_unused_220_ = lean_ctor_get(v___x_206_, 0);
lean_dec(v_unused_220_);
v___x_212_ = v___x_206_;
v_isShared_213_ = v_isSharedCheck_219_;
goto v_resetjp_211_;
}
else
{
lean_inc(v_diag_210_);
lean_inc(v_postponed_209_);
lean_inc(v_zetaDeltaFVarIds_208_);
lean_inc(v_cache_207_);
lean_dec(v___x_206_);
v___x_212_ = lean_box(0);
v_isShared_213_ = v_isSharedCheck_219_;
goto v_resetjp_211_;
}
v_resetjp_211_:
{
lean_object* v___x_215_; 
if (v_isShared_213_ == 0)
{
lean_ctor_set(v___x_212_, 0, v_snd_205_);
v___x_215_ = v___x_212_;
goto v_reusejp_214_;
}
else
{
lean_object* v_reuseFailAlloc_218_; 
v_reuseFailAlloc_218_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_218_, 0, v_snd_205_);
lean_ctor_set(v_reuseFailAlloc_218_, 1, v_cache_207_);
lean_ctor_set(v_reuseFailAlloc_218_, 2, v_zetaDeltaFVarIds_208_);
lean_ctor_set(v_reuseFailAlloc_218_, 3, v_postponed_209_);
lean_ctor_set(v_reuseFailAlloc_218_, 4, v_diag_210_);
v___x_215_ = v_reuseFailAlloc_218_;
goto v_reusejp_214_;
}
v_reusejp_214_:
{
lean_object* v___x_216_; lean_object* v___x_217_; 
v___x_216_ = lean_st_ref_set(v___y_197_, v___x_215_);
v___x_217_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_217_, 0, v_fst_204_);
return v___x_217_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__1___redArg___boxed(lean_object* v_e_221_, lean_object* v___y_222_, lean_object* v___y_223_){
_start:
{
lean_object* v_res_224_; 
v_res_224_ = lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__1___redArg(v_e_221_, v___y_222_);
lean_dec(v___y_222_);
return v_res_224_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__1(lean_object* v_e_225_, lean_object* v___y_226_, lean_object* v___y_227_, lean_object* v___y_228_, lean_object* v___y_229_){
_start:
{
lean_object* v___x_231_; 
v___x_231_ = lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__1___redArg(v_e_225_, v___y_227_);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__1___boxed(lean_object* v_e_232_, lean_object* v___y_233_, lean_object* v___y_234_, lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_){
_start:
{
lean_object* v_res_238_; 
v_res_238_ = lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__1(v_e_232_, v___y_233_, v___y_234_, v___y_235_, v___y_236_);
lean_dec(v___y_236_);
lean_dec_ref(v___y_235_);
lean_dec(v___y_234_);
lean_dec_ref(v___y_233_);
return v_res_238_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_getDelayedMVarAssignment_x3f___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__2___redArg(lean_object* v_mvarId_239_, lean_object* v___y_240_){
_start:
{
lean_object* v___x_242_; lean_object* v_mctx_243_; lean_object* v___x_244_; lean_object* v___x_245_; 
v___x_242_ = lean_st_ref_get(v___y_240_);
v_mctx_243_ = lean_ctor_get(v___x_242_, 0);
lean_inc_ref(v_mctx_243_);
lean_dec(v___x_242_);
v___x_244_ = l_Lean_MetavarContext_getDelayedMVarAssignmentCore_x3f(v_mctx_243_, v_mvarId_239_);
lean_dec_ref(v_mctx_243_);
v___x_245_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_245_, 0, v___x_244_);
return v___x_245_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_getDelayedMVarAssignment_x3f___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__2___redArg___boxed(lean_object* v_mvarId_246_, lean_object* v___y_247_, lean_object* v___y_248_){
_start:
{
lean_object* v_res_249_; 
v_res_249_ = lp_aesop_Lean_getDelayedMVarAssignment_x3f___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__2___redArg(v_mvarId_246_, v___y_247_);
lean_dec(v___y_247_);
lean_dec(v_mvarId_246_);
return v_res_249_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_getDelayedMVarAssignment_x3f___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__2(lean_object* v_mvarId_250_, lean_object* v___y_251_, lean_object* v___y_252_, lean_object* v___y_253_, lean_object* v___y_254_){
_start:
{
lean_object* v___x_256_; 
v___x_256_ = lp_aesop_Lean_getDelayedMVarAssignment_x3f___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__2___redArg(v_mvarId_250_, v___y_252_);
return v___x_256_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_getDelayedMVarAssignment_x3f___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__2___boxed(lean_object* v_mvarId_257_, lean_object* v___y_258_, lean_object* v___y_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_){
_start:
{
lean_object* v_res_263_; 
v_res_263_ = lp_aesop_Lean_getDelayedMVarAssignment_x3f___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__2(v_mvarId_257_, v___y_258_, v___y_259_, v___y_260_, v___y_261_);
lean_dec(v___y_261_);
lean_dec_ref(v___y_260_);
lean_dec(v___y_259_);
lean_dec_ref(v___y_258_);
lean_dec(v_mvarId_257_);
return v_res_263_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isDeclared___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__10___redArg(lean_object* v_mvarId_264_, lean_object* v___y_265_){
_start:
{
lean_object* v___x_267_; lean_object* v_mctx_268_; uint8_t v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; 
v___x_267_ = lean_st_ref_get(v___y_265_);
v_mctx_268_ = lean_ctor_get(v___x_267_, 0);
lean_inc_ref(v_mctx_268_);
lean_dec(v___x_267_);
v___x_269_ = lp_batteries_Lean_MetavarContext_isExprMVarDeclared(v_mctx_268_, v_mvarId_264_);
lean_dec_ref(v_mctx_268_);
v___x_270_ = lean_box(v___x_269_);
v___x_271_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_271_, 0, v___x_270_);
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isDeclared___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__10___redArg___boxed(lean_object* v_mvarId_272_, lean_object* v___y_273_, lean_object* v___y_274_){
_start:
{
lean_object* v_res_275_; 
v_res_275_ = lp_aesop_Lean_MVarId_isDeclared___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__10___redArg(v_mvarId_272_, v___y_273_);
lean_dec(v___y_273_);
lean_dec(v_mvarId_272_);
return v_res_275_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isDeclared___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__10(lean_object* v_mvarId_276_, lean_object* v___y_277_, lean_object* v___y_278_, lean_object* v___y_279_, lean_object* v___y_280_){
_start:
{
lean_object* v___x_282_; 
v___x_282_ = lp_aesop_Lean_MVarId_isDeclared___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__10___redArg(v_mvarId_276_, v___y_278_);
return v___x_282_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isDeclared___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__10___boxed(lean_object* v_mvarId_283_, lean_object* v___y_284_, lean_object* v___y_285_, lean_object* v___y_286_, lean_object* v___y_287_, lean_object* v___y_288_){
_start:
{
lean_object* v_res_289_; 
v_res_289_ = lp_aesop_Lean_MVarId_isDeclared___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__10(v_mvarId_283_, v___y_284_, v___y_285_, v___y_286_, v___y_287_);
lean_dec(v___y_287_);
lean_dec_ref(v___y_286_);
lean_dec(v___y_285_);
lean_dec_ref(v___y_284_);
lean_dec(v_mvarId_283_);
return v_res_289_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__0(lean_object* v_mvarId_290_, lean_object* v___y_291_, lean_object* v___y_292_, lean_object* v___y_293_, lean_object* v___y_294_){
_start:
{
lean_object* v___x_296_; 
v___x_296_ = lp_aesop_Lean_getExprMVarAssignment_x3f___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__0___redArg(v_mvarId_290_, v___y_292_);
if (lean_obj_tag(v___x_296_) == 0)
{
lean_object* v_a_297_; 
v_a_297_ = lean_ctor_get(v___x_296_, 0);
lean_inc(v_a_297_);
lean_dec_ref_known(v___x_296_, 1);
if (lean_obj_tag(v_a_297_) == 1)
{
lean_object* v_val_298_; lean_object* v___x_300_; uint8_t v_isShared_301_; uint8_t v_isSharedCheck_323_; 
v_val_298_ = lean_ctor_get(v_a_297_, 0);
v_isSharedCheck_323_ = !lean_is_exclusive(v_a_297_);
if (v_isSharedCheck_323_ == 0)
{
v___x_300_ = v_a_297_;
v_isShared_301_ = v_isSharedCheck_323_;
goto v_resetjp_299_;
}
else
{
lean_inc(v_val_298_);
lean_dec(v_a_297_);
v___x_300_ = lean_box(0);
v_isShared_301_ = v_isSharedCheck_323_;
goto v_resetjp_299_;
}
v_resetjp_299_:
{
lean_object* v___x_302_; 
v___x_302_ = lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__1___redArg(v_val_298_, v___y_292_);
if (lean_obj_tag(v___x_302_) == 0)
{
lean_object* v_a_303_; lean_object* v___x_305_; uint8_t v_isShared_306_; uint8_t v_isSharedCheck_314_; 
v_a_303_ = lean_ctor_get(v___x_302_, 0);
v_isSharedCheck_314_ = !lean_is_exclusive(v___x_302_);
if (v_isSharedCheck_314_ == 0)
{
v___x_305_ = v___x_302_;
v_isShared_306_ = v_isSharedCheck_314_;
goto v_resetjp_304_;
}
else
{
lean_inc(v_a_303_);
lean_dec(v___x_302_);
v___x_305_ = lean_box(0);
v_isShared_306_ = v_isSharedCheck_314_;
goto v_resetjp_304_;
}
v_resetjp_304_:
{
lean_object* v___x_307_; lean_object* v___x_309_; 
v___x_307_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_307_, 0, v_a_303_);
if (v_isShared_301_ == 0)
{
lean_ctor_set(v___x_300_, 0, v___x_307_);
v___x_309_ = v___x_300_;
goto v_reusejp_308_;
}
else
{
lean_object* v_reuseFailAlloc_313_; 
v_reuseFailAlloc_313_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_313_, 0, v___x_307_);
v___x_309_ = v_reuseFailAlloc_313_;
goto v_reusejp_308_;
}
v_reusejp_308_:
{
lean_object* v___x_311_; 
if (v_isShared_306_ == 0)
{
lean_ctor_set(v___x_305_, 0, v___x_309_);
v___x_311_ = v___x_305_;
goto v_reusejp_310_;
}
else
{
lean_object* v_reuseFailAlloc_312_; 
v_reuseFailAlloc_312_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_312_, 0, v___x_309_);
v___x_311_ = v_reuseFailAlloc_312_;
goto v_reusejp_310_;
}
v_reusejp_310_:
{
return v___x_311_;
}
}
}
}
else
{
lean_object* v_a_315_; lean_object* v___x_317_; uint8_t v_isShared_318_; uint8_t v_isSharedCheck_322_; 
lean_del_object(v___x_300_);
v_a_315_ = lean_ctor_get(v___x_302_, 0);
v_isSharedCheck_322_ = !lean_is_exclusive(v___x_302_);
if (v_isSharedCheck_322_ == 0)
{
v___x_317_ = v___x_302_;
v_isShared_318_ = v_isSharedCheck_322_;
goto v_resetjp_316_;
}
else
{
lean_inc(v_a_315_);
lean_dec(v___x_302_);
v___x_317_ = lean_box(0);
v_isShared_318_ = v_isSharedCheck_322_;
goto v_resetjp_316_;
}
v_resetjp_316_:
{
lean_object* v___x_320_; 
if (v_isShared_318_ == 0)
{
v___x_320_ = v___x_317_;
goto v_reusejp_319_;
}
else
{
lean_object* v_reuseFailAlloc_321_; 
v_reuseFailAlloc_321_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_321_, 0, v_a_315_);
v___x_320_ = v_reuseFailAlloc_321_;
goto v_reusejp_319_;
}
v_reusejp_319_:
{
return v___x_320_;
}
}
}
}
}
else
{
lean_object* v___x_324_; 
lean_dec(v_a_297_);
v___x_324_ = lp_aesop_Lean_getDelayedMVarAssignment_x3f___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__2___redArg(v_mvarId_290_, v___y_292_);
if (lean_obj_tag(v___x_324_) == 0)
{
lean_object* v_a_325_; lean_object* v___x_327_; uint8_t v_isShared_328_; uint8_t v_isSharedCheck_345_; 
v_a_325_ = lean_ctor_get(v___x_324_, 0);
v_isSharedCheck_345_ = !lean_is_exclusive(v___x_324_);
if (v_isSharedCheck_345_ == 0)
{
v___x_327_ = v___x_324_;
v_isShared_328_ = v_isSharedCheck_345_;
goto v_resetjp_326_;
}
else
{
lean_inc(v_a_325_);
lean_dec(v___x_324_);
v___x_327_ = lean_box(0);
v_isShared_328_ = v_isSharedCheck_345_;
goto v_resetjp_326_;
}
v_resetjp_326_:
{
if (lean_obj_tag(v_a_325_) == 1)
{
lean_object* v_val_329_; lean_object* v___x_331_; uint8_t v_isShared_332_; uint8_t v_isSharedCheck_340_; 
v_val_329_ = lean_ctor_get(v_a_325_, 0);
v_isSharedCheck_340_ = !lean_is_exclusive(v_a_325_);
if (v_isSharedCheck_340_ == 0)
{
v___x_331_ = v_a_325_;
v_isShared_332_ = v_isSharedCheck_340_;
goto v_resetjp_330_;
}
else
{
lean_inc(v_val_329_);
lean_dec(v_a_325_);
v___x_331_ = lean_box(0);
v_isShared_332_ = v_isSharedCheck_340_;
goto v_resetjp_330_;
}
v_resetjp_330_:
{
lean_object* v___x_333_; lean_object* v___x_335_; 
v___x_333_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_333_, 0, v_val_329_);
if (v_isShared_332_ == 0)
{
lean_ctor_set(v___x_331_, 0, v___x_333_);
v___x_335_ = v___x_331_;
goto v_reusejp_334_;
}
else
{
lean_object* v_reuseFailAlloc_339_; 
v_reuseFailAlloc_339_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_339_, 0, v___x_333_);
v___x_335_ = v_reuseFailAlloc_339_;
goto v_reusejp_334_;
}
v_reusejp_334_:
{
lean_object* v___x_337_; 
if (v_isShared_328_ == 0)
{
lean_ctor_set(v___x_327_, 0, v___x_335_);
v___x_337_ = v___x_327_;
goto v_reusejp_336_;
}
else
{
lean_object* v_reuseFailAlloc_338_; 
v_reuseFailAlloc_338_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_338_, 0, v___x_335_);
v___x_337_ = v_reuseFailAlloc_338_;
goto v_reusejp_336_;
}
v_reusejp_336_:
{
return v___x_337_;
}
}
}
}
else
{
lean_object* v___x_341_; lean_object* v___x_343_; 
lean_dec(v_a_325_);
v___x_341_ = lean_box(0);
if (v_isShared_328_ == 0)
{
lean_ctor_set(v___x_327_, 0, v___x_341_);
v___x_343_ = v___x_327_;
goto v_reusejp_342_;
}
else
{
lean_object* v_reuseFailAlloc_344_; 
v_reuseFailAlloc_344_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_344_, 0, v___x_341_);
v___x_343_ = v_reuseFailAlloc_344_;
goto v_reusejp_342_;
}
v_reusejp_342_:
{
return v___x_343_;
}
}
}
}
else
{
lean_object* v_a_346_; lean_object* v___x_348_; uint8_t v_isShared_349_; uint8_t v_isSharedCheck_353_; 
v_a_346_ = lean_ctor_get(v___x_324_, 0);
v_isSharedCheck_353_ = !lean_is_exclusive(v___x_324_);
if (v_isSharedCheck_353_ == 0)
{
v___x_348_ = v___x_324_;
v_isShared_349_ = v_isSharedCheck_353_;
goto v_resetjp_347_;
}
else
{
lean_inc(v_a_346_);
lean_dec(v___x_324_);
v___x_348_ = lean_box(0);
v_isShared_349_ = v_isSharedCheck_353_;
goto v_resetjp_347_;
}
v_resetjp_347_:
{
lean_object* v___x_351_; 
if (v_isShared_349_ == 0)
{
v___x_351_ = v___x_348_;
goto v_reusejp_350_;
}
else
{
lean_object* v_reuseFailAlloc_352_; 
v_reuseFailAlloc_352_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_352_, 0, v_a_346_);
v___x_351_ = v_reuseFailAlloc_352_;
goto v_reusejp_350_;
}
v_reusejp_350_:
{
return v___x_351_;
}
}
}
}
}
else
{
lean_object* v_a_354_; lean_object* v___x_356_; uint8_t v_isShared_357_; uint8_t v_isSharedCheck_361_; 
v_a_354_ = lean_ctor_get(v___x_296_, 0);
v_isSharedCheck_361_ = !lean_is_exclusive(v___x_296_);
if (v_isSharedCheck_361_ == 0)
{
v___x_356_ = v___x_296_;
v_isShared_357_ = v_isSharedCheck_361_;
goto v_resetjp_355_;
}
else
{
lean_inc(v_a_354_);
lean_dec(v___x_296_);
v___x_356_ = lean_box(0);
v_isShared_357_ = v_isSharedCheck_361_;
goto v_resetjp_355_;
}
v_resetjp_355_:
{
lean_object* v___x_359_; 
if (v_isShared_357_ == 0)
{
v___x_359_ = v___x_356_;
goto v_reusejp_358_;
}
else
{
lean_object* v_reuseFailAlloc_360_; 
v_reuseFailAlloc_360_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_360_, 0, v_a_354_);
v___x_359_ = v_reuseFailAlloc_360_;
goto v_reusejp_358_;
}
v_reusejp_358_:
{
return v___x_359_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__0___boxed(lean_object* v_mvarId_362_, lean_object* v___y_363_, lean_object* v___y_364_, lean_object* v___y_365_, lean_object* v___y_366_, lean_object* v___y_367_){
_start:
{
lean_object* v_res_368_; 
v_res_368_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__0(v_mvarId_362_, v___y_363_, v___y_364_, v___y_365_, v___y_366_);
lean_dec(v___y_366_);
lean_dec_ref(v___y_365_);
lean_dec(v___y_364_);
lean_dec_ref(v___y_363_);
lean_dec(v_mvarId_362_);
return v_res_368_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__4_spec__4(lean_object* v_opts_369_, lean_object* v_opt_370_){
_start:
{
lean_object* v_name_371_; lean_object* v_defValue_372_; lean_object* v_map_373_; lean_object* v___x_374_; 
v_name_371_ = lean_ctor_get(v_opt_370_, 0);
v_defValue_372_ = lean_ctor_get(v_opt_370_, 1);
v_map_373_ = lean_ctor_get(v_opts_369_, 0);
v___x_374_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_373_, v_name_371_);
if (lean_obj_tag(v___x_374_) == 0)
{
uint8_t v___x_375_; 
v___x_375_ = lean_unbox(v_defValue_372_);
return v___x_375_;
}
else
{
lean_object* v_val_376_; 
v_val_376_ = lean_ctor_get(v___x_374_, 0);
lean_inc(v_val_376_);
lean_dec_ref_known(v___x_374_, 1);
if (lean_obj_tag(v_val_376_) == 1)
{
uint8_t v_v_377_; 
v_v_377_ = lean_ctor_get_uint8(v_val_376_, 0);
lean_dec_ref_known(v_val_376_, 0);
return v_v_377_;
}
else
{
uint8_t v___x_378_; 
lean_dec(v_val_376_);
v___x_378_ = lean_unbox(v_defValue_372_);
return v___x_378_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__4_spec__4___boxed(lean_object* v_opts_379_, lean_object* v_opt_380_){
_start:
{
uint8_t v_res_381_; lean_object* v_r_382_; 
v_res_381_ = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__4_spec__4(v_opts_379_, v_opt_380_);
lean_dec_ref(v_opt_380_);
lean_dec_ref(v_opts_379_);
v_r_382_ = lean_box(v_res_381_);
return v_r_382_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__4___redArg(lean_object* v_opt_383_, lean_object* v___y_384_){
_start:
{
lean_object* v_options_386_; lean_object* v_option_387_; uint8_t v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; 
v_options_386_ = lean_ctor_get(v___y_384_, 2);
v_option_387_ = lean_ctor_get(v_opt_383_, 1);
v___x_388_ = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__4_spec__4(v_options_386_, v_option_387_);
v___x_389_ = lean_box(v___x_388_);
v___x_390_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_390_, 0, v___x_389_);
return v___x_390_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__4___redArg___boxed(lean_object* v_opt_391_, lean_object* v___y_392_, lean_object* v___y_393_){
_start:
{
lean_object* v_res_394_; 
v_res_394_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__4___redArg(v_opt_391_, v___y_392_);
lean_dec_ref(v___y_392_);
lean_dec_ref(v_opt_391_);
return v_res_394_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6_spec__7(lean_object* v_msgData_395_, lean_object* v___y_396_, lean_object* v___y_397_, lean_object* v___y_398_, lean_object* v___y_399_){
_start:
{
lean_object* v___x_401_; lean_object* v_env_402_; lean_object* v___x_403_; lean_object* v_mctx_404_; lean_object* v_lctx_405_; lean_object* v_options_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; 
v___x_401_ = lean_st_ref_get(v___y_399_);
v_env_402_ = lean_ctor_get(v___x_401_, 0);
lean_inc_ref(v_env_402_);
lean_dec(v___x_401_);
v___x_403_ = lean_st_ref_get(v___y_397_);
v_mctx_404_ = lean_ctor_get(v___x_403_, 0);
lean_inc_ref(v_mctx_404_);
lean_dec(v___x_403_);
v_lctx_405_ = lean_ctor_get(v___y_396_, 2);
v_options_406_ = lean_ctor_get(v___y_398_, 2);
lean_inc_ref(v_options_406_);
lean_inc_ref(v_lctx_405_);
v___x_407_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_407_, 0, v_env_402_);
lean_ctor_set(v___x_407_, 1, v_mctx_404_);
lean_ctor_set(v___x_407_, 2, v_lctx_405_);
lean_ctor_set(v___x_407_, 3, v_options_406_);
v___x_408_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_408_, 0, v___x_407_);
lean_ctor_set(v___x_408_, 1, v_msgData_395_);
v___x_409_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_409_, 0, v___x_408_);
return v___x_409_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6_spec__7___boxed(lean_object* v_msgData_410_, lean_object* v___y_411_, lean_object* v___y_412_, lean_object* v___y_413_, lean_object* v___y_414_, lean_object* v___y_415_){
_start:
{
lean_object* v_res_416_; 
v_res_416_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6_spec__7(v_msgData_410_, v___y_411_, v___y_412_, v___y_413_, v___y_414_);
lean_dec(v___y_414_);
lean_dec_ref(v___y_413_);
lean_dec(v___y_412_);
lean_dec_ref(v___y_411_);
return v_res_416_;
}
}
static double _init_lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6___closed__0(void){
_start:
{
lean_object* v___x_417_; double v___x_418_; 
v___x_417_ = lean_unsigned_to_nat(0u);
v___x_418_ = lean_float_of_nat(v___x_417_);
return v___x_418_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6(lean_object* v_cls_422_, lean_object* v_msg_423_, lean_object* v___y_424_, lean_object* v___y_425_, lean_object* v___y_426_, lean_object* v___y_427_){
_start:
{
lean_object* v_ref_429_; lean_object* v___x_430_; lean_object* v_a_431_; lean_object* v___x_433_; uint8_t v_isShared_434_; uint8_t v_isSharedCheck_475_; 
v_ref_429_ = lean_ctor_get(v___y_426_, 5);
v___x_430_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6_spec__7(v_msg_423_, v___y_424_, v___y_425_, v___y_426_, v___y_427_);
v_a_431_ = lean_ctor_get(v___x_430_, 0);
v_isSharedCheck_475_ = !lean_is_exclusive(v___x_430_);
if (v_isSharedCheck_475_ == 0)
{
v___x_433_ = v___x_430_;
v_isShared_434_ = v_isSharedCheck_475_;
goto v_resetjp_432_;
}
else
{
lean_inc(v_a_431_);
lean_dec(v___x_430_);
v___x_433_ = lean_box(0);
v_isShared_434_ = v_isSharedCheck_475_;
goto v_resetjp_432_;
}
v_resetjp_432_:
{
lean_object* v___x_435_; lean_object* v_traceState_436_; lean_object* v_env_437_; lean_object* v_nextMacroScope_438_; lean_object* v_ngen_439_; lean_object* v_auxDeclNGen_440_; lean_object* v_cache_441_; lean_object* v_messages_442_; lean_object* v_infoState_443_; lean_object* v_snapshotTasks_444_; lean_object* v___x_446_; uint8_t v_isShared_447_; uint8_t v_isSharedCheck_474_; 
v___x_435_ = lean_st_ref_take(v___y_427_);
v_traceState_436_ = lean_ctor_get(v___x_435_, 4);
v_env_437_ = lean_ctor_get(v___x_435_, 0);
v_nextMacroScope_438_ = lean_ctor_get(v___x_435_, 1);
v_ngen_439_ = lean_ctor_get(v___x_435_, 2);
v_auxDeclNGen_440_ = lean_ctor_get(v___x_435_, 3);
v_cache_441_ = lean_ctor_get(v___x_435_, 5);
v_messages_442_ = lean_ctor_get(v___x_435_, 6);
v_infoState_443_ = lean_ctor_get(v___x_435_, 7);
v_snapshotTasks_444_ = lean_ctor_get(v___x_435_, 8);
v_isSharedCheck_474_ = !lean_is_exclusive(v___x_435_);
if (v_isSharedCheck_474_ == 0)
{
v___x_446_ = v___x_435_;
v_isShared_447_ = v_isSharedCheck_474_;
goto v_resetjp_445_;
}
else
{
lean_inc(v_snapshotTasks_444_);
lean_inc(v_infoState_443_);
lean_inc(v_messages_442_);
lean_inc(v_cache_441_);
lean_inc(v_traceState_436_);
lean_inc(v_auxDeclNGen_440_);
lean_inc(v_ngen_439_);
lean_inc(v_nextMacroScope_438_);
lean_inc(v_env_437_);
lean_dec(v___x_435_);
v___x_446_ = lean_box(0);
v_isShared_447_ = v_isSharedCheck_474_;
goto v_resetjp_445_;
}
v_resetjp_445_:
{
uint64_t v_tid_448_; lean_object* v_traces_449_; lean_object* v___x_451_; uint8_t v_isShared_452_; uint8_t v_isSharedCheck_473_; 
v_tid_448_ = lean_ctor_get_uint64(v_traceState_436_, sizeof(void*)*1);
v_traces_449_ = lean_ctor_get(v_traceState_436_, 0);
v_isSharedCheck_473_ = !lean_is_exclusive(v_traceState_436_);
if (v_isSharedCheck_473_ == 0)
{
v___x_451_ = v_traceState_436_;
v_isShared_452_ = v_isSharedCheck_473_;
goto v_resetjp_450_;
}
else
{
lean_inc(v_traces_449_);
lean_dec(v_traceState_436_);
v___x_451_ = lean_box(0);
v_isShared_452_ = v_isSharedCheck_473_;
goto v_resetjp_450_;
}
v_resetjp_450_:
{
lean_object* v___x_453_; double v___x_454_; uint8_t v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_463_; 
v___x_453_ = lean_box(0);
v___x_454_ = lean_float_once(&lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6___closed__0, &lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6___closed__0_once, _init_lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6___closed__0);
v___x_455_ = 0;
v___x_456_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6___closed__1));
v___x_457_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_457_, 0, v_cls_422_);
lean_ctor_set(v___x_457_, 1, v___x_453_);
lean_ctor_set(v___x_457_, 2, v___x_456_);
lean_ctor_set_float(v___x_457_, sizeof(void*)*3, v___x_454_);
lean_ctor_set_float(v___x_457_, sizeof(void*)*3 + 8, v___x_454_);
lean_ctor_set_uint8(v___x_457_, sizeof(void*)*3 + 16, v___x_455_);
v___x_458_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6___closed__2));
v___x_459_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_459_, 0, v___x_457_);
lean_ctor_set(v___x_459_, 1, v_a_431_);
lean_ctor_set(v___x_459_, 2, v___x_458_);
lean_inc(v_ref_429_);
v___x_460_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_460_, 0, v_ref_429_);
lean_ctor_set(v___x_460_, 1, v___x_459_);
v___x_461_ = l_Lean_PersistentArray_push___redArg(v_traces_449_, v___x_460_);
if (v_isShared_452_ == 0)
{
lean_ctor_set(v___x_451_, 0, v___x_461_);
v___x_463_ = v___x_451_;
goto v_reusejp_462_;
}
else
{
lean_object* v_reuseFailAlloc_472_; 
v_reuseFailAlloc_472_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_472_, 0, v___x_461_);
lean_ctor_set_uint64(v_reuseFailAlloc_472_, sizeof(void*)*1, v_tid_448_);
v___x_463_ = v_reuseFailAlloc_472_;
goto v_reusejp_462_;
}
v_reusejp_462_:
{
lean_object* v___x_465_; 
if (v_isShared_447_ == 0)
{
lean_ctor_set(v___x_446_, 4, v___x_463_);
v___x_465_ = v___x_446_;
goto v_reusejp_464_;
}
else
{
lean_object* v_reuseFailAlloc_471_; 
v_reuseFailAlloc_471_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_471_, 0, v_env_437_);
lean_ctor_set(v_reuseFailAlloc_471_, 1, v_nextMacroScope_438_);
lean_ctor_set(v_reuseFailAlloc_471_, 2, v_ngen_439_);
lean_ctor_set(v_reuseFailAlloc_471_, 3, v_auxDeclNGen_440_);
lean_ctor_set(v_reuseFailAlloc_471_, 4, v___x_463_);
lean_ctor_set(v_reuseFailAlloc_471_, 5, v_cache_441_);
lean_ctor_set(v_reuseFailAlloc_471_, 6, v_messages_442_);
lean_ctor_set(v_reuseFailAlloc_471_, 7, v_infoState_443_);
lean_ctor_set(v_reuseFailAlloc_471_, 8, v_snapshotTasks_444_);
v___x_465_ = v_reuseFailAlloc_471_;
goto v_reusejp_464_;
}
v_reusejp_464_:
{
lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_469_; 
v___x_466_ = lean_st_ref_set(v___y_427_, v___x_465_);
v___x_467_ = lean_box(0);
if (v_isShared_434_ == 0)
{
lean_ctor_set(v___x_433_, 0, v___x_467_);
v___x_469_ = v___x_433_;
goto v_reusejp_468_;
}
else
{
lean_object* v_reuseFailAlloc_470_; 
v_reuseFailAlloc_470_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_470_, 0, v___x_467_);
v___x_469_ = v_reuseFailAlloc_470_;
goto v_reusejp_468_;
}
v_reusejp_468_:
{
return v___x_469_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6___boxed(lean_object* v_cls_476_, lean_object* v_msg_477_, lean_object* v___y_478_, lean_object* v___y_479_, lean_object* v___y_480_, lean_object* v___y_481_, lean_object* v___y_482_){
_start:
{
lean_object* v_res_483_; 
v_res_483_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6(v_cls_476_, v_msg_477_, v___y_478_, v___y_479_, v___y_480_, v___y_481_);
lean_dec(v___y_481_);
lean_dec_ref(v___y_480_);
lean_dec(v___y_479_);
lean_dec_ref(v___y_478_);
return v_res_483_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__25___redArg(lean_object* v_t_484_, lean_object* v_k_485_){
_start:
{
if (lean_obj_tag(v_t_484_) == 0)
{
lean_object* v_k_486_; lean_object* v_v_487_; lean_object* v_l_488_; lean_object* v_r_489_; uint8_t v___x_490_; 
v_k_486_ = lean_ctor_get(v_t_484_, 1);
v_v_487_ = lean_ctor_get(v_t_484_, 2);
v_l_488_ = lean_ctor_get(v_t_484_, 3);
v_r_489_ = lean_ctor_get(v_t_484_, 4);
v___x_490_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_k_485_, v_k_486_);
switch(v___x_490_)
{
case 0:
{
v_t_484_ = v_l_488_;
goto _start;
}
case 1:
{
lean_object* v___x_492_; 
lean_inc(v_v_487_);
v___x_492_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_492_, 0, v_v_487_);
return v___x_492_;
}
default: 
{
v_t_484_ = v_r_489_;
goto _start;
}
}
}
else
{
lean_object* v___x_494_; 
v___x_494_ = lean_box(0);
return v___x_494_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__25___redArg___boxed(lean_object* v_t_495_, lean_object* v_k_496_){
_start:
{
lean_object* v_res_497_; 
v_res_497_ = lp_aesop_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__25___redArg(v_t_495_, v_k_496_);
lean_dec(v_k_496_);
lean_dec(v_t_495_);
return v_res_497_;
}
}
static lean_object* _init_lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26___closed__0(void){
_start:
{
lean_object* v___x_498_; 
v___x_498_ = l_instMonadEIO(lean_box(0));
return v___x_498_;
}
}
LEAN_EXPORT lean_object* lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26(lean_object* v_msg_503_, lean_object* v___y_504_, lean_object* v___y_505_, lean_object* v___y_506_, lean_object* v___y_507_){
_start:
{
lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v_toApplicative_511_; lean_object* v___x_513_; uint8_t v_isShared_514_; uint8_t v_isSharedCheck_572_; 
v___x_509_ = lean_obj_once(&lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26___closed__0, &lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26___closed__0_once, _init_lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26___closed__0);
v___x_510_ = l_StateRefT_x27_instMonad___redArg(v___x_509_);
v_toApplicative_511_ = lean_ctor_get(v___x_510_, 0);
v_isSharedCheck_572_ = !lean_is_exclusive(v___x_510_);
if (v_isSharedCheck_572_ == 0)
{
lean_object* v_unused_573_; 
v_unused_573_ = lean_ctor_get(v___x_510_, 1);
lean_dec(v_unused_573_);
v___x_513_ = v___x_510_;
v_isShared_514_ = v_isSharedCheck_572_;
goto v_resetjp_512_;
}
else
{
lean_inc(v_toApplicative_511_);
lean_dec(v___x_510_);
v___x_513_ = lean_box(0);
v_isShared_514_ = v_isSharedCheck_572_;
goto v_resetjp_512_;
}
v_resetjp_512_:
{
lean_object* v_toFunctor_515_; lean_object* v_toSeq_516_; lean_object* v_toSeqLeft_517_; lean_object* v_toSeqRight_518_; lean_object* v___x_520_; uint8_t v_isShared_521_; uint8_t v_isSharedCheck_570_; 
v_toFunctor_515_ = lean_ctor_get(v_toApplicative_511_, 0);
v_toSeq_516_ = lean_ctor_get(v_toApplicative_511_, 2);
v_toSeqLeft_517_ = lean_ctor_get(v_toApplicative_511_, 3);
v_toSeqRight_518_ = lean_ctor_get(v_toApplicative_511_, 4);
v_isSharedCheck_570_ = !lean_is_exclusive(v_toApplicative_511_);
if (v_isSharedCheck_570_ == 0)
{
lean_object* v_unused_571_; 
v_unused_571_ = lean_ctor_get(v_toApplicative_511_, 1);
lean_dec(v_unused_571_);
v___x_520_ = v_toApplicative_511_;
v_isShared_521_ = v_isSharedCheck_570_;
goto v_resetjp_519_;
}
else
{
lean_inc(v_toSeqRight_518_);
lean_inc(v_toSeqLeft_517_);
lean_inc(v_toSeq_516_);
lean_inc(v_toFunctor_515_);
lean_dec(v_toApplicative_511_);
v___x_520_ = lean_box(0);
v_isShared_521_ = v_isSharedCheck_570_;
goto v_resetjp_519_;
}
v_resetjp_519_:
{
lean_object* v___f_522_; lean_object* v___f_523_; lean_object* v___f_524_; lean_object* v___f_525_; lean_object* v___x_526_; lean_object* v___f_527_; lean_object* v___f_528_; lean_object* v___f_529_; lean_object* v___x_531_; 
v___f_522_ = ((lean_object*)(lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26___closed__1));
v___f_523_ = ((lean_object*)(lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26___closed__2));
lean_inc_ref(v_toFunctor_515_);
v___f_524_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_524_, 0, v_toFunctor_515_);
v___f_525_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_525_, 0, v_toFunctor_515_);
v___x_526_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_526_, 0, v___f_524_);
lean_ctor_set(v___x_526_, 1, v___f_525_);
v___f_527_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_527_, 0, v_toSeqRight_518_);
v___f_528_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_528_, 0, v_toSeqLeft_517_);
v___f_529_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_529_, 0, v_toSeq_516_);
if (v_isShared_521_ == 0)
{
lean_ctor_set(v___x_520_, 4, v___f_527_);
lean_ctor_set(v___x_520_, 3, v___f_528_);
lean_ctor_set(v___x_520_, 2, v___f_529_);
lean_ctor_set(v___x_520_, 1, v___f_522_);
lean_ctor_set(v___x_520_, 0, v___x_526_);
v___x_531_ = v___x_520_;
goto v_reusejp_530_;
}
else
{
lean_object* v_reuseFailAlloc_569_; 
v_reuseFailAlloc_569_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_569_, 0, v___x_526_);
lean_ctor_set(v_reuseFailAlloc_569_, 1, v___f_522_);
lean_ctor_set(v_reuseFailAlloc_569_, 2, v___f_529_);
lean_ctor_set(v_reuseFailAlloc_569_, 3, v___f_528_);
lean_ctor_set(v_reuseFailAlloc_569_, 4, v___f_527_);
v___x_531_ = v_reuseFailAlloc_569_;
goto v_reusejp_530_;
}
v_reusejp_530_:
{
lean_object* v___x_533_; 
if (v_isShared_514_ == 0)
{
lean_ctor_set(v___x_513_, 1, v___f_523_);
lean_ctor_set(v___x_513_, 0, v___x_531_);
v___x_533_ = v___x_513_;
goto v_reusejp_532_;
}
else
{
lean_object* v_reuseFailAlloc_568_; 
v_reuseFailAlloc_568_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_568_, 0, v___x_531_);
lean_ctor_set(v_reuseFailAlloc_568_, 1, v___f_523_);
v___x_533_ = v_reuseFailAlloc_568_;
goto v_reusejp_532_;
}
v_reusejp_532_:
{
lean_object* v___x_534_; lean_object* v_toApplicative_535_; lean_object* v___x_537_; uint8_t v_isShared_538_; uint8_t v_isSharedCheck_566_; 
v___x_534_ = l_StateRefT_x27_instMonad___redArg(v___x_533_);
v_toApplicative_535_ = lean_ctor_get(v___x_534_, 0);
v_isSharedCheck_566_ = !lean_is_exclusive(v___x_534_);
if (v_isSharedCheck_566_ == 0)
{
lean_object* v_unused_567_; 
v_unused_567_ = lean_ctor_get(v___x_534_, 1);
lean_dec(v_unused_567_);
v___x_537_ = v___x_534_;
v_isShared_538_ = v_isSharedCheck_566_;
goto v_resetjp_536_;
}
else
{
lean_inc(v_toApplicative_535_);
lean_dec(v___x_534_);
v___x_537_ = lean_box(0);
v_isShared_538_ = v_isSharedCheck_566_;
goto v_resetjp_536_;
}
v_resetjp_536_:
{
lean_object* v_toFunctor_539_; lean_object* v_toSeq_540_; lean_object* v_toSeqLeft_541_; lean_object* v_toSeqRight_542_; lean_object* v___x_544_; uint8_t v_isShared_545_; uint8_t v_isSharedCheck_564_; 
v_toFunctor_539_ = lean_ctor_get(v_toApplicative_535_, 0);
v_toSeq_540_ = lean_ctor_get(v_toApplicative_535_, 2);
v_toSeqLeft_541_ = lean_ctor_get(v_toApplicative_535_, 3);
v_toSeqRight_542_ = lean_ctor_get(v_toApplicative_535_, 4);
v_isSharedCheck_564_ = !lean_is_exclusive(v_toApplicative_535_);
if (v_isSharedCheck_564_ == 0)
{
lean_object* v_unused_565_; 
v_unused_565_ = lean_ctor_get(v_toApplicative_535_, 1);
lean_dec(v_unused_565_);
v___x_544_ = v_toApplicative_535_;
v_isShared_545_ = v_isSharedCheck_564_;
goto v_resetjp_543_;
}
else
{
lean_inc(v_toSeqRight_542_);
lean_inc(v_toSeqLeft_541_);
lean_inc(v_toSeq_540_);
lean_inc(v_toFunctor_539_);
lean_dec(v_toApplicative_535_);
v___x_544_ = lean_box(0);
v_isShared_545_ = v_isSharedCheck_564_;
goto v_resetjp_543_;
}
v_resetjp_543_:
{
lean_object* v___f_546_; lean_object* v___f_547_; lean_object* v___f_548_; lean_object* v___f_549_; lean_object* v___x_550_; lean_object* v___f_551_; lean_object* v___f_552_; lean_object* v___f_553_; lean_object* v___x_555_; 
v___f_546_ = ((lean_object*)(lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26___closed__3));
v___f_547_ = ((lean_object*)(lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26___closed__4));
lean_inc_ref(v_toFunctor_539_);
v___f_548_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_548_, 0, v_toFunctor_539_);
v___f_549_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_549_, 0, v_toFunctor_539_);
v___x_550_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_550_, 0, v___f_548_);
lean_ctor_set(v___x_550_, 1, v___f_549_);
v___f_551_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_551_, 0, v_toSeqRight_542_);
v___f_552_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_552_, 0, v_toSeqLeft_541_);
v___f_553_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_553_, 0, v_toSeq_540_);
if (v_isShared_545_ == 0)
{
lean_ctor_set(v___x_544_, 4, v___f_551_);
lean_ctor_set(v___x_544_, 3, v___f_552_);
lean_ctor_set(v___x_544_, 2, v___f_553_);
lean_ctor_set(v___x_544_, 1, v___f_546_);
lean_ctor_set(v___x_544_, 0, v___x_550_);
v___x_555_ = v___x_544_;
goto v_reusejp_554_;
}
else
{
lean_object* v_reuseFailAlloc_563_; 
v_reuseFailAlloc_563_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_563_, 0, v___x_550_);
lean_ctor_set(v_reuseFailAlloc_563_, 1, v___f_546_);
lean_ctor_set(v_reuseFailAlloc_563_, 2, v___f_553_);
lean_ctor_set(v_reuseFailAlloc_563_, 3, v___f_552_);
lean_ctor_set(v_reuseFailAlloc_563_, 4, v___f_551_);
v___x_555_ = v_reuseFailAlloc_563_;
goto v_reusejp_554_;
}
v_reusejp_554_:
{
lean_object* v___x_557_; 
if (v_isShared_538_ == 0)
{
lean_ctor_set(v___x_537_, 1, v___f_547_);
lean_ctor_set(v___x_537_, 0, v___x_555_);
v___x_557_ = v___x_537_;
goto v_reusejp_556_;
}
else
{
lean_object* v_reuseFailAlloc_562_; 
v_reuseFailAlloc_562_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_562_, 0, v___x_555_);
lean_ctor_set(v_reuseFailAlloc_562_, 1, v___f_547_);
v___x_557_ = v_reuseFailAlloc_562_;
goto v_reusejp_556_;
}
v_reusejp_556_:
{
lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_13816__overap_560_; lean_object* v___x_561_; 
v___x_558_ = l_Lean_instInhabitedLocalContext_default;
v___x_559_ = l_instInhabitedOfMonad___redArg(v___x_557_, v___x_558_);
v___x_13816__overap_560_ = lean_panic_fn_borrowed(v___x_559_, v_msg_503_);
lean_dec(v___x_559_);
lean_inc(v___y_507_);
lean_inc_ref(v___y_506_);
lean_inc(v___y_505_);
lean_inc_ref(v___y_504_);
v___x_561_ = lean_apply_5(v___x_13816__overap_560_, v___y_504_, v___y_505_, v___y_506_, v___y_507_, lean_box(0));
return v___x_561_;
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
LEAN_EXPORT lean_object* lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26___boxed(lean_object* v_msg_574_, lean_object* v___y_575_, lean_object* v___y_576_, lean_object* v___y_577_, lean_object* v___y_578_, lean_object* v___y_579_){
_start:
{
lean_object* v_res_580_; 
v_res_580_ = lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26(v_msg_574_, v___y_575_, v___y_576_, v___y_577_, v___y_578_);
lean_dec(v___y_578_);
lean_dec_ref(v___y_577_);
lean_dec(v___y_576_);
lean_dec_ref(v___y_575_);
return v_res_580_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34(lean_object* v_auxDeclToFullName_585_, lean_object* v_as_586_, size_t v_i_587_, size_t v_stop_588_, lean_object* v_b_589_, lean_object* v___y_590_, lean_object* v___y_591_, lean_object* v___y_592_, lean_object* v___y_593_){
_start:
{
lean_object* v_a_596_; uint8_t v___x_600_; 
v___x_600_ = lean_usize_dec_eq(v_i_587_, v_stop_588_);
if (v___x_600_ == 0)
{
lean_object* v___x_601_; 
v___x_601_ = lean_array_uget_borrowed(v_as_586_, v_i_587_);
if (lean_obj_tag(v___x_601_) == 0)
{
v_a_596_ = v_b_589_;
goto v___jp_595_;
}
else
{
lean_object* v_val_602_; 
v_val_602_ = lean_ctor_get(v___x_601_, 0);
if (lean_obj_tag(v_val_602_) == 0)
{
uint8_t v_kind_603_; 
v_kind_603_ = lean_ctor_get_uint8(v_val_602_, sizeof(void*)*4 + 1);
if (v_kind_603_ == 2)
{
lean_object* v_fvarId_604_; lean_object* v_userName_605_; lean_object* v_type_606_; lean_object* v___x_607_; 
v_fvarId_604_ = lean_ctor_get(v_val_602_, 1);
v_userName_605_ = lean_ctor_get(v_val_602_, 2);
v_type_606_ = lean_ctor_get(v_val_602_, 3);
lean_inc_ref(v_type_606_);
v___x_607_ = lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__1___redArg(v_type_606_, v___y_591_);
if (lean_obj_tag(v___x_607_) == 0)
{
lean_object* v_a_608_; lean_object* v___x_609_; 
v_a_608_ = lean_ctor_get(v___x_607_, 0);
lean_inc(v_a_608_);
lean_dec_ref_known(v___x_607_, 1);
v___x_609_ = lp_aesop_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__25___redArg(v_auxDeclToFullName_585_, v_fvarId_604_);
if (lean_obj_tag(v___x_609_) == 1)
{
lean_object* v_val_610_; lean_object* v___x_611_; 
v_val_610_ = lean_ctor_get(v___x_609_, 0);
lean_inc(v_val_610_);
lean_dec_ref_known(v___x_609_, 1);
lean_inc(v_userName_605_);
lean_inc(v_fvarId_604_);
v___x_611_ = l_Lean_LocalContext_mkAuxDecl(v_b_589_, v_fvarId_604_, v_userName_605_, v_a_608_, v_val_610_);
v_a_596_ = v___x_611_;
goto v___jp_595_;
}
else
{
lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; uint8_t v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; 
lean_dec(v___x_609_);
lean_dec(v_a_608_);
lean_dec_ref(v_b_589_);
v___x_612_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34___closed__0));
v___x_613_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34___closed__1));
v___x_614_ = lean_unsigned_to_nat(635u);
v___x_615_ = lean_unsigned_to_nat(12u);
v___x_616_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34___closed__2));
v___x_617_ = 1;
lean_inc(v_userName_605_);
v___x_618_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_userName_605_, v___x_617_);
v___x_619_ = lean_string_append(v___x_616_, v___x_618_);
lean_dec_ref(v___x_618_);
v___x_620_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34___closed__3));
v___x_621_ = lean_string_append(v___x_619_, v___x_620_);
v___x_622_ = l_mkPanicMessageWithDecl(v___x_612_, v___x_613_, v___x_614_, v___x_615_, v___x_621_);
lean_dec_ref(v___x_621_);
v___x_623_ = lp_aesop_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__26(v___x_622_, v___y_590_, v___y_591_, v___y_592_, v___y_593_);
if (lean_obj_tag(v___x_623_) == 0)
{
lean_object* v_a_624_; 
v_a_624_ = lean_ctor_get(v___x_623_, 0);
lean_inc(v_a_624_);
lean_dec_ref_known(v___x_623_, 1);
v_a_596_ = v_a_624_;
goto v___jp_595_;
}
else
{
return v___x_623_;
}
}
}
else
{
lean_object* v_a_625_; lean_object* v___x_627_; uint8_t v_isShared_628_; uint8_t v_isSharedCheck_632_; 
lean_dec_ref(v_b_589_);
v_a_625_ = lean_ctor_get(v___x_607_, 0);
v_isSharedCheck_632_ = !lean_is_exclusive(v___x_607_);
if (v_isSharedCheck_632_ == 0)
{
v___x_627_ = v___x_607_;
v_isShared_628_ = v_isSharedCheck_632_;
goto v_resetjp_626_;
}
else
{
lean_inc(v_a_625_);
lean_dec(v___x_607_);
v___x_627_ = lean_box(0);
v_isShared_628_ = v_isSharedCheck_632_;
goto v_resetjp_626_;
}
v_resetjp_626_:
{
lean_object* v___x_630_; 
if (v_isShared_628_ == 0)
{
v___x_630_ = v___x_627_;
goto v_reusejp_629_;
}
else
{
lean_object* v_reuseFailAlloc_631_; 
v_reuseFailAlloc_631_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_631_, 0, v_a_625_);
v___x_630_ = v_reuseFailAlloc_631_;
goto v_reusejp_629_;
}
v_reusejp_629_:
{
return v___x_630_;
}
}
}
}
else
{
lean_object* v_fvarId_633_; lean_object* v_userName_634_; lean_object* v_type_635_; uint8_t v_bi_636_; lean_object* v___x_637_; 
v_fvarId_633_ = lean_ctor_get(v_val_602_, 1);
v_userName_634_ = lean_ctor_get(v_val_602_, 2);
v_type_635_ = lean_ctor_get(v_val_602_, 3);
v_bi_636_ = lean_ctor_get_uint8(v_val_602_, sizeof(void*)*4);
lean_inc_ref(v_type_635_);
v___x_637_ = lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__1___redArg(v_type_635_, v___y_591_);
if (lean_obj_tag(v___x_637_) == 0)
{
lean_object* v_a_638_; lean_object* v___x_639_; 
v_a_638_ = lean_ctor_get(v___x_637_, 0);
lean_inc(v_a_638_);
lean_dec_ref_known(v___x_637_, 1);
lean_inc(v_userName_634_);
lean_inc(v_fvarId_633_);
v___x_639_ = l_Lean_LocalContext_mkLocalDecl(v_b_589_, v_fvarId_633_, v_userName_634_, v_a_638_, v_bi_636_, v_kind_603_);
v_a_596_ = v___x_639_;
goto v___jp_595_;
}
else
{
lean_object* v_a_640_; lean_object* v___x_642_; uint8_t v_isShared_643_; uint8_t v_isSharedCheck_647_; 
lean_dec_ref(v_b_589_);
v_a_640_ = lean_ctor_get(v___x_637_, 0);
v_isSharedCheck_647_ = !lean_is_exclusive(v___x_637_);
if (v_isSharedCheck_647_ == 0)
{
v___x_642_ = v___x_637_;
v_isShared_643_ = v_isSharedCheck_647_;
goto v_resetjp_641_;
}
else
{
lean_inc(v_a_640_);
lean_dec(v___x_637_);
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
}
}
else
{
lean_object* v_fvarId_648_; lean_object* v_userName_649_; lean_object* v_type_650_; lean_object* v_value_651_; uint8_t v_nondep_652_; uint8_t v_kind_653_; lean_object* v___x_654_; 
v_fvarId_648_ = lean_ctor_get(v_val_602_, 1);
v_userName_649_ = lean_ctor_get(v_val_602_, 2);
v_type_650_ = lean_ctor_get(v_val_602_, 3);
v_value_651_ = lean_ctor_get(v_val_602_, 4);
v_nondep_652_ = lean_ctor_get_uint8(v_val_602_, sizeof(void*)*5);
v_kind_653_ = lean_ctor_get_uint8(v_val_602_, sizeof(void*)*5 + 1);
lean_inc_ref(v_type_650_);
v___x_654_ = lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__1___redArg(v_type_650_, v___y_591_);
if (lean_obj_tag(v___x_654_) == 0)
{
lean_object* v_a_655_; lean_object* v___x_656_; 
v_a_655_ = lean_ctor_get(v___x_654_, 0);
lean_inc(v_a_655_);
lean_dec_ref_known(v___x_654_, 1);
lean_inc_ref(v_value_651_);
v___x_656_ = lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__1___redArg(v_value_651_, v___y_591_);
if (lean_obj_tag(v___x_656_) == 0)
{
lean_object* v_a_657_; lean_object* v___x_658_; 
v_a_657_ = lean_ctor_get(v___x_656_, 0);
lean_inc(v_a_657_);
lean_dec_ref_known(v___x_656_, 1);
lean_inc(v_userName_649_);
lean_inc(v_fvarId_648_);
v___x_658_ = l_Lean_LocalContext_mkLetDecl(v_b_589_, v_fvarId_648_, v_userName_649_, v_a_655_, v_a_657_, v_nondep_652_, v_kind_653_);
v_a_596_ = v___x_658_;
goto v___jp_595_;
}
else
{
lean_object* v_a_659_; lean_object* v___x_661_; uint8_t v_isShared_662_; uint8_t v_isSharedCheck_666_; 
lean_dec(v_a_655_);
lean_dec_ref(v_b_589_);
v_a_659_ = lean_ctor_get(v___x_656_, 0);
v_isSharedCheck_666_ = !lean_is_exclusive(v___x_656_);
if (v_isSharedCheck_666_ == 0)
{
v___x_661_ = v___x_656_;
v_isShared_662_ = v_isSharedCheck_666_;
goto v_resetjp_660_;
}
else
{
lean_inc(v_a_659_);
lean_dec(v___x_656_);
v___x_661_ = lean_box(0);
v_isShared_662_ = v_isSharedCheck_666_;
goto v_resetjp_660_;
}
v_resetjp_660_:
{
lean_object* v___x_664_; 
if (v_isShared_662_ == 0)
{
v___x_664_ = v___x_661_;
goto v_reusejp_663_;
}
else
{
lean_object* v_reuseFailAlloc_665_; 
v_reuseFailAlloc_665_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_665_, 0, v_a_659_);
v___x_664_ = v_reuseFailAlloc_665_;
goto v_reusejp_663_;
}
v_reusejp_663_:
{
return v___x_664_;
}
}
}
}
else
{
lean_object* v_a_667_; lean_object* v___x_669_; uint8_t v_isShared_670_; uint8_t v_isSharedCheck_674_; 
lean_dec_ref(v_b_589_);
v_a_667_ = lean_ctor_get(v___x_654_, 0);
v_isSharedCheck_674_ = !lean_is_exclusive(v___x_654_);
if (v_isSharedCheck_674_ == 0)
{
v___x_669_ = v___x_654_;
v_isShared_670_ = v_isSharedCheck_674_;
goto v_resetjp_668_;
}
else
{
lean_inc(v_a_667_);
lean_dec(v___x_654_);
v___x_669_ = lean_box(0);
v_isShared_670_ = v_isSharedCheck_674_;
goto v_resetjp_668_;
}
v_resetjp_668_:
{
lean_object* v___x_672_; 
if (v_isShared_670_ == 0)
{
v___x_672_ = v___x_669_;
goto v_reusejp_671_;
}
else
{
lean_object* v_reuseFailAlloc_673_; 
v_reuseFailAlloc_673_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_673_, 0, v_a_667_);
v___x_672_ = v_reuseFailAlloc_673_;
goto v_reusejp_671_;
}
v_reusejp_671_:
{
return v___x_672_;
}
}
}
}
}
}
else
{
lean_object* v___x_675_; 
v___x_675_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_675_, 0, v_b_589_);
return v___x_675_;
}
v___jp_595_:
{
size_t v___x_597_; size_t v___x_598_; 
v___x_597_ = ((size_t)1ULL);
v___x_598_ = lean_usize_add(v_i_587_, v___x_597_);
v_i_587_ = v___x_598_;
v_b_589_ = v_a_596_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34___boxed(lean_object* v_auxDeclToFullName_676_, lean_object* v_as_677_, lean_object* v_i_678_, lean_object* v_stop_679_, lean_object* v_b_680_, lean_object* v___y_681_, lean_object* v___y_682_, lean_object* v___y_683_, lean_object* v___y_684_, lean_object* v___y_685_){
_start:
{
size_t v_i_boxed_686_; size_t v_stop_boxed_687_; lean_object* v_res_688_; 
v_i_boxed_686_ = lean_unbox_usize(v_i_678_);
lean_dec(v_i_678_);
v_stop_boxed_687_ = lean_unbox_usize(v_stop_679_);
lean_dec(v_stop_679_);
v_res_688_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34(v_auxDeclToFullName_676_, v_as_677_, v_i_boxed_686_, v_stop_boxed_687_, v_b_680_, v___y_681_, v___y_682_, v___y_683_, v___y_684_);
lean_dec(v___y_684_);
lean_dec_ref(v___y_683_);
lean_dec(v___y_682_);
lean_dec_ref(v___y_681_);
lean_dec_ref(v_as_677_);
lean_dec(v_auxDeclToFullName_676_);
return v_res_688_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__35(lean_object* v_auxDeclToFullName_689_, lean_object* v_x_690_, lean_object* v_x_691_, lean_object* v___y_692_, lean_object* v___y_693_, lean_object* v___y_694_, lean_object* v___y_695_){
_start:
{
if (lean_obj_tag(v_x_690_) == 0)
{
lean_object* v_cs_697_; lean_object* v___x_699_; uint8_t v_isShared_700_; uint8_t v_isSharedCheck_717_; 
v_cs_697_ = lean_ctor_get(v_x_690_, 0);
v_isSharedCheck_717_ = !lean_is_exclusive(v_x_690_);
if (v_isSharedCheck_717_ == 0)
{
v___x_699_ = v_x_690_;
v_isShared_700_ = v_isSharedCheck_717_;
goto v_resetjp_698_;
}
else
{
lean_inc(v_cs_697_);
lean_dec(v_x_690_);
v___x_699_ = lean_box(0);
v_isShared_700_ = v_isSharedCheck_717_;
goto v_resetjp_698_;
}
v_resetjp_698_:
{
lean_object* v___x_701_; lean_object* v___x_702_; uint8_t v___x_703_; 
v___x_701_ = lean_unsigned_to_nat(0u);
v___x_702_ = lean_array_get_size(v_cs_697_);
v___x_703_ = lean_nat_dec_lt(v___x_701_, v___x_702_);
if (v___x_703_ == 0)
{
lean_object* v___x_705_; 
lean_dec_ref(v_cs_697_);
if (v_isShared_700_ == 0)
{
lean_ctor_set(v___x_699_, 0, v_x_691_);
v___x_705_ = v___x_699_;
goto v_reusejp_704_;
}
else
{
lean_object* v_reuseFailAlloc_706_; 
v_reuseFailAlloc_706_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_706_, 0, v_x_691_);
v___x_705_ = v_reuseFailAlloc_706_;
goto v_reusejp_704_;
}
v_reusejp_704_:
{
return v___x_705_;
}
}
else
{
uint8_t v___x_707_; 
v___x_707_ = lean_nat_dec_le(v___x_702_, v___x_702_);
if (v___x_707_ == 0)
{
if (v___x_703_ == 0)
{
lean_object* v___x_709_; 
lean_dec_ref(v_cs_697_);
if (v_isShared_700_ == 0)
{
lean_ctor_set(v___x_699_, 0, v_x_691_);
v___x_709_ = v___x_699_;
goto v_reusejp_708_;
}
else
{
lean_object* v_reuseFailAlloc_710_; 
v_reuseFailAlloc_710_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_710_, 0, v_x_691_);
v___x_709_ = v_reuseFailAlloc_710_;
goto v_reusejp_708_;
}
v_reusejp_708_:
{
return v___x_709_;
}
}
else
{
size_t v___x_711_; size_t v___x_712_; lean_object* v___x_713_; 
lean_del_object(v___x_699_);
v___x_711_ = ((size_t)0ULL);
v___x_712_ = lean_usize_of_nat(v___x_702_);
v___x_713_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__33_spec__35(v_auxDeclToFullName_689_, v_cs_697_, v___x_711_, v___x_712_, v_x_691_, v___y_692_, v___y_693_, v___y_694_, v___y_695_);
lean_dec_ref(v_cs_697_);
return v___x_713_;
}
}
else
{
size_t v___x_714_; size_t v___x_715_; lean_object* v___x_716_; 
lean_del_object(v___x_699_);
v___x_714_ = ((size_t)0ULL);
v___x_715_ = lean_usize_of_nat(v___x_702_);
v___x_716_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__33_spec__35(v_auxDeclToFullName_689_, v_cs_697_, v___x_714_, v___x_715_, v_x_691_, v___y_692_, v___y_693_, v___y_694_, v___y_695_);
lean_dec_ref(v_cs_697_);
return v___x_716_;
}
}
}
}
else
{
lean_object* v_vs_718_; lean_object* v___x_720_; uint8_t v_isShared_721_; uint8_t v_isSharedCheck_738_; 
v_vs_718_ = lean_ctor_get(v_x_690_, 0);
v_isSharedCheck_738_ = !lean_is_exclusive(v_x_690_);
if (v_isSharedCheck_738_ == 0)
{
v___x_720_ = v_x_690_;
v_isShared_721_ = v_isSharedCheck_738_;
goto v_resetjp_719_;
}
else
{
lean_inc(v_vs_718_);
lean_dec(v_x_690_);
v___x_720_ = lean_box(0);
v_isShared_721_ = v_isSharedCheck_738_;
goto v_resetjp_719_;
}
v_resetjp_719_:
{
lean_object* v___x_722_; lean_object* v___x_723_; uint8_t v___x_724_; 
v___x_722_ = lean_unsigned_to_nat(0u);
v___x_723_ = lean_array_get_size(v_vs_718_);
v___x_724_ = lean_nat_dec_lt(v___x_722_, v___x_723_);
if (v___x_724_ == 0)
{
lean_object* v___x_726_; 
lean_dec_ref(v_vs_718_);
if (v_isShared_721_ == 0)
{
lean_ctor_set_tag(v___x_720_, 0);
lean_ctor_set(v___x_720_, 0, v_x_691_);
v___x_726_ = v___x_720_;
goto v_reusejp_725_;
}
else
{
lean_object* v_reuseFailAlloc_727_; 
v_reuseFailAlloc_727_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_727_, 0, v_x_691_);
v___x_726_ = v_reuseFailAlloc_727_;
goto v_reusejp_725_;
}
v_reusejp_725_:
{
return v___x_726_;
}
}
else
{
uint8_t v___x_728_; 
v___x_728_ = lean_nat_dec_le(v___x_723_, v___x_723_);
if (v___x_728_ == 0)
{
if (v___x_724_ == 0)
{
lean_object* v___x_730_; 
lean_dec_ref(v_vs_718_);
if (v_isShared_721_ == 0)
{
lean_ctor_set_tag(v___x_720_, 0);
lean_ctor_set(v___x_720_, 0, v_x_691_);
v___x_730_ = v___x_720_;
goto v_reusejp_729_;
}
else
{
lean_object* v_reuseFailAlloc_731_; 
v_reuseFailAlloc_731_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_731_, 0, v_x_691_);
v___x_730_ = v_reuseFailAlloc_731_;
goto v_reusejp_729_;
}
v_reusejp_729_:
{
return v___x_730_;
}
}
else
{
size_t v___x_732_; size_t v___x_733_; lean_object* v___x_734_; 
lean_del_object(v___x_720_);
v___x_732_ = ((size_t)0ULL);
v___x_733_ = lean_usize_of_nat(v___x_723_);
v___x_734_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34(v_auxDeclToFullName_689_, v_vs_718_, v___x_732_, v___x_733_, v_x_691_, v___y_692_, v___y_693_, v___y_694_, v___y_695_);
lean_dec_ref(v_vs_718_);
return v___x_734_;
}
}
else
{
size_t v___x_735_; size_t v___x_736_; lean_object* v___x_737_; 
lean_del_object(v___x_720_);
v___x_735_ = ((size_t)0ULL);
v___x_736_ = lean_usize_of_nat(v___x_723_);
v___x_737_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34(v_auxDeclToFullName_689_, v_vs_718_, v___x_735_, v___x_736_, v_x_691_, v___y_692_, v___y_693_, v___y_694_, v___y_695_);
lean_dec_ref(v_vs_718_);
return v___x_737_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__33_spec__35(lean_object* v_auxDeclToFullName_739_, lean_object* v_as_740_, size_t v_i_741_, size_t v_stop_742_, lean_object* v_b_743_, lean_object* v___y_744_, lean_object* v___y_745_, lean_object* v___y_746_, lean_object* v___y_747_){
_start:
{
uint8_t v___x_749_; 
v___x_749_ = lean_usize_dec_eq(v_i_741_, v_stop_742_);
if (v___x_749_ == 0)
{
lean_object* v___x_750_; lean_object* v___x_751_; 
v___x_750_ = lean_array_uget_borrowed(v_as_740_, v_i_741_);
lean_inc(v___x_750_);
v___x_751_ = lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__35(v_auxDeclToFullName_739_, v___x_750_, v_b_743_, v___y_744_, v___y_745_, v___y_746_, v___y_747_);
if (lean_obj_tag(v___x_751_) == 0)
{
lean_object* v_a_752_; size_t v___x_753_; size_t v___x_754_; 
v_a_752_ = lean_ctor_get(v___x_751_, 0);
lean_inc(v_a_752_);
lean_dec_ref_known(v___x_751_, 1);
v___x_753_ = ((size_t)1ULL);
v___x_754_ = lean_usize_add(v_i_741_, v___x_753_);
v_i_741_ = v___x_754_;
v_b_743_ = v_a_752_;
goto _start;
}
else
{
return v___x_751_;
}
}
else
{
lean_object* v___x_756_; 
v___x_756_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_756_, 0, v_b_743_);
return v___x_756_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__33_spec__35___boxed(lean_object* v_auxDeclToFullName_757_, lean_object* v_as_758_, lean_object* v_i_759_, lean_object* v_stop_760_, lean_object* v_b_761_, lean_object* v___y_762_, lean_object* v___y_763_, lean_object* v___y_764_, lean_object* v___y_765_, lean_object* v___y_766_){
_start:
{
size_t v_i_boxed_767_; size_t v_stop_boxed_768_; lean_object* v_res_769_; 
v_i_boxed_767_ = lean_unbox_usize(v_i_759_);
lean_dec(v_i_759_);
v_stop_boxed_768_ = lean_unbox_usize(v_stop_760_);
lean_dec(v_stop_760_);
v_res_769_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__33_spec__35(v_auxDeclToFullName_757_, v_as_758_, v_i_boxed_767_, v_stop_boxed_768_, v_b_761_, v___y_762_, v___y_763_, v___y_764_, v___y_765_);
lean_dec(v___y_765_);
lean_dec_ref(v___y_764_);
lean_dec(v___y_763_);
lean_dec_ref(v___y_762_);
lean_dec_ref(v_as_758_);
lean_dec(v_auxDeclToFullName_757_);
return v_res_769_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__35___boxed(lean_object* v_auxDeclToFullName_770_, lean_object* v_x_771_, lean_object* v_x_772_, lean_object* v___y_773_, lean_object* v___y_774_, lean_object* v___y_775_, lean_object* v___y_776_, lean_object* v___y_777_){
_start:
{
lean_object* v_res_778_; 
v_res_778_ = lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__35(v_auxDeclToFullName_770_, v_x_771_, v_x_772_, v___y_773_, v___y_774_, v___y_775_, v___y_776_);
lean_dec(v___y_776_);
lean_dec_ref(v___y_775_);
lean_dec(v___y_774_);
lean_dec_ref(v___y_773_);
lean_dec(v_auxDeclToFullName_770_);
return v_res_778_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__33___closed__0(void){
_start:
{
lean_object* v___x_779_; 
v___x_779_ = l_Lean_instInhabitedPersistentArrayNode_default(lean_box(0));
return v___x_779_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__33(lean_object* v_auxDeclToFullName_780_, lean_object* v_x_781_, size_t v_x_782_, size_t v_x_783_, lean_object* v_x_784_, lean_object* v___y_785_, lean_object* v___y_786_, lean_object* v___y_787_, lean_object* v___y_788_){
_start:
{
if (lean_obj_tag(v_x_781_) == 0)
{
lean_object* v_cs_790_; lean_object* v___x_791_; size_t v___x_792_; lean_object* v_j_793_; lean_object* v___x_794_; size_t v___x_795_; size_t v___x_796_; size_t v___x_797_; size_t v___x_798_; size_t v___x_799_; size_t v___x_800_; lean_object* v___x_801_; 
v_cs_790_ = lean_ctor_get(v_x_781_, 0);
lean_inc_ref(v_cs_790_);
lean_dec_ref_known(v_x_781_, 1);
v___x_791_ = lean_obj_once(&lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__33___closed__0, &lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__33___closed__0_once, _init_lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__33___closed__0);
v___x_792_ = lean_usize_shift_right(v_x_782_, v_x_783_);
v_j_793_ = lean_usize_to_nat(v___x_792_);
v___x_794_ = lean_array_get_borrowed(v___x_791_, v_cs_790_, v_j_793_);
v___x_795_ = ((size_t)1ULL);
v___x_796_ = lean_usize_shift_left(v___x_795_, v_x_783_);
v___x_797_ = lean_usize_sub(v___x_796_, v___x_795_);
v___x_798_ = lean_usize_land(v_x_782_, v___x_797_);
v___x_799_ = ((size_t)5ULL);
v___x_800_ = lean_usize_sub(v_x_783_, v___x_799_);
lean_inc(v___x_794_);
v___x_801_ = lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__33(v_auxDeclToFullName_780_, v___x_794_, v___x_798_, v___x_800_, v_x_784_, v___y_785_, v___y_786_, v___y_787_, v___y_788_);
if (lean_obj_tag(v___x_801_) == 0)
{
lean_object* v_a_802_; lean_object* v___x_803_; lean_object* v___x_804_; lean_object* v___x_805_; uint8_t v___x_806_; 
v_a_802_ = lean_ctor_get(v___x_801_, 0);
lean_inc(v_a_802_);
v___x_803_ = lean_unsigned_to_nat(1u);
v___x_804_ = lean_nat_add(v_j_793_, v___x_803_);
lean_dec(v_j_793_);
v___x_805_ = lean_array_get_size(v_cs_790_);
v___x_806_ = lean_nat_dec_lt(v___x_804_, v___x_805_);
if (v___x_806_ == 0)
{
lean_dec(v___x_804_);
lean_dec(v_a_802_);
lean_dec_ref(v_cs_790_);
return v___x_801_;
}
else
{
uint8_t v___x_807_; 
v___x_807_ = lean_nat_dec_le(v___x_805_, v___x_805_);
if (v___x_807_ == 0)
{
if (v___x_806_ == 0)
{
lean_dec(v___x_804_);
lean_dec(v_a_802_);
lean_dec_ref(v_cs_790_);
return v___x_801_;
}
else
{
size_t v___x_808_; size_t v___x_809_; lean_object* v___x_810_; 
lean_dec_ref_known(v___x_801_, 1);
v___x_808_ = lean_usize_of_nat(v___x_804_);
lean_dec(v___x_804_);
v___x_809_ = lean_usize_of_nat(v___x_805_);
v___x_810_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__33_spec__35(v_auxDeclToFullName_780_, v_cs_790_, v___x_808_, v___x_809_, v_a_802_, v___y_785_, v___y_786_, v___y_787_, v___y_788_);
lean_dec_ref(v_cs_790_);
return v___x_810_;
}
}
else
{
size_t v___x_811_; size_t v___x_812_; lean_object* v___x_813_; 
lean_dec_ref_known(v___x_801_, 1);
v___x_811_ = lean_usize_of_nat(v___x_804_);
lean_dec(v___x_804_);
v___x_812_ = lean_usize_of_nat(v___x_805_);
v___x_813_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__33_spec__35(v_auxDeclToFullName_780_, v_cs_790_, v___x_811_, v___x_812_, v_a_802_, v___y_785_, v___y_786_, v___y_787_, v___y_788_);
lean_dec_ref(v_cs_790_);
return v___x_813_;
}
}
}
else
{
lean_dec(v_j_793_);
lean_dec_ref(v_cs_790_);
return v___x_801_;
}
}
else
{
lean_object* v_vs_814_; lean_object* v___x_816_; uint8_t v_isShared_817_; uint8_t v_isSharedCheck_834_; 
v_vs_814_ = lean_ctor_get(v_x_781_, 0);
v_isSharedCheck_834_ = !lean_is_exclusive(v_x_781_);
if (v_isSharedCheck_834_ == 0)
{
v___x_816_ = v_x_781_;
v_isShared_817_ = v_isSharedCheck_834_;
goto v_resetjp_815_;
}
else
{
lean_inc(v_vs_814_);
lean_dec(v_x_781_);
v___x_816_ = lean_box(0);
v_isShared_817_ = v_isSharedCheck_834_;
goto v_resetjp_815_;
}
v_resetjp_815_:
{
lean_object* v___x_818_; lean_object* v___x_819_; uint8_t v___x_820_; 
v___x_818_ = lean_usize_to_nat(v_x_782_);
v___x_819_ = lean_array_get_size(v_vs_814_);
v___x_820_ = lean_nat_dec_lt(v___x_818_, v___x_819_);
if (v___x_820_ == 0)
{
lean_object* v___x_822_; 
lean_dec(v___x_818_);
lean_dec_ref(v_vs_814_);
if (v_isShared_817_ == 0)
{
lean_ctor_set_tag(v___x_816_, 0);
lean_ctor_set(v___x_816_, 0, v_x_784_);
v___x_822_ = v___x_816_;
goto v_reusejp_821_;
}
else
{
lean_object* v_reuseFailAlloc_823_; 
v_reuseFailAlloc_823_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_823_, 0, v_x_784_);
v___x_822_ = v_reuseFailAlloc_823_;
goto v_reusejp_821_;
}
v_reusejp_821_:
{
return v___x_822_;
}
}
else
{
uint8_t v___x_824_; 
v___x_824_ = lean_nat_dec_le(v___x_819_, v___x_819_);
if (v___x_824_ == 0)
{
if (v___x_820_ == 0)
{
lean_object* v___x_826_; 
lean_dec(v___x_818_);
lean_dec_ref(v_vs_814_);
if (v_isShared_817_ == 0)
{
lean_ctor_set_tag(v___x_816_, 0);
lean_ctor_set(v___x_816_, 0, v_x_784_);
v___x_826_ = v___x_816_;
goto v_reusejp_825_;
}
else
{
lean_object* v_reuseFailAlloc_827_; 
v_reuseFailAlloc_827_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_827_, 0, v_x_784_);
v___x_826_ = v_reuseFailAlloc_827_;
goto v_reusejp_825_;
}
v_reusejp_825_:
{
return v___x_826_;
}
}
else
{
size_t v___x_828_; size_t v___x_829_; lean_object* v___x_830_; 
lean_del_object(v___x_816_);
v___x_828_ = lean_usize_of_nat(v___x_818_);
lean_dec(v___x_818_);
v___x_829_ = lean_usize_of_nat(v___x_819_);
v___x_830_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34(v_auxDeclToFullName_780_, v_vs_814_, v___x_828_, v___x_829_, v_x_784_, v___y_785_, v___y_786_, v___y_787_, v___y_788_);
lean_dec_ref(v_vs_814_);
return v___x_830_;
}
}
else
{
size_t v___x_831_; size_t v___x_832_; lean_object* v___x_833_; 
lean_del_object(v___x_816_);
v___x_831_ = lean_usize_of_nat(v___x_818_);
lean_dec(v___x_818_);
v___x_832_ = lean_usize_of_nat(v___x_819_);
v___x_833_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34(v_auxDeclToFullName_780_, v_vs_814_, v___x_831_, v___x_832_, v_x_784_, v___y_785_, v___y_786_, v___y_787_, v___y_788_);
lean_dec_ref(v_vs_814_);
return v___x_833_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__33___boxed(lean_object* v_auxDeclToFullName_835_, lean_object* v_x_836_, lean_object* v_x_837_, lean_object* v_x_838_, lean_object* v_x_839_, lean_object* v___y_840_, lean_object* v___y_841_, lean_object* v___y_842_, lean_object* v___y_843_, lean_object* v___y_844_){
_start:
{
size_t v_x_17433__boxed_845_; size_t v_x_17434__boxed_846_; lean_object* v_res_847_; 
v_x_17433__boxed_845_ = lean_unbox_usize(v_x_837_);
lean_dec(v_x_837_);
v_x_17434__boxed_846_ = lean_unbox_usize(v_x_838_);
lean_dec(v_x_838_);
v_res_847_ = lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__33(v_auxDeclToFullName_835_, v_x_836_, v_x_17433__boxed_845_, v_x_17434__boxed_846_, v_x_839_, v___y_840_, v___y_841_, v___y_842_, v___y_843_);
lean_dec(v___y_843_);
lean_dec_ref(v___y_842_);
lean_dec(v___y_841_);
lean_dec_ref(v___y_840_);
lean_dec(v_auxDeclToFullName_835_);
return v_res_847_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30(lean_object* v_auxDeclToFullName_848_, lean_object* v_t_849_, lean_object* v_init_850_, lean_object* v_start_851_, lean_object* v___y_852_, lean_object* v___y_853_, lean_object* v___y_854_, lean_object* v___y_855_){
_start:
{
lean_object* v___x_857_; uint8_t v___x_858_; 
v___x_857_ = lean_unsigned_to_nat(0u);
v___x_858_ = lean_nat_dec_eq(v_start_851_, v___x_857_);
if (v___x_858_ == 0)
{
lean_object* v_root_859_; lean_object* v_tail_860_; size_t v_shift_861_; lean_object* v_tailOff_862_; uint8_t v___x_863_; 
v_root_859_ = lean_ctor_get(v_t_849_, 0);
lean_inc_ref(v_root_859_);
v_tail_860_ = lean_ctor_get(v_t_849_, 1);
lean_inc_ref(v_tail_860_);
v_shift_861_ = lean_ctor_get_usize(v_t_849_, 4);
v_tailOff_862_ = lean_ctor_get(v_t_849_, 3);
lean_inc(v_tailOff_862_);
lean_dec_ref(v_t_849_);
v___x_863_ = lean_nat_dec_le(v_tailOff_862_, v_start_851_);
if (v___x_863_ == 0)
{
size_t v___x_864_; lean_object* v___x_865_; 
lean_dec(v_tailOff_862_);
v___x_864_ = lean_usize_of_nat(v_start_851_);
v___x_865_ = lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__33(v_auxDeclToFullName_848_, v_root_859_, v___x_864_, v_shift_861_, v_init_850_, v___y_852_, v___y_853_, v___y_854_, v___y_855_);
if (lean_obj_tag(v___x_865_) == 0)
{
lean_object* v_a_866_; lean_object* v___x_867_; uint8_t v___x_868_; 
v_a_866_ = lean_ctor_get(v___x_865_, 0);
lean_inc(v_a_866_);
v___x_867_ = lean_array_get_size(v_tail_860_);
v___x_868_ = lean_nat_dec_lt(v___x_857_, v___x_867_);
if (v___x_868_ == 0)
{
lean_dec(v_a_866_);
lean_dec_ref(v_tail_860_);
return v___x_865_;
}
else
{
uint8_t v___x_869_; 
v___x_869_ = lean_nat_dec_le(v___x_867_, v___x_867_);
if (v___x_869_ == 0)
{
if (v___x_868_ == 0)
{
lean_dec(v_a_866_);
lean_dec_ref(v_tail_860_);
return v___x_865_;
}
else
{
size_t v___x_870_; size_t v___x_871_; lean_object* v___x_872_; 
lean_dec_ref_known(v___x_865_, 1);
v___x_870_ = ((size_t)0ULL);
v___x_871_ = lean_usize_of_nat(v___x_867_);
v___x_872_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34(v_auxDeclToFullName_848_, v_tail_860_, v___x_870_, v___x_871_, v_a_866_, v___y_852_, v___y_853_, v___y_854_, v___y_855_);
lean_dec_ref(v_tail_860_);
return v___x_872_;
}
}
else
{
size_t v___x_873_; size_t v___x_874_; lean_object* v___x_875_; 
lean_dec_ref_known(v___x_865_, 1);
v___x_873_ = ((size_t)0ULL);
v___x_874_ = lean_usize_of_nat(v___x_867_);
v___x_875_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34(v_auxDeclToFullName_848_, v_tail_860_, v___x_873_, v___x_874_, v_a_866_, v___y_852_, v___y_853_, v___y_854_, v___y_855_);
lean_dec_ref(v_tail_860_);
return v___x_875_;
}
}
}
else
{
lean_dec_ref(v_tail_860_);
return v___x_865_;
}
}
else
{
lean_object* v___x_876_; lean_object* v___x_877_; uint8_t v___x_878_; 
lean_dec_ref(v_root_859_);
v___x_876_ = lean_nat_sub(v_start_851_, v_tailOff_862_);
lean_dec(v_tailOff_862_);
v___x_877_ = lean_array_get_size(v_tail_860_);
v___x_878_ = lean_nat_dec_lt(v___x_876_, v___x_877_);
if (v___x_878_ == 0)
{
lean_object* v___x_879_; 
lean_dec(v___x_876_);
lean_dec_ref(v_tail_860_);
v___x_879_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_879_, 0, v_init_850_);
return v___x_879_;
}
else
{
uint8_t v___x_880_; 
v___x_880_ = lean_nat_dec_le(v___x_877_, v___x_877_);
if (v___x_880_ == 0)
{
if (v___x_878_ == 0)
{
lean_object* v___x_881_; 
lean_dec(v___x_876_);
lean_dec_ref(v_tail_860_);
v___x_881_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_881_, 0, v_init_850_);
return v___x_881_;
}
else
{
size_t v___x_882_; size_t v___x_883_; lean_object* v___x_884_; 
v___x_882_ = lean_usize_of_nat(v___x_876_);
lean_dec(v___x_876_);
v___x_883_ = lean_usize_of_nat(v___x_877_);
v___x_884_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34(v_auxDeclToFullName_848_, v_tail_860_, v___x_882_, v___x_883_, v_init_850_, v___y_852_, v___y_853_, v___y_854_, v___y_855_);
lean_dec_ref(v_tail_860_);
return v___x_884_;
}
}
else
{
size_t v___x_885_; size_t v___x_886_; lean_object* v___x_887_; 
v___x_885_ = lean_usize_of_nat(v___x_876_);
lean_dec(v___x_876_);
v___x_886_ = lean_usize_of_nat(v___x_877_);
v___x_887_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34(v_auxDeclToFullName_848_, v_tail_860_, v___x_885_, v___x_886_, v_init_850_, v___y_852_, v___y_853_, v___y_854_, v___y_855_);
lean_dec_ref(v_tail_860_);
return v___x_887_;
}
}
}
}
else
{
lean_object* v_root_888_; lean_object* v_tail_889_; lean_object* v___x_890_; 
v_root_888_ = lean_ctor_get(v_t_849_, 0);
lean_inc_ref(v_root_888_);
v_tail_889_ = lean_ctor_get(v_t_849_, 1);
lean_inc_ref(v_tail_889_);
lean_dec_ref(v_t_849_);
v___x_890_ = lp_aesop___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__35(v_auxDeclToFullName_848_, v_root_888_, v_init_850_, v___y_852_, v___y_853_, v___y_854_, v___y_855_);
if (lean_obj_tag(v___x_890_) == 0)
{
lean_object* v_a_891_; lean_object* v___x_892_; uint8_t v___x_893_; 
v_a_891_ = lean_ctor_get(v___x_890_, 0);
lean_inc(v_a_891_);
v___x_892_ = lean_array_get_size(v_tail_889_);
v___x_893_ = lean_nat_dec_lt(v___x_857_, v___x_892_);
if (v___x_893_ == 0)
{
lean_dec(v_a_891_);
lean_dec_ref(v_tail_889_);
return v___x_890_;
}
else
{
uint8_t v___x_894_; 
v___x_894_ = lean_nat_dec_le(v___x_892_, v___x_892_);
if (v___x_894_ == 0)
{
if (v___x_893_ == 0)
{
lean_dec(v_a_891_);
lean_dec_ref(v_tail_889_);
return v___x_890_;
}
else
{
size_t v___x_895_; size_t v___x_896_; lean_object* v___x_897_; 
lean_dec_ref_known(v___x_890_, 1);
v___x_895_ = ((size_t)0ULL);
v___x_896_ = lean_usize_of_nat(v___x_892_);
v___x_897_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34(v_auxDeclToFullName_848_, v_tail_889_, v___x_895_, v___x_896_, v_a_891_, v___y_852_, v___y_853_, v___y_854_, v___y_855_);
lean_dec_ref(v_tail_889_);
return v___x_897_;
}
}
else
{
size_t v___x_898_; size_t v___x_899_; lean_object* v___x_900_; 
lean_dec_ref_known(v___x_890_, 1);
v___x_898_ = ((size_t)0ULL);
v___x_899_ = lean_usize_of_nat(v___x_892_);
v___x_900_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30_spec__34(v_auxDeclToFullName_848_, v_tail_889_, v___x_898_, v___x_899_, v_a_891_, v___y_852_, v___y_853_, v___y_854_, v___y_855_);
lean_dec_ref(v_tail_889_);
return v___x_900_;
}
}
}
else
{
lean_dec_ref(v_tail_889_);
return v___x_890_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30___boxed(lean_object* v_auxDeclToFullName_901_, lean_object* v_t_902_, lean_object* v_init_903_, lean_object* v_start_904_, lean_object* v___y_905_, lean_object* v___y_906_, lean_object* v___y_907_, lean_object* v___y_908_, lean_object* v___y_909_){
_start:
{
lean_object* v_res_910_; 
v_res_910_ = lp_aesop_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30(v_auxDeclToFullName_901_, v_t_902_, v_init_903_, v_start_904_, v___y_905_, v___y_906_, v___y_907_, v___y_908_);
lean_dec(v___y_908_);
lean_dec_ref(v___y_907_);
lean_dec(v___y_906_);
lean_dec_ref(v___y_905_);
lean_dec(v_start_904_);
lean_dec(v_auxDeclToFullName_901_);
return v_res_910_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27(lean_object* v_auxDeclToFullName_911_, lean_object* v_lctx_912_, lean_object* v_init_913_, lean_object* v_start_914_, lean_object* v___y_915_, lean_object* v___y_916_, lean_object* v___y_917_, lean_object* v___y_918_){
_start:
{
lean_object* v_decls_920_; lean_object* v___x_921_; 
v_decls_920_ = lean_ctor_get(v_lctx_912_, 1);
lean_inc_ref(v_decls_920_);
lean_dec_ref(v_lctx_912_);
v___x_921_ = lp_aesop_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27_spec__30(v_auxDeclToFullName_911_, v_decls_920_, v_init_913_, v_start_914_, v___y_915_, v___y_916_, v___y_917_, v___y_918_);
return v___x_921_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27___boxed(lean_object* v_auxDeclToFullName_922_, lean_object* v_lctx_923_, lean_object* v_init_924_, lean_object* v_start_925_, lean_object* v___y_926_, lean_object* v___y_927_, lean_object* v___y_928_, lean_object* v___y_929_, lean_object* v___y_930_){
_start:
{
lean_object* v_res_931_; 
v_res_931_ = lp_aesop_Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27(v_auxDeclToFullName_922_, v_lctx_923_, v_init_924_, v_start_925_, v___y_926_, v___y_927_, v___y_928_, v___y_929_);
lean_dec(v___y_929_);
lean_dec_ref(v___y_928_);
lean_dec(v___y_927_);
lean_dec_ref(v___y_926_);
lean_dec(v_start_925_);
lean_dec(v_auxDeclToFullName_922_);
return v_res_931_;
}
}
static lean_object* _init_lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__0(void){
_start:
{
lean_object* v___x_932_; 
v___x_932_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_932_;
}
}
static lean_object* _init_lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__1(void){
_start:
{
lean_object* v___x_933_; lean_object* v___x_934_; 
v___x_933_ = lean_obj_once(&lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__0, &lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__0_once, _init_lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__0);
v___x_934_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_934_, 0, v___x_933_);
return v___x_934_;
}
}
static lean_object* _init_lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__2(void){
_start:
{
lean_object* v___x_935_; lean_object* v___x_936_; lean_object* v___x_937_; 
v___x_935_ = lean_unsigned_to_nat(32u);
v___x_936_ = lean_mk_empty_array_with_capacity(v___x_935_);
v___x_937_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_937_, 0, v___x_936_);
return v___x_937_;
}
}
static lean_object* _init_lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__3(void){
_start:
{
size_t v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; lean_object* v___x_942_; lean_object* v___x_943_; 
v___x_938_ = ((size_t)5ULL);
v___x_939_ = lean_unsigned_to_nat(0u);
v___x_940_ = lean_unsigned_to_nat(32u);
v___x_941_ = lean_mk_empty_array_with_capacity(v___x_940_);
v___x_942_ = lean_obj_once(&lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__2, &lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__2_once, _init_lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__2);
v___x_943_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_943_, 0, v___x_942_);
lean_ctor_set(v___x_943_, 1, v___x_941_);
lean_ctor_set(v___x_943_, 2, v___x_939_);
lean_ctor_set(v___x_943_, 3, v___x_939_);
lean_ctor_set_usize(v___x_943_, 4, v___x_938_);
return v___x_943_;
}
}
static lean_object* _init_lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__4(void){
_start:
{
lean_object* v___x_944_; lean_object* v___x_945_; lean_object* v___x_946_; lean_object* v___x_947_; 
v___x_944_ = lean_box(1);
v___x_945_ = lean_obj_once(&lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__3, &lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__3_once, _init_lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__3);
v___x_946_ = lean_obj_once(&lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__1, &lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__1_once, _init_lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__1);
v___x_947_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_947_, 0, v___x_946_);
lean_ctor_set(v___x_947_, 1, v___x_945_);
lean_ctor_set(v___x_947_, 2, v___x_944_);
return v___x_947_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19(lean_object* v_lctx_948_, lean_object* v___y_949_, lean_object* v___y_950_, lean_object* v___y_951_, lean_object* v___y_952_){
_start:
{
lean_object* v_auxDeclToFullName_954_; lean_object* v___x_955_; lean_object* v___x_956_; lean_object* v___x_957_; 
v_auxDeclToFullName_954_ = lean_ctor_get(v_lctx_948_, 2);
lean_inc(v_auxDeclToFullName_954_);
v___x_955_ = lean_unsigned_to_nat(0u);
v___x_956_ = lean_obj_once(&lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__4, &lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__4_once, _init_lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___closed__4);
v___x_957_ = lp_aesop_Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__27(v_auxDeclToFullName_954_, v_lctx_948_, v___x_956_, v___x_955_, v___y_949_, v___y_950_, v___y_951_, v___y_952_);
lean_dec(v_auxDeclToFullName_954_);
return v___x_957_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19___boxed(lean_object* v_lctx_958_, lean_object* v___y_959_, lean_object* v___y_960_, lean_object* v___y_961_, lean_object* v___y_962_, lean_object* v___y_963_){
_start:
{
lean_object* v_res_964_; 
v_res_964_ = lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19(v_lctx_958_, v___y_959_, v___y_960_, v___y_961_, v___y_962_);
lean_dec(v___y_962_);
lean_dec_ref(v___y_961_);
lean_dec(v___y_960_);
lean_dec_ref(v___y_959_);
return v_res_964_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17_spec__22_spec__30___redArg(lean_object* v_x_965_, lean_object* v_x_966_, lean_object* v_x_967_, lean_object* v_x_968_){
_start:
{
lean_object* v_ks_969_; lean_object* v_vs_970_; lean_object* v___x_972_; uint8_t v_isShared_973_; uint8_t v_isSharedCheck_994_; 
v_ks_969_ = lean_ctor_get(v_x_965_, 0);
v_vs_970_ = lean_ctor_get(v_x_965_, 1);
v_isSharedCheck_994_ = !lean_is_exclusive(v_x_965_);
if (v_isSharedCheck_994_ == 0)
{
v___x_972_ = v_x_965_;
v_isShared_973_ = v_isSharedCheck_994_;
goto v_resetjp_971_;
}
else
{
lean_inc(v_vs_970_);
lean_inc(v_ks_969_);
lean_dec(v_x_965_);
v___x_972_ = lean_box(0);
v_isShared_973_ = v_isSharedCheck_994_;
goto v_resetjp_971_;
}
v_resetjp_971_:
{
lean_object* v___x_974_; uint8_t v___x_975_; 
v___x_974_ = lean_array_get_size(v_ks_969_);
v___x_975_ = lean_nat_dec_lt(v_x_966_, v___x_974_);
if (v___x_975_ == 0)
{
lean_object* v___x_976_; lean_object* v___x_977_; lean_object* v___x_979_; 
lean_dec(v_x_966_);
v___x_976_ = lean_array_push(v_ks_969_, v_x_967_);
v___x_977_ = lean_array_push(v_vs_970_, v_x_968_);
if (v_isShared_973_ == 0)
{
lean_ctor_set(v___x_972_, 1, v___x_977_);
lean_ctor_set(v___x_972_, 0, v___x_976_);
v___x_979_ = v___x_972_;
goto v_reusejp_978_;
}
else
{
lean_object* v_reuseFailAlloc_980_; 
v_reuseFailAlloc_980_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_980_, 0, v___x_976_);
lean_ctor_set(v_reuseFailAlloc_980_, 1, v___x_977_);
v___x_979_ = v_reuseFailAlloc_980_;
goto v_reusejp_978_;
}
v_reusejp_978_:
{
return v___x_979_;
}
}
else
{
lean_object* v_k_x27_981_; uint8_t v___x_982_; 
v_k_x27_981_ = lean_array_fget_borrowed(v_ks_969_, v_x_966_);
v___x_982_ = l_Lean_instBEqMVarId_beq(v_x_967_, v_k_x27_981_);
if (v___x_982_ == 0)
{
lean_object* v___x_984_; 
if (v_isShared_973_ == 0)
{
v___x_984_ = v___x_972_;
goto v_reusejp_983_;
}
else
{
lean_object* v_reuseFailAlloc_988_; 
v_reuseFailAlloc_988_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_988_, 0, v_ks_969_);
lean_ctor_set(v_reuseFailAlloc_988_, 1, v_vs_970_);
v___x_984_ = v_reuseFailAlloc_988_;
goto v_reusejp_983_;
}
v_reusejp_983_:
{
lean_object* v___x_985_; lean_object* v___x_986_; 
v___x_985_ = lean_unsigned_to_nat(1u);
v___x_986_ = lean_nat_add(v_x_966_, v___x_985_);
lean_dec(v_x_966_);
v_x_965_ = v___x_984_;
v_x_966_ = v___x_986_;
goto _start;
}
}
else
{
lean_object* v___x_989_; lean_object* v___x_990_; lean_object* v___x_992_; 
v___x_989_ = lean_array_fset(v_ks_969_, v_x_966_, v_x_967_);
v___x_990_ = lean_array_fset(v_vs_970_, v_x_966_, v_x_968_);
lean_dec(v_x_966_);
if (v_isShared_973_ == 0)
{
lean_ctor_set(v___x_972_, 1, v___x_990_);
lean_ctor_set(v___x_972_, 0, v___x_989_);
v___x_992_ = v___x_972_;
goto v_reusejp_991_;
}
else
{
lean_object* v_reuseFailAlloc_993_; 
v_reuseFailAlloc_993_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_993_, 0, v___x_989_);
lean_ctor_set(v_reuseFailAlloc_993_, 1, v___x_990_);
v___x_992_ = v_reuseFailAlloc_993_;
goto v_reusejp_991_;
}
v_reusejp_991_:
{
return v___x_992_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17_spec__22___redArg(lean_object* v_n_995_, lean_object* v_k_996_, lean_object* v_v_997_){
_start:
{
lean_object* v___x_998_; lean_object* v___x_999_; 
v___x_998_ = lean_unsigned_to_nat(0u);
v___x_999_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17_spec__22_spec__30___redArg(v_n_995_, v___x_998_, v_k_996_, v_v_997_);
return v___x_999_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17___redArg___closed__0(void){
_start:
{
lean_object* v___x_1000_; 
v___x_1000_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_1000_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17___redArg(lean_object* v_x_1001_, size_t v_x_1002_, size_t v_x_1003_, lean_object* v_x_1004_, lean_object* v_x_1005_){
_start:
{
if (lean_obj_tag(v_x_1001_) == 0)
{
lean_object* v_es_1006_; size_t v___x_1007_; size_t v___x_1008_; lean_object* v_j_1009_; lean_object* v___x_1010_; uint8_t v___x_1011_; 
v_es_1006_ = lean_ctor_get(v_x_1001_, 0);
v___x_1007_ = ((size_t)31ULL);
v___x_1008_ = lean_usize_land(v_x_1002_, v___x_1007_);
v_j_1009_ = lean_usize_to_nat(v___x_1008_);
v___x_1010_ = lean_array_get_size(v_es_1006_);
v___x_1011_ = lean_nat_dec_lt(v_j_1009_, v___x_1010_);
if (v___x_1011_ == 0)
{
lean_dec(v_j_1009_);
lean_dec(v_x_1005_);
lean_dec(v_x_1004_);
return v_x_1001_;
}
else
{
lean_object* v___x_1013_; uint8_t v_isShared_1014_; uint8_t v_isSharedCheck_1050_; 
lean_inc_ref(v_es_1006_);
v_isSharedCheck_1050_ = !lean_is_exclusive(v_x_1001_);
if (v_isSharedCheck_1050_ == 0)
{
lean_object* v_unused_1051_; 
v_unused_1051_ = lean_ctor_get(v_x_1001_, 0);
lean_dec(v_unused_1051_);
v___x_1013_ = v_x_1001_;
v_isShared_1014_ = v_isSharedCheck_1050_;
goto v_resetjp_1012_;
}
else
{
lean_dec(v_x_1001_);
v___x_1013_ = lean_box(0);
v_isShared_1014_ = v_isSharedCheck_1050_;
goto v_resetjp_1012_;
}
v_resetjp_1012_:
{
lean_object* v_v_1015_; lean_object* v___x_1016_; lean_object* v_xs_x27_1017_; lean_object* v___y_1019_; 
v_v_1015_ = lean_array_fget(v_es_1006_, v_j_1009_);
v___x_1016_ = lean_box(0);
v_xs_x27_1017_ = lean_array_fset(v_es_1006_, v_j_1009_, v___x_1016_);
switch(lean_obj_tag(v_v_1015_))
{
case 0:
{
lean_object* v_key_1024_; lean_object* v_val_1025_; lean_object* v___x_1027_; uint8_t v_isShared_1028_; uint8_t v_isSharedCheck_1035_; 
v_key_1024_ = lean_ctor_get(v_v_1015_, 0);
v_val_1025_ = lean_ctor_get(v_v_1015_, 1);
v_isSharedCheck_1035_ = !lean_is_exclusive(v_v_1015_);
if (v_isSharedCheck_1035_ == 0)
{
v___x_1027_ = v_v_1015_;
v_isShared_1028_ = v_isSharedCheck_1035_;
goto v_resetjp_1026_;
}
else
{
lean_inc(v_val_1025_);
lean_inc(v_key_1024_);
lean_dec(v_v_1015_);
v___x_1027_ = lean_box(0);
v_isShared_1028_ = v_isSharedCheck_1035_;
goto v_resetjp_1026_;
}
v_resetjp_1026_:
{
uint8_t v___x_1029_; 
v___x_1029_ = l_Lean_instBEqMVarId_beq(v_x_1004_, v_key_1024_);
if (v___x_1029_ == 0)
{
lean_object* v___x_1030_; lean_object* v___x_1031_; 
lean_del_object(v___x_1027_);
v___x_1030_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_1024_, v_val_1025_, v_x_1004_, v_x_1005_);
v___x_1031_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1031_, 0, v___x_1030_);
v___y_1019_ = v___x_1031_;
goto v___jp_1018_;
}
else
{
lean_object* v___x_1033_; 
lean_dec(v_val_1025_);
lean_dec(v_key_1024_);
if (v_isShared_1028_ == 0)
{
lean_ctor_set(v___x_1027_, 1, v_x_1005_);
lean_ctor_set(v___x_1027_, 0, v_x_1004_);
v___x_1033_ = v___x_1027_;
goto v_reusejp_1032_;
}
else
{
lean_object* v_reuseFailAlloc_1034_; 
v_reuseFailAlloc_1034_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1034_, 0, v_x_1004_);
lean_ctor_set(v_reuseFailAlloc_1034_, 1, v_x_1005_);
v___x_1033_ = v_reuseFailAlloc_1034_;
goto v_reusejp_1032_;
}
v_reusejp_1032_:
{
v___y_1019_ = v___x_1033_;
goto v___jp_1018_;
}
}
}
}
case 1:
{
lean_object* v_node_1036_; lean_object* v___x_1038_; uint8_t v_isShared_1039_; uint8_t v_isSharedCheck_1048_; 
v_node_1036_ = lean_ctor_get(v_v_1015_, 0);
v_isSharedCheck_1048_ = !lean_is_exclusive(v_v_1015_);
if (v_isSharedCheck_1048_ == 0)
{
v___x_1038_ = v_v_1015_;
v_isShared_1039_ = v_isSharedCheck_1048_;
goto v_resetjp_1037_;
}
else
{
lean_inc(v_node_1036_);
lean_dec(v_v_1015_);
v___x_1038_ = lean_box(0);
v_isShared_1039_ = v_isSharedCheck_1048_;
goto v_resetjp_1037_;
}
v_resetjp_1037_:
{
size_t v___x_1040_; size_t v___x_1041_; size_t v___x_1042_; size_t v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1046_; 
v___x_1040_ = ((size_t)5ULL);
v___x_1041_ = lean_usize_shift_right(v_x_1002_, v___x_1040_);
v___x_1042_ = ((size_t)1ULL);
v___x_1043_ = lean_usize_add(v_x_1003_, v___x_1042_);
v___x_1044_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17___redArg(v_node_1036_, v___x_1041_, v___x_1043_, v_x_1004_, v_x_1005_);
if (v_isShared_1039_ == 0)
{
lean_ctor_set(v___x_1038_, 0, v___x_1044_);
v___x_1046_ = v___x_1038_;
goto v_reusejp_1045_;
}
else
{
lean_object* v_reuseFailAlloc_1047_; 
v_reuseFailAlloc_1047_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1047_, 0, v___x_1044_);
v___x_1046_ = v_reuseFailAlloc_1047_;
goto v_reusejp_1045_;
}
v_reusejp_1045_:
{
v___y_1019_ = v___x_1046_;
goto v___jp_1018_;
}
}
}
default: 
{
lean_object* v___x_1049_; 
v___x_1049_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1049_, 0, v_x_1004_);
lean_ctor_set(v___x_1049_, 1, v_x_1005_);
v___y_1019_ = v___x_1049_;
goto v___jp_1018_;
}
}
v___jp_1018_:
{
lean_object* v___x_1020_; lean_object* v___x_1022_; 
v___x_1020_ = lean_array_fset(v_xs_x27_1017_, v_j_1009_, v___y_1019_);
lean_dec(v_j_1009_);
if (v_isShared_1014_ == 0)
{
lean_ctor_set(v___x_1013_, 0, v___x_1020_);
v___x_1022_ = v___x_1013_;
goto v_reusejp_1021_;
}
else
{
lean_object* v_reuseFailAlloc_1023_; 
v_reuseFailAlloc_1023_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1023_, 0, v___x_1020_);
v___x_1022_ = v_reuseFailAlloc_1023_;
goto v_reusejp_1021_;
}
v_reusejp_1021_:
{
return v___x_1022_;
}
}
}
}
}
else
{
lean_object* v_ks_1052_; lean_object* v_vs_1053_; lean_object* v___x_1055_; uint8_t v_isShared_1056_; uint8_t v_isSharedCheck_1073_; 
v_ks_1052_ = lean_ctor_get(v_x_1001_, 0);
v_vs_1053_ = lean_ctor_get(v_x_1001_, 1);
v_isSharedCheck_1073_ = !lean_is_exclusive(v_x_1001_);
if (v_isSharedCheck_1073_ == 0)
{
v___x_1055_ = v_x_1001_;
v_isShared_1056_ = v_isSharedCheck_1073_;
goto v_resetjp_1054_;
}
else
{
lean_inc(v_vs_1053_);
lean_inc(v_ks_1052_);
lean_dec(v_x_1001_);
v___x_1055_ = lean_box(0);
v_isShared_1056_ = v_isSharedCheck_1073_;
goto v_resetjp_1054_;
}
v_resetjp_1054_:
{
lean_object* v___x_1058_; 
if (v_isShared_1056_ == 0)
{
v___x_1058_ = v___x_1055_;
goto v_reusejp_1057_;
}
else
{
lean_object* v_reuseFailAlloc_1072_; 
v_reuseFailAlloc_1072_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1072_, 0, v_ks_1052_);
lean_ctor_set(v_reuseFailAlloc_1072_, 1, v_vs_1053_);
v___x_1058_ = v_reuseFailAlloc_1072_;
goto v_reusejp_1057_;
}
v_reusejp_1057_:
{
lean_object* v_newNode_1059_; uint8_t v___y_1061_; size_t v___x_1067_; uint8_t v___x_1068_; 
v_newNode_1059_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17_spec__22___redArg(v___x_1058_, v_x_1004_, v_x_1005_);
v___x_1067_ = ((size_t)7ULL);
v___x_1068_ = lean_usize_dec_le(v___x_1067_, v_x_1003_);
if (v___x_1068_ == 0)
{
lean_object* v___x_1069_; lean_object* v___x_1070_; uint8_t v___x_1071_; 
v___x_1069_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_1059_);
v___x_1070_ = lean_unsigned_to_nat(4u);
v___x_1071_ = lean_nat_dec_lt(v___x_1069_, v___x_1070_);
lean_dec(v___x_1069_);
v___y_1061_ = v___x_1071_;
goto v___jp_1060_;
}
else
{
v___y_1061_ = v___x_1068_;
goto v___jp_1060_;
}
v___jp_1060_:
{
if (v___y_1061_ == 0)
{
lean_object* v_ks_1062_; lean_object* v_vs_1063_; lean_object* v___x_1064_; lean_object* v___x_1065_; lean_object* v___x_1066_; 
v_ks_1062_ = lean_ctor_get(v_newNode_1059_, 0);
lean_inc_ref(v_ks_1062_);
v_vs_1063_ = lean_ctor_get(v_newNode_1059_, 1);
lean_inc_ref(v_vs_1063_);
lean_dec_ref(v_newNode_1059_);
v___x_1064_ = lean_unsigned_to_nat(0u);
v___x_1065_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17___redArg___closed__0, &lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17___redArg___closed__0_once, _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17___redArg___closed__0);
v___x_1066_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17_spec__23___redArg(v_x_1003_, v_ks_1062_, v_vs_1063_, v___x_1064_, v___x_1065_);
lean_dec_ref(v_vs_1063_);
lean_dec_ref(v_ks_1062_);
return v___x_1066_;
}
else
{
return v_newNode_1059_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17_spec__23___redArg(size_t v_depth_1074_, lean_object* v_keys_1075_, lean_object* v_vals_1076_, lean_object* v_i_1077_, lean_object* v_entries_1078_){
_start:
{
lean_object* v___x_1079_; uint8_t v___x_1080_; 
v___x_1079_ = lean_array_get_size(v_keys_1075_);
v___x_1080_ = lean_nat_dec_lt(v_i_1077_, v___x_1079_);
if (v___x_1080_ == 0)
{
lean_dec(v_i_1077_);
return v_entries_1078_;
}
else
{
lean_object* v_k_1081_; lean_object* v_v_1082_; uint64_t v___x_1083_; size_t v_h_1084_; size_t v___x_1085_; lean_object* v___x_1086_; size_t v___x_1087_; size_t v___x_1088_; size_t v___x_1089_; size_t v_h_1090_; lean_object* v___x_1091_; lean_object* v___x_1092_; 
v_k_1081_ = lean_array_fget_borrowed(v_keys_1075_, v_i_1077_);
v_v_1082_ = lean_array_fget_borrowed(v_vals_1076_, v_i_1077_);
v___x_1083_ = l_Lean_instHashableMVarId_hash(v_k_1081_);
v_h_1084_ = lean_uint64_to_usize(v___x_1083_);
v___x_1085_ = ((size_t)5ULL);
v___x_1086_ = lean_unsigned_to_nat(1u);
v___x_1087_ = ((size_t)1ULL);
v___x_1088_ = lean_usize_sub(v_depth_1074_, v___x_1087_);
v___x_1089_ = lean_usize_mul(v___x_1085_, v___x_1088_);
v_h_1090_ = lean_usize_shift_right(v_h_1084_, v___x_1089_);
v___x_1091_ = lean_nat_add(v_i_1077_, v___x_1086_);
lean_dec(v_i_1077_);
lean_inc(v_v_1082_);
lean_inc(v_k_1081_);
v___x_1092_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17___redArg(v_entries_1078_, v_h_1090_, v_depth_1074_, v_k_1081_, v_v_1082_);
v_i_1077_ = v___x_1091_;
v_entries_1078_ = v___x_1092_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17_spec__23___redArg___boxed(lean_object* v_depth_1094_, lean_object* v_keys_1095_, lean_object* v_vals_1096_, lean_object* v_i_1097_, lean_object* v_entries_1098_){
_start:
{
size_t v_depth_boxed_1099_; lean_object* v_res_1100_; 
v_depth_boxed_1099_ = lean_unbox_usize(v_depth_1094_);
lean_dec(v_depth_1094_);
v_res_1100_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17_spec__23___redArg(v_depth_boxed_1099_, v_keys_1095_, v_vals_1096_, v_i_1097_, v_entries_1098_);
lean_dec_ref(v_vals_1096_);
lean_dec_ref(v_keys_1095_);
return v_res_1100_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17___redArg___boxed(lean_object* v_x_1101_, lean_object* v_x_1102_, lean_object* v_x_1103_, lean_object* v_x_1104_, lean_object* v_x_1105_){
_start:
{
size_t v_x_17807__boxed_1106_; size_t v_x_17808__boxed_1107_; lean_object* v_res_1108_; 
v_x_17807__boxed_1106_ = lean_unbox_usize(v_x_1102_);
lean_dec(v_x_1102_);
v_x_17808__boxed_1107_ = lean_unbox_usize(v_x_1103_);
lean_dec(v_x_1103_);
v_res_1108_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17___redArg(v_x_1101_, v_x_17807__boxed_1106_, v_x_17808__boxed_1107_, v_x_1104_, v_x_1105_);
return v_res_1108_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12___redArg(lean_object* v_x_1109_, lean_object* v_x_1110_, lean_object* v_x_1111_){
_start:
{
uint64_t v___x_1112_; size_t v___x_1113_; size_t v___x_1114_; lean_object* v___x_1115_; 
v___x_1112_ = l_Lean_instHashableMVarId_hash(v_x_1110_);
v___x_1113_ = lean_uint64_to_usize(v___x_1112_);
v___x_1114_ = ((size_t)1ULL);
v___x_1115_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17___redArg(v_x_1109_, v___x_1113_, v___x_1114_, v_x_1110_, v_x_1111_);
return v___x_1115_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15(lean_object* v_mvarId_1116_, lean_object* v___y_1117_, lean_object* v___y_1118_, lean_object* v___y_1119_, lean_object* v___y_1120_){
_start:
{
lean_object* v___x_1122_; lean_object* v_mctx_1123_; lean_object* v_mvarDecl_1124_; lean_object* v_userName_1125_; lean_object* v_lctx_1126_; lean_object* v_type_1127_; lean_object* v_depth_1128_; lean_object* v_localInstances_1129_; uint8_t v_kind_1130_; lean_object* v_numScopeArgs_1131_; lean_object* v_index_1132_; lean_object* v___x_1134_; uint8_t v_isShared_1135_; uint8_t v_isSharedCheck_1195_; 
v___x_1122_ = lean_st_ref_get(v___y_1118_);
v_mctx_1123_ = lean_ctor_get(v___x_1122_, 0);
lean_inc_ref(v_mctx_1123_);
lean_dec(v___x_1122_);
lean_inc(v_mvarId_1116_);
v_mvarDecl_1124_ = l_Lean_MetavarContext_getDecl(v_mctx_1123_, v_mvarId_1116_);
lean_dec_ref(v_mctx_1123_);
v_userName_1125_ = lean_ctor_get(v_mvarDecl_1124_, 0);
v_lctx_1126_ = lean_ctor_get(v_mvarDecl_1124_, 1);
v_type_1127_ = lean_ctor_get(v_mvarDecl_1124_, 2);
v_depth_1128_ = lean_ctor_get(v_mvarDecl_1124_, 3);
v_localInstances_1129_ = lean_ctor_get(v_mvarDecl_1124_, 4);
v_kind_1130_ = lean_ctor_get_uint8(v_mvarDecl_1124_, sizeof(void*)*7);
v_numScopeArgs_1131_ = lean_ctor_get(v_mvarDecl_1124_, 5);
v_index_1132_ = lean_ctor_get(v_mvarDecl_1124_, 6);
v_isSharedCheck_1195_ = !lean_is_exclusive(v_mvarDecl_1124_);
if (v_isSharedCheck_1195_ == 0)
{
v___x_1134_ = v_mvarDecl_1124_;
v_isShared_1135_ = v_isSharedCheck_1195_;
goto v_resetjp_1133_;
}
else
{
lean_inc(v_index_1132_);
lean_inc(v_numScopeArgs_1131_);
lean_inc(v_localInstances_1129_);
lean_inc(v_depth_1128_);
lean_inc(v_type_1127_);
lean_inc(v_lctx_1126_);
lean_inc(v_userName_1125_);
lean_dec(v_mvarDecl_1124_);
v___x_1134_ = lean_box(0);
v_isShared_1135_ = v_isSharedCheck_1195_;
goto v_resetjp_1133_;
}
v_resetjp_1133_:
{
lean_object* v___x_1136_; 
v___x_1136_ = lp_aesop_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19(v_lctx_1126_, v___y_1117_, v___y_1118_, v___y_1119_, v___y_1120_);
if (lean_obj_tag(v___x_1136_) == 0)
{
lean_object* v_a_1137_; lean_object* v___x_1138_; lean_object* v_a_1139_; lean_object* v___x_1141_; uint8_t v_isShared_1142_; uint8_t v_isSharedCheck_1186_; 
v_a_1137_ = lean_ctor_get(v___x_1136_, 0);
lean_inc(v_a_1137_);
lean_dec_ref_known(v___x_1136_, 1);
v___x_1138_ = lp_aesop_Lean_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__1___redArg(v_type_1127_, v___y_1118_);
v_a_1139_ = lean_ctor_get(v___x_1138_, 0);
v_isSharedCheck_1186_ = !lean_is_exclusive(v___x_1138_);
if (v_isSharedCheck_1186_ == 0)
{
v___x_1141_ = v___x_1138_;
v_isShared_1142_ = v_isSharedCheck_1186_;
goto v_resetjp_1140_;
}
else
{
lean_inc(v_a_1139_);
lean_dec(v___x_1138_);
v___x_1141_ = lean_box(0);
v_isShared_1142_ = v_isSharedCheck_1186_;
goto v_resetjp_1140_;
}
v_resetjp_1140_:
{
lean_object* v___x_1143_; lean_object* v___x_1144_; lean_object* v_fst_1145_; lean_object* v_snd_1146_; lean_object* v___x_1147_; lean_object* v_mctx_1148_; lean_object* v_cache_1149_; lean_object* v_zetaDeltaFVarIds_1150_; lean_object* v_postponed_1151_; lean_object* v_diag_1152_; lean_object* v___x_1154_; uint8_t v_isShared_1155_; uint8_t v_isSharedCheck_1185_; 
v___x_1143_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1143_, 0, v_a_1137_);
lean_ctor_set(v___x_1143_, 1, v_a_1139_);
v___x_1144_ = lean_sharecommon_quick(v___x_1143_);
lean_dec_ref_known(v___x_1143_, 2);
v_fst_1145_ = lean_ctor_get(v___x_1144_, 0);
lean_inc(v_fst_1145_);
v_snd_1146_ = lean_ctor_get(v___x_1144_, 1);
lean_inc(v_snd_1146_);
lean_dec(v___x_1144_);
v___x_1147_ = lean_st_ref_take(v___y_1118_);
v_mctx_1148_ = lean_ctor_get(v___x_1147_, 0);
v_cache_1149_ = lean_ctor_get(v___x_1147_, 1);
v_zetaDeltaFVarIds_1150_ = lean_ctor_get(v___x_1147_, 2);
v_postponed_1151_ = lean_ctor_get(v___x_1147_, 3);
v_diag_1152_ = lean_ctor_get(v___x_1147_, 4);
v_isSharedCheck_1185_ = !lean_is_exclusive(v___x_1147_);
if (v_isSharedCheck_1185_ == 0)
{
v___x_1154_ = v___x_1147_;
v_isShared_1155_ = v_isSharedCheck_1185_;
goto v_resetjp_1153_;
}
else
{
lean_inc(v_diag_1152_);
lean_inc(v_postponed_1151_);
lean_inc(v_zetaDeltaFVarIds_1150_);
lean_inc(v_cache_1149_);
lean_inc(v_mctx_1148_);
lean_dec(v___x_1147_);
v___x_1154_ = lean_box(0);
v_isShared_1155_ = v_isSharedCheck_1185_;
goto v_resetjp_1153_;
}
v_resetjp_1153_:
{
lean_object* v_depth_1156_; lean_object* v_levelAssignDepth_1157_; lean_object* v_lmvarCounter_1158_; lean_object* v_mvarCounter_1159_; lean_object* v_lDecls_1160_; lean_object* v_decls_1161_; lean_object* v_userNames_1162_; lean_object* v_lAssignment_1163_; lean_object* v_eAssignment_1164_; lean_object* v_dAssignment_1165_; lean_object* v___x_1167_; uint8_t v_isShared_1168_; uint8_t v_isSharedCheck_1184_; 
v_depth_1156_ = lean_ctor_get(v_mctx_1148_, 0);
v_levelAssignDepth_1157_ = lean_ctor_get(v_mctx_1148_, 1);
v_lmvarCounter_1158_ = lean_ctor_get(v_mctx_1148_, 2);
v_mvarCounter_1159_ = lean_ctor_get(v_mctx_1148_, 3);
v_lDecls_1160_ = lean_ctor_get(v_mctx_1148_, 4);
v_decls_1161_ = lean_ctor_get(v_mctx_1148_, 5);
v_userNames_1162_ = lean_ctor_get(v_mctx_1148_, 6);
v_lAssignment_1163_ = lean_ctor_get(v_mctx_1148_, 7);
v_eAssignment_1164_ = lean_ctor_get(v_mctx_1148_, 8);
v_dAssignment_1165_ = lean_ctor_get(v_mctx_1148_, 9);
v_isSharedCheck_1184_ = !lean_is_exclusive(v_mctx_1148_);
if (v_isSharedCheck_1184_ == 0)
{
v___x_1167_ = v_mctx_1148_;
v_isShared_1168_ = v_isSharedCheck_1184_;
goto v_resetjp_1166_;
}
else
{
lean_inc(v_dAssignment_1165_);
lean_inc(v_eAssignment_1164_);
lean_inc(v_lAssignment_1163_);
lean_inc(v_userNames_1162_);
lean_inc(v_decls_1161_);
lean_inc(v_lDecls_1160_);
lean_inc(v_mvarCounter_1159_);
lean_inc(v_lmvarCounter_1158_);
lean_inc(v_levelAssignDepth_1157_);
lean_inc(v_depth_1156_);
lean_dec(v_mctx_1148_);
v___x_1167_ = lean_box(0);
v_isShared_1168_ = v_isSharedCheck_1184_;
goto v_resetjp_1166_;
}
v_resetjp_1166_:
{
lean_object* v___x_1170_; 
if (v_isShared_1135_ == 0)
{
lean_ctor_set(v___x_1134_, 2, v_snd_1146_);
lean_ctor_set(v___x_1134_, 1, v_fst_1145_);
v___x_1170_ = v___x_1134_;
goto v_reusejp_1169_;
}
else
{
lean_object* v_reuseFailAlloc_1183_; 
v_reuseFailAlloc_1183_ = lean_alloc_ctor(0, 7, 1);
lean_ctor_set(v_reuseFailAlloc_1183_, 0, v_userName_1125_);
lean_ctor_set(v_reuseFailAlloc_1183_, 1, v_fst_1145_);
lean_ctor_set(v_reuseFailAlloc_1183_, 2, v_snd_1146_);
lean_ctor_set(v_reuseFailAlloc_1183_, 3, v_depth_1128_);
lean_ctor_set(v_reuseFailAlloc_1183_, 4, v_localInstances_1129_);
lean_ctor_set(v_reuseFailAlloc_1183_, 5, v_numScopeArgs_1131_);
lean_ctor_set(v_reuseFailAlloc_1183_, 6, v_index_1132_);
lean_ctor_set_uint8(v_reuseFailAlloc_1183_, sizeof(void*)*7, v_kind_1130_);
v___x_1170_ = v_reuseFailAlloc_1183_;
goto v_reusejp_1169_;
}
v_reusejp_1169_:
{
lean_object* v___x_1171_; lean_object* v___x_1173_; 
v___x_1171_ = lp_aesop_Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12___redArg(v_decls_1161_, v_mvarId_1116_, v___x_1170_);
if (v_isShared_1168_ == 0)
{
lean_ctor_set(v___x_1167_, 5, v___x_1171_);
v___x_1173_ = v___x_1167_;
goto v_reusejp_1172_;
}
else
{
lean_object* v_reuseFailAlloc_1182_; 
v_reuseFailAlloc_1182_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1182_, 0, v_depth_1156_);
lean_ctor_set(v_reuseFailAlloc_1182_, 1, v_levelAssignDepth_1157_);
lean_ctor_set(v_reuseFailAlloc_1182_, 2, v_lmvarCounter_1158_);
lean_ctor_set(v_reuseFailAlloc_1182_, 3, v_mvarCounter_1159_);
lean_ctor_set(v_reuseFailAlloc_1182_, 4, v_lDecls_1160_);
lean_ctor_set(v_reuseFailAlloc_1182_, 5, v___x_1171_);
lean_ctor_set(v_reuseFailAlloc_1182_, 6, v_userNames_1162_);
lean_ctor_set(v_reuseFailAlloc_1182_, 7, v_lAssignment_1163_);
lean_ctor_set(v_reuseFailAlloc_1182_, 8, v_eAssignment_1164_);
lean_ctor_set(v_reuseFailAlloc_1182_, 9, v_dAssignment_1165_);
v___x_1173_ = v_reuseFailAlloc_1182_;
goto v_reusejp_1172_;
}
v_reusejp_1172_:
{
lean_object* v___x_1175_; 
if (v_isShared_1155_ == 0)
{
lean_ctor_set(v___x_1154_, 0, v___x_1173_);
v___x_1175_ = v___x_1154_;
goto v_reusejp_1174_;
}
else
{
lean_object* v_reuseFailAlloc_1181_; 
v_reuseFailAlloc_1181_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1181_, 0, v___x_1173_);
lean_ctor_set(v_reuseFailAlloc_1181_, 1, v_cache_1149_);
lean_ctor_set(v_reuseFailAlloc_1181_, 2, v_zetaDeltaFVarIds_1150_);
lean_ctor_set(v_reuseFailAlloc_1181_, 3, v_postponed_1151_);
lean_ctor_set(v_reuseFailAlloc_1181_, 4, v_diag_1152_);
v___x_1175_ = v_reuseFailAlloc_1181_;
goto v_reusejp_1174_;
}
v_reusejp_1174_:
{
lean_object* v___x_1176_; lean_object* v___x_1177_; lean_object* v___x_1179_; 
v___x_1176_ = lean_st_ref_set(v___y_1118_, v___x_1175_);
v___x_1177_ = lean_box(0);
if (v_isShared_1142_ == 0)
{
lean_ctor_set(v___x_1141_, 0, v___x_1177_);
v___x_1179_ = v___x_1141_;
goto v_reusejp_1178_;
}
else
{
lean_object* v_reuseFailAlloc_1180_; 
v_reuseFailAlloc_1180_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1180_, 0, v___x_1177_);
v___x_1179_ = v_reuseFailAlloc_1180_;
goto v_reusejp_1178_;
}
v_reusejp_1178_:
{
return v___x_1179_;
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
lean_object* v_a_1187_; lean_object* v___x_1189_; uint8_t v_isShared_1190_; uint8_t v_isSharedCheck_1194_; 
lean_del_object(v___x_1134_);
lean_dec(v_index_1132_);
lean_dec(v_numScopeArgs_1131_);
lean_dec_ref(v_localInstances_1129_);
lean_dec(v_depth_1128_);
lean_dec_ref(v_type_1127_);
lean_dec(v_userName_1125_);
lean_dec(v_mvarId_1116_);
v_a_1187_ = lean_ctor_get(v___x_1136_, 0);
v_isSharedCheck_1194_ = !lean_is_exclusive(v___x_1136_);
if (v_isSharedCheck_1194_ == 0)
{
v___x_1189_ = v___x_1136_;
v_isShared_1190_ = v_isSharedCheck_1194_;
goto v_resetjp_1188_;
}
else
{
lean_inc(v_a_1187_);
lean_dec(v___x_1136_);
v___x_1189_ = lean_box(0);
v_isShared_1190_ = v_isSharedCheck_1194_;
goto v_resetjp_1188_;
}
v_resetjp_1188_:
{
lean_object* v___x_1192_; 
if (v_isShared_1190_ == 0)
{
v___x_1192_ = v___x_1189_;
goto v_reusejp_1191_;
}
else
{
lean_object* v_reuseFailAlloc_1193_; 
v_reuseFailAlloc_1193_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1193_, 0, v_a_1187_);
v___x_1192_ = v_reuseFailAlloc_1193_;
goto v_reusejp_1191_;
}
v_reusejp_1191_:
{
return v___x_1192_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15___boxed(lean_object* v_mvarId_1196_, lean_object* v___y_1197_, lean_object* v___y_1198_, lean_object* v___y_1199_, lean_object* v___y_1200_, lean_object* v___y_1201_){
_start:
{
lean_object* v_res_1202_; 
v_res_1202_ = lp_aesop_Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15(v_mvarId_1196_, v___y_1197_, v___y_1198_, v___y_1199_, v___y_1200_);
lean_dec(v___y_1200_);
lean_dec_ref(v___y_1199_);
lean_dec(v___y_1198_);
lean_dec_ref(v___y_1197_);
return v_res_1202_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__17___redArg(lean_object* v_msg_1203_, lean_object* v___y_1204_, lean_object* v___y_1205_, lean_object* v___y_1206_, lean_object* v___y_1207_){
_start:
{
lean_object* v_ref_1209_; lean_object* v___x_1210_; lean_object* v_a_1211_; lean_object* v___x_1213_; uint8_t v_isShared_1214_; uint8_t v_isSharedCheck_1219_; 
v_ref_1209_ = lean_ctor_get(v___y_1206_, 5);
v___x_1210_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6_spec__7(v_msg_1203_, v___y_1204_, v___y_1205_, v___y_1206_, v___y_1207_);
v_a_1211_ = lean_ctor_get(v___x_1210_, 0);
v_isSharedCheck_1219_ = !lean_is_exclusive(v___x_1210_);
if (v_isSharedCheck_1219_ == 0)
{
v___x_1213_ = v___x_1210_;
v_isShared_1214_ = v_isSharedCheck_1219_;
goto v_resetjp_1212_;
}
else
{
lean_inc(v_a_1211_);
lean_dec(v___x_1210_);
v___x_1213_ = lean_box(0);
v_isShared_1214_ = v_isSharedCheck_1219_;
goto v_resetjp_1212_;
}
v_resetjp_1212_:
{
lean_object* v___x_1215_; lean_object* v___x_1217_; 
lean_inc(v_ref_1209_);
v___x_1215_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1215_, 0, v_ref_1209_);
lean_ctor_set(v___x_1215_, 1, v_a_1211_);
if (v_isShared_1214_ == 0)
{
lean_ctor_set_tag(v___x_1213_, 1);
lean_ctor_set(v___x_1213_, 0, v___x_1215_);
v___x_1217_ = v___x_1213_;
goto v_reusejp_1216_;
}
else
{
lean_object* v_reuseFailAlloc_1218_; 
v_reuseFailAlloc_1218_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1218_, 0, v___x_1215_);
v___x_1217_ = v_reuseFailAlloc_1218_;
goto v_reusejp_1216_;
}
v_reusejp_1216_:
{
return v___x_1217_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__17___redArg___boxed(lean_object* v_msg_1220_, lean_object* v___y_1221_, lean_object* v___y_1222_, lean_object* v___y_1223_, lean_object* v___y_1224_, lean_object* v___y_1225_){
_start:
{
lean_object* v_res_1226_; 
v_res_1226_ = lp_aesop_Lean_throwError___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__17___redArg(v_msg_1220_, v___y_1221_, v___y_1222_, v___y_1223_, v___y_1224_);
lean_dec(v___y_1224_);
lean_dec_ref(v___y_1223_);
lean_dec(v___y_1222_);
lean_dec_ref(v___y_1221_);
return v_res_1226_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16_spec__21_spec__24___redArg(lean_object* v_keys_1227_, lean_object* v_vals_1228_, lean_object* v_i_1229_, lean_object* v_k_1230_){
_start:
{
lean_object* v___x_1231_; uint8_t v___x_1232_; 
v___x_1231_ = lean_array_get_size(v_keys_1227_);
v___x_1232_ = lean_nat_dec_lt(v_i_1229_, v___x_1231_);
if (v___x_1232_ == 0)
{
lean_object* v___x_1233_; 
lean_dec(v_i_1229_);
v___x_1233_ = lean_box(0);
return v___x_1233_;
}
else
{
lean_object* v_k_x27_1234_; uint8_t v___x_1235_; 
v_k_x27_1234_ = lean_array_fget_borrowed(v_keys_1227_, v_i_1229_);
v___x_1235_ = l_Lean_instBEqMVarId_beq(v_k_1230_, v_k_x27_1234_);
if (v___x_1235_ == 0)
{
lean_object* v___x_1236_; lean_object* v___x_1237_; 
v___x_1236_ = lean_unsigned_to_nat(1u);
v___x_1237_ = lean_nat_add(v_i_1229_, v___x_1236_);
lean_dec(v_i_1229_);
v_i_1229_ = v___x_1237_;
goto _start;
}
else
{
lean_object* v___x_1239_; lean_object* v___x_1240_; 
v___x_1239_ = lean_array_fget_borrowed(v_vals_1228_, v_i_1229_);
lean_dec(v_i_1229_);
lean_inc(v___x_1239_);
v___x_1240_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1240_, 0, v___x_1239_);
return v___x_1240_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16_spec__21_spec__24___redArg___boxed(lean_object* v_keys_1241_, lean_object* v_vals_1242_, lean_object* v_i_1243_, lean_object* v_k_1244_){
_start:
{
lean_object* v_res_1245_; 
v_res_1245_ = lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16_spec__21_spec__24___redArg(v_keys_1241_, v_vals_1242_, v_i_1243_, v_k_1244_);
lean_dec(v_k_1244_);
lean_dec_ref(v_vals_1242_);
lean_dec_ref(v_keys_1241_);
return v_res_1245_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16_spec__21___redArg(lean_object* v_x_1246_, size_t v_x_1247_, lean_object* v_x_1248_){
_start:
{
if (lean_obj_tag(v_x_1246_) == 0)
{
lean_object* v_es_1249_; lean_object* v___x_1250_; size_t v___x_1251_; size_t v___x_1252_; lean_object* v_j_1253_; lean_object* v___x_1254_; 
v_es_1249_ = lean_ctor_get(v_x_1246_, 0);
v___x_1250_ = lean_box(2);
v___x_1251_ = ((size_t)31ULL);
v___x_1252_ = lean_usize_land(v_x_1247_, v___x_1251_);
v_j_1253_ = lean_usize_to_nat(v___x_1252_);
v___x_1254_ = lean_array_get_borrowed(v___x_1250_, v_es_1249_, v_j_1253_);
lean_dec(v_j_1253_);
switch(lean_obj_tag(v___x_1254_))
{
case 0:
{
lean_object* v_key_1255_; lean_object* v_val_1256_; uint8_t v___x_1257_; 
v_key_1255_ = lean_ctor_get(v___x_1254_, 0);
v_val_1256_ = lean_ctor_get(v___x_1254_, 1);
v___x_1257_ = l_Lean_instBEqMVarId_beq(v_x_1248_, v_key_1255_);
if (v___x_1257_ == 0)
{
lean_object* v___x_1258_; 
v___x_1258_ = lean_box(0);
return v___x_1258_;
}
else
{
lean_object* v___x_1259_; 
lean_inc(v_val_1256_);
v___x_1259_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1259_, 0, v_val_1256_);
return v___x_1259_;
}
}
case 1:
{
lean_object* v_node_1260_; size_t v___x_1261_; size_t v___x_1262_; 
v_node_1260_ = lean_ctor_get(v___x_1254_, 0);
v___x_1261_ = ((size_t)5ULL);
v___x_1262_ = lean_usize_shift_right(v_x_1247_, v___x_1261_);
v_x_1246_ = v_node_1260_;
v_x_1247_ = v___x_1262_;
goto _start;
}
default: 
{
lean_object* v___x_1264_; 
v___x_1264_ = lean_box(0);
return v___x_1264_;
}
}
}
else
{
lean_object* v_ks_1265_; lean_object* v_vs_1266_; lean_object* v___x_1267_; lean_object* v___x_1268_; 
v_ks_1265_ = lean_ctor_get(v_x_1246_, 0);
v_vs_1266_ = lean_ctor_get(v_x_1246_, 1);
v___x_1267_ = lean_unsigned_to_nat(0u);
v___x_1268_ = lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16_spec__21_spec__24___redArg(v_ks_1265_, v_vs_1266_, v___x_1267_, v_x_1248_);
return v___x_1268_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16_spec__21___redArg___boxed(lean_object* v_x_1269_, lean_object* v_x_1270_, lean_object* v_x_1271_){
_start:
{
size_t v_x_18145__boxed_1272_; lean_object* v_res_1273_; 
v_x_18145__boxed_1272_ = lean_unbox_usize(v_x_1270_);
lean_dec(v_x_1270_);
v_res_1273_ = lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16_spec__21___redArg(v_x_1269_, v_x_18145__boxed_1272_, v_x_1271_);
lean_dec(v_x_1271_);
lean_dec_ref(v_x_1269_);
return v_res_1273_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16___redArg(lean_object* v_x_1274_, lean_object* v_x_1275_){
_start:
{
uint64_t v___x_1276_; size_t v___x_1277_; lean_object* v___x_1278_; 
v___x_1276_ = l_Lean_instHashableMVarId_hash(v_x_1275_);
v___x_1277_ = lean_uint64_to_usize(v___x_1276_);
v___x_1278_ = lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16_spec__21___redArg(v_x_1274_, v___x_1277_, v_x_1275_);
return v___x_1278_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16___redArg___boxed(lean_object* v_x_1279_, lean_object* v_x_1280_){
_start:
{
lean_object* v_res_1281_; 
v_res_1281_ = lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16___redArg(v_x_1279_, v_x_1280_);
lean_dec(v_x_1280_);
lean_dec_ref(v_x_1279_);
return v_res_1281_;
}
}
static lean_object* _init_lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14___closed__1(void){
_start:
{
lean_object* v___x_1283_; lean_object* v___x_1284_; 
v___x_1283_ = ((lean_object*)(lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14___closed__0));
v___x_1284_ = l_Lean_stringToMessageData(v___x_1283_);
return v___x_1284_;
}
}
static lean_object* _init_lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14___closed__3(void){
_start:
{
lean_object* v___x_1286_; lean_object* v___x_1287_; 
v___x_1286_ = ((lean_object*)(lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14___closed__2));
v___x_1287_ = l_Lean_stringToMessageData(v___x_1286_);
return v___x_1287_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14(lean_object* v_mctx_1288_, lean_object* v_mvarId_1289_, lean_object* v___y_1290_, lean_object* v___y_1291_, lean_object* v___y_1292_, lean_object* v___y_1293_){
_start:
{
lean_object* v_decls_1295_; lean_object* v___x_1296_; 
v_decls_1295_ = lean_ctor_get(v_mctx_1288_, 5);
v___x_1296_ = lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16___redArg(v_decls_1295_, v_mvarId_1289_);
if (lean_obj_tag(v___x_1296_) == 1)
{
lean_object* v_val_1297_; lean_object* v___x_1299_; uint8_t v_isShared_1300_; uint8_t v_isSharedCheck_1304_; 
lean_dec(v_mvarId_1289_);
v_val_1297_ = lean_ctor_get(v___x_1296_, 0);
v_isSharedCheck_1304_ = !lean_is_exclusive(v___x_1296_);
if (v_isSharedCheck_1304_ == 0)
{
v___x_1299_ = v___x_1296_;
v_isShared_1300_ = v_isSharedCheck_1304_;
goto v_resetjp_1298_;
}
else
{
lean_inc(v_val_1297_);
lean_dec(v___x_1296_);
v___x_1299_ = lean_box(0);
v_isShared_1300_ = v_isSharedCheck_1304_;
goto v_resetjp_1298_;
}
v_resetjp_1298_:
{
lean_object* v___x_1302_; 
if (v_isShared_1300_ == 0)
{
lean_ctor_set_tag(v___x_1299_, 0);
v___x_1302_ = v___x_1299_;
goto v_reusejp_1301_;
}
else
{
lean_object* v_reuseFailAlloc_1303_; 
v_reuseFailAlloc_1303_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1303_, 0, v_val_1297_);
v___x_1302_ = v_reuseFailAlloc_1303_;
goto v_reusejp_1301_;
}
v_reusejp_1301_:
{
return v___x_1302_;
}
}
}
else
{
lean_object* v___x_1305_; lean_object* v___x_1306_; lean_object* v___x_1307_; lean_object* v___x_1308_; lean_object* v___x_1309_; lean_object* v___x_1310_; 
lean_dec(v___x_1296_);
v___x_1305_ = lean_obj_once(&lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14___closed__1, &lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14___closed__1_once, _init_lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14___closed__1);
v___x_1306_ = l_Lean_MessageData_ofName(v_mvarId_1289_);
v___x_1307_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1307_, 0, v___x_1305_);
lean_ctor_set(v___x_1307_, 1, v___x_1306_);
v___x_1308_ = lean_obj_once(&lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14___closed__3, &lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14___closed__3_once, _init_lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14___closed__3);
v___x_1309_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1309_, 0, v___x_1307_);
lean_ctor_set(v___x_1309_, 1, v___x_1308_);
v___x_1310_ = lp_aesop_Lean_throwError___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__17___redArg(v___x_1309_, v___y_1290_, v___y_1291_, v___y_1292_, v___y_1293_);
return v___x_1310_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14___boxed(lean_object* v_mctx_1311_, lean_object* v_mvarId_1312_, lean_object* v___y_1313_, lean_object* v___y_1314_, lean_object* v___y_1315_, lean_object* v___y_1316_, lean_object* v___y_1317_){
_start:
{
lean_object* v_res_1318_; 
v_res_1318_ = lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14(v_mctx_1311_, v_mvarId_1312_, v___y_1313_, v___y_1314_, v___y_1315_, v___y_1316_);
lean_dec(v___y_1316_);
lean_dec_ref(v___y_1315_);
lean_dec(v___y_1314_);
lean_dec_ref(v___y_1313_);
lean_dec_ref(v_mctx_1311_);
return v_res_1318_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11(lean_object* v_mvarId_1319_, lean_object* v___y_1320_, lean_object* v___y_1321_, lean_object* v___y_1322_, lean_object* v___y_1323_){
_start:
{
lean_object* v___x_1325_; lean_object* v_mctx_1326_; lean_object* v___x_1327_; 
v___x_1325_ = lean_st_ref_get(v___y_1321_);
v_mctx_1326_ = lean_ctor_get(v___x_1325_, 0);
lean_inc_ref(v_mctx_1326_);
lean_dec(v___x_1325_);
lean_inc(v_mvarId_1319_);
v___x_1327_ = lp_aesop_Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14(v_mctx_1326_, v_mvarId_1319_, v___y_1320_, v___y_1321_, v___y_1322_, v___y_1323_);
lean_dec_ref(v_mctx_1326_);
if (lean_obj_tag(v___x_1327_) == 0)
{
lean_object* v___x_1328_; 
lean_dec_ref_known(v___x_1327_, 1);
v___x_1328_ = lp_aesop_Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15(v_mvarId_1319_, v___y_1320_, v___y_1321_, v___y_1322_, v___y_1323_);
return v___x_1328_;
}
else
{
lean_object* v_a_1329_; lean_object* v___x_1331_; uint8_t v_isShared_1332_; uint8_t v_isSharedCheck_1336_; 
lean_dec(v_mvarId_1319_);
v_a_1329_ = lean_ctor_get(v___x_1327_, 0);
v_isSharedCheck_1336_ = !lean_is_exclusive(v___x_1327_);
if (v_isSharedCheck_1336_ == 0)
{
v___x_1331_ = v___x_1327_;
v_isShared_1332_ = v_isSharedCheck_1336_;
goto v_resetjp_1330_;
}
else
{
lean_inc(v_a_1329_);
lean_dec(v___x_1327_);
v___x_1331_ = lean_box(0);
v_isShared_1332_ = v_isSharedCheck_1336_;
goto v_resetjp_1330_;
}
v_resetjp_1330_:
{
lean_object* v___x_1334_; 
if (v_isShared_1332_ == 0)
{
v___x_1334_ = v___x_1331_;
goto v_reusejp_1333_;
}
else
{
lean_object* v_reuseFailAlloc_1335_; 
v_reuseFailAlloc_1335_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1335_, 0, v_a_1329_);
v___x_1334_ = v_reuseFailAlloc_1335_;
goto v_reusejp_1333_;
}
v_reusejp_1333_:
{
return v___x_1334_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11___boxed(lean_object* v_mvarId_1337_, lean_object* v___y_1338_, lean_object* v___y_1339_, lean_object* v___y_1340_, lean_object* v___y_1341_, lean_object* v___y_1342_){
_start:
{
lean_object* v_res_1343_; 
v_res_1343_ = lp_aesop_Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11(v_mvarId_1337_, v___y_1338_, v___y_1339_, v___y_1340_, v___y_1341_);
lean_dec(v___y_1341_);
lean_dec_ref(v___y_1340_);
lean_dec(v___y_1339_);
lean_dec_ref(v___y_1338_);
return v_res_1343_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1___closed__1(void){
_start:
{
lean_object* v___x_1345_; lean_object* v___x_1346_; 
v___x_1345_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1___closed__0));
v___x_1346_ = l_Lean_stringToMessageData(v___x_1345_);
return v___x_1346_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1___closed__3(void){
_start:
{
lean_object* v___x_1348_; lean_object* v___x_1349_; 
v___x_1348_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1___closed__2));
v___x_1349_ = l_Lean_stringToMessageData(v___x_1348_);
return v___x_1349_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1(lean_object* v_mvarId_1350_, uint8_t v___x_1351_, lean_object* v___y_1352_, lean_object* v___y_1353_, lean_object* v___y_1354_, lean_object* v___y_1355_){
_start:
{
lean_object* v___x_1357_; 
lean_inc(v_mvarId_1350_);
v___x_1357_ = lp_aesop_Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11(v_mvarId_1350_, v___y_1352_, v___y_1353_, v___y_1354_, v___y_1355_);
if (lean_obj_tag(v___x_1357_) == 0)
{
lean_object* v___x_1358_; 
lean_dec_ref_known(v___x_1357_, 1);
lean_inc(v_mvarId_1350_);
v___x_1358_ = l_Lean_MVarId_getDecl(v_mvarId_1350_, v___y_1352_, v___y_1353_, v___y_1354_, v___y_1355_);
if (lean_obj_tag(v___x_1358_) == 0)
{
lean_object* v_a_1359_; lean_object* v___x_1360_; 
v_a_1359_ = lean_ctor_get(v___x_1358_, 0);
lean_inc(v_a_1359_);
lean_dec_ref_known(v___x_1358_, 1);
lean_inc(v_mvarId_1350_);
v___x_1360_ = l_Lean_MVarId_getMVarDependencies(v_mvarId_1350_, v___x_1351_, v___y_1352_, v___y_1353_, v___y_1354_, v___y_1355_);
if (lean_obj_tag(v___x_1360_) == 0)
{
lean_object* v_a_1361_; lean_object* v___x_1363_; uint8_t v_isShared_1364_; uint8_t v_isSharedCheck_1400_; 
v_a_1361_ = lean_ctor_get(v___x_1360_, 0);
v_isSharedCheck_1400_ = !lean_is_exclusive(v___x_1360_);
if (v_isSharedCheck_1400_ == 0)
{
v___x_1363_ = v___x_1360_;
v_isShared_1364_ = v_isSharedCheck_1400_;
goto v_resetjp_1362_;
}
else
{
lean_inc(v_a_1361_);
lean_dec(v___x_1360_);
v___x_1363_ = lean_box(0);
v_isShared_1364_ = v_isSharedCheck_1400_;
goto v_resetjp_1362_;
}
v_resetjp_1362_:
{
lean_object* v___x_1370_; lean_object* v___x_1371_; 
v___x_1370_ = lp_aesop_Aesop_TraceOption_extraction;
v___x_1371_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__4___redArg(v___x_1370_, v___y_1354_);
if (lean_obj_tag(v___x_1371_) == 0)
{
lean_object* v_a_1372_; uint8_t v___x_1373_; 
v_a_1372_ = lean_ctor_get(v___x_1371_, 0);
lean_inc(v_a_1372_);
lean_dec_ref_known(v___x_1371_, 1);
v___x_1373_ = lean_unbox(v_a_1372_);
lean_dec(v_a_1372_);
if (v___x_1373_ == 0)
{
lean_dec(v_mvarId_1350_);
goto v___jp_1365_;
}
else
{
lean_object* v_traceClass_1374_; lean_object* v___x_1375_; lean_object* v___x_1376_; lean_object* v___x_1377_; lean_object* v___x_1378_; lean_object* v___x_1379_; lean_object* v___x_1380_; lean_object* v___x_1381_; lean_object* v___x_1382_; lean_object* v___x_1383_; 
v_traceClass_1374_ = lean_ctor_get(v___x_1370_, 0);
v___x_1375_ = lean_obj_once(&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1___closed__1, &lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1___closed__1_once, _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1___closed__1);
lean_inc(v_mvarId_1350_);
v___x_1376_ = l_Lean_MessageData_ofName(v_mvarId_1350_);
v___x_1377_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1377_, 0, v___x_1375_);
lean_ctor_set(v___x_1377_, 1, v___x_1376_);
v___x_1378_ = lean_obj_once(&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1___closed__3, &lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1___closed__3_once, _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1___closed__3);
v___x_1379_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1379_, 0, v___x_1377_);
lean_ctor_set(v___x_1379_, 1, v___x_1378_);
v___x_1380_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1380_, 0, v_mvarId_1350_);
v___x_1381_ = l_Lean_indentD(v___x_1380_);
v___x_1382_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1382_, 0, v___x_1379_);
lean_ctor_set(v___x_1382_, 1, v___x_1381_);
lean_inc(v_traceClass_1374_);
v___x_1383_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6(v_traceClass_1374_, v___x_1382_, v___y_1352_, v___y_1353_, v___y_1354_, v___y_1355_);
if (lean_obj_tag(v___x_1383_) == 0)
{
lean_dec_ref_known(v___x_1383_, 1);
goto v___jp_1365_;
}
else
{
lean_object* v_a_1384_; lean_object* v___x_1386_; uint8_t v_isShared_1387_; uint8_t v_isSharedCheck_1391_; 
lean_del_object(v___x_1363_);
lean_dec(v_a_1361_);
lean_dec(v_a_1359_);
v_a_1384_ = lean_ctor_get(v___x_1383_, 0);
v_isSharedCheck_1391_ = !lean_is_exclusive(v___x_1383_);
if (v_isSharedCheck_1391_ == 0)
{
v___x_1386_ = v___x_1383_;
v_isShared_1387_ = v_isSharedCheck_1391_;
goto v_resetjp_1385_;
}
else
{
lean_inc(v_a_1384_);
lean_dec(v___x_1383_);
v___x_1386_ = lean_box(0);
v_isShared_1387_ = v_isSharedCheck_1391_;
goto v_resetjp_1385_;
}
v_resetjp_1385_:
{
lean_object* v___x_1389_; 
if (v_isShared_1387_ == 0)
{
v___x_1389_ = v___x_1386_;
goto v_reusejp_1388_;
}
else
{
lean_object* v_reuseFailAlloc_1390_; 
v_reuseFailAlloc_1390_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1390_, 0, v_a_1384_);
v___x_1389_ = v_reuseFailAlloc_1390_;
goto v_reusejp_1388_;
}
v_reusejp_1388_:
{
return v___x_1389_;
}
}
}
}
}
else
{
lean_object* v_a_1392_; lean_object* v___x_1394_; uint8_t v_isShared_1395_; uint8_t v_isSharedCheck_1399_; 
lean_del_object(v___x_1363_);
lean_dec(v_a_1361_);
lean_dec(v_a_1359_);
lean_dec(v_mvarId_1350_);
v_a_1392_ = lean_ctor_get(v___x_1371_, 0);
v_isSharedCheck_1399_ = !lean_is_exclusive(v___x_1371_);
if (v_isSharedCheck_1399_ == 0)
{
v___x_1394_ = v___x_1371_;
v_isShared_1395_ = v_isSharedCheck_1399_;
goto v_resetjp_1393_;
}
else
{
lean_inc(v_a_1392_);
lean_dec(v___x_1371_);
v___x_1394_ = lean_box(0);
v_isShared_1395_ = v_isSharedCheck_1399_;
goto v_resetjp_1393_;
}
v_resetjp_1393_:
{
lean_object* v___x_1397_; 
if (v_isShared_1395_ == 0)
{
v___x_1397_ = v___x_1394_;
goto v_reusejp_1396_;
}
else
{
lean_object* v_reuseFailAlloc_1398_; 
v_reuseFailAlloc_1398_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1398_, 0, v_a_1392_);
v___x_1397_ = v_reuseFailAlloc_1398_;
goto v_reusejp_1396_;
}
v_reusejp_1396_:
{
return v___x_1397_;
}
}
}
v___jp_1365_:
{
lean_object* v___x_1366_; lean_object* v___x_1368_; 
v___x_1366_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1366_, 0, v_a_1359_);
lean_ctor_set(v___x_1366_, 1, v_a_1361_);
if (v_isShared_1364_ == 0)
{
lean_ctor_set(v___x_1363_, 0, v___x_1366_);
v___x_1368_ = v___x_1363_;
goto v_reusejp_1367_;
}
else
{
lean_object* v_reuseFailAlloc_1369_; 
v_reuseFailAlloc_1369_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1369_, 0, v___x_1366_);
v___x_1368_ = v_reuseFailAlloc_1369_;
goto v_reusejp_1367_;
}
v_reusejp_1367_:
{
return v___x_1368_;
}
}
}
}
else
{
lean_object* v_a_1401_; lean_object* v___x_1403_; uint8_t v_isShared_1404_; uint8_t v_isSharedCheck_1408_; 
lean_dec(v_a_1359_);
lean_dec(v_mvarId_1350_);
v_a_1401_ = lean_ctor_get(v___x_1360_, 0);
v_isSharedCheck_1408_ = !lean_is_exclusive(v___x_1360_);
if (v_isSharedCheck_1408_ == 0)
{
v___x_1403_ = v___x_1360_;
v_isShared_1404_ = v_isSharedCheck_1408_;
goto v_resetjp_1402_;
}
else
{
lean_inc(v_a_1401_);
lean_dec(v___x_1360_);
v___x_1403_ = lean_box(0);
v_isShared_1404_ = v_isSharedCheck_1408_;
goto v_resetjp_1402_;
}
v_resetjp_1402_:
{
lean_object* v___x_1406_; 
if (v_isShared_1404_ == 0)
{
v___x_1406_ = v___x_1403_;
goto v_reusejp_1405_;
}
else
{
lean_object* v_reuseFailAlloc_1407_; 
v_reuseFailAlloc_1407_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1407_, 0, v_a_1401_);
v___x_1406_ = v_reuseFailAlloc_1407_;
goto v_reusejp_1405_;
}
v_reusejp_1405_:
{
return v___x_1406_;
}
}
}
}
else
{
lean_object* v_a_1409_; lean_object* v___x_1411_; uint8_t v_isShared_1412_; uint8_t v_isSharedCheck_1416_; 
lean_dec(v_mvarId_1350_);
v_a_1409_ = lean_ctor_get(v___x_1358_, 0);
v_isSharedCheck_1416_ = !lean_is_exclusive(v___x_1358_);
if (v_isSharedCheck_1416_ == 0)
{
v___x_1411_ = v___x_1358_;
v_isShared_1412_ = v_isSharedCheck_1416_;
goto v_resetjp_1410_;
}
else
{
lean_inc(v_a_1409_);
lean_dec(v___x_1358_);
v___x_1411_ = lean_box(0);
v_isShared_1412_ = v_isSharedCheck_1416_;
goto v_resetjp_1410_;
}
v_resetjp_1410_:
{
lean_object* v___x_1414_; 
if (v_isShared_1412_ == 0)
{
v___x_1414_ = v___x_1411_;
goto v_reusejp_1413_;
}
else
{
lean_object* v_reuseFailAlloc_1415_; 
v_reuseFailAlloc_1415_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1415_, 0, v_a_1409_);
v___x_1414_ = v_reuseFailAlloc_1415_;
goto v_reusejp_1413_;
}
v_reusejp_1413_:
{
return v___x_1414_;
}
}
}
}
else
{
lean_object* v_a_1417_; lean_object* v___x_1419_; uint8_t v_isShared_1420_; uint8_t v_isSharedCheck_1424_; 
lean_dec(v_mvarId_1350_);
v_a_1417_ = lean_ctor_get(v___x_1357_, 0);
v_isSharedCheck_1424_ = !lean_is_exclusive(v___x_1357_);
if (v_isSharedCheck_1424_ == 0)
{
v___x_1419_ = v___x_1357_;
v_isShared_1420_ = v_isSharedCheck_1424_;
goto v_resetjp_1418_;
}
else
{
lean_inc(v_a_1417_);
lean_dec(v___x_1357_);
v___x_1419_ = lean_box(0);
v_isShared_1420_ = v_isSharedCheck_1424_;
goto v_resetjp_1418_;
}
v_resetjp_1418_:
{
lean_object* v___x_1422_; 
if (v_isShared_1420_ == 0)
{
v___x_1422_ = v___x_1419_;
goto v_reusejp_1421_;
}
else
{
lean_object* v_reuseFailAlloc_1423_; 
v_reuseFailAlloc_1423_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1423_, 0, v_a_1417_);
v___x_1422_ = v_reuseFailAlloc_1423_;
goto v_reusejp_1421_;
}
v_reusejp_1421_:
{
return v___x_1422_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1___boxed(lean_object* v_mvarId_1425_, lean_object* v___x_1426_, lean_object* v___y_1427_, lean_object* v___y_1428_, lean_object* v___y_1429_, lean_object* v___y_1430_, lean_object* v___y_1431_){
_start:
{
uint8_t v___x_18316__boxed_1432_; lean_object* v_res_1433_; 
v___x_18316__boxed_1432_ = lean_unbox(v___x_1426_);
v_res_1433_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1(v_mvarId_1425_, v___x_18316__boxed_1432_, v___y_1427_, v___y_1428_, v___y_1429_, v___y_1430_);
lean_dec(v___y_1430_);
lean_dec_ref(v___y_1429_);
lean_dec(v___y_1428_);
lean_dec_ref(v___y_1427_);
return v_res_1433_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_assignDelayedMVar___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__7___redArg(lean_object* v_mvarId_1434_, lean_object* v_fvars_1435_, lean_object* v_mvarIdPending_1436_, lean_object* v___y_1437_){
_start:
{
lean_object* v___x_1439_; lean_object* v_mctx_1440_; lean_object* v_cache_1441_; lean_object* v_zetaDeltaFVarIds_1442_; lean_object* v_postponed_1443_; lean_object* v_diag_1444_; lean_object* v___x_1446_; uint8_t v_isShared_1447_; uint8_t v_isSharedCheck_1473_; 
v___x_1439_ = lean_st_ref_take(v___y_1437_);
v_mctx_1440_ = lean_ctor_get(v___x_1439_, 0);
v_cache_1441_ = lean_ctor_get(v___x_1439_, 1);
v_zetaDeltaFVarIds_1442_ = lean_ctor_get(v___x_1439_, 2);
v_postponed_1443_ = lean_ctor_get(v___x_1439_, 3);
v_diag_1444_ = lean_ctor_get(v___x_1439_, 4);
v_isSharedCheck_1473_ = !lean_is_exclusive(v___x_1439_);
if (v_isSharedCheck_1473_ == 0)
{
v___x_1446_ = v___x_1439_;
v_isShared_1447_ = v_isSharedCheck_1473_;
goto v_resetjp_1445_;
}
else
{
lean_inc(v_diag_1444_);
lean_inc(v_postponed_1443_);
lean_inc(v_zetaDeltaFVarIds_1442_);
lean_inc(v_cache_1441_);
lean_inc(v_mctx_1440_);
lean_dec(v___x_1439_);
v___x_1446_ = lean_box(0);
v_isShared_1447_ = v_isSharedCheck_1473_;
goto v_resetjp_1445_;
}
v_resetjp_1445_:
{
lean_object* v_depth_1448_; lean_object* v_levelAssignDepth_1449_; lean_object* v_lmvarCounter_1450_; lean_object* v_mvarCounter_1451_; lean_object* v_lDecls_1452_; lean_object* v_decls_1453_; lean_object* v_userNames_1454_; lean_object* v_lAssignment_1455_; lean_object* v_eAssignment_1456_; lean_object* v_dAssignment_1457_; lean_object* v___x_1459_; uint8_t v_isShared_1460_; uint8_t v_isSharedCheck_1472_; 
v_depth_1448_ = lean_ctor_get(v_mctx_1440_, 0);
v_levelAssignDepth_1449_ = lean_ctor_get(v_mctx_1440_, 1);
v_lmvarCounter_1450_ = lean_ctor_get(v_mctx_1440_, 2);
v_mvarCounter_1451_ = lean_ctor_get(v_mctx_1440_, 3);
v_lDecls_1452_ = lean_ctor_get(v_mctx_1440_, 4);
v_decls_1453_ = lean_ctor_get(v_mctx_1440_, 5);
v_userNames_1454_ = lean_ctor_get(v_mctx_1440_, 6);
v_lAssignment_1455_ = lean_ctor_get(v_mctx_1440_, 7);
v_eAssignment_1456_ = lean_ctor_get(v_mctx_1440_, 8);
v_dAssignment_1457_ = lean_ctor_get(v_mctx_1440_, 9);
v_isSharedCheck_1472_ = !lean_is_exclusive(v_mctx_1440_);
if (v_isSharedCheck_1472_ == 0)
{
v___x_1459_ = v_mctx_1440_;
v_isShared_1460_ = v_isSharedCheck_1472_;
goto v_resetjp_1458_;
}
else
{
lean_inc(v_dAssignment_1457_);
lean_inc(v_eAssignment_1456_);
lean_inc(v_lAssignment_1455_);
lean_inc(v_userNames_1454_);
lean_inc(v_decls_1453_);
lean_inc(v_lDecls_1452_);
lean_inc(v_mvarCounter_1451_);
lean_inc(v_lmvarCounter_1450_);
lean_inc(v_levelAssignDepth_1449_);
lean_inc(v_depth_1448_);
lean_dec(v_mctx_1440_);
v___x_1459_ = lean_box(0);
v_isShared_1460_ = v_isSharedCheck_1472_;
goto v_resetjp_1458_;
}
v_resetjp_1458_:
{
lean_object* v___x_1461_; lean_object* v___x_1462_; lean_object* v___x_1464_; 
v___x_1461_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1461_, 0, v_fvars_1435_);
lean_ctor_set(v___x_1461_, 1, v_mvarIdPending_1436_);
v___x_1462_ = lp_aesop_Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12___redArg(v_dAssignment_1457_, v_mvarId_1434_, v___x_1461_);
if (v_isShared_1460_ == 0)
{
lean_ctor_set(v___x_1459_, 9, v___x_1462_);
v___x_1464_ = v___x_1459_;
goto v_reusejp_1463_;
}
else
{
lean_object* v_reuseFailAlloc_1471_; 
v_reuseFailAlloc_1471_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1471_, 0, v_depth_1448_);
lean_ctor_set(v_reuseFailAlloc_1471_, 1, v_levelAssignDepth_1449_);
lean_ctor_set(v_reuseFailAlloc_1471_, 2, v_lmvarCounter_1450_);
lean_ctor_set(v_reuseFailAlloc_1471_, 3, v_mvarCounter_1451_);
lean_ctor_set(v_reuseFailAlloc_1471_, 4, v_lDecls_1452_);
lean_ctor_set(v_reuseFailAlloc_1471_, 5, v_decls_1453_);
lean_ctor_set(v_reuseFailAlloc_1471_, 6, v_userNames_1454_);
lean_ctor_set(v_reuseFailAlloc_1471_, 7, v_lAssignment_1455_);
lean_ctor_set(v_reuseFailAlloc_1471_, 8, v_eAssignment_1456_);
lean_ctor_set(v_reuseFailAlloc_1471_, 9, v___x_1462_);
v___x_1464_ = v_reuseFailAlloc_1471_;
goto v_reusejp_1463_;
}
v_reusejp_1463_:
{
lean_object* v___x_1466_; 
if (v_isShared_1447_ == 0)
{
lean_ctor_set(v___x_1446_, 0, v___x_1464_);
v___x_1466_ = v___x_1446_;
goto v_reusejp_1465_;
}
else
{
lean_object* v_reuseFailAlloc_1470_; 
v_reuseFailAlloc_1470_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1470_, 0, v___x_1464_);
lean_ctor_set(v_reuseFailAlloc_1470_, 1, v_cache_1441_);
lean_ctor_set(v_reuseFailAlloc_1470_, 2, v_zetaDeltaFVarIds_1442_);
lean_ctor_set(v_reuseFailAlloc_1470_, 3, v_postponed_1443_);
lean_ctor_set(v_reuseFailAlloc_1470_, 4, v_diag_1444_);
v___x_1466_ = v_reuseFailAlloc_1470_;
goto v_reusejp_1465_;
}
v_reusejp_1465_:
{
lean_object* v___x_1467_; lean_object* v___x_1468_; lean_object* v___x_1469_; 
v___x_1467_ = lean_st_ref_set(v___y_1437_, v___x_1466_);
v___x_1468_ = lean_box(0);
v___x_1469_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1469_, 0, v___x_1468_);
return v___x_1469_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_assignDelayedMVar___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__7___redArg___boxed(lean_object* v_mvarId_1474_, lean_object* v_fvars_1475_, lean_object* v_mvarIdPending_1476_, lean_object* v___y_1477_, lean_object* v___y_1478_){
_start:
{
lean_object* v_res_1479_; 
v_res_1479_ = lp_aesop_Lean_assignDelayedMVar___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__7___redArg(v_mvarId_1474_, v_fvars_1475_, v_mvarIdPending_1476_, v___y_1477_);
lean_dec(v___y_1477_);
return v_res_1479_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11_spec__13_spec__18___redArg(lean_object* v_keys_1480_, lean_object* v_i_1481_, lean_object* v_k_1482_){
_start:
{
lean_object* v___x_1483_; uint8_t v___x_1484_; 
v___x_1483_ = lean_array_get_size(v_keys_1480_);
v___x_1484_ = lean_nat_dec_lt(v_i_1481_, v___x_1483_);
if (v___x_1484_ == 0)
{
lean_dec(v_i_1481_);
return v___x_1484_;
}
else
{
lean_object* v_k_x27_1485_; uint8_t v___x_1486_; 
v_k_x27_1485_ = lean_array_fget_borrowed(v_keys_1480_, v_i_1481_);
v___x_1486_ = l_Lean_instBEqMVarId_beq(v_k_1482_, v_k_x27_1485_);
if (v___x_1486_ == 0)
{
lean_object* v___x_1487_; lean_object* v___x_1488_; 
v___x_1487_ = lean_unsigned_to_nat(1u);
v___x_1488_ = lean_nat_add(v_i_1481_, v___x_1487_);
lean_dec(v_i_1481_);
v_i_1481_ = v___x_1488_;
goto _start;
}
else
{
lean_dec(v_i_1481_);
return v___x_1486_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11_spec__13_spec__18___redArg___boxed(lean_object* v_keys_1490_, lean_object* v_i_1491_, lean_object* v_k_1492_){
_start:
{
uint8_t v_res_1493_; lean_object* v_r_1494_; 
v_res_1493_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11_spec__13_spec__18___redArg(v_keys_1490_, v_i_1491_, v_k_1492_);
lean_dec(v_k_1492_);
lean_dec_ref(v_keys_1490_);
v_r_1494_ = lean_box(v_res_1493_);
return v_r_1494_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11_spec__13___redArg(lean_object* v_x_1495_, size_t v_x_1496_, lean_object* v_x_1497_){
_start:
{
if (lean_obj_tag(v_x_1495_) == 0)
{
lean_object* v_es_1498_; lean_object* v___x_1499_; size_t v___x_1500_; size_t v___x_1501_; lean_object* v_j_1502_; lean_object* v___x_1503_; 
v_es_1498_ = lean_ctor_get(v_x_1495_, 0);
v___x_1499_ = lean_box(2);
v___x_1500_ = ((size_t)31ULL);
v___x_1501_ = lean_usize_land(v_x_1496_, v___x_1500_);
v_j_1502_ = lean_usize_to_nat(v___x_1501_);
v___x_1503_ = lean_array_get_borrowed(v___x_1499_, v_es_1498_, v_j_1502_);
lean_dec(v_j_1502_);
switch(lean_obj_tag(v___x_1503_))
{
case 0:
{
lean_object* v_key_1504_; uint8_t v___x_1505_; 
v_key_1504_ = lean_ctor_get(v___x_1503_, 0);
v___x_1505_ = l_Lean_instBEqMVarId_beq(v_x_1497_, v_key_1504_);
return v___x_1505_;
}
case 1:
{
lean_object* v_node_1506_; size_t v___x_1507_; size_t v___x_1508_; 
v_node_1506_ = lean_ctor_get(v___x_1503_, 0);
v___x_1507_ = ((size_t)5ULL);
v___x_1508_ = lean_usize_shift_right(v_x_1496_, v___x_1507_);
v_x_1495_ = v_node_1506_;
v_x_1496_ = v___x_1508_;
goto _start;
}
default: 
{
uint8_t v___x_1510_; 
v___x_1510_ = 0;
return v___x_1510_;
}
}
}
else
{
lean_object* v_ks_1511_; lean_object* v___x_1512_; uint8_t v___x_1513_; 
v_ks_1511_ = lean_ctor_get(v_x_1495_, 0);
v___x_1512_ = lean_unsigned_to_nat(0u);
v___x_1513_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11_spec__13_spec__18___redArg(v_ks_1511_, v___x_1512_, v_x_1497_);
return v___x_1513_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11_spec__13___redArg___boxed(lean_object* v_x_1514_, lean_object* v_x_1515_, lean_object* v_x_1516_){
_start:
{
size_t v_x_18535__boxed_1517_; uint8_t v_res_1518_; lean_object* v_r_1519_; 
v_x_18535__boxed_1517_ = lean_unbox_usize(v_x_1515_);
lean_dec(v_x_1515_);
v_res_1518_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11_spec__13___redArg(v_x_1514_, v_x_18535__boxed_1517_, v_x_1516_);
lean_dec(v_x_1516_);
lean_dec_ref(v_x_1514_);
v_r_1519_ = lean_box(v_res_1518_);
return v_r_1519_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11___redArg(lean_object* v_x_1520_, lean_object* v_x_1521_){
_start:
{
uint64_t v___x_1522_; size_t v___x_1523_; uint8_t v___x_1524_; 
v___x_1522_ = l_Lean_instHashableMVarId_hash(v_x_1521_);
v___x_1523_ = lean_uint64_to_usize(v___x_1522_);
v___x_1524_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11_spec__13___redArg(v_x_1520_, v___x_1523_, v_x_1521_);
return v___x_1524_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11___redArg___boxed(lean_object* v_x_1525_, lean_object* v_x_1526_){
_start:
{
uint8_t v_res_1527_; lean_object* v_r_1528_; 
v_res_1527_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11___redArg(v_x_1525_, v_x_1526_);
lean_dec(v_x_1526_);
lean_dec_ref(v_x_1525_);
v_r_1528_ = lean_box(v_res_1527_);
return v_r_1528_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9___redArg(lean_object* v_mvarId_1529_, lean_object* v___y_1530_){
_start:
{
lean_object* v___x_1532_; lean_object* v_mctx_1533_; lean_object* v_eAssignment_1534_; lean_object* v_dAssignment_1535_; uint8_t v___x_1536_; 
v___x_1532_ = lean_st_ref_get(v___y_1530_);
v_mctx_1533_ = lean_ctor_get(v___x_1532_, 0);
lean_inc_ref(v_mctx_1533_);
lean_dec(v___x_1532_);
v_eAssignment_1534_ = lean_ctor_get(v_mctx_1533_, 8);
lean_inc_ref(v_eAssignment_1534_);
v_dAssignment_1535_ = lean_ctor_get(v_mctx_1533_, 9);
lean_inc_ref(v_dAssignment_1535_);
lean_dec_ref(v_mctx_1533_);
v___x_1536_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11___redArg(v_eAssignment_1534_, v_mvarId_1529_);
lean_dec_ref(v_eAssignment_1534_);
if (v___x_1536_ == 0)
{
uint8_t v___x_1537_; lean_object* v___x_1538_; lean_object* v___x_1539_; 
v___x_1537_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11___redArg(v_dAssignment_1535_, v_mvarId_1529_);
lean_dec_ref(v_dAssignment_1535_);
v___x_1538_ = lean_box(v___x_1537_);
v___x_1539_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1539_, 0, v___x_1538_);
return v___x_1539_;
}
else
{
lean_object* v___x_1540_; lean_object* v___x_1541_; 
lean_dec_ref(v_dAssignment_1535_);
v___x_1540_ = lean_box(v___x_1536_);
v___x_1541_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1541_, 0, v___x_1540_);
return v___x_1541_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9___redArg___boxed(lean_object* v_mvarId_1542_, lean_object* v___y_1543_, lean_object* v___y_1544_){
_start:
{
lean_object* v_res_1545_; 
v_res_1545_ = lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9___redArg(v_mvarId_1542_, v___y_1543_);
lean_dec(v___y_1543_);
lean_dec(v_mvarId_1542_);
return v_res_1545_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__8(lean_object* v_a_1546_, lean_object* v_a_1547_){
_start:
{
if (lean_obj_tag(v_a_1546_) == 0)
{
lean_object* v___x_1548_; 
v___x_1548_ = l_List_reverse___redArg(v_a_1547_);
return v___x_1548_;
}
else
{
lean_object* v_head_1549_; lean_object* v_tail_1550_; lean_object* v___x_1552_; uint8_t v_isShared_1553_; uint8_t v_isSharedCheck_1559_; 
v_head_1549_ = lean_ctor_get(v_a_1546_, 0);
v_tail_1550_ = lean_ctor_get(v_a_1546_, 1);
v_isSharedCheck_1559_ = !lean_is_exclusive(v_a_1546_);
if (v_isSharedCheck_1559_ == 0)
{
v___x_1552_ = v_a_1546_;
v_isShared_1553_ = v_isSharedCheck_1559_;
goto v_resetjp_1551_;
}
else
{
lean_inc(v_tail_1550_);
lean_inc(v_head_1549_);
lean_dec(v_a_1546_);
v___x_1552_ = lean_box(0);
v_isShared_1553_ = v_isSharedCheck_1559_;
goto v_resetjp_1551_;
}
v_resetjp_1551_:
{
lean_object* v___x_1554_; lean_object* v___x_1556_; 
v___x_1554_ = l_Lean_MessageData_ofExpr(v_head_1549_);
if (v_isShared_1553_ == 0)
{
lean_ctor_set(v___x_1552_, 1, v_a_1547_);
lean_ctor_set(v___x_1552_, 0, v___x_1554_);
v___x_1556_ = v___x_1552_;
goto v_reusejp_1555_;
}
else
{
lean_object* v_reuseFailAlloc_1558_; 
v_reuseFailAlloc_1558_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1558_, 0, v___x_1554_);
lean_ctor_set(v_reuseFailAlloc_1558_, 1, v_a_1547_);
v___x_1556_ = v_reuseFailAlloc_1558_;
goto v_reusejp_1555_;
}
v_reusejp_1555_:
{
v_a_1546_ = v_tail_1550_;
v_a_1547_ = v___x_1556_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__5___redArg(lean_object* v_mvarId_1560_, lean_object* v_val_1561_, lean_object* v___y_1562_){
_start:
{
lean_object* v___x_1564_; lean_object* v_mctx_1565_; lean_object* v_cache_1566_; lean_object* v_zetaDeltaFVarIds_1567_; lean_object* v_postponed_1568_; lean_object* v_diag_1569_; lean_object* v___x_1571_; uint8_t v_isShared_1572_; uint8_t v_isSharedCheck_1597_; 
v___x_1564_ = lean_st_ref_take(v___y_1562_);
v_mctx_1565_ = lean_ctor_get(v___x_1564_, 0);
v_cache_1566_ = lean_ctor_get(v___x_1564_, 1);
v_zetaDeltaFVarIds_1567_ = lean_ctor_get(v___x_1564_, 2);
v_postponed_1568_ = lean_ctor_get(v___x_1564_, 3);
v_diag_1569_ = lean_ctor_get(v___x_1564_, 4);
v_isSharedCheck_1597_ = !lean_is_exclusive(v___x_1564_);
if (v_isSharedCheck_1597_ == 0)
{
v___x_1571_ = v___x_1564_;
v_isShared_1572_ = v_isSharedCheck_1597_;
goto v_resetjp_1570_;
}
else
{
lean_inc(v_diag_1569_);
lean_inc(v_postponed_1568_);
lean_inc(v_zetaDeltaFVarIds_1567_);
lean_inc(v_cache_1566_);
lean_inc(v_mctx_1565_);
lean_dec(v___x_1564_);
v___x_1571_ = lean_box(0);
v_isShared_1572_ = v_isSharedCheck_1597_;
goto v_resetjp_1570_;
}
v_resetjp_1570_:
{
lean_object* v_depth_1573_; lean_object* v_levelAssignDepth_1574_; lean_object* v_lmvarCounter_1575_; lean_object* v_mvarCounter_1576_; lean_object* v_lDecls_1577_; lean_object* v_decls_1578_; lean_object* v_userNames_1579_; lean_object* v_lAssignment_1580_; lean_object* v_eAssignment_1581_; lean_object* v_dAssignment_1582_; lean_object* v___x_1584_; uint8_t v_isShared_1585_; uint8_t v_isSharedCheck_1596_; 
v_depth_1573_ = lean_ctor_get(v_mctx_1565_, 0);
v_levelAssignDepth_1574_ = lean_ctor_get(v_mctx_1565_, 1);
v_lmvarCounter_1575_ = lean_ctor_get(v_mctx_1565_, 2);
v_mvarCounter_1576_ = lean_ctor_get(v_mctx_1565_, 3);
v_lDecls_1577_ = lean_ctor_get(v_mctx_1565_, 4);
v_decls_1578_ = lean_ctor_get(v_mctx_1565_, 5);
v_userNames_1579_ = lean_ctor_get(v_mctx_1565_, 6);
v_lAssignment_1580_ = lean_ctor_get(v_mctx_1565_, 7);
v_eAssignment_1581_ = lean_ctor_get(v_mctx_1565_, 8);
v_dAssignment_1582_ = lean_ctor_get(v_mctx_1565_, 9);
v_isSharedCheck_1596_ = !lean_is_exclusive(v_mctx_1565_);
if (v_isSharedCheck_1596_ == 0)
{
v___x_1584_ = v_mctx_1565_;
v_isShared_1585_ = v_isSharedCheck_1596_;
goto v_resetjp_1583_;
}
else
{
lean_inc(v_dAssignment_1582_);
lean_inc(v_eAssignment_1581_);
lean_inc(v_lAssignment_1580_);
lean_inc(v_userNames_1579_);
lean_inc(v_decls_1578_);
lean_inc(v_lDecls_1577_);
lean_inc(v_mvarCounter_1576_);
lean_inc(v_lmvarCounter_1575_);
lean_inc(v_levelAssignDepth_1574_);
lean_inc(v_depth_1573_);
lean_dec(v_mctx_1565_);
v___x_1584_ = lean_box(0);
v_isShared_1585_ = v_isSharedCheck_1596_;
goto v_resetjp_1583_;
}
v_resetjp_1583_:
{
lean_object* v___x_1586_; lean_object* v___x_1588_; 
v___x_1586_ = lp_aesop_Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12___redArg(v_eAssignment_1581_, v_mvarId_1560_, v_val_1561_);
if (v_isShared_1585_ == 0)
{
lean_ctor_set(v___x_1584_, 8, v___x_1586_);
v___x_1588_ = v___x_1584_;
goto v_reusejp_1587_;
}
else
{
lean_object* v_reuseFailAlloc_1595_; 
v_reuseFailAlloc_1595_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1595_, 0, v_depth_1573_);
lean_ctor_set(v_reuseFailAlloc_1595_, 1, v_levelAssignDepth_1574_);
lean_ctor_set(v_reuseFailAlloc_1595_, 2, v_lmvarCounter_1575_);
lean_ctor_set(v_reuseFailAlloc_1595_, 3, v_mvarCounter_1576_);
lean_ctor_set(v_reuseFailAlloc_1595_, 4, v_lDecls_1577_);
lean_ctor_set(v_reuseFailAlloc_1595_, 5, v_decls_1578_);
lean_ctor_set(v_reuseFailAlloc_1595_, 6, v_userNames_1579_);
lean_ctor_set(v_reuseFailAlloc_1595_, 7, v_lAssignment_1580_);
lean_ctor_set(v_reuseFailAlloc_1595_, 8, v___x_1586_);
lean_ctor_set(v_reuseFailAlloc_1595_, 9, v_dAssignment_1582_);
v___x_1588_ = v_reuseFailAlloc_1595_;
goto v_reusejp_1587_;
}
v_reusejp_1587_:
{
lean_object* v___x_1590_; 
if (v_isShared_1572_ == 0)
{
lean_ctor_set(v___x_1571_, 0, v___x_1588_);
v___x_1590_ = v___x_1571_;
goto v_reusejp_1589_;
}
else
{
lean_object* v_reuseFailAlloc_1594_; 
v_reuseFailAlloc_1594_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1594_, 0, v___x_1588_);
lean_ctor_set(v_reuseFailAlloc_1594_, 1, v_cache_1566_);
lean_ctor_set(v_reuseFailAlloc_1594_, 2, v_zetaDeltaFVarIds_1567_);
lean_ctor_set(v_reuseFailAlloc_1594_, 3, v_postponed_1568_);
lean_ctor_set(v_reuseFailAlloc_1594_, 4, v_diag_1569_);
v___x_1590_ = v_reuseFailAlloc_1594_;
goto v_reusejp_1589_;
}
v_reusejp_1589_:
{
lean_object* v___x_1591_; lean_object* v___x_1592_; lean_object* v___x_1593_; 
v___x_1591_ = lean_st_ref_set(v___y_1562_, v___x_1590_);
v___x_1592_ = lean_box(0);
v___x_1593_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1593_, 0, v___x_1592_);
return v___x_1593_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__5___redArg___boxed(lean_object* v_mvarId_1598_, lean_object* v_val_1599_, lean_object* v___y_1600_, lean_object* v___y_1601_){
_start:
{
lean_object* v_res_1602_; 
v_res_1602_ = lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__5___redArg(v_mvarId_1598_, v_val_1599_, v___y_1600_);
lean_dec(v___y_1600_);
return v_res_1602_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__3(lean_object* v_s_1603_, lean_object* v_as_1604_, size_t v_sz_1605_, size_t v_i_1606_, lean_object* v_b_1607_, lean_object* v___y_1608_, lean_object* v___y_1609_, lean_object* v___y_1610_, lean_object* v___y_1611_){
_start:
{
uint8_t v___x_1613_; 
v___x_1613_ = lean_usize_dec_lt(v_i_1606_, v_sz_1605_);
if (v___x_1613_ == 0)
{
lean_object* v___x_1614_; 
lean_dec_ref(v_s_1603_);
v___x_1614_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1614_, 0, v_b_1607_);
return v___x_1614_;
}
else
{
lean_object* v_a_1615_; lean_object* v___x_1616_; 
v_a_1615_ = lean_array_uget_borrowed(v_as_1604_, v_i_1606_);
lean_inc(v_a_1615_);
lean_inc_ref(v_s_1603_);
v___x_1616_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar(v_s_1603_, v_a_1615_, v___y_1608_, v___y_1609_, v___y_1610_, v___y_1611_);
if (lean_obj_tag(v___x_1616_) == 0)
{
lean_object* v___x_1617_; size_t v___x_1618_; size_t v___x_1619_; 
lean_dec_ref_known(v___x_1616_, 1);
v___x_1617_ = lean_box(0);
v___x_1618_ = ((size_t)1ULL);
v___x_1619_ = lean_usize_add(v_i_1606_, v___x_1618_);
v_i_1606_ = v___x_1619_;
v_b_1607_ = v___x_1617_;
goto _start;
}
else
{
lean_dec_ref(v_s_1603_);
return v___x_1616_;
}
}
}
}
static lean_object* _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__1(void){
_start:
{
lean_object* v___x_1622_; lean_object* v___x_1623_; 
v___x_1622_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__0));
v___x_1623_ = l_Lean_stringToMessageData(v___x_1622_);
return v___x_1623_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__3(void){
_start:
{
lean_object* v___x_1625_; lean_object* v___x_1626_; 
v___x_1625_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__2));
v___x_1626_ = l_Lean_stringToMessageData(v___x_1625_);
return v___x_1626_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__5(void){
_start:
{
lean_object* v___x_1628_; lean_object* v___x_1629_; 
v___x_1628_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__4));
v___x_1629_ = l_Lean_stringToMessageData(v___x_1628_);
return v___x_1629_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__7(void){
_start:
{
lean_object* v___x_1631_; lean_object* v___x_1632_; 
v___x_1631_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__6));
v___x_1632_ = l_Lean_stringToMessageData(v___x_1631_);
return v___x_1632_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__14(lean_object* v_s_1633_, lean_object* v_as_1634_, size_t v_sz_1635_, size_t v_i_1636_, lean_object* v_b_1637_, lean_object* v___y_1638_, lean_object* v___y_1639_, lean_object* v___y_1640_, lean_object* v___y_1641_){
_start:
{
uint8_t v___x_1643_; 
v___x_1643_ = lean_usize_dec_lt(v_i_1636_, v_sz_1635_);
if (v___x_1643_ == 0)
{
lean_object* v___x_1644_; 
lean_dec_ref(v_s_1633_);
v___x_1644_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1644_, 0, v_b_1637_);
return v___x_1644_;
}
else
{
lean_object* v_a_1645_; lean_object* v___x_1646_; 
v_a_1645_ = lean_array_uget_borrowed(v_as_1634_, v_i_1636_);
lean_inc(v_a_1645_);
lean_inc_ref(v_s_1633_);
v___x_1646_ = lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__13(v_s_1633_, v_a_1645_, v_b_1637_, v___y_1638_, v___y_1639_, v___y_1640_, v___y_1641_);
if (lean_obj_tag(v___x_1646_) == 0)
{
lean_object* v_a_1647_; lean_object* v___x_1649_; uint8_t v_isShared_1650_; uint8_t v_isSharedCheck_1659_; 
v_a_1647_ = lean_ctor_get(v___x_1646_, 0);
v_isSharedCheck_1659_ = !lean_is_exclusive(v___x_1646_);
if (v_isSharedCheck_1659_ == 0)
{
v___x_1649_ = v___x_1646_;
v_isShared_1650_ = v_isSharedCheck_1659_;
goto v_resetjp_1648_;
}
else
{
lean_inc(v_a_1647_);
lean_dec(v___x_1646_);
v___x_1649_ = lean_box(0);
v_isShared_1650_ = v_isSharedCheck_1659_;
goto v_resetjp_1648_;
}
v_resetjp_1648_:
{
if (lean_obj_tag(v_a_1647_) == 0)
{
lean_object* v_a_1651_; lean_object* v___x_1653_; 
lean_dec_ref(v_s_1633_);
v_a_1651_ = lean_ctor_get(v_a_1647_, 0);
lean_inc(v_a_1651_);
lean_dec_ref_known(v_a_1647_, 1);
if (v_isShared_1650_ == 0)
{
lean_ctor_set(v___x_1649_, 0, v_a_1651_);
v___x_1653_ = v___x_1649_;
goto v_reusejp_1652_;
}
else
{
lean_object* v_reuseFailAlloc_1654_; 
v_reuseFailAlloc_1654_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1654_, 0, v_a_1651_);
v___x_1653_ = v_reuseFailAlloc_1654_;
goto v_reusejp_1652_;
}
v_reusejp_1652_:
{
return v___x_1653_;
}
}
else
{
lean_object* v_a_1655_; size_t v___x_1656_; size_t v___x_1657_; 
lean_del_object(v___x_1649_);
v_a_1655_ = lean_ctor_get(v_a_1647_, 0);
lean_inc(v_a_1655_);
lean_dec_ref_known(v_a_1647_, 1);
v___x_1656_ = ((size_t)1ULL);
v___x_1657_ = lean_usize_add(v_i_1636_, v___x_1656_);
v_i_1636_ = v___x_1657_;
v_b_1637_ = v_a_1655_;
goto _start;
}
}
}
else
{
lean_object* v_a_1660_; lean_object* v___x_1662_; uint8_t v_isShared_1663_; uint8_t v_isSharedCheck_1667_; 
lean_dec_ref(v_s_1633_);
v_a_1660_ = lean_ctor_get(v___x_1646_, 0);
v_isSharedCheck_1667_ = !lean_is_exclusive(v___x_1646_);
if (v_isSharedCheck_1667_ == 0)
{
v___x_1662_ = v___x_1646_;
v_isShared_1663_ = v_isSharedCheck_1667_;
goto v_resetjp_1661_;
}
else
{
lean_inc(v_a_1660_);
lean_dec(v___x_1646_);
v___x_1662_ = lean_box(0);
v_isShared_1663_ = v_isSharedCheck_1667_;
goto v_resetjp_1661_;
}
v_resetjp_1661_:
{
lean_object* v___x_1665_; 
if (v_isShared_1663_ == 0)
{
v___x_1665_ = v___x_1662_;
goto v_reusejp_1664_;
}
else
{
lean_object* v_reuseFailAlloc_1666_; 
v_reuseFailAlloc_1666_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1666_, 0, v_a_1660_);
v___x_1665_ = v_reuseFailAlloc_1666_;
goto v_reusejp_1664_;
}
v_reusejp_1664_:
{
return v___x_1665_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar(lean_object* v_s_1668_, lean_object* v_mvarId_1669_, lean_object* v_a_1670_, lean_object* v_a_1671_, lean_object* v_a_1672_, lean_object* v_a_1673_){
_start:
{
lean_object* v___x_1675_; 
v___x_1675_ = lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9___redArg(v_mvarId_1669_, v_a_1671_);
if (lean_obj_tag(v___x_1675_) == 0)
{
lean_object* v_a_1676_; lean_object* v___x_1678_; uint8_t v_isShared_1679_; uint8_t v_isSharedCheck_1866_; 
v_a_1676_ = lean_ctor_get(v___x_1675_, 0);
v_isSharedCheck_1866_ = !lean_is_exclusive(v___x_1675_);
if (v_isSharedCheck_1866_ == 0)
{
v___x_1678_ = v___x_1675_;
v_isShared_1679_ = v_isSharedCheck_1866_;
goto v_resetjp_1677_;
}
else
{
lean_inc(v_a_1676_);
lean_dec(v___x_1675_);
v___x_1678_ = lean_box(0);
v_isShared_1679_ = v_isSharedCheck_1866_;
goto v_resetjp_1677_;
}
v_resetjp_1677_:
{
uint8_t v___x_1680_; 
v___x_1680_ = lean_unbox(v_a_1676_);
lean_dec(v_a_1676_);
if (v___x_1680_ == 0)
{
lean_object* v___x_1681_; 
lean_del_object(v___x_1678_);
v___x_1681_ = lp_aesop_Lean_MVarId_isDeclared___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__10___redArg(v_mvarId_1669_, v_a_1671_);
if (lean_obj_tag(v___x_1681_) == 0)
{
lean_object* v_a_1682_; lean_object* v___f_1683_; lean_object* v___y_1685_; lean_object* v___y_1686_; lean_object* v___y_1687_; lean_object* v___y_1688_; uint8_t v___x_1801_; 
v_a_1682_ = lean_ctor_get(v___x_1681_, 0);
lean_inc(v_a_1682_);
lean_dec_ref_known(v___x_1681_, 1);
lean_inc(v_mvarId_1669_);
v___f_1683_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__0___boxed), 6, 1);
lean_closure_set(v___f_1683_, 0, v_mvarId_1669_);
v___x_1801_ = lean_unbox(v_a_1682_);
lean_dec(v_a_1682_);
if (v___x_1801_ == 0)
{
uint8_t v___x_1802_; lean_object* v___x_1803_; lean_object* v___f_1804_; lean_object* v___x_1805_; 
v___x_1802_ = 1;
v___x_1803_ = lean_box(v___x_1802_);
lean_inc(v_mvarId_1669_);
v___f_1804_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___lam__1___boxed), 7, 2);
lean_closure_set(v___f_1804_, 0, v_mvarId_1669_);
lean_closure_set(v___f_1804_, 1, v___x_1803_);
lean_inc_ref(v_s_1668_);
v___x_1805_ = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(v_s_1668_, v___f_1804_, v_a_1670_, v_a_1671_, v_a_1672_, v_a_1673_);
if (lean_obj_tag(v___x_1805_) == 0)
{
lean_object* v_a_1806_; lean_object* v_fst_1807_; lean_object* v_snd_1808_; lean_object* v___x_1809_; lean_object* v_mctx_1810_; lean_object* v_cache_1811_; lean_object* v_zetaDeltaFVarIds_1812_; lean_object* v_postponed_1813_; lean_object* v_diag_1814_; lean_object* v___x_1816_; uint8_t v_isShared_1817_; uint8_t v_isSharedCheck_1845_; 
v_a_1806_ = lean_ctor_get(v___x_1805_, 0);
lean_inc(v_a_1806_);
lean_dec_ref_known(v___x_1805_, 1);
v_fst_1807_ = lean_ctor_get(v_a_1806_, 0);
lean_inc(v_fst_1807_);
v_snd_1808_ = lean_ctor_get(v_a_1806_, 1);
lean_inc(v_snd_1808_);
lean_dec(v_a_1806_);
v___x_1809_ = lean_st_ref_take(v_a_1671_);
v_mctx_1810_ = lean_ctor_get(v___x_1809_, 0);
v_cache_1811_ = lean_ctor_get(v___x_1809_, 1);
v_zetaDeltaFVarIds_1812_ = lean_ctor_get(v___x_1809_, 2);
v_postponed_1813_ = lean_ctor_get(v___x_1809_, 3);
v_diag_1814_ = lean_ctor_get(v___x_1809_, 4);
v_isSharedCheck_1845_ = !lean_is_exclusive(v___x_1809_);
if (v_isSharedCheck_1845_ == 0)
{
v___x_1816_ = v___x_1809_;
v_isShared_1817_ = v_isSharedCheck_1845_;
goto v_resetjp_1815_;
}
else
{
lean_inc(v_diag_1814_);
lean_inc(v_postponed_1813_);
lean_inc(v_zetaDeltaFVarIds_1812_);
lean_inc(v_cache_1811_);
lean_inc(v_mctx_1810_);
lean_dec(v___x_1809_);
v___x_1816_ = lean_box(0);
v_isShared_1817_ = v_isSharedCheck_1845_;
goto v_resetjp_1815_;
}
v_resetjp_1815_:
{
lean_object* v_depth_1818_; lean_object* v_levelAssignDepth_1819_; lean_object* v_lmvarCounter_1820_; lean_object* v_mvarCounter_1821_; lean_object* v_lDecls_1822_; lean_object* v_decls_1823_; lean_object* v_userNames_1824_; lean_object* v_lAssignment_1825_; lean_object* v_eAssignment_1826_; lean_object* v_dAssignment_1827_; lean_object* v___x_1829_; uint8_t v_isShared_1830_; uint8_t v_isSharedCheck_1844_; 
v_depth_1818_ = lean_ctor_get(v_mctx_1810_, 0);
v_levelAssignDepth_1819_ = lean_ctor_get(v_mctx_1810_, 1);
v_lmvarCounter_1820_ = lean_ctor_get(v_mctx_1810_, 2);
v_mvarCounter_1821_ = lean_ctor_get(v_mctx_1810_, 3);
v_lDecls_1822_ = lean_ctor_get(v_mctx_1810_, 4);
v_decls_1823_ = lean_ctor_get(v_mctx_1810_, 5);
v_userNames_1824_ = lean_ctor_get(v_mctx_1810_, 6);
v_lAssignment_1825_ = lean_ctor_get(v_mctx_1810_, 7);
v_eAssignment_1826_ = lean_ctor_get(v_mctx_1810_, 8);
v_dAssignment_1827_ = lean_ctor_get(v_mctx_1810_, 9);
v_isSharedCheck_1844_ = !lean_is_exclusive(v_mctx_1810_);
if (v_isSharedCheck_1844_ == 0)
{
v___x_1829_ = v_mctx_1810_;
v_isShared_1830_ = v_isSharedCheck_1844_;
goto v_resetjp_1828_;
}
else
{
lean_inc(v_dAssignment_1827_);
lean_inc(v_eAssignment_1826_);
lean_inc(v_lAssignment_1825_);
lean_inc(v_userNames_1824_);
lean_inc(v_decls_1823_);
lean_inc(v_lDecls_1822_);
lean_inc(v_mvarCounter_1821_);
lean_inc(v_lmvarCounter_1820_);
lean_inc(v_levelAssignDepth_1819_);
lean_inc(v_depth_1818_);
lean_dec(v_mctx_1810_);
v___x_1829_ = lean_box(0);
v_isShared_1830_ = v_isSharedCheck_1844_;
goto v_resetjp_1828_;
}
v_resetjp_1828_:
{
lean_object* v___x_1831_; lean_object* v___x_1833_; 
lean_inc(v_mvarId_1669_);
v___x_1831_ = lp_aesop_Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12___redArg(v_decls_1823_, v_mvarId_1669_, v_fst_1807_);
if (v_isShared_1830_ == 0)
{
lean_ctor_set(v___x_1829_, 5, v___x_1831_);
v___x_1833_ = v___x_1829_;
goto v_reusejp_1832_;
}
else
{
lean_object* v_reuseFailAlloc_1843_; 
v_reuseFailAlloc_1843_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1843_, 0, v_depth_1818_);
lean_ctor_set(v_reuseFailAlloc_1843_, 1, v_levelAssignDepth_1819_);
lean_ctor_set(v_reuseFailAlloc_1843_, 2, v_lmvarCounter_1820_);
lean_ctor_set(v_reuseFailAlloc_1843_, 3, v_mvarCounter_1821_);
lean_ctor_set(v_reuseFailAlloc_1843_, 4, v_lDecls_1822_);
lean_ctor_set(v_reuseFailAlloc_1843_, 5, v___x_1831_);
lean_ctor_set(v_reuseFailAlloc_1843_, 6, v_userNames_1824_);
lean_ctor_set(v_reuseFailAlloc_1843_, 7, v_lAssignment_1825_);
lean_ctor_set(v_reuseFailAlloc_1843_, 8, v_eAssignment_1826_);
lean_ctor_set(v_reuseFailAlloc_1843_, 9, v_dAssignment_1827_);
v___x_1833_ = v_reuseFailAlloc_1843_;
goto v_reusejp_1832_;
}
v_reusejp_1832_:
{
lean_object* v___x_1835_; 
if (v_isShared_1817_ == 0)
{
lean_ctor_set(v___x_1816_, 0, v___x_1833_);
v___x_1835_ = v___x_1816_;
goto v_reusejp_1834_;
}
else
{
lean_object* v_reuseFailAlloc_1842_; 
v_reuseFailAlloc_1842_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1842_, 0, v___x_1833_);
lean_ctor_set(v_reuseFailAlloc_1842_, 1, v_cache_1811_);
lean_ctor_set(v_reuseFailAlloc_1842_, 2, v_zetaDeltaFVarIds_1812_);
lean_ctor_set(v_reuseFailAlloc_1842_, 3, v_postponed_1813_);
lean_ctor_set(v_reuseFailAlloc_1842_, 4, v_diag_1814_);
v___x_1835_ = v_reuseFailAlloc_1842_;
goto v_reusejp_1834_;
}
v_reusejp_1834_:
{
lean_object* v___x_1836_; lean_object* v_buckets_1837_; lean_object* v___x_1838_; size_t v_sz_1839_; size_t v___x_1840_; lean_object* v___x_1841_; 
v___x_1836_ = lean_st_ref_set(v_a_1671_, v___x_1835_);
v_buckets_1837_ = lean_ctor_get(v_snd_1808_, 1);
lean_inc_ref(v_buckets_1837_);
lean_dec(v_snd_1808_);
v___x_1838_ = lean_box(0);
v_sz_1839_ = lean_array_size(v_buckets_1837_);
v___x_1840_ = ((size_t)0ULL);
lean_inc_ref(v_s_1668_);
v___x_1841_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__14(v_s_1668_, v_buckets_1837_, v_sz_1839_, v___x_1840_, v___x_1838_, v_a_1670_, v_a_1671_, v_a_1672_, v_a_1673_);
lean_dec_ref(v_buckets_1837_);
if (lean_obj_tag(v___x_1841_) == 0)
{
lean_dec_ref_known(v___x_1841_, 1);
v___y_1685_ = v_a_1670_;
v___y_1686_ = v_a_1671_;
v___y_1687_ = v_a_1672_;
v___y_1688_ = v_a_1673_;
goto v___jp_1684_;
}
else
{
lean_dec_ref(v___f_1683_);
lean_dec(v_mvarId_1669_);
lean_dec_ref(v_s_1668_);
return v___x_1841_;
}
}
}
}
}
}
else
{
lean_object* v_a_1846_; lean_object* v___x_1848_; uint8_t v_isShared_1849_; uint8_t v_isSharedCheck_1853_; 
lean_dec_ref(v___f_1683_);
lean_dec(v_mvarId_1669_);
lean_dec_ref(v_s_1668_);
v_a_1846_ = lean_ctor_get(v___x_1805_, 0);
v_isSharedCheck_1853_ = !lean_is_exclusive(v___x_1805_);
if (v_isSharedCheck_1853_ == 0)
{
v___x_1848_ = v___x_1805_;
v_isShared_1849_ = v_isSharedCheck_1853_;
goto v_resetjp_1847_;
}
else
{
lean_inc(v_a_1846_);
lean_dec(v___x_1805_);
v___x_1848_ = lean_box(0);
v_isShared_1849_ = v_isSharedCheck_1853_;
goto v_resetjp_1847_;
}
v_resetjp_1847_:
{
lean_object* v___x_1851_; 
if (v_isShared_1849_ == 0)
{
v___x_1851_ = v___x_1848_;
goto v_reusejp_1850_;
}
else
{
lean_object* v_reuseFailAlloc_1852_; 
v_reuseFailAlloc_1852_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1852_, 0, v_a_1846_);
v___x_1851_ = v_reuseFailAlloc_1852_;
goto v_reusejp_1850_;
}
v_reusejp_1850_:
{
return v___x_1851_;
}
}
}
}
else
{
v___y_1685_ = v_a_1670_;
v___y_1686_ = v_a_1671_;
v___y_1687_ = v_a_1672_;
v___y_1688_ = v_a_1673_;
goto v___jp_1684_;
}
v___jp_1684_:
{
lean_object* v___x_1689_; 
lean_inc_ref(v_s_1668_);
v___x_1689_ = lp_batteries_Lean_Meta_SavedState_runMetaM_x27___redArg(v_s_1668_, v___f_1683_, v___y_1685_, v___y_1686_, v___y_1687_, v___y_1688_);
if (lean_obj_tag(v___x_1689_) == 0)
{
lean_object* v_a_1690_; lean_object* v___x_1692_; uint8_t v_isShared_1693_; uint8_t v_isSharedCheck_1792_; 
v_a_1690_ = lean_ctor_get(v___x_1689_, 0);
v_isSharedCheck_1792_ = !lean_is_exclusive(v___x_1689_);
if (v_isSharedCheck_1792_ == 0)
{
v___x_1692_ = v___x_1689_;
v_isShared_1693_ = v_isSharedCheck_1792_;
goto v_resetjp_1691_;
}
else
{
lean_inc(v_a_1690_);
lean_dec(v___x_1689_);
v___x_1692_ = lean_box(0);
v_isShared_1693_ = v_isSharedCheck_1792_;
goto v_resetjp_1691_;
}
v_resetjp_1691_:
{
if (lean_obj_tag(v_a_1690_) == 0)
{
lean_object* v___x_1694_; lean_object* v___x_1696_; 
lean_dec(v_mvarId_1669_);
lean_dec_ref(v_s_1668_);
v___x_1694_ = lean_box(0);
if (v_isShared_1693_ == 0)
{
lean_ctor_set(v___x_1692_, 0, v___x_1694_);
v___x_1696_ = v___x_1692_;
goto v_reusejp_1695_;
}
else
{
lean_object* v_reuseFailAlloc_1697_; 
v_reuseFailAlloc_1697_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1697_, 0, v___x_1694_);
v___x_1696_ = v_reuseFailAlloc_1697_;
goto v_reusejp_1695_;
}
v_reusejp_1695_:
{
return v___x_1696_;
}
}
else
{
lean_object* v_val_1698_; 
lean_del_object(v___x_1692_);
v_val_1698_ = lean_ctor_get(v_a_1690_, 0);
lean_inc(v_val_1698_);
lean_dec_ref_known(v_a_1690_, 1);
if (lean_obj_tag(v_val_1698_) == 0)
{
lean_object* v_val_1699_; lean_object* v___x_1700_; 
v_val_1699_ = lean_ctor_get(v_val_1698_, 0);
lean_inc_n(v_val_1699_, 2);
lean_dec_ref_known(v_val_1698_, 1);
v___x_1700_ = l_Lean_Meta_getMVars(v_val_1699_, v___y_1685_, v___y_1686_, v___y_1687_, v___y_1688_);
if (lean_obj_tag(v___x_1700_) == 0)
{
lean_object* v_a_1701_; lean_object* v___x_1702_; size_t v_sz_1703_; size_t v___x_1704_; lean_object* v___x_1705_; 
v_a_1701_ = lean_ctor_get(v___x_1700_, 0);
lean_inc(v_a_1701_);
lean_dec_ref_known(v___x_1700_, 1);
v___x_1702_ = lean_box(0);
v_sz_1703_ = lean_array_size(v_a_1701_);
v___x_1704_ = ((size_t)0ULL);
v___x_1705_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__3(v_s_1668_, v_a_1701_, v_sz_1703_, v___x_1704_, v___x_1702_, v___y_1685_, v___y_1686_, v___y_1687_, v___y_1688_);
lean_dec(v_a_1701_);
if (lean_obj_tag(v___x_1705_) == 0)
{
lean_object* v___x_1706_; lean_object* v___x_1707_; 
lean_dec_ref_known(v___x_1705_, 1);
v___x_1706_ = lp_aesop_Aesop_TraceOption_extraction;
v___x_1707_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__4___redArg(v___x_1706_, v___y_1687_);
if (lean_obj_tag(v___x_1707_) == 0)
{
lean_object* v_a_1708_; uint8_t v___x_1709_; 
v_a_1708_ = lean_ctor_get(v___x_1707_, 0);
lean_inc(v_a_1708_);
lean_dec_ref_known(v___x_1707_, 1);
v___x_1709_ = lean_unbox(v_a_1708_);
lean_dec(v_a_1708_);
if (v___x_1709_ == 0)
{
lean_object* v___x_1710_; 
v___x_1710_ = lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__5___redArg(v_mvarId_1669_, v_val_1699_, v___y_1686_);
return v___x_1710_;
}
else
{
lean_object* v_traceClass_1711_; lean_object* v___x_1712_; lean_object* v___x_1713_; lean_object* v___x_1714_; lean_object* v___x_1715_; lean_object* v___x_1716_; lean_object* v___x_1717_; lean_object* v___x_1718_; lean_object* v___x_1719_; lean_object* v___x_1720_; 
v_traceClass_1711_ = lean_ctor_get(v___x_1706_, 0);
v___x_1712_ = lean_obj_once(&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__1, &lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__1_once, _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__1);
lean_inc(v_mvarId_1669_);
v___x_1713_ = l_Lean_MessageData_ofName(v_mvarId_1669_);
v___x_1714_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1714_, 0, v___x_1712_);
lean_ctor_set(v___x_1714_, 1, v___x_1713_);
v___x_1715_ = lean_obj_once(&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__3, &lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__3_once, _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__3);
v___x_1716_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1716_, 0, v___x_1714_);
lean_ctor_set(v___x_1716_, 1, v___x_1715_);
v___x_1717_ = lean_expr_dbg_to_string(v_val_1699_);
v___x_1718_ = l_Lean_stringToMessageData(v___x_1717_);
v___x_1719_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1719_, 0, v___x_1716_);
lean_ctor_set(v___x_1719_, 1, v___x_1718_);
lean_inc(v_traceClass_1711_);
v___x_1720_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6(v_traceClass_1711_, v___x_1719_, v___y_1685_, v___y_1686_, v___y_1687_, v___y_1688_);
if (lean_obj_tag(v___x_1720_) == 0)
{
lean_object* v___x_1721_; 
lean_dec_ref_known(v___x_1720_, 1);
v___x_1721_ = lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__5___redArg(v_mvarId_1669_, v_val_1699_, v___y_1686_);
return v___x_1721_;
}
else
{
lean_dec(v_val_1699_);
lean_dec(v_mvarId_1669_);
return v___x_1720_;
}
}
}
else
{
lean_object* v_a_1722_; lean_object* v___x_1724_; uint8_t v_isShared_1725_; uint8_t v_isSharedCheck_1729_; 
lean_dec(v_val_1699_);
lean_dec(v_mvarId_1669_);
v_a_1722_ = lean_ctor_get(v___x_1707_, 0);
v_isSharedCheck_1729_ = !lean_is_exclusive(v___x_1707_);
if (v_isSharedCheck_1729_ == 0)
{
v___x_1724_ = v___x_1707_;
v_isShared_1725_ = v_isSharedCheck_1729_;
goto v_resetjp_1723_;
}
else
{
lean_inc(v_a_1722_);
lean_dec(v___x_1707_);
v___x_1724_ = lean_box(0);
v_isShared_1725_ = v_isSharedCheck_1729_;
goto v_resetjp_1723_;
}
v_resetjp_1723_:
{
lean_object* v___x_1727_; 
if (v_isShared_1725_ == 0)
{
v___x_1727_ = v___x_1724_;
goto v_reusejp_1726_;
}
else
{
lean_object* v_reuseFailAlloc_1728_; 
v_reuseFailAlloc_1728_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1728_, 0, v_a_1722_);
v___x_1727_ = v_reuseFailAlloc_1728_;
goto v_reusejp_1726_;
}
v_reusejp_1726_:
{
return v___x_1727_;
}
}
}
}
else
{
lean_dec(v_val_1699_);
lean_dec(v_mvarId_1669_);
return v___x_1705_;
}
}
else
{
lean_object* v_a_1730_; lean_object* v___x_1732_; uint8_t v_isShared_1733_; uint8_t v_isSharedCheck_1737_; 
lean_dec(v_val_1699_);
lean_dec(v_mvarId_1669_);
lean_dec_ref(v_s_1668_);
v_a_1730_ = lean_ctor_get(v___x_1700_, 0);
v_isSharedCheck_1737_ = !lean_is_exclusive(v___x_1700_);
if (v_isSharedCheck_1737_ == 0)
{
v___x_1732_ = v___x_1700_;
v_isShared_1733_ = v_isSharedCheck_1737_;
goto v_resetjp_1731_;
}
else
{
lean_inc(v_a_1730_);
lean_dec(v___x_1700_);
v___x_1732_ = lean_box(0);
v_isShared_1733_ = v_isSharedCheck_1737_;
goto v_resetjp_1731_;
}
v_resetjp_1731_:
{
lean_object* v___x_1735_; 
if (v_isShared_1733_ == 0)
{
v___x_1735_ = v___x_1732_;
goto v_reusejp_1734_;
}
else
{
lean_object* v_reuseFailAlloc_1736_; 
v_reuseFailAlloc_1736_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1736_, 0, v_a_1730_);
v___x_1735_ = v_reuseFailAlloc_1736_;
goto v_reusejp_1734_;
}
v_reusejp_1734_:
{
return v___x_1735_;
}
}
}
}
else
{
lean_object* v_val_1738_; lean_object* v_fvars_1739_; lean_object* v_mvarIdPending_1740_; lean_object* v___x_1742_; uint8_t v_isShared_1743_; uint8_t v_isSharedCheck_1791_; 
v_val_1738_ = lean_ctor_get(v_val_1698_, 0);
lean_inc(v_val_1738_);
lean_dec_ref_known(v_val_1698_, 1);
v_fvars_1739_ = lean_ctor_get(v_val_1738_, 0);
v_mvarIdPending_1740_ = lean_ctor_get(v_val_1738_, 1);
v_isSharedCheck_1791_ = !lean_is_exclusive(v_val_1738_);
if (v_isSharedCheck_1791_ == 0)
{
v___x_1742_ = v_val_1738_;
v_isShared_1743_ = v_isSharedCheck_1791_;
goto v_resetjp_1741_;
}
else
{
lean_inc(v_mvarIdPending_1740_);
lean_inc(v_fvars_1739_);
lean_dec(v_val_1738_);
v___x_1742_ = lean_box(0);
v_isShared_1743_ = v_isSharedCheck_1791_;
goto v_resetjp_1741_;
}
v_resetjp_1741_:
{
lean_object* v___x_1744_; lean_object* v___x_1745_; 
lean_inc(v_mvarIdPending_1740_);
v___x_1744_ = l_Lean_mkMVar(v_mvarIdPending_1740_);
v___x_1745_ = l_Lean_Meta_getMVars(v___x_1744_, v___y_1685_, v___y_1686_, v___y_1687_, v___y_1688_);
if (lean_obj_tag(v___x_1745_) == 0)
{
lean_object* v_a_1746_; lean_object* v___x_1747_; size_t v_sz_1748_; size_t v___x_1749_; lean_object* v___x_1750_; 
v_a_1746_ = lean_ctor_get(v___x_1745_, 0);
lean_inc(v_a_1746_);
lean_dec_ref_known(v___x_1745_, 1);
v___x_1747_ = lean_box(0);
v_sz_1748_ = lean_array_size(v_a_1746_);
v___x_1749_ = ((size_t)0ULL);
v___x_1750_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__3(v_s_1668_, v_a_1746_, v_sz_1748_, v___x_1749_, v___x_1747_, v___y_1685_, v___y_1686_, v___y_1687_, v___y_1688_);
lean_dec(v_a_1746_);
if (lean_obj_tag(v___x_1750_) == 0)
{
lean_object* v___x_1751_; lean_object* v___x_1752_; 
lean_dec_ref_known(v___x_1750_, 1);
v___x_1751_ = lp_aesop_Aesop_TraceOption_extraction;
v___x_1752_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__4___redArg(v___x_1751_, v___y_1687_);
if (lean_obj_tag(v___x_1752_) == 0)
{
lean_object* v_a_1753_; uint8_t v___x_1754_; 
v_a_1753_ = lean_ctor_get(v___x_1752_, 0);
lean_inc(v_a_1753_);
lean_dec_ref_known(v___x_1752_, 1);
v___x_1754_ = lean_unbox(v_a_1753_);
lean_dec(v_a_1753_);
if (v___x_1754_ == 0)
{
lean_object* v___x_1755_; 
lean_del_object(v___x_1742_);
v___x_1755_ = lp_aesop_Lean_assignDelayedMVar___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__7___redArg(v_mvarId_1669_, v_fvars_1739_, v_mvarIdPending_1740_, v___y_1686_);
return v___x_1755_;
}
else
{
lean_object* v_traceClass_1756_; lean_object* v___x_1757_; lean_object* v___x_1758_; lean_object* v___x_1760_; 
v_traceClass_1756_ = lean_ctor_get(v___x_1751_, 0);
v___x_1757_ = lean_obj_once(&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__5, &lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__5_once, _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__5);
lean_inc(v_mvarId_1669_);
v___x_1758_ = l_Lean_MessageData_ofName(v_mvarId_1669_);
if (v_isShared_1743_ == 0)
{
lean_ctor_set_tag(v___x_1742_, 7);
lean_ctor_set(v___x_1742_, 1, v___x_1758_);
lean_ctor_set(v___x_1742_, 0, v___x_1757_);
v___x_1760_ = v___x_1742_;
goto v_reusejp_1759_;
}
else
{
lean_object* v_reuseFailAlloc_1774_; 
v_reuseFailAlloc_1774_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1774_, 0, v___x_1757_);
lean_ctor_set(v_reuseFailAlloc_1774_, 1, v___x_1758_);
v___x_1760_ = v_reuseFailAlloc_1774_;
goto v_reusejp_1759_;
}
v_reusejp_1759_:
{
lean_object* v___x_1761_; lean_object* v___x_1762_; lean_object* v___x_1763_; lean_object* v___x_1764_; lean_object* v___x_1765_; lean_object* v___x_1766_; lean_object* v___x_1767_; lean_object* v___x_1768_; lean_object* v___x_1769_; lean_object* v___x_1770_; lean_object* v___x_1771_; lean_object* v___x_1772_; 
v___x_1761_ = lean_obj_once(&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__3, &lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__3_once, _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__3);
v___x_1762_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1762_, 0, v___x_1760_);
lean_ctor_set(v___x_1762_, 1, v___x_1761_);
lean_inc_ref(v_fvars_1739_);
v___x_1763_ = lean_array_to_list(v_fvars_1739_);
v___x_1764_ = lean_box(0);
v___x_1765_ = lp_aesop_List_mapTR_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__8(v___x_1763_, v___x_1764_);
v___x_1766_ = l_Lean_MessageData_ofList(v___x_1765_);
v___x_1767_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1767_, 0, v___x_1762_);
lean_ctor_set(v___x_1767_, 1, v___x_1766_);
v___x_1768_ = lean_obj_once(&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__7, &lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__7_once, _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___closed__7);
v___x_1769_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1769_, 0, v___x_1767_);
lean_ctor_set(v___x_1769_, 1, v___x_1768_);
lean_inc(v_mvarIdPending_1740_);
v___x_1770_ = l_Lean_MessageData_ofName(v_mvarIdPending_1740_);
v___x_1771_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1771_, 0, v___x_1769_);
lean_ctor_set(v___x_1771_, 1, v___x_1770_);
lean_inc(v_traceClass_1756_);
v___x_1772_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6(v_traceClass_1756_, v___x_1771_, v___y_1685_, v___y_1686_, v___y_1687_, v___y_1688_);
if (lean_obj_tag(v___x_1772_) == 0)
{
lean_object* v___x_1773_; 
lean_dec_ref_known(v___x_1772_, 1);
v___x_1773_ = lp_aesop_Lean_assignDelayedMVar___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__7___redArg(v_mvarId_1669_, v_fvars_1739_, v_mvarIdPending_1740_, v___y_1686_);
return v___x_1773_;
}
else
{
lean_dec(v_mvarIdPending_1740_);
lean_dec_ref(v_fvars_1739_);
lean_dec(v_mvarId_1669_);
return v___x_1772_;
}
}
}
}
else
{
lean_object* v_a_1775_; lean_object* v___x_1777_; uint8_t v_isShared_1778_; uint8_t v_isSharedCheck_1782_; 
lean_del_object(v___x_1742_);
lean_dec(v_mvarIdPending_1740_);
lean_dec_ref(v_fvars_1739_);
lean_dec(v_mvarId_1669_);
v_a_1775_ = lean_ctor_get(v___x_1752_, 0);
v_isSharedCheck_1782_ = !lean_is_exclusive(v___x_1752_);
if (v_isSharedCheck_1782_ == 0)
{
v___x_1777_ = v___x_1752_;
v_isShared_1778_ = v_isSharedCheck_1782_;
goto v_resetjp_1776_;
}
else
{
lean_inc(v_a_1775_);
lean_dec(v___x_1752_);
v___x_1777_ = lean_box(0);
v_isShared_1778_ = v_isSharedCheck_1782_;
goto v_resetjp_1776_;
}
v_resetjp_1776_:
{
lean_object* v___x_1780_; 
if (v_isShared_1778_ == 0)
{
v___x_1780_ = v___x_1777_;
goto v_reusejp_1779_;
}
else
{
lean_object* v_reuseFailAlloc_1781_; 
v_reuseFailAlloc_1781_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1781_, 0, v_a_1775_);
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
else
{
lean_del_object(v___x_1742_);
lean_dec(v_mvarIdPending_1740_);
lean_dec_ref(v_fvars_1739_);
lean_dec(v_mvarId_1669_);
return v___x_1750_;
}
}
else
{
lean_object* v_a_1783_; lean_object* v___x_1785_; uint8_t v_isShared_1786_; uint8_t v_isSharedCheck_1790_; 
lean_del_object(v___x_1742_);
lean_dec(v_mvarIdPending_1740_);
lean_dec_ref(v_fvars_1739_);
lean_dec(v_mvarId_1669_);
lean_dec_ref(v_s_1668_);
v_a_1783_ = lean_ctor_get(v___x_1745_, 0);
v_isSharedCheck_1790_ = !lean_is_exclusive(v___x_1745_);
if (v_isSharedCheck_1790_ == 0)
{
v___x_1785_ = v___x_1745_;
v_isShared_1786_ = v_isSharedCheck_1790_;
goto v_resetjp_1784_;
}
else
{
lean_inc(v_a_1783_);
lean_dec(v___x_1745_);
v___x_1785_ = lean_box(0);
v_isShared_1786_ = v_isSharedCheck_1790_;
goto v_resetjp_1784_;
}
v_resetjp_1784_:
{
lean_object* v___x_1788_; 
if (v_isShared_1786_ == 0)
{
v___x_1788_ = v___x_1785_;
goto v_reusejp_1787_;
}
else
{
lean_object* v_reuseFailAlloc_1789_; 
v_reuseFailAlloc_1789_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1789_, 0, v_a_1783_);
v___x_1788_ = v_reuseFailAlloc_1789_;
goto v_reusejp_1787_;
}
v_reusejp_1787_:
{
return v___x_1788_;
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
lean_object* v_a_1793_; lean_object* v___x_1795_; uint8_t v_isShared_1796_; uint8_t v_isSharedCheck_1800_; 
lean_dec(v_mvarId_1669_);
lean_dec_ref(v_s_1668_);
v_a_1793_ = lean_ctor_get(v___x_1689_, 0);
v_isSharedCheck_1800_ = !lean_is_exclusive(v___x_1689_);
if (v_isSharedCheck_1800_ == 0)
{
v___x_1795_ = v___x_1689_;
v_isShared_1796_ = v_isSharedCheck_1800_;
goto v_resetjp_1794_;
}
else
{
lean_inc(v_a_1793_);
lean_dec(v___x_1689_);
v___x_1795_ = lean_box(0);
v_isShared_1796_ = v_isSharedCheck_1800_;
goto v_resetjp_1794_;
}
v_resetjp_1794_:
{
lean_object* v___x_1798_; 
if (v_isShared_1796_ == 0)
{
v___x_1798_ = v___x_1795_;
goto v_reusejp_1797_;
}
else
{
lean_object* v_reuseFailAlloc_1799_; 
v_reuseFailAlloc_1799_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1799_, 0, v_a_1793_);
v___x_1798_ = v_reuseFailAlloc_1799_;
goto v_reusejp_1797_;
}
v_reusejp_1797_:
{
return v___x_1798_;
}
}
}
}
}
else
{
lean_object* v_a_1854_; lean_object* v___x_1856_; uint8_t v_isShared_1857_; uint8_t v_isSharedCheck_1861_; 
lean_dec(v_mvarId_1669_);
lean_dec_ref(v_s_1668_);
v_a_1854_ = lean_ctor_get(v___x_1681_, 0);
v_isSharedCheck_1861_ = !lean_is_exclusive(v___x_1681_);
if (v_isSharedCheck_1861_ == 0)
{
v___x_1856_ = v___x_1681_;
v_isShared_1857_ = v_isSharedCheck_1861_;
goto v_resetjp_1855_;
}
else
{
lean_inc(v_a_1854_);
lean_dec(v___x_1681_);
v___x_1856_ = lean_box(0);
v_isShared_1857_ = v_isSharedCheck_1861_;
goto v_resetjp_1855_;
}
v_resetjp_1855_:
{
lean_object* v___x_1859_; 
if (v_isShared_1857_ == 0)
{
v___x_1859_ = v___x_1856_;
goto v_reusejp_1858_;
}
else
{
lean_object* v_reuseFailAlloc_1860_; 
v_reuseFailAlloc_1860_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1860_, 0, v_a_1854_);
v___x_1859_ = v_reuseFailAlloc_1860_;
goto v_reusejp_1858_;
}
v_reusejp_1858_:
{
return v___x_1859_;
}
}
}
}
else
{
lean_object* v___x_1862_; lean_object* v___x_1864_; 
lean_dec(v_mvarId_1669_);
lean_dec_ref(v_s_1668_);
v___x_1862_ = lean_box(0);
if (v_isShared_1679_ == 0)
{
lean_ctor_set(v___x_1678_, 0, v___x_1862_);
v___x_1864_ = v___x_1678_;
goto v_reusejp_1863_;
}
else
{
lean_object* v_reuseFailAlloc_1865_; 
v_reuseFailAlloc_1865_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1865_, 0, v___x_1862_);
v___x_1864_ = v_reuseFailAlloc_1865_;
goto v_reusejp_1863_;
}
v_reusejp_1863_:
{
return v___x_1864_;
}
}
}
}
else
{
lean_object* v_a_1867_; lean_object* v___x_1869_; uint8_t v_isShared_1870_; uint8_t v_isSharedCheck_1874_; 
lean_dec(v_mvarId_1669_);
lean_dec_ref(v_s_1668_);
v_a_1867_ = lean_ctor_get(v___x_1675_, 0);
v_isSharedCheck_1874_ = !lean_is_exclusive(v___x_1675_);
if (v_isSharedCheck_1874_ == 0)
{
v___x_1869_ = v___x_1675_;
v_isShared_1870_ = v_isSharedCheck_1874_;
goto v_resetjp_1868_;
}
else
{
lean_inc(v_a_1867_);
lean_dec(v___x_1675_);
v___x_1869_ = lean_box(0);
v_isShared_1870_ = v_isSharedCheck_1874_;
goto v_resetjp_1868_;
}
v_resetjp_1868_:
{
lean_object* v___x_1872_; 
if (v_isShared_1870_ == 0)
{
v___x_1872_ = v___x_1869_;
goto v_reusejp_1871_;
}
else
{
lean_object* v_reuseFailAlloc_1873_; 
v_reuseFailAlloc_1873_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1873_, 0, v_a_1867_);
v___x_1872_ = v_reuseFailAlloc_1873_;
goto v_reusejp_1871_;
}
v_reusejp_1871_:
{
return v___x_1872_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__13(lean_object* v_s_1875_, lean_object* v_a_1876_, lean_object* v_a_1877_, lean_object* v___y_1878_, lean_object* v___y_1879_, lean_object* v___y_1880_, lean_object* v___y_1881_){
_start:
{
if (lean_obj_tag(v_a_1876_) == 0)
{
lean_object* v___x_1883_; lean_object* v___x_1884_; 
lean_dec_ref(v_s_1875_);
v___x_1883_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1883_, 0, v_a_1877_);
v___x_1884_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1884_, 0, v___x_1883_);
return v___x_1884_;
}
else
{
lean_object* v_key_1885_; lean_object* v_tail_1886_; lean_object* v___x_1887_; 
v_key_1885_ = lean_ctor_get(v_a_1876_, 0);
lean_inc(v_key_1885_);
v_tail_1886_ = lean_ctor_get(v_a_1876_, 2);
lean_inc(v_tail_1886_);
lean_dec_ref_known(v_a_1876_, 3);
lean_inc_ref(v_s_1875_);
v___x_1887_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar(v_s_1875_, v_key_1885_, v___y_1878_, v___y_1879_, v___y_1880_, v___y_1881_);
if (lean_obj_tag(v___x_1887_) == 0)
{
lean_object* v___x_1888_; 
lean_dec_ref_known(v___x_1887_, 1);
v___x_1888_ = lean_box(0);
v_a_1876_ = v_tail_1886_;
v_a_1877_ = v___x_1888_;
goto _start;
}
else
{
lean_object* v_a_1890_; lean_object* v___x_1892_; uint8_t v_isShared_1893_; uint8_t v_isSharedCheck_1897_; 
lean_dec(v_tail_1886_);
lean_dec_ref(v_s_1875_);
v_a_1890_ = lean_ctor_get(v___x_1887_, 0);
v_isSharedCheck_1897_ = !lean_is_exclusive(v___x_1887_);
if (v_isSharedCheck_1897_ == 0)
{
v___x_1892_ = v___x_1887_;
v_isShared_1893_ = v_isSharedCheck_1897_;
goto v_resetjp_1891_;
}
else
{
lean_inc(v_a_1890_);
lean_dec(v___x_1887_);
v___x_1892_ = lean_box(0);
v_isShared_1893_ = v_isSharedCheck_1897_;
goto v_resetjp_1891_;
}
v_resetjp_1891_:
{
lean_object* v___x_1895_; 
if (v_isShared_1893_ == 0)
{
v___x_1895_ = v___x_1892_;
goto v_reusejp_1894_;
}
else
{
lean_object* v_reuseFailAlloc_1896_; 
v_reuseFailAlloc_1896_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1896_, 0, v_a_1890_);
v___x_1895_ = v_reuseFailAlloc_1896_;
goto v_reusejp_1894_;
}
v_reusejp_1894_:
{
return v___x_1895_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__13___boxed(lean_object* v_s_1898_, lean_object* v_a_1899_, lean_object* v_a_1900_, lean_object* v___y_1901_, lean_object* v___y_1902_, lean_object* v___y_1903_, lean_object* v___y_1904_, lean_object* v___y_1905_){
_start:
{
lean_object* v_res_1906_; 
v_res_1906_ = lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__13(v_s_1898_, v_a_1899_, v_a_1900_, v___y_1901_, v___y_1902_, v___y_1903_, v___y_1904_);
lean_dec(v___y_1904_);
lean_dec_ref(v___y_1903_);
lean_dec(v___y_1902_);
lean_dec_ref(v___y_1901_);
return v_res_1906_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__3___boxed(lean_object* v_s_1907_, lean_object* v_as_1908_, lean_object* v_sz_1909_, lean_object* v_i_1910_, lean_object* v_b_1911_, lean_object* v___y_1912_, lean_object* v___y_1913_, lean_object* v___y_1914_, lean_object* v___y_1915_, lean_object* v___y_1916_){
_start:
{
size_t v_sz_boxed_1917_; size_t v_i_boxed_1918_; lean_object* v_res_1919_; 
v_sz_boxed_1917_ = lean_unbox_usize(v_sz_1909_);
lean_dec(v_sz_1909_);
v_i_boxed_1918_ = lean_unbox_usize(v_i_1910_);
lean_dec(v_i_1910_);
v_res_1919_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__3(v_s_1907_, v_as_1908_, v_sz_boxed_1917_, v_i_boxed_1918_, v_b_1911_, v___y_1912_, v___y_1913_, v___y_1914_, v___y_1915_);
lean_dec(v___y_1915_);
lean_dec_ref(v___y_1914_);
lean_dec(v___y_1913_);
lean_dec_ref(v___y_1912_);
lean_dec_ref(v_as_1908_);
return v_res_1919_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__14___boxed(lean_object* v_s_1920_, lean_object* v_as_1921_, lean_object* v_sz_1922_, lean_object* v_i_1923_, lean_object* v_b_1924_, lean_object* v___y_1925_, lean_object* v___y_1926_, lean_object* v___y_1927_, lean_object* v___y_1928_, lean_object* v___y_1929_){
_start:
{
size_t v_sz_boxed_1930_; size_t v_i_boxed_1931_; lean_object* v_res_1932_; 
v_sz_boxed_1930_ = lean_unbox_usize(v_sz_1922_);
lean_dec(v_sz_1922_);
v_i_boxed_1931_ = lean_unbox_usize(v_i_1923_);
lean_dec(v_i_1923_);
v_res_1932_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__14(v_s_1920_, v_as_1921_, v_sz_boxed_1930_, v_i_boxed_1931_, v_b_1924_, v___y_1925_, v___y_1926_, v___y_1927_, v___y_1928_);
lean_dec(v___y_1928_);
lean_dec_ref(v___y_1927_);
lean_dec(v___y_1926_);
lean_dec_ref(v___y_1925_);
lean_dec_ref(v_as_1921_);
return v_res_1932_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar___boxed(lean_object* v_s_1933_, lean_object* v_mvarId_1934_, lean_object* v_a_1935_, lean_object* v_a_1936_, lean_object* v_a_1937_, lean_object* v_a_1938_, lean_object* v_a_1939_){
_start:
{
lean_object* v_res_1940_; 
v_res_1940_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar(v_s_1933_, v_mvarId_1934_, v_a_1935_, v_a_1936_, v_a_1937_, v_a_1938_);
lean_dec(v_a_1938_);
lean_dec_ref(v_a_1937_);
lean_dec(v_a_1936_);
lean_dec_ref(v_a_1935_);
return v_res_1940_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__4(lean_object* v_opt_1941_, lean_object* v___y_1942_, lean_object* v___y_1943_, lean_object* v___y_1944_, lean_object* v___y_1945_){
_start:
{
lean_object* v___x_1947_; 
v___x_1947_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__4___redArg(v_opt_1941_, v___y_1944_);
return v___x_1947_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__4___boxed(lean_object* v_opt_1948_, lean_object* v___y_1949_, lean_object* v___y_1950_, lean_object* v___y_1951_, lean_object* v___y_1952_, lean_object* v___y_1953_){
_start:
{
lean_object* v_res_1954_; 
v_res_1954_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__4(v_opt_1948_, v___y_1949_, v___y_1950_, v___y_1951_, v___y_1952_);
lean_dec(v___y_1952_);
lean_dec_ref(v___y_1951_);
lean_dec(v___y_1950_);
lean_dec_ref(v___y_1949_);
lean_dec_ref(v_opt_1948_);
return v_res_1954_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__5(lean_object* v_mvarId_1955_, lean_object* v_val_1956_, lean_object* v___y_1957_, lean_object* v___y_1958_, lean_object* v___y_1959_, lean_object* v___y_1960_){
_start:
{
lean_object* v___x_1962_; 
v___x_1962_ = lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__5___redArg(v_mvarId_1955_, v_val_1956_, v___y_1958_);
return v___x_1962_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__5___boxed(lean_object* v_mvarId_1963_, lean_object* v_val_1964_, lean_object* v___y_1965_, lean_object* v___y_1966_, lean_object* v___y_1967_, lean_object* v___y_1968_, lean_object* v___y_1969_){
_start:
{
lean_object* v_res_1970_; 
v_res_1970_ = lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__5(v_mvarId_1963_, v_val_1964_, v___y_1965_, v___y_1966_, v___y_1967_, v___y_1968_);
lean_dec(v___y_1968_);
lean_dec_ref(v___y_1967_);
lean_dec(v___y_1966_);
lean_dec_ref(v___y_1965_);
return v_res_1970_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_assignDelayedMVar___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__7(lean_object* v_mvarId_1971_, lean_object* v_fvars_1972_, lean_object* v_mvarIdPending_1973_, lean_object* v___y_1974_, lean_object* v___y_1975_, lean_object* v___y_1976_, lean_object* v___y_1977_){
_start:
{
lean_object* v___x_1979_; 
v___x_1979_ = lp_aesop_Lean_assignDelayedMVar___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__7___redArg(v_mvarId_1971_, v_fvars_1972_, v_mvarIdPending_1973_, v___y_1975_);
return v___x_1979_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_assignDelayedMVar___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__7___boxed(lean_object* v_mvarId_1980_, lean_object* v_fvars_1981_, lean_object* v_mvarIdPending_1982_, lean_object* v___y_1983_, lean_object* v___y_1984_, lean_object* v___y_1985_, lean_object* v___y_1986_, lean_object* v___y_1987_){
_start:
{
lean_object* v_res_1988_; 
v_res_1988_ = lp_aesop_Lean_assignDelayedMVar___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__7(v_mvarId_1980_, v_fvars_1981_, v_mvarIdPending_1982_, v___y_1983_, v___y_1984_, v___y_1985_, v___y_1986_);
lean_dec(v___y_1986_);
lean_dec_ref(v___y_1985_);
lean_dec(v___y_1984_);
lean_dec_ref(v___y_1983_);
return v_res_1988_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9(lean_object* v_mvarId_1989_, lean_object* v___y_1990_, lean_object* v___y_1991_, lean_object* v___y_1992_, lean_object* v___y_1993_){
_start:
{
lean_object* v___x_1995_; 
v___x_1995_ = lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9___redArg(v_mvarId_1989_, v___y_1991_);
return v___x_1995_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9___boxed(lean_object* v_mvarId_1996_, lean_object* v___y_1997_, lean_object* v___y_1998_, lean_object* v___y_1999_, lean_object* v___y_2000_, lean_object* v___y_2001_){
_start:
{
lean_object* v_res_2002_; 
v_res_2002_ = lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9(v_mvarId_1996_, v___y_1997_, v___y_1998_, v___y_1999_, v___y_2000_);
lean_dec(v___y_2000_);
lean_dec_ref(v___y_1999_);
lean_dec(v___y_1998_);
lean_dec_ref(v___y_1997_);
lean_dec(v_mvarId_1996_);
return v_res_2002_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12(lean_object* v_00_u03b2_2003_, lean_object* v_x_2004_, lean_object* v_x_2005_, lean_object* v_x_2006_){
_start:
{
lean_object* v___x_2007_; 
v___x_2007_ = lp_aesop_Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12___redArg(v_x_2004_, v_x_2005_, v_x_2006_);
return v___x_2007_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11(lean_object* v_00_u03b2_2008_, lean_object* v_x_2009_, lean_object* v_x_2010_){
_start:
{
uint8_t v___x_2011_; 
v___x_2011_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11___redArg(v_x_2009_, v_x_2010_);
return v___x_2011_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11___boxed(lean_object* v_00_u03b2_2012_, lean_object* v_x_2013_, lean_object* v_x_2014_){
_start:
{
uint8_t v_res_2015_; lean_object* v_r_2016_; 
v_res_2015_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11(v_00_u03b2_2012_, v_x_2013_, v_x_2014_);
lean_dec(v_x_2014_);
lean_dec_ref(v_x_2013_);
v_r_2016_ = lean_box(v_res_2015_);
return v_r_2016_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17(lean_object* v_00_u03b2_2017_, lean_object* v_x_2018_, size_t v_x_2019_, size_t v_x_2020_, lean_object* v_x_2021_, lean_object* v_x_2022_){
_start:
{
lean_object* v___x_2023_; 
v___x_2023_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17___redArg(v_x_2018_, v_x_2019_, v_x_2020_, v_x_2021_, v_x_2022_);
return v___x_2023_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17___boxed(lean_object* v_00_u03b2_2024_, lean_object* v_x_2025_, lean_object* v_x_2026_, lean_object* v_x_2027_, lean_object* v_x_2028_, lean_object* v_x_2029_){
_start:
{
size_t v_x_19317__boxed_2030_; size_t v_x_19318__boxed_2031_; lean_object* v_res_2032_; 
v_x_19317__boxed_2030_ = lean_unbox_usize(v_x_2026_);
lean_dec(v_x_2026_);
v_x_19318__boxed_2031_ = lean_unbox_usize(v_x_2027_);
lean_dec(v_x_2027_);
v_res_2032_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17(v_00_u03b2_2024_, v_x_2025_, v_x_19317__boxed_2030_, v_x_19318__boxed_2031_, v_x_2028_, v_x_2029_);
return v_res_2032_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11_spec__13(lean_object* v_00_u03b2_2033_, lean_object* v_x_2034_, size_t v_x_2035_, lean_object* v_x_2036_){
_start:
{
uint8_t v___x_2037_; 
v___x_2037_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11_spec__13___redArg(v_x_2034_, v_x_2035_, v_x_2036_);
return v___x_2037_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11_spec__13___boxed(lean_object* v_00_u03b2_2038_, lean_object* v_x_2039_, lean_object* v_x_2040_, lean_object* v_x_2041_){
_start:
{
size_t v_x_19334__boxed_2042_; uint8_t v_res_2043_; lean_object* v_r_2044_; 
v_x_19334__boxed_2042_ = lean_unbox_usize(v_x_2040_);
lean_dec(v_x_2040_);
v_res_2043_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11_spec__13(v_00_u03b2_2038_, v_x_2039_, v_x_19334__boxed_2042_, v_x_2041_);
lean_dec(v_x_2041_);
lean_dec_ref(v_x_2039_);
v_r_2044_ = lean_box(v_res_2043_);
return v_r_2044_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16(lean_object* v_00_u03b2_2045_, lean_object* v_x_2046_, lean_object* v_x_2047_){
_start:
{
lean_object* v___x_2048_; 
v___x_2048_ = lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16___redArg(v_x_2046_, v_x_2047_);
return v___x_2048_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16___boxed(lean_object* v_00_u03b2_2049_, lean_object* v_x_2050_, lean_object* v_x_2051_){
_start:
{
lean_object* v_res_2052_; 
v_res_2052_ = lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16(v_00_u03b2_2049_, v_x_2050_, v_x_2051_);
lean_dec(v_x_2051_);
lean_dec_ref(v_x_2050_);
return v_res_2052_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__17(lean_object* v_00_u03b1_2053_, lean_object* v_msg_2054_, lean_object* v___y_2055_, lean_object* v___y_2056_, lean_object* v___y_2057_, lean_object* v___y_2058_){
_start:
{
lean_object* v___x_2060_; 
v___x_2060_ = lp_aesop_Lean_throwError___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__17___redArg(v_msg_2054_, v___y_2055_, v___y_2056_, v___y_2057_, v___y_2058_);
return v___x_2060_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__17___boxed(lean_object* v_00_u03b1_2061_, lean_object* v_msg_2062_, lean_object* v___y_2063_, lean_object* v___y_2064_, lean_object* v___y_2065_, lean_object* v___y_2066_, lean_object* v___y_2067_){
_start:
{
lean_object* v_res_2068_; 
v_res_2068_ = lp_aesop_Lean_throwError___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__17(v_00_u03b1_2061_, v_msg_2062_, v___y_2063_, v___y_2064_, v___y_2065_, v___y_2066_);
lean_dec(v___y_2066_);
lean_dec_ref(v___y_2065_);
lean_dec(v___y_2064_);
lean_dec_ref(v___y_2063_);
return v_res_2068_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17_spec__22(lean_object* v_00_u03b2_2069_, lean_object* v_n_2070_, lean_object* v_k_2071_, lean_object* v_v_2072_){
_start:
{
lean_object* v___x_2073_; 
v___x_2073_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17_spec__22___redArg(v_n_2070_, v_k_2071_, v_v_2072_);
return v___x_2073_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17_spec__23(lean_object* v_00_u03b2_2074_, size_t v_depth_2075_, lean_object* v_keys_2076_, lean_object* v_vals_2077_, lean_object* v_heq_2078_, lean_object* v_i_2079_, lean_object* v_entries_2080_){
_start:
{
lean_object* v___x_2081_; 
v___x_2081_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17_spec__23___redArg(v_depth_2075_, v_keys_2076_, v_vals_2077_, v_i_2079_, v_entries_2080_);
return v___x_2081_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17_spec__23___boxed(lean_object* v_00_u03b2_2082_, lean_object* v_depth_2083_, lean_object* v_keys_2084_, lean_object* v_vals_2085_, lean_object* v_heq_2086_, lean_object* v_i_2087_, lean_object* v_entries_2088_){
_start:
{
size_t v_depth_boxed_2089_; lean_object* v_res_2090_; 
v_depth_boxed_2089_ = lean_unbox_usize(v_depth_2083_);
lean_dec(v_depth_2083_);
v_res_2090_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17_spec__23(v_00_u03b2_2082_, v_depth_boxed_2089_, v_keys_2084_, v_vals_2085_, v_heq_2086_, v_i_2087_, v_entries_2088_);
lean_dec_ref(v_vals_2085_);
lean_dec_ref(v_keys_2084_);
return v_res_2090_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11_spec__13_spec__18(lean_object* v_00_u03b2_2091_, lean_object* v_keys_2092_, lean_object* v_vals_2093_, lean_object* v_heq_2094_, lean_object* v_i_2095_, lean_object* v_k_2096_){
_start:
{
uint8_t v___x_2097_; 
v___x_2097_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11_spec__13_spec__18___redArg(v_keys_2092_, v_i_2095_, v_k_2096_);
return v___x_2097_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11_spec__13_spec__18___boxed(lean_object* v_00_u03b2_2098_, lean_object* v_keys_2099_, lean_object* v_vals_2100_, lean_object* v_heq_2101_, lean_object* v_i_2102_, lean_object* v_k_2103_){
_start:
{
uint8_t v_res_2104_; lean_object* v_r_2105_; 
v_res_2104_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__9_spec__11_spec__13_spec__18(v_00_u03b2_2098_, v_keys_2099_, v_vals_2100_, v_heq_2101_, v_i_2102_, v_k_2103_);
lean_dec(v_k_2103_);
lean_dec_ref(v_vals_2100_);
lean_dec_ref(v_keys_2099_);
v_r_2105_ = lean_box(v_res_2104_);
return v_r_2105_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16_spec__21(lean_object* v_00_u03b2_2106_, lean_object* v_x_2107_, size_t v_x_2108_, lean_object* v_x_2109_){
_start:
{
lean_object* v___x_2110_; 
v___x_2110_ = lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16_spec__21___redArg(v_x_2107_, v_x_2108_, v_x_2109_);
return v___x_2110_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16_spec__21___boxed(lean_object* v_00_u03b2_2111_, lean_object* v_x_2112_, lean_object* v_x_2113_, lean_object* v_x_2114_){
_start:
{
size_t v_x_19376__boxed_2115_; lean_object* v_res_2116_; 
v_x_19376__boxed_2115_ = lean_unbox_usize(v_x_2113_);
lean_dec(v_x_2113_);
v_res_2116_ = lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16_spec__21(v_00_u03b2_2111_, v_x_2112_, v_x_19376__boxed_2115_, v_x_2114_);
lean_dec(v_x_2114_);
lean_dec_ref(v_x_2112_);
return v_res_2116_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__25(lean_object* v_00_u03b4_2117_, lean_object* v_t_2118_, lean_object* v_k_2119_){
_start:
{
lean_object* v___x_2120_; 
v___x_2120_ = lp_aesop_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__25___redArg(v_t_2118_, v_k_2119_);
return v___x_2120_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__25___boxed(lean_object* v_00_u03b4_2121_, lean_object* v_t_2122_, lean_object* v_k_2123_){
_start:
{
lean_object* v_res_2124_; 
v_res_2124_ = lp_aesop_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__15_spec__19_spec__25(v_00_u03b4_2121_, v_t_2122_, v_k_2123_);
lean_dec(v_k_2123_);
lean_dec(v_t_2122_);
return v_res_2124_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17_spec__22_spec__30(lean_object* v_00_u03b2_2125_, lean_object* v_x_2126_, lean_object* v_x_2127_, lean_object* v_x_2128_, lean_object* v_x_2129_){
_start:
{
lean_object* v___x_2130_; 
v___x_2130_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__12_spec__17_spec__22_spec__30___redArg(v_x_2126_, v_x_2127_, v_x_2128_, v_x_2129_);
return v___x_2130_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16_spec__21_spec__24(lean_object* v_00_u03b2_2131_, lean_object* v_keys_2132_, lean_object* v_vals_2133_, lean_object* v_heq_2134_, lean_object* v_i_2135_, lean_object* v_k_2136_){
_start:
{
lean_object* v___x_2137_; 
v___x_2137_ = lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16_spec__21_spec__24___redArg(v_keys_2132_, v_vals_2133_, v_i_2135_, v_k_2136_);
return v___x_2137_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16_spec__21_spec__24___boxed(lean_object* v_00_u03b2_2138_, lean_object* v_keys_2139_, lean_object* v_vals_2140_, lean_object* v_heq_2141_, lean_object* v_i_2142_, lean_object* v_k_2143_){
_start:
{
lean_object* v_res_2144_; 
v_res_2144_ = lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__16_spec__21_spec__24(v_00_u03b2_2138_, v_keys_2139_, v_vals_2140_, v_heq_2141_, v_i_2142_, v_k_2143_);
lean_dec(v_k_2143_);
lean_dec_ref(v_vals_2140_);
lean_dec_ref(v_keys_2139_);
return v_res_2144_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__1(void){
_start:
{
lean_object* v___x_2146_; lean_object* v___x_2147_; 
v___x_2146_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__0));
v___x_2147_ = l_Lean_stringToMessageData(v___x_2146_);
return v___x_2147_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__3(void){
_start:
{
lean_object* v___x_2149_; lean_object* v___x_2150_; 
v___x_2149_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__2));
v___x_2150_ = l_Lean_stringToMessageData(v___x_2149_);
return v___x_2150_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__5(void){
_start:
{
lean_object* v___x_2152_; lean_object* v___x_2153_; 
v___x_2152_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__4));
v___x_2153_ = l_Lean_stringToMessageData(v___x_2152_);
return v___x_2153_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__7(void){
_start:
{
lean_object* v___x_2155_; lean_object* v___x_2156_; 
v___x_2155_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__6));
v___x_2156_ = l_Lean_stringToMessageData(v___x_2155_);
return v___x_2156_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal(lean_object* v_parentEnv_2157_, lean_object* v_g_2158_, lean_object* v_a_2159_, lean_object* v_a_2160_, lean_object* v_a_2161_, lean_object* v_a_2162_){
_start:
{
lean_object* v___y_2165_; lean_object* v___y_2166_; lean_object* v___y_2167_; lean_object* v___y_2168_; lean_object* v___x_2250_; lean_object* v___x_2251_; lean_object* v_a_2252_; lean_object* v___x_2254_; uint8_t v_isShared_2255_; uint8_t v_isSharedCheck_2278_; 
v___x_2250_ = lp_aesop_Aesop_TraceOption_extraction;
v___x_2251_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__4___redArg(v___x_2250_, v_a_2161_);
v_a_2252_ = lean_ctor_get(v___x_2251_, 0);
v_isSharedCheck_2278_ = !lean_is_exclusive(v___x_2251_);
if (v_isSharedCheck_2278_ == 0)
{
v___x_2254_ = v___x_2251_;
v_isShared_2255_ = v_isSharedCheck_2278_;
goto v_resetjp_2253_;
}
else
{
lean_inc(v_a_2252_);
lean_dec(v___x_2251_);
v___x_2254_ = lean_box(0);
v_isShared_2255_ = v_isSharedCheck_2278_;
goto v_resetjp_2253_;
}
v___jp_2164_:
{
lean_object* v___x_2169_; lean_object* v_elimGoal_2170_; lean_object* v___x_2171_; lean_object* v_normalizationState_2172_; 
v___x_2169_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_2170_ = lean_ctor_get(v___x_2169_, 1);
lean_inc_ref(v_elimGoal_2170_);
v___x_2171_ = lean_apply_1(v_elimGoal_2170_, v_g_2158_);
v_normalizationState_2172_ = lean_ctor_get(v___x_2171_, 6);
lean_inc(v_normalizationState_2172_);
switch(lean_obj_tag(v_normalizationState_2172_))
{
case 0:
{
lean_object* v_id_2173_; lean_object* v___x_2174_; lean_object* v___x_2175_; lean_object* v___x_2176_; lean_object* v___x_2177_; lean_object* v___x_2178_; lean_object* v___x_2179_; lean_object* v___x_2180_; lean_object* v___x_2181_; lean_object* v___x_2182_; lean_object* v___x_2183_; 
lean_dec_ref(v_parentEnv_2157_);
v_id_2173_ = lean_ctor_get(v___x_2171_, 0);
lean_inc(v_id_2173_);
lean_dec_ref(v___x_2171_);
v___x_2174_ = lean_obj_once(&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__1, &lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__1_once, _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__1);
v___x_2175_ = lean_obj_once(&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__3, &lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__3_once, _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__3);
v___x_2176_ = l_Nat_reprFast(v_id_2173_);
v___x_2177_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2177_, 0, v___x_2176_);
v___x_2178_ = l_Lean_MessageData_ofFormat(v___x_2177_);
v___x_2179_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2179_, 0, v___x_2175_);
lean_ctor_set(v___x_2179_, 1, v___x_2178_);
v___x_2180_ = lean_obj_once(&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__5, &lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__5_once, _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__5);
v___x_2181_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2181_, 0, v___x_2179_);
lean_ctor_set(v___x_2181_, 1, v___x_2180_);
v___x_2182_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2182_, 0, v___x_2174_);
lean_ctor_set(v___x_2182_, 1, v___x_2181_);
v___x_2183_ = lp_aesop_Lean_throwError___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__17___redArg(v___x_2182_, v___y_2165_, v___y_2166_, v___y_2167_, v___y_2168_);
return v___x_2183_;
}
case 1:
{
lean_object* v_postState_2184_; lean_object* v_core_2185_; lean_object* v_toState_2186_; lean_object* v___x_2188_; uint8_t v_isShared_2189_; uint8_t v_isSharedCheck_2224_; 
v_postState_2184_ = lean_ctor_get(v_normalizationState_2172_, 1);
lean_inc_ref(v_postState_2184_);
v_core_2185_ = lean_ctor_get(v_postState_2184_, 0);
lean_inc_ref(v_core_2185_);
v_toState_2186_ = lean_ctor_get(v_core_2185_, 0);
v_isSharedCheck_2224_ = !lean_is_exclusive(v_core_2185_);
if (v_isSharedCheck_2224_ == 0)
{
lean_object* v_unused_2225_; 
v_unused_2225_ = lean_ctor_get(v_core_2185_, 1);
lean_dec(v_unused_2225_);
v___x_2188_ = v_core_2185_;
v_isShared_2189_ = v_isSharedCheck_2224_;
goto v_resetjp_2187_;
}
else
{
lean_inc(v_toState_2186_);
lean_dec(v_core_2185_);
v___x_2188_ = lean_box(0);
v_isShared_2189_ = v_isSharedCheck_2224_;
goto v_resetjp_2187_;
}
v_resetjp_2187_:
{
lean_object* v_children_2190_; lean_object* v_preNormGoal_2191_; lean_object* v_postGoal_2192_; lean_object* v_env_2193_; lean_object* v___x_2194_; lean_object* v___x_2196_; uint8_t v_isShared_2197_; uint8_t v_isSharedCheck_2222_; 
v_children_2190_ = lean_ctor_get(v___x_2171_, 2);
lean_inc_ref(v_children_2190_);
v_preNormGoal_2191_ = lean_ctor_get(v___x_2171_, 5);
lean_inc(v_preNormGoal_2191_);
lean_dec_ref(v___x_2171_);
v_postGoal_2192_ = lean_ctor_get(v_normalizationState_2172_, 0);
lean_inc(v_postGoal_2192_);
lean_dec_ref_known(v_normalizationState_2172_, 3);
v_env_2193_ = lean_ctor_get(v_toState_2186_, 0);
lean_inc_ref_n(v_env_2193_, 2);
lean_dec_ref(v_toState_2186_);
v___x_2194_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications(v_parentEnv_2157_, v_env_2193_, v___y_2167_, v___y_2168_);
v_isSharedCheck_2222_ = !lean_is_exclusive(v___x_2194_);
if (v_isSharedCheck_2222_ == 0)
{
lean_object* v_unused_2223_; 
v_unused_2223_ = lean_ctor_get(v___x_2194_, 0);
lean_dec(v_unused_2223_);
v___x_2196_ = v___x_2194_;
v_isShared_2197_ = v_isSharedCheck_2222_;
goto v_resetjp_2195_;
}
else
{
lean_dec(v___x_2194_);
v___x_2196_ = lean_box(0);
v_isShared_2197_ = v_isSharedCheck_2222_;
goto v_resetjp_2195_;
}
v_resetjp_2195_:
{
lean_object* v___x_2198_; 
v___x_2198_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar(v_postState_2184_, v_preNormGoal_2191_, v___y_2165_, v___y_2166_, v___y_2167_, v___y_2168_);
if (lean_obj_tag(v___x_2198_) == 0)
{
lean_object* v___x_2200_; uint8_t v_isShared_2201_; uint8_t v_isSharedCheck_2212_; 
v_isSharedCheck_2212_ = !lean_is_exclusive(v___x_2198_);
if (v_isSharedCheck_2212_ == 0)
{
lean_object* v_unused_2213_; 
v_unused_2213_ = lean_ctor_get(v___x_2198_, 0);
lean_dec(v_unused_2213_);
v___x_2200_ = v___x_2198_;
v_isShared_2201_ = v_isSharedCheck_2212_;
goto v_resetjp_2199_;
}
else
{
lean_dec(v___x_2198_);
v___x_2200_ = lean_box(0);
v_isShared_2201_ = v_isSharedCheck_2212_;
goto v_resetjp_2199_;
}
v_resetjp_2199_:
{
lean_object* v___x_2203_; 
if (v_isShared_2189_ == 0)
{
lean_ctor_set(v___x_2188_, 1, v_env_2193_);
lean_ctor_set(v___x_2188_, 0, v_children_2190_);
v___x_2203_ = v___x_2188_;
goto v_reusejp_2202_;
}
else
{
lean_object* v_reuseFailAlloc_2211_; 
v_reuseFailAlloc_2211_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2211_, 0, v_children_2190_);
lean_ctor_set(v_reuseFailAlloc_2211_, 1, v_env_2193_);
v___x_2203_ = v_reuseFailAlloc_2211_;
goto v_reusejp_2202_;
}
v_reusejp_2202_:
{
lean_object* v___x_2204_; lean_object* v___x_2206_; 
v___x_2204_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2204_, 0, v_postGoal_2192_);
lean_ctor_set(v___x_2204_, 1, v___x_2203_);
if (v_isShared_2197_ == 0)
{
lean_ctor_set_tag(v___x_2196_, 1);
lean_ctor_set(v___x_2196_, 0, v___x_2204_);
v___x_2206_ = v___x_2196_;
goto v_reusejp_2205_;
}
else
{
lean_object* v_reuseFailAlloc_2210_; 
v_reuseFailAlloc_2210_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2210_, 0, v___x_2204_);
v___x_2206_ = v_reuseFailAlloc_2210_;
goto v_reusejp_2205_;
}
v_reusejp_2205_:
{
lean_object* v___x_2208_; 
if (v_isShared_2201_ == 0)
{
lean_ctor_set(v___x_2200_, 0, v___x_2206_);
v___x_2208_ = v___x_2200_;
goto v_reusejp_2207_;
}
else
{
lean_object* v_reuseFailAlloc_2209_; 
v_reuseFailAlloc_2209_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2209_, 0, v___x_2206_);
v___x_2208_ = v_reuseFailAlloc_2209_;
goto v_reusejp_2207_;
}
v_reusejp_2207_:
{
return v___x_2208_;
}
}
}
}
}
else
{
lean_object* v_a_2214_; lean_object* v___x_2216_; uint8_t v_isShared_2217_; uint8_t v_isSharedCheck_2221_; 
lean_del_object(v___x_2196_);
lean_dec_ref(v_env_2193_);
lean_dec(v_postGoal_2192_);
lean_dec_ref(v_children_2190_);
lean_del_object(v___x_2188_);
v_a_2214_ = lean_ctor_get(v___x_2198_, 0);
v_isSharedCheck_2221_ = !lean_is_exclusive(v___x_2198_);
if (v_isSharedCheck_2221_ == 0)
{
v___x_2216_ = v___x_2198_;
v_isShared_2217_ = v_isSharedCheck_2221_;
goto v_resetjp_2215_;
}
else
{
lean_inc(v_a_2214_);
lean_dec(v___x_2198_);
v___x_2216_ = lean_box(0);
v_isShared_2217_ = v_isSharedCheck_2221_;
goto v_resetjp_2215_;
}
v_resetjp_2215_:
{
lean_object* v___x_2219_; 
if (v_isShared_2217_ == 0)
{
v___x_2219_ = v___x_2216_;
goto v_reusejp_2218_;
}
else
{
lean_object* v_reuseFailAlloc_2220_; 
v_reuseFailAlloc_2220_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2220_, 0, v_a_2214_);
v___x_2219_ = v_reuseFailAlloc_2220_;
goto v_reusejp_2218_;
}
v_reusejp_2218_:
{
return v___x_2219_;
}
}
}
}
}
}
default: 
{
lean_object* v_postState_2226_; lean_object* v_core_2227_; lean_object* v_toState_2228_; lean_object* v_preNormGoal_2229_; lean_object* v_env_2230_; lean_object* v___x_2231_; lean_object* v___x_2232_; 
v_postState_2226_ = lean_ctor_get(v_normalizationState_2172_, 0);
lean_inc_ref(v_postState_2226_);
lean_dec_ref_known(v_normalizationState_2172_, 2);
v_core_2227_ = lean_ctor_get(v_postState_2226_, 0);
v_toState_2228_ = lean_ctor_get(v_core_2227_, 0);
v_preNormGoal_2229_ = lean_ctor_get(v___x_2171_, 5);
lean_inc(v_preNormGoal_2229_);
lean_dec_ref(v___x_2171_);
v_env_2230_ = lean_ctor_get(v_toState_2228_, 0);
lean_inc_ref(v_env_2230_);
v___x_2231_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications(v_parentEnv_2157_, v_env_2230_, v___y_2167_, v___y_2168_);
lean_dec_ref(v___x_2231_);
v___x_2232_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar(v_postState_2226_, v_preNormGoal_2229_, v___y_2165_, v___y_2166_, v___y_2167_, v___y_2168_);
if (lean_obj_tag(v___x_2232_) == 0)
{
lean_object* v___x_2234_; uint8_t v_isShared_2235_; uint8_t v_isSharedCheck_2240_; 
v_isSharedCheck_2240_ = !lean_is_exclusive(v___x_2232_);
if (v_isSharedCheck_2240_ == 0)
{
lean_object* v_unused_2241_; 
v_unused_2241_ = lean_ctor_get(v___x_2232_, 0);
lean_dec(v_unused_2241_);
v___x_2234_ = v___x_2232_;
v_isShared_2235_ = v_isSharedCheck_2240_;
goto v_resetjp_2233_;
}
else
{
lean_dec(v___x_2232_);
v___x_2234_ = lean_box(0);
v_isShared_2235_ = v_isSharedCheck_2240_;
goto v_resetjp_2233_;
}
v_resetjp_2233_:
{
lean_object* v___x_2236_; lean_object* v___x_2238_; 
v___x_2236_ = lean_box(0);
if (v_isShared_2235_ == 0)
{
lean_ctor_set(v___x_2234_, 0, v___x_2236_);
v___x_2238_ = v___x_2234_;
goto v_reusejp_2237_;
}
else
{
lean_object* v_reuseFailAlloc_2239_; 
v_reuseFailAlloc_2239_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2239_, 0, v___x_2236_);
v___x_2238_ = v_reuseFailAlloc_2239_;
goto v_reusejp_2237_;
}
v_reusejp_2237_:
{
return v___x_2238_;
}
}
}
else
{
lean_object* v_a_2242_; lean_object* v___x_2244_; uint8_t v_isShared_2245_; uint8_t v_isSharedCheck_2249_; 
v_a_2242_ = lean_ctor_get(v___x_2232_, 0);
v_isSharedCheck_2249_ = !lean_is_exclusive(v___x_2232_);
if (v_isSharedCheck_2249_ == 0)
{
v___x_2244_ = v___x_2232_;
v_isShared_2245_ = v_isSharedCheck_2249_;
goto v_resetjp_2243_;
}
else
{
lean_inc(v_a_2242_);
lean_dec(v___x_2232_);
v___x_2244_ = lean_box(0);
v_isShared_2245_ = v_isSharedCheck_2249_;
goto v_resetjp_2243_;
}
v_resetjp_2243_:
{
lean_object* v___x_2247_; 
if (v_isShared_2245_ == 0)
{
v___x_2247_ = v___x_2244_;
goto v_reusejp_2246_;
}
else
{
lean_object* v_reuseFailAlloc_2248_; 
v_reuseFailAlloc_2248_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2248_, 0, v_a_2242_);
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
v_resetjp_2253_:
{
uint8_t v___x_2256_; 
v___x_2256_ = lean_unbox(v_a_2252_);
lean_dec(v_a_2252_);
if (v___x_2256_ == 0)
{
lean_del_object(v___x_2254_);
v___y_2165_ = v_a_2159_;
v___y_2166_ = v_a_2160_;
v___y_2167_ = v_a_2161_;
v___y_2168_ = v_a_2162_;
goto v___jp_2164_;
}
else
{
lean_object* v_traceClass_2257_; lean_object* v___x_2258_; lean_object* v_elimGoal_2259_; lean_object* v___x_2260_; lean_object* v_id_2261_; lean_object* v___x_2262_; lean_object* v___x_2263_; lean_object* v___x_2265_; 
v_traceClass_2257_ = lean_ctor_get(v___x_2250_, 0);
v___x_2258_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_2259_ = lean_ctor_get(v___x_2258_, 1);
lean_inc_ref(v_elimGoal_2259_);
lean_inc(v_g_2158_);
v___x_2260_ = lean_apply_1(v_elimGoal_2259_, v_g_2158_);
v_id_2261_ = lean_ctor_get(v___x_2260_, 0);
lean_inc(v_id_2261_);
lean_dec_ref(v___x_2260_);
v___x_2262_ = lean_obj_once(&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__7, &lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__7_once, _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__7);
v___x_2263_ = l_Nat_reprFast(v_id_2261_);
if (v_isShared_2255_ == 0)
{
lean_ctor_set_tag(v___x_2254_, 3);
lean_ctor_set(v___x_2254_, 0, v___x_2263_);
v___x_2265_ = v___x_2254_;
goto v_reusejp_2264_;
}
else
{
lean_object* v_reuseFailAlloc_2277_; 
v_reuseFailAlloc_2277_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2277_, 0, v___x_2263_);
v___x_2265_ = v_reuseFailAlloc_2277_;
goto v_reusejp_2264_;
}
v_reusejp_2264_:
{
lean_object* v___x_2266_; lean_object* v___x_2267_; lean_object* v___x_2268_; 
v___x_2266_ = l_Lean_MessageData_ofFormat(v___x_2265_);
v___x_2267_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2267_, 0, v___x_2262_);
lean_ctor_set(v___x_2267_, 1, v___x_2266_);
lean_inc(v_traceClass_2257_);
v___x_2268_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6(v_traceClass_2257_, v___x_2267_, v_a_2159_, v_a_2160_, v_a_2161_, v_a_2162_);
if (lean_obj_tag(v___x_2268_) == 0)
{
lean_dec_ref_known(v___x_2268_, 1);
v___y_2165_ = v_a_2159_;
v___y_2166_ = v_a_2160_;
v___y_2167_ = v_a_2161_;
v___y_2168_ = v_a_2162_;
goto v___jp_2164_;
}
else
{
lean_object* v_a_2269_; lean_object* v___x_2271_; uint8_t v_isShared_2272_; uint8_t v_isSharedCheck_2276_; 
lean_dec(v_g_2158_);
lean_dec_ref(v_parentEnv_2157_);
v_a_2269_ = lean_ctor_get(v___x_2268_, 0);
v_isSharedCheck_2276_ = !lean_is_exclusive(v___x_2268_);
if (v_isSharedCheck_2276_ == 0)
{
v___x_2271_ = v___x_2268_;
v_isShared_2272_ = v_isSharedCheck_2276_;
goto v_resetjp_2270_;
}
else
{
lean_inc(v_a_2269_);
lean_dec(v___x_2268_);
v___x_2271_ = lean_box(0);
v_isShared_2272_ = v_isSharedCheck_2276_;
goto v_resetjp_2270_;
}
v_resetjp_2270_:
{
lean_object* v___x_2274_; 
if (v_isShared_2272_ == 0)
{
v___x_2274_ = v___x_2271_;
goto v_reusejp_2273_;
}
else
{
lean_object* v_reuseFailAlloc_2275_; 
v_reuseFailAlloc_2275_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2275_, 0, v_a_2269_);
v___x_2274_ = v_reuseFailAlloc_2275_;
goto v_reusejp_2273_;
}
v_reusejp_2273_:
{
return v___x_2274_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___boxed(lean_object* v_parentEnv_2279_, lean_object* v_g_2280_, lean_object* v_a_2281_, lean_object* v_a_2282_, lean_object* v_a_2283_, lean_object* v_a_2284_, lean_object* v_a_2285_){
_start:
{
lean_object* v_res_2286_; 
v_res_2286_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal(v_parentEnv_2279_, v_g_2280_, v_a_2281_, v_a_2282_, v_a_2283_, v_a_2284_);
lean_dec(v_a_2284_);
lean_dec_ref(v_a_2283_);
lean_dec(v_a_2282_);
lean_dec_ref(v_a_2281_);
return v_res_2286_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitRapp___closed__1(void){
_start:
{
lean_object* v___x_2288_; lean_object* v___x_2289_; 
v___x_2288_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitRapp___closed__0));
v___x_2289_ = l_Lean_stringToMessageData(v___x_2288_);
return v___x_2289_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitRapp(lean_object* v_parentEnv_2290_, lean_object* v_parentGoal_2291_, lean_object* v_r_2292_, lean_object* v_a_2293_, lean_object* v_a_2294_, lean_object* v_a_2295_, lean_object* v_a_2296_){
_start:
{
lean_object* v___y_2299_; lean_object* v___y_2300_; lean_object* v___y_2301_; lean_object* v___y_2302_; lean_object* v___x_2350_; lean_object* v___x_2351_; lean_object* v_a_2352_; lean_object* v___x_2354_; uint8_t v_isShared_2355_; uint8_t v_isSharedCheck_2378_; 
v___x_2350_ = lp_aesop_Aesop_TraceOption_extraction;
v___x_2351_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__4___redArg(v___x_2350_, v_a_2295_);
v_a_2352_ = lean_ctor_get(v___x_2351_, 0);
v_isSharedCheck_2378_ = !lean_is_exclusive(v___x_2351_);
if (v_isSharedCheck_2378_ == 0)
{
v___x_2354_ = v___x_2351_;
v_isShared_2355_ = v_isSharedCheck_2378_;
goto v_resetjp_2353_;
}
else
{
lean_inc(v_a_2352_);
lean_dec(v___x_2351_);
v___x_2354_ = lean_box(0);
v_isShared_2355_ = v_isSharedCheck_2378_;
goto v_resetjp_2353_;
}
v___jp_2298_:
{
lean_object* v___x_2303_; lean_object* v_elimRapp_2304_; lean_object* v___x_2305_; lean_object* v_metaState_2306_; lean_object* v_core_2307_; lean_object* v_toState_2308_; lean_object* v___x_2310_; uint8_t v_isShared_2311_; uint8_t v_isSharedCheck_2348_; 
v___x_2303_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_2304_ = lean_ctor_get(v___x_2303_, 3);
lean_inc_ref(v_elimRapp_2304_);
v___x_2305_ = lean_apply_1(v_elimRapp_2304_, v_r_2292_);
v_metaState_2306_ = lean_ctor_get(v___x_2305_, 6);
lean_inc_ref(v_metaState_2306_);
v_core_2307_ = lean_ctor_get(v_metaState_2306_, 0);
lean_inc_ref(v_core_2307_);
v_toState_2308_ = lean_ctor_get(v_core_2307_, 0);
v_isSharedCheck_2348_ = !lean_is_exclusive(v_core_2307_);
if (v_isSharedCheck_2348_ == 0)
{
lean_object* v_unused_2349_; 
v_unused_2349_ = lean_ctor_get(v_core_2307_, 1);
lean_dec(v_unused_2349_);
v___x_2310_ = v_core_2307_;
v_isShared_2311_ = v_isSharedCheck_2348_;
goto v_resetjp_2309_;
}
else
{
lean_inc(v_toState_2308_);
lean_dec(v_core_2307_);
v___x_2310_ = lean_box(0);
v_isShared_2311_ = v_isSharedCheck_2348_;
goto v_resetjp_2309_;
}
v_resetjp_2309_:
{
lean_object* v_children_2312_; lean_object* v_assignedMVars_2313_; lean_object* v_env_2314_; lean_object* v___x_2315_; lean_object* v___x_2316_; 
v_children_2312_ = lean_ctor_get(v___x_2305_, 2);
lean_inc_ref(v_children_2312_);
v_assignedMVars_2313_ = lean_ctor_get(v___x_2305_, 8);
lean_inc_ref(v_assignedMVars_2313_);
lean_dec_ref(v___x_2305_);
v_env_2314_ = lean_ctor_get(v_toState_2308_, 0);
lean_inc_ref_n(v_env_2314_, 2);
lean_dec_ref(v_toState_2308_);
v___x_2315_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyEnvModifications(v_parentEnv_2290_, v_env_2314_, v___y_2301_, v___y_2302_);
lean_dec_ref(v___x_2315_);
lean_inc_ref(v_metaState_2306_);
v___x_2316_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar(v_metaState_2306_, v_parentGoal_2291_, v___y_2299_, v___y_2300_, v___y_2301_, v___y_2302_);
if (lean_obj_tag(v___x_2316_) == 0)
{
lean_object* v___x_2317_; size_t v_sz_2318_; size_t v___x_2319_; lean_object* v___x_2320_; 
lean_dec_ref_known(v___x_2316_, 1);
v___x_2317_ = lean_box(0);
v_sz_2318_ = lean_array_size(v_assignedMVars_2313_);
v___x_2319_ = ((size_t)0ULL);
v___x_2320_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__3(v_metaState_2306_, v_assignedMVars_2313_, v_sz_2318_, v___x_2319_, v___x_2317_, v___y_2299_, v___y_2300_, v___y_2301_, v___y_2302_);
lean_dec_ref(v_assignedMVars_2313_);
if (lean_obj_tag(v___x_2320_) == 0)
{
lean_object* v___x_2322_; uint8_t v_isShared_2323_; uint8_t v_isSharedCheck_2330_; 
v_isSharedCheck_2330_ = !lean_is_exclusive(v___x_2320_);
if (v_isSharedCheck_2330_ == 0)
{
lean_object* v_unused_2331_; 
v_unused_2331_ = lean_ctor_get(v___x_2320_, 0);
lean_dec(v_unused_2331_);
v___x_2322_ = v___x_2320_;
v_isShared_2323_ = v_isSharedCheck_2330_;
goto v_resetjp_2321_;
}
else
{
lean_dec(v___x_2320_);
v___x_2322_ = lean_box(0);
v_isShared_2323_ = v_isSharedCheck_2330_;
goto v_resetjp_2321_;
}
v_resetjp_2321_:
{
lean_object* v___x_2325_; 
if (v_isShared_2311_ == 0)
{
lean_ctor_set(v___x_2310_, 1, v_env_2314_);
lean_ctor_set(v___x_2310_, 0, v_children_2312_);
v___x_2325_ = v___x_2310_;
goto v_reusejp_2324_;
}
else
{
lean_object* v_reuseFailAlloc_2329_; 
v_reuseFailAlloc_2329_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2329_, 0, v_children_2312_);
lean_ctor_set(v_reuseFailAlloc_2329_, 1, v_env_2314_);
v___x_2325_ = v_reuseFailAlloc_2329_;
goto v_reusejp_2324_;
}
v_reusejp_2324_:
{
lean_object* v___x_2327_; 
if (v_isShared_2323_ == 0)
{
lean_ctor_set(v___x_2322_, 0, v___x_2325_);
v___x_2327_ = v___x_2322_;
goto v_reusejp_2326_;
}
else
{
lean_object* v_reuseFailAlloc_2328_; 
v_reuseFailAlloc_2328_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2328_, 0, v___x_2325_);
v___x_2327_ = v_reuseFailAlloc_2328_;
goto v_reusejp_2326_;
}
v_reusejp_2326_:
{
return v___x_2327_;
}
}
}
}
else
{
lean_object* v_a_2332_; lean_object* v___x_2334_; uint8_t v_isShared_2335_; uint8_t v_isSharedCheck_2339_; 
lean_dec_ref(v_env_2314_);
lean_dec_ref(v_children_2312_);
lean_del_object(v___x_2310_);
v_a_2332_ = lean_ctor_get(v___x_2320_, 0);
v_isSharedCheck_2339_ = !lean_is_exclusive(v___x_2320_);
if (v_isSharedCheck_2339_ == 0)
{
v___x_2334_ = v___x_2320_;
v_isShared_2335_ = v_isSharedCheck_2339_;
goto v_resetjp_2333_;
}
else
{
lean_inc(v_a_2332_);
lean_dec(v___x_2320_);
v___x_2334_ = lean_box(0);
v_isShared_2335_ = v_isSharedCheck_2339_;
goto v_resetjp_2333_;
}
v_resetjp_2333_:
{
lean_object* v___x_2337_; 
if (v_isShared_2335_ == 0)
{
v___x_2337_ = v___x_2334_;
goto v_reusejp_2336_;
}
else
{
lean_object* v_reuseFailAlloc_2338_; 
v_reuseFailAlloc_2338_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2338_, 0, v_a_2332_);
v___x_2337_ = v_reuseFailAlloc_2338_;
goto v_reusejp_2336_;
}
v_reusejp_2336_:
{
return v___x_2337_;
}
}
}
}
else
{
lean_object* v_a_2340_; lean_object* v___x_2342_; uint8_t v_isShared_2343_; uint8_t v_isSharedCheck_2347_; 
lean_dec_ref(v_env_2314_);
lean_dec_ref(v_assignedMVars_2313_);
lean_dec_ref(v_children_2312_);
lean_del_object(v___x_2310_);
lean_dec_ref(v_metaState_2306_);
v_a_2340_ = lean_ctor_get(v___x_2316_, 0);
v_isSharedCheck_2347_ = !lean_is_exclusive(v___x_2316_);
if (v_isSharedCheck_2347_ == 0)
{
v___x_2342_ = v___x_2316_;
v_isShared_2343_ = v_isSharedCheck_2347_;
goto v_resetjp_2341_;
}
else
{
lean_inc(v_a_2340_);
lean_dec(v___x_2316_);
v___x_2342_ = lean_box(0);
v_isShared_2343_ = v_isSharedCheck_2347_;
goto v_resetjp_2341_;
}
v_resetjp_2341_:
{
lean_object* v___x_2345_; 
if (v_isShared_2343_ == 0)
{
v___x_2345_ = v___x_2342_;
goto v_reusejp_2344_;
}
else
{
lean_object* v_reuseFailAlloc_2346_; 
v_reuseFailAlloc_2346_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2346_, 0, v_a_2340_);
v___x_2345_ = v_reuseFailAlloc_2346_;
goto v_reusejp_2344_;
}
v_reusejp_2344_:
{
return v___x_2345_;
}
}
}
}
}
v_resetjp_2353_:
{
uint8_t v___x_2356_; 
v___x_2356_ = lean_unbox(v_a_2352_);
lean_dec(v_a_2352_);
if (v___x_2356_ == 0)
{
lean_del_object(v___x_2354_);
v___y_2299_ = v_a_2293_;
v___y_2300_ = v_a_2294_;
v___y_2301_ = v_a_2295_;
v___y_2302_ = v_a_2296_;
goto v___jp_2298_;
}
else
{
lean_object* v_traceClass_2357_; lean_object* v___x_2358_; lean_object* v_elimRapp_2359_; lean_object* v___x_2360_; lean_object* v_id_2361_; lean_object* v___x_2362_; lean_object* v___x_2363_; lean_object* v___x_2365_; 
v_traceClass_2357_ = lean_ctor_get(v___x_2350_, 0);
v___x_2358_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_2359_ = lean_ctor_get(v___x_2358_, 3);
lean_inc_ref(v_elimRapp_2359_);
lean_inc(v_r_2292_);
v___x_2360_ = lean_apply_1(v_elimRapp_2359_, v_r_2292_);
v_id_2361_ = lean_ctor_get(v___x_2360_, 0);
lean_inc(v_id_2361_);
lean_dec_ref(v___x_2360_);
v___x_2362_ = lean_obj_once(&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitRapp___closed__1, &lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitRapp___closed__1_once, _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitRapp___closed__1);
v___x_2363_ = l_Nat_reprFast(v_id_2361_);
if (v_isShared_2355_ == 0)
{
lean_ctor_set_tag(v___x_2354_, 3);
lean_ctor_set(v___x_2354_, 0, v___x_2363_);
v___x_2365_ = v___x_2354_;
goto v_reusejp_2364_;
}
else
{
lean_object* v_reuseFailAlloc_2377_; 
v_reuseFailAlloc_2377_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2377_, 0, v___x_2363_);
v___x_2365_ = v_reuseFailAlloc_2377_;
goto v_reusejp_2364_;
}
v_reusejp_2364_:
{
lean_object* v___x_2366_; lean_object* v___x_2367_; lean_object* v___x_2368_; 
v___x_2366_ = l_Lean_MessageData_ofFormat(v___x_2365_);
v___x_2367_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2367_, 0, v___x_2362_);
lean_ctor_set(v___x_2367_, 1, v___x_2366_);
lean_inc(v_traceClass_2357_);
v___x_2368_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6(v_traceClass_2357_, v___x_2367_, v_a_2293_, v_a_2294_, v_a_2295_, v_a_2296_);
if (lean_obj_tag(v___x_2368_) == 0)
{
lean_dec_ref_known(v___x_2368_, 1);
v___y_2299_ = v_a_2293_;
v___y_2300_ = v_a_2294_;
v___y_2301_ = v_a_2295_;
v___y_2302_ = v_a_2296_;
goto v___jp_2298_;
}
else
{
lean_object* v_a_2369_; lean_object* v___x_2371_; uint8_t v_isShared_2372_; uint8_t v_isSharedCheck_2376_; 
lean_dec(v_r_2292_);
lean_dec(v_parentGoal_2291_);
lean_dec_ref(v_parentEnv_2290_);
v_a_2369_ = lean_ctor_get(v___x_2368_, 0);
v_isSharedCheck_2376_ = !lean_is_exclusive(v___x_2368_);
if (v_isSharedCheck_2376_ == 0)
{
v___x_2371_ = v___x_2368_;
v_isShared_2372_ = v_isSharedCheck_2376_;
goto v_resetjp_2370_;
}
else
{
lean_inc(v_a_2369_);
lean_dec(v___x_2368_);
v___x_2371_ = lean_box(0);
v_isShared_2372_ = v_isSharedCheck_2376_;
goto v_resetjp_2370_;
}
v_resetjp_2370_:
{
lean_object* v___x_2374_; 
if (v_isShared_2372_ == 0)
{
v___x_2374_ = v___x_2371_;
goto v_reusejp_2373_;
}
else
{
lean_object* v_reuseFailAlloc_2375_; 
v_reuseFailAlloc_2375_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2375_, 0, v_a_2369_);
v___x_2374_ = v_reuseFailAlloc_2375_;
goto v_reusejp_2373_;
}
v_reusejp_2373_:
{
return v___x_2374_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitRapp___boxed(lean_object* v_parentEnv_2379_, lean_object* v_parentGoal_2380_, lean_object* v_r_2381_, lean_object* v_a_2382_, lean_object* v_a_2383_, lean_object* v_a_2384_, lean_object* v_a_2385_, lean_object* v_a_2386_){
_start:
{
lean_object* v_res_2387_; 
v_res_2387_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitRapp(v_parentEnv_2379_, v_parentGoal_2380_, v_r_2381_, v_a_2382_, v_a_2383_, v_a_2384_, v_a_2385_);
lean_dec(v_a_2385_);
lean_dec_ref(v_a_2384_);
lean_dec(v_a_2383_);
lean_dec_ref(v_a_2382_);
return v_res_2387_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__4___redArg(size_t v_sz_2388_, size_t v_i_2389_, lean_object* v_bs_2390_){
_start:
{
uint8_t v___x_2392_; 
v___x_2392_ = lean_usize_dec_lt(v_i_2389_, v_sz_2388_);
if (v___x_2392_ == 0)
{
lean_object* v___x_2393_; 
v___x_2393_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2393_, 0, v_bs_2390_);
return v___x_2393_;
}
else
{
lean_object* v_v_2394_; lean_object* v___x_2395_; lean_object* v___x_2396_; lean_object* v_elimGoal_2397_; lean_object* v___x_2398_; lean_object* v_id_2399_; lean_object* v___x_2400_; lean_object* v_bs_x27_2401_; size_t v___x_2402_; size_t v___x_2403_; lean_object* v___x_2404_; 
v_v_2394_ = lean_array_uget_borrowed(v_bs_2390_, v_i_2389_);
v___x_2395_ = lean_st_ref_get(v_v_2394_);
v___x_2396_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_2397_ = lean_ctor_get(v___x_2396_, 1);
lean_inc_ref(v_elimGoal_2397_);
v___x_2398_ = lean_apply_1(v_elimGoal_2397_, v___x_2395_);
v_id_2399_ = lean_ctor_get(v___x_2398_, 0);
lean_inc(v_id_2399_);
lean_dec_ref(v___x_2398_);
v___x_2400_ = lean_unsigned_to_nat(0u);
v_bs_x27_2401_ = lean_array_uset(v_bs_2390_, v_i_2389_, v___x_2400_);
v___x_2402_ = ((size_t)1ULL);
v___x_2403_ = lean_usize_add(v_i_2389_, v___x_2402_);
v___x_2404_ = lean_array_uset(v_bs_x27_2401_, v_i_2389_, v_id_2399_);
v_i_2389_ = v___x_2403_;
v_bs_2390_ = v___x_2404_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__4___redArg___boxed(lean_object* v_sz_2406_, lean_object* v_i_2407_, lean_object* v_bs_2408_, lean_object* v___y_2409_){
_start:
{
size_t v_sz_boxed_2410_; size_t v_i_boxed_2411_; lean_object* v_res_2412_; 
v_sz_boxed_2410_ = lean_unbox_usize(v_sz_2406_);
lean_dec(v_sz_2406_);
v_i_boxed_2411_ = lean_unbox_usize(v_i_2407_);
lean_dec(v_i_2407_);
v_res_2412_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__4___redArg(v_sz_boxed_2410_, v_i_boxed_2411_, v_bs_2408_);
return v_res_2412_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__6___redArg(lean_object* v_as_2416_, size_t v_sz_2417_, size_t v_i_2418_, lean_object* v_b_2419_){
_start:
{
uint8_t v___x_2421_; 
v___x_2421_ = lean_usize_dec_lt(v_i_2418_, v_sz_2417_);
if (v___x_2421_ == 0)
{
lean_object* v___x_2422_; 
v___x_2422_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2422_, 0, v_b_2419_);
return v___x_2422_;
}
else
{
lean_object* v_a_2423_; lean_object* v___x_2424_; lean_object* v___x_2425_; lean_object* v_elimGoal_2426_; lean_object* v___x_2427_; uint8_t v_state_2428_; lean_object* v___x_2429_; uint8_t v___x_2430_; 
lean_dec_ref(v_b_2419_);
v_a_2423_ = lean_array_uget_borrowed(v_as_2416_, v_i_2418_);
v___x_2424_ = lean_st_ref_get(v_a_2423_);
v___x_2425_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_2426_ = lean_ctor_get(v___x_2425_, 1);
lean_inc_ref(v_elimGoal_2426_);
v___x_2427_ = lean_apply_1(v_elimGoal_2426_, v___x_2424_);
v_state_2428_ = lean_ctor_get_uint8(v___x_2427_, sizeof(void*)*14 + 8);
lean_dec_ref(v___x_2427_);
v___x_2429_ = lean_box(0);
v___x_2430_ = lp_aesop_Aesop_GoalState_isProven(v_state_2428_);
if (v___x_2430_ == 0)
{
lean_object* v___x_2431_; size_t v___x_2432_; size_t v___x_2433_; 
v___x_2431_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__6___redArg___closed__0));
v___x_2432_ = ((size_t)1ULL);
v___x_2433_ = lean_usize_add(v_i_2418_, v___x_2432_);
v_i_2418_ = v___x_2433_;
v_b_2419_ = v___x_2431_;
goto _start;
}
else
{
lean_object* v___x_2435_; lean_object* v___x_2436_; lean_object* v___x_2437_; lean_object* v___x_2438_; 
lean_inc(v_a_2423_);
v___x_2435_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2435_, 0, v_a_2423_);
v___x_2436_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2436_, 0, v___x_2435_);
v___x_2437_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2437_, 0, v___x_2436_);
lean_ctor_set(v___x_2437_, 1, v___x_2429_);
v___x_2438_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2438_, 0, v___x_2437_);
return v___x_2438_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__6___redArg___boxed(lean_object* v_as_2439_, lean_object* v_sz_2440_, lean_object* v_i_2441_, lean_object* v_b_2442_, lean_object* v___y_2443_){
_start:
{
size_t v_sz_boxed_2444_; size_t v_i_boxed_2445_; lean_object* v_res_2446_; 
v_sz_boxed_2444_ = lean_unbox_usize(v_sz_2440_);
lean_dec(v_sz_2440_);
v_i_boxed_2445_ = lean_unbox_usize(v_i_2441_);
lean_dec(v_i_2441_);
v_res_2446_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__6___redArg(v_as_2439_, v_sz_boxed_2444_, v_i_boxed_2445_, v_b_2442_);
lean_dec_ref(v_as_2439_);
return v_res_2446_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__5(lean_object* v_a_2447_, lean_object* v_a_2448_){
_start:
{
if (lean_obj_tag(v_a_2447_) == 0)
{
lean_object* v___x_2449_; 
v___x_2449_ = l_List_reverse___redArg(v_a_2448_);
return v___x_2449_;
}
else
{
lean_object* v_head_2450_; lean_object* v_tail_2451_; lean_object* v___x_2453_; uint8_t v_isShared_2454_; uint8_t v_isSharedCheck_2462_; 
v_head_2450_ = lean_ctor_get(v_a_2447_, 0);
v_tail_2451_ = lean_ctor_get(v_a_2447_, 1);
v_isSharedCheck_2462_ = !lean_is_exclusive(v_a_2447_);
if (v_isSharedCheck_2462_ == 0)
{
v___x_2453_ = v_a_2447_;
v_isShared_2454_ = v_isSharedCheck_2462_;
goto v_resetjp_2452_;
}
else
{
lean_inc(v_tail_2451_);
lean_inc(v_head_2450_);
lean_dec(v_a_2447_);
v___x_2453_ = lean_box(0);
v_isShared_2454_ = v_isSharedCheck_2462_;
goto v_resetjp_2452_;
}
v_resetjp_2452_:
{
lean_object* v___x_2455_; lean_object* v___x_2456_; lean_object* v___x_2457_; lean_object* v___x_2459_; 
v___x_2455_ = l_Nat_reprFast(v_head_2450_);
v___x_2456_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2456_, 0, v___x_2455_);
v___x_2457_ = l_Lean_MessageData_ofFormat(v___x_2456_);
if (v_isShared_2454_ == 0)
{
lean_ctor_set(v___x_2453_, 1, v_a_2448_);
lean_ctor_set(v___x_2453_, 0, v___x_2457_);
v___x_2459_ = v___x_2453_;
goto v_reusejp_2458_;
}
else
{
lean_object* v_reuseFailAlloc_2461_; 
v_reuseFailAlloc_2461_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2461_, 0, v___x_2457_);
lean_ctor_set(v_reuseFailAlloc_2461_, 1, v_a_2448_);
v___x_2459_ = v_reuseFailAlloc_2461_;
goto v_reusejp_2458_;
}
v_reusejp_2458_:
{
v_a_2447_ = v_tail_2451_;
v_a_2448_ = v___x_2459_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal_spec__0___redArg(lean_object* v_as_2463_, size_t v_sz_2464_, size_t v_i_2465_, lean_object* v_b_2466_){
_start:
{
uint8_t v___x_2468_; 
v___x_2468_ = lean_usize_dec_lt(v_i_2465_, v_sz_2464_);
if (v___x_2468_ == 0)
{
lean_object* v___x_2469_; 
v___x_2469_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2469_, 0, v_b_2466_);
return v___x_2469_;
}
else
{
lean_object* v_a_2470_; lean_object* v___x_2471_; lean_object* v___x_2472_; lean_object* v_elimRapp_2473_; lean_object* v___x_2474_; uint8_t v_state_2475_; lean_object* v___x_2476_; uint8_t v___x_2477_; 
lean_dec_ref(v_b_2466_);
v_a_2470_ = lean_array_uget_borrowed(v_as_2463_, v_i_2465_);
v___x_2471_ = lean_st_ref_get(v_a_2470_);
v___x_2472_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_2473_ = lean_ctor_get(v___x_2472_, 3);
lean_inc_ref(v_elimRapp_2473_);
v___x_2474_ = lean_apply_1(v_elimRapp_2473_, v___x_2471_);
v_state_2475_ = lean_ctor_get_uint8(v___x_2474_, sizeof(void*)*9 + 8);
lean_dec_ref(v___x_2474_);
v___x_2476_ = lean_box(0);
v___x_2477_ = lp_aesop_Aesop_NodeState_isProven(v_state_2475_);
if (v___x_2477_ == 0)
{
lean_object* v___x_2478_; size_t v___x_2479_; size_t v___x_2480_; 
v___x_2478_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__6___redArg___closed__0));
v___x_2479_ = ((size_t)1ULL);
v___x_2480_ = lean_usize_add(v_i_2465_, v___x_2479_);
v_i_2465_ = v___x_2480_;
v_b_2466_ = v___x_2478_;
goto _start;
}
else
{
lean_object* v___x_2482_; lean_object* v___x_2483_; lean_object* v___x_2484_; lean_object* v___x_2485_; 
lean_inc(v_a_2470_);
v___x_2482_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2482_, 0, v_a_2470_);
v___x_2483_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2483_, 0, v___x_2482_);
v___x_2484_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2484_, 0, v___x_2483_);
lean_ctor_set(v___x_2484_, 1, v___x_2476_);
v___x_2485_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2485_, 0, v___x_2484_);
return v___x_2485_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal_spec__0___redArg___boxed(lean_object* v_as_2486_, lean_object* v_sz_2487_, lean_object* v_i_2488_, lean_object* v_b_2489_, lean_object* v___y_2490_){
_start:
{
size_t v_sz_boxed_2491_; size_t v_i_boxed_2492_; lean_object* v_res_2493_; 
v_sz_boxed_2491_ = lean_unbox_usize(v_sz_2487_);
lean_dec(v_sz_2487_);
v_i_boxed_2492_ = lean_unbox_usize(v_i_2488_);
lean_dec(v_i_2488_);
v_res_2493_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal_spec__0___redArg(v_as_2486_, v_sz_boxed_2491_, v_i_boxed_2492_, v_b_2489_);
lean_dec_ref(v_as_2486_);
return v_res_2493_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster___closed__1(void){
_start:
{
lean_object* v___x_2495_; lean_object* v___x_2496_; 
v___x_2495_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster___closed__0));
v___x_2496_ = l_Lean_stringToMessageData(v___x_2495_);
return v___x_2496_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster___closed__3(void){
_start:
{
lean_object* v___x_2498_; lean_object* v___x_2499_; 
v___x_2498_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster___closed__2));
v___x_2499_ = l_Lean_stringToMessageData(v___x_2498_);
return v___x_2499_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal___closed__1(void){
_start:
{
lean_object* v___x_2501_; lean_object* v___x_2502_; 
v___x_2501_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal___closed__0));
v___x_2502_ = l_Lean_stringToMessageData(v___x_2501_);
return v___x_2502_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofRapp(lean_object* v_parentEnv_2503_, lean_object* v_parentGoal_2504_, lean_object* v_r_2505_, lean_object* v_a_2506_, lean_object* v_a_2507_, lean_object* v_a_2508_, lean_object* v_a_2509_){
_start:
{
lean_object* v___x_2511_; 
v___x_2511_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitRapp(v_parentEnv_2503_, v_parentGoal_2504_, v_r_2505_, v_a_2506_, v_a_2507_, v_a_2508_, v_a_2509_);
if (lean_obj_tag(v___x_2511_) == 0)
{
lean_object* v_a_2512_; lean_object* v___x_2514_; uint8_t v_isShared_2515_; uint8_t v_isSharedCheck_2535_; 
v_a_2512_ = lean_ctor_get(v___x_2511_, 0);
v_isSharedCheck_2535_ = !lean_is_exclusive(v___x_2511_);
if (v_isSharedCheck_2535_ == 0)
{
v___x_2514_ = v___x_2511_;
v_isShared_2515_ = v_isSharedCheck_2535_;
goto v_resetjp_2513_;
}
else
{
lean_inc(v_a_2512_);
lean_dec(v___x_2511_);
v___x_2514_ = lean_box(0);
v_isShared_2515_ = v_isSharedCheck_2535_;
goto v_resetjp_2513_;
}
v_resetjp_2513_:
{
lean_object* v_fst_2516_; lean_object* v_snd_2517_; lean_object* v___x_2518_; lean_object* v___x_2519_; lean_object* v___x_2520_; uint8_t v___x_2521_; 
v_fst_2516_ = lean_ctor_get(v_a_2512_, 0);
lean_inc(v_fst_2516_);
v_snd_2517_ = lean_ctor_get(v_a_2512_, 1);
lean_inc(v_snd_2517_);
lean_dec(v_a_2512_);
v___x_2518_ = lean_unsigned_to_nat(0u);
v___x_2519_ = lean_array_get_size(v_fst_2516_);
v___x_2520_ = lean_box(0);
v___x_2521_ = lean_nat_dec_lt(v___x_2518_, v___x_2519_);
if (v___x_2521_ == 0)
{
lean_object* v___x_2523_; 
lean_dec(v_snd_2517_);
lean_dec(v_fst_2516_);
if (v_isShared_2515_ == 0)
{
lean_ctor_set(v___x_2514_, 0, v___x_2520_);
v___x_2523_ = v___x_2514_;
goto v_reusejp_2522_;
}
else
{
lean_object* v_reuseFailAlloc_2524_; 
v_reuseFailAlloc_2524_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2524_, 0, v___x_2520_);
v___x_2523_ = v_reuseFailAlloc_2524_;
goto v_reusejp_2522_;
}
v_reusejp_2522_:
{
return v___x_2523_;
}
}
else
{
uint8_t v___x_2525_; 
v___x_2525_ = lean_nat_dec_le(v___x_2519_, v___x_2519_);
if (v___x_2525_ == 0)
{
if (v___x_2521_ == 0)
{
lean_object* v___x_2527_; 
lean_dec(v_snd_2517_);
lean_dec(v_fst_2516_);
if (v_isShared_2515_ == 0)
{
lean_ctor_set(v___x_2514_, 0, v___x_2520_);
v___x_2527_ = v___x_2514_;
goto v_reusejp_2526_;
}
else
{
lean_object* v_reuseFailAlloc_2528_; 
v_reuseFailAlloc_2528_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2528_, 0, v___x_2520_);
v___x_2527_ = v_reuseFailAlloc_2528_;
goto v_reusejp_2526_;
}
v_reusejp_2526_:
{
return v___x_2527_;
}
}
else
{
size_t v___x_2529_; size_t v___x_2530_; lean_object* v___x_2531_; 
lean_del_object(v___x_2514_);
v___x_2529_ = ((size_t)0ULL);
v___x_2530_ = lean_usize_of_nat(v___x_2519_);
v___x_2531_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofRapp_spec__2(v_snd_2517_, v_fst_2516_, v___x_2529_, v___x_2530_, v___x_2520_, v_a_2506_, v_a_2507_, v_a_2508_, v_a_2509_);
lean_dec(v_fst_2516_);
return v___x_2531_;
}
}
else
{
size_t v___x_2532_; size_t v___x_2533_; lean_object* v___x_2534_; 
lean_del_object(v___x_2514_);
v___x_2532_ = ((size_t)0ULL);
v___x_2533_ = lean_usize_of_nat(v___x_2519_);
v___x_2534_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofRapp_spec__2(v_snd_2517_, v_fst_2516_, v___x_2532_, v___x_2533_, v___x_2520_, v_a_2506_, v_a_2507_, v_a_2508_, v_a_2509_);
lean_dec(v_fst_2516_);
return v___x_2534_;
}
}
}
}
else
{
lean_object* v_a_2536_; lean_object* v___x_2538_; uint8_t v_isShared_2539_; uint8_t v_isSharedCheck_2543_; 
v_a_2536_ = lean_ctor_get(v___x_2511_, 0);
v_isSharedCheck_2543_ = !lean_is_exclusive(v___x_2511_);
if (v_isSharedCheck_2543_ == 0)
{
v___x_2538_ = v___x_2511_;
v_isShared_2539_ = v_isSharedCheck_2543_;
goto v_resetjp_2537_;
}
else
{
lean_inc(v_a_2536_);
lean_dec(v___x_2511_);
v___x_2538_ = lean_box(0);
v_isShared_2539_ = v_isSharedCheck_2543_;
goto v_resetjp_2537_;
}
v_resetjp_2537_:
{
lean_object* v___x_2541_; 
if (v_isShared_2539_ == 0)
{
v___x_2541_ = v___x_2538_;
goto v_reusejp_2540_;
}
else
{
lean_object* v_reuseFailAlloc_2542_; 
v_reuseFailAlloc_2542_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2542_, 0, v_a_2536_);
v___x_2541_ = v_reuseFailAlloc_2542_;
goto v_reusejp_2540_;
}
v_reusejp_2540_:
{
return v___x_2541_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal(lean_object* v_parentEnv_2544_, lean_object* v_g_2545_, lean_object* v_a_2546_, lean_object* v_a_2547_, lean_object* v_a_2548_, lean_object* v_a_2549_){
_start:
{
lean_object* v___x_2566_; 
lean_inc(v_g_2545_);
v___x_2566_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal(v_parentEnv_2544_, v_g_2545_, v_a_2546_, v_a_2547_, v_a_2548_, v_a_2549_);
if (lean_obj_tag(v___x_2566_) == 0)
{
lean_object* v_a_2567_; lean_object* v___x_2569_; uint8_t v_isShared_2570_; uint8_t v_isSharedCheck_2598_; 
v_a_2567_ = lean_ctor_get(v___x_2566_, 0);
v_isSharedCheck_2598_ = !lean_is_exclusive(v___x_2566_);
if (v_isSharedCheck_2598_ == 0)
{
v___x_2569_ = v___x_2566_;
v_isShared_2570_ = v_isSharedCheck_2598_;
goto v_resetjp_2568_;
}
else
{
lean_inc(v_a_2567_);
lean_dec(v___x_2566_);
v___x_2569_ = lean_box(0);
v_isShared_2570_ = v_isSharedCheck_2598_;
goto v_resetjp_2568_;
}
v_resetjp_2568_:
{
if (lean_obj_tag(v_a_2567_) == 1)
{
lean_object* v_val_2571_; lean_object* v_snd_2572_; lean_object* v_fst_2573_; lean_object* v_fst_2574_; lean_object* v_snd_2575_; lean_object* v___x_2576_; size_t v_sz_2577_; size_t v___x_2578_; lean_object* v___x_2579_; 
lean_del_object(v___x_2569_);
v_val_2571_ = lean_ctor_get(v_a_2567_, 0);
lean_inc(v_val_2571_);
lean_dec_ref_known(v_a_2567_, 1);
v_snd_2572_ = lean_ctor_get(v_val_2571_, 1);
lean_inc(v_snd_2572_);
v_fst_2573_ = lean_ctor_get(v_val_2571_, 0);
lean_inc(v_fst_2573_);
lean_dec(v_val_2571_);
v_fst_2574_ = lean_ctor_get(v_snd_2572_, 0);
lean_inc(v_fst_2574_);
v_snd_2575_ = lean_ctor_get(v_snd_2572_, 1);
lean_inc(v_snd_2575_);
lean_dec(v_snd_2572_);
v___x_2576_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__6___redArg___closed__0));
v_sz_2577_ = lean_array_size(v_fst_2574_);
v___x_2578_ = ((size_t)0ULL);
v___x_2579_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal_spec__0___redArg(v_fst_2574_, v_sz_2577_, v___x_2578_, v___x_2576_);
lean_dec(v_fst_2574_);
if (lean_obj_tag(v___x_2579_) == 0)
{
lean_object* v_a_2580_; lean_object* v_fst_2581_; 
v_a_2580_ = lean_ctor_get(v___x_2579_, 0);
lean_inc(v_a_2580_);
lean_dec_ref_known(v___x_2579_, 1);
v_fst_2581_ = lean_ctor_get(v_a_2580_, 0);
lean_inc(v_fst_2581_);
lean_dec(v_a_2580_);
if (lean_obj_tag(v_fst_2581_) == 0)
{
lean_dec(v_snd_2575_);
lean_dec(v_fst_2573_);
goto v___jp_2551_;
}
else
{
lean_object* v_val_2582_; 
v_val_2582_ = lean_ctor_get(v_fst_2581_, 0);
lean_inc(v_val_2582_);
lean_dec_ref_known(v_fst_2581_, 1);
if (lean_obj_tag(v_val_2582_) == 1)
{
lean_object* v_val_2583_; lean_object* v___x_2584_; lean_object* v___x_2585_; 
lean_dec(v_g_2545_);
v_val_2583_ = lean_ctor_get(v_val_2582_, 0);
lean_inc(v_val_2583_);
lean_dec_ref_known(v_val_2582_, 1);
v___x_2584_ = lean_st_ref_get(v_val_2583_);
lean_dec(v_val_2583_);
v___x_2585_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofRapp(v_snd_2575_, v_fst_2573_, v___x_2584_, v_a_2546_, v_a_2547_, v_a_2548_, v_a_2549_);
return v___x_2585_;
}
else
{
lean_dec(v_val_2582_);
lean_dec(v_snd_2575_);
lean_dec(v_fst_2573_);
goto v___jp_2551_;
}
}
}
else
{
lean_object* v_a_2586_; lean_object* v___x_2588_; uint8_t v_isShared_2589_; uint8_t v_isSharedCheck_2593_; 
lean_dec(v_snd_2575_);
lean_dec(v_fst_2573_);
lean_dec(v_g_2545_);
v_a_2586_ = lean_ctor_get(v___x_2579_, 0);
v_isSharedCheck_2593_ = !lean_is_exclusive(v___x_2579_);
if (v_isSharedCheck_2593_ == 0)
{
v___x_2588_ = v___x_2579_;
v_isShared_2589_ = v_isSharedCheck_2593_;
goto v_resetjp_2587_;
}
else
{
lean_inc(v_a_2586_);
lean_dec(v___x_2579_);
v___x_2588_ = lean_box(0);
v_isShared_2589_ = v_isSharedCheck_2593_;
goto v_resetjp_2587_;
}
v_resetjp_2587_:
{
lean_object* v___x_2591_; 
if (v_isShared_2589_ == 0)
{
v___x_2591_ = v___x_2588_;
goto v_reusejp_2590_;
}
else
{
lean_object* v_reuseFailAlloc_2592_; 
v_reuseFailAlloc_2592_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2592_, 0, v_a_2586_);
v___x_2591_ = v_reuseFailAlloc_2592_;
goto v_reusejp_2590_;
}
v_reusejp_2590_:
{
return v___x_2591_;
}
}
}
}
else
{
lean_object* v___x_2594_; lean_object* v___x_2596_; 
lean_dec(v_a_2567_);
lean_dec(v_g_2545_);
v___x_2594_ = lean_box(0);
if (v_isShared_2570_ == 0)
{
lean_ctor_set(v___x_2569_, 0, v___x_2594_);
v___x_2596_ = v___x_2569_;
goto v_reusejp_2595_;
}
else
{
lean_object* v_reuseFailAlloc_2597_; 
v_reuseFailAlloc_2597_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2597_, 0, v___x_2594_);
v___x_2596_ = v_reuseFailAlloc_2597_;
goto v_reusejp_2595_;
}
v_reusejp_2595_:
{
return v___x_2596_;
}
}
}
}
else
{
lean_object* v_a_2599_; lean_object* v___x_2601_; uint8_t v_isShared_2602_; uint8_t v_isSharedCheck_2606_; 
lean_dec(v_g_2545_);
v_a_2599_ = lean_ctor_get(v___x_2566_, 0);
v_isSharedCheck_2606_ = !lean_is_exclusive(v___x_2566_);
if (v_isSharedCheck_2606_ == 0)
{
v___x_2601_ = v___x_2566_;
v_isShared_2602_ = v_isSharedCheck_2606_;
goto v_resetjp_2600_;
}
else
{
lean_inc(v_a_2599_);
lean_dec(v___x_2566_);
v___x_2601_ = lean_box(0);
v_isShared_2602_ = v_isSharedCheck_2606_;
goto v_resetjp_2600_;
}
v_resetjp_2600_:
{
lean_object* v___x_2604_; 
if (v_isShared_2602_ == 0)
{
v___x_2604_ = v___x_2601_;
goto v_reusejp_2603_;
}
else
{
lean_object* v_reuseFailAlloc_2605_; 
v_reuseFailAlloc_2605_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2605_, 0, v_a_2599_);
v___x_2604_ = v_reuseFailAlloc_2605_;
goto v_reusejp_2603_;
}
v_reusejp_2603_:
{
return v___x_2604_;
}
}
}
v___jp_2551_:
{
lean_object* v___x_2552_; lean_object* v_elimGoal_2553_; lean_object* v___x_2554_; lean_object* v_id_2555_; lean_object* v___x_2556_; lean_object* v___x_2557_; lean_object* v___x_2558_; lean_object* v___x_2559_; lean_object* v___x_2560_; lean_object* v___x_2561_; lean_object* v___x_2562_; lean_object* v___x_2563_; lean_object* v___x_2564_; lean_object* v___x_2565_; 
v___x_2552_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_2553_ = lean_ctor_get(v___x_2552_, 1);
lean_inc_ref(v_elimGoal_2553_);
v___x_2554_ = lean_apply_1(v_elimGoal_2553_, v_g_2545_);
v_id_2555_ = lean_ctor_get(v___x_2554_, 0);
lean_inc(v_id_2555_);
lean_dec_ref(v___x_2554_);
v___x_2556_ = lean_obj_once(&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__1, &lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__1_once, _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__1);
v___x_2557_ = lean_obj_once(&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__3, &lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__3_once, _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__3);
v___x_2558_ = l_Nat_reprFast(v_id_2555_);
v___x_2559_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2559_, 0, v___x_2558_);
v___x_2560_ = l_Lean_MessageData_ofFormat(v___x_2559_);
v___x_2561_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2561_, 0, v___x_2557_);
lean_ctor_set(v___x_2561_, 1, v___x_2560_);
v___x_2562_ = lean_obj_once(&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal___closed__1, &lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal___closed__1_once, _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal___closed__1);
v___x_2563_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2563_, 0, v___x_2561_);
lean_ctor_set(v___x_2563_, 1, v___x_2562_);
v___x_2564_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2564_, 0, v___x_2556_);
lean_ctor_set(v___x_2564_, 1, v___x_2563_);
v___x_2565_ = lp_aesop_Lean_throwError___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__17___redArg(v___x_2564_, v_a_2546_, v_a_2547_, v_a_2548_, v_a_2549_);
return v___x_2565_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster(lean_object* v_parentEnv_2607_, lean_object* v_c_2608_, lean_object* v_a_2609_, lean_object* v_a_2610_, lean_object* v_a_2611_, lean_object* v_a_2612_){
_start:
{
lean_object* v___x_2614_; lean_object* v_elimMVarCluster_2615_; lean_object* v___x_2616_; lean_object* v_goals_2617_; lean_object* v___x_2642_; size_t v_sz_2643_; size_t v___x_2644_; lean_object* v___x_2645_; 
v___x_2614_ = lp_aesop_Aesop_treeImpl;
v_elimMVarCluster_2615_ = lean_ctor_get(v___x_2614_, 5);
lean_inc_ref(v_elimMVarCluster_2615_);
v___x_2616_ = lean_apply_1(v_elimMVarCluster_2615_, v_c_2608_);
v_goals_2617_ = lean_ctor_get(v___x_2616_, 1);
lean_inc_ref(v_goals_2617_);
lean_dec_ref(v___x_2616_);
v___x_2642_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__6___redArg___closed__0));
v_sz_2643_ = lean_array_size(v_goals_2617_);
v___x_2644_ = ((size_t)0ULL);
v___x_2645_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__6___redArg(v_goals_2617_, v_sz_2643_, v___x_2644_, v___x_2642_);
if (lean_obj_tag(v___x_2645_) == 0)
{
lean_object* v_a_2646_; lean_object* v_fst_2647_; 
v_a_2646_ = lean_ctor_get(v___x_2645_, 0);
lean_inc(v_a_2646_);
lean_dec_ref_known(v___x_2645_, 1);
v_fst_2647_ = lean_ctor_get(v_a_2646_, 0);
lean_inc(v_fst_2647_);
lean_dec(v_a_2646_);
if (lean_obj_tag(v_fst_2647_) == 0)
{
lean_dec_ref(v_parentEnv_2607_);
goto v___jp_2618_;
}
else
{
lean_object* v_val_2648_; 
v_val_2648_ = lean_ctor_get(v_fst_2647_, 0);
lean_inc(v_val_2648_);
lean_dec_ref_known(v_fst_2647_, 1);
if (lean_obj_tag(v_val_2648_) == 1)
{
lean_object* v_val_2649_; lean_object* v___x_2650_; lean_object* v___x_2651_; 
lean_dec_ref(v_goals_2617_);
v_val_2649_ = lean_ctor_get(v_val_2648_, 0);
lean_inc(v_val_2649_);
lean_dec_ref_known(v_val_2648_, 1);
v___x_2650_ = lean_st_ref_get(v_val_2649_);
lean_dec(v_val_2649_);
v___x_2651_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal(v_parentEnv_2607_, v___x_2650_, v_a_2609_, v_a_2610_, v_a_2611_, v_a_2612_);
return v___x_2651_;
}
else
{
lean_dec(v_val_2648_);
lean_dec_ref(v_parentEnv_2607_);
goto v___jp_2618_;
}
}
}
else
{
lean_object* v_a_2652_; lean_object* v___x_2654_; uint8_t v_isShared_2655_; uint8_t v_isSharedCheck_2659_; 
lean_dec_ref(v_goals_2617_);
lean_dec_ref(v_parentEnv_2607_);
v_a_2652_ = lean_ctor_get(v___x_2645_, 0);
v_isSharedCheck_2659_ = !lean_is_exclusive(v___x_2645_);
if (v_isSharedCheck_2659_ == 0)
{
v___x_2654_ = v___x_2645_;
v_isShared_2655_ = v_isSharedCheck_2659_;
goto v_resetjp_2653_;
}
else
{
lean_inc(v_a_2652_);
lean_dec(v___x_2645_);
v___x_2654_ = lean_box(0);
v_isShared_2655_ = v_isSharedCheck_2659_;
goto v_resetjp_2653_;
}
v_resetjp_2653_:
{
lean_object* v___x_2657_; 
if (v_isShared_2655_ == 0)
{
v___x_2657_ = v___x_2654_;
goto v_reusejp_2656_;
}
else
{
lean_object* v_reuseFailAlloc_2658_; 
v_reuseFailAlloc_2658_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2658_, 0, v_a_2652_);
v___x_2657_ = v_reuseFailAlloc_2658_;
goto v_reusejp_2656_;
}
v_reusejp_2656_:
{
return v___x_2657_;
}
}
}
v___jp_2618_:
{
size_t v_sz_2619_; size_t v___x_2620_; lean_object* v___x_2621_; 
v_sz_2619_ = lean_array_size(v_goals_2617_);
v___x_2620_ = ((size_t)0ULL);
v___x_2621_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__4___redArg(v_sz_2619_, v___x_2620_, v_goals_2617_);
if (lean_obj_tag(v___x_2621_) == 0)
{
lean_object* v_a_2622_; lean_object* v___x_2623_; lean_object* v___x_2624_; lean_object* v___x_2625_; lean_object* v___x_2626_; lean_object* v___x_2627_; lean_object* v___x_2628_; lean_object* v___x_2629_; lean_object* v___x_2630_; lean_object* v___x_2631_; lean_object* v___x_2632_; lean_object* v___x_2633_; 
v_a_2622_ = lean_ctor_get(v___x_2621_, 0);
lean_inc(v_a_2622_);
lean_dec_ref_known(v___x_2621_, 1);
v___x_2623_ = lean_obj_once(&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__1, &lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__1_once, _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal___closed__1);
v___x_2624_ = lean_obj_once(&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster___closed__1, &lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster___closed__1_once, _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster___closed__1);
v___x_2625_ = lean_array_to_list(v_a_2622_);
v___x_2626_ = lean_box(0);
v___x_2627_ = lp_aesop_List_mapTR_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__5(v___x_2625_, v___x_2626_);
v___x_2628_ = l_Lean_MessageData_ofList(v___x_2627_);
v___x_2629_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2629_, 0, v___x_2624_);
lean_ctor_set(v___x_2629_, 1, v___x_2628_);
v___x_2630_ = lean_obj_once(&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster___closed__3, &lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster___closed__3_once, _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster___closed__3);
v___x_2631_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2631_, 0, v___x_2629_);
lean_ctor_set(v___x_2631_, 1, v___x_2630_);
v___x_2632_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2632_, 0, v___x_2623_);
lean_ctor_set(v___x_2632_, 1, v___x_2631_);
v___x_2633_ = lp_aesop_Lean_throwError___at___00Lean_MetavarContext_getExprMVarDecl___at___00Lean_MVarId_instantiateMVars___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__11_spec__14_spec__17___redArg(v___x_2632_, v_a_2609_, v_a_2610_, v_a_2611_, v_a_2612_);
return v___x_2633_;
}
else
{
lean_object* v_a_2634_; lean_object* v___x_2636_; uint8_t v_isShared_2637_; uint8_t v_isSharedCheck_2641_; 
v_a_2634_ = lean_ctor_get(v___x_2621_, 0);
v_isSharedCheck_2641_ = !lean_is_exclusive(v___x_2621_);
if (v_isSharedCheck_2641_ == 0)
{
v___x_2636_ = v___x_2621_;
v_isShared_2637_ = v_isSharedCheck_2641_;
goto v_resetjp_2635_;
}
else
{
lean_inc(v_a_2634_);
lean_dec(v___x_2621_);
v___x_2636_ = lean_box(0);
v_isShared_2637_ = v_isSharedCheck_2641_;
goto v_resetjp_2635_;
}
v_resetjp_2635_:
{
lean_object* v___x_2639_; 
if (v_isShared_2637_ == 0)
{
v___x_2639_ = v___x_2636_;
goto v_reusejp_2638_;
}
else
{
lean_object* v_reuseFailAlloc_2640_; 
v_reuseFailAlloc_2640_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2640_, 0, v_a_2634_);
v___x_2639_ = v_reuseFailAlloc_2640_;
goto v_reusejp_2638_;
}
v_reusejp_2638_:
{
return v___x_2639_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofRapp_spec__2(lean_object* v_snd_2660_, lean_object* v_as_2661_, size_t v_i_2662_, size_t v_stop_2663_, lean_object* v_b_2664_, lean_object* v___y_2665_, lean_object* v___y_2666_, lean_object* v___y_2667_, lean_object* v___y_2668_){
_start:
{
uint8_t v___x_2670_; 
v___x_2670_ = lean_usize_dec_eq(v_i_2662_, v_stop_2663_);
if (v___x_2670_ == 0)
{
lean_object* v___x_2671_; lean_object* v___x_2672_; lean_object* v___x_2673_; 
v___x_2671_ = lean_array_uget_borrowed(v_as_2661_, v_i_2662_);
v___x_2672_ = lean_st_ref_get(v___x_2671_);
lean_inc_ref(v_snd_2660_);
v___x_2673_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster(v_snd_2660_, v___x_2672_, v___y_2665_, v___y_2666_, v___y_2667_, v___y_2668_);
if (lean_obj_tag(v___x_2673_) == 0)
{
lean_object* v_a_2674_; size_t v___x_2675_; size_t v___x_2676_; 
v_a_2674_ = lean_ctor_get(v___x_2673_, 0);
lean_inc(v_a_2674_);
lean_dec_ref_known(v___x_2673_, 1);
v___x_2675_ = ((size_t)1ULL);
v___x_2676_ = lean_usize_add(v_i_2662_, v___x_2675_);
v_i_2662_ = v___x_2676_;
v_b_2664_ = v_a_2674_;
goto _start;
}
else
{
lean_dec_ref(v_snd_2660_);
return v___x_2673_;
}
}
else
{
lean_object* v___x_2678_; 
lean_dec_ref(v_snd_2660_);
v___x_2678_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2678_, 0, v_b_2664_);
return v___x_2678_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofRapp_spec__2___boxed(lean_object* v_snd_2679_, lean_object* v_as_2680_, lean_object* v_i_2681_, lean_object* v_stop_2682_, lean_object* v_b_2683_, lean_object* v___y_2684_, lean_object* v___y_2685_, lean_object* v___y_2686_, lean_object* v___y_2687_, lean_object* v___y_2688_){
_start:
{
size_t v_i_boxed_2689_; size_t v_stop_boxed_2690_; lean_object* v_res_2691_; 
v_i_boxed_2689_ = lean_unbox_usize(v_i_2681_);
lean_dec(v_i_2681_);
v_stop_boxed_2690_ = lean_unbox_usize(v_stop_2682_);
lean_dec(v_stop_2682_);
v_res_2691_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofRapp_spec__2(v_snd_2679_, v_as_2680_, v_i_boxed_2689_, v_stop_boxed_2690_, v_b_2683_, v___y_2684_, v___y_2685_, v___y_2686_, v___y_2687_);
lean_dec(v___y_2687_);
lean_dec_ref(v___y_2686_);
lean_dec(v___y_2685_);
lean_dec_ref(v___y_2684_);
lean_dec_ref(v_as_2680_);
return v_res_2691_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofRapp___boxed(lean_object* v_parentEnv_2692_, lean_object* v_parentGoal_2693_, lean_object* v_r_2694_, lean_object* v_a_2695_, lean_object* v_a_2696_, lean_object* v_a_2697_, lean_object* v_a_2698_, lean_object* v_a_2699_){
_start:
{
lean_object* v_res_2700_; 
v_res_2700_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofRapp(v_parentEnv_2692_, v_parentGoal_2693_, v_r_2694_, v_a_2695_, v_a_2696_, v_a_2697_, v_a_2698_);
lean_dec(v_a_2698_);
lean_dec_ref(v_a_2697_);
lean_dec(v_a_2696_);
lean_dec_ref(v_a_2695_);
return v_res_2700_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster___boxed(lean_object* v_parentEnv_2701_, lean_object* v_c_2702_, lean_object* v_a_2703_, lean_object* v_a_2704_, lean_object* v_a_2705_, lean_object* v_a_2706_, lean_object* v_a_2707_){
_start:
{
lean_object* v_res_2708_; 
v_res_2708_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster(v_parentEnv_2701_, v_c_2702_, v_a_2703_, v_a_2704_, v_a_2705_, v_a_2706_);
lean_dec(v_a_2706_);
lean_dec_ref(v_a_2705_);
lean_dec(v_a_2704_);
lean_dec_ref(v_a_2703_);
return v_res_2708_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal___boxed(lean_object* v_parentEnv_2709_, lean_object* v_g_2710_, lean_object* v_a_2711_, lean_object* v_a_2712_, lean_object* v_a_2713_, lean_object* v_a_2714_, lean_object* v_a_2715_){
_start:
{
lean_object* v_res_2716_; 
v_res_2716_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal(v_parentEnv_2709_, v_g_2710_, v_a_2711_, v_a_2712_, v_a_2713_, v_a_2714_);
lean_dec(v_a_2714_);
lean_dec_ref(v_a_2713_);
lean_dec(v_a_2712_);
lean_dec_ref(v_a_2711_);
return v_res_2716_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal_spec__0(lean_object* v_as_2717_, size_t v_sz_2718_, size_t v_i_2719_, lean_object* v_b_2720_, lean_object* v___y_2721_, lean_object* v___y_2722_, lean_object* v___y_2723_, lean_object* v___y_2724_){
_start:
{
lean_object* v___x_2726_; 
v___x_2726_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal_spec__0___redArg(v_as_2717_, v_sz_2718_, v_i_2719_, v_b_2720_);
return v___x_2726_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal_spec__0___boxed(lean_object* v_as_2727_, lean_object* v_sz_2728_, lean_object* v_i_2729_, lean_object* v_b_2730_, lean_object* v___y_2731_, lean_object* v___y_2732_, lean_object* v___y_2733_, lean_object* v___y_2734_, lean_object* v___y_2735_){
_start:
{
size_t v_sz_boxed_2736_; size_t v_i_boxed_2737_; lean_object* v_res_2738_; 
v_sz_boxed_2736_ = lean_unbox_usize(v_sz_2728_);
lean_dec(v_sz_2728_);
v_i_boxed_2737_ = lean_unbox_usize(v_i_2729_);
lean_dec(v_i_2729_);
v_res_2738_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal_spec__0(v_as_2727_, v_sz_boxed_2736_, v_i_boxed_2737_, v_b_2730_, v___y_2731_, v___y_2732_, v___y_2733_, v___y_2734_);
lean_dec(v___y_2734_);
lean_dec_ref(v___y_2733_);
lean_dec(v___y_2732_);
lean_dec_ref(v___y_2731_);
lean_dec_ref(v_as_2727_);
return v_res_2738_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__4(size_t v_sz_2739_, size_t v_i_2740_, lean_object* v_bs_2741_, lean_object* v___y_2742_, lean_object* v___y_2743_, lean_object* v___y_2744_, lean_object* v___y_2745_){
_start:
{
lean_object* v___x_2747_; 
v___x_2747_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__4___redArg(v_sz_2739_, v_i_2740_, v_bs_2741_);
return v___x_2747_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__4___boxed(lean_object* v_sz_2748_, lean_object* v_i_2749_, lean_object* v_bs_2750_, lean_object* v___y_2751_, lean_object* v___y_2752_, lean_object* v___y_2753_, lean_object* v___y_2754_, lean_object* v___y_2755_){
_start:
{
size_t v_sz_boxed_2756_; size_t v_i_boxed_2757_; lean_object* v_res_2758_; 
v_sz_boxed_2756_ = lean_unbox_usize(v_sz_2748_);
lean_dec(v_sz_2748_);
v_i_boxed_2757_ = lean_unbox_usize(v_i_2749_);
lean_dec(v_i_2749_);
v_res_2758_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__4(v_sz_boxed_2756_, v_i_boxed_2757_, v_bs_2750_, v___y_2751_, v___y_2752_, v___y_2753_, v___y_2754_);
lean_dec(v___y_2754_);
lean_dec_ref(v___y_2753_);
lean_dec(v___y_2752_);
lean_dec_ref(v___y_2751_);
return v_res_2758_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__6(lean_object* v_as_2759_, size_t v_sz_2760_, size_t v_i_2761_, lean_object* v_b_2762_, lean_object* v___y_2763_, lean_object* v___y_2764_, lean_object* v___y_2765_, lean_object* v___y_2766_){
_start:
{
lean_object* v___x_2768_; 
v___x_2768_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__6___redArg(v_as_2759_, v_sz_2760_, v_i_2761_, v_b_2762_);
return v___x_2768_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__6___boxed(lean_object* v_as_2769_, lean_object* v_sz_2770_, lean_object* v_i_2771_, lean_object* v_b_2772_, lean_object* v___y_2773_, lean_object* v___y_2774_, lean_object* v___y_2775_, lean_object* v___y_2776_, lean_object* v___y_2777_){
_start:
{
size_t v_sz_boxed_2778_; size_t v_i_boxed_2779_; lean_object* v_res_2780_; 
v_sz_boxed_2778_ = lean_unbox_usize(v_sz_2770_);
lean_dec(v_sz_2770_);
v_i_boxed_2779_ = lean_unbox_usize(v_i_2771_);
lean_dec(v_i_2771_);
v_res_2780_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractProofMVarCluster_spec__6(v_as_2769_, v_sz_boxed_2778_, v_i_boxed_2779_, v_b_2772_, v___y_2773_, v___y_2774_, v___y_2775_, v___y_2776_);
lean_dec(v___y_2776_);
lean_dec_ref(v___y_2775_);
lean_dec(v___y_2774_);
lean_dec_ref(v___y_2773_);
lean_dec_ref(v_as_2769_);
return v_res_2780_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal_spec__0___redArg(lean_object* v_msg_2781_, lean_object* v___y_2782_, lean_object* v___y_2783_, lean_object* v___y_2784_, lean_object* v___y_2785_){
_start:
{
lean_object* v_ref_2787_; lean_object* v___x_2788_; lean_object* v_a_2789_; lean_object* v___x_2791_; uint8_t v_isShared_2792_; uint8_t v_isSharedCheck_2797_; 
v_ref_2787_ = lean_ctor_get(v___y_2784_, 5);
v___x_2788_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_copyExprMVar_spec__6_spec__7(v_msg_2781_, v___y_2782_, v___y_2783_, v___y_2784_, v___y_2785_);
v_a_2789_ = lean_ctor_get(v___x_2788_, 0);
v_isSharedCheck_2797_ = !lean_is_exclusive(v___x_2788_);
if (v_isSharedCheck_2797_ == 0)
{
v___x_2791_ = v___x_2788_;
v_isShared_2792_ = v_isSharedCheck_2797_;
goto v_resetjp_2790_;
}
else
{
lean_inc(v_a_2789_);
lean_dec(v___x_2788_);
v___x_2791_ = lean_box(0);
v_isShared_2792_ = v_isSharedCheck_2797_;
goto v_resetjp_2790_;
}
v_resetjp_2790_:
{
lean_object* v___x_2793_; lean_object* v___x_2795_; 
lean_inc(v_ref_2787_);
v___x_2793_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2793_, 0, v_ref_2787_);
lean_ctor_set(v___x_2793_, 1, v_a_2789_);
if (v_isShared_2792_ == 0)
{
lean_ctor_set_tag(v___x_2791_, 1);
lean_ctor_set(v___x_2791_, 0, v___x_2793_);
v___x_2795_ = v___x_2791_;
goto v_reusejp_2794_;
}
else
{
lean_object* v_reuseFailAlloc_2796_; 
v_reuseFailAlloc_2796_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2796_, 0, v___x_2793_);
v___x_2795_ = v_reuseFailAlloc_2796_;
goto v_reusejp_2794_;
}
v_reusejp_2794_:
{
return v___x_2795_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal_spec__0___redArg___boxed(lean_object* v_msg_2798_, lean_object* v___y_2799_, lean_object* v___y_2800_, lean_object* v___y_2801_, lean_object* v___y_2802_, lean_object* v___y_2803_){
_start:
{
lean_object* v_res_2804_; 
v_res_2804_ = lp_aesop_Lean_throwError___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal_spec__0___redArg(v_msg_2798_, v___y_2799_, v___y_2800_, v___y_2801_, v___y_2802_);
lean_dec(v___y_2802_);
lean_dec_ref(v___y_2801_);
lean_dec(v___y_2800_);
lean_dec_ref(v___y_2799_);
return v_res_2804_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixMVarCluster(lean_object* v_parentEnv_2805_, lean_object* v_c_2806_, lean_object* v_a_2807_, lean_object* v_a_2808_, lean_object* v_a_2809_, lean_object* v_a_2810_, lean_object* v_a_2811_){
_start:
{
lean_object* v___x_2813_; lean_object* v_elimMVarCluster_2814_; lean_object* v___x_2815_; lean_object* v_goals_2816_; lean_object* v___x_2817_; lean_object* v___x_2818_; lean_object* v___x_2819_; uint8_t v___x_2820_; 
v___x_2813_ = lp_aesop_Aesop_treeImpl;
v_elimMVarCluster_2814_ = lean_ctor_get(v___x_2813_, 5);
lean_inc_ref(v_elimMVarCluster_2814_);
v___x_2815_ = lean_apply_1(v_elimMVarCluster_2814_, v_c_2806_);
v_goals_2816_ = lean_ctor_get(v___x_2815_, 1);
lean_inc_ref(v_goals_2816_);
lean_dec_ref(v___x_2815_);
v___x_2817_ = lean_unsigned_to_nat(0u);
v___x_2818_ = lean_array_get_size(v_goals_2816_);
v___x_2819_ = lean_box(0);
v___x_2820_ = lean_nat_dec_lt(v___x_2817_, v___x_2818_);
if (v___x_2820_ == 0)
{
lean_object* v___x_2821_; 
lean_dec_ref(v_goals_2816_);
lean_dec_ref(v_parentEnv_2805_);
v___x_2821_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2821_, 0, v___x_2819_);
return v___x_2821_;
}
else
{
uint8_t v___x_2822_; 
v___x_2822_ = lean_nat_dec_le(v___x_2818_, v___x_2818_);
if (v___x_2822_ == 0)
{
if (v___x_2820_ == 0)
{
lean_object* v___x_2823_; 
lean_dec_ref(v_goals_2816_);
lean_dec_ref(v_parentEnv_2805_);
v___x_2823_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2823_, 0, v___x_2819_);
return v___x_2823_;
}
else
{
size_t v___x_2824_; size_t v___x_2825_; lean_object* v___x_2826_; 
v___x_2824_ = ((size_t)0ULL);
v___x_2825_ = lean_usize_of_nat(v___x_2818_);
v___x_2826_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixMVarCluster_spec__4(v_parentEnv_2805_, v_goals_2816_, v___x_2824_, v___x_2825_, v___x_2819_, v_a_2807_, v_a_2808_, v_a_2809_, v_a_2810_, v_a_2811_);
lean_dec_ref(v_goals_2816_);
return v___x_2826_;
}
}
else
{
size_t v___x_2827_; size_t v___x_2828_; lean_object* v___x_2829_; 
v___x_2827_ = ((size_t)0ULL);
v___x_2828_ = lean_usize_of_nat(v___x_2818_);
v___x_2829_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixMVarCluster_spec__4(v_parentEnv_2805_, v_goals_2816_, v___x_2827_, v___x_2828_, v___x_2819_, v_a_2807_, v_a_2808_, v_a_2809_, v_a_2810_, v_a_2811_);
lean_dec_ref(v_goals_2816_);
return v___x_2829_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixRapp_spec__2(lean_object* v_snd_2830_, lean_object* v_as_2831_, size_t v_i_2832_, size_t v_stop_2833_, lean_object* v_b_2834_, lean_object* v___y_2835_, lean_object* v___y_2836_, lean_object* v___y_2837_, lean_object* v___y_2838_, lean_object* v___y_2839_){
_start:
{
uint8_t v___x_2841_; 
v___x_2841_ = lean_usize_dec_eq(v_i_2832_, v_stop_2833_);
if (v___x_2841_ == 0)
{
lean_object* v___x_2842_; lean_object* v___x_2843_; lean_object* v___x_2844_; 
v___x_2842_ = lean_array_uget_borrowed(v_as_2831_, v_i_2832_);
v___x_2843_ = lean_st_ref_get(v___x_2842_);
lean_inc_ref(v_snd_2830_);
v___x_2844_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixMVarCluster(v_snd_2830_, v___x_2843_, v___y_2835_, v___y_2836_, v___y_2837_, v___y_2838_, v___y_2839_);
if (lean_obj_tag(v___x_2844_) == 0)
{
lean_object* v_a_2845_; size_t v___x_2846_; size_t v___x_2847_; 
v_a_2845_ = lean_ctor_get(v___x_2844_, 0);
lean_inc(v_a_2845_);
lean_dec_ref_known(v___x_2844_, 1);
v___x_2846_ = ((size_t)1ULL);
v___x_2847_ = lean_usize_add(v_i_2832_, v___x_2846_);
v_i_2832_ = v___x_2847_;
v_b_2834_ = v_a_2845_;
goto _start;
}
else
{
lean_dec_ref(v_snd_2830_);
return v___x_2844_;
}
}
else
{
lean_object* v___x_2849_; 
lean_dec_ref(v_snd_2830_);
v___x_2849_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2849_, 0, v_b_2834_);
return v___x_2849_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixRapp(lean_object* v_parentEnv_2850_, lean_object* v_parentGoal_2851_, lean_object* v_r_2852_, lean_object* v_a_2853_, lean_object* v_a_2854_, lean_object* v_a_2855_, lean_object* v_a_2856_, lean_object* v_a_2857_){
_start:
{
lean_object* v___x_2859_; 
v___x_2859_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitRapp(v_parentEnv_2850_, v_parentGoal_2851_, v_r_2852_, v_a_2854_, v_a_2855_, v_a_2856_, v_a_2857_);
if (lean_obj_tag(v___x_2859_) == 0)
{
lean_object* v_a_2860_; lean_object* v___x_2862_; uint8_t v_isShared_2863_; uint8_t v_isSharedCheck_2883_; 
v_a_2860_ = lean_ctor_get(v___x_2859_, 0);
v_isSharedCheck_2883_ = !lean_is_exclusive(v___x_2859_);
if (v_isSharedCheck_2883_ == 0)
{
v___x_2862_ = v___x_2859_;
v_isShared_2863_ = v_isSharedCheck_2883_;
goto v_resetjp_2861_;
}
else
{
lean_inc(v_a_2860_);
lean_dec(v___x_2859_);
v___x_2862_ = lean_box(0);
v_isShared_2863_ = v_isSharedCheck_2883_;
goto v_resetjp_2861_;
}
v_resetjp_2861_:
{
lean_object* v_fst_2864_; lean_object* v_snd_2865_; lean_object* v___x_2866_; lean_object* v___x_2867_; lean_object* v___x_2868_; uint8_t v___x_2869_; 
v_fst_2864_ = lean_ctor_get(v_a_2860_, 0);
lean_inc(v_fst_2864_);
v_snd_2865_ = lean_ctor_get(v_a_2860_, 1);
lean_inc(v_snd_2865_);
lean_dec(v_a_2860_);
v___x_2866_ = lean_unsigned_to_nat(0u);
v___x_2867_ = lean_array_get_size(v_fst_2864_);
v___x_2868_ = lean_box(0);
v___x_2869_ = lean_nat_dec_lt(v___x_2866_, v___x_2867_);
if (v___x_2869_ == 0)
{
lean_object* v___x_2871_; 
lean_dec(v_snd_2865_);
lean_dec(v_fst_2864_);
if (v_isShared_2863_ == 0)
{
lean_ctor_set(v___x_2862_, 0, v___x_2868_);
v___x_2871_ = v___x_2862_;
goto v_reusejp_2870_;
}
else
{
lean_object* v_reuseFailAlloc_2872_; 
v_reuseFailAlloc_2872_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2872_, 0, v___x_2868_);
v___x_2871_ = v_reuseFailAlloc_2872_;
goto v_reusejp_2870_;
}
v_reusejp_2870_:
{
return v___x_2871_;
}
}
else
{
uint8_t v___x_2873_; 
v___x_2873_ = lean_nat_dec_le(v___x_2867_, v___x_2867_);
if (v___x_2873_ == 0)
{
if (v___x_2869_ == 0)
{
lean_object* v___x_2875_; 
lean_dec(v_snd_2865_);
lean_dec(v_fst_2864_);
if (v_isShared_2863_ == 0)
{
lean_ctor_set(v___x_2862_, 0, v___x_2868_);
v___x_2875_ = v___x_2862_;
goto v_reusejp_2874_;
}
else
{
lean_object* v_reuseFailAlloc_2876_; 
v_reuseFailAlloc_2876_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2876_, 0, v___x_2868_);
v___x_2875_ = v_reuseFailAlloc_2876_;
goto v_reusejp_2874_;
}
v_reusejp_2874_:
{
return v___x_2875_;
}
}
else
{
size_t v___x_2877_; size_t v___x_2878_; lean_object* v___x_2879_; 
lean_del_object(v___x_2862_);
v___x_2877_ = ((size_t)0ULL);
v___x_2878_ = lean_usize_of_nat(v___x_2867_);
v___x_2879_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixRapp_spec__2(v_snd_2865_, v_fst_2864_, v___x_2877_, v___x_2878_, v___x_2868_, v_a_2853_, v_a_2854_, v_a_2855_, v_a_2856_, v_a_2857_);
lean_dec(v_fst_2864_);
return v___x_2879_;
}
}
else
{
size_t v___x_2880_; size_t v___x_2881_; lean_object* v___x_2882_; 
lean_del_object(v___x_2862_);
v___x_2880_ = ((size_t)0ULL);
v___x_2881_ = lean_usize_of_nat(v___x_2867_);
v___x_2882_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixRapp_spec__2(v_snd_2865_, v_fst_2864_, v___x_2880_, v___x_2881_, v___x_2868_, v_a_2853_, v_a_2854_, v_a_2855_, v_a_2856_, v_a_2857_);
lean_dec(v_fst_2864_);
return v___x_2882_;
}
}
}
}
else
{
lean_object* v_a_2884_; lean_object* v___x_2886_; uint8_t v_isShared_2887_; uint8_t v_isSharedCheck_2891_; 
v_a_2884_ = lean_ctor_get(v___x_2859_, 0);
v_isSharedCheck_2891_ = !lean_is_exclusive(v___x_2859_);
if (v_isSharedCheck_2891_ == 0)
{
v___x_2886_ = v___x_2859_;
v_isShared_2887_ = v_isSharedCheck_2891_;
goto v_resetjp_2885_;
}
else
{
lean_inc(v_a_2884_);
lean_dec(v___x_2859_);
v___x_2886_ = lean_box(0);
v_isShared_2887_ = v_isSharedCheck_2891_;
goto v_resetjp_2885_;
}
v_resetjp_2885_:
{
lean_object* v___x_2889_; 
if (v_isShared_2887_ == 0)
{
v___x_2889_ = v___x_2886_;
goto v_reusejp_2888_;
}
else
{
lean_object* v_reuseFailAlloc_2890_; 
v_reuseFailAlloc_2890_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2890_, 0, v_a_2884_);
v___x_2889_ = v_reuseFailAlloc_2890_;
goto v_reusejp_2888_;
}
v_reusejp_2888_:
{
return v___x_2889_;
}
}
}
}
}
static lean_object* _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal___closed__1(void){
_start:
{
lean_object* v___x_2893_; lean_object* v___x_2894_; 
v___x_2893_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal___closed__0));
v___x_2894_ = l_Lean_stringToMessageData(v___x_2893_);
return v___x_2894_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal___closed__3(void){
_start:
{
lean_object* v___x_2896_; lean_object* v___x_2897_; 
v___x_2896_ = ((lean_object*)(lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal___closed__2));
v___x_2897_ = l_Lean_stringToMessageData(v___x_2896_);
return v___x_2897_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal(lean_object* v_parentEnv_2898_, lean_object* v_g_2899_, lean_object* v_a_2900_, lean_object* v_a_2901_, lean_object* v_a_2902_, lean_object* v_a_2903_, lean_object* v_a_2904_){
_start:
{
lean_object* v___x_2906_; 
lean_inc(v_g_2899_);
v___x_2906_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_visitGoal(v_parentEnv_2898_, v_g_2899_, v_a_2901_, v_a_2902_, v_a_2903_, v_a_2904_);
if (lean_obj_tag(v___x_2906_) == 0)
{
lean_object* v_a_2907_; lean_object* v___x_2909_; uint8_t v_isShared_2910_; uint8_t v_isSharedCheck_2973_; 
v_a_2907_ = lean_ctor_get(v___x_2906_, 0);
v_isSharedCheck_2973_ = !lean_is_exclusive(v___x_2906_);
if (v_isSharedCheck_2973_ == 0)
{
v___x_2909_ = v___x_2906_;
v_isShared_2910_ = v_isSharedCheck_2973_;
goto v_resetjp_2908_;
}
else
{
lean_inc(v_a_2907_);
lean_dec(v___x_2906_);
v___x_2909_ = lean_box(0);
v_isShared_2910_ = v_isSharedCheck_2973_;
goto v_resetjp_2908_;
}
v_resetjp_2908_:
{
if (lean_obj_tag(v_a_2907_) == 1)
{
lean_object* v_val_2911_; lean_object* v___x_2913_; uint8_t v_isShared_2914_; uint8_t v_isSharedCheck_2968_; 
v_val_2911_ = lean_ctor_get(v_a_2907_, 0);
v_isSharedCheck_2968_ = !lean_is_exclusive(v_a_2907_);
if (v_isSharedCheck_2968_ == 0)
{
v___x_2913_ = v_a_2907_;
v_isShared_2914_ = v_isSharedCheck_2968_;
goto v_resetjp_2912_;
}
else
{
lean_inc(v_val_2911_);
lean_dec(v_a_2907_);
v___x_2913_ = lean_box(0);
v_isShared_2914_ = v_isSharedCheck_2968_;
goto v_resetjp_2912_;
}
v_resetjp_2912_:
{
lean_object* v_snd_2915_; lean_object* v_fst_2916_; lean_object* v___x_2918_; uint8_t v_isShared_2919_; uint8_t v_isSharedCheck_2967_; 
v_snd_2915_ = lean_ctor_get(v_val_2911_, 1);
v_fst_2916_ = lean_ctor_get(v_val_2911_, 0);
v_isSharedCheck_2967_ = !lean_is_exclusive(v_val_2911_);
if (v_isSharedCheck_2967_ == 0)
{
v___x_2918_ = v_val_2911_;
v_isShared_2919_ = v_isSharedCheck_2967_;
goto v_resetjp_2917_;
}
else
{
lean_inc(v_snd_2915_);
lean_inc(v_fst_2916_);
lean_dec(v_val_2911_);
v___x_2918_ = lean_box(0);
v_isShared_2919_ = v_isSharedCheck_2967_;
goto v_resetjp_2917_;
}
v_resetjp_2917_:
{
lean_object* v_snd_2920_; lean_object* v___x_2922_; uint8_t v_isShared_2923_; uint8_t v_isSharedCheck_2965_; 
v_snd_2920_ = lean_ctor_get(v_snd_2915_, 1);
v_isSharedCheck_2965_ = !lean_is_exclusive(v_snd_2915_);
if (v_isSharedCheck_2965_ == 0)
{
lean_object* v_unused_2966_; 
v_unused_2966_ = lean_ctor_get(v_snd_2915_, 0);
lean_dec(v_unused_2966_);
v___x_2922_ = v_snd_2915_;
v_isShared_2923_ = v_isSharedCheck_2965_;
goto v_resetjp_2921_;
}
else
{
lean_inc(v_snd_2920_);
lean_dec(v_snd_2915_);
v___x_2922_ = lean_box(0);
v_isShared_2923_ = v_isSharedCheck_2965_;
goto v_resetjp_2921_;
}
v_resetjp_2921_:
{
lean_object* v___x_2924_; lean_object* v___y_2926_; lean_object* v___y_2927_; lean_object* v___y_2928_; lean_object* v___y_2929_; lean_object* v___y_2930_; lean_object* v___x_2944_; lean_object* v___x_2945_; uint8_t v___x_2946_; 
lean_inc(v_g_2899_);
v___x_2924_ = lp_aesop_Aesop_Goal_safeRapps(v_g_2899_);
v___x_2944_ = lean_unsigned_to_nat(1u);
v___x_2945_ = lean_array_get_size(v___x_2924_);
v___x_2946_ = lean_nat_dec_lt(v___x_2944_, v___x_2945_);
if (v___x_2946_ == 0)
{
lean_del_object(v___x_2922_);
lean_del_object(v___x_2918_);
lean_del_object(v___x_2913_);
lean_dec(v_g_2899_);
v___y_2926_ = v_a_2900_;
v___y_2927_ = v_a_2901_;
v___y_2928_ = v_a_2902_;
v___y_2929_ = v_a_2903_;
v___y_2930_ = v_a_2904_;
goto v___jp_2925_;
}
else
{
lean_object* v___x_2947_; lean_object* v_elimGoal_2948_; lean_object* v___x_2949_; lean_object* v_id_2950_; lean_object* v___x_2951_; lean_object* v___x_2952_; lean_object* v___x_2954_; 
v___x_2947_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_2948_ = lean_ctor_get(v___x_2947_, 1);
lean_inc_ref(v_elimGoal_2948_);
v___x_2949_ = lean_apply_1(v_elimGoal_2948_, v_g_2899_);
v_id_2950_ = lean_ctor_get(v___x_2949_, 0);
lean_inc(v_id_2950_);
lean_dec_ref(v___x_2949_);
v___x_2951_ = lean_obj_once(&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal___closed__1, &lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal___closed__1_once, _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal___closed__1);
v___x_2952_ = l_Nat_reprFast(v_id_2950_);
if (v_isShared_2914_ == 0)
{
lean_ctor_set_tag(v___x_2913_, 3);
lean_ctor_set(v___x_2913_, 0, v___x_2952_);
v___x_2954_ = v___x_2913_;
goto v_reusejp_2953_;
}
else
{
lean_object* v_reuseFailAlloc_2964_; 
v_reuseFailAlloc_2964_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2964_, 0, v___x_2952_);
v___x_2954_ = v_reuseFailAlloc_2964_;
goto v_reusejp_2953_;
}
v_reusejp_2953_:
{
lean_object* v___x_2955_; lean_object* v___x_2957_; 
v___x_2955_ = l_Lean_MessageData_ofFormat(v___x_2954_);
if (v_isShared_2923_ == 0)
{
lean_ctor_set_tag(v___x_2922_, 7);
lean_ctor_set(v___x_2922_, 1, v___x_2955_);
lean_ctor_set(v___x_2922_, 0, v___x_2951_);
v___x_2957_ = v___x_2922_;
goto v_reusejp_2956_;
}
else
{
lean_object* v_reuseFailAlloc_2963_; 
v_reuseFailAlloc_2963_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2963_, 0, v___x_2951_);
lean_ctor_set(v_reuseFailAlloc_2963_, 1, v___x_2955_);
v___x_2957_ = v_reuseFailAlloc_2963_;
goto v_reusejp_2956_;
}
v_reusejp_2956_:
{
lean_object* v___x_2958_; lean_object* v___x_2960_; 
v___x_2958_ = lean_obj_once(&lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal___closed__3, &lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal___closed__3_once, _init_lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal___closed__3);
if (v_isShared_2919_ == 0)
{
lean_ctor_set_tag(v___x_2918_, 7);
lean_ctor_set(v___x_2918_, 1, v___x_2958_);
lean_ctor_set(v___x_2918_, 0, v___x_2957_);
v___x_2960_ = v___x_2918_;
goto v_reusejp_2959_;
}
else
{
lean_object* v_reuseFailAlloc_2962_; 
v_reuseFailAlloc_2962_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2962_, 0, v___x_2957_);
lean_ctor_set(v_reuseFailAlloc_2962_, 1, v___x_2958_);
v___x_2960_ = v_reuseFailAlloc_2962_;
goto v_reusejp_2959_;
}
v_reusejp_2959_:
{
lean_object* v___x_2961_; 
v___x_2961_ = lp_aesop_Lean_throwError___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal_spec__0___redArg(v___x_2960_, v_a_2901_, v_a_2902_, v_a_2903_, v_a_2904_);
if (lean_obj_tag(v___x_2961_) == 0)
{
lean_dec_ref_known(v___x_2961_, 1);
v___y_2926_ = v_a_2900_;
v___y_2927_ = v_a_2901_;
v___y_2928_ = v_a_2902_;
v___y_2929_ = v_a_2903_;
v___y_2930_ = v_a_2904_;
goto v___jp_2925_;
}
else
{
lean_dec_ref(v___x_2924_);
lean_dec(v_snd_2920_);
lean_dec(v_fst_2916_);
lean_del_object(v___x_2909_);
return v___x_2961_;
}
}
}
}
}
v___jp_2925_:
{
lean_object* v___x_2931_; lean_object* v___x_2932_; uint8_t v___x_2933_; 
v___x_2931_ = lean_unsigned_to_nat(0u);
v___x_2932_ = lean_array_get_size(v___x_2924_);
v___x_2933_ = lean_nat_dec_lt(v___x_2931_, v___x_2932_);
if (v___x_2933_ == 0)
{
lean_object* v___x_2934_; lean_object* v___x_2935_; lean_object* v___x_2936_; lean_object* v___x_2937_; lean_object* v___x_2939_; 
lean_dec_ref(v___x_2924_);
lean_dec(v_snd_2920_);
v___x_2934_ = lean_st_ref_take(v___y_2926_);
v___x_2935_ = lean_array_push(v___x_2934_, v_fst_2916_);
v___x_2936_ = lean_st_ref_set(v___y_2926_, v___x_2935_);
v___x_2937_ = lean_box(0);
if (v_isShared_2910_ == 0)
{
lean_ctor_set(v___x_2909_, 0, v___x_2937_);
v___x_2939_ = v___x_2909_;
goto v_reusejp_2938_;
}
else
{
lean_object* v_reuseFailAlloc_2940_; 
v_reuseFailAlloc_2940_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2940_, 0, v___x_2937_);
v___x_2939_ = v_reuseFailAlloc_2940_;
goto v_reusejp_2938_;
}
v_reusejp_2938_:
{
return v___x_2939_;
}
}
else
{
lean_object* v___x_2941_; lean_object* v___x_2942_; lean_object* v___x_2943_; 
lean_del_object(v___x_2909_);
v___x_2941_ = lean_array_fget(v___x_2924_, v___x_2931_);
lean_dec_ref(v___x_2924_);
v___x_2942_ = lean_st_ref_get(v___x_2941_);
lean_dec(v___x_2941_);
v___x_2943_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixRapp(v_snd_2920_, v_fst_2916_, v___x_2942_, v___y_2926_, v___y_2927_, v___y_2928_, v___y_2929_, v___y_2930_);
return v___x_2943_;
}
}
}
}
}
}
else
{
lean_object* v___x_2969_; lean_object* v___x_2971_; 
lean_dec(v_a_2907_);
lean_dec(v_g_2899_);
v___x_2969_ = lean_box(0);
if (v_isShared_2910_ == 0)
{
lean_ctor_set(v___x_2909_, 0, v___x_2969_);
v___x_2971_ = v___x_2909_;
goto v_reusejp_2970_;
}
else
{
lean_object* v_reuseFailAlloc_2972_; 
v_reuseFailAlloc_2972_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2972_, 0, v___x_2969_);
v___x_2971_ = v_reuseFailAlloc_2972_;
goto v_reusejp_2970_;
}
v_reusejp_2970_:
{
return v___x_2971_;
}
}
}
}
else
{
lean_object* v_a_2974_; lean_object* v___x_2976_; uint8_t v_isShared_2977_; uint8_t v_isSharedCheck_2981_; 
lean_dec(v_g_2899_);
v_a_2974_ = lean_ctor_get(v___x_2906_, 0);
v_isSharedCheck_2981_ = !lean_is_exclusive(v___x_2906_);
if (v_isSharedCheck_2981_ == 0)
{
v___x_2976_ = v___x_2906_;
v_isShared_2977_ = v_isSharedCheck_2981_;
goto v_resetjp_2975_;
}
else
{
lean_inc(v_a_2974_);
lean_dec(v___x_2906_);
v___x_2976_ = lean_box(0);
v_isShared_2977_ = v_isSharedCheck_2981_;
goto v_resetjp_2975_;
}
v_resetjp_2975_:
{
lean_object* v___x_2979_; 
if (v_isShared_2977_ == 0)
{
v___x_2979_ = v___x_2976_;
goto v_reusejp_2978_;
}
else
{
lean_object* v_reuseFailAlloc_2980_; 
v_reuseFailAlloc_2980_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2980_, 0, v_a_2974_);
v___x_2979_ = v_reuseFailAlloc_2980_;
goto v_reusejp_2978_;
}
v_reusejp_2978_:
{
return v___x_2979_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixMVarCluster_spec__4(lean_object* v_parentEnv_2982_, lean_object* v_as_2983_, size_t v_i_2984_, size_t v_stop_2985_, lean_object* v_b_2986_, lean_object* v___y_2987_, lean_object* v___y_2988_, lean_object* v___y_2989_, lean_object* v___y_2990_, lean_object* v___y_2991_){
_start:
{
uint8_t v___x_2993_; 
v___x_2993_ = lean_usize_dec_eq(v_i_2984_, v_stop_2985_);
if (v___x_2993_ == 0)
{
lean_object* v___x_2994_; lean_object* v___x_2995_; lean_object* v___x_2996_; 
v___x_2994_ = lean_array_uget_borrowed(v_as_2983_, v_i_2984_);
v___x_2995_ = lean_st_ref_get(v___x_2994_);
lean_inc_ref(v_parentEnv_2982_);
v___x_2996_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal(v_parentEnv_2982_, v___x_2995_, v___y_2987_, v___y_2988_, v___y_2989_, v___y_2990_, v___y_2991_);
if (lean_obj_tag(v___x_2996_) == 0)
{
lean_object* v_a_2997_; size_t v___x_2998_; size_t v___x_2999_; 
v_a_2997_ = lean_ctor_get(v___x_2996_, 0);
lean_inc(v_a_2997_);
lean_dec_ref_known(v___x_2996_, 1);
v___x_2998_ = ((size_t)1ULL);
v___x_2999_ = lean_usize_add(v_i_2984_, v___x_2998_);
v_i_2984_ = v___x_2999_;
v_b_2986_ = v_a_2997_;
goto _start;
}
else
{
lean_dec_ref(v_parentEnv_2982_);
return v___x_2996_;
}
}
else
{
lean_object* v___x_3001_; 
lean_dec_ref(v_parentEnv_2982_);
v___x_3001_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3001_, 0, v_b_2986_);
return v___x_3001_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixMVarCluster_spec__4___boxed(lean_object* v_parentEnv_3002_, lean_object* v_as_3003_, lean_object* v_i_3004_, lean_object* v_stop_3005_, lean_object* v_b_3006_, lean_object* v___y_3007_, lean_object* v___y_3008_, lean_object* v___y_3009_, lean_object* v___y_3010_, lean_object* v___y_3011_, lean_object* v___y_3012_){
_start:
{
size_t v_i_boxed_3013_; size_t v_stop_boxed_3014_; lean_object* v_res_3015_; 
v_i_boxed_3013_ = lean_unbox_usize(v_i_3004_);
lean_dec(v_i_3004_);
v_stop_boxed_3014_ = lean_unbox_usize(v_stop_3005_);
lean_dec(v_stop_3005_);
v_res_3015_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixMVarCluster_spec__4(v_parentEnv_3002_, v_as_3003_, v_i_boxed_3013_, v_stop_boxed_3014_, v_b_3006_, v___y_3007_, v___y_3008_, v___y_3009_, v___y_3010_, v___y_3011_);
lean_dec(v___y_3011_);
lean_dec_ref(v___y_3010_);
lean_dec(v___y_3009_);
lean_dec_ref(v___y_3008_);
lean_dec(v___y_3007_);
lean_dec_ref(v_as_3003_);
return v_res_3015_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixRapp_spec__2___boxed(lean_object* v_snd_3016_, lean_object* v_as_3017_, lean_object* v_i_3018_, lean_object* v_stop_3019_, lean_object* v_b_3020_, lean_object* v___y_3021_, lean_object* v___y_3022_, lean_object* v___y_3023_, lean_object* v___y_3024_, lean_object* v___y_3025_, lean_object* v___y_3026_){
_start:
{
size_t v_i_boxed_3027_; size_t v_stop_boxed_3028_; lean_object* v_res_3029_; 
v_i_boxed_3027_ = lean_unbox_usize(v_i_3018_);
lean_dec(v_i_3018_);
v_stop_boxed_3028_ = lean_unbox_usize(v_stop_3019_);
lean_dec(v_stop_3019_);
v_res_3029_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixRapp_spec__2(v_snd_3016_, v_as_3017_, v_i_boxed_3027_, v_stop_boxed_3028_, v_b_3020_, v___y_3021_, v___y_3022_, v___y_3023_, v___y_3024_, v___y_3025_);
lean_dec(v___y_3025_);
lean_dec_ref(v___y_3024_);
lean_dec(v___y_3023_);
lean_dec_ref(v___y_3022_);
lean_dec(v___y_3021_);
lean_dec_ref(v_as_3017_);
return v_res_3029_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixMVarCluster___boxed(lean_object* v_parentEnv_3030_, lean_object* v_c_3031_, lean_object* v_a_3032_, lean_object* v_a_3033_, lean_object* v_a_3034_, lean_object* v_a_3035_, lean_object* v_a_3036_, lean_object* v_a_3037_){
_start:
{
lean_object* v_res_3038_; 
v_res_3038_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixMVarCluster(v_parentEnv_3030_, v_c_3031_, v_a_3032_, v_a_3033_, v_a_3034_, v_a_3035_, v_a_3036_);
lean_dec(v_a_3036_);
lean_dec_ref(v_a_3035_);
lean_dec(v_a_3034_);
lean_dec_ref(v_a_3033_);
lean_dec(v_a_3032_);
return v_res_3038_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixRapp___boxed(lean_object* v_parentEnv_3039_, lean_object* v_parentGoal_3040_, lean_object* v_r_3041_, lean_object* v_a_3042_, lean_object* v_a_3043_, lean_object* v_a_3044_, lean_object* v_a_3045_, lean_object* v_a_3046_, lean_object* v_a_3047_){
_start:
{
lean_object* v_res_3048_; 
v_res_3048_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixRapp(v_parentEnv_3039_, v_parentGoal_3040_, v_r_3041_, v_a_3042_, v_a_3043_, v_a_3044_, v_a_3045_, v_a_3046_);
lean_dec(v_a_3046_);
lean_dec_ref(v_a_3045_);
lean_dec(v_a_3044_);
lean_dec_ref(v_a_3043_);
lean_dec(v_a_3042_);
return v_res_3048_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal___boxed(lean_object* v_parentEnv_3049_, lean_object* v_g_3050_, lean_object* v_a_3051_, lean_object* v_a_3052_, lean_object* v_a_3053_, lean_object* v_a_3054_, lean_object* v_a_3055_, lean_object* v_a_3056_){
_start:
{
lean_object* v_res_3057_; 
v_res_3057_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal(v_parentEnv_3049_, v_g_3050_, v_a_3051_, v_a_3052_, v_a_3053_, v_a_3054_, v_a_3055_);
lean_dec(v_a_3055_);
lean_dec_ref(v_a_3054_);
lean_dec(v_a_3053_);
lean_dec_ref(v_a_3052_);
lean_dec(v_a_3051_);
return v_res_3057_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal_spec__0(lean_object* v_00_u03b1_3058_, lean_object* v_msg_3059_, lean_object* v___y_3060_, lean_object* v___y_3061_, lean_object* v___y_3062_, lean_object* v___y_3063_, lean_object* v___y_3064_){
_start:
{
lean_object* v___x_3066_; 
v___x_3066_ = lp_aesop_Lean_throwError___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal_spec__0___redArg(v_msg_3059_, v___y_3061_, v___y_3062_, v___y_3063_, v___y_3064_);
return v___x_3066_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal_spec__0___boxed(lean_object* v_00_u03b1_3067_, lean_object* v_msg_3068_, lean_object* v___y_3069_, lean_object* v___y_3070_, lean_object* v___y_3071_, lean_object* v___y_3072_, lean_object* v___y_3073_, lean_object* v___y_3074_){
_start:
{
lean_object* v_res_3075_; 
v_res_3075_ = lp_aesop_Lean_throwError___at___00__private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal_spec__0(v_00_u03b1_3067_, v_msg_3068_, v___y_3069_, v___y_3070_, v___y_3071_, v___y_3072_, v___y_3073_);
lean_dec(v___y_3073_);
lean_dec_ref(v___y_3072_);
lean_dec(v___y_3071_);
lean_dec_ref(v___y_3070_);
lean_dec(v___y_3069_);
return v_res_3075_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_extractProof(lean_object* v_root_3076_, lean_object* v_a_3077_, lean_object* v_a_3078_, lean_object* v_a_3079_, lean_object* v_a_3080_){
_start:
{
lean_object* v___x_3082_; lean_object* v_env_3083_; lean_object* v___x_3084_; 
v___x_3082_ = lean_st_ref_get(v_a_3080_);
v_env_3083_ = lean_ctor_get(v___x_3082_, 0);
lean_inc_ref(v_env_3083_);
lean_dec(v___x_3082_);
v___x_3084_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractProofGoal(v_env_3083_, v_root_3076_, v_a_3077_, v_a_3078_, v_a_3079_, v_a_3080_);
return v___x_3084_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_extractProof___boxed(lean_object* v_root_3085_, lean_object* v_a_3086_, lean_object* v_a_3087_, lean_object* v_a_3088_, lean_object* v_a_3089_, lean_object* v_a_3090_){
_start:
{
lean_object* v_res_3091_; 
v_res_3091_ = lp_aesop_Aesop_Goal_extractProof(v_root_3085_, v_a_3086_, v_a_3087_, v_a_3088_, v_a_3089_);
lean_dec(v_a_3089_);
lean_dec_ref(v_a_3088_);
lean_dec(v_a_3087_);
lean_dec_ref(v_a_3086_);
return v_res_3091_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_extractProof(lean_object* v_a_3092_, lean_object* v_a_3093_, lean_object* v_a_3094_, lean_object* v_a_3095_, lean_object* v_a_3096_, lean_object* v_a_3097_, lean_object* v_a_3098_){
_start:
{
lean_object* v___x_3100_; 
v___x_3100_ = lp_aesop_Aesop_getRootGoal(v_a_3092_, v_a_3093_, v_a_3094_, v_a_3095_, v_a_3096_, v_a_3097_, v_a_3098_);
if (lean_obj_tag(v___x_3100_) == 0)
{
lean_object* v_a_3101_; lean_object* v___x_3102_; lean_object* v___x_3103_; 
v_a_3101_ = lean_ctor_get(v___x_3100_, 0);
lean_inc(v_a_3101_);
lean_dec_ref_known(v___x_3100_, 1);
v___x_3102_ = lean_st_ref_get(v_a_3101_);
lean_dec(v_a_3101_);
v___x_3103_ = lp_aesop_Aesop_Goal_extractProof(v___x_3102_, v_a_3095_, v_a_3096_, v_a_3097_, v_a_3098_);
return v___x_3103_;
}
else
{
lean_object* v_a_3104_; lean_object* v___x_3106_; uint8_t v_isShared_3107_; uint8_t v_isSharedCheck_3111_; 
v_a_3104_ = lean_ctor_get(v___x_3100_, 0);
v_isSharedCheck_3111_ = !lean_is_exclusive(v___x_3100_);
if (v_isSharedCheck_3111_ == 0)
{
v___x_3106_ = v___x_3100_;
v_isShared_3107_ = v_isSharedCheck_3111_;
goto v_resetjp_3105_;
}
else
{
lean_inc(v_a_3104_);
lean_dec(v___x_3100_);
v___x_3106_ = lean_box(0);
v_isShared_3107_ = v_isSharedCheck_3111_;
goto v_resetjp_3105_;
}
v_resetjp_3105_:
{
lean_object* v___x_3109_; 
if (v_isShared_3107_ == 0)
{
v___x_3109_ = v___x_3106_;
goto v_reusejp_3108_;
}
else
{
lean_object* v_reuseFailAlloc_3110_; 
v_reuseFailAlloc_3110_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3110_, 0, v_a_3104_);
v___x_3109_ = v_reuseFailAlloc_3110_;
goto v_reusejp_3108_;
}
v_reusejp_3108_:
{
return v___x_3109_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_extractProof___boxed(lean_object* v_a_3112_, lean_object* v_a_3113_, lean_object* v_a_3114_, lean_object* v_a_3115_, lean_object* v_a_3116_, lean_object* v_a_3117_, lean_object* v_a_3118_, lean_object* v_a_3119_){
_start:
{
lean_object* v_res_3120_; 
v_res_3120_ = lp_aesop_Aesop_extractProof(v_a_3112_, v_a_3113_, v_a_3114_, v_a_3115_, v_a_3116_, v_a_3117_, v_a_3118_);
lean_dec(v_a_3118_);
lean_dec_ref(v_a_3117_);
lean_dec(v_a_3116_);
lean_dec_ref(v_a_3115_);
lean_dec(v_a_3114_);
lean_dec(v_a_3113_);
lean_dec_ref(v_a_3112_);
return v_res_3120_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_extractSafePrefix(lean_object* v_root_3123_, lean_object* v_a_3124_, lean_object* v_a_3125_, lean_object* v_a_3126_, lean_object* v_a_3127_){
_start:
{
lean_object* v___x_3129_; lean_object* v___x_3130_; lean_object* v___x_3131_; lean_object* v_env_3132_; lean_object* v___x_3133_; 
v___x_3129_ = lean_st_ref_get(v_a_3127_);
v___x_3130_ = ((lean_object*)(lp_aesop_Aesop_Goal_extractSafePrefix___closed__0));
v___x_3131_ = lean_st_mk_ref(v___x_3130_);
v_env_3132_ = lean_ctor_get(v___x_3129_, 0);
lean_inc_ref(v_env_3132_);
lean_dec(v___x_3129_);
v___x_3133_ = lp_aesop___private_Aesop_Tree_ExtractProof_0__Aesop_extractSafePrefixGoal(v_env_3132_, v_root_3123_, v___x_3131_, v_a_3124_, v_a_3125_, v_a_3126_, v_a_3127_);
if (lean_obj_tag(v___x_3133_) == 0)
{
lean_object* v___x_3135_; uint8_t v_isShared_3136_; uint8_t v_isSharedCheck_3141_; 
v_isSharedCheck_3141_ = !lean_is_exclusive(v___x_3133_);
if (v_isSharedCheck_3141_ == 0)
{
lean_object* v_unused_3142_; 
v_unused_3142_ = lean_ctor_get(v___x_3133_, 0);
lean_dec(v_unused_3142_);
v___x_3135_ = v___x_3133_;
v_isShared_3136_ = v_isSharedCheck_3141_;
goto v_resetjp_3134_;
}
else
{
lean_dec(v___x_3133_);
v___x_3135_ = lean_box(0);
v_isShared_3136_ = v_isSharedCheck_3141_;
goto v_resetjp_3134_;
}
v_resetjp_3134_:
{
lean_object* v___x_3137_; lean_object* v___x_3139_; 
v___x_3137_ = lean_st_ref_get(v___x_3131_);
lean_dec(v___x_3131_);
if (v_isShared_3136_ == 0)
{
lean_ctor_set(v___x_3135_, 0, v___x_3137_);
v___x_3139_ = v___x_3135_;
goto v_reusejp_3138_;
}
else
{
lean_object* v_reuseFailAlloc_3140_; 
v_reuseFailAlloc_3140_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3140_, 0, v___x_3137_);
v___x_3139_ = v_reuseFailAlloc_3140_;
goto v_reusejp_3138_;
}
v_reusejp_3138_:
{
return v___x_3139_;
}
}
}
else
{
lean_object* v_a_3143_; lean_object* v___x_3145_; uint8_t v_isShared_3146_; uint8_t v_isSharedCheck_3150_; 
lean_dec(v___x_3131_);
v_a_3143_ = lean_ctor_get(v___x_3133_, 0);
v_isSharedCheck_3150_ = !lean_is_exclusive(v___x_3133_);
if (v_isSharedCheck_3150_ == 0)
{
v___x_3145_ = v___x_3133_;
v_isShared_3146_ = v_isSharedCheck_3150_;
goto v_resetjp_3144_;
}
else
{
lean_inc(v_a_3143_);
lean_dec(v___x_3133_);
v___x_3145_ = lean_box(0);
v_isShared_3146_ = v_isSharedCheck_3150_;
goto v_resetjp_3144_;
}
v_resetjp_3144_:
{
lean_object* v___x_3148_; 
if (v_isShared_3146_ == 0)
{
v___x_3148_ = v___x_3145_;
goto v_reusejp_3147_;
}
else
{
lean_object* v_reuseFailAlloc_3149_; 
v_reuseFailAlloc_3149_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3149_, 0, v_a_3143_);
v___x_3148_ = v_reuseFailAlloc_3149_;
goto v_reusejp_3147_;
}
v_reusejp_3147_:
{
return v___x_3148_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Goal_extractSafePrefix___boxed(lean_object* v_root_3151_, lean_object* v_a_3152_, lean_object* v_a_3153_, lean_object* v_a_3154_, lean_object* v_a_3155_, lean_object* v_a_3156_){
_start:
{
lean_object* v_res_3157_; 
v_res_3157_ = lp_aesop_Aesop_Goal_extractSafePrefix(v_root_3151_, v_a_3152_, v_a_3153_, v_a_3154_, v_a_3155_);
lean_dec(v_a_3155_);
lean_dec_ref(v_a_3154_);
lean_dec(v_a_3153_);
lean_dec_ref(v_a_3152_);
return v_res_3157_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_extractSafePrefix(lean_object* v_a_3158_, lean_object* v_a_3159_, lean_object* v_a_3160_, lean_object* v_a_3161_, lean_object* v_a_3162_, lean_object* v_a_3163_, lean_object* v_a_3164_){
_start:
{
lean_object* v___x_3166_; 
v___x_3166_ = lp_aesop_Aesop_getRootGoal(v_a_3158_, v_a_3159_, v_a_3160_, v_a_3161_, v_a_3162_, v_a_3163_, v_a_3164_);
if (lean_obj_tag(v___x_3166_) == 0)
{
lean_object* v_a_3167_; lean_object* v___x_3168_; lean_object* v___x_3169_; 
v_a_3167_ = lean_ctor_get(v___x_3166_, 0);
lean_inc(v_a_3167_);
lean_dec_ref_known(v___x_3166_, 1);
v___x_3168_ = lean_st_ref_get(v_a_3167_);
lean_dec(v_a_3167_);
v___x_3169_ = lp_aesop_Aesop_Goal_extractSafePrefix(v___x_3168_, v_a_3161_, v_a_3162_, v_a_3163_, v_a_3164_);
return v___x_3169_;
}
else
{
lean_object* v_a_3170_; lean_object* v___x_3172_; uint8_t v_isShared_3173_; uint8_t v_isSharedCheck_3177_; 
v_a_3170_ = lean_ctor_get(v___x_3166_, 0);
v_isSharedCheck_3177_ = !lean_is_exclusive(v___x_3166_);
if (v_isSharedCheck_3177_ == 0)
{
v___x_3172_ = v___x_3166_;
v_isShared_3173_ = v_isSharedCheck_3177_;
goto v_resetjp_3171_;
}
else
{
lean_inc(v_a_3170_);
lean_dec(v___x_3166_);
v___x_3172_ = lean_box(0);
v_isShared_3173_ = v_isSharedCheck_3177_;
goto v_resetjp_3171_;
}
v_resetjp_3171_:
{
lean_object* v___x_3175_; 
if (v_isShared_3173_ == 0)
{
v___x_3175_ = v___x_3172_;
goto v_reusejp_3174_;
}
else
{
lean_object* v_reuseFailAlloc_3176_; 
v_reuseFailAlloc_3176_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3176_, 0, v_a_3170_);
v___x_3175_ = v_reuseFailAlloc_3176_;
goto v_reusejp_3174_;
}
v_reusejp_3174_:
{
return v___x_3175_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_extractSafePrefix___boxed(lean_object* v_a_3178_, lean_object* v_a_3179_, lean_object* v_a_3180_, lean_object* v_a_3181_, lean_object* v_a_3182_, lean_object* v_a_3183_, lean_object* v_a_3184_, lean_object* v_a_3185_){
_start:
{
lean_object* v_res_3186_; 
v_res_3186_ = lp_aesop_Aesop_extractSafePrefix(v_a_3178_, v_a_3179_, v_a_3180_, v_a_3181_, v_a_3182_, v_a_3183_, v_a_3184_);
lean_dec(v_a_3184_);
lean_dec_ref(v_a_3183_);
lean_dec(v_a_3182_);
lean_dec_ref(v_a_3181_);
lean_dec(v_a_3180_);
lean_dec(v_a_3179_);
lean_dec_ref(v_a_3178_);
return v_res_3186_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tree_TreeM(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_SavedState(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Tree_ExtractProof(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_TreeM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_SavedState(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Tree_ExtractProof(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Tree_TreeM(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_Meta_SavedState(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Tree_ExtractProof(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_TreeM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Meta_SavedState(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_ExtractProof(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Tree_ExtractProof(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Tree_ExtractProof(builtin);
}
#ifdef __cplusplus
}
#endif
