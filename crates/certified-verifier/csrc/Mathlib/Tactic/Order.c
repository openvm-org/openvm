// Lean compiler output
// Module: Mathlib.Tactic.Order
// Imports: public import Init public meta import Init public meta import Lean.Elab.Tactic.Omega public meta import Mathlib.Tactic.Order.CollectFacts public meta import Mathlib.Tactic.Order.Graph.Basic public import Mathlib.Tactic.ByContra public import Mathlib.Tactic.Order.CollectFacts public import Mathlib.Tactic.Order.Graph.Basic public import Mathlib.Tactic.Order.Graph.Tarjan public import Mathlib.Tactic.Order.Preprocessing public import Mathlib.Tactic.Order.ToInt public import Mathlib.Util.ElabWithoutMVars
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
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Meta_ppExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_usize_to_nat(size_t);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_st_ref_take(lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_constructLeGraph(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProof___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_addEdge(lean_object*, lean_object*);
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
lean_object* l_instInhabitedOfMonad___redArg(lean_object*, lean_object*);
lean_object* l_instInhabitedForall___redArg___lam__0___boxed(lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
uint64_t lean_uint64_of_nat(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lp_mathlib_Mathlib_Tactic_Order_instBEqOrderType_beq(uint8_t, uint8_t);
lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs(lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkEqRefl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Qq_getLevelQ_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* lp_Qq_Qq_synthInstanceQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Order_ToInt_translateToInt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_Elab_Tactic_Omega_omega(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Order_replaceBotTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Order_preprocessFacts(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_String_intercalate(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Order_collectFacts(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_AtomM_run___redArg(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_saveState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_SavedState_restore___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_registerTraceClass(lean_object*, uint8_t, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_elabTerm(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_elabTermWithoutNewMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Syntax_TSepArray_getElems___redArg(lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__0_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "order"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__0_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__0_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__1_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__0_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(92, 250, 133, 237, 39, 132, 42, 246)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__1_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__1_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__2_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__2_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__2_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__3_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__2_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__3_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__3_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__4_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__4_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__4_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__5_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__3_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__4_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__5_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__5_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__6_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__6_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__6_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__7_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__5_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__6_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__7_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__7_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__8_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Order"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__8_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__8_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__9_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__7_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__8_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(18, 206, 225, 242, 248, 198, 182, 154)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__9_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__9_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__10_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__9_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(147, 224, 55, 143, 136, 178, 82, 108)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__10_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__10_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__11_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__10_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__4_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(110, 157, 105, 219, 42, 217, 7, 47)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__11_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__11_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__12_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__11_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__6_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(147, 16, 235, 178, 46, 76, 63, 219)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__12_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__12_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__13_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__12_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__8_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(94, 64, 71, 77, 55, 162, 28, 214)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__13_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__13_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__14_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__14_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__14_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__15_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__13_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__14_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(19, 237, 34, 149, 30, 74, 200, 138)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__15_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__15_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__16_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__16_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__16_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__17_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__15_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__16_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(38, 58, 60, 161, 38, 210, 53, 122)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__17_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__17_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__18_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__17_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__4_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(63, 81, 93, 129, 27, 50, 21, 33)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__18_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__18_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__19_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__18_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__6_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(142, 57, 247, 164, 220, 244, 107, 236)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__19_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__19_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__20_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__19_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__8_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(151, 220, 3, 42, 68, 170, 47, 24)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__20_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__20_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__21_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__21_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__22_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__22_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__22_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__23_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__23_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__24_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__24_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__24_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__25_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__25_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__26_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__26_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2____boxed(lean_object*);
static lean_once_cell_t lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2___closed__0;
static const lean_closure_object lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2___closed__1 = (const lean_object*)&lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2___closed__1_value;
static const lean_closure_object lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2___closed__2 = (const lean_object*)&lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2___closed__2_value;
static const lean_closure_object lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2___closed__3 = (const lean_object*)&lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2___closed__3_value;
static const lean_closure_object lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2___closed__4 = (const lean_object*)&lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2_spec__4(lean_object*);
static const lean_string_object lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "Std.Data.DHashMap.Internal.AssocList.Basic"};
static const lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2___closed__0 = (const lean_object*)&lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2___closed__0_value;
static const lean_string_object lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "Std.DHashMap.Internal.AssocList.get!"};
static const lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2___closed__1 = (const lean_object*)&lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2___closed__1_value;
static const lean_string_object lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "key is not present in hash table"};
static const lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2___closed__2 = (const lean_object*)&lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2___closed__2_value;
static lean_once_cell_t lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "le_antisymm"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__1_value),LEAN_SCALAR_PTR_LITERAL(60, 197, 251, 149, 196, 83, 222, 132)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__2_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "Mathlib.Tactic.Order"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__3_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 45, .m_capacity = 45, .m_length = 44, .m_data = "Mathlib.Tactic.Order.findContradictionWithNe"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__4_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 49, .m_capacity = 49, .m_length = 48, .m_data = "Cannot find path in strongly connected component"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__5 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__6;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__7;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_findContradictionWithNe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_findContradictionWithNe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNle_spec__0___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNle_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_findContradictionWithNle(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_findContradictionWithNle___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNle_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNle_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "le_of_not_lt_le"};
static const lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__4_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__6_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__8_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(86, 184, 242, 98, 125, 170, 160, 198)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(238, 15, 36, 226, 139, 79, 197, 150)}};
static const lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "Mathlib.Tactic.Order.updateGraphWithNltInfSup"};
static const lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__2_value;
static const lean_string_object lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Non-nlt fact in nltFacts."};
static const lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__4;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "sup_le"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__0 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(217, 62, 157, 97, 234, 82, 162, 188)}};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__1 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__1_value;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "le_inf"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__2 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(86, 126, 92, 163, 75, 1, 21, 111)}};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__3 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__3_value;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 40, .m_capacity = 40, .m_length = 39, .m_data = "Non-isInf or isSup fact in infSupFacts."};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__4 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__5;
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__10(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__5_spec__5_spec__6_spec__13___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__5_spec__5_spec__6___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__5_spec__5___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__7_spec__8(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__7_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__7___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__6(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__9(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__11(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Order_updateGraphWithNltInfSup___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Order_updateGraphWithNltInfSup___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Order_updateGraphWithNltInfSup___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Order_updateGraphWithNltInfSup___closed__1;
static const lean_array_object lp_mathlib_Mathlib_Tactic_Order_updateGraphWithNltInfSup___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_updateGraphWithNltInfSup___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_updateGraphWithNltInfSup___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_updateGraphWithNltInfSup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_updateGraphWithNltInfSup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__5_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__5_spec__5_spec__6(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__5_spec__5_spec__6_spec__13(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Order_instOrdProdNatExpr__mathlib___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_instOrdProdNatExpr__mathlib___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Order_instOrdProdNatExpr__mathlib___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Order_instOrdProdNatExpr__mathlib___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_instOrdProdNatExpr__mathlib___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instOrdProdNatExpr__mathlib___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Order_instOrdProdNatExpr__mathlib = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instOrdProdNatExpr__mathlib___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Order_orderCoreImp_spec__8___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Order_orderCoreImp_spec__8___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Order_orderCoreImp_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Order_orderCoreImp_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Order_orderCoreImp_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Order_orderCoreImp_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "#"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0___redArg___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0___redArg(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Order_orderCoreImp_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Order_orderCoreImp_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___redArg___closed__0;
static const lean_string_object lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___redArg___closed__1_value;
static const lean_array_object lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " = #"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " ≠ #"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " ≤ #"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__2_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 3, .m_data = "¬ #"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__3_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " < #"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__4_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 5, .m_data = " := ⊤"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__5 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__5_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 5, .m_data = " := ⊥"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__6 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__6_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " := #"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__7 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__7_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " ⊓ #"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__8 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__8_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " ⊔ #"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__9 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Order_orderCoreImp_spec__2_spec__3(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Order_orderCoreImp_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_Order_orderCoreImp_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_Order_orderCoreImp_spec__2___closed__0 = (const lean_object*)&lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_Order_orderCoreImp_spec__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_Order_orderCoreImp_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_Order_orderCoreImp_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3_spec__11_spec__13___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3_spec__11___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3_spec__12___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__0 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__1 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__2 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__2_value)}};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__3 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__3_value;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "LinearOrder"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__4 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__4_value),LEAN_SCALAR_PTR_LITERAL(99, 217, 57, 95, 10, 230, 120, 46)}};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__5 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__5_value;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__6 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__7;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "Processed facts:\n"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__8 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__9;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "Collected facts:\n"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__10 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__10_value;
static lean_once_cell_t lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__11;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "Working on type "};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__12 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__12_value;
static lean_once_cell_t lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__13;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " ("};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__14 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__14_value;
static lean_once_cell_t lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__15;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__16 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__16_value;
static lean_once_cell_t lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__17;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "linear order"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__18 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__18_value;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "partial order"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__19 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__19_value;
static const lean_string_object lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "preorder"};
static const lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__20 = (const lean_object*)&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__20_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_orderCoreImp_spec__6(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_orderCoreImp_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "No contradiction found.\n\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 58, .m_capacity = 58, .m_length = 57, .m_data = "Additional diagnostic information may be available using "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "the `set_option trace.order true` command."};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__10;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "Collected atoms:\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__11_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__12;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCoreImp(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Order_orderCoreImp_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Order_orderCoreImp_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3_spec__11(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3_spec__12(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3_spec__11_spec__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCore___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCore___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Order_orderCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Order_orderCore___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCore___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderCore___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCore(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "orderArgs"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__4_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__6_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__8_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(86, 184, 242, 98, 125, 170, 160, 198)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__0_value),LEAN_SCALAR_PTR_LITERAL(205, 91, 50, 25, 142, 178, 228, 14)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__4_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " only"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " ["};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__11_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 10}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__13_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__14_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__16_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__18_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__21_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__22_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 9}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__1_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__23_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__24_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Order_orderArgs = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__24_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "order_core"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__13_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__0_value),LEAN_SCALAR_PTR_LITERAL(79, 53, 15, 141, 41, 201, 190, 24)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__24_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__4_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__5_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__6_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__7_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__8_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__8_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__1(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1___lam__2(lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "tacticOrder_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__4_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__6_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__8_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(86, 184, 242, 98, 125, 170, 160, 198)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(199, 243, 227, 168, 152, 105, 113, 149)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__0_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__24_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Order_tacticOrder__ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cdot"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(238, 151, 138, 49, 249, 18, 254, 242)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "cdotTk"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(117, 126, 44, 217, 38, 3, 69, 145)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "·"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__8_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__6_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__10_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__6_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__10_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "intros"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__14_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__6_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__14_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(26, 175, 18, 116, 252, 50, 128, 45)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__14_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__15;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "ByContra"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "byContra!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__4_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__18_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__18_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__6_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__18_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__18_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(81, 42, 251, 71, 231, 71, 43, 201)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__18_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(183, 203, 28, 162, 126, 183, 195, 112)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "by_contra!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__21_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__21_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__21_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__21_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__6_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__21_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__20_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "rcasesPatMed"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__23_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__23_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__23_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__23_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__6_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__23_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__22_value),LEAN_SCALAR_PTR_LITERAL(253, 13, 65, 195, 228, 27, 47, 149)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "rcasesPat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "one"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__26_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__26_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__26_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__26_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__26_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__6_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__26_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__26_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__24_value),LEAN_SCALAR_PTR_LITERAL(162, 181, 165, 225, 136, 177, 169, 19)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__26_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__25_value),LEAN_SCALAR_PTR_LITERAL(186, 152, 172, 228, 11, 240, 156, 168)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__26_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "_order_neg_goal"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__27_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__28;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__27_value),LEAN_SCALAR_PTR_LITERAL(2, 243, 101, 125, 196, 243, 129, 243)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__29_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__21_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_49_ = lean_unsigned_to_nat(2841239826u);
v___x_50_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__20_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_));
v___x_51_ = l_Lean_Name_num___override(v___x_50_, v___x_49_);
return v___x_51_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__23_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; 
v___x_53_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__22_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_));
v___x_54_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__21_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__21_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__21_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_);
v___x_55_ = l_Lean_Name_str___override(v___x_54_, v___x_53_);
return v___x_55_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__25_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; 
v___x_57_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__24_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_));
v___x_58_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__23_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__23_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__23_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_);
v___x_59_ = l_Lean_Name_str___override(v___x_58_, v___x_57_);
return v___x_59_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__26_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; 
v___x_60_ = lean_unsigned_to_nat(2u);
v___x_61_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__25_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__25_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__25_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_);
v___x_62_ = l_Lean_Name_num___override(v___x_61_, v___x_60_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_64_; uint8_t v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; 
v___x_64_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__1_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_));
v___x_65_ = 0;
v___x_66_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__26_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__26_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__26_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_);
v___x_67_ = l_Lean_registerTraceClass(v___x_64_, v___x_65_, v___x_66_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2____boxed(lean_object* v_a_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_();
return v_res_69_;
}
}
static lean_object* _init_lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2___closed__0(void){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = l_instMonadEIO(lean_box(0));
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2(lean_object* v_msg_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_){
_start:
{
lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v_toApplicative_85_; lean_object* v___x_87_; uint8_t v_isShared_88_; uint8_t v_isSharedCheck_148_; 
v___x_83_ = lean_obj_once(&lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2___closed__0, &lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2___closed__0_once, _init_lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2___closed__0);
v___x_84_ = l_StateRefT_x27_instMonad___redArg(v___x_83_);
v_toApplicative_85_ = lean_ctor_get(v___x_84_, 0);
v_isSharedCheck_148_ = !lean_is_exclusive(v___x_84_);
if (v_isSharedCheck_148_ == 0)
{
lean_object* v_unused_149_; 
v_unused_149_ = lean_ctor_get(v___x_84_, 1);
lean_dec(v_unused_149_);
v___x_87_ = v___x_84_;
v_isShared_88_ = v_isSharedCheck_148_;
goto v_resetjp_86_;
}
else
{
lean_inc(v_toApplicative_85_);
lean_dec(v___x_84_);
v___x_87_ = lean_box(0);
v_isShared_88_ = v_isSharedCheck_148_;
goto v_resetjp_86_;
}
v_resetjp_86_:
{
lean_object* v_toFunctor_89_; lean_object* v_toSeq_90_; lean_object* v_toSeqLeft_91_; lean_object* v_toSeqRight_92_; lean_object* v___x_94_; uint8_t v_isShared_95_; uint8_t v_isSharedCheck_146_; 
v_toFunctor_89_ = lean_ctor_get(v_toApplicative_85_, 0);
v_toSeq_90_ = lean_ctor_get(v_toApplicative_85_, 2);
v_toSeqLeft_91_ = lean_ctor_get(v_toApplicative_85_, 3);
v_toSeqRight_92_ = lean_ctor_get(v_toApplicative_85_, 4);
v_isSharedCheck_146_ = !lean_is_exclusive(v_toApplicative_85_);
if (v_isSharedCheck_146_ == 0)
{
lean_object* v_unused_147_; 
v_unused_147_ = lean_ctor_get(v_toApplicative_85_, 1);
lean_dec(v_unused_147_);
v___x_94_ = v_toApplicative_85_;
v_isShared_95_ = v_isSharedCheck_146_;
goto v_resetjp_93_;
}
else
{
lean_inc(v_toSeqRight_92_);
lean_inc(v_toSeqLeft_91_);
lean_inc(v_toSeq_90_);
lean_inc(v_toFunctor_89_);
lean_dec(v_toApplicative_85_);
v___x_94_ = lean_box(0);
v_isShared_95_ = v_isSharedCheck_146_;
goto v_resetjp_93_;
}
v_resetjp_93_:
{
lean_object* v___f_96_; lean_object* v___f_97_; lean_object* v___f_98_; lean_object* v___f_99_; lean_object* v___x_100_; lean_object* v___f_101_; lean_object* v___f_102_; lean_object* v___f_103_; lean_object* v___x_105_; 
v___f_96_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2___closed__1));
v___f_97_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2___closed__2));
lean_inc_ref(v_toFunctor_89_);
v___f_98_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_98_, 0, v_toFunctor_89_);
v___f_99_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_99_, 0, v_toFunctor_89_);
v___x_100_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_100_, 0, v___f_98_);
lean_ctor_set(v___x_100_, 1, v___f_99_);
v___f_101_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_101_, 0, v_toSeqRight_92_);
v___f_102_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_102_, 0, v_toSeqLeft_91_);
v___f_103_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_103_, 0, v_toSeq_90_);
if (v_isShared_95_ == 0)
{
lean_ctor_set(v___x_94_, 4, v___f_101_);
lean_ctor_set(v___x_94_, 3, v___f_102_);
lean_ctor_set(v___x_94_, 2, v___f_103_);
lean_ctor_set(v___x_94_, 1, v___f_96_);
lean_ctor_set(v___x_94_, 0, v___x_100_);
v___x_105_ = v___x_94_;
goto v_reusejp_104_;
}
else
{
lean_object* v_reuseFailAlloc_145_; 
v_reuseFailAlloc_145_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_145_, 0, v___x_100_);
lean_ctor_set(v_reuseFailAlloc_145_, 1, v___f_96_);
lean_ctor_set(v_reuseFailAlloc_145_, 2, v___f_103_);
lean_ctor_set(v_reuseFailAlloc_145_, 3, v___f_102_);
lean_ctor_set(v_reuseFailAlloc_145_, 4, v___f_101_);
v___x_105_ = v_reuseFailAlloc_145_;
goto v_reusejp_104_;
}
v_reusejp_104_:
{
lean_object* v___x_107_; 
if (v_isShared_88_ == 0)
{
lean_ctor_set(v___x_87_, 1, v___f_97_);
lean_ctor_set(v___x_87_, 0, v___x_105_);
v___x_107_ = v___x_87_;
goto v_reusejp_106_;
}
else
{
lean_object* v_reuseFailAlloc_144_; 
v_reuseFailAlloc_144_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_144_, 0, v___x_105_);
lean_ctor_set(v_reuseFailAlloc_144_, 1, v___f_97_);
v___x_107_ = v_reuseFailAlloc_144_;
goto v_reusejp_106_;
}
v_reusejp_106_:
{
lean_object* v___x_108_; lean_object* v_toApplicative_109_; lean_object* v___x_111_; uint8_t v_isShared_112_; uint8_t v_isSharedCheck_142_; 
v___x_108_ = l_StateRefT_x27_instMonad___redArg(v___x_107_);
v_toApplicative_109_ = lean_ctor_get(v___x_108_, 0);
v_isSharedCheck_142_ = !lean_is_exclusive(v___x_108_);
if (v_isSharedCheck_142_ == 0)
{
lean_object* v_unused_143_; 
v_unused_143_ = lean_ctor_get(v___x_108_, 1);
lean_dec(v_unused_143_);
v___x_111_ = v___x_108_;
v_isShared_112_ = v_isSharedCheck_142_;
goto v_resetjp_110_;
}
else
{
lean_inc(v_toApplicative_109_);
lean_dec(v___x_108_);
v___x_111_ = lean_box(0);
v_isShared_112_ = v_isSharedCheck_142_;
goto v_resetjp_110_;
}
v_resetjp_110_:
{
lean_object* v_toFunctor_113_; lean_object* v_toSeq_114_; lean_object* v_toSeqLeft_115_; lean_object* v_toSeqRight_116_; lean_object* v___x_118_; uint8_t v_isShared_119_; uint8_t v_isSharedCheck_140_; 
v_toFunctor_113_ = lean_ctor_get(v_toApplicative_109_, 0);
v_toSeq_114_ = lean_ctor_get(v_toApplicative_109_, 2);
v_toSeqLeft_115_ = lean_ctor_get(v_toApplicative_109_, 3);
v_toSeqRight_116_ = lean_ctor_get(v_toApplicative_109_, 4);
v_isSharedCheck_140_ = !lean_is_exclusive(v_toApplicative_109_);
if (v_isSharedCheck_140_ == 0)
{
lean_object* v_unused_141_; 
v_unused_141_ = lean_ctor_get(v_toApplicative_109_, 1);
lean_dec(v_unused_141_);
v___x_118_ = v_toApplicative_109_;
v_isShared_119_ = v_isSharedCheck_140_;
goto v_resetjp_117_;
}
else
{
lean_inc(v_toSeqRight_116_);
lean_inc(v_toSeqLeft_115_);
lean_inc(v_toSeq_114_);
lean_inc(v_toFunctor_113_);
lean_dec(v_toApplicative_109_);
v___x_118_ = lean_box(0);
v_isShared_119_ = v_isSharedCheck_140_;
goto v_resetjp_117_;
}
v_resetjp_117_:
{
lean_object* v___f_120_; lean_object* v___f_121_; lean_object* v___f_122_; lean_object* v___f_123_; lean_object* v___x_124_; lean_object* v___f_125_; lean_object* v___f_126_; lean_object* v___f_127_; lean_object* v___x_129_; 
v___f_120_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2___closed__3));
v___f_121_ = ((lean_object*)(lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2___closed__4));
lean_inc_ref(v_toFunctor_113_);
v___f_122_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_122_, 0, v_toFunctor_113_);
v___f_123_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_123_, 0, v_toFunctor_113_);
v___x_124_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_124_, 0, v___f_122_);
lean_ctor_set(v___x_124_, 1, v___f_123_);
v___f_125_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_125_, 0, v_toSeqRight_116_);
v___f_126_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_126_, 0, v_toSeqLeft_115_);
v___f_127_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_127_, 0, v_toSeq_114_);
if (v_isShared_119_ == 0)
{
lean_ctor_set(v___x_118_, 4, v___f_125_);
lean_ctor_set(v___x_118_, 3, v___f_126_);
lean_ctor_set(v___x_118_, 2, v___f_127_);
lean_ctor_set(v___x_118_, 1, v___f_120_);
lean_ctor_set(v___x_118_, 0, v___x_124_);
v___x_129_ = v___x_118_;
goto v_reusejp_128_;
}
else
{
lean_object* v_reuseFailAlloc_139_; 
v_reuseFailAlloc_139_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_139_, 0, v___x_124_);
lean_ctor_set(v_reuseFailAlloc_139_, 1, v___f_120_);
lean_ctor_set(v_reuseFailAlloc_139_, 2, v___f_127_);
lean_ctor_set(v_reuseFailAlloc_139_, 3, v___f_126_);
lean_ctor_set(v_reuseFailAlloc_139_, 4, v___f_125_);
v___x_129_ = v_reuseFailAlloc_139_;
goto v_reusejp_128_;
}
v_reusejp_128_:
{
lean_object* v___x_131_; 
if (v_isShared_112_ == 0)
{
lean_ctor_set(v___x_111_, 1, v___f_121_);
lean_ctor_set(v___x_111_, 0, v___x_129_);
v___x_131_ = v___x_111_;
goto v_reusejp_130_;
}
else
{
lean_object* v_reuseFailAlloc_138_; 
v_reuseFailAlloc_138_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_138_, 0, v___x_129_);
lean_ctor_set(v_reuseFailAlloc_138_, 1, v___f_121_);
v___x_131_ = v_reuseFailAlloc_138_;
goto v_reusejp_130_;
}
v_reusejp_130_:
{
lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___f_135_; lean_object* v___x_6003__overap_136_; lean_object* v___x_137_; 
v___x_132_ = l_StateRefT_x27_instMonad___redArg(v___x_131_);
v___x_133_ = lean_box(0);
v___x_134_ = l_instInhabitedOfMonad___redArg(v___x_132_, v___x_133_);
v___f_135_ = lean_alloc_closure((void*)(l_instInhabitedForall___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_135_, 0, v___x_134_);
v___x_6003__overap_136_ = lean_panic_fn_borrowed(v___f_135_, v_msg_75_);
lean_dec_ref(v___f_135_);
lean_inc(v___y_81_);
lean_inc_ref(v___y_80_);
lean_inc(v___y_79_);
lean_inc_ref(v___y_78_);
lean_inc(v___y_77_);
lean_inc_ref(v___y_76_);
v___x_137_ = lean_apply_7(v___x_6003__overap_136_, v___y_76_, v___y_77_, v___y_78_, v___y_79_, v___y_80_, v___y_81_, lean_box(0));
return v___x_137_;
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
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2___boxed(lean_object* v_msg_150_, lean_object* v___y_151_, lean_object* v___y_152_, lean_object* v___y_153_, lean_object* v___y_154_, lean_object* v___y_155_, lean_object* v___y_156_, lean_object* v___y_157_){
_start:
{
lean_object* v_res_158_; 
v_res_158_ = lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2(v_msg_150_, v___y_151_, v___y_152_, v___y_153_, v___y_154_, v___y_155_, v___y_156_);
lean_dec(v___y_156_);
lean_dec_ref(v___y_155_);
lean_dec(v___y_154_);
lean_dec_ref(v___y_153_);
lean_dec(v___y_152_);
lean_dec_ref(v___y_151_);
return v_res_158_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0_spec__0___redArg(lean_object* v_a_159_, lean_object* v_x_160_){
_start:
{
if (lean_obj_tag(v_x_160_) == 0)
{
uint8_t v___x_161_; 
v___x_161_ = 0;
return v___x_161_;
}
else
{
lean_object* v_key_162_; lean_object* v_tail_163_; uint8_t v___x_164_; 
v_key_162_ = lean_ctor_get(v_x_160_, 0);
v_tail_163_ = lean_ctor_get(v_x_160_, 2);
v___x_164_ = lean_nat_dec_eq(v_key_162_, v_a_159_);
if (v___x_164_ == 0)
{
v_x_160_ = v_tail_163_;
goto _start;
}
else
{
return v___x_164_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0_spec__0___redArg___boxed(lean_object* v_a_166_, lean_object* v_x_167_){
_start:
{
uint8_t v_res_168_; lean_object* v_r_169_; 
v_res_168_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0_spec__0___redArg(v_a_166_, v_x_167_);
lean_dec(v_x_167_);
lean_dec(v_a_166_);
v_r_169_ = lean_box(v_res_168_);
return v_r_169_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0___redArg(lean_object* v_m_170_, lean_object* v_a_171_){
_start:
{
lean_object* v_buckets_172_; lean_object* v___x_173_; uint64_t v___x_174_; uint64_t v___x_175_; uint64_t v___x_176_; uint64_t v_fold_177_; uint64_t v___x_178_; uint64_t v___x_179_; uint64_t v___x_180_; size_t v___x_181_; size_t v___x_182_; size_t v___x_183_; size_t v___x_184_; size_t v___x_185_; lean_object* v___x_186_; uint8_t v___x_187_; 
v_buckets_172_ = lean_ctor_get(v_m_170_, 1);
v___x_173_ = lean_array_get_size(v_buckets_172_);
v___x_174_ = lean_uint64_of_nat(v_a_171_);
v___x_175_ = 32ULL;
v___x_176_ = lean_uint64_shift_right(v___x_174_, v___x_175_);
v_fold_177_ = lean_uint64_xor(v___x_174_, v___x_176_);
v___x_178_ = 16ULL;
v___x_179_ = lean_uint64_shift_right(v_fold_177_, v___x_178_);
v___x_180_ = lean_uint64_xor(v_fold_177_, v___x_179_);
v___x_181_ = lean_uint64_to_usize(v___x_180_);
v___x_182_ = lean_usize_of_nat(v___x_173_);
v___x_183_ = ((size_t)1ULL);
v___x_184_ = lean_usize_sub(v___x_182_, v___x_183_);
v___x_185_ = lean_usize_land(v___x_181_, v___x_184_);
v___x_186_ = lean_array_uget_borrowed(v_buckets_172_, v___x_185_);
v___x_187_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0_spec__0___redArg(v_a_171_, v___x_186_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0___redArg___boxed(lean_object* v_m_188_, lean_object* v_a_189_){
_start:
{
uint8_t v_res_190_; lean_object* v_r_191_; 
v_res_190_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0___redArg(v_m_188_, v_a_189_);
lean_dec(v_a_189_);
lean_dec_ref(v_m_188_);
v_r_191_ = lean_box(v_res_190_);
return v_r_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2_spec__4(lean_object* v_msg_192_){
_start:
{
lean_object* v___x_193_; lean_object* v___x_194_; 
v___x_193_ = lean_unsigned_to_nat(0u);
v___x_194_ = lean_panic_fn_borrowed(v___x_193_, v_msg_192_);
return v___x_194_;
}
}
static lean_object* _init_lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2___closed__3(void){
_start:
{
lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; 
v___x_198_ = ((lean_object*)(lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2___closed__2));
v___x_199_ = lean_unsigned_to_nat(11u);
v___x_200_ = lean_unsigned_to_nat(163u);
v___x_201_ = ((lean_object*)(lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2___closed__1));
v___x_202_ = ((lean_object*)(lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2___closed__0));
v___x_203_ = l_mkPanicMessageWithDecl(v___x_202_, v___x_201_, v___x_200_, v___x_199_, v___x_198_);
return v___x_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2(lean_object* v_a_204_, lean_object* v_x_205_){
_start:
{
if (lean_obj_tag(v_x_205_) == 0)
{
lean_object* v___x_206_; lean_object* v___x_207_; 
v___x_206_ = lean_obj_once(&lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2___closed__3, &lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2___closed__3_once, _init_lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2___closed__3);
v___x_207_ = lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2_spec__4(v___x_206_);
return v___x_207_;
}
else
{
lean_object* v_key_208_; lean_object* v_value_209_; lean_object* v_tail_210_; uint8_t v___x_211_; 
v_key_208_ = lean_ctor_get(v_x_205_, 0);
v_value_209_ = lean_ctor_get(v_x_205_, 1);
v_tail_210_ = lean_ctor_get(v_x_205_, 2);
v___x_211_ = lean_nat_dec_eq(v_key_208_, v_a_204_);
if (v___x_211_ == 0)
{
v_x_205_ = v_tail_210_;
goto _start;
}
else
{
lean_inc(v_value_209_);
return v_value_209_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2___boxed(lean_object* v_a_213_, lean_object* v_x_214_){
_start:
{
lean_object* v_res_215_; 
v_res_215_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2(v_a_213_, v_x_214_);
lean_dec(v_x_214_);
lean_dec(v_a_213_);
return v_res_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1(lean_object* v_m_216_, lean_object* v_a_217_){
_start:
{
lean_object* v_buckets_218_; lean_object* v___x_219_; uint64_t v___x_220_; uint64_t v___x_221_; uint64_t v___x_222_; uint64_t v_fold_223_; uint64_t v___x_224_; uint64_t v___x_225_; uint64_t v___x_226_; size_t v___x_227_; size_t v___x_228_; size_t v___x_229_; size_t v___x_230_; size_t v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; 
v_buckets_218_ = lean_ctor_get(v_m_216_, 1);
v___x_219_ = lean_array_get_size(v_buckets_218_);
v___x_220_ = lean_uint64_of_nat(v_a_217_);
v___x_221_ = 32ULL;
v___x_222_ = lean_uint64_shift_right(v___x_220_, v___x_221_);
v_fold_223_ = lean_uint64_xor(v___x_220_, v___x_222_);
v___x_224_ = 16ULL;
v___x_225_ = lean_uint64_shift_right(v_fold_223_, v___x_224_);
v___x_226_ = lean_uint64_xor(v_fold_223_, v___x_225_);
v___x_227_ = lean_uint64_to_usize(v___x_226_);
v___x_228_ = lean_usize_of_nat(v___x_219_);
v___x_229_ = ((size_t)1ULL);
v___x_230_ = lean_usize_sub(v___x_228_, v___x_229_);
v___x_231_ = lean_usize_land(v___x_227_, v___x_230_);
v___x_232_ = lean_array_uget_borrowed(v_buckets_218_, v___x_231_);
v___x_233_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1_spec__2(v_a_217_, v___x_232_);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1___boxed(lean_object* v_m_234_, lean_object* v_a_235_){
_start:
{
lean_object* v_res_236_; 
v_res_236_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1(v_m_234_, v_a_235_);
lean_dec(v_a_235_);
lean_dec_ref(v_m_234_);
return v_res_236_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__6(void){
_start:
{
lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; 
v___x_246_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__5));
v___x_247_ = lean_unsigned_to_nat(8u);
v___x_248_ = lean_unsigned_to_nat(174u);
v___x_249_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__4));
v___x_250_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__3));
v___x_251_ = l_mkPanicMessageWithDecl(v___x_250_, v___x_249_, v___x_248_, v___x_247_, v___x_246_);
return v___x_251_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__7(void){
_start:
{
lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; 
v___x_252_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__5));
v___x_253_ = lean_unsigned_to_nat(8u);
v___x_254_ = lean_unsigned_to_nat(172u);
v___x_255_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__4));
v___x_256_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__3));
v___x_257_ = l_mkPanicMessageWithDecl(v___x_256_, v___x_255_, v___x_254_, v___x_253_, v___x_252_);
return v___x_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3(lean_object* v_scc_258_, lean_object* v_graph_259_, lean_object* v_as_260_, size_t v_sz_261_, size_t v_i_262_, lean_object* v_b_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_, lean_object* v___y_268_, lean_object* v___y_269_){
_start:
{
lean_object* v_a_272_; uint8_t v___x_276_; 
v___x_276_ = lean_usize_dec_lt(v_i_262_, v_sz_261_);
if (v___x_276_ == 0)
{
lean_object* v___x_277_; 
v___x_277_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_277_, 0, v_b_263_);
return v___x_277_;
}
else
{
lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v_a_280_; 
lean_dec_ref(v_b_263_);
v___x_278_ = lean_box(0);
v___x_279_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__0));
v_a_280_ = lean_array_uget_borrowed(v_as_260_, v_i_262_);
if (lean_obj_tag(v_a_280_) == 1)
{
lean_object* v_lhs_281_; lean_object* v_rhs_282_; lean_object* v_proof_283_; uint8_t v___x_284_; 
v_lhs_281_ = lean_ctor_get(v_a_280_, 0);
v_rhs_282_ = lean_ctor_get(v_a_280_, 1);
v_proof_283_ = lean_ctor_get(v_a_280_, 2);
v___x_284_ = lean_nat_dec_eq(v_lhs_281_, v_rhs_282_);
if (v___x_284_ == 0)
{
uint8_t v___x_285_; 
v___x_285_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0___redArg(v_scc_258_, v_lhs_281_);
if (v___x_285_ == 0)
{
v_a_272_ = v___x_279_;
goto v___jp_271_;
}
else
{
if (v___x_284_ == 0)
{
uint8_t v___x_286_; 
v___x_286_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0___redArg(v_scc_258_, v_rhs_282_);
if (v___x_286_ == 0)
{
v_a_272_ = v___x_279_;
goto v___jp_271_;
}
else
{
lean_object* v___x_287_; lean_object* v___x_288_; uint8_t v___x_289_; 
v___x_287_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1(v_scc_258_, v_lhs_281_);
v___x_288_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__1(v_scc_258_, v_rhs_282_);
v___x_289_ = lean_nat_dec_eq(v___x_287_, v___x_288_);
lean_dec(v___x_288_);
lean_dec(v___x_287_);
if (v___x_289_ == 0)
{
v_a_272_ = v___x_279_;
goto v___jp_271_;
}
else
{
lean_object* v___x_290_; 
lean_inc(v_lhs_281_);
v___x_290_ = lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProof___redArg(v_graph_259_, v_lhs_281_, v_rhs_282_, v___y_265_, v___y_266_, v___y_267_, v___y_268_, v___y_269_);
if (lean_obj_tag(v___x_290_) == 0)
{
lean_object* v_a_291_; 
v_a_291_ = lean_ctor_get(v___x_290_, 0);
lean_inc(v_a_291_);
lean_dec_ref_known(v___x_290_, 1);
if (lean_obj_tag(v_a_291_) == 1)
{
lean_object* v_val_292_; lean_object* v___x_294_; uint8_t v_isShared_295_; uint8_t v_isSharedCheck_351_; 
v_val_292_ = lean_ctor_get(v_a_291_, 0);
v_isSharedCheck_351_ = !lean_is_exclusive(v_a_291_);
if (v_isSharedCheck_351_ == 0)
{
v___x_294_ = v_a_291_;
v_isShared_295_ = v_isSharedCheck_351_;
goto v_resetjp_293_;
}
else
{
lean_inc(v_val_292_);
lean_dec(v_a_291_);
v___x_294_ = lean_box(0);
v_isShared_295_ = v_isSharedCheck_351_;
goto v_resetjp_293_;
}
v_resetjp_293_:
{
lean_object* v___x_296_; 
lean_inc(v_rhs_282_);
v___x_296_ = lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProof___redArg(v_graph_259_, v_rhs_282_, v_lhs_281_, v___y_265_, v___y_266_, v___y_267_, v___y_268_, v___y_269_);
if (lean_obj_tag(v___x_296_) == 0)
{
lean_object* v_a_297_; 
v_a_297_ = lean_ctor_get(v___x_296_, 0);
lean_inc(v_a_297_);
lean_dec_ref_known(v___x_296_, 1);
if (lean_obj_tag(v_a_297_) == 1)
{
lean_object* v_val_298_; lean_object* v___x_300_; uint8_t v_isShared_301_; uint8_t v_isSharedCheck_332_; 
v_val_298_ = lean_ctor_get(v_a_297_, 0);
v_isSharedCheck_332_ = !lean_is_exclusive(v_a_297_);
if (v_isSharedCheck_332_ == 0)
{
v___x_300_ = v_a_297_;
v_isShared_301_ = v_isSharedCheck_332_;
goto v_resetjp_299_;
}
else
{
lean_inc(v_val_298_);
lean_dec(v_a_297_);
v___x_300_ = lean_box(0);
v_isShared_301_ = v_isSharedCheck_332_;
goto v_resetjp_299_;
}
v_resetjp_299_:
{
lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; 
v___x_302_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__2));
v___x_303_ = lean_unsigned_to_nat(2u);
v___x_304_ = lean_mk_empty_array_with_capacity(v___x_303_);
v___x_305_ = lean_array_push(v___x_304_, v_val_292_);
v___x_306_ = lean_array_push(v___x_305_, v_val_298_);
v___x_307_ = l_Lean_Meta_mkAppM(v___x_302_, v___x_306_, v___y_266_, v___y_267_, v___y_268_, v___y_269_);
if (lean_obj_tag(v___x_307_) == 0)
{
lean_object* v_a_308_; lean_object* v___x_310_; uint8_t v_isShared_311_; uint8_t v_isSharedCheck_323_; 
v_a_308_ = lean_ctor_get(v___x_307_, 0);
v_isSharedCheck_323_ = !lean_is_exclusive(v___x_307_);
if (v_isSharedCheck_323_ == 0)
{
v___x_310_ = v___x_307_;
v_isShared_311_ = v_isSharedCheck_323_;
goto v_resetjp_309_;
}
else
{
lean_inc(v_a_308_);
lean_dec(v___x_307_);
v___x_310_ = lean_box(0);
v_isShared_311_ = v_isSharedCheck_323_;
goto v_resetjp_309_;
}
v_resetjp_309_:
{
lean_object* v___x_312_; lean_object* v___x_314_; 
lean_inc_ref(v_proof_283_);
v___x_312_ = l_Lean_Expr_app___override(v_proof_283_, v_a_308_);
if (v_isShared_301_ == 0)
{
lean_ctor_set(v___x_300_, 0, v___x_312_);
v___x_314_ = v___x_300_;
goto v_reusejp_313_;
}
else
{
lean_object* v_reuseFailAlloc_322_; 
v_reuseFailAlloc_322_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_322_, 0, v___x_312_);
v___x_314_ = v_reuseFailAlloc_322_;
goto v_reusejp_313_;
}
v_reusejp_313_:
{
lean_object* v___x_316_; 
if (v_isShared_295_ == 0)
{
lean_ctor_set(v___x_294_, 0, v___x_314_);
v___x_316_ = v___x_294_;
goto v_reusejp_315_;
}
else
{
lean_object* v_reuseFailAlloc_321_; 
v_reuseFailAlloc_321_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_321_, 0, v___x_314_);
v___x_316_ = v_reuseFailAlloc_321_;
goto v_reusejp_315_;
}
v_reusejp_315_:
{
lean_object* v___x_317_; lean_object* v___x_319_; 
v___x_317_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_317_, 0, v___x_316_);
lean_ctor_set(v___x_317_, 1, v___x_278_);
if (v_isShared_311_ == 0)
{
lean_ctor_set(v___x_310_, 0, v___x_317_);
v___x_319_ = v___x_310_;
goto v_reusejp_318_;
}
else
{
lean_object* v_reuseFailAlloc_320_; 
v_reuseFailAlloc_320_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_320_, 0, v___x_317_);
v___x_319_ = v_reuseFailAlloc_320_;
goto v_reusejp_318_;
}
v_reusejp_318_:
{
return v___x_319_;
}
}
}
}
}
else
{
lean_object* v_a_324_; lean_object* v___x_326_; uint8_t v_isShared_327_; uint8_t v_isSharedCheck_331_; 
lean_del_object(v___x_300_);
lean_del_object(v___x_294_);
v_a_324_ = lean_ctor_get(v___x_307_, 0);
v_isSharedCheck_331_ = !lean_is_exclusive(v___x_307_);
if (v_isSharedCheck_331_ == 0)
{
v___x_326_ = v___x_307_;
v_isShared_327_ = v_isSharedCheck_331_;
goto v_resetjp_325_;
}
else
{
lean_inc(v_a_324_);
lean_dec(v___x_307_);
v___x_326_ = lean_box(0);
v_isShared_327_ = v_isSharedCheck_331_;
goto v_resetjp_325_;
}
v_resetjp_325_:
{
lean_object* v___x_329_; 
if (v_isShared_327_ == 0)
{
v___x_329_ = v___x_326_;
goto v_reusejp_328_;
}
else
{
lean_object* v_reuseFailAlloc_330_; 
v_reuseFailAlloc_330_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_330_, 0, v_a_324_);
v___x_329_ = v_reuseFailAlloc_330_;
goto v_reusejp_328_;
}
v_reusejp_328_:
{
return v___x_329_;
}
}
}
}
}
else
{
lean_object* v___x_333_; lean_object* v___x_334_; 
lean_dec(v_a_297_);
lean_del_object(v___x_294_);
lean_dec(v_val_292_);
v___x_333_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__6, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__6_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__6);
v___x_334_ = lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2(v___x_333_, v___y_264_, v___y_265_, v___y_266_, v___y_267_, v___y_268_, v___y_269_);
if (lean_obj_tag(v___x_334_) == 0)
{
lean_dec_ref_known(v___x_334_, 1);
v_a_272_ = v___x_279_;
goto v___jp_271_;
}
else
{
lean_object* v_a_335_; lean_object* v___x_337_; uint8_t v_isShared_338_; uint8_t v_isSharedCheck_342_; 
v_a_335_ = lean_ctor_get(v___x_334_, 0);
v_isSharedCheck_342_ = !lean_is_exclusive(v___x_334_);
if (v_isSharedCheck_342_ == 0)
{
v___x_337_ = v___x_334_;
v_isShared_338_ = v_isSharedCheck_342_;
goto v_resetjp_336_;
}
else
{
lean_inc(v_a_335_);
lean_dec(v___x_334_);
v___x_337_ = lean_box(0);
v_isShared_338_ = v_isSharedCheck_342_;
goto v_resetjp_336_;
}
v_resetjp_336_:
{
lean_object* v___x_340_; 
if (v_isShared_338_ == 0)
{
v___x_340_ = v___x_337_;
goto v_reusejp_339_;
}
else
{
lean_object* v_reuseFailAlloc_341_; 
v_reuseFailAlloc_341_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_341_, 0, v_a_335_);
v___x_340_ = v_reuseFailAlloc_341_;
goto v_reusejp_339_;
}
v_reusejp_339_:
{
return v___x_340_;
}
}
}
}
}
else
{
lean_object* v_a_343_; lean_object* v___x_345_; uint8_t v_isShared_346_; uint8_t v_isSharedCheck_350_; 
lean_del_object(v___x_294_);
lean_dec(v_val_292_);
v_a_343_ = lean_ctor_get(v___x_296_, 0);
v_isSharedCheck_350_ = !lean_is_exclusive(v___x_296_);
if (v_isSharedCheck_350_ == 0)
{
v___x_345_ = v___x_296_;
v_isShared_346_ = v_isSharedCheck_350_;
goto v_resetjp_344_;
}
else
{
lean_inc(v_a_343_);
lean_dec(v___x_296_);
v___x_345_ = lean_box(0);
v_isShared_346_ = v_isSharedCheck_350_;
goto v_resetjp_344_;
}
v_resetjp_344_:
{
lean_object* v___x_348_; 
if (v_isShared_346_ == 0)
{
v___x_348_ = v___x_345_;
goto v_reusejp_347_;
}
else
{
lean_object* v_reuseFailAlloc_349_; 
v_reuseFailAlloc_349_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_349_, 0, v_a_343_);
v___x_348_ = v_reuseFailAlloc_349_;
goto v_reusejp_347_;
}
v_reusejp_347_:
{
return v___x_348_;
}
}
}
}
}
else
{
lean_object* v___x_352_; lean_object* v___x_353_; 
lean_dec(v_a_291_);
v___x_352_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__7, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__7_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__7);
v___x_353_ = lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2(v___x_352_, v___y_264_, v___y_265_, v___y_266_, v___y_267_, v___y_268_, v___y_269_);
if (lean_obj_tag(v___x_353_) == 0)
{
lean_dec_ref_known(v___x_353_, 1);
v_a_272_ = v___x_279_;
goto v___jp_271_;
}
else
{
lean_object* v_a_354_; lean_object* v___x_356_; uint8_t v_isShared_357_; uint8_t v_isSharedCheck_361_; 
v_a_354_ = lean_ctor_get(v___x_353_, 0);
v_isSharedCheck_361_ = !lean_is_exclusive(v___x_353_);
if (v_isSharedCheck_361_ == 0)
{
v___x_356_ = v___x_353_;
v_isShared_357_ = v_isSharedCheck_361_;
goto v_resetjp_355_;
}
else
{
lean_inc(v_a_354_);
lean_dec(v___x_353_);
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
else
{
lean_object* v_a_362_; lean_object* v___x_364_; uint8_t v_isShared_365_; uint8_t v_isSharedCheck_369_; 
v_a_362_ = lean_ctor_get(v___x_290_, 0);
v_isSharedCheck_369_ = !lean_is_exclusive(v___x_290_);
if (v_isSharedCheck_369_ == 0)
{
v___x_364_ = v___x_290_;
v_isShared_365_ = v_isSharedCheck_369_;
goto v_resetjp_363_;
}
else
{
lean_inc(v_a_362_);
lean_dec(v___x_290_);
v___x_364_ = lean_box(0);
v_isShared_365_ = v_isSharedCheck_369_;
goto v_resetjp_363_;
}
v_resetjp_363_:
{
lean_object* v___x_367_; 
if (v_isShared_365_ == 0)
{
v___x_367_ = v___x_364_;
goto v_reusejp_366_;
}
else
{
lean_object* v_reuseFailAlloc_368_; 
v_reuseFailAlloc_368_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_368_, 0, v_a_362_);
v___x_367_ = v_reuseFailAlloc_368_;
goto v_reusejp_366_;
}
v_reusejp_366_:
{
return v___x_367_;
}
}
}
}
}
}
else
{
v_a_272_ = v___x_279_;
goto v___jp_271_;
}
}
}
else
{
lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; 
v___x_370_ = lean_st_ref_get(v___y_265_);
v___x_371_ = l_Lean_instInhabitedExpr;
v___x_372_ = lean_array_get(v___x_371_, v___x_370_, v_lhs_281_);
lean_dec(v___x_370_);
v___x_373_ = l_Lean_Meta_mkEqRefl(v___x_372_, v___y_266_, v___y_267_, v___y_268_, v___y_269_);
if (lean_obj_tag(v___x_373_) == 0)
{
lean_object* v_a_374_; lean_object* v___x_376_; uint8_t v_isShared_377_; uint8_t v_isSharedCheck_385_; 
v_a_374_ = lean_ctor_get(v___x_373_, 0);
v_isSharedCheck_385_ = !lean_is_exclusive(v___x_373_);
if (v_isSharedCheck_385_ == 0)
{
v___x_376_ = v___x_373_;
v_isShared_377_ = v_isSharedCheck_385_;
goto v_resetjp_375_;
}
else
{
lean_inc(v_a_374_);
lean_dec(v___x_373_);
v___x_376_ = lean_box(0);
v_isShared_377_ = v_isSharedCheck_385_;
goto v_resetjp_375_;
}
v_resetjp_375_:
{
lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_383_; 
lean_inc_ref(v_proof_283_);
v___x_378_ = l_Lean_Expr_app___override(v_proof_283_, v_a_374_);
v___x_379_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_379_, 0, v___x_378_);
v___x_380_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_380_, 0, v___x_379_);
v___x_381_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_381_, 0, v___x_380_);
lean_ctor_set(v___x_381_, 1, v___x_278_);
if (v_isShared_377_ == 0)
{
lean_ctor_set(v___x_376_, 0, v___x_381_);
v___x_383_ = v___x_376_;
goto v_reusejp_382_;
}
else
{
lean_object* v_reuseFailAlloc_384_; 
v_reuseFailAlloc_384_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_384_, 0, v___x_381_);
v___x_383_ = v_reuseFailAlloc_384_;
goto v_reusejp_382_;
}
v_reusejp_382_:
{
return v___x_383_;
}
}
}
else
{
lean_object* v_a_386_; lean_object* v___x_388_; uint8_t v_isShared_389_; uint8_t v_isSharedCheck_393_; 
v_a_386_ = lean_ctor_get(v___x_373_, 0);
v_isSharedCheck_393_ = !lean_is_exclusive(v___x_373_);
if (v_isSharedCheck_393_ == 0)
{
v___x_388_ = v___x_373_;
v_isShared_389_ = v_isSharedCheck_393_;
goto v_resetjp_387_;
}
else
{
lean_inc(v_a_386_);
lean_dec(v___x_373_);
v___x_388_ = lean_box(0);
v_isShared_389_ = v_isSharedCheck_393_;
goto v_resetjp_387_;
}
v_resetjp_387_:
{
lean_object* v___x_391_; 
if (v_isShared_389_ == 0)
{
v___x_391_ = v___x_388_;
goto v_reusejp_390_;
}
else
{
lean_object* v_reuseFailAlloc_392_; 
v_reuseFailAlloc_392_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_392_, 0, v_a_386_);
v___x_391_ = v_reuseFailAlloc_392_;
goto v_reusejp_390_;
}
v_reusejp_390_:
{
return v___x_391_;
}
}
}
}
}
else
{
v_a_272_ = v___x_279_;
goto v___jp_271_;
}
}
v___jp_271_:
{
size_t v___x_273_; size_t v___x_274_; 
v___x_273_ = ((size_t)1ULL);
v___x_274_ = lean_usize_add(v_i_262_, v___x_273_);
lean_inc_ref(v_a_272_);
v_i_262_ = v___x_274_;
v_b_263_ = v_a_272_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___boxed(lean_object* v_scc_394_, lean_object* v_graph_395_, lean_object* v_as_396_, lean_object* v_sz_397_, lean_object* v_i_398_, lean_object* v_b_399_, lean_object* v___y_400_, lean_object* v___y_401_, lean_object* v___y_402_, lean_object* v___y_403_, lean_object* v___y_404_, lean_object* v___y_405_, lean_object* v___y_406_){
_start:
{
size_t v_sz_boxed_407_; size_t v_i_boxed_408_; lean_object* v_res_409_; 
v_sz_boxed_407_ = lean_unbox_usize(v_sz_397_);
lean_dec(v_sz_397_);
v_i_boxed_408_ = lean_unbox_usize(v_i_398_);
lean_dec(v_i_398_);
v_res_409_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3(v_scc_394_, v_graph_395_, v_as_396_, v_sz_boxed_407_, v_i_boxed_408_, v_b_399_, v___y_400_, v___y_401_, v___y_402_, v___y_403_, v___y_404_, v___y_405_);
lean_dec(v___y_405_);
lean_dec_ref(v___y_404_);
lean_dec(v___y_403_);
lean_dec_ref(v___y_402_);
lean_dec(v___y_401_);
lean_dec_ref(v___y_400_);
lean_dec_ref(v_as_396_);
lean_dec_ref(v_graph_395_);
lean_dec_ref(v_scc_394_);
return v_res_409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_findContradictionWithNe(lean_object* v_graph_410_, lean_object* v_facts_411_, lean_object* v_a_412_, lean_object* v_a_413_, lean_object* v_a_414_, lean_object* v_a_415_, lean_object* v_a_416_, lean_object* v_a_417_){
_start:
{
lean_object* v_scc_419_; lean_object* v___x_420_; lean_object* v___x_421_; size_t v_sz_422_; size_t v___x_423_; lean_object* v___x_424_; 
v_scc_419_ = lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs(v_graph_410_);
v___x_420_ = lean_box(0);
v___x_421_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__0));
v_sz_422_ = lean_array_size(v_facts_411_);
v___x_423_ = ((size_t)0ULL);
v___x_424_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3(v_scc_419_, v_graph_410_, v_facts_411_, v_sz_422_, v___x_423_, v___x_421_, v_a_412_, v_a_413_, v_a_414_, v_a_415_, v_a_416_, v_a_417_);
lean_dec_ref(v_scc_419_);
if (lean_obj_tag(v___x_424_) == 0)
{
lean_object* v_a_425_; lean_object* v___x_427_; uint8_t v_isShared_428_; uint8_t v_isSharedCheck_437_; 
v_a_425_ = lean_ctor_get(v___x_424_, 0);
v_isSharedCheck_437_ = !lean_is_exclusive(v___x_424_);
if (v_isSharedCheck_437_ == 0)
{
v___x_427_ = v___x_424_;
v_isShared_428_ = v_isSharedCheck_437_;
goto v_resetjp_426_;
}
else
{
lean_inc(v_a_425_);
lean_dec(v___x_424_);
v___x_427_ = lean_box(0);
v_isShared_428_ = v_isSharedCheck_437_;
goto v_resetjp_426_;
}
v_resetjp_426_:
{
lean_object* v_fst_429_; 
v_fst_429_ = lean_ctor_get(v_a_425_, 0);
lean_inc(v_fst_429_);
lean_dec(v_a_425_);
if (lean_obj_tag(v_fst_429_) == 0)
{
lean_object* v___x_431_; 
if (v_isShared_428_ == 0)
{
lean_ctor_set(v___x_427_, 0, v___x_420_);
v___x_431_ = v___x_427_;
goto v_reusejp_430_;
}
else
{
lean_object* v_reuseFailAlloc_432_; 
v_reuseFailAlloc_432_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_432_, 0, v___x_420_);
v___x_431_ = v_reuseFailAlloc_432_;
goto v_reusejp_430_;
}
v_reusejp_430_:
{
return v___x_431_;
}
}
else
{
lean_object* v_val_433_; lean_object* v___x_435_; 
v_val_433_ = lean_ctor_get(v_fst_429_, 0);
lean_inc(v_val_433_);
lean_dec_ref_known(v_fst_429_, 1);
if (v_isShared_428_ == 0)
{
lean_ctor_set(v___x_427_, 0, v_val_433_);
v___x_435_ = v___x_427_;
goto v_reusejp_434_;
}
else
{
lean_object* v_reuseFailAlloc_436_; 
v_reuseFailAlloc_436_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_436_, 0, v_val_433_);
v___x_435_ = v_reuseFailAlloc_436_;
goto v_reusejp_434_;
}
v_reusejp_434_:
{
return v___x_435_;
}
}
}
}
else
{
lean_object* v_a_438_; lean_object* v___x_440_; uint8_t v_isShared_441_; uint8_t v_isSharedCheck_445_; 
v_a_438_ = lean_ctor_get(v___x_424_, 0);
v_isSharedCheck_445_ = !lean_is_exclusive(v___x_424_);
if (v_isSharedCheck_445_ == 0)
{
v___x_440_ = v___x_424_;
v_isShared_441_ = v_isSharedCheck_445_;
goto v_resetjp_439_;
}
else
{
lean_inc(v_a_438_);
lean_dec(v___x_424_);
v___x_440_ = lean_box(0);
v_isShared_441_ = v_isSharedCheck_445_;
goto v_resetjp_439_;
}
v_resetjp_439_:
{
lean_object* v___x_443_; 
if (v_isShared_441_ == 0)
{
v___x_443_ = v___x_440_;
goto v_reusejp_442_;
}
else
{
lean_object* v_reuseFailAlloc_444_; 
v_reuseFailAlloc_444_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_444_, 0, v_a_438_);
v___x_443_ = v_reuseFailAlloc_444_;
goto v_reusejp_442_;
}
v_reusejp_442_:
{
return v___x_443_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_findContradictionWithNe___boxed(lean_object* v_graph_446_, lean_object* v_facts_447_, lean_object* v_a_448_, lean_object* v_a_449_, lean_object* v_a_450_, lean_object* v_a_451_, lean_object* v_a_452_, lean_object* v_a_453_, lean_object* v_a_454_){
_start:
{
lean_object* v_res_455_; 
v_res_455_ = lp_mathlib_Mathlib_Tactic_Order_findContradictionWithNe(v_graph_446_, v_facts_447_, v_a_448_, v_a_449_, v_a_450_, v_a_451_, v_a_452_, v_a_453_);
lean_dec(v_a_453_);
lean_dec_ref(v_a_452_);
lean_dec(v_a_451_);
lean_dec_ref(v_a_450_);
lean_dec(v_a_449_);
lean_dec_ref(v_a_448_);
lean_dec_ref(v_facts_447_);
lean_dec_ref(v_graph_446_);
return v_res_455_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0(lean_object* v_00_u03b2_456_, lean_object* v_m_457_, lean_object* v_a_458_){
_start:
{
uint8_t v___x_459_; 
v___x_459_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0___redArg(v_m_457_, v_a_458_);
return v___x_459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0___boxed(lean_object* v_00_u03b2_460_, lean_object* v_m_461_, lean_object* v_a_462_){
_start:
{
uint8_t v_res_463_; lean_object* v_r_464_; 
v_res_463_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0(v_00_u03b2_460_, v_m_461_, v_a_462_);
lean_dec(v_a_462_);
lean_dec_ref(v_m_461_);
v_r_464_ = lean_box(v_res_463_);
return v_r_464_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0_spec__0(lean_object* v_00_u03b2_465_, lean_object* v_a_466_, lean_object* v_x_467_){
_start:
{
uint8_t v___x_468_; 
v___x_468_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0_spec__0___redArg(v_a_466_, v_x_467_);
return v___x_468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0_spec__0___boxed(lean_object* v_00_u03b2_469_, lean_object* v_a_470_, lean_object* v_x_471_){
_start:
{
uint8_t v_res_472_; lean_object* v_r_473_; 
v_res_472_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0_spec__0(v_00_u03b2_469_, v_a_470_, v_x_471_);
lean_dec(v_x_471_);
lean_dec(v_a_470_);
v_r_473_ = lean_box(v_res_472_);
return v_r_473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNle_spec__0___redArg(lean_object* v_g_474_, lean_object* v_as_475_, size_t v_sz_476_, size_t v_i_477_, lean_object* v_b_478_, lean_object* v___y_479_, lean_object* v___y_480_, lean_object* v___y_481_, lean_object* v___y_482_, lean_object* v___y_483_){
_start:
{
lean_object* v_a_486_; uint8_t v___x_490_; 
v___x_490_ = lean_usize_dec_lt(v_i_477_, v_sz_476_);
if (v___x_490_ == 0)
{
lean_object* v___x_491_; 
v___x_491_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_491_, 0, v_b_478_);
return v___x_491_;
}
else
{
lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v_a_494_; 
lean_dec_ref(v_b_478_);
v___x_492_ = lean_box(0);
v___x_493_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__0));
v_a_494_ = lean_array_uget_borrowed(v_as_475_, v_i_477_);
if (lean_obj_tag(v_a_494_) == 3)
{
lean_object* v_lhs_495_; lean_object* v_rhs_496_; lean_object* v_proof_497_; lean_object* v___x_498_; 
v_lhs_495_ = lean_ctor_get(v_a_494_, 0);
v_rhs_496_ = lean_ctor_get(v_a_494_, 1);
v_proof_497_ = lean_ctor_get(v_a_494_, 2);
lean_inc(v_lhs_495_);
v___x_498_ = lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProof___redArg(v_g_474_, v_lhs_495_, v_rhs_496_, v___y_479_, v___y_480_, v___y_481_, v___y_482_, v___y_483_);
if (lean_obj_tag(v___x_498_) == 0)
{
lean_object* v_a_499_; lean_object* v___x_501_; uint8_t v_isShared_502_; uint8_t v_isSharedCheck_517_; 
v_a_499_ = lean_ctor_get(v___x_498_, 0);
v_isSharedCheck_517_ = !lean_is_exclusive(v___x_498_);
if (v_isSharedCheck_517_ == 0)
{
v___x_501_ = v___x_498_;
v_isShared_502_ = v_isSharedCheck_517_;
goto v_resetjp_500_;
}
else
{
lean_inc(v_a_499_);
lean_dec(v___x_498_);
v___x_501_ = lean_box(0);
v_isShared_502_ = v_isSharedCheck_517_;
goto v_resetjp_500_;
}
v_resetjp_500_:
{
if (lean_obj_tag(v_a_499_) == 1)
{
lean_object* v_val_503_; lean_object* v___x_505_; uint8_t v_isShared_506_; uint8_t v_isSharedCheck_516_; 
v_val_503_ = lean_ctor_get(v_a_499_, 0);
v_isSharedCheck_516_ = !lean_is_exclusive(v_a_499_);
if (v_isSharedCheck_516_ == 0)
{
v___x_505_ = v_a_499_;
v_isShared_506_ = v_isSharedCheck_516_;
goto v_resetjp_504_;
}
else
{
lean_inc(v_val_503_);
lean_dec(v_a_499_);
v___x_505_ = lean_box(0);
v_isShared_506_ = v_isSharedCheck_516_;
goto v_resetjp_504_;
}
v_resetjp_504_:
{
lean_object* v___x_507_; lean_object* v___x_509_; 
lean_inc_ref(v_proof_497_);
v___x_507_ = l_Lean_Expr_app___override(v_proof_497_, v_val_503_);
if (v_isShared_506_ == 0)
{
lean_ctor_set(v___x_505_, 0, v___x_507_);
v___x_509_ = v___x_505_;
goto v_reusejp_508_;
}
else
{
lean_object* v_reuseFailAlloc_515_; 
v_reuseFailAlloc_515_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_515_, 0, v___x_507_);
v___x_509_ = v_reuseFailAlloc_515_;
goto v_reusejp_508_;
}
v_reusejp_508_:
{
lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_513_; 
v___x_510_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_510_, 0, v___x_509_);
v___x_511_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_511_, 0, v___x_510_);
lean_ctor_set(v___x_511_, 1, v___x_492_);
if (v_isShared_502_ == 0)
{
lean_ctor_set(v___x_501_, 0, v___x_511_);
v___x_513_ = v___x_501_;
goto v_reusejp_512_;
}
else
{
lean_object* v_reuseFailAlloc_514_; 
v_reuseFailAlloc_514_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_514_, 0, v___x_511_);
v___x_513_ = v_reuseFailAlloc_514_;
goto v_reusejp_512_;
}
v_reusejp_512_:
{
return v___x_513_;
}
}
}
}
else
{
lean_del_object(v___x_501_);
lean_dec(v_a_499_);
v_a_486_ = v___x_493_;
goto v___jp_485_;
}
}
}
else
{
lean_object* v_a_518_; lean_object* v___x_520_; uint8_t v_isShared_521_; uint8_t v_isSharedCheck_525_; 
v_a_518_ = lean_ctor_get(v___x_498_, 0);
v_isSharedCheck_525_ = !lean_is_exclusive(v___x_498_);
if (v_isSharedCheck_525_ == 0)
{
v___x_520_ = v___x_498_;
v_isShared_521_ = v_isSharedCheck_525_;
goto v_resetjp_519_;
}
else
{
lean_inc(v_a_518_);
lean_dec(v___x_498_);
v___x_520_ = lean_box(0);
v_isShared_521_ = v_isSharedCheck_525_;
goto v_resetjp_519_;
}
v_resetjp_519_:
{
lean_object* v___x_523_; 
if (v_isShared_521_ == 0)
{
v___x_523_ = v___x_520_;
goto v_reusejp_522_;
}
else
{
lean_object* v_reuseFailAlloc_524_; 
v_reuseFailAlloc_524_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_524_, 0, v_a_518_);
v___x_523_ = v_reuseFailAlloc_524_;
goto v_reusejp_522_;
}
v_reusejp_522_:
{
return v___x_523_;
}
}
}
}
else
{
v_a_486_ = v___x_493_;
goto v___jp_485_;
}
}
v___jp_485_:
{
size_t v___x_487_; size_t v___x_488_; 
v___x_487_ = ((size_t)1ULL);
v___x_488_ = lean_usize_add(v_i_477_, v___x_487_);
lean_inc_ref(v_a_486_);
v_i_477_ = v___x_488_;
v_b_478_ = v_a_486_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNle_spec__0___redArg___boxed(lean_object* v_g_526_, lean_object* v_as_527_, lean_object* v_sz_528_, lean_object* v_i_529_, lean_object* v_b_530_, lean_object* v___y_531_, lean_object* v___y_532_, lean_object* v___y_533_, lean_object* v___y_534_, lean_object* v___y_535_, lean_object* v___y_536_){
_start:
{
size_t v_sz_boxed_537_; size_t v_i_boxed_538_; lean_object* v_res_539_; 
v_sz_boxed_537_ = lean_unbox_usize(v_sz_528_);
lean_dec(v_sz_528_);
v_i_boxed_538_ = lean_unbox_usize(v_i_529_);
lean_dec(v_i_529_);
v_res_539_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNle_spec__0___redArg(v_g_526_, v_as_527_, v_sz_boxed_537_, v_i_boxed_538_, v_b_530_, v___y_531_, v___y_532_, v___y_533_, v___y_534_, v___y_535_);
lean_dec(v___y_535_);
lean_dec_ref(v___y_534_);
lean_dec(v___y_533_);
lean_dec_ref(v___y_532_);
lean_dec(v___y_531_);
lean_dec_ref(v_as_527_);
lean_dec_ref(v_g_526_);
return v_res_539_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_findContradictionWithNle(lean_object* v_g_540_, lean_object* v_facts_541_, lean_object* v_a_542_, lean_object* v_a_543_, lean_object* v_a_544_, lean_object* v_a_545_, lean_object* v_a_546_, lean_object* v_a_547_){
_start:
{
lean_object* v___x_549_; lean_object* v___x_550_; size_t v_sz_551_; size_t v___x_552_; lean_object* v___x_553_; 
v___x_549_ = lean_box(0);
v___x_550_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__0));
v_sz_551_ = lean_array_size(v_facts_541_);
v___x_552_ = ((size_t)0ULL);
v___x_553_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNle_spec__0___redArg(v_g_540_, v_facts_541_, v_sz_551_, v___x_552_, v___x_550_, v_a_543_, v_a_544_, v_a_545_, v_a_546_, v_a_547_);
if (lean_obj_tag(v___x_553_) == 0)
{
lean_object* v_a_554_; lean_object* v___x_556_; uint8_t v_isShared_557_; uint8_t v_isSharedCheck_566_; 
v_a_554_ = lean_ctor_get(v___x_553_, 0);
v_isSharedCheck_566_ = !lean_is_exclusive(v___x_553_);
if (v_isSharedCheck_566_ == 0)
{
v___x_556_ = v___x_553_;
v_isShared_557_ = v_isSharedCheck_566_;
goto v_resetjp_555_;
}
else
{
lean_inc(v_a_554_);
lean_dec(v___x_553_);
v___x_556_ = lean_box(0);
v_isShared_557_ = v_isSharedCheck_566_;
goto v_resetjp_555_;
}
v_resetjp_555_:
{
lean_object* v_fst_558_; 
v_fst_558_ = lean_ctor_get(v_a_554_, 0);
lean_inc(v_fst_558_);
lean_dec(v_a_554_);
if (lean_obj_tag(v_fst_558_) == 0)
{
lean_object* v___x_560_; 
if (v_isShared_557_ == 0)
{
lean_ctor_set(v___x_556_, 0, v___x_549_);
v___x_560_ = v___x_556_;
goto v_reusejp_559_;
}
else
{
lean_object* v_reuseFailAlloc_561_; 
v_reuseFailAlloc_561_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_561_, 0, v___x_549_);
v___x_560_ = v_reuseFailAlloc_561_;
goto v_reusejp_559_;
}
v_reusejp_559_:
{
return v___x_560_;
}
}
else
{
lean_object* v_val_562_; lean_object* v___x_564_; 
v_val_562_ = lean_ctor_get(v_fst_558_, 0);
lean_inc(v_val_562_);
lean_dec_ref_known(v_fst_558_, 1);
if (v_isShared_557_ == 0)
{
lean_ctor_set(v___x_556_, 0, v_val_562_);
v___x_564_ = v___x_556_;
goto v_reusejp_563_;
}
else
{
lean_object* v_reuseFailAlloc_565_; 
v_reuseFailAlloc_565_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_565_, 0, v_val_562_);
v___x_564_ = v_reuseFailAlloc_565_;
goto v_reusejp_563_;
}
v_reusejp_563_:
{
return v___x_564_;
}
}
}
}
else
{
lean_object* v_a_567_; lean_object* v___x_569_; uint8_t v_isShared_570_; uint8_t v_isSharedCheck_574_; 
v_a_567_ = lean_ctor_get(v___x_553_, 0);
v_isSharedCheck_574_ = !lean_is_exclusive(v___x_553_);
if (v_isSharedCheck_574_ == 0)
{
v___x_569_ = v___x_553_;
v_isShared_570_ = v_isSharedCheck_574_;
goto v_resetjp_568_;
}
else
{
lean_inc(v_a_567_);
lean_dec(v___x_553_);
v___x_569_ = lean_box(0);
v_isShared_570_ = v_isSharedCheck_574_;
goto v_resetjp_568_;
}
v_resetjp_568_:
{
lean_object* v___x_572_; 
if (v_isShared_570_ == 0)
{
v___x_572_ = v___x_569_;
goto v_reusejp_571_;
}
else
{
lean_object* v_reuseFailAlloc_573_; 
v_reuseFailAlloc_573_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_573_, 0, v_a_567_);
v___x_572_ = v_reuseFailAlloc_573_;
goto v_reusejp_571_;
}
v_reusejp_571_:
{
return v___x_572_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_findContradictionWithNle___boxed(lean_object* v_g_575_, lean_object* v_facts_576_, lean_object* v_a_577_, lean_object* v_a_578_, lean_object* v_a_579_, lean_object* v_a_580_, lean_object* v_a_581_, lean_object* v_a_582_, lean_object* v_a_583_){
_start:
{
lean_object* v_res_584_; 
v_res_584_ = lp_mathlib_Mathlib_Tactic_Order_findContradictionWithNle(v_g_575_, v_facts_576_, v_a_577_, v_a_578_, v_a_579_, v_a_580_, v_a_581_, v_a_582_);
lean_dec(v_a_582_);
lean_dec_ref(v_a_581_);
lean_dec(v_a_580_);
lean_dec_ref(v_a_579_);
lean_dec(v_a_578_);
lean_dec_ref(v_a_577_);
lean_dec_ref(v_facts_576_);
lean_dec_ref(v_g_575_);
return v_res_584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNle_spec__0(lean_object* v_g_585_, lean_object* v_as_586_, size_t v_sz_587_, size_t v_i_588_, lean_object* v_b_589_, lean_object* v___y_590_, lean_object* v___y_591_, lean_object* v___y_592_, lean_object* v___y_593_, lean_object* v___y_594_, lean_object* v___y_595_){
_start:
{
lean_object* v___x_597_; 
v___x_597_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNle_spec__0___redArg(v_g_585_, v_as_586_, v_sz_587_, v_i_588_, v_b_589_, v___y_591_, v___y_592_, v___y_593_, v___y_594_, v___y_595_);
return v___x_597_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNle_spec__0___boxed(lean_object* v_g_598_, lean_object* v_as_599_, lean_object* v_sz_600_, lean_object* v_i_601_, lean_object* v_b_602_, lean_object* v___y_603_, lean_object* v___y_604_, lean_object* v___y_605_, lean_object* v___y_606_, lean_object* v___y_607_, lean_object* v___y_608_, lean_object* v___y_609_){
_start:
{
size_t v_sz_boxed_610_; size_t v_i_boxed_611_; lean_object* v_res_612_; 
v_sz_boxed_610_ = lean_unbox_usize(v_sz_600_);
lean_dec(v_sz_600_);
v_i_boxed_611_ = lean_unbox_usize(v_i_601_);
lean_dec(v_i_601_);
v_res_612_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNle_spec__0(v_g_598_, v_as_599_, v_sz_boxed_610_, v_i_boxed_611_, v_b_602_, v___y_603_, v___y_604_, v___y_605_, v___y_606_, v___y_607_, v___y_608_);
lean_dec(v___y_608_);
lean_dec_ref(v___y_607_);
lean_dec(v___y_606_);
lean_dec_ref(v___y_605_);
lean_dec(v___y_604_);
lean_dec_ref(v___y_603_);
lean_dec_ref(v_as_599_);
lean_dec_ref(v_g_598_);
return v_res_612_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__4(void){
_start:
{
lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; 
v___x_621_ = ((lean_object*)(lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__3));
v___x_622_ = lean_unsigned_to_nat(46u);
v___x_623_ = lean_unsigned_to_nat(208u);
v___x_624_ = ((lean_object*)(lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__2));
v___x_625_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__3));
v___x_626_ = l_mkPanicMessageWithDecl(v___x_625_, v___x_624_, v___x_623_, v___x_622_, v___x_621_);
return v___x_626_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg(lean_object* v___y_627_, lean_object* v_range_628_, lean_object* v_b_629_, lean_object* v_i_630_, lean_object* v___y_631_, lean_object* v___y_632_, lean_object* v___y_633_, lean_object* v___y_634_, lean_object* v___y_635_, lean_object* v___y_636_){
_start:
{
lean_object* v_stop_638_; lean_object* v_step_639_; lean_object* v_a_641_; uint8_t v___x_644_; 
v_stop_638_ = lean_ctor_get(v_range_628_, 1);
v_step_639_ = lean_ctor_get(v_range_628_, 2);
v___x_644_ = lean_nat_dec_lt(v_i_630_, v_stop_638_);
if (v___x_644_ == 0)
{
lean_object* v___x_645_; 
lean_dec(v_i_630_);
v___x_645_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_645_, 0, v_b_629_);
return v___x_645_;
}
else
{
lean_object* v_snd_646_; lean_object* v_fst_647_; lean_object* v___x_649_; uint8_t v_isShared_650_; uint8_t v_isSharedCheck_734_; 
v_snd_646_ = lean_ctor_get(v_b_629_, 1);
v_fst_647_ = lean_ctor_get(v_b_629_, 0);
v_isSharedCheck_734_ = !lean_is_exclusive(v_b_629_);
if (v_isSharedCheck_734_ == 0)
{
v___x_649_ = v_b_629_;
v_isShared_650_ = v_isSharedCheck_734_;
goto v_resetjp_648_;
}
else
{
lean_inc(v_snd_646_);
lean_inc(v_fst_647_);
lean_dec(v_b_629_);
v___x_649_ = lean_box(0);
v_isShared_650_ = v_isSharedCheck_734_;
goto v_resetjp_648_;
}
v_resetjp_648_:
{
lean_object* v_fst_651_; lean_object* v_snd_652_; lean_object* v___x_654_; uint8_t v_isShared_655_; uint8_t v_isSharedCheck_733_; 
v_fst_651_ = lean_ctor_get(v_snd_646_, 0);
v_snd_652_ = lean_ctor_get(v_snd_646_, 1);
v_isSharedCheck_733_ = !lean_is_exclusive(v_snd_646_);
if (v_isSharedCheck_733_ == 0)
{
v___x_654_ = v_snd_646_;
v_isShared_655_ = v_isSharedCheck_733_;
goto v_resetjp_653_;
}
else
{
lean_inc(v_snd_652_);
lean_inc(v_fst_651_);
lean_dec(v_snd_646_);
v___x_654_ = lean_box(0);
v_isShared_655_ = v_isSharedCheck_733_;
goto v_resetjp_653_;
}
v_resetjp_653_:
{
lean_object* v___x_656_; uint8_t v___x_657_; 
v___x_656_ = lean_array_fget_borrowed(v_fst_647_, v_i_630_);
v___x_657_ = lean_unbox(v___x_656_);
if (v___x_657_ == 0)
{
lean_object* v___x_658_; 
v___x_658_ = lean_array_fget(v___y_627_, v_i_630_);
if (lean_obj_tag(v___x_658_) == 5)
{
lean_object* v_lhs_659_; lean_object* v_rhs_660_; lean_object* v_proof_661_; lean_object* v___x_663_; uint8_t v_isShared_664_; uint8_t v_isSharedCheck_710_; 
v_lhs_659_ = lean_ctor_get(v___x_658_, 0);
v_rhs_660_ = lean_ctor_get(v___x_658_, 1);
v_proof_661_ = lean_ctor_get(v___x_658_, 2);
v_isSharedCheck_710_ = !lean_is_exclusive(v___x_658_);
if (v_isSharedCheck_710_ == 0)
{
v___x_663_ = v___x_658_;
v_isShared_664_ = v_isSharedCheck_710_;
goto v_resetjp_662_;
}
else
{
lean_inc(v_proof_661_);
lean_inc(v_rhs_660_);
lean_inc(v_lhs_659_);
lean_dec(v___x_658_);
v___x_663_ = lean_box(0);
v_isShared_664_ = v_isSharedCheck_710_;
goto v_resetjp_662_;
}
v_resetjp_662_:
{
lean_object* v___x_665_; 
lean_inc(v_lhs_659_);
v___x_665_ = lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProof___redArg(v_fst_651_, v_lhs_659_, v_rhs_660_, v___y_632_, v___y_633_, v___y_634_, v___y_635_, v___y_636_);
if (lean_obj_tag(v___x_665_) == 0)
{
lean_object* v_a_666_; 
v_a_666_ = lean_ctor_get(v___x_665_, 0);
lean_inc(v_a_666_);
lean_dec_ref_known(v___x_665_, 1);
if (lean_obj_tag(v_a_666_) == 1)
{
lean_object* v_val_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v___x_673_; 
lean_dec(v_snd_652_);
v_val_667_ = lean_ctor_get(v_a_666_, 0);
lean_inc(v_val_667_);
lean_dec_ref_known(v_a_666_, 1);
v___x_668_ = ((lean_object*)(lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__1));
v___x_669_ = lean_unsigned_to_nat(2u);
v___x_670_ = lean_mk_empty_array_with_capacity(v___x_669_);
v___x_671_ = lean_array_push(v___x_670_, v_proof_661_);
v___x_672_ = lean_array_push(v___x_671_, v_val_667_);
v___x_673_ = l_Lean_Meta_mkAppM(v___x_668_, v___x_672_, v___y_633_, v___y_634_, v___y_635_, v___y_636_);
if (lean_obj_tag(v___x_673_) == 0)
{
lean_object* v_a_674_; lean_object* v___x_676_; 
v_a_674_ = lean_ctor_get(v___x_673_, 0);
lean_inc(v_a_674_);
lean_dec_ref_known(v___x_673_, 1);
if (v_isShared_664_ == 0)
{
lean_ctor_set_tag(v___x_663_, 0);
lean_ctor_set(v___x_663_, 2, v_a_674_);
lean_ctor_set(v___x_663_, 1, v_lhs_659_);
lean_ctor_set(v___x_663_, 0, v_rhs_660_);
v___x_676_ = v___x_663_;
goto v_reusejp_675_;
}
else
{
lean_object* v_reuseFailAlloc_687_; 
v_reuseFailAlloc_687_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_687_, 0, v_rhs_660_);
lean_ctor_set(v_reuseFailAlloc_687_, 1, v_lhs_659_);
lean_ctor_set(v_reuseFailAlloc_687_, 2, v_a_674_);
v___x_676_ = v_reuseFailAlloc_687_;
goto v_reusejp_675_;
}
v_reusejp_675_:
{
lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_682_; 
v___x_677_ = lp_mathlib_Mathlib_Tactic_Order_Graph_addEdge(v_fst_651_, v___x_676_);
v___x_678_ = lean_box(v___x_644_);
v___x_679_ = lean_array_fset(v_fst_647_, v_i_630_, v___x_678_);
v___x_680_ = lean_box(v___x_644_);
if (v_isShared_655_ == 0)
{
lean_ctor_set(v___x_654_, 1, v___x_680_);
lean_ctor_set(v___x_654_, 0, v___x_677_);
v___x_682_ = v___x_654_;
goto v_reusejp_681_;
}
else
{
lean_object* v_reuseFailAlloc_686_; 
v_reuseFailAlloc_686_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_686_, 0, v___x_677_);
lean_ctor_set(v_reuseFailAlloc_686_, 1, v___x_680_);
v___x_682_ = v_reuseFailAlloc_686_;
goto v_reusejp_681_;
}
v_reusejp_681_:
{
lean_object* v___x_684_; 
if (v_isShared_650_ == 0)
{
lean_ctor_set(v___x_649_, 1, v___x_682_);
lean_ctor_set(v___x_649_, 0, v___x_679_);
v___x_684_ = v___x_649_;
goto v_reusejp_683_;
}
else
{
lean_object* v_reuseFailAlloc_685_; 
v_reuseFailAlloc_685_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_685_, 0, v___x_679_);
lean_ctor_set(v_reuseFailAlloc_685_, 1, v___x_682_);
v___x_684_ = v_reuseFailAlloc_685_;
goto v_reusejp_683_;
}
v_reusejp_683_:
{
v_a_641_ = v___x_684_;
goto v___jp_640_;
}
}
}
}
else
{
lean_object* v_a_688_; lean_object* v___x_690_; uint8_t v_isShared_691_; uint8_t v_isSharedCheck_695_; 
lean_del_object(v___x_663_);
lean_dec(v_rhs_660_);
lean_dec(v_lhs_659_);
lean_del_object(v___x_654_);
lean_dec(v_fst_651_);
lean_del_object(v___x_649_);
lean_dec(v_fst_647_);
lean_dec(v_i_630_);
v_a_688_ = lean_ctor_get(v___x_673_, 0);
v_isSharedCheck_695_ = !lean_is_exclusive(v___x_673_);
if (v_isSharedCheck_695_ == 0)
{
v___x_690_ = v___x_673_;
v_isShared_691_ = v_isSharedCheck_695_;
goto v_resetjp_689_;
}
else
{
lean_inc(v_a_688_);
lean_dec(v___x_673_);
v___x_690_ = lean_box(0);
v_isShared_691_ = v_isSharedCheck_695_;
goto v_resetjp_689_;
}
v_resetjp_689_:
{
lean_object* v___x_693_; 
if (v_isShared_691_ == 0)
{
v___x_693_ = v___x_690_;
goto v_reusejp_692_;
}
else
{
lean_object* v_reuseFailAlloc_694_; 
v_reuseFailAlloc_694_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_694_, 0, v_a_688_);
v___x_693_ = v_reuseFailAlloc_694_;
goto v_reusejp_692_;
}
v_reusejp_692_:
{
return v___x_693_;
}
}
}
}
else
{
lean_object* v___x_697_; 
lean_dec(v_a_666_);
lean_del_object(v___x_663_);
lean_dec_ref(v_proof_661_);
lean_dec(v_rhs_660_);
lean_dec(v_lhs_659_);
if (v_isShared_655_ == 0)
{
v___x_697_ = v___x_654_;
goto v_reusejp_696_;
}
else
{
lean_object* v_reuseFailAlloc_701_; 
v_reuseFailAlloc_701_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_701_, 0, v_fst_651_);
lean_ctor_set(v_reuseFailAlloc_701_, 1, v_snd_652_);
v___x_697_ = v_reuseFailAlloc_701_;
goto v_reusejp_696_;
}
v_reusejp_696_:
{
lean_object* v___x_699_; 
if (v_isShared_650_ == 0)
{
lean_ctor_set(v___x_649_, 1, v___x_697_);
v___x_699_ = v___x_649_;
goto v_reusejp_698_;
}
else
{
lean_object* v_reuseFailAlloc_700_; 
v_reuseFailAlloc_700_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_700_, 0, v_fst_647_);
lean_ctor_set(v_reuseFailAlloc_700_, 1, v___x_697_);
v___x_699_ = v_reuseFailAlloc_700_;
goto v_reusejp_698_;
}
v_reusejp_698_:
{
v_a_641_ = v___x_699_;
goto v___jp_640_;
}
}
}
}
else
{
lean_object* v_a_702_; lean_object* v___x_704_; uint8_t v_isShared_705_; uint8_t v_isSharedCheck_709_; 
lean_del_object(v___x_663_);
lean_dec_ref(v_proof_661_);
lean_dec(v_rhs_660_);
lean_dec(v_lhs_659_);
lean_del_object(v___x_654_);
lean_dec(v_snd_652_);
lean_dec(v_fst_651_);
lean_del_object(v___x_649_);
lean_dec(v_fst_647_);
lean_dec(v_i_630_);
v_a_702_ = lean_ctor_get(v___x_665_, 0);
v_isSharedCheck_709_ = !lean_is_exclusive(v___x_665_);
if (v_isSharedCheck_709_ == 0)
{
v___x_704_ = v___x_665_;
v_isShared_705_ = v_isSharedCheck_709_;
goto v_resetjp_703_;
}
else
{
lean_inc(v_a_702_);
lean_dec(v___x_665_);
v___x_704_ = lean_box(0);
v_isShared_705_ = v_isSharedCheck_709_;
goto v_resetjp_703_;
}
v_resetjp_703_:
{
lean_object* v___x_707_; 
if (v_isShared_705_ == 0)
{
v___x_707_ = v___x_704_;
goto v_reusejp_706_;
}
else
{
lean_object* v_reuseFailAlloc_708_; 
v_reuseFailAlloc_708_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_708_, 0, v_a_702_);
v___x_707_ = v_reuseFailAlloc_708_;
goto v_reusejp_706_;
}
v_reusejp_706_:
{
return v___x_707_;
}
}
}
}
}
else
{
lean_object* v___x_711_; lean_object* v___x_712_; 
lean_dec(v___x_658_);
v___x_711_ = lean_obj_once(&lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__4, &lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__4_once, _init_lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__4);
v___x_712_ = lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2(v___x_711_, v___y_631_, v___y_632_, v___y_633_, v___y_634_, v___y_635_, v___y_636_);
if (lean_obj_tag(v___x_712_) == 0)
{
lean_object* v___x_714_; 
lean_dec_ref_known(v___x_712_, 1);
if (v_isShared_655_ == 0)
{
v___x_714_ = v___x_654_;
goto v_reusejp_713_;
}
else
{
lean_object* v_reuseFailAlloc_718_; 
v_reuseFailAlloc_718_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_718_, 0, v_fst_651_);
lean_ctor_set(v_reuseFailAlloc_718_, 1, v_snd_652_);
v___x_714_ = v_reuseFailAlloc_718_;
goto v_reusejp_713_;
}
v_reusejp_713_:
{
lean_object* v___x_716_; 
if (v_isShared_650_ == 0)
{
lean_ctor_set(v___x_649_, 1, v___x_714_);
v___x_716_ = v___x_649_;
goto v_reusejp_715_;
}
else
{
lean_object* v_reuseFailAlloc_717_; 
v_reuseFailAlloc_717_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_717_, 0, v_fst_647_);
lean_ctor_set(v_reuseFailAlloc_717_, 1, v___x_714_);
v___x_716_ = v_reuseFailAlloc_717_;
goto v_reusejp_715_;
}
v_reusejp_715_:
{
v_a_641_ = v___x_716_;
goto v___jp_640_;
}
}
}
else
{
lean_object* v_a_719_; lean_object* v___x_721_; uint8_t v_isShared_722_; uint8_t v_isSharedCheck_726_; 
lean_del_object(v___x_654_);
lean_dec(v_snd_652_);
lean_dec(v_fst_651_);
lean_del_object(v___x_649_);
lean_dec(v_fst_647_);
lean_dec(v_i_630_);
v_a_719_ = lean_ctor_get(v___x_712_, 0);
v_isSharedCheck_726_ = !lean_is_exclusive(v___x_712_);
if (v_isSharedCheck_726_ == 0)
{
v___x_721_ = v___x_712_;
v_isShared_722_ = v_isSharedCheck_726_;
goto v_resetjp_720_;
}
else
{
lean_inc(v_a_719_);
lean_dec(v___x_712_);
v___x_721_ = lean_box(0);
v_isShared_722_ = v_isSharedCheck_726_;
goto v_resetjp_720_;
}
v_resetjp_720_:
{
lean_object* v___x_724_; 
if (v_isShared_722_ == 0)
{
v___x_724_ = v___x_721_;
goto v_reusejp_723_;
}
else
{
lean_object* v_reuseFailAlloc_725_; 
v_reuseFailAlloc_725_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_725_, 0, v_a_719_);
v___x_724_ = v_reuseFailAlloc_725_;
goto v_reusejp_723_;
}
v_reusejp_723_:
{
return v___x_724_;
}
}
}
}
}
else
{
lean_object* v___x_728_; 
if (v_isShared_655_ == 0)
{
v___x_728_ = v___x_654_;
goto v_reusejp_727_;
}
else
{
lean_object* v_reuseFailAlloc_732_; 
v_reuseFailAlloc_732_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_732_, 0, v_fst_651_);
lean_ctor_set(v_reuseFailAlloc_732_, 1, v_snd_652_);
v___x_728_ = v_reuseFailAlloc_732_;
goto v_reusejp_727_;
}
v_reusejp_727_:
{
lean_object* v___x_730_; 
if (v_isShared_650_ == 0)
{
lean_ctor_set(v___x_649_, 1, v___x_728_);
v___x_730_ = v___x_649_;
goto v_reusejp_729_;
}
else
{
lean_object* v_reuseFailAlloc_731_; 
v_reuseFailAlloc_731_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_731_, 0, v_fst_647_);
lean_ctor_set(v_reuseFailAlloc_731_, 1, v___x_728_);
v___x_730_ = v_reuseFailAlloc_731_;
goto v_reusejp_729_;
}
v_reusejp_729_:
{
v_a_641_ = v___x_730_;
goto v___jp_640_;
}
}
}
}
}
}
v___jp_640_:
{
lean_object* v___x_642_; 
v___x_642_ = lean_nat_add(v_i_630_, v_step_639_);
lean_dec(v_i_630_);
v_b_629_ = v_a_641_;
v_i_630_ = v___x_642_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___boxed(lean_object* v___y_735_, lean_object* v_range_736_, lean_object* v_b_737_, lean_object* v_i_738_, lean_object* v___y_739_, lean_object* v___y_740_, lean_object* v___y_741_, lean_object* v___y_742_, lean_object* v___y_743_, lean_object* v___y_744_, lean_object* v___y_745_){
_start:
{
lean_object* v_res_746_; 
v_res_746_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg(v___y_735_, v_range_736_, v_b_737_, v_i_738_, v___y_739_, v___y_740_, v___y_741_, v___y_742_, v___y_743_, v___y_744_);
lean_dec(v___y_744_);
lean_dec_ref(v___y_743_);
lean_dec(v___y_742_);
lean_dec_ref(v___y_741_);
lean_dec(v___y_740_);
lean_dec_ref(v___y_739_);
lean_dec_ref(v_range_736_);
lean_dec_ref(v___y_735_);
return v_res_746_;
}
}
static lean_object* _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__5(void){
_start:
{
lean_object* v___x_754_; lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; 
v___x_754_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__4));
v___x_755_ = lean_unsigned_to_nat(15u);
v___x_756_ = lean_unsigned_to_nat(228u);
v___x_757_ = ((lean_object*)(lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg___closed__2));
v___x_758_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__3___closed__3));
v___x_759_ = l_mkPanicMessageWithDecl(v___x_758_, v___x_757_, v___x_756_, v___x_755_, v___x_754_);
return v___x_759_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0(lean_object* v_a_760_, lean_object* v_a_761_, lean_object* v_a_762_, lean_object* v___y_763_, lean_object* v___y_764_, lean_object* v___y_765_, lean_object* v___y_766_, lean_object* v___y_767_, lean_object* v___y_768_){
_start:
{
if (lean_obj_tag(v_a_761_) == 0)
{
lean_object* v___x_770_; lean_object* v___x_771_; 
lean_dec_ref(v_a_760_);
v___x_770_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_770_, 0, v_a_762_);
v___x_771_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_771_, 0, v___x_770_);
return v___x_771_;
}
else
{
switch(lean_obj_tag(v_a_760_))
{
case 9:
{
lean_object* v_key_772_; lean_object* v_tail_773_; lean_object* v___x_775_; uint8_t v_isShared_776_; uint8_t v_isSharedCheck_855_; 
v_key_772_ = lean_ctor_get(v_a_761_, 0);
v_tail_773_ = lean_ctor_get(v_a_761_, 2);
v_isSharedCheck_855_ = !lean_is_exclusive(v_a_761_);
if (v_isSharedCheck_855_ == 0)
{
lean_object* v_unused_856_; 
v_unused_856_ = lean_ctor_get(v_a_761_, 1);
lean_dec(v_unused_856_);
v___x_775_ = v_a_761_;
v_isShared_776_ = v_isSharedCheck_855_;
goto v_resetjp_774_;
}
else
{
lean_inc(v_tail_773_);
lean_inc(v_key_772_);
lean_dec(v_a_761_);
v___x_775_ = lean_box(0);
v_isShared_776_ = v_isSharedCheck_855_;
goto v_resetjp_774_;
}
v_resetjp_774_:
{
lean_object* v_fst_777_; lean_object* v_snd_778_; lean_object* v___x_780_; uint8_t v_isShared_781_; uint8_t v_isSharedCheck_854_; 
v_fst_777_ = lean_ctor_get(v_a_762_, 0);
v_snd_778_ = lean_ctor_get(v_a_762_, 1);
v_isSharedCheck_854_ = !lean_is_exclusive(v_a_762_);
if (v_isSharedCheck_854_ == 0)
{
v___x_780_ = v_a_762_;
v_isShared_781_ = v_isSharedCheck_854_;
goto v_resetjp_779_;
}
else
{
lean_inc(v_snd_778_);
lean_inc(v_fst_777_);
lean_dec(v_a_762_);
v___x_780_ = lean_box(0);
v_isShared_781_ = v_isSharedCheck_854_;
goto v_resetjp_779_;
}
v_resetjp_779_:
{
lean_object* v_lhs_782_; lean_object* v_rhs_783_; lean_object* v_res_784_; lean_object* v___x_785_; 
v_lhs_782_ = lean_ctor_get(v_a_760_, 0);
v_rhs_783_ = lean_ctor_get(v_a_760_, 1);
v_res_784_ = lean_ctor_get(v_a_760_, 2);
lean_inc(v_lhs_782_);
v___x_785_ = lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProof___redArg(v_fst_777_, v_lhs_782_, v_key_772_, v___y_764_, v___y_765_, v___y_766_, v___y_767_, v___y_768_);
if (lean_obj_tag(v___x_785_) == 0)
{
lean_object* v_a_786_; 
v_a_786_ = lean_ctor_get(v___x_785_, 0);
lean_inc(v_a_786_);
lean_dec_ref_known(v___x_785_, 1);
if (lean_obj_tag(v_a_786_) == 1)
{
lean_object* v_val_787_; lean_object* v___x_788_; 
v_val_787_ = lean_ctor_get(v_a_786_, 0);
lean_inc(v_val_787_);
lean_dec_ref_known(v_a_786_, 1);
lean_inc(v_rhs_783_);
v___x_788_ = lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProof___redArg(v_fst_777_, v_rhs_783_, v_key_772_, v___y_764_, v___y_765_, v___y_766_, v___y_767_, v___y_768_);
if (lean_obj_tag(v___x_788_) == 0)
{
lean_object* v_a_789_; 
v_a_789_ = lean_ctor_get(v___x_788_, 0);
lean_inc(v_a_789_);
lean_dec_ref_known(v___x_788_, 1);
if (lean_obj_tag(v_a_789_) == 1)
{
lean_object* v_val_790_; lean_object* v___x_791_; 
v_val_790_ = lean_ctor_get(v_a_789_, 0);
lean_inc(v_val_790_);
lean_dec_ref_known(v_a_789_, 1);
lean_inc(v_res_784_);
v___x_791_ = lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProof___redArg(v_fst_777_, v_res_784_, v_key_772_, v___y_764_, v___y_765_, v___y_766_, v___y_767_, v___y_768_);
if (lean_obj_tag(v___x_791_) == 0)
{
lean_object* v_a_792_; 
v_a_792_ = lean_ctor_get(v___x_791_, 0);
lean_inc(v_a_792_);
lean_dec_ref_known(v___x_791_, 1);
if (lean_obj_tag(v_a_792_) == 0)
{
lean_object* v___x_793_; lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___x_796_; lean_object* v___x_797_; lean_object* v___x_798_; 
lean_dec(v_snd_778_);
v___x_793_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__1));
v___x_794_ = lean_unsigned_to_nat(2u);
v___x_795_ = lean_mk_empty_array_with_capacity(v___x_794_);
v___x_796_ = lean_array_push(v___x_795_, v_val_787_);
v___x_797_ = lean_array_push(v___x_796_, v_val_790_);
v___x_798_ = l_Lean_Meta_mkAppM(v___x_793_, v___x_797_, v___y_765_, v___y_766_, v___y_767_, v___y_768_);
if (lean_obj_tag(v___x_798_) == 0)
{
lean_object* v_a_799_; uint8_t v___x_800_; lean_object* v___x_802_; 
v_a_799_ = lean_ctor_get(v___x_798_, 0);
lean_inc(v_a_799_);
lean_dec_ref_known(v___x_798_, 1);
v___x_800_ = 1;
lean_inc(v_res_784_);
if (v_isShared_776_ == 0)
{
lean_ctor_set_tag(v___x_775_, 0);
lean_ctor_set(v___x_775_, 2, v_a_799_);
lean_ctor_set(v___x_775_, 1, v_key_772_);
lean_ctor_set(v___x_775_, 0, v_res_784_);
v___x_802_ = v___x_775_;
goto v_reusejp_801_;
}
else
{
lean_object* v_reuseFailAlloc_809_; 
v_reuseFailAlloc_809_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_809_, 0, v_res_784_);
lean_ctor_set(v_reuseFailAlloc_809_, 1, v_key_772_);
lean_ctor_set(v_reuseFailAlloc_809_, 2, v_a_799_);
v___x_802_ = v_reuseFailAlloc_809_;
goto v_reusejp_801_;
}
v_reusejp_801_:
{
lean_object* v___x_803_; lean_object* v___x_804_; lean_object* v___x_806_; 
v___x_803_ = lp_mathlib_Mathlib_Tactic_Order_Graph_addEdge(v_fst_777_, v___x_802_);
v___x_804_ = lean_box(v___x_800_);
if (v_isShared_781_ == 0)
{
lean_ctor_set(v___x_780_, 1, v___x_804_);
lean_ctor_set(v___x_780_, 0, v___x_803_);
v___x_806_ = v___x_780_;
goto v_reusejp_805_;
}
else
{
lean_object* v_reuseFailAlloc_808_; 
v_reuseFailAlloc_808_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_808_, 0, v___x_803_);
lean_ctor_set(v_reuseFailAlloc_808_, 1, v___x_804_);
v___x_806_ = v_reuseFailAlloc_808_;
goto v_reusejp_805_;
}
v_reusejp_805_:
{
v_a_761_ = v_tail_773_;
v_a_762_ = v___x_806_;
goto _start;
}
}
}
else
{
lean_object* v_a_810_; lean_object* v___x_812_; uint8_t v_isShared_813_; uint8_t v_isSharedCheck_817_; 
lean_del_object(v___x_780_);
lean_dec(v_fst_777_);
lean_del_object(v___x_775_);
lean_dec(v_tail_773_);
lean_dec(v_key_772_);
lean_dec_ref_known(v_a_760_, 3);
v_a_810_ = lean_ctor_get(v___x_798_, 0);
v_isSharedCheck_817_ = !lean_is_exclusive(v___x_798_);
if (v_isSharedCheck_817_ == 0)
{
v___x_812_ = v___x_798_;
v_isShared_813_ = v_isSharedCheck_817_;
goto v_resetjp_811_;
}
else
{
lean_inc(v_a_810_);
lean_dec(v___x_798_);
v___x_812_ = lean_box(0);
v_isShared_813_ = v_isSharedCheck_817_;
goto v_resetjp_811_;
}
v_resetjp_811_:
{
lean_object* v___x_815_; 
if (v_isShared_813_ == 0)
{
v___x_815_ = v___x_812_;
goto v_reusejp_814_;
}
else
{
lean_object* v_reuseFailAlloc_816_; 
v_reuseFailAlloc_816_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_816_, 0, v_a_810_);
v___x_815_ = v_reuseFailAlloc_816_;
goto v_reusejp_814_;
}
v_reusejp_814_:
{
return v___x_815_;
}
}
}
}
else
{
lean_object* v___x_819_; 
lean_dec_ref_known(v_a_792_, 1);
lean_dec(v_val_790_);
lean_dec(v_val_787_);
lean_del_object(v___x_775_);
lean_dec(v_key_772_);
if (v_isShared_781_ == 0)
{
v___x_819_ = v___x_780_;
goto v_reusejp_818_;
}
else
{
lean_object* v_reuseFailAlloc_821_; 
v_reuseFailAlloc_821_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_821_, 0, v_fst_777_);
lean_ctor_set(v_reuseFailAlloc_821_, 1, v_snd_778_);
v___x_819_ = v_reuseFailAlloc_821_;
goto v_reusejp_818_;
}
v_reusejp_818_:
{
v_a_761_ = v_tail_773_;
v_a_762_ = v___x_819_;
goto _start;
}
}
}
else
{
lean_object* v_a_822_; lean_object* v___x_824_; uint8_t v_isShared_825_; uint8_t v_isSharedCheck_829_; 
lean_dec(v_val_790_);
lean_dec(v_val_787_);
lean_del_object(v___x_780_);
lean_dec(v_snd_778_);
lean_dec(v_fst_777_);
lean_del_object(v___x_775_);
lean_dec(v_tail_773_);
lean_dec_ref_known(v_a_760_, 3);
lean_dec(v_key_772_);
v_a_822_ = lean_ctor_get(v___x_791_, 0);
v_isSharedCheck_829_ = !lean_is_exclusive(v___x_791_);
if (v_isSharedCheck_829_ == 0)
{
v___x_824_ = v___x_791_;
v_isShared_825_ = v_isSharedCheck_829_;
goto v_resetjp_823_;
}
else
{
lean_inc(v_a_822_);
lean_dec(v___x_791_);
v___x_824_ = lean_box(0);
v_isShared_825_ = v_isSharedCheck_829_;
goto v_resetjp_823_;
}
v_resetjp_823_:
{
lean_object* v___x_827_; 
if (v_isShared_825_ == 0)
{
v___x_827_ = v___x_824_;
goto v_reusejp_826_;
}
else
{
lean_object* v_reuseFailAlloc_828_; 
v_reuseFailAlloc_828_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_828_, 0, v_a_822_);
v___x_827_ = v_reuseFailAlloc_828_;
goto v_reusejp_826_;
}
v_reusejp_826_:
{
return v___x_827_;
}
}
}
}
else
{
lean_object* v___x_831_; 
lean_dec(v_a_789_);
lean_dec(v_val_787_);
lean_del_object(v___x_775_);
lean_dec(v_key_772_);
if (v_isShared_781_ == 0)
{
v___x_831_ = v___x_780_;
goto v_reusejp_830_;
}
else
{
lean_object* v_reuseFailAlloc_833_; 
v_reuseFailAlloc_833_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_833_, 0, v_fst_777_);
lean_ctor_set(v_reuseFailAlloc_833_, 1, v_snd_778_);
v___x_831_ = v_reuseFailAlloc_833_;
goto v_reusejp_830_;
}
v_reusejp_830_:
{
v_a_761_ = v_tail_773_;
v_a_762_ = v___x_831_;
goto _start;
}
}
}
else
{
lean_object* v_a_834_; lean_object* v___x_836_; uint8_t v_isShared_837_; uint8_t v_isSharedCheck_841_; 
lean_dec(v_val_787_);
lean_del_object(v___x_780_);
lean_dec(v_snd_778_);
lean_dec(v_fst_777_);
lean_del_object(v___x_775_);
lean_dec(v_tail_773_);
lean_dec_ref_known(v_a_760_, 3);
lean_dec(v_key_772_);
v_a_834_ = lean_ctor_get(v___x_788_, 0);
v_isSharedCheck_841_ = !lean_is_exclusive(v___x_788_);
if (v_isSharedCheck_841_ == 0)
{
v___x_836_ = v___x_788_;
v_isShared_837_ = v_isSharedCheck_841_;
goto v_resetjp_835_;
}
else
{
lean_inc(v_a_834_);
lean_dec(v___x_788_);
v___x_836_ = lean_box(0);
v_isShared_837_ = v_isSharedCheck_841_;
goto v_resetjp_835_;
}
v_resetjp_835_:
{
lean_object* v___x_839_; 
if (v_isShared_837_ == 0)
{
v___x_839_ = v___x_836_;
goto v_reusejp_838_;
}
else
{
lean_object* v_reuseFailAlloc_840_; 
v_reuseFailAlloc_840_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_840_, 0, v_a_834_);
v___x_839_ = v_reuseFailAlloc_840_;
goto v_reusejp_838_;
}
v_reusejp_838_:
{
return v___x_839_;
}
}
}
}
else
{
lean_object* v___x_843_; 
lean_dec(v_a_786_);
lean_del_object(v___x_775_);
lean_dec(v_key_772_);
if (v_isShared_781_ == 0)
{
v___x_843_ = v___x_780_;
goto v_reusejp_842_;
}
else
{
lean_object* v_reuseFailAlloc_845_; 
v_reuseFailAlloc_845_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_845_, 0, v_fst_777_);
lean_ctor_set(v_reuseFailAlloc_845_, 1, v_snd_778_);
v___x_843_ = v_reuseFailAlloc_845_;
goto v_reusejp_842_;
}
v_reusejp_842_:
{
v_a_761_ = v_tail_773_;
v_a_762_ = v___x_843_;
goto _start;
}
}
}
else
{
lean_object* v_a_846_; lean_object* v___x_848_; uint8_t v_isShared_849_; uint8_t v_isSharedCheck_853_; 
lean_del_object(v___x_780_);
lean_dec(v_snd_778_);
lean_dec(v_fst_777_);
lean_del_object(v___x_775_);
lean_dec(v_tail_773_);
lean_dec(v_key_772_);
lean_dec_ref_known(v_a_760_, 3);
v_a_846_ = lean_ctor_get(v___x_785_, 0);
v_isSharedCheck_853_ = !lean_is_exclusive(v___x_785_);
if (v_isSharedCheck_853_ == 0)
{
v___x_848_ = v___x_785_;
v_isShared_849_ = v_isSharedCheck_853_;
goto v_resetjp_847_;
}
else
{
lean_inc(v_a_846_);
lean_dec(v___x_785_);
v___x_848_ = lean_box(0);
v_isShared_849_ = v_isSharedCheck_853_;
goto v_resetjp_847_;
}
v_resetjp_847_:
{
lean_object* v___x_851_; 
if (v_isShared_849_ == 0)
{
v___x_851_ = v___x_848_;
goto v_reusejp_850_;
}
else
{
lean_object* v_reuseFailAlloc_852_; 
v_reuseFailAlloc_852_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_852_, 0, v_a_846_);
v___x_851_ = v_reuseFailAlloc_852_;
goto v_reusejp_850_;
}
v_reusejp_850_:
{
return v___x_851_;
}
}
}
}
}
}
case 8:
{
lean_object* v_key_857_; lean_object* v_tail_858_; lean_object* v___x_860_; uint8_t v_isShared_861_; uint8_t v_isSharedCheck_940_; 
v_key_857_ = lean_ctor_get(v_a_761_, 0);
v_tail_858_ = lean_ctor_get(v_a_761_, 2);
v_isSharedCheck_940_ = !lean_is_exclusive(v_a_761_);
if (v_isSharedCheck_940_ == 0)
{
lean_object* v_unused_941_; 
v_unused_941_ = lean_ctor_get(v_a_761_, 1);
lean_dec(v_unused_941_);
v___x_860_ = v_a_761_;
v_isShared_861_ = v_isSharedCheck_940_;
goto v_resetjp_859_;
}
else
{
lean_inc(v_tail_858_);
lean_inc(v_key_857_);
lean_dec(v_a_761_);
v___x_860_ = lean_box(0);
v_isShared_861_ = v_isSharedCheck_940_;
goto v_resetjp_859_;
}
v_resetjp_859_:
{
lean_object* v_fst_862_; lean_object* v_snd_863_; lean_object* v___x_865_; uint8_t v_isShared_866_; uint8_t v_isSharedCheck_939_; 
v_fst_862_ = lean_ctor_get(v_a_762_, 0);
v_snd_863_ = lean_ctor_get(v_a_762_, 1);
v_isSharedCheck_939_ = !lean_is_exclusive(v_a_762_);
if (v_isSharedCheck_939_ == 0)
{
v___x_865_ = v_a_762_;
v_isShared_866_ = v_isSharedCheck_939_;
goto v_resetjp_864_;
}
else
{
lean_inc(v_snd_863_);
lean_inc(v_fst_862_);
lean_dec(v_a_762_);
v___x_865_ = lean_box(0);
v_isShared_866_ = v_isSharedCheck_939_;
goto v_resetjp_864_;
}
v_resetjp_864_:
{
lean_object* v_lhs_867_; lean_object* v_rhs_868_; lean_object* v_res_869_; lean_object* v___x_870_; 
v_lhs_867_ = lean_ctor_get(v_a_760_, 0);
v_rhs_868_ = lean_ctor_get(v_a_760_, 1);
v_res_869_ = lean_ctor_get(v_a_760_, 2);
lean_inc(v_key_857_);
v___x_870_ = lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProof___redArg(v_fst_862_, v_key_857_, v_lhs_867_, v___y_764_, v___y_765_, v___y_766_, v___y_767_, v___y_768_);
if (lean_obj_tag(v___x_870_) == 0)
{
lean_object* v_a_871_; 
v_a_871_ = lean_ctor_get(v___x_870_, 0);
lean_inc(v_a_871_);
lean_dec_ref_known(v___x_870_, 1);
if (lean_obj_tag(v_a_871_) == 1)
{
lean_object* v_val_872_; lean_object* v___x_873_; 
v_val_872_ = lean_ctor_get(v_a_871_, 0);
lean_inc(v_val_872_);
lean_dec_ref_known(v_a_871_, 1);
lean_inc(v_key_857_);
v___x_873_ = lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProof___redArg(v_fst_862_, v_key_857_, v_rhs_868_, v___y_764_, v___y_765_, v___y_766_, v___y_767_, v___y_768_);
if (lean_obj_tag(v___x_873_) == 0)
{
lean_object* v_a_874_; 
v_a_874_ = lean_ctor_get(v___x_873_, 0);
lean_inc(v_a_874_);
lean_dec_ref_known(v___x_873_, 1);
if (lean_obj_tag(v_a_874_) == 1)
{
lean_object* v_val_875_; lean_object* v___x_876_; 
v_val_875_ = lean_ctor_get(v_a_874_, 0);
lean_inc(v_val_875_);
lean_dec_ref_known(v_a_874_, 1);
lean_inc(v_key_857_);
v___x_876_ = lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProof___redArg(v_fst_862_, v_key_857_, v_res_869_, v___y_764_, v___y_765_, v___y_766_, v___y_767_, v___y_768_);
if (lean_obj_tag(v___x_876_) == 0)
{
lean_object* v_a_877_; 
v_a_877_ = lean_ctor_get(v___x_876_, 0);
lean_inc(v_a_877_);
lean_dec_ref_known(v___x_876_, 1);
if (lean_obj_tag(v_a_877_) == 0)
{
lean_object* v___x_878_; lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v___x_881_; lean_object* v___x_882_; lean_object* v___x_883_; 
lean_dec(v_snd_863_);
v___x_878_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__3));
v___x_879_ = lean_unsigned_to_nat(2u);
v___x_880_ = lean_mk_empty_array_with_capacity(v___x_879_);
v___x_881_ = lean_array_push(v___x_880_, v_val_872_);
v___x_882_ = lean_array_push(v___x_881_, v_val_875_);
v___x_883_ = l_Lean_Meta_mkAppM(v___x_878_, v___x_882_, v___y_765_, v___y_766_, v___y_767_, v___y_768_);
if (lean_obj_tag(v___x_883_) == 0)
{
lean_object* v_a_884_; uint8_t v___x_885_; lean_object* v___x_887_; 
v_a_884_ = lean_ctor_get(v___x_883_, 0);
lean_inc(v_a_884_);
lean_dec_ref_known(v___x_883_, 1);
v___x_885_ = 1;
lean_inc(v_res_869_);
if (v_isShared_861_ == 0)
{
lean_ctor_set_tag(v___x_860_, 0);
lean_ctor_set(v___x_860_, 2, v_a_884_);
lean_ctor_set(v___x_860_, 1, v_res_869_);
v___x_887_ = v___x_860_;
goto v_reusejp_886_;
}
else
{
lean_object* v_reuseFailAlloc_894_; 
v_reuseFailAlloc_894_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_894_, 0, v_key_857_);
lean_ctor_set(v_reuseFailAlloc_894_, 1, v_res_869_);
lean_ctor_set(v_reuseFailAlloc_894_, 2, v_a_884_);
v___x_887_ = v_reuseFailAlloc_894_;
goto v_reusejp_886_;
}
v_reusejp_886_:
{
lean_object* v___x_888_; lean_object* v___x_889_; lean_object* v___x_891_; 
v___x_888_ = lp_mathlib_Mathlib_Tactic_Order_Graph_addEdge(v_fst_862_, v___x_887_);
v___x_889_ = lean_box(v___x_885_);
if (v_isShared_866_ == 0)
{
lean_ctor_set(v___x_865_, 1, v___x_889_);
lean_ctor_set(v___x_865_, 0, v___x_888_);
v___x_891_ = v___x_865_;
goto v_reusejp_890_;
}
else
{
lean_object* v_reuseFailAlloc_893_; 
v_reuseFailAlloc_893_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_893_, 0, v___x_888_);
lean_ctor_set(v_reuseFailAlloc_893_, 1, v___x_889_);
v___x_891_ = v_reuseFailAlloc_893_;
goto v_reusejp_890_;
}
v_reusejp_890_:
{
v_a_761_ = v_tail_858_;
v_a_762_ = v___x_891_;
goto _start;
}
}
}
else
{
lean_object* v_a_895_; lean_object* v___x_897_; uint8_t v_isShared_898_; uint8_t v_isSharedCheck_902_; 
lean_del_object(v___x_865_);
lean_dec(v_fst_862_);
lean_del_object(v___x_860_);
lean_dec(v_tail_858_);
lean_dec(v_key_857_);
lean_dec_ref_known(v_a_760_, 3);
v_a_895_ = lean_ctor_get(v___x_883_, 0);
v_isSharedCheck_902_ = !lean_is_exclusive(v___x_883_);
if (v_isSharedCheck_902_ == 0)
{
v___x_897_ = v___x_883_;
v_isShared_898_ = v_isSharedCheck_902_;
goto v_resetjp_896_;
}
else
{
lean_inc(v_a_895_);
lean_dec(v___x_883_);
v___x_897_ = lean_box(0);
v_isShared_898_ = v_isSharedCheck_902_;
goto v_resetjp_896_;
}
v_resetjp_896_:
{
lean_object* v___x_900_; 
if (v_isShared_898_ == 0)
{
v___x_900_ = v___x_897_;
goto v_reusejp_899_;
}
else
{
lean_object* v_reuseFailAlloc_901_; 
v_reuseFailAlloc_901_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_901_, 0, v_a_895_);
v___x_900_ = v_reuseFailAlloc_901_;
goto v_reusejp_899_;
}
v_reusejp_899_:
{
return v___x_900_;
}
}
}
}
else
{
lean_object* v___x_904_; 
lean_dec_ref_known(v_a_877_, 1);
lean_dec(v_val_875_);
lean_dec(v_val_872_);
lean_del_object(v___x_860_);
lean_dec(v_key_857_);
if (v_isShared_866_ == 0)
{
v___x_904_ = v___x_865_;
goto v_reusejp_903_;
}
else
{
lean_object* v_reuseFailAlloc_906_; 
v_reuseFailAlloc_906_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_906_, 0, v_fst_862_);
lean_ctor_set(v_reuseFailAlloc_906_, 1, v_snd_863_);
v___x_904_ = v_reuseFailAlloc_906_;
goto v_reusejp_903_;
}
v_reusejp_903_:
{
v_a_761_ = v_tail_858_;
v_a_762_ = v___x_904_;
goto _start;
}
}
}
else
{
lean_object* v_a_907_; lean_object* v___x_909_; uint8_t v_isShared_910_; uint8_t v_isSharedCheck_914_; 
lean_dec(v_val_875_);
lean_dec(v_val_872_);
lean_del_object(v___x_865_);
lean_dec(v_snd_863_);
lean_dec(v_fst_862_);
lean_del_object(v___x_860_);
lean_dec(v_tail_858_);
lean_dec(v_key_857_);
lean_dec_ref_known(v_a_760_, 3);
v_a_907_ = lean_ctor_get(v___x_876_, 0);
v_isSharedCheck_914_ = !lean_is_exclusive(v___x_876_);
if (v_isSharedCheck_914_ == 0)
{
v___x_909_ = v___x_876_;
v_isShared_910_ = v_isSharedCheck_914_;
goto v_resetjp_908_;
}
else
{
lean_inc(v_a_907_);
lean_dec(v___x_876_);
v___x_909_ = lean_box(0);
v_isShared_910_ = v_isSharedCheck_914_;
goto v_resetjp_908_;
}
v_resetjp_908_:
{
lean_object* v___x_912_; 
if (v_isShared_910_ == 0)
{
v___x_912_ = v___x_909_;
goto v_reusejp_911_;
}
else
{
lean_object* v_reuseFailAlloc_913_; 
v_reuseFailAlloc_913_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_913_, 0, v_a_907_);
v___x_912_ = v_reuseFailAlloc_913_;
goto v_reusejp_911_;
}
v_reusejp_911_:
{
return v___x_912_;
}
}
}
}
else
{
lean_object* v___x_916_; 
lean_dec(v_a_874_);
lean_dec(v_val_872_);
lean_del_object(v___x_860_);
lean_dec(v_key_857_);
if (v_isShared_866_ == 0)
{
v___x_916_ = v___x_865_;
goto v_reusejp_915_;
}
else
{
lean_object* v_reuseFailAlloc_918_; 
v_reuseFailAlloc_918_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_918_, 0, v_fst_862_);
lean_ctor_set(v_reuseFailAlloc_918_, 1, v_snd_863_);
v___x_916_ = v_reuseFailAlloc_918_;
goto v_reusejp_915_;
}
v_reusejp_915_:
{
v_a_761_ = v_tail_858_;
v_a_762_ = v___x_916_;
goto _start;
}
}
}
else
{
lean_object* v_a_919_; lean_object* v___x_921_; uint8_t v_isShared_922_; uint8_t v_isSharedCheck_926_; 
lean_dec(v_val_872_);
lean_del_object(v___x_865_);
lean_dec(v_snd_863_);
lean_dec(v_fst_862_);
lean_del_object(v___x_860_);
lean_dec(v_tail_858_);
lean_dec(v_key_857_);
lean_dec_ref_known(v_a_760_, 3);
v_a_919_ = lean_ctor_get(v___x_873_, 0);
v_isSharedCheck_926_ = !lean_is_exclusive(v___x_873_);
if (v_isSharedCheck_926_ == 0)
{
v___x_921_ = v___x_873_;
v_isShared_922_ = v_isSharedCheck_926_;
goto v_resetjp_920_;
}
else
{
lean_inc(v_a_919_);
lean_dec(v___x_873_);
v___x_921_ = lean_box(0);
v_isShared_922_ = v_isSharedCheck_926_;
goto v_resetjp_920_;
}
v_resetjp_920_:
{
lean_object* v___x_924_; 
if (v_isShared_922_ == 0)
{
v___x_924_ = v___x_921_;
goto v_reusejp_923_;
}
else
{
lean_object* v_reuseFailAlloc_925_; 
v_reuseFailAlloc_925_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_925_, 0, v_a_919_);
v___x_924_ = v_reuseFailAlloc_925_;
goto v_reusejp_923_;
}
v_reusejp_923_:
{
return v___x_924_;
}
}
}
}
else
{
lean_object* v___x_928_; 
lean_dec(v_a_871_);
lean_del_object(v___x_860_);
lean_dec(v_key_857_);
if (v_isShared_866_ == 0)
{
v___x_928_ = v___x_865_;
goto v_reusejp_927_;
}
else
{
lean_object* v_reuseFailAlloc_930_; 
v_reuseFailAlloc_930_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_930_, 0, v_fst_862_);
lean_ctor_set(v_reuseFailAlloc_930_, 1, v_snd_863_);
v___x_928_ = v_reuseFailAlloc_930_;
goto v_reusejp_927_;
}
v_reusejp_927_:
{
v_a_761_ = v_tail_858_;
v_a_762_ = v___x_928_;
goto _start;
}
}
}
else
{
lean_object* v_a_931_; lean_object* v___x_933_; uint8_t v_isShared_934_; uint8_t v_isSharedCheck_938_; 
lean_del_object(v___x_865_);
lean_dec(v_snd_863_);
lean_dec(v_fst_862_);
lean_del_object(v___x_860_);
lean_dec(v_tail_858_);
lean_dec(v_key_857_);
lean_dec_ref_known(v_a_760_, 3);
v_a_931_ = lean_ctor_get(v___x_870_, 0);
v_isSharedCheck_938_ = !lean_is_exclusive(v___x_870_);
if (v_isSharedCheck_938_ == 0)
{
v___x_933_ = v___x_870_;
v_isShared_934_ = v_isSharedCheck_938_;
goto v_resetjp_932_;
}
else
{
lean_inc(v_a_931_);
lean_dec(v___x_870_);
v___x_933_ = lean_box(0);
v_isShared_934_ = v_isSharedCheck_938_;
goto v_resetjp_932_;
}
v_resetjp_932_:
{
lean_object* v___x_936_; 
if (v_isShared_934_ == 0)
{
v___x_936_ = v___x_933_;
goto v_reusejp_935_;
}
else
{
lean_object* v_reuseFailAlloc_937_; 
v_reuseFailAlloc_937_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_937_, 0, v_a_931_);
v___x_936_ = v_reuseFailAlloc_937_;
goto v_reusejp_935_;
}
v_reusejp_935_:
{
return v___x_936_;
}
}
}
}
}
}
default: 
{
lean_object* v_tail_942_; lean_object* v_fst_943_; lean_object* v_snd_944_; lean_object* v___x_946_; uint8_t v_isShared_947_; uint8_t v_isSharedCheck_962_; 
v_tail_942_ = lean_ctor_get(v_a_761_, 2);
lean_inc(v_tail_942_);
lean_dec_ref_known(v_a_761_, 3);
v_fst_943_ = lean_ctor_get(v_a_762_, 0);
v_snd_944_ = lean_ctor_get(v_a_762_, 1);
v_isSharedCheck_962_ = !lean_is_exclusive(v_a_762_);
if (v_isSharedCheck_962_ == 0)
{
v___x_946_ = v_a_762_;
v_isShared_947_ = v_isSharedCheck_962_;
goto v_resetjp_945_;
}
else
{
lean_inc(v_snd_944_);
lean_inc(v_fst_943_);
lean_dec(v_a_762_);
v___x_946_ = lean_box(0);
v_isShared_947_ = v_isSharedCheck_962_;
goto v_resetjp_945_;
}
v_resetjp_945_:
{
lean_object* v___x_948_; lean_object* v___x_949_; 
v___x_948_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__5, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__5_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___closed__5);
v___x_949_ = lp_mathlib_panic___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__2(v___x_948_, v___y_763_, v___y_764_, v___y_765_, v___y_766_, v___y_767_, v___y_768_);
if (lean_obj_tag(v___x_949_) == 0)
{
lean_object* v___x_951_; 
lean_dec_ref_known(v___x_949_, 1);
if (v_isShared_947_ == 0)
{
v___x_951_ = v___x_946_;
goto v_reusejp_950_;
}
else
{
lean_object* v_reuseFailAlloc_953_; 
v_reuseFailAlloc_953_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_953_, 0, v_fst_943_);
lean_ctor_set(v_reuseFailAlloc_953_, 1, v_snd_944_);
v___x_951_ = v_reuseFailAlloc_953_;
goto v_reusejp_950_;
}
v_reusejp_950_:
{
v_a_761_ = v_tail_942_;
v_a_762_ = v___x_951_;
goto _start;
}
}
else
{
lean_object* v_a_954_; lean_object* v___x_956_; uint8_t v_isShared_957_; uint8_t v_isSharedCheck_961_; 
lean_del_object(v___x_946_);
lean_dec(v_snd_944_);
lean_dec(v_fst_943_);
lean_dec(v_tail_942_);
lean_dec_ref(v_a_760_);
v_a_954_ = lean_ctor_get(v___x_949_, 0);
v_isSharedCheck_961_ = !lean_is_exclusive(v___x_949_);
if (v_isSharedCheck_961_ == 0)
{
v___x_956_ = v___x_949_;
v_isShared_957_ = v_isSharedCheck_961_;
goto v_resetjp_955_;
}
else
{
lean_inc(v_a_954_);
lean_dec(v___x_949_);
v___x_956_ = lean_box(0);
v_isShared_957_ = v_isSharedCheck_961_;
goto v_resetjp_955_;
}
v_resetjp_955_:
{
lean_object* v___x_959_; 
if (v_isShared_957_ == 0)
{
v___x_959_ = v___x_956_;
goto v_reusejp_958_;
}
else
{
lean_object* v_reuseFailAlloc_960_; 
v_reuseFailAlloc_960_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_960_, 0, v_a_954_);
v___x_959_ = v_reuseFailAlloc_960_;
goto v_reusejp_958_;
}
v_reusejp_958_:
{
return v___x_959_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0___boxed(lean_object* v_a_963_, lean_object* v_a_964_, lean_object* v_a_965_, lean_object* v___y_966_, lean_object* v___y_967_, lean_object* v___y_968_, lean_object* v___y_969_, lean_object* v___y_970_, lean_object* v___y_971_, lean_object* v___y_972_){
_start:
{
lean_object* v_res_973_; 
v_res_973_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0(v_a_963_, v_a_964_, v_a_965_, v___y_966_, v___y_967_, v___y_968_, v___y_969_, v___y_970_, v___y_971_);
lean_dec(v___y_971_);
lean_dec_ref(v___y_970_);
lean_dec(v___y_969_);
lean_dec_ref(v___y_968_);
lean_dec(v___y_967_);
lean_dec_ref(v___y_966_);
return v_res_973_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__1(lean_object* v_a_974_, lean_object* v_as_975_, size_t v_sz_976_, size_t v_i_977_, lean_object* v_b_978_, lean_object* v___y_979_, lean_object* v___y_980_, lean_object* v___y_981_, lean_object* v___y_982_, lean_object* v___y_983_, lean_object* v___y_984_){
_start:
{
uint8_t v___x_986_; 
v___x_986_ = lean_usize_dec_lt(v_i_977_, v_sz_976_);
if (v___x_986_ == 0)
{
lean_object* v___x_987_; 
lean_dec_ref(v_a_974_);
v___x_987_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_987_, 0, v_b_978_);
return v___x_987_;
}
else
{
lean_object* v_a_988_; lean_object* v___x_989_; 
v_a_988_ = lean_array_uget_borrowed(v_as_975_, v_i_977_);
lean_inc(v_a_988_);
lean_inc_ref(v_a_974_);
v___x_989_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__0(v_a_974_, v_a_988_, v_b_978_, v___y_979_, v___y_980_, v___y_981_, v___y_982_, v___y_983_, v___y_984_);
if (lean_obj_tag(v___x_989_) == 0)
{
lean_object* v_a_990_; lean_object* v___x_992_; uint8_t v_isShared_993_; uint8_t v_isSharedCheck_1002_; 
v_a_990_ = lean_ctor_get(v___x_989_, 0);
v_isSharedCheck_1002_ = !lean_is_exclusive(v___x_989_);
if (v_isSharedCheck_1002_ == 0)
{
v___x_992_ = v___x_989_;
v_isShared_993_ = v_isSharedCheck_1002_;
goto v_resetjp_991_;
}
else
{
lean_inc(v_a_990_);
lean_dec(v___x_989_);
v___x_992_ = lean_box(0);
v_isShared_993_ = v_isSharedCheck_1002_;
goto v_resetjp_991_;
}
v_resetjp_991_:
{
if (lean_obj_tag(v_a_990_) == 0)
{
lean_object* v_a_994_; lean_object* v___x_996_; 
lean_dec_ref(v_a_974_);
v_a_994_ = lean_ctor_get(v_a_990_, 0);
lean_inc(v_a_994_);
lean_dec_ref_known(v_a_990_, 1);
if (v_isShared_993_ == 0)
{
lean_ctor_set(v___x_992_, 0, v_a_994_);
v___x_996_ = v___x_992_;
goto v_reusejp_995_;
}
else
{
lean_object* v_reuseFailAlloc_997_; 
v_reuseFailAlloc_997_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_997_, 0, v_a_994_);
v___x_996_ = v_reuseFailAlloc_997_;
goto v_reusejp_995_;
}
v_reusejp_995_:
{
return v___x_996_;
}
}
else
{
lean_object* v_a_998_; size_t v___x_999_; size_t v___x_1000_; 
lean_del_object(v___x_992_);
v_a_998_ = lean_ctor_get(v_a_990_, 0);
lean_inc(v_a_998_);
lean_dec_ref_known(v_a_990_, 1);
v___x_999_ = ((size_t)1ULL);
v___x_1000_ = lean_usize_add(v_i_977_, v___x_999_);
v_i_977_ = v___x_1000_;
v_b_978_ = v_a_998_;
goto _start;
}
}
}
else
{
lean_object* v_a_1003_; lean_object* v___x_1005_; uint8_t v_isShared_1006_; uint8_t v_isSharedCheck_1010_; 
lean_dec_ref(v_a_974_);
v_a_1003_ = lean_ctor_get(v___x_989_, 0);
v_isSharedCheck_1010_ = !lean_is_exclusive(v___x_989_);
if (v_isSharedCheck_1010_ == 0)
{
v___x_1005_ = v___x_989_;
v_isShared_1006_ = v_isSharedCheck_1010_;
goto v_resetjp_1004_;
}
else
{
lean_inc(v_a_1003_);
lean_dec(v___x_989_);
v___x_1005_ = lean_box(0);
v_isShared_1006_ = v_isSharedCheck_1010_;
goto v_resetjp_1004_;
}
v_resetjp_1004_:
{
lean_object* v___x_1008_; 
if (v_isShared_1006_ == 0)
{
v___x_1008_ = v___x_1005_;
goto v_reusejp_1007_;
}
else
{
lean_object* v_reuseFailAlloc_1009_; 
v_reuseFailAlloc_1009_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1009_, 0, v_a_1003_);
v___x_1008_ = v_reuseFailAlloc_1009_;
goto v_reusejp_1007_;
}
v_reusejp_1007_:
{
return v___x_1008_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__1___boxed(lean_object* v_a_1011_, lean_object* v_as_1012_, lean_object* v_sz_1013_, lean_object* v_i_1014_, lean_object* v_b_1015_, lean_object* v___y_1016_, lean_object* v___y_1017_, lean_object* v___y_1018_, lean_object* v___y_1019_, lean_object* v___y_1020_, lean_object* v___y_1021_, lean_object* v___y_1022_){
_start:
{
size_t v_sz_boxed_1023_; size_t v_i_boxed_1024_; lean_object* v_res_1025_; 
v_sz_boxed_1023_ = lean_unbox_usize(v_sz_1013_);
lean_dec(v_sz_1013_);
v_i_boxed_1024_ = lean_unbox_usize(v_i_1014_);
lean_dec(v_i_1014_);
v_res_1025_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__1(v_a_1011_, v_as_1012_, v_sz_boxed_1023_, v_i_boxed_1024_, v_b_1015_, v___y_1016_, v___y_1017_, v___y_1018_, v___y_1019_, v___y_1020_, v___y_1021_);
lean_dec(v___y_1021_);
lean_dec_ref(v___y_1020_);
lean_dec(v___y_1019_);
lean_dec_ref(v___y_1018_);
lean_dec(v___y_1017_);
lean_dec_ref(v___y_1016_);
lean_dec_ref(v_as_1012_);
return v_res_1025_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__3(lean_object* v___y_1026_, lean_object* v_as_1027_, size_t v_sz_1028_, size_t v_i_1029_, lean_object* v_b_1030_, lean_object* v___y_1031_, lean_object* v___y_1032_, lean_object* v___y_1033_, lean_object* v___y_1034_, lean_object* v___y_1035_, lean_object* v___y_1036_){
_start:
{
uint8_t v___x_1038_; 
v___x_1038_ = lean_usize_dec_lt(v_i_1029_, v_sz_1028_);
if (v___x_1038_ == 0)
{
lean_object* v___x_1039_; 
v___x_1039_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1039_, 0, v_b_1030_);
return v___x_1039_;
}
else
{
lean_object* v_fst_1040_; lean_object* v_snd_1041_; lean_object* v___x_1043_; uint8_t v_isShared_1044_; uint8_t v_isSharedCheck_1066_; 
v_fst_1040_ = lean_ctor_get(v_b_1030_, 0);
v_snd_1041_ = lean_ctor_get(v_b_1030_, 1);
v_isSharedCheck_1066_ = !lean_is_exclusive(v_b_1030_);
if (v_isSharedCheck_1066_ == 0)
{
v___x_1043_ = v_b_1030_;
v_isShared_1044_ = v_isSharedCheck_1066_;
goto v_resetjp_1042_;
}
else
{
lean_inc(v_snd_1041_);
lean_inc(v_fst_1040_);
lean_dec(v_b_1030_);
v___x_1043_ = lean_box(0);
v_isShared_1044_ = v_isSharedCheck_1066_;
goto v_resetjp_1042_;
}
v_resetjp_1042_:
{
lean_object* v_buckets_1045_; lean_object* v_a_1046_; lean_object* v___x_1048_; 
v_buckets_1045_ = lean_ctor_get(v___y_1026_, 1);
v_a_1046_ = lean_array_uget_borrowed(v_as_1027_, v_i_1029_);
if (v_isShared_1044_ == 0)
{
v___x_1048_ = v___x_1043_;
goto v_reusejp_1047_;
}
else
{
lean_object* v_reuseFailAlloc_1065_; 
v_reuseFailAlloc_1065_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1065_, 0, v_fst_1040_);
lean_ctor_set(v_reuseFailAlloc_1065_, 1, v_snd_1041_);
v___x_1048_ = v_reuseFailAlloc_1065_;
goto v_reusejp_1047_;
}
v_reusejp_1047_:
{
size_t v_sz_1049_; size_t v___x_1050_; lean_object* v___x_1051_; 
v_sz_1049_ = lean_array_size(v_buckets_1045_);
v___x_1050_ = ((size_t)0ULL);
lean_inc(v_a_1046_);
v___x_1051_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__1(v_a_1046_, v_buckets_1045_, v_sz_1049_, v___x_1050_, v___x_1048_, v___y_1031_, v___y_1032_, v___y_1033_, v___y_1034_, v___y_1035_, v___y_1036_);
if (lean_obj_tag(v___x_1051_) == 0)
{
lean_object* v_a_1052_; lean_object* v_fst_1053_; lean_object* v_snd_1054_; lean_object* v___x_1056_; uint8_t v_isShared_1057_; uint8_t v_isSharedCheck_1064_; 
v_a_1052_ = lean_ctor_get(v___x_1051_, 0);
lean_inc(v_a_1052_);
lean_dec_ref_known(v___x_1051_, 1);
v_fst_1053_ = lean_ctor_get(v_a_1052_, 0);
v_snd_1054_ = lean_ctor_get(v_a_1052_, 1);
v_isSharedCheck_1064_ = !lean_is_exclusive(v_a_1052_);
if (v_isSharedCheck_1064_ == 0)
{
v___x_1056_ = v_a_1052_;
v_isShared_1057_ = v_isSharedCheck_1064_;
goto v_resetjp_1055_;
}
else
{
lean_inc(v_snd_1054_);
lean_inc(v_fst_1053_);
lean_dec(v_a_1052_);
v___x_1056_ = lean_box(0);
v_isShared_1057_ = v_isSharedCheck_1064_;
goto v_resetjp_1055_;
}
v_resetjp_1055_:
{
lean_object* v___x_1059_; 
if (v_isShared_1057_ == 0)
{
v___x_1059_ = v___x_1056_;
goto v_reusejp_1058_;
}
else
{
lean_object* v_reuseFailAlloc_1063_; 
v_reuseFailAlloc_1063_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1063_, 0, v_fst_1053_);
lean_ctor_set(v_reuseFailAlloc_1063_, 1, v_snd_1054_);
v___x_1059_ = v_reuseFailAlloc_1063_;
goto v_reusejp_1058_;
}
v_reusejp_1058_:
{
size_t v___x_1060_; size_t v___x_1061_; 
v___x_1060_ = ((size_t)1ULL);
v___x_1061_ = lean_usize_add(v_i_1029_, v___x_1060_);
v_i_1029_ = v___x_1061_;
v_b_1030_ = v___x_1059_;
goto _start;
}
}
}
else
{
return v___x_1051_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__3___boxed(lean_object* v___y_1067_, lean_object* v_as_1068_, lean_object* v_sz_1069_, lean_object* v_i_1070_, lean_object* v_b_1071_, lean_object* v___y_1072_, lean_object* v___y_1073_, lean_object* v___y_1074_, lean_object* v___y_1075_, lean_object* v___y_1076_, lean_object* v___y_1077_, lean_object* v___y_1078_){
_start:
{
size_t v_sz_boxed_1079_; size_t v_i_boxed_1080_; lean_object* v_res_1081_; 
v_sz_boxed_1079_ = lean_unbox_usize(v_sz_1069_);
lean_dec(v_sz_1069_);
v_i_boxed_1080_ = lean_unbox_usize(v_i_1070_);
lean_dec(v_i_1070_);
v_res_1081_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__3(v___y_1067_, v_as_1068_, v_sz_boxed_1079_, v_i_boxed_1080_, v_b_1071_, v___y_1072_, v___y_1073_, v___y_1074_, v___y_1075_, v___y_1076_, v___y_1077_);
lean_dec(v___y_1077_);
lean_dec_ref(v___y_1076_);
lean_dec(v___y_1075_);
lean_dec_ref(v___y_1074_);
lean_dec(v___y_1073_);
lean_dec_ref(v___y_1072_);
lean_dec_ref(v_as_1068_);
lean_dec_ref(v___y_1067_);
return v_res_1081_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__4___redArg(lean_object* v___x_1082_, lean_object* v___y_1083_, lean_object* v___y_1084_, lean_object* v___y_1085_, lean_object* v_a_1086_, lean_object* v___y_1087_, lean_object* v___y_1088_, lean_object* v___y_1089_, lean_object* v___y_1090_, lean_object* v___y_1091_, lean_object* v___y_1092_){
_start:
{
lean_object* v_fst_1094_; lean_object* v_snd_1095_; lean_object* v___x_1097_; uint8_t v_isShared_1098_; uint8_t v_isSharedCheck_1169_; 
v_fst_1094_ = lean_ctor_get(v_a_1086_, 0);
v_snd_1095_ = lean_ctor_get(v_a_1086_, 1);
v_isSharedCheck_1169_ = !lean_is_exclusive(v_a_1086_);
if (v_isSharedCheck_1169_ == 0)
{
v___x_1097_ = v_a_1086_;
v_isShared_1098_ = v_isSharedCheck_1169_;
goto v_resetjp_1096_;
}
else
{
lean_inc(v_snd_1095_);
lean_inc(v_fst_1094_);
lean_dec(v_a_1086_);
v___x_1097_ = lean_box(0);
v_isShared_1098_ = v_isSharedCheck_1169_;
goto v_resetjp_1096_;
}
v_resetjp_1096_:
{
lean_object* v___x_1099_; uint8_t v_changed_1100_; lean_object* v___x_1101_; lean_object* v___x_1102_; lean_object* v___x_1103_; lean_object* v___x_1105_; 
v___x_1099_ = lean_unsigned_to_nat(0u);
v_changed_1100_ = 0;
v___x_1101_ = lean_unsigned_to_nat(1u);
lean_inc(v___x_1082_);
v___x_1102_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1102_, 0, v___x_1099_);
lean_ctor_set(v___x_1102_, 1, v___x_1082_);
lean_ctor_set(v___x_1102_, 2, v___x_1101_);
v___x_1103_ = lean_box(v_changed_1100_);
if (v_isShared_1098_ == 0)
{
lean_ctor_set(v___x_1097_, 1, v___x_1103_);
lean_ctor_set(v___x_1097_, 0, v_snd_1095_);
v___x_1105_ = v___x_1097_;
goto v_reusejp_1104_;
}
else
{
lean_object* v_reuseFailAlloc_1168_; 
v_reuseFailAlloc_1168_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1168_, 0, v_snd_1095_);
lean_ctor_set(v_reuseFailAlloc_1168_, 1, v___x_1103_);
v___x_1105_ = v_reuseFailAlloc_1168_;
goto v_reusejp_1104_;
}
v_reusejp_1104_:
{
lean_object* v___x_1106_; lean_object* v___x_1107_; 
v___x_1106_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1106_, 0, v_fst_1094_);
lean_ctor_set(v___x_1106_, 1, v___x_1105_);
v___x_1107_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg(v___y_1083_, v___x_1102_, v___x_1106_, v___x_1099_, v___y_1087_, v___y_1088_, v___y_1089_, v___y_1090_, v___y_1091_, v___y_1092_);
lean_dec_ref_known(v___x_1102_, 3);
if (lean_obj_tag(v___x_1107_) == 0)
{
lean_object* v_a_1108_; lean_object* v_snd_1109_; lean_object* v_fst_1110_; lean_object* v_fst_1111_; lean_object* v_snd_1112_; lean_object* v___x_1114_; uint8_t v_isShared_1115_; uint8_t v_isSharedCheck_1159_; 
v_a_1108_ = lean_ctor_get(v___x_1107_, 0);
lean_inc(v_a_1108_);
lean_dec_ref_known(v___x_1107_, 1);
v_snd_1109_ = lean_ctor_get(v_a_1108_, 1);
lean_inc(v_snd_1109_);
v_fst_1110_ = lean_ctor_get(v_a_1108_, 0);
lean_inc(v_fst_1110_);
lean_dec(v_a_1108_);
v_fst_1111_ = lean_ctor_get(v_snd_1109_, 0);
v_snd_1112_ = lean_ctor_get(v_snd_1109_, 1);
v_isSharedCheck_1159_ = !lean_is_exclusive(v_snd_1109_);
if (v_isSharedCheck_1159_ == 0)
{
v___x_1114_ = v_snd_1109_;
v_isShared_1115_ = v_isSharedCheck_1159_;
goto v_resetjp_1113_;
}
else
{
lean_inc(v_snd_1112_);
lean_inc(v_fst_1111_);
lean_dec(v_snd_1109_);
v___x_1114_ = lean_box(0);
v_isShared_1115_ = v_isSharedCheck_1159_;
goto v_resetjp_1113_;
}
v_resetjp_1113_:
{
lean_object* v___x_1117_; 
if (v_isShared_1115_ == 0)
{
v___x_1117_ = v___x_1114_;
goto v_reusejp_1116_;
}
else
{
lean_object* v_reuseFailAlloc_1158_; 
v_reuseFailAlloc_1158_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1158_, 0, v_fst_1111_);
lean_ctor_set(v_reuseFailAlloc_1158_, 1, v_snd_1112_);
v___x_1117_ = v_reuseFailAlloc_1158_;
goto v_reusejp_1116_;
}
v_reusejp_1116_:
{
size_t v_sz_1118_; size_t v___x_1119_; lean_object* v___x_1120_; 
v_sz_1118_ = lean_array_size(v___y_1084_);
v___x_1119_ = ((size_t)0ULL);
v___x_1120_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__3(v___y_1085_, v___y_1084_, v_sz_1118_, v___x_1119_, v___x_1117_, v___y_1087_, v___y_1088_, v___y_1089_, v___y_1090_, v___y_1091_, v___y_1092_);
if (lean_obj_tag(v___x_1120_) == 0)
{
lean_object* v_a_1121_; lean_object* v___x_1123_; uint8_t v_isShared_1124_; uint8_t v_isSharedCheck_1149_; 
v_a_1121_ = lean_ctor_get(v___x_1120_, 0);
v_isSharedCheck_1149_ = !lean_is_exclusive(v___x_1120_);
if (v_isSharedCheck_1149_ == 0)
{
v___x_1123_ = v___x_1120_;
v_isShared_1124_ = v_isSharedCheck_1149_;
goto v_resetjp_1122_;
}
else
{
lean_inc(v_a_1121_);
lean_dec(v___x_1120_);
v___x_1123_ = lean_box(0);
v_isShared_1124_ = v_isSharedCheck_1149_;
goto v_resetjp_1122_;
}
v_resetjp_1122_:
{
lean_object* v_snd_1125_; uint8_t v___x_1126_; 
v_snd_1125_ = lean_ctor_get(v_a_1121_, 1);
v___x_1126_ = lean_unbox(v_snd_1125_);
if (v___x_1126_ == 0)
{
lean_object* v_fst_1127_; lean_object* v___x_1129_; uint8_t v_isShared_1130_; uint8_t v_isSharedCheck_1137_; 
lean_dec(v___x_1082_);
v_fst_1127_ = lean_ctor_get(v_a_1121_, 0);
v_isSharedCheck_1137_ = !lean_is_exclusive(v_a_1121_);
if (v_isSharedCheck_1137_ == 0)
{
lean_object* v_unused_1138_; 
v_unused_1138_ = lean_ctor_get(v_a_1121_, 1);
lean_dec(v_unused_1138_);
v___x_1129_ = v_a_1121_;
v_isShared_1130_ = v_isSharedCheck_1137_;
goto v_resetjp_1128_;
}
else
{
lean_inc(v_fst_1127_);
lean_dec(v_a_1121_);
v___x_1129_ = lean_box(0);
v_isShared_1130_ = v_isSharedCheck_1137_;
goto v_resetjp_1128_;
}
v_resetjp_1128_:
{
lean_object* v___x_1132_; 
if (v_isShared_1130_ == 0)
{
lean_ctor_set(v___x_1129_, 1, v_fst_1127_);
lean_ctor_set(v___x_1129_, 0, v_fst_1110_);
v___x_1132_ = v___x_1129_;
goto v_reusejp_1131_;
}
else
{
lean_object* v_reuseFailAlloc_1136_; 
v_reuseFailAlloc_1136_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1136_, 0, v_fst_1110_);
lean_ctor_set(v_reuseFailAlloc_1136_, 1, v_fst_1127_);
v___x_1132_ = v_reuseFailAlloc_1136_;
goto v_reusejp_1131_;
}
v_reusejp_1131_:
{
lean_object* v___x_1134_; 
if (v_isShared_1124_ == 0)
{
lean_ctor_set(v___x_1123_, 0, v___x_1132_);
v___x_1134_ = v___x_1123_;
goto v_reusejp_1133_;
}
else
{
lean_object* v_reuseFailAlloc_1135_; 
v_reuseFailAlloc_1135_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1135_, 0, v___x_1132_);
v___x_1134_ = v_reuseFailAlloc_1135_;
goto v_reusejp_1133_;
}
v_reusejp_1133_:
{
return v___x_1134_;
}
}
}
}
else
{
lean_object* v_fst_1139_; lean_object* v___x_1141_; uint8_t v_isShared_1142_; uint8_t v_isSharedCheck_1147_; 
lean_del_object(v___x_1123_);
v_fst_1139_ = lean_ctor_get(v_a_1121_, 0);
v_isSharedCheck_1147_ = !lean_is_exclusive(v_a_1121_);
if (v_isSharedCheck_1147_ == 0)
{
lean_object* v_unused_1148_; 
v_unused_1148_ = lean_ctor_get(v_a_1121_, 1);
lean_dec(v_unused_1148_);
v___x_1141_ = v_a_1121_;
v_isShared_1142_ = v_isSharedCheck_1147_;
goto v_resetjp_1140_;
}
else
{
lean_inc(v_fst_1139_);
lean_dec(v_a_1121_);
v___x_1141_ = lean_box(0);
v_isShared_1142_ = v_isSharedCheck_1147_;
goto v_resetjp_1140_;
}
v_resetjp_1140_:
{
lean_object* v___x_1144_; 
if (v_isShared_1142_ == 0)
{
lean_ctor_set(v___x_1141_, 1, v_fst_1139_);
lean_ctor_set(v___x_1141_, 0, v_fst_1110_);
v___x_1144_ = v___x_1141_;
goto v_reusejp_1143_;
}
else
{
lean_object* v_reuseFailAlloc_1146_; 
v_reuseFailAlloc_1146_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1146_, 0, v_fst_1110_);
lean_ctor_set(v_reuseFailAlloc_1146_, 1, v_fst_1139_);
v___x_1144_ = v_reuseFailAlloc_1146_;
goto v_reusejp_1143_;
}
v_reusejp_1143_:
{
v_a_1086_ = v___x_1144_;
goto _start;
}
}
}
}
}
else
{
lean_object* v_a_1150_; lean_object* v___x_1152_; uint8_t v_isShared_1153_; uint8_t v_isSharedCheck_1157_; 
lean_dec(v_fst_1110_);
lean_dec(v___x_1082_);
v_a_1150_ = lean_ctor_get(v___x_1120_, 0);
v_isSharedCheck_1157_ = !lean_is_exclusive(v___x_1120_);
if (v_isSharedCheck_1157_ == 0)
{
v___x_1152_ = v___x_1120_;
v_isShared_1153_ = v_isSharedCheck_1157_;
goto v_resetjp_1151_;
}
else
{
lean_inc(v_a_1150_);
lean_dec(v___x_1120_);
v___x_1152_ = lean_box(0);
v_isShared_1153_ = v_isSharedCheck_1157_;
goto v_resetjp_1151_;
}
v_resetjp_1151_:
{
lean_object* v___x_1155_; 
if (v_isShared_1153_ == 0)
{
v___x_1155_ = v___x_1152_;
goto v_reusejp_1154_;
}
else
{
lean_object* v_reuseFailAlloc_1156_; 
v_reuseFailAlloc_1156_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1156_, 0, v_a_1150_);
v___x_1155_ = v_reuseFailAlloc_1156_;
goto v_reusejp_1154_;
}
v_reusejp_1154_:
{
return v___x_1155_;
}
}
}
}
}
}
else
{
lean_object* v_a_1160_; lean_object* v___x_1162_; uint8_t v_isShared_1163_; uint8_t v_isSharedCheck_1167_; 
lean_dec(v___x_1082_);
v_a_1160_ = lean_ctor_get(v___x_1107_, 0);
v_isSharedCheck_1167_ = !lean_is_exclusive(v___x_1107_);
if (v_isSharedCheck_1167_ == 0)
{
v___x_1162_ = v___x_1107_;
v_isShared_1163_ = v_isSharedCheck_1167_;
goto v_resetjp_1161_;
}
else
{
lean_inc(v_a_1160_);
lean_dec(v___x_1107_);
v___x_1162_ = lean_box(0);
v_isShared_1163_ = v_isSharedCheck_1167_;
goto v_resetjp_1161_;
}
v_resetjp_1161_:
{
lean_object* v___x_1165_; 
if (v_isShared_1163_ == 0)
{
v___x_1165_ = v___x_1162_;
goto v_reusejp_1164_;
}
else
{
lean_object* v_reuseFailAlloc_1166_; 
v_reuseFailAlloc_1166_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1166_, 0, v_a_1160_);
v___x_1165_ = v_reuseFailAlloc_1166_;
goto v_reusejp_1164_;
}
v_reusejp_1164_:
{
return v___x_1165_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__4___redArg___boxed(lean_object* v___x_1170_, lean_object* v___y_1171_, lean_object* v___y_1172_, lean_object* v___y_1173_, lean_object* v_a_1174_, lean_object* v___y_1175_, lean_object* v___y_1176_, lean_object* v___y_1177_, lean_object* v___y_1178_, lean_object* v___y_1179_, lean_object* v___y_1180_, lean_object* v___y_1181_){
_start:
{
lean_object* v_res_1182_; 
v_res_1182_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__4___redArg(v___x_1170_, v___y_1171_, v___y_1172_, v___y_1173_, v_a_1174_, v___y_1175_, v___y_1176_, v___y_1177_, v___y_1178_, v___y_1179_, v___y_1180_);
lean_dec(v___y_1180_);
lean_dec_ref(v___y_1179_);
lean_dec(v___y_1178_);
lean_dec_ref(v___y_1177_);
lean_dec(v___y_1176_);
lean_dec_ref(v___y_1175_);
lean_dec_ref(v___y_1173_);
lean_dec_ref(v___y_1172_);
lean_dec_ref(v___y_1171_);
return v_res_1182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__10(lean_object* v_as_1183_, size_t v_i_1184_, size_t v_stop_1185_, lean_object* v_b_1186_){
_start:
{
lean_object* v___y_1188_; uint8_t v___x_1192_; 
v___x_1192_ = lean_usize_dec_eq(v_i_1184_, v_stop_1185_);
if (v___x_1192_ == 0)
{
lean_object* v___x_1193_; 
v___x_1193_ = lean_array_uget_borrowed(v_as_1183_, v_i_1184_);
switch(lean_obj_tag(v___x_1193_))
{
case 8:
{
lean_object* v___x_1194_; 
lean_inc_ref(v___x_1193_);
v___x_1194_ = lean_array_push(v_b_1186_, v___x_1193_);
v___y_1188_ = v___x_1194_;
goto v___jp_1187_;
}
case 9:
{
lean_object* v___x_1195_; 
lean_inc_ref(v___x_1193_);
v___x_1195_ = lean_array_push(v_b_1186_, v___x_1193_);
v___y_1188_ = v___x_1195_;
goto v___jp_1187_;
}
default: 
{
v___y_1188_ = v_b_1186_;
goto v___jp_1187_;
}
}
}
else
{
return v_b_1186_;
}
v___jp_1187_:
{
size_t v___x_1189_; size_t v___x_1190_; 
v___x_1189_ = ((size_t)1ULL);
v___x_1190_ = lean_usize_add(v_i_1184_, v___x_1189_);
v_i_1184_ = v___x_1190_;
v_b_1186_ = v___y_1188_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__10___boxed(lean_object* v_as_1196_, lean_object* v_i_1197_, lean_object* v_stop_1198_, lean_object* v_b_1199_){
_start:
{
size_t v_i_boxed_1200_; size_t v_stop_boxed_1201_; lean_object* v_res_1202_; 
v_i_boxed_1200_ = lean_unbox_usize(v_i_1197_);
lean_dec(v_i_1197_);
v_stop_boxed_1201_ = lean_unbox_usize(v_stop_1198_);
lean_dec(v_stop_1198_);
v_res_1202_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__10(v_as_1196_, v_i_boxed_1200_, v_stop_boxed_1201_, v_b_1199_);
lean_dec_ref(v_as_1196_);
return v_res_1202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__5_spec__5_spec__6_spec__13___redArg(lean_object* v_x_1203_, lean_object* v_x_1204_){
_start:
{
if (lean_obj_tag(v_x_1204_) == 0)
{
return v_x_1203_;
}
else
{
lean_object* v_key_1205_; lean_object* v_value_1206_; lean_object* v_tail_1207_; lean_object* v___x_1209_; uint8_t v_isShared_1210_; uint8_t v_isSharedCheck_1230_; 
v_key_1205_ = lean_ctor_get(v_x_1204_, 0);
v_value_1206_ = lean_ctor_get(v_x_1204_, 1);
v_tail_1207_ = lean_ctor_get(v_x_1204_, 2);
v_isSharedCheck_1230_ = !lean_is_exclusive(v_x_1204_);
if (v_isSharedCheck_1230_ == 0)
{
v___x_1209_ = v_x_1204_;
v_isShared_1210_ = v_isSharedCheck_1230_;
goto v_resetjp_1208_;
}
else
{
lean_inc(v_tail_1207_);
lean_inc(v_value_1206_);
lean_inc(v_key_1205_);
lean_dec(v_x_1204_);
v___x_1209_ = lean_box(0);
v_isShared_1210_ = v_isSharedCheck_1230_;
goto v_resetjp_1208_;
}
v_resetjp_1208_:
{
lean_object* v___x_1211_; uint64_t v___x_1212_; uint64_t v___x_1213_; uint64_t v___x_1214_; uint64_t v_fold_1215_; uint64_t v___x_1216_; uint64_t v___x_1217_; uint64_t v___x_1218_; size_t v___x_1219_; size_t v___x_1220_; size_t v___x_1221_; size_t v___x_1222_; size_t v___x_1223_; lean_object* v___x_1224_; lean_object* v___x_1226_; 
v___x_1211_ = lean_array_get_size(v_x_1203_);
v___x_1212_ = lean_uint64_of_nat(v_key_1205_);
v___x_1213_ = 32ULL;
v___x_1214_ = lean_uint64_shift_right(v___x_1212_, v___x_1213_);
v_fold_1215_ = lean_uint64_xor(v___x_1212_, v___x_1214_);
v___x_1216_ = 16ULL;
v___x_1217_ = lean_uint64_shift_right(v_fold_1215_, v___x_1216_);
v___x_1218_ = lean_uint64_xor(v_fold_1215_, v___x_1217_);
v___x_1219_ = lean_uint64_to_usize(v___x_1218_);
v___x_1220_ = lean_usize_of_nat(v___x_1211_);
v___x_1221_ = ((size_t)1ULL);
v___x_1222_ = lean_usize_sub(v___x_1220_, v___x_1221_);
v___x_1223_ = lean_usize_land(v___x_1219_, v___x_1222_);
v___x_1224_ = lean_array_uget_borrowed(v_x_1203_, v___x_1223_);
lean_inc(v___x_1224_);
if (v_isShared_1210_ == 0)
{
lean_ctor_set(v___x_1209_, 2, v___x_1224_);
v___x_1226_ = v___x_1209_;
goto v_reusejp_1225_;
}
else
{
lean_object* v_reuseFailAlloc_1229_; 
v_reuseFailAlloc_1229_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1229_, 0, v_key_1205_);
lean_ctor_set(v_reuseFailAlloc_1229_, 1, v_value_1206_);
lean_ctor_set(v_reuseFailAlloc_1229_, 2, v___x_1224_);
v___x_1226_ = v_reuseFailAlloc_1229_;
goto v_reusejp_1225_;
}
v_reusejp_1225_:
{
lean_object* v___x_1227_; 
v___x_1227_ = lean_array_uset(v_x_1203_, v___x_1223_, v___x_1226_);
v_x_1203_ = v___x_1227_;
v_x_1204_ = v_tail_1207_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__5_spec__5_spec__6___redArg(lean_object* v_i_1231_, lean_object* v_source_1232_, lean_object* v_target_1233_){
_start:
{
lean_object* v___x_1234_; uint8_t v___x_1235_; 
v___x_1234_ = lean_array_get_size(v_source_1232_);
v___x_1235_ = lean_nat_dec_lt(v_i_1231_, v___x_1234_);
if (v___x_1235_ == 0)
{
lean_dec_ref(v_source_1232_);
lean_dec(v_i_1231_);
return v_target_1233_;
}
else
{
lean_object* v_es_1236_; lean_object* v___x_1237_; lean_object* v_source_1238_; lean_object* v_target_1239_; lean_object* v___x_1240_; lean_object* v___x_1241_; 
v_es_1236_ = lean_array_fget(v_source_1232_, v_i_1231_);
v___x_1237_ = lean_box(0);
v_source_1238_ = lean_array_fset(v_source_1232_, v_i_1231_, v___x_1237_);
v_target_1239_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__5_spec__5_spec__6_spec__13___redArg(v_target_1233_, v_es_1236_);
v___x_1240_ = lean_unsigned_to_nat(1u);
v___x_1241_ = lean_nat_add(v_i_1231_, v___x_1240_);
lean_dec(v_i_1231_);
v_i_1231_ = v___x_1241_;
v_source_1232_ = v_source_1238_;
v_target_1233_ = v_target_1239_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__5_spec__5___redArg(lean_object* v_data_1243_){
_start:
{
lean_object* v___x_1244_; lean_object* v___x_1245_; lean_object* v_nbuckets_1246_; lean_object* v___x_1247_; lean_object* v___x_1248_; lean_object* v___x_1249_; lean_object* v___x_1250_; 
v___x_1244_ = lean_array_get_size(v_data_1243_);
v___x_1245_ = lean_unsigned_to_nat(2u);
v_nbuckets_1246_ = lean_nat_mul(v___x_1244_, v___x_1245_);
v___x_1247_ = lean_unsigned_to_nat(0u);
v___x_1248_ = lean_box(0);
v___x_1249_ = lean_mk_array(v_nbuckets_1246_, v___x_1248_);
v___x_1250_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__5_spec__5_spec__6___redArg(v___x_1247_, v_data_1243_, v___x_1249_);
return v___x_1250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__5___redArg(lean_object* v_m_1251_, lean_object* v_a_1252_, lean_object* v_b_1253_){
_start:
{
lean_object* v_size_1254_; lean_object* v_buckets_1255_; lean_object* v___x_1256_; uint64_t v___x_1257_; uint64_t v___x_1258_; uint64_t v___x_1259_; uint64_t v_fold_1260_; uint64_t v___x_1261_; uint64_t v___x_1262_; uint64_t v___x_1263_; size_t v___x_1264_; size_t v___x_1265_; size_t v___x_1266_; size_t v___x_1267_; size_t v___x_1268_; lean_object* v_bkt_1269_; uint8_t v___x_1270_; 
v_size_1254_ = lean_ctor_get(v_m_1251_, 0);
v_buckets_1255_ = lean_ctor_get(v_m_1251_, 1);
v___x_1256_ = lean_array_get_size(v_buckets_1255_);
v___x_1257_ = lean_uint64_of_nat(v_a_1252_);
v___x_1258_ = 32ULL;
v___x_1259_ = lean_uint64_shift_right(v___x_1257_, v___x_1258_);
v_fold_1260_ = lean_uint64_xor(v___x_1257_, v___x_1259_);
v___x_1261_ = 16ULL;
v___x_1262_ = lean_uint64_shift_right(v_fold_1260_, v___x_1261_);
v___x_1263_ = lean_uint64_xor(v_fold_1260_, v___x_1262_);
v___x_1264_ = lean_uint64_to_usize(v___x_1263_);
v___x_1265_ = lean_usize_of_nat(v___x_1256_);
v___x_1266_ = ((size_t)1ULL);
v___x_1267_ = lean_usize_sub(v___x_1265_, v___x_1266_);
v___x_1268_ = lean_usize_land(v___x_1264_, v___x_1267_);
v_bkt_1269_ = lean_array_uget_borrowed(v_buckets_1255_, v___x_1268_);
v___x_1270_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_findContradictionWithNe_spec__0_spec__0___redArg(v_a_1252_, v_bkt_1269_);
if (v___x_1270_ == 0)
{
lean_object* v___x_1272_; uint8_t v_isShared_1273_; uint8_t v_isSharedCheck_1291_; 
lean_inc_ref(v_buckets_1255_);
lean_inc(v_size_1254_);
v_isSharedCheck_1291_ = !lean_is_exclusive(v_m_1251_);
if (v_isSharedCheck_1291_ == 0)
{
lean_object* v_unused_1292_; lean_object* v_unused_1293_; 
v_unused_1292_ = lean_ctor_get(v_m_1251_, 1);
lean_dec(v_unused_1292_);
v_unused_1293_ = lean_ctor_get(v_m_1251_, 0);
lean_dec(v_unused_1293_);
v___x_1272_ = v_m_1251_;
v_isShared_1273_ = v_isSharedCheck_1291_;
goto v_resetjp_1271_;
}
else
{
lean_dec(v_m_1251_);
v___x_1272_ = lean_box(0);
v_isShared_1273_ = v_isSharedCheck_1291_;
goto v_resetjp_1271_;
}
v_resetjp_1271_:
{
lean_object* v___x_1274_; lean_object* v_size_x27_1275_; lean_object* v___x_1276_; lean_object* v_buckets_x27_1277_; lean_object* v___x_1278_; lean_object* v___x_1279_; lean_object* v___x_1280_; lean_object* v___x_1281_; lean_object* v___x_1282_; uint8_t v___x_1283_; 
v___x_1274_ = lean_unsigned_to_nat(1u);
v_size_x27_1275_ = lean_nat_add(v_size_1254_, v___x_1274_);
lean_dec(v_size_1254_);
lean_inc(v_bkt_1269_);
v___x_1276_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1276_, 0, v_a_1252_);
lean_ctor_set(v___x_1276_, 1, v_b_1253_);
lean_ctor_set(v___x_1276_, 2, v_bkt_1269_);
v_buckets_x27_1277_ = lean_array_uset(v_buckets_1255_, v___x_1268_, v___x_1276_);
v___x_1278_ = lean_unsigned_to_nat(4u);
v___x_1279_ = lean_nat_mul(v_size_x27_1275_, v___x_1278_);
v___x_1280_ = lean_unsigned_to_nat(3u);
v___x_1281_ = lean_nat_div(v___x_1279_, v___x_1280_);
lean_dec(v___x_1279_);
v___x_1282_ = lean_array_get_size(v_buckets_x27_1277_);
v___x_1283_ = lean_nat_dec_le(v___x_1281_, v___x_1282_);
lean_dec(v___x_1281_);
if (v___x_1283_ == 0)
{
lean_object* v_val_1284_; lean_object* v___x_1286_; 
v_val_1284_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__5_spec__5___redArg(v_buckets_x27_1277_);
if (v_isShared_1273_ == 0)
{
lean_ctor_set(v___x_1272_, 1, v_val_1284_);
lean_ctor_set(v___x_1272_, 0, v_size_x27_1275_);
v___x_1286_ = v___x_1272_;
goto v_reusejp_1285_;
}
else
{
lean_object* v_reuseFailAlloc_1287_; 
v_reuseFailAlloc_1287_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1287_, 0, v_size_x27_1275_);
lean_ctor_set(v_reuseFailAlloc_1287_, 1, v_val_1284_);
v___x_1286_ = v_reuseFailAlloc_1287_;
goto v_reusejp_1285_;
}
v_reusejp_1285_:
{
return v___x_1286_;
}
}
else
{
lean_object* v___x_1289_; 
if (v_isShared_1273_ == 0)
{
lean_ctor_set(v___x_1272_, 1, v_buckets_x27_1277_);
lean_ctor_set(v___x_1272_, 0, v_size_x27_1275_);
v___x_1289_ = v___x_1272_;
goto v_reusejp_1288_;
}
else
{
lean_object* v_reuseFailAlloc_1290_; 
v_reuseFailAlloc_1290_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1290_, 0, v_size_x27_1275_);
lean_ctor_set(v_reuseFailAlloc_1290_, 1, v_buckets_x27_1277_);
v___x_1289_ = v_reuseFailAlloc_1290_;
goto v_reusejp_1288_;
}
v_reusejp_1288_:
{
return v___x_1289_;
}
}
}
}
else
{
lean_dec(v_b_1253_);
lean_dec(v_a_1252_);
return v_m_1251_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__7_spec__8(lean_object* v_as_1294_, size_t v_sz_1295_, size_t v_i_1296_, lean_object* v_b_1297_){
_start:
{
uint8_t v___x_1298_; 
v___x_1298_ = lean_usize_dec_lt(v_i_1296_, v_sz_1295_);
if (v___x_1298_ == 0)
{
return v_b_1297_;
}
else
{
lean_object* v_a_1299_; lean_object* v___x_1300_; lean_object* v_r_1301_; size_t v___x_1302_; size_t v___x_1303_; 
v_a_1299_ = lean_array_uget_borrowed(v_as_1294_, v_i_1296_);
v___x_1300_ = lean_box(0);
lean_inc(v_a_1299_);
v_r_1301_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__5___redArg(v_b_1297_, v_a_1299_, v___x_1300_);
v___x_1302_ = ((size_t)1ULL);
v___x_1303_ = lean_usize_add(v_i_1296_, v___x_1302_);
v_i_1296_ = v___x_1303_;
v_b_1297_ = v_r_1301_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__7_spec__8___boxed(lean_object* v_as_1305_, lean_object* v_sz_1306_, lean_object* v_i_1307_, lean_object* v_b_1308_){
_start:
{
size_t v_sz_boxed_1309_; size_t v_i_boxed_1310_; lean_object* v_res_1311_; 
v_sz_boxed_1309_ = lean_unbox_usize(v_sz_1306_);
lean_dec(v_sz_1306_);
v_i_boxed_1310_ = lean_unbox_usize(v_i_1307_);
lean_dec(v_i_1307_);
v_res_1311_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__7_spec__8(v_as_1305_, v_sz_boxed_1309_, v_i_boxed_1310_, v_b_1308_);
lean_dec_ref(v_as_1305_);
return v_res_1311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__7(lean_object* v_m_1312_, lean_object* v_l_1313_){
_start:
{
size_t v_sz_1314_; size_t v___x_1315_; lean_object* v___x_1316_; 
v_sz_1314_ = lean_array_size(v_l_1313_);
v___x_1315_ = ((size_t)0ULL);
v___x_1316_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__7_spec__8(v_l_1313_, v_sz_1314_, v___x_1315_, v_m_1312_);
return v___x_1316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__7___boxed(lean_object* v_m_1317_, lean_object* v_l_1318_){
_start:
{
lean_object* v_res_1319_; 
v_res_1319_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__7(v_m_1317_, v_l_1318_);
lean_dec_ref(v_l_1318_);
return v_res_1319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__6(size_t v_sz_1320_, size_t v_i_1321_, lean_object* v_bs_1322_){
_start:
{
uint8_t v___x_1323_; 
v___x_1323_ = lean_usize_dec_lt(v_i_1321_, v_sz_1320_);
if (v___x_1323_ == 0)
{
return v_bs_1322_;
}
else
{
lean_object* v_v_1324_; lean_object* v_dst_1325_; lean_object* v___x_1326_; lean_object* v_bs_x27_1327_; size_t v___x_1328_; size_t v___x_1329_; lean_object* v___x_1330_; 
v_v_1324_ = lean_array_uget_borrowed(v_bs_1322_, v_i_1321_);
v_dst_1325_ = lean_ctor_get(v_v_1324_, 1);
lean_inc(v_dst_1325_);
v___x_1326_ = lean_unsigned_to_nat(0u);
v_bs_x27_1327_ = lean_array_uset(v_bs_1322_, v_i_1321_, v___x_1326_);
v___x_1328_ = ((size_t)1ULL);
v___x_1329_ = lean_usize_add(v_i_1321_, v___x_1328_);
v___x_1330_ = lean_array_uset(v_bs_x27_1327_, v_i_1321_, v_dst_1325_);
v_i_1321_ = v___x_1329_;
v_bs_1322_ = v___x_1330_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__6___boxed(lean_object* v_sz_1332_, lean_object* v_i_1333_, lean_object* v_bs_1334_){
_start:
{
size_t v_sz_boxed_1335_; size_t v_i_boxed_1336_; lean_object* v_res_1337_; 
v_sz_boxed_1335_ = lean_unbox_usize(v_sz_1332_);
lean_dec(v_sz_1332_);
v_i_boxed_1336_ = lean_unbox_usize(v_i_1333_);
lean_dec(v_i_1333_);
v_res_1337_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__6(v_sz_boxed_1335_, v_i_boxed_1336_, v_bs_1334_);
return v_res_1337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__8(lean_object* v_x_1338_, lean_object* v_x_1339_){
_start:
{
if (lean_obj_tag(v_x_1339_) == 0)
{
return v_x_1338_;
}
else
{
lean_object* v_key_1340_; lean_object* v_value_1341_; lean_object* v_tail_1342_; lean_object* v___x_1343_; lean_object* v___x_1344_; size_t v_sz_1345_; size_t v___x_1346_; lean_object* v___x_1347_; lean_object* v___x_1348_; 
v_key_1340_ = lean_ctor_get(v_x_1339_, 0);
lean_inc(v_key_1340_);
v_value_1341_ = lean_ctor_get(v_x_1339_, 1);
lean_inc(v_value_1341_);
v_tail_1342_ = lean_ctor_get(v_x_1339_, 2);
lean_inc(v_tail_1342_);
lean_dec_ref_known(v_x_1339_, 3);
v___x_1343_ = lean_box(0);
v___x_1344_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__5___redArg(v_x_1338_, v_key_1340_, v___x_1343_);
v_sz_1345_ = lean_array_size(v_value_1341_);
v___x_1346_ = ((size_t)0ULL);
v___x_1347_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__6(v_sz_1345_, v___x_1346_, v_value_1341_);
v___x_1348_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__7(v___x_1344_, v___x_1347_);
lean_dec_ref(v___x_1347_);
v_x_1338_ = v___x_1348_;
v_x_1339_ = v_tail_1342_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__9(lean_object* v_as_1350_, size_t v_i_1351_, size_t v_stop_1352_, lean_object* v_b_1353_){
_start:
{
uint8_t v___x_1354_; 
v___x_1354_ = lean_usize_dec_eq(v_i_1351_, v_stop_1352_);
if (v___x_1354_ == 0)
{
lean_object* v___x_1355_; lean_object* v___x_1356_; size_t v___x_1357_; size_t v___x_1358_; 
v___x_1355_ = lean_array_uget_borrowed(v_as_1350_, v_i_1351_);
lean_inc(v___x_1355_);
v___x_1356_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__8(v_b_1353_, v___x_1355_);
v___x_1357_ = ((size_t)1ULL);
v___x_1358_ = lean_usize_add(v_i_1351_, v___x_1357_);
v_i_1351_ = v___x_1358_;
v_b_1353_ = v___x_1356_;
goto _start;
}
else
{
return v_b_1353_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__9___boxed(lean_object* v_as_1360_, lean_object* v_i_1361_, lean_object* v_stop_1362_, lean_object* v_b_1363_){
_start:
{
size_t v_i_boxed_1364_; size_t v_stop_boxed_1365_; lean_object* v_res_1366_; 
v_i_boxed_1364_ = lean_unbox_usize(v_i_1361_);
lean_dec(v_i_1361_);
v_stop_boxed_1365_ = lean_unbox_usize(v_stop_1362_);
lean_dec(v_stop_1362_);
v_res_1366_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__9(v_as_1360_, v_i_boxed_1364_, v_stop_boxed_1365_, v_b_1363_);
lean_dec_ref(v_as_1360_);
return v_res_1366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__11(lean_object* v_as_1367_, size_t v_i_1368_, size_t v_stop_1369_, lean_object* v_b_1370_){
_start:
{
lean_object* v___y_1372_; uint8_t v___x_1376_; 
v___x_1376_ = lean_usize_dec_eq(v_i_1368_, v_stop_1369_);
if (v___x_1376_ == 0)
{
lean_object* v___x_1377_; 
v___x_1377_ = lean_array_uget_borrowed(v_as_1367_, v_i_1368_);
if (lean_obj_tag(v___x_1377_) == 5)
{
lean_object* v___x_1378_; 
lean_inc_ref(v___x_1377_);
v___x_1378_ = lean_array_push(v_b_1370_, v___x_1377_);
v___y_1372_ = v___x_1378_;
goto v___jp_1371_;
}
else
{
v___y_1372_ = v_b_1370_;
goto v___jp_1371_;
}
}
else
{
return v_b_1370_;
}
v___jp_1371_:
{
size_t v___x_1373_; size_t v___x_1374_; 
v___x_1373_ = ((size_t)1ULL);
v___x_1374_ = lean_usize_add(v_i_1368_, v___x_1373_);
v_i_1368_ = v___x_1374_;
v_b_1370_ = v___y_1372_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__11___boxed(lean_object* v_as_1379_, lean_object* v_i_1380_, lean_object* v_stop_1381_, lean_object* v_b_1382_){
_start:
{
size_t v_i_boxed_1383_; size_t v_stop_boxed_1384_; lean_object* v_res_1385_; 
v_i_boxed_1383_ = lean_unbox_usize(v_i_1380_);
lean_dec(v_i_1380_);
v_stop_boxed_1384_ = lean_unbox_usize(v_stop_1381_);
lean_dec(v_stop_1381_);
v_res_1385_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__11(v_as_1379_, v_i_boxed_1383_, v_stop_boxed_1384_, v_b_1382_);
lean_dec_ref(v_as_1379_);
return v_res_1385_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order_updateGraphWithNltInfSup___closed__0(void){
_start:
{
lean_object* v___x_1386_; lean_object* v___x_1387_; lean_object* v___x_1388_; 
v___x_1386_ = lean_box(0);
v___x_1387_ = lean_unsigned_to_nat(16u);
v___x_1388_ = lean_mk_array(v___x_1387_, v___x_1386_);
return v___x_1388_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order_updateGraphWithNltInfSup___closed__1(void){
_start:
{
lean_object* v___x_1389_; lean_object* v___x_1390_; lean_object* v___x_1391_; 
v___x_1389_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Order_updateGraphWithNltInfSup___closed__0, &lp_mathlib_Mathlib_Tactic_Order_updateGraphWithNltInfSup___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Order_updateGraphWithNltInfSup___closed__0);
v___x_1390_ = lean_unsigned_to_nat(0u);
v___x_1391_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1391_, 0, v___x_1390_);
lean_ctor_set(v___x_1391_, 1, v___x_1389_);
return v___x_1391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_updateGraphWithNltInfSup(lean_object* v_g_1394_, lean_object* v_facts_1395_, lean_object* v_a_1396_, lean_object* v_a_1397_, lean_object* v_a_1398_, lean_object* v_a_1399_, lean_object* v_a_1400_, lean_object* v_a_1401_){
_start:
{
lean_object* v___y_1404_; lean_object* v___y_1405_; lean_object* v___y_1406_; lean_object* v___y_1407_; lean_object* v___y_1408_; lean_object* v___x_1428_; lean_object* v___y_1430_; lean_object* v___y_1431_; lean_object* v___y_1432_; lean_object* v___y_1433_; lean_object* v___x_1445_; lean_object* v___y_1447_; lean_object* v___x_1461_; uint8_t v___x_1462_; 
v___x_1428_ = lean_unsigned_to_nat(0u);
v___x_1445_ = lean_array_get_size(v_facts_1395_);
v___x_1461_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_updateGraphWithNltInfSup___closed__2));
v___x_1462_ = lean_nat_dec_lt(v___x_1428_, v___x_1445_);
if (v___x_1462_ == 0)
{
v___y_1447_ = v___x_1461_;
goto v___jp_1446_;
}
else
{
uint8_t v___x_1463_; 
v___x_1463_ = lean_nat_dec_le(v___x_1445_, v___x_1445_);
if (v___x_1463_ == 0)
{
if (v___x_1462_ == 0)
{
v___y_1447_ = v___x_1461_;
goto v___jp_1446_;
}
else
{
size_t v___x_1464_; size_t v___x_1465_; lean_object* v___x_1466_; 
v___x_1464_ = ((size_t)0ULL);
v___x_1465_ = lean_usize_of_nat(v___x_1445_);
v___x_1466_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__11(v_facts_1395_, v___x_1464_, v___x_1465_, v___x_1461_);
v___y_1447_ = v___x_1466_;
goto v___jp_1446_;
}
}
else
{
size_t v___x_1467_; size_t v___x_1468_; lean_object* v___x_1469_; 
v___x_1467_ = ((size_t)0ULL);
v___x_1468_ = lean_usize_of_nat(v___x_1445_);
v___x_1469_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__11(v_facts_1395_, v___x_1467_, v___x_1468_, v___x_1461_);
v___y_1447_ = v___x_1469_;
goto v___jp_1446_;
}
}
v___jp_1403_:
{
lean_object* v___x_1409_; lean_object* v___x_1410_; 
v___x_1409_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1409_, 0, v___y_1406_);
lean_ctor_set(v___x_1409_, 1, v_g_1394_);
v___x_1410_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__4___redArg(v___y_1404_, v___y_1407_, v___y_1405_, v___y_1408_, v___x_1409_, v_a_1396_, v_a_1397_, v_a_1398_, v_a_1399_, v_a_1400_, v_a_1401_);
lean_dec_ref(v___y_1408_);
lean_dec_ref(v___y_1405_);
lean_dec_ref(v___y_1407_);
if (lean_obj_tag(v___x_1410_) == 0)
{
lean_object* v_a_1411_; lean_object* v___x_1413_; uint8_t v_isShared_1414_; uint8_t v_isSharedCheck_1419_; 
v_a_1411_ = lean_ctor_get(v___x_1410_, 0);
v_isSharedCheck_1419_ = !lean_is_exclusive(v___x_1410_);
if (v_isSharedCheck_1419_ == 0)
{
v___x_1413_ = v___x_1410_;
v_isShared_1414_ = v_isSharedCheck_1419_;
goto v_resetjp_1412_;
}
else
{
lean_inc(v_a_1411_);
lean_dec(v___x_1410_);
v___x_1413_ = lean_box(0);
v_isShared_1414_ = v_isSharedCheck_1419_;
goto v_resetjp_1412_;
}
v_resetjp_1412_:
{
lean_object* v_snd_1415_; lean_object* v___x_1417_; 
v_snd_1415_ = lean_ctor_get(v_a_1411_, 1);
lean_inc(v_snd_1415_);
lean_dec(v_a_1411_);
if (v_isShared_1414_ == 0)
{
lean_ctor_set(v___x_1413_, 0, v_snd_1415_);
v___x_1417_ = v___x_1413_;
goto v_reusejp_1416_;
}
else
{
lean_object* v_reuseFailAlloc_1418_; 
v_reuseFailAlloc_1418_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1418_, 0, v_snd_1415_);
v___x_1417_ = v_reuseFailAlloc_1418_;
goto v_reusejp_1416_;
}
v_reusejp_1416_:
{
return v___x_1417_;
}
}
}
else
{
lean_object* v_a_1420_; lean_object* v___x_1422_; uint8_t v_isShared_1423_; uint8_t v_isSharedCheck_1427_; 
v_a_1420_ = lean_ctor_get(v___x_1410_, 0);
v_isSharedCheck_1427_ = !lean_is_exclusive(v___x_1410_);
if (v_isSharedCheck_1427_ == 0)
{
v___x_1422_ = v___x_1410_;
v_isShared_1423_ = v_isSharedCheck_1427_;
goto v_resetjp_1421_;
}
else
{
lean_inc(v_a_1420_);
lean_dec(v___x_1410_);
v___x_1422_ = lean_box(0);
v_isShared_1423_ = v_isSharedCheck_1427_;
goto v_resetjp_1421_;
}
v_resetjp_1421_:
{
lean_object* v___x_1425_; 
if (v_isShared_1423_ == 0)
{
v___x_1425_ = v___x_1422_;
goto v_reusejp_1424_;
}
else
{
lean_object* v_reuseFailAlloc_1426_; 
v_reuseFailAlloc_1426_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1426_, 0, v_a_1420_);
v___x_1425_ = v_reuseFailAlloc_1426_;
goto v_reusejp_1424_;
}
v_reusejp_1424_:
{
return v___x_1425_;
}
}
}
}
v___jp_1429_:
{
lean_object* v_buckets_1434_; lean_object* v___x_1435_; lean_object* v___x_1436_; uint8_t v___x_1437_; 
v_buckets_1434_ = lean_ctor_get(v_g_1394_, 1);
v___x_1435_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Order_updateGraphWithNltInfSup___closed__1, &lp_mathlib_Mathlib_Tactic_Order_updateGraphWithNltInfSup___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Order_updateGraphWithNltInfSup___closed__1);
v___x_1436_ = lean_array_get_size(v_buckets_1434_);
v___x_1437_ = lean_nat_dec_lt(v___x_1428_, v___x_1436_);
if (v___x_1437_ == 0)
{
v___y_1404_ = v___y_1430_;
v___y_1405_ = v___y_1433_;
v___y_1406_ = v___y_1431_;
v___y_1407_ = v___y_1432_;
v___y_1408_ = v___x_1435_;
goto v___jp_1403_;
}
else
{
uint8_t v___x_1438_; 
v___x_1438_ = lean_nat_dec_le(v___x_1436_, v___x_1436_);
if (v___x_1438_ == 0)
{
if (v___x_1437_ == 0)
{
v___y_1404_ = v___y_1430_;
v___y_1405_ = v___y_1433_;
v___y_1406_ = v___y_1431_;
v___y_1407_ = v___y_1432_;
v___y_1408_ = v___x_1435_;
goto v___jp_1403_;
}
else
{
size_t v___x_1439_; size_t v___x_1440_; lean_object* v___x_1441_; 
v___x_1439_ = ((size_t)0ULL);
v___x_1440_ = lean_usize_of_nat(v___x_1436_);
v___x_1441_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__9(v_buckets_1434_, v___x_1439_, v___x_1440_, v___x_1435_);
v___y_1404_ = v___y_1430_;
v___y_1405_ = v___y_1433_;
v___y_1406_ = v___y_1431_;
v___y_1407_ = v___y_1432_;
v___y_1408_ = v___x_1441_;
goto v___jp_1403_;
}
}
else
{
size_t v___x_1442_; size_t v___x_1443_; lean_object* v___x_1444_; 
v___x_1442_ = ((size_t)0ULL);
v___x_1443_ = lean_usize_of_nat(v___x_1436_);
v___x_1444_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__9(v_buckets_1434_, v___x_1442_, v___x_1443_, v___x_1435_);
v___y_1404_ = v___y_1430_;
v___y_1405_ = v___y_1433_;
v___y_1406_ = v___y_1431_;
v___y_1407_ = v___y_1432_;
v___y_1408_ = v___x_1444_;
goto v___jp_1403_;
}
}
}
v___jp_1446_:
{
lean_object* v___x_1448_; uint8_t v_changed_1449_; lean_object* v___x_1450_; lean_object* v_usedNltFacts_1451_; lean_object* v___x_1452_; uint8_t v___x_1453_; 
v___x_1448_ = lean_array_get_size(v___y_1447_);
v_changed_1449_ = 0;
v___x_1450_ = lean_box(v_changed_1449_);
v_usedNltFacts_1451_ = lean_mk_array(v___x_1448_, v___x_1450_);
v___x_1452_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_updateGraphWithNltInfSup___closed__2));
v___x_1453_ = lean_nat_dec_lt(v___x_1428_, v___x_1445_);
if (v___x_1453_ == 0)
{
v___y_1430_ = v___x_1448_;
v___y_1431_ = v_usedNltFacts_1451_;
v___y_1432_ = v___y_1447_;
v___y_1433_ = v___x_1452_;
goto v___jp_1429_;
}
else
{
uint8_t v___x_1454_; 
v___x_1454_ = lean_nat_dec_le(v___x_1445_, v___x_1445_);
if (v___x_1454_ == 0)
{
if (v___x_1453_ == 0)
{
v___y_1430_ = v___x_1448_;
v___y_1431_ = v_usedNltFacts_1451_;
v___y_1432_ = v___y_1447_;
v___y_1433_ = v___x_1452_;
goto v___jp_1429_;
}
else
{
size_t v___x_1455_; size_t v___x_1456_; lean_object* v___x_1457_; 
v___x_1455_ = ((size_t)0ULL);
v___x_1456_ = lean_usize_of_nat(v___x_1445_);
v___x_1457_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__10(v_facts_1395_, v___x_1455_, v___x_1456_, v___x_1452_);
v___y_1430_ = v___x_1448_;
v___y_1431_ = v_usedNltFacts_1451_;
v___y_1432_ = v___y_1447_;
v___y_1433_ = v___x_1457_;
goto v___jp_1429_;
}
}
else
{
size_t v___x_1458_; size_t v___x_1459_; lean_object* v___x_1460_; 
v___x_1458_ = ((size_t)0ULL);
v___x_1459_ = lean_usize_of_nat(v___x_1445_);
v___x_1460_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__10(v_facts_1395_, v___x_1458_, v___x_1459_, v___x_1452_);
v___y_1430_ = v___x_1448_;
v___y_1431_ = v_usedNltFacts_1451_;
v___y_1432_ = v___y_1447_;
v___y_1433_ = v___x_1460_;
goto v___jp_1429_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_updateGraphWithNltInfSup___boxed(lean_object* v_g_1470_, lean_object* v_facts_1471_, lean_object* v_a_1472_, lean_object* v_a_1473_, lean_object* v_a_1474_, lean_object* v_a_1475_, lean_object* v_a_1476_, lean_object* v_a_1477_, lean_object* v_a_1478_){
_start:
{
lean_object* v_res_1479_; 
v_res_1479_ = lp_mathlib_Mathlib_Tactic_Order_updateGraphWithNltInfSup(v_g_1470_, v_facts_1471_, v_a_1472_, v_a_1473_, v_a_1474_, v_a_1475_, v_a_1476_, v_a_1477_);
lean_dec(v_a_1477_);
lean_dec_ref(v_a_1476_);
lean_dec(v_a_1475_);
lean_dec_ref(v_a_1474_);
lean_dec(v_a_1473_);
lean_dec_ref(v_a_1472_);
lean_dec_ref(v_facts_1471_);
return v_res_1479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2(lean_object* v___y_1480_, lean_object* v___x_1481_, lean_object* v_range_1482_, lean_object* v_b_1483_, lean_object* v_i_1484_, lean_object* v_hs_1485_, lean_object* v_hl_1486_, lean_object* v___y_1487_, lean_object* v___y_1488_, lean_object* v___y_1489_, lean_object* v___y_1490_, lean_object* v___y_1491_, lean_object* v___y_1492_){
_start:
{
lean_object* v___x_1494_; 
v___x_1494_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___redArg(v___y_1480_, v_range_1482_, v_b_1483_, v_i_1484_, v___y_1487_, v___y_1488_, v___y_1489_, v___y_1490_, v___y_1491_, v___y_1492_);
return v___x_1494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2___boxed(lean_object* v___y_1495_, lean_object* v___x_1496_, lean_object* v_range_1497_, lean_object* v_b_1498_, lean_object* v_i_1499_, lean_object* v_hs_1500_, lean_object* v_hl_1501_, lean_object* v___y_1502_, lean_object* v___y_1503_, lean_object* v___y_1504_, lean_object* v___y_1505_, lean_object* v___y_1506_, lean_object* v___y_1507_, lean_object* v___y_1508_){
_start:
{
lean_object* v_res_1509_; 
v_res_1509_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__2(v___y_1495_, v___x_1496_, v_range_1497_, v_b_1498_, v_i_1499_, v_hs_1500_, v_hl_1501_, v___y_1502_, v___y_1503_, v___y_1504_, v___y_1505_, v___y_1506_, v___y_1507_);
lean_dec(v___y_1507_);
lean_dec_ref(v___y_1506_);
lean_dec(v___y_1505_);
lean_dec_ref(v___y_1504_);
lean_dec(v___y_1503_);
lean_dec_ref(v___y_1502_);
lean_dec_ref(v_range_1497_);
lean_dec(v___x_1496_);
lean_dec_ref(v___y_1495_);
return v_res_1509_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__4(lean_object* v___x_1510_, lean_object* v___y_1511_, lean_object* v___y_1512_, lean_object* v___y_1513_, lean_object* v_inst_1514_, lean_object* v_a_1515_, lean_object* v___y_1516_, lean_object* v___y_1517_, lean_object* v___y_1518_, lean_object* v___y_1519_, lean_object* v___y_1520_, lean_object* v___y_1521_){
_start:
{
lean_object* v___x_1523_; 
v___x_1523_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__4___redArg(v___x_1510_, v___y_1511_, v___y_1512_, v___y_1513_, v_a_1515_, v___y_1516_, v___y_1517_, v___y_1518_, v___y_1519_, v___y_1520_, v___y_1521_);
return v___x_1523_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__4___boxed(lean_object* v___x_1524_, lean_object* v___y_1525_, lean_object* v___y_1526_, lean_object* v___y_1527_, lean_object* v_inst_1528_, lean_object* v_a_1529_, lean_object* v___y_1530_, lean_object* v___y_1531_, lean_object* v___y_1532_, lean_object* v___y_1533_, lean_object* v___y_1534_, lean_object* v___y_1535_, lean_object* v___y_1536_){
_start:
{
lean_object* v_res_1537_; 
v_res_1537_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__4(v___x_1524_, v___y_1525_, v___y_1526_, v___y_1527_, v_inst_1528_, v_a_1529_, v___y_1530_, v___y_1531_, v___y_1532_, v___y_1533_, v___y_1534_, v___y_1535_);
lean_dec(v___y_1535_);
lean_dec_ref(v___y_1534_);
lean_dec(v___y_1533_);
lean_dec_ref(v___y_1532_);
lean_dec(v___y_1531_);
lean_dec_ref(v___y_1530_);
lean_dec_ref(v___y_1527_);
lean_dec_ref(v___y_1526_);
lean_dec_ref(v___y_1525_);
return v_res_1537_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__5(lean_object* v_00_u03b2_1538_, lean_object* v_m_1539_, lean_object* v_a_1540_, lean_object* v_b_1541_){
_start:
{
lean_object* v___x_1542_; 
v___x_1542_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__5___redArg(v_m_1539_, v_a_1540_, v_b_1541_);
return v___x_1542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__5_spec__5(lean_object* v_00_u03b2_1543_, lean_object* v_data_1544_){
_start:
{
lean_object* v___x_1545_; 
v___x_1545_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__5_spec__5___redArg(v_data_1544_);
return v___x_1545_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__5_spec__5_spec__6(lean_object* v_00_u03b2_1546_, lean_object* v_i_1547_, lean_object* v_source_1548_, lean_object* v_target_1549_){
_start:
{
lean_object* v___x_1550_; 
v___x_1550_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__5_spec__5_spec__6___redArg(v_i_1547_, v_source_1548_, v_target_1549_);
return v___x_1550_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__5_spec__5_spec__6_spec__13(lean_object* v_00_u03b2_1551_, lean_object* v_x_1552_, lean_object* v_x_1553_){
_start:
{
lean_object* v___x_1554_; 
v___x_1554_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_updateGraphWithNltInfSup_spec__5_spec__5_spec__6_spec__13___redArg(v_x_1552_, v_x_1553_);
return v___x_1554_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Order_instOrdProdNatExpr__mathlib___lam__0(lean_object* v_x_1555_, lean_object* v_y_1556_){
_start:
{
lean_object* v_fst_1557_; lean_object* v_fst_1558_; uint8_t v___x_1559_; 
v_fst_1557_ = lean_ctor_get(v_x_1555_, 0);
v_fst_1558_ = lean_ctor_get(v_y_1556_, 0);
v___x_1559_ = lean_nat_dec_lt(v_fst_1557_, v_fst_1558_);
if (v___x_1559_ == 0)
{
uint8_t v___x_1560_; 
v___x_1560_ = lean_nat_dec_eq(v_fst_1557_, v_fst_1558_);
if (v___x_1560_ == 0)
{
uint8_t v___x_1561_; 
v___x_1561_ = 2;
return v___x_1561_;
}
else
{
uint8_t v___x_1562_; 
v___x_1562_ = 1;
return v___x_1562_;
}
}
else
{
uint8_t v___x_1563_; 
v___x_1563_ = 0;
return v___x_1563_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_instOrdProdNatExpr__mathlib___lam__0___boxed(lean_object* v_x_1564_, lean_object* v_y_1565_){
_start:
{
uint8_t v_res_1566_; lean_object* v_r_1567_; 
v_res_1566_ = lp_mathlib_Mathlib_Tactic_Order_instOrdProdNatExpr__mathlib___lam__0(v_x_1564_, v_y_1565_);
lean_dec_ref(v_y_1565_);
lean_dec_ref(v_x_1564_);
v_r_1567_ = lean_box(v_res_1566_);
return v_r_1567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Order_orderCoreImp_spec__8___redArg___lam__0(lean_object* v_x_1570_, lean_object* v___y_1571_, lean_object* v___y_1572_, lean_object* v___y_1573_, lean_object* v___y_1574_, lean_object* v___y_1575_, lean_object* v___y_1576_){
_start:
{
lean_object* v___x_1578_; 
lean_inc(v___y_1572_);
lean_inc_ref(v___y_1571_);
v___x_1578_ = lean_apply_7(v_x_1570_, v___y_1571_, v___y_1572_, v___y_1573_, v___y_1574_, v___y_1575_, v___y_1576_, lean_box(0));
return v___x_1578_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Order_orderCoreImp_spec__8___redArg___lam__0___boxed(lean_object* v_x_1579_, lean_object* v___y_1580_, lean_object* v___y_1581_, lean_object* v___y_1582_, lean_object* v___y_1583_, lean_object* v___y_1584_, lean_object* v___y_1585_, lean_object* v___y_1586_){
_start:
{
lean_object* v_res_1587_; 
v_res_1587_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Order_orderCoreImp_spec__8___redArg___lam__0(v_x_1579_, v___y_1580_, v___y_1581_, v___y_1582_, v___y_1583_, v___y_1584_, v___y_1585_);
lean_dec(v___y_1581_);
lean_dec_ref(v___y_1580_);
return v_res_1587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Order_orderCoreImp_spec__8___redArg(lean_object* v_mvarId_1588_, lean_object* v_x_1589_, lean_object* v___y_1590_, lean_object* v___y_1591_, lean_object* v___y_1592_, lean_object* v___y_1593_, lean_object* v___y_1594_, lean_object* v___y_1595_){
_start:
{
lean_object* v___f_1597_; lean_object* v___x_1598_; 
lean_inc(v___y_1591_);
lean_inc_ref(v___y_1590_);
v___f_1597_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Order_orderCoreImp_spec__8___redArg___lam__0___boxed), 8, 3);
lean_closure_set(v___f_1597_, 0, v_x_1589_);
lean_closure_set(v___f_1597_, 1, v___y_1590_);
lean_closure_set(v___f_1597_, 2, v___y_1591_);
v___x_1598_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_1588_, v___f_1597_, v___y_1592_, v___y_1593_, v___y_1594_, v___y_1595_);
if (lean_obj_tag(v___x_1598_) == 0)
{
return v___x_1598_;
}
else
{
lean_object* v_a_1599_; lean_object* v___x_1601_; uint8_t v_isShared_1602_; uint8_t v_isSharedCheck_1606_; 
v_a_1599_ = lean_ctor_get(v___x_1598_, 0);
v_isSharedCheck_1606_ = !lean_is_exclusive(v___x_1598_);
if (v_isSharedCheck_1606_ == 0)
{
v___x_1601_ = v___x_1598_;
v_isShared_1602_ = v_isSharedCheck_1606_;
goto v_resetjp_1600_;
}
else
{
lean_inc(v_a_1599_);
lean_dec(v___x_1598_);
v___x_1601_ = lean_box(0);
v_isShared_1602_ = v_isSharedCheck_1606_;
goto v_resetjp_1600_;
}
v_resetjp_1600_:
{
lean_object* v___x_1604_; 
if (v_isShared_1602_ == 0)
{
v___x_1604_ = v___x_1601_;
goto v_reusejp_1603_;
}
else
{
lean_object* v_reuseFailAlloc_1605_; 
v_reuseFailAlloc_1605_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1605_, 0, v_a_1599_);
v___x_1604_ = v_reuseFailAlloc_1605_;
goto v_reusejp_1603_;
}
v_reusejp_1603_:
{
return v___x_1604_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Order_orderCoreImp_spec__8___redArg___boxed(lean_object* v_mvarId_1607_, lean_object* v_x_1608_, lean_object* v___y_1609_, lean_object* v___y_1610_, lean_object* v___y_1611_, lean_object* v___y_1612_, lean_object* v___y_1613_, lean_object* v___y_1614_, lean_object* v___y_1615_){
_start:
{
lean_object* v_res_1616_; 
v_res_1616_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Order_orderCoreImp_spec__8___redArg(v_mvarId_1607_, v_x_1608_, v___y_1609_, v___y_1610_, v___y_1611_, v___y_1612_, v___y_1613_, v___y_1614_);
lean_dec(v___y_1614_);
lean_dec_ref(v___y_1613_);
lean_dec(v___y_1612_);
lean_dec_ref(v___y_1611_);
lean_dec(v___y_1610_);
lean_dec_ref(v___y_1609_);
return v_res_1616_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Order_orderCoreImp_spec__8(lean_object* v_00_u03b1_1617_, lean_object* v_mvarId_1618_, lean_object* v_x_1619_, lean_object* v___y_1620_, lean_object* v___y_1621_, lean_object* v___y_1622_, lean_object* v___y_1623_, lean_object* v___y_1624_, lean_object* v___y_1625_){
_start:
{
lean_object* v___x_1627_; 
v___x_1627_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Order_orderCoreImp_spec__8___redArg(v_mvarId_1618_, v_x_1619_, v___y_1620_, v___y_1621_, v___y_1622_, v___y_1623_, v___y_1624_, v___y_1625_);
return v___x_1627_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Order_orderCoreImp_spec__8___boxed(lean_object* v_00_u03b1_1628_, lean_object* v_mvarId_1629_, lean_object* v_x_1630_, lean_object* v___y_1631_, lean_object* v___y_1632_, lean_object* v___y_1633_, lean_object* v___y_1634_, lean_object* v___y_1635_, lean_object* v___y_1636_, lean_object* v___y_1637_){
_start:
{
lean_object* v_res_1638_; 
v_res_1638_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Order_orderCoreImp_spec__8(v_00_u03b1_1628_, v_mvarId_1629_, v_x_1630_, v___y_1631_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_, v___y_1636_);
lean_dec(v___y_1636_);
lean_dec_ref(v___y_1635_);
lean_dec(v___y_1634_);
lean_dec_ref(v___y_1633_);
lean_dec(v___y_1632_);
lean_dec_ref(v___y_1631_);
return v_res_1638_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0___redArg(size_t v_sz_1641_, size_t v_i_1642_, lean_object* v_bs_1643_, lean_object* v___y_1644_, lean_object* v___y_1645_, lean_object* v___y_1646_, lean_object* v___y_1647_){
_start:
{
uint8_t v___x_1649_; 
v___x_1649_ = lean_usize_dec_lt(v_i_1642_, v_sz_1641_);
if (v___x_1649_ == 0)
{
lean_object* v___x_1650_; 
v___x_1650_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1650_, 0, v_bs_1643_);
return v___x_1650_;
}
else
{
lean_object* v_v_1651_; lean_object* v___x_1652_; 
v_v_1651_ = lean_array_uget_borrowed(v_bs_1643_, v_i_1642_);
lean_inc(v_v_1651_);
v___x_1652_ = l_Lean_Meta_ppExpr(v_v_1651_, v___y_1644_, v___y_1645_, v___y_1646_, v___y_1647_);
if (lean_obj_tag(v___x_1652_) == 0)
{
lean_object* v_a_1653_; lean_object* v___x_1654_; lean_object* v_bs_x27_1655_; lean_object* v___x_1656_; lean_object* v___x_1657_; lean_object* v___x_1658_; lean_object* v___x_1659_; lean_object* v___x_1660_; lean_object* v___x_1661_; lean_object* v___x_1662_; lean_object* v___x_1663_; lean_object* v___x_1664_; size_t v___x_1665_; size_t v___x_1666_; lean_object* v___x_1667_; 
v_a_1653_ = lean_ctor_get(v___x_1652_, 0);
lean_inc(v_a_1653_);
lean_dec_ref_known(v___x_1652_, 1);
v___x_1654_ = lean_unsigned_to_nat(0u);
v_bs_x27_1655_ = lean_array_uset(v_bs_1643_, v_i_1642_, v___x_1654_);
v___x_1656_ = lean_usize_to_nat(v_i_1642_);
v___x_1657_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0___redArg___closed__0));
v___x_1658_ = l_Nat_reprFast(v___x_1656_);
v___x_1659_ = lean_string_append(v___x_1657_, v___x_1658_);
lean_dec_ref(v___x_1658_);
v___x_1660_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0___redArg___closed__1));
v___x_1661_ = lean_string_append(v___x_1659_, v___x_1660_);
v___x_1662_ = l_Std_Format_defWidth;
v___x_1663_ = l_Std_Format_pretty(v_a_1653_, v___x_1662_, v___x_1654_, v___x_1654_);
v___x_1664_ = lean_string_append(v___x_1661_, v___x_1663_);
lean_dec_ref(v___x_1663_);
v___x_1665_ = ((size_t)1ULL);
v___x_1666_ = lean_usize_add(v_i_1642_, v___x_1665_);
v___x_1667_ = lean_array_uset(v_bs_x27_1655_, v_i_1642_, v___x_1664_);
v_i_1642_ = v___x_1666_;
v_bs_1643_ = v___x_1667_;
goto _start;
}
else
{
lean_object* v_a_1669_; lean_object* v___x_1671_; uint8_t v_isShared_1672_; uint8_t v_isSharedCheck_1676_; 
lean_dec_ref(v_bs_1643_);
v_a_1669_ = lean_ctor_get(v___x_1652_, 0);
v_isSharedCheck_1676_ = !lean_is_exclusive(v___x_1652_);
if (v_isSharedCheck_1676_ == 0)
{
v___x_1671_ = v___x_1652_;
v_isShared_1672_ = v_isSharedCheck_1676_;
goto v_resetjp_1670_;
}
else
{
lean_inc(v_a_1669_);
lean_dec(v___x_1652_);
v___x_1671_ = lean_box(0);
v_isShared_1672_ = v_isSharedCheck_1676_;
goto v_resetjp_1670_;
}
v_resetjp_1670_:
{
lean_object* v___x_1674_; 
if (v_isShared_1672_ == 0)
{
v___x_1674_ = v___x_1671_;
goto v_reusejp_1673_;
}
else
{
lean_object* v_reuseFailAlloc_1675_; 
v_reuseFailAlloc_1675_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1675_, 0, v_a_1669_);
v___x_1674_ = v_reuseFailAlloc_1675_;
goto v_reusejp_1673_;
}
v_reusejp_1673_:
{
return v___x_1674_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0___redArg___boxed(lean_object* v_sz_1677_, lean_object* v_i_1678_, lean_object* v_bs_1679_, lean_object* v___y_1680_, lean_object* v___y_1681_, lean_object* v___y_1682_, lean_object* v___y_1683_, lean_object* v___y_1684_){
_start:
{
size_t v_sz_boxed_1685_; size_t v_i_boxed_1686_; lean_object* v_res_1687_; 
v_sz_boxed_1685_ = lean_unbox_usize(v_sz_1677_);
lean_dec(v_sz_1677_);
v_i_boxed_1686_ = lean_unbox_usize(v_i_1678_);
lean_dec(v_i_1678_);
v_res_1687_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0___redArg(v_sz_boxed_1685_, v_i_boxed_1686_, v_bs_1679_, v___y_1680_, v___y_1681_, v___y_1682_, v___y_1683_);
lean_dec(v___y_1683_);
lean_dec_ref(v___y_1682_);
lean_dec(v___y_1681_);
lean_dec_ref(v___y_1680_);
return v_res_1687_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4_spec__6(lean_object* v_msgData_1688_, lean_object* v___y_1689_, lean_object* v___y_1690_, lean_object* v___y_1691_, lean_object* v___y_1692_){
_start:
{
lean_object* v___x_1694_; lean_object* v_env_1695_; lean_object* v___x_1696_; lean_object* v_mctx_1697_; lean_object* v_lctx_1698_; lean_object* v_options_1699_; lean_object* v___x_1700_; lean_object* v___x_1701_; lean_object* v___x_1702_; 
v___x_1694_ = lean_st_ref_get(v___y_1692_);
v_env_1695_ = lean_ctor_get(v___x_1694_, 0);
lean_inc_ref(v_env_1695_);
lean_dec(v___x_1694_);
v___x_1696_ = lean_st_ref_get(v___y_1690_);
v_mctx_1697_ = lean_ctor_get(v___x_1696_, 0);
lean_inc_ref(v_mctx_1697_);
lean_dec(v___x_1696_);
v_lctx_1698_ = lean_ctor_get(v___y_1689_, 2);
v_options_1699_ = lean_ctor_get(v___y_1691_, 2);
lean_inc_ref(v_options_1699_);
lean_inc_ref(v_lctx_1698_);
v___x_1700_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1700_, 0, v_env_1695_);
lean_ctor_set(v___x_1700_, 1, v_mctx_1697_);
lean_ctor_set(v___x_1700_, 2, v_lctx_1698_);
lean_ctor_set(v___x_1700_, 3, v_options_1699_);
v___x_1701_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1701_, 0, v___x_1700_);
lean_ctor_set(v___x_1701_, 1, v_msgData_1688_);
v___x_1702_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1702_, 0, v___x_1701_);
return v___x_1702_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4_spec__6___boxed(lean_object* v_msgData_1703_, lean_object* v___y_1704_, lean_object* v___y_1705_, lean_object* v___y_1706_, lean_object* v___y_1707_, lean_object* v___y_1708_){
_start:
{
lean_object* v_res_1709_; 
v_res_1709_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4_spec__6(v_msgData_1703_, v___y_1704_, v___y_1705_, v___y_1706_, v___y_1707_);
lean_dec(v___y_1707_);
lean_dec_ref(v___y_1706_);
lean_dec(v___y_1705_);
lean_dec_ref(v___y_1704_);
return v_res_1709_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Order_orderCoreImp_spec__7___redArg(lean_object* v_msg_1710_, lean_object* v___y_1711_, lean_object* v___y_1712_, lean_object* v___y_1713_, lean_object* v___y_1714_){
_start:
{
lean_object* v_ref_1716_; lean_object* v___x_1717_; lean_object* v_a_1718_; lean_object* v___x_1720_; uint8_t v_isShared_1721_; uint8_t v_isSharedCheck_1726_; 
v_ref_1716_ = lean_ctor_get(v___y_1713_, 5);
v___x_1717_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4_spec__6(v_msg_1710_, v___y_1711_, v___y_1712_, v___y_1713_, v___y_1714_);
v_a_1718_ = lean_ctor_get(v___x_1717_, 0);
v_isSharedCheck_1726_ = !lean_is_exclusive(v___x_1717_);
if (v_isSharedCheck_1726_ == 0)
{
v___x_1720_ = v___x_1717_;
v_isShared_1721_ = v_isSharedCheck_1726_;
goto v_resetjp_1719_;
}
else
{
lean_inc(v_a_1718_);
lean_dec(v___x_1717_);
v___x_1720_ = lean_box(0);
v_isShared_1721_ = v_isSharedCheck_1726_;
goto v_resetjp_1719_;
}
v_resetjp_1719_:
{
lean_object* v___x_1722_; lean_object* v___x_1724_; 
lean_inc(v_ref_1716_);
v___x_1722_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1722_, 0, v_ref_1716_);
lean_ctor_set(v___x_1722_, 1, v_a_1718_);
if (v_isShared_1721_ == 0)
{
lean_ctor_set_tag(v___x_1720_, 1);
lean_ctor_set(v___x_1720_, 0, v___x_1722_);
v___x_1724_ = v___x_1720_;
goto v_reusejp_1723_;
}
else
{
lean_object* v_reuseFailAlloc_1725_; 
v_reuseFailAlloc_1725_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1725_, 0, v___x_1722_);
v___x_1724_ = v_reuseFailAlloc_1725_;
goto v_reusejp_1723_;
}
v_reusejp_1723_:
{
return v___x_1724_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Order_orderCoreImp_spec__7___redArg___boxed(lean_object* v_msg_1727_, lean_object* v___y_1728_, lean_object* v___y_1729_, lean_object* v___y_1730_, lean_object* v___y_1731_, lean_object* v___y_1732_){
_start:
{
lean_object* v_res_1733_; 
v_res_1733_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Order_orderCoreImp_spec__7___redArg(v_msg_1727_, v___y_1728_, v___y_1729_, v___y_1730_, v___y_1731_);
lean_dec(v___y_1731_);
lean_dec_ref(v___y_1730_);
lean_dec(v___y_1729_);
lean_dec_ref(v___y_1728_);
return v_res_1733_;
}
}
static double _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___redArg___closed__0(void){
_start:
{
lean_object* v___x_1734_; double v___x_1735_; 
v___x_1734_ = lean_unsigned_to_nat(0u);
v___x_1735_ = lean_float_of_nat(v___x_1734_);
return v___x_1735_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___redArg(lean_object* v_cls_1739_, lean_object* v_msg_1740_, lean_object* v___y_1741_, lean_object* v___y_1742_, lean_object* v___y_1743_, lean_object* v___y_1744_){
_start:
{
lean_object* v_ref_1746_; lean_object* v___x_1747_; lean_object* v_a_1748_; lean_object* v___x_1750_; uint8_t v_isShared_1751_; uint8_t v_isSharedCheck_1792_; 
v_ref_1746_ = lean_ctor_get(v___y_1743_, 5);
v___x_1747_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4_spec__6(v_msg_1740_, v___y_1741_, v___y_1742_, v___y_1743_, v___y_1744_);
v_a_1748_ = lean_ctor_get(v___x_1747_, 0);
v_isSharedCheck_1792_ = !lean_is_exclusive(v___x_1747_);
if (v_isSharedCheck_1792_ == 0)
{
v___x_1750_ = v___x_1747_;
v_isShared_1751_ = v_isSharedCheck_1792_;
goto v_resetjp_1749_;
}
else
{
lean_inc(v_a_1748_);
lean_dec(v___x_1747_);
v___x_1750_ = lean_box(0);
v_isShared_1751_ = v_isSharedCheck_1792_;
goto v_resetjp_1749_;
}
v_resetjp_1749_:
{
lean_object* v___x_1752_; lean_object* v_traceState_1753_; lean_object* v_env_1754_; lean_object* v_nextMacroScope_1755_; lean_object* v_ngen_1756_; lean_object* v_auxDeclNGen_1757_; lean_object* v_cache_1758_; lean_object* v_messages_1759_; lean_object* v_infoState_1760_; lean_object* v_snapshotTasks_1761_; lean_object* v___x_1763_; uint8_t v_isShared_1764_; uint8_t v_isSharedCheck_1791_; 
v___x_1752_ = lean_st_ref_take(v___y_1744_);
v_traceState_1753_ = lean_ctor_get(v___x_1752_, 4);
v_env_1754_ = lean_ctor_get(v___x_1752_, 0);
v_nextMacroScope_1755_ = lean_ctor_get(v___x_1752_, 1);
v_ngen_1756_ = lean_ctor_get(v___x_1752_, 2);
v_auxDeclNGen_1757_ = lean_ctor_get(v___x_1752_, 3);
v_cache_1758_ = lean_ctor_get(v___x_1752_, 5);
v_messages_1759_ = lean_ctor_get(v___x_1752_, 6);
v_infoState_1760_ = lean_ctor_get(v___x_1752_, 7);
v_snapshotTasks_1761_ = lean_ctor_get(v___x_1752_, 8);
v_isSharedCheck_1791_ = !lean_is_exclusive(v___x_1752_);
if (v_isSharedCheck_1791_ == 0)
{
v___x_1763_ = v___x_1752_;
v_isShared_1764_ = v_isSharedCheck_1791_;
goto v_resetjp_1762_;
}
else
{
lean_inc(v_snapshotTasks_1761_);
lean_inc(v_infoState_1760_);
lean_inc(v_messages_1759_);
lean_inc(v_cache_1758_);
lean_inc(v_traceState_1753_);
lean_inc(v_auxDeclNGen_1757_);
lean_inc(v_ngen_1756_);
lean_inc(v_nextMacroScope_1755_);
lean_inc(v_env_1754_);
lean_dec(v___x_1752_);
v___x_1763_ = lean_box(0);
v_isShared_1764_ = v_isSharedCheck_1791_;
goto v_resetjp_1762_;
}
v_resetjp_1762_:
{
uint64_t v_tid_1765_; lean_object* v_traces_1766_; lean_object* v___x_1768_; uint8_t v_isShared_1769_; uint8_t v_isSharedCheck_1790_; 
v_tid_1765_ = lean_ctor_get_uint64(v_traceState_1753_, sizeof(void*)*1);
v_traces_1766_ = lean_ctor_get(v_traceState_1753_, 0);
v_isSharedCheck_1790_ = !lean_is_exclusive(v_traceState_1753_);
if (v_isSharedCheck_1790_ == 0)
{
v___x_1768_ = v_traceState_1753_;
v_isShared_1769_ = v_isSharedCheck_1790_;
goto v_resetjp_1767_;
}
else
{
lean_inc(v_traces_1766_);
lean_dec(v_traceState_1753_);
v___x_1768_ = lean_box(0);
v_isShared_1769_ = v_isSharedCheck_1790_;
goto v_resetjp_1767_;
}
v_resetjp_1767_:
{
lean_object* v___x_1770_; double v___x_1771_; uint8_t v___x_1772_; lean_object* v___x_1773_; lean_object* v___x_1774_; lean_object* v___x_1775_; lean_object* v___x_1776_; lean_object* v___x_1777_; lean_object* v___x_1778_; lean_object* v___x_1780_; 
v___x_1770_ = lean_box(0);
v___x_1771_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___redArg___closed__0, &lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___redArg___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___redArg___closed__0);
v___x_1772_ = 0;
v___x_1773_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___redArg___closed__1));
v___x_1774_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_1774_, 0, v_cls_1739_);
lean_ctor_set(v___x_1774_, 1, v___x_1770_);
lean_ctor_set(v___x_1774_, 2, v___x_1773_);
lean_ctor_set_float(v___x_1774_, sizeof(void*)*3, v___x_1771_);
lean_ctor_set_float(v___x_1774_, sizeof(void*)*3 + 8, v___x_1771_);
lean_ctor_set_uint8(v___x_1774_, sizeof(void*)*3 + 16, v___x_1772_);
v___x_1775_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___redArg___closed__2));
v___x_1776_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_1776_, 0, v___x_1774_);
lean_ctor_set(v___x_1776_, 1, v_a_1748_);
lean_ctor_set(v___x_1776_, 2, v___x_1775_);
lean_inc(v_ref_1746_);
v___x_1777_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1777_, 0, v_ref_1746_);
lean_ctor_set(v___x_1777_, 1, v___x_1776_);
v___x_1778_ = l_Lean_PersistentArray_push___redArg(v_traces_1766_, v___x_1777_);
if (v_isShared_1769_ == 0)
{
lean_ctor_set(v___x_1768_, 0, v___x_1778_);
v___x_1780_ = v___x_1768_;
goto v_reusejp_1779_;
}
else
{
lean_object* v_reuseFailAlloc_1789_; 
v_reuseFailAlloc_1789_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1789_, 0, v___x_1778_);
lean_ctor_set_uint64(v_reuseFailAlloc_1789_, sizeof(void*)*1, v_tid_1765_);
v___x_1780_ = v_reuseFailAlloc_1789_;
goto v_reusejp_1779_;
}
v_reusejp_1779_:
{
lean_object* v___x_1782_; 
if (v_isShared_1764_ == 0)
{
lean_ctor_set(v___x_1763_, 4, v___x_1780_);
v___x_1782_ = v___x_1763_;
goto v_reusejp_1781_;
}
else
{
lean_object* v_reuseFailAlloc_1788_; 
v_reuseFailAlloc_1788_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1788_, 0, v_env_1754_);
lean_ctor_set(v_reuseFailAlloc_1788_, 1, v_nextMacroScope_1755_);
lean_ctor_set(v_reuseFailAlloc_1788_, 2, v_ngen_1756_);
lean_ctor_set(v_reuseFailAlloc_1788_, 3, v_auxDeclNGen_1757_);
lean_ctor_set(v_reuseFailAlloc_1788_, 4, v___x_1780_);
lean_ctor_set(v_reuseFailAlloc_1788_, 5, v_cache_1758_);
lean_ctor_set(v_reuseFailAlloc_1788_, 6, v_messages_1759_);
lean_ctor_set(v_reuseFailAlloc_1788_, 7, v_infoState_1760_);
lean_ctor_set(v_reuseFailAlloc_1788_, 8, v_snapshotTasks_1761_);
v___x_1782_ = v_reuseFailAlloc_1788_;
goto v_reusejp_1781_;
}
v_reusejp_1781_:
{
lean_object* v___x_1783_; lean_object* v___x_1784_; lean_object* v___x_1786_; 
v___x_1783_ = lean_st_ref_set(v___y_1744_, v___x_1782_);
v___x_1784_ = lean_box(0);
if (v_isShared_1751_ == 0)
{
lean_ctor_set(v___x_1750_, 0, v___x_1784_);
v___x_1786_ = v___x_1750_;
goto v_reusejp_1785_;
}
else
{
lean_object* v_reuseFailAlloc_1787_; 
v_reuseFailAlloc_1787_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1787_, 0, v___x_1784_);
v___x_1786_ = v_reuseFailAlloc_1787_;
goto v_reusejp_1785_;
}
v_reusejp_1785_:
{
return v___x_1786_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___redArg___boxed(lean_object* v_cls_1793_, lean_object* v_msg_1794_, lean_object* v___y_1795_, lean_object* v___y_1796_, lean_object* v___y_1797_, lean_object* v___y_1798_, lean_object* v___y_1799_){
_start:
{
lean_object* v_res_1800_; 
v_res_1800_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___redArg(v_cls_1793_, v_msg_1794_, v___y_1795_, v___y_1796_, v___y_1797_, v___y_1798_);
lean_dec(v___y_1798_);
lean_dec_ref(v___y_1797_);
lean_dec(v___y_1796_);
lean_dec_ref(v___y_1795_);
return v_res_1800_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3(size_t v_sz_1811_, size_t v_i_1812_, lean_object* v_bs_1813_){
_start:
{
uint8_t v___x_1814_; 
v___x_1814_ = lean_usize_dec_lt(v_i_1812_, v_sz_1811_);
if (v___x_1814_ == 0)
{
return v_bs_1813_;
}
else
{
lean_object* v_v_1815_; lean_object* v___x_1816_; lean_object* v_bs_x27_1817_; lean_object* v___y_1819_; 
v_v_1815_ = lean_array_uget(v_bs_1813_, v_i_1812_);
v___x_1816_ = lean_unsigned_to_nat(0u);
v_bs_x27_1817_ = lean_array_uset(v_bs_1813_, v_i_1812_, v___x_1816_);
switch(lean_obj_tag(v_v_1815_))
{
case 0:
{
lean_object* v_lhs_1824_; lean_object* v_rhs_1825_; lean_object* v___x_1826_; lean_object* v___x_1827_; lean_object* v___x_1828_; lean_object* v___x_1829_; lean_object* v___x_1830_; lean_object* v___x_1831_; lean_object* v___x_1832_; 
v_lhs_1824_ = lean_ctor_get(v_v_1815_, 0);
lean_inc(v_lhs_1824_);
v_rhs_1825_ = lean_ctor_get(v_v_1815_, 1);
lean_inc(v_rhs_1825_);
lean_dec_ref_known(v_v_1815_, 3);
v___x_1826_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0___redArg___closed__0));
v___x_1827_ = l_Nat_reprFast(v_lhs_1824_);
v___x_1828_ = lean_string_append(v___x_1826_, v___x_1827_);
lean_dec_ref(v___x_1827_);
v___x_1829_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__0));
v___x_1830_ = lean_string_append(v___x_1828_, v___x_1829_);
v___x_1831_ = l_Nat_reprFast(v_rhs_1825_);
v___x_1832_ = lean_string_append(v___x_1830_, v___x_1831_);
lean_dec_ref(v___x_1831_);
v___y_1819_ = v___x_1832_;
goto v___jp_1818_;
}
case 1:
{
lean_object* v_lhs_1833_; lean_object* v_rhs_1834_; lean_object* v___x_1835_; lean_object* v___x_1836_; lean_object* v___x_1837_; lean_object* v___x_1838_; lean_object* v___x_1839_; lean_object* v___x_1840_; lean_object* v___x_1841_; 
v_lhs_1833_ = lean_ctor_get(v_v_1815_, 0);
lean_inc(v_lhs_1833_);
v_rhs_1834_ = lean_ctor_get(v_v_1815_, 1);
lean_inc(v_rhs_1834_);
lean_dec_ref_known(v_v_1815_, 3);
v___x_1835_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0___redArg___closed__0));
v___x_1836_ = l_Nat_reprFast(v_lhs_1833_);
v___x_1837_ = lean_string_append(v___x_1835_, v___x_1836_);
lean_dec_ref(v___x_1836_);
v___x_1838_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__1));
v___x_1839_ = lean_string_append(v___x_1837_, v___x_1838_);
v___x_1840_ = l_Nat_reprFast(v_rhs_1834_);
v___x_1841_ = lean_string_append(v___x_1839_, v___x_1840_);
lean_dec_ref(v___x_1840_);
v___y_1819_ = v___x_1841_;
goto v___jp_1818_;
}
case 2:
{
lean_object* v_lhs_1842_; lean_object* v_rhs_1843_; lean_object* v___x_1844_; lean_object* v___x_1845_; lean_object* v___x_1846_; lean_object* v___x_1847_; lean_object* v___x_1848_; lean_object* v___x_1849_; lean_object* v___x_1850_; 
v_lhs_1842_ = lean_ctor_get(v_v_1815_, 0);
lean_inc(v_lhs_1842_);
v_rhs_1843_ = lean_ctor_get(v_v_1815_, 1);
lean_inc(v_rhs_1843_);
lean_dec_ref_known(v_v_1815_, 3);
v___x_1844_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0___redArg___closed__0));
v___x_1845_ = l_Nat_reprFast(v_lhs_1842_);
v___x_1846_ = lean_string_append(v___x_1844_, v___x_1845_);
lean_dec_ref(v___x_1845_);
v___x_1847_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__2));
v___x_1848_ = lean_string_append(v___x_1846_, v___x_1847_);
v___x_1849_ = l_Nat_reprFast(v_rhs_1843_);
v___x_1850_ = lean_string_append(v___x_1848_, v___x_1849_);
lean_dec_ref(v___x_1849_);
v___y_1819_ = v___x_1850_;
goto v___jp_1818_;
}
case 3:
{
lean_object* v_lhs_1851_; lean_object* v_rhs_1852_; lean_object* v___x_1853_; lean_object* v___x_1854_; lean_object* v___x_1855_; lean_object* v___x_1856_; lean_object* v___x_1857_; lean_object* v___x_1858_; lean_object* v___x_1859_; 
v_lhs_1851_ = lean_ctor_get(v_v_1815_, 0);
lean_inc(v_lhs_1851_);
v_rhs_1852_ = lean_ctor_get(v_v_1815_, 1);
lean_inc(v_rhs_1852_);
lean_dec_ref_known(v_v_1815_, 3);
v___x_1853_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__3));
v___x_1854_ = l_Nat_reprFast(v_lhs_1851_);
v___x_1855_ = lean_string_append(v___x_1853_, v___x_1854_);
lean_dec_ref(v___x_1854_);
v___x_1856_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__2));
v___x_1857_ = lean_string_append(v___x_1855_, v___x_1856_);
v___x_1858_ = l_Nat_reprFast(v_rhs_1852_);
v___x_1859_ = lean_string_append(v___x_1857_, v___x_1858_);
lean_dec_ref(v___x_1858_);
v___y_1819_ = v___x_1859_;
goto v___jp_1818_;
}
case 4:
{
lean_object* v_lhs_1860_; lean_object* v_rhs_1861_; lean_object* v___x_1862_; lean_object* v___x_1863_; lean_object* v___x_1864_; lean_object* v___x_1865_; lean_object* v___x_1866_; lean_object* v___x_1867_; lean_object* v___x_1868_; 
v_lhs_1860_ = lean_ctor_get(v_v_1815_, 0);
lean_inc(v_lhs_1860_);
v_rhs_1861_ = lean_ctor_get(v_v_1815_, 1);
lean_inc(v_rhs_1861_);
lean_dec_ref_known(v_v_1815_, 3);
v___x_1862_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0___redArg___closed__0));
v___x_1863_ = l_Nat_reprFast(v_lhs_1860_);
v___x_1864_ = lean_string_append(v___x_1862_, v___x_1863_);
lean_dec_ref(v___x_1863_);
v___x_1865_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__4));
v___x_1866_ = lean_string_append(v___x_1864_, v___x_1865_);
v___x_1867_ = l_Nat_reprFast(v_rhs_1861_);
v___x_1868_ = lean_string_append(v___x_1866_, v___x_1867_);
lean_dec_ref(v___x_1867_);
v___y_1819_ = v___x_1868_;
goto v___jp_1818_;
}
case 5:
{
lean_object* v_lhs_1869_; lean_object* v_rhs_1870_; lean_object* v___x_1871_; lean_object* v___x_1872_; lean_object* v___x_1873_; lean_object* v___x_1874_; lean_object* v___x_1875_; lean_object* v___x_1876_; lean_object* v___x_1877_; 
v_lhs_1869_ = lean_ctor_get(v_v_1815_, 0);
lean_inc(v_lhs_1869_);
v_rhs_1870_ = lean_ctor_get(v_v_1815_, 1);
lean_inc(v_rhs_1870_);
lean_dec_ref_known(v_v_1815_, 3);
v___x_1871_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__3));
v___x_1872_ = l_Nat_reprFast(v_lhs_1869_);
v___x_1873_ = lean_string_append(v___x_1871_, v___x_1872_);
lean_dec_ref(v___x_1872_);
v___x_1874_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__4));
v___x_1875_ = lean_string_append(v___x_1873_, v___x_1874_);
v___x_1876_ = l_Nat_reprFast(v_rhs_1870_);
v___x_1877_ = lean_string_append(v___x_1875_, v___x_1876_);
lean_dec_ref(v___x_1876_);
v___y_1819_ = v___x_1877_;
goto v___jp_1818_;
}
case 6:
{
lean_object* v_idx_1878_; lean_object* v___x_1879_; lean_object* v___x_1880_; lean_object* v___x_1881_; lean_object* v___x_1882_; lean_object* v___x_1883_; 
v_idx_1878_ = lean_ctor_get(v_v_1815_, 0);
lean_inc(v_idx_1878_);
lean_dec_ref_known(v_v_1815_, 1);
v___x_1879_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0___redArg___closed__0));
v___x_1880_ = l_Nat_reprFast(v_idx_1878_);
v___x_1881_ = lean_string_append(v___x_1879_, v___x_1880_);
lean_dec_ref(v___x_1880_);
v___x_1882_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__5));
v___x_1883_ = lean_string_append(v___x_1881_, v___x_1882_);
v___y_1819_ = v___x_1883_;
goto v___jp_1818_;
}
case 7:
{
lean_object* v_idx_1884_; lean_object* v___x_1885_; lean_object* v___x_1886_; lean_object* v___x_1887_; lean_object* v___x_1888_; lean_object* v___x_1889_; 
v_idx_1884_ = lean_ctor_get(v_v_1815_, 0);
lean_inc(v_idx_1884_);
lean_dec_ref_known(v_v_1815_, 1);
v___x_1885_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0___redArg___closed__0));
v___x_1886_ = l_Nat_reprFast(v_idx_1884_);
v___x_1887_ = lean_string_append(v___x_1885_, v___x_1886_);
lean_dec_ref(v___x_1886_);
v___x_1888_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__6));
v___x_1889_ = lean_string_append(v___x_1887_, v___x_1888_);
v___y_1819_ = v___x_1889_;
goto v___jp_1818_;
}
case 8:
{
lean_object* v_lhs_1890_; lean_object* v_rhs_1891_; lean_object* v_res_1892_; lean_object* v___x_1893_; lean_object* v___x_1894_; lean_object* v___x_1895_; lean_object* v___x_1896_; lean_object* v___x_1897_; lean_object* v___x_1898_; lean_object* v___x_1899_; lean_object* v___x_1900_; lean_object* v___x_1901_; lean_object* v___x_1902_; lean_object* v___x_1903_; 
v_lhs_1890_ = lean_ctor_get(v_v_1815_, 0);
lean_inc(v_lhs_1890_);
v_rhs_1891_ = lean_ctor_get(v_v_1815_, 1);
lean_inc(v_rhs_1891_);
v_res_1892_ = lean_ctor_get(v_v_1815_, 2);
lean_inc(v_res_1892_);
lean_dec_ref_known(v_v_1815_, 3);
v___x_1893_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0___redArg___closed__0));
v___x_1894_ = l_Nat_reprFast(v_res_1892_);
v___x_1895_ = lean_string_append(v___x_1893_, v___x_1894_);
lean_dec_ref(v___x_1894_);
v___x_1896_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__7));
v___x_1897_ = lean_string_append(v___x_1895_, v___x_1896_);
v___x_1898_ = l_Nat_reprFast(v_lhs_1890_);
v___x_1899_ = lean_string_append(v___x_1897_, v___x_1898_);
lean_dec_ref(v___x_1898_);
v___x_1900_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__8));
v___x_1901_ = lean_string_append(v___x_1899_, v___x_1900_);
v___x_1902_ = l_Nat_reprFast(v_rhs_1891_);
v___x_1903_ = lean_string_append(v___x_1901_, v___x_1902_);
lean_dec_ref(v___x_1902_);
v___y_1819_ = v___x_1903_;
goto v___jp_1818_;
}
default: 
{
lean_object* v_lhs_1904_; lean_object* v_rhs_1905_; lean_object* v_res_1906_; lean_object* v___x_1907_; lean_object* v___x_1908_; lean_object* v___x_1909_; lean_object* v___x_1910_; lean_object* v___x_1911_; lean_object* v___x_1912_; lean_object* v___x_1913_; lean_object* v___x_1914_; lean_object* v___x_1915_; lean_object* v___x_1916_; lean_object* v___x_1917_; 
v_lhs_1904_ = lean_ctor_get(v_v_1815_, 0);
lean_inc(v_lhs_1904_);
v_rhs_1905_ = lean_ctor_get(v_v_1815_, 1);
lean_inc(v_rhs_1905_);
v_res_1906_ = lean_ctor_get(v_v_1815_, 2);
lean_inc(v_res_1906_);
lean_dec_ref_known(v_v_1815_, 3);
v___x_1907_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0___redArg___closed__0));
v___x_1908_ = l_Nat_reprFast(v_res_1906_);
v___x_1909_ = lean_string_append(v___x_1907_, v___x_1908_);
lean_dec_ref(v___x_1908_);
v___x_1910_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__7));
v___x_1911_ = lean_string_append(v___x_1909_, v___x_1910_);
v___x_1912_ = l_Nat_reprFast(v_lhs_1904_);
v___x_1913_ = lean_string_append(v___x_1911_, v___x_1912_);
lean_dec_ref(v___x_1912_);
v___x_1914_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___closed__9));
v___x_1915_ = lean_string_append(v___x_1913_, v___x_1914_);
v___x_1916_ = l_Nat_reprFast(v_rhs_1905_);
v___x_1917_ = lean_string_append(v___x_1915_, v___x_1916_);
lean_dec_ref(v___x_1916_);
v___y_1819_ = v___x_1917_;
goto v___jp_1818_;
}
}
v___jp_1818_:
{
size_t v___x_1820_; size_t v___x_1821_; lean_object* v___x_1822_; 
v___x_1820_ = ((size_t)1ULL);
v___x_1821_ = lean_usize_add(v_i_1812_, v___x_1820_);
v___x_1822_ = lean_array_uset(v_bs_x27_1817_, v_i_1812_, v___y_1819_);
v_i_1812_ = v___x_1821_;
v_bs_1813_ = v___x_1822_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3___boxed(lean_object* v_sz_1918_, lean_object* v_i_1919_, lean_object* v_bs_1920_){
_start:
{
size_t v_sz_boxed_1921_; size_t v_i_boxed_1922_; lean_object* v_res_1923_; 
v_sz_boxed_1921_ = lean_unbox_usize(v_sz_1918_);
lean_dec(v_sz_1918_);
v_i_boxed_1922_ = lean_unbox_usize(v_i_1919_);
lean_dec(v_i_1919_);
v_res_1923_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3(v_sz_boxed_1921_, v_i_boxed_1922_, v_bs_1920_);
return v_res_1923_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___lam__0(lean_object* v___x_1927_, lean_object* v_____do__lift_1928_, lean_object* v___y_1929_, lean_object* v___y_1930_, lean_object* v___y_1931_, lean_object* v___y_1932_, lean_object* v___y_1933_, lean_object* v___y_1934_){
_start:
{
lean_object* v_options_1936_; uint8_t v_hasTrace_1937_; 
v_options_1936_ = lean_ctor_get(v___y_1933_, 2);
v_hasTrace_1937_ = lean_ctor_get_uint8(v_options_1936_, sizeof(void*)*1);
if (v_hasTrace_1937_ == 0)
{
lean_object* v___x_1938_; lean_object* v___x_1939_; 
lean_dec(v___x_1927_);
v___x_1938_ = lean_box(v_hasTrace_1937_);
v___x_1939_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1939_, 0, v___x_1938_);
return v___x_1939_;
}
else
{
lean_object* v___x_1940_; lean_object* v___x_1941_; uint8_t v___x_1942_; lean_object* v___x_1943_; lean_object* v___x_1944_; 
v___x_1940_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___lam__0___closed__1));
v___x_1941_ = l_Lean_Name_append(v___x_1940_, v___x_1927_);
v___x_1942_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_____do__lift_1928_, v_options_1936_, v___x_1941_);
lean_dec(v___x_1941_);
v___x_1943_ = lean_box(v___x_1942_);
v___x_1944_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1944_, 0, v___x_1943_);
return v___x_1944_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___lam__0___boxed(lean_object* v___x_1945_, lean_object* v_____do__lift_1946_, lean_object* v___y_1947_, lean_object* v___y_1948_, lean_object* v___y_1949_, lean_object* v___y_1950_, lean_object* v___y_1951_, lean_object* v___y_1952_, lean_object* v___y_1953_){
_start:
{
lean_object* v_res_1954_; 
v_res_1954_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___lam__0(v___x_1945_, v_____do__lift_1946_, v___y_1947_, v___y_1948_, v___y_1949_, v___y_1950_, v___y_1951_, v___y_1952_);
lean_dec(v___y_1952_);
lean_dec_ref(v___y_1951_);
lean_dec(v___y_1950_);
lean_dec_ref(v___y_1949_);
lean_dec(v___y_1948_);
lean_dec_ref(v___y_1947_);
lean_dec_ref(v_____do__lift_1946_);
return v_res_1954_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Order_orderCoreImp_spec__2_spec__3(lean_object* v_as_1955_, size_t v_i_1956_, size_t v_stop_1957_, lean_object* v_b_1958_){
_start:
{
lean_object* v___y_1960_; uint8_t v___x_1964_; 
v___x_1964_ = lean_usize_dec_eq(v_i_1956_, v_stop_1957_);
if (v___x_1964_ == 0)
{
lean_object* v___x_1965_; 
v___x_1965_ = lean_array_uget_borrowed(v_as_1955_, v_i_1956_);
switch(lean_obj_tag(v___x_1965_))
{
case 0:
{
lean_object* v_proof_1966_; lean_object* v___x_1967_; 
v_proof_1966_ = lean_ctor_get(v___x_1965_, 2);
lean_inc_ref(v_proof_1966_);
v___x_1967_ = lean_array_push(v_b_1958_, v_proof_1966_);
v___y_1960_ = v___x_1967_;
goto v___jp_1959_;
}
case 1:
{
lean_object* v_proof_1968_; lean_object* v___x_1969_; 
v_proof_1968_ = lean_ctor_get(v___x_1965_, 2);
lean_inc_ref(v_proof_1968_);
v___x_1969_ = lean_array_push(v_b_1958_, v_proof_1968_);
v___y_1960_ = v___x_1969_;
goto v___jp_1959_;
}
case 2:
{
lean_object* v_proof_1970_; lean_object* v___x_1971_; 
v_proof_1970_ = lean_ctor_get(v___x_1965_, 2);
lean_inc_ref(v_proof_1970_);
v___x_1971_ = lean_array_push(v_b_1958_, v_proof_1970_);
v___y_1960_ = v___x_1971_;
goto v___jp_1959_;
}
case 3:
{
lean_object* v_proof_1972_; lean_object* v___x_1973_; 
v_proof_1972_ = lean_ctor_get(v___x_1965_, 2);
lean_inc_ref(v_proof_1972_);
v___x_1973_ = lean_array_push(v_b_1958_, v_proof_1972_);
v___y_1960_ = v___x_1973_;
goto v___jp_1959_;
}
case 4:
{
lean_object* v_proof_1974_; lean_object* v___x_1975_; 
v_proof_1974_ = lean_ctor_get(v___x_1965_, 2);
lean_inc_ref(v_proof_1974_);
v___x_1975_ = lean_array_push(v_b_1958_, v_proof_1974_);
v___y_1960_ = v___x_1975_;
goto v___jp_1959_;
}
case 5:
{
lean_object* v_proof_1976_; lean_object* v___x_1977_; 
v_proof_1976_ = lean_ctor_get(v___x_1965_, 2);
lean_inc_ref(v_proof_1976_);
v___x_1977_ = lean_array_push(v_b_1958_, v_proof_1976_);
v___y_1960_ = v___x_1977_;
goto v___jp_1959_;
}
default: 
{
v___y_1960_ = v_b_1958_;
goto v___jp_1959_;
}
}
}
else
{
return v_b_1958_;
}
v___jp_1959_:
{
size_t v___x_1961_; size_t v___x_1962_; 
v___x_1961_ = ((size_t)1ULL);
v___x_1962_ = lean_usize_add(v_i_1956_, v___x_1961_);
v_i_1956_ = v___x_1962_;
v_b_1958_ = v___y_1960_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Order_orderCoreImp_spec__2_spec__3___boxed(lean_object* v_as_1978_, lean_object* v_i_1979_, lean_object* v_stop_1980_, lean_object* v_b_1981_){
_start:
{
size_t v_i_boxed_1982_; size_t v_stop_boxed_1983_; lean_object* v_res_1984_; 
v_i_boxed_1982_ = lean_unbox_usize(v_i_1979_);
lean_dec(v_i_1979_);
v_stop_boxed_1983_ = lean_unbox_usize(v_stop_1980_);
lean_dec(v_stop_1980_);
v_res_1984_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Order_orderCoreImp_spec__2_spec__3(v_as_1978_, v_i_boxed_1982_, v_stop_boxed_1983_, v_b_1981_);
lean_dec_ref(v_as_1978_);
return v_res_1984_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_Order_orderCoreImp_spec__2(lean_object* v_as_1987_, lean_object* v_start_1988_, lean_object* v_stop_1989_){
_start:
{
lean_object* v___x_1990_; uint8_t v___x_1991_; 
v___x_1990_ = ((lean_object*)(lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_Order_orderCoreImp_spec__2___closed__0));
v___x_1991_ = lean_nat_dec_lt(v_start_1988_, v_stop_1989_);
if (v___x_1991_ == 0)
{
return v___x_1990_;
}
else
{
lean_object* v___x_1992_; uint8_t v___x_1993_; 
v___x_1992_ = lean_array_get_size(v_as_1987_);
v___x_1993_ = lean_nat_dec_le(v_stop_1989_, v___x_1992_);
if (v___x_1993_ == 0)
{
uint8_t v___x_1994_; 
v___x_1994_ = lean_nat_dec_lt(v_start_1988_, v___x_1992_);
if (v___x_1994_ == 0)
{
return v___x_1990_;
}
else
{
size_t v___x_1995_; size_t v___x_1996_; lean_object* v___x_1997_; 
v___x_1995_ = lean_usize_of_nat(v_start_1988_);
v___x_1996_ = lean_usize_of_nat(v___x_1992_);
v___x_1997_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Order_orderCoreImp_spec__2_spec__3(v_as_1987_, v___x_1995_, v___x_1996_, v___x_1990_);
return v___x_1997_;
}
}
else
{
size_t v___x_1998_; size_t v___x_1999_; lean_object* v___x_2000_; 
v___x_1998_ = lean_usize_of_nat(v_start_1988_);
v___x_1999_ = lean_usize_of_nat(v_stop_1989_);
v___x_2000_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00Mathlib_Tactic_Order_orderCoreImp_spec__2_spec__3(v_as_1987_, v___x_1998_, v___x_1999_, v___x_1990_);
return v___x_2000_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_Order_orderCoreImp_spec__2___boxed(lean_object* v_as_2001_, lean_object* v_start_2002_, lean_object* v_stop_2003_){
_start:
{
lean_object* v_res_2004_; 
v_res_2004_ = lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_Order_orderCoreImp_spec__2(v_as_2001_, v_start_2002_, v_stop_2003_);
lean_dec(v_stop_2003_);
lean_dec(v_start_2002_);
lean_dec_ref(v_as_2001_);
return v_res_2004_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3_spec__11_spec__13___redArg(lean_object* v_x_2005_, lean_object* v_x_2006_, lean_object* v_x_2007_, lean_object* v_x_2008_){
_start:
{
lean_object* v_ks_2009_; lean_object* v_vs_2010_; lean_object* v___x_2012_; uint8_t v_isShared_2013_; uint8_t v_isSharedCheck_2034_; 
v_ks_2009_ = lean_ctor_get(v_x_2005_, 0);
v_vs_2010_ = lean_ctor_get(v_x_2005_, 1);
v_isSharedCheck_2034_ = !lean_is_exclusive(v_x_2005_);
if (v_isSharedCheck_2034_ == 0)
{
v___x_2012_ = v_x_2005_;
v_isShared_2013_ = v_isSharedCheck_2034_;
goto v_resetjp_2011_;
}
else
{
lean_inc(v_vs_2010_);
lean_inc(v_ks_2009_);
lean_dec(v_x_2005_);
v___x_2012_ = lean_box(0);
v_isShared_2013_ = v_isSharedCheck_2034_;
goto v_resetjp_2011_;
}
v_resetjp_2011_:
{
lean_object* v___x_2014_; uint8_t v___x_2015_; 
v___x_2014_ = lean_array_get_size(v_ks_2009_);
v___x_2015_ = lean_nat_dec_lt(v_x_2006_, v___x_2014_);
if (v___x_2015_ == 0)
{
lean_object* v___x_2016_; lean_object* v___x_2017_; lean_object* v___x_2019_; 
lean_dec(v_x_2006_);
v___x_2016_ = lean_array_push(v_ks_2009_, v_x_2007_);
v___x_2017_ = lean_array_push(v_vs_2010_, v_x_2008_);
if (v_isShared_2013_ == 0)
{
lean_ctor_set(v___x_2012_, 1, v___x_2017_);
lean_ctor_set(v___x_2012_, 0, v___x_2016_);
v___x_2019_ = v___x_2012_;
goto v_reusejp_2018_;
}
else
{
lean_object* v_reuseFailAlloc_2020_; 
v_reuseFailAlloc_2020_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2020_, 0, v___x_2016_);
lean_ctor_set(v_reuseFailAlloc_2020_, 1, v___x_2017_);
v___x_2019_ = v_reuseFailAlloc_2020_;
goto v_reusejp_2018_;
}
v_reusejp_2018_:
{
return v___x_2019_;
}
}
else
{
lean_object* v_k_x27_2021_; uint8_t v___x_2022_; 
v_k_x27_2021_ = lean_array_fget_borrowed(v_ks_2009_, v_x_2006_);
v___x_2022_ = l_Lean_instBEqMVarId_beq(v_x_2007_, v_k_x27_2021_);
if (v___x_2022_ == 0)
{
lean_object* v___x_2024_; 
if (v_isShared_2013_ == 0)
{
v___x_2024_ = v___x_2012_;
goto v_reusejp_2023_;
}
else
{
lean_object* v_reuseFailAlloc_2028_; 
v_reuseFailAlloc_2028_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2028_, 0, v_ks_2009_);
lean_ctor_set(v_reuseFailAlloc_2028_, 1, v_vs_2010_);
v___x_2024_ = v_reuseFailAlloc_2028_;
goto v_reusejp_2023_;
}
v_reusejp_2023_:
{
lean_object* v___x_2025_; lean_object* v___x_2026_; 
v___x_2025_ = lean_unsigned_to_nat(1u);
v___x_2026_ = lean_nat_add(v_x_2006_, v___x_2025_);
lean_dec(v_x_2006_);
v_x_2005_ = v___x_2024_;
v_x_2006_ = v___x_2026_;
goto _start;
}
}
else
{
lean_object* v___x_2029_; lean_object* v___x_2030_; lean_object* v___x_2032_; 
v___x_2029_ = lean_array_fset(v_ks_2009_, v_x_2006_, v_x_2007_);
v___x_2030_ = lean_array_fset(v_vs_2010_, v_x_2006_, v_x_2008_);
lean_dec(v_x_2006_);
if (v_isShared_2013_ == 0)
{
lean_ctor_set(v___x_2012_, 1, v___x_2030_);
lean_ctor_set(v___x_2012_, 0, v___x_2029_);
v___x_2032_ = v___x_2012_;
goto v_reusejp_2031_;
}
else
{
lean_object* v_reuseFailAlloc_2033_; 
v_reuseFailAlloc_2033_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2033_, 0, v___x_2029_);
lean_ctor_set(v_reuseFailAlloc_2033_, 1, v___x_2030_);
v___x_2032_ = v_reuseFailAlloc_2033_;
goto v_reusejp_2031_;
}
v_reusejp_2031_:
{
return v___x_2032_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3_spec__11___redArg(lean_object* v_n_2035_, lean_object* v_k_2036_, lean_object* v_v_2037_){
_start:
{
lean_object* v___x_2038_; lean_object* v___x_2039_; 
v___x_2038_ = lean_unsigned_to_nat(0u);
v___x_2039_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3_spec__11_spec__13___redArg(v_n_2035_, v___x_2038_, v_k_2036_, v_v_2037_);
return v___x_2039_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_2040_; 
v___x_2040_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_2040_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3___redArg(lean_object* v_x_2041_, size_t v_x_2042_, size_t v_x_2043_, lean_object* v_x_2044_, lean_object* v_x_2045_){
_start:
{
if (lean_obj_tag(v_x_2041_) == 0)
{
lean_object* v_es_2046_; size_t v___x_2047_; size_t v___x_2048_; lean_object* v_j_2049_; lean_object* v___x_2050_; uint8_t v___x_2051_; 
v_es_2046_ = lean_ctor_get(v_x_2041_, 0);
v___x_2047_ = ((size_t)31ULL);
v___x_2048_ = lean_usize_land(v_x_2042_, v___x_2047_);
v_j_2049_ = lean_usize_to_nat(v___x_2048_);
v___x_2050_ = lean_array_get_size(v_es_2046_);
v___x_2051_ = lean_nat_dec_lt(v_j_2049_, v___x_2050_);
if (v___x_2051_ == 0)
{
lean_dec(v_j_2049_);
lean_dec(v_x_2045_);
lean_dec(v_x_2044_);
return v_x_2041_;
}
else
{
lean_object* v___x_2053_; uint8_t v_isShared_2054_; uint8_t v_isSharedCheck_2090_; 
lean_inc_ref(v_es_2046_);
v_isSharedCheck_2090_ = !lean_is_exclusive(v_x_2041_);
if (v_isSharedCheck_2090_ == 0)
{
lean_object* v_unused_2091_; 
v_unused_2091_ = lean_ctor_get(v_x_2041_, 0);
lean_dec(v_unused_2091_);
v___x_2053_ = v_x_2041_;
v_isShared_2054_ = v_isSharedCheck_2090_;
goto v_resetjp_2052_;
}
else
{
lean_dec(v_x_2041_);
v___x_2053_ = lean_box(0);
v_isShared_2054_ = v_isSharedCheck_2090_;
goto v_resetjp_2052_;
}
v_resetjp_2052_:
{
lean_object* v_v_2055_; lean_object* v___x_2056_; lean_object* v_xs_x27_2057_; lean_object* v___y_2059_; 
v_v_2055_ = lean_array_fget(v_es_2046_, v_j_2049_);
v___x_2056_ = lean_box(0);
v_xs_x27_2057_ = lean_array_fset(v_es_2046_, v_j_2049_, v___x_2056_);
switch(lean_obj_tag(v_v_2055_))
{
case 0:
{
lean_object* v_key_2064_; lean_object* v_val_2065_; lean_object* v___x_2067_; uint8_t v_isShared_2068_; uint8_t v_isSharedCheck_2075_; 
v_key_2064_ = lean_ctor_get(v_v_2055_, 0);
v_val_2065_ = lean_ctor_get(v_v_2055_, 1);
v_isSharedCheck_2075_ = !lean_is_exclusive(v_v_2055_);
if (v_isSharedCheck_2075_ == 0)
{
v___x_2067_ = v_v_2055_;
v_isShared_2068_ = v_isSharedCheck_2075_;
goto v_resetjp_2066_;
}
else
{
lean_inc(v_val_2065_);
lean_inc(v_key_2064_);
lean_dec(v_v_2055_);
v___x_2067_ = lean_box(0);
v_isShared_2068_ = v_isSharedCheck_2075_;
goto v_resetjp_2066_;
}
v_resetjp_2066_:
{
uint8_t v___x_2069_; 
v___x_2069_ = l_Lean_instBEqMVarId_beq(v_x_2044_, v_key_2064_);
if (v___x_2069_ == 0)
{
lean_object* v___x_2070_; lean_object* v___x_2071_; 
lean_del_object(v___x_2067_);
v___x_2070_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_2064_, v_val_2065_, v_x_2044_, v_x_2045_);
v___x_2071_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2071_, 0, v___x_2070_);
v___y_2059_ = v___x_2071_;
goto v___jp_2058_;
}
else
{
lean_object* v___x_2073_; 
lean_dec(v_val_2065_);
lean_dec(v_key_2064_);
if (v_isShared_2068_ == 0)
{
lean_ctor_set(v___x_2067_, 1, v_x_2045_);
lean_ctor_set(v___x_2067_, 0, v_x_2044_);
v___x_2073_ = v___x_2067_;
goto v_reusejp_2072_;
}
else
{
lean_object* v_reuseFailAlloc_2074_; 
v_reuseFailAlloc_2074_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2074_, 0, v_x_2044_);
lean_ctor_set(v_reuseFailAlloc_2074_, 1, v_x_2045_);
v___x_2073_ = v_reuseFailAlloc_2074_;
goto v_reusejp_2072_;
}
v_reusejp_2072_:
{
v___y_2059_ = v___x_2073_;
goto v___jp_2058_;
}
}
}
}
case 1:
{
lean_object* v_node_2076_; lean_object* v___x_2078_; uint8_t v_isShared_2079_; uint8_t v_isSharedCheck_2088_; 
v_node_2076_ = lean_ctor_get(v_v_2055_, 0);
v_isSharedCheck_2088_ = !lean_is_exclusive(v_v_2055_);
if (v_isSharedCheck_2088_ == 0)
{
v___x_2078_ = v_v_2055_;
v_isShared_2079_ = v_isSharedCheck_2088_;
goto v_resetjp_2077_;
}
else
{
lean_inc(v_node_2076_);
lean_dec(v_v_2055_);
v___x_2078_ = lean_box(0);
v_isShared_2079_ = v_isSharedCheck_2088_;
goto v_resetjp_2077_;
}
v_resetjp_2077_:
{
size_t v___x_2080_; size_t v___x_2081_; size_t v___x_2082_; size_t v___x_2083_; lean_object* v___x_2084_; lean_object* v___x_2086_; 
v___x_2080_ = ((size_t)5ULL);
v___x_2081_ = lean_usize_shift_right(v_x_2042_, v___x_2080_);
v___x_2082_ = ((size_t)1ULL);
v___x_2083_ = lean_usize_add(v_x_2043_, v___x_2082_);
v___x_2084_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3___redArg(v_node_2076_, v___x_2081_, v___x_2083_, v_x_2044_, v_x_2045_);
if (v_isShared_2079_ == 0)
{
lean_ctor_set(v___x_2078_, 0, v___x_2084_);
v___x_2086_ = v___x_2078_;
goto v_reusejp_2085_;
}
else
{
lean_object* v_reuseFailAlloc_2087_; 
v_reuseFailAlloc_2087_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2087_, 0, v___x_2084_);
v___x_2086_ = v_reuseFailAlloc_2087_;
goto v_reusejp_2085_;
}
v_reusejp_2085_:
{
v___y_2059_ = v___x_2086_;
goto v___jp_2058_;
}
}
}
default: 
{
lean_object* v___x_2089_; 
v___x_2089_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2089_, 0, v_x_2044_);
lean_ctor_set(v___x_2089_, 1, v_x_2045_);
v___y_2059_ = v___x_2089_;
goto v___jp_2058_;
}
}
v___jp_2058_:
{
lean_object* v___x_2060_; lean_object* v___x_2062_; 
v___x_2060_ = lean_array_fset(v_xs_x27_2057_, v_j_2049_, v___y_2059_);
lean_dec(v_j_2049_);
if (v_isShared_2054_ == 0)
{
lean_ctor_set(v___x_2053_, 0, v___x_2060_);
v___x_2062_ = v___x_2053_;
goto v_reusejp_2061_;
}
else
{
lean_object* v_reuseFailAlloc_2063_; 
v_reuseFailAlloc_2063_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2063_, 0, v___x_2060_);
v___x_2062_ = v_reuseFailAlloc_2063_;
goto v_reusejp_2061_;
}
v_reusejp_2061_:
{
return v___x_2062_;
}
}
}
}
}
else
{
lean_object* v_ks_2092_; lean_object* v_vs_2093_; lean_object* v___x_2095_; uint8_t v_isShared_2096_; uint8_t v_isSharedCheck_2113_; 
v_ks_2092_ = lean_ctor_get(v_x_2041_, 0);
v_vs_2093_ = lean_ctor_get(v_x_2041_, 1);
v_isSharedCheck_2113_ = !lean_is_exclusive(v_x_2041_);
if (v_isSharedCheck_2113_ == 0)
{
v___x_2095_ = v_x_2041_;
v_isShared_2096_ = v_isSharedCheck_2113_;
goto v_resetjp_2094_;
}
else
{
lean_inc(v_vs_2093_);
lean_inc(v_ks_2092_);
lean_dec(v_x_2041_);
v___x_2095_ = lean_box(0);
v_isShared_2096_ = v_isSharedCheck_2113_;
goto v_resetjp_2094_;
}
v_resetjp_2094_:
{
lean_object* v___x_2098_; 
if (v_isShared_2096_ == 0)
{
v___x_2098_ = v___x_2095_;
goto v_reusejp_2097_;
}
else
{
lean_object* v_reuseFailAlloc_2112_; 
v_reuseFailAlloc_2112_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2112_, 0, v_ks_2092_);
lean_ctor_set(v_reuseFailAlloc_2112_, 1, v_vs_2093_);
v___x_2098_ = v_reuseFailAlloc_2112_;
goto v_reusejp_2097_;
}
v_reusejp_2097_:
{
lean_object* v_newNode_2099_; uint8_t v___y_2101_; size_t v___x_2107_; uint8_t v___x_2108_; 
v_newNode_2099_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3_spec__11___redArg(v___x_2098_, v_x_2044_, v_x_2045_);
v___x_2107_ = ((size_t)7ULL);
v___x_2108_ = lean_usize_dec_le(v___x_2107_, v_x_2043_);
if (v___x_2108_ == 0)
{
lean_object* v___x_2109_; lean_object* v___x_2110_; uint8_t v___x_2111_; 
v___x_2109_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_2099_);
v___x_2110_ = lean_unsigned_to_nat(4u);
v___x_2111_ = lean_nat_dec_lt(v___x_2109_, v___x_2110_);
lean_dec(v___x_2109_);
v___y_2101_ = v___x_2111_;
goto v___jp_2100_;
}
else
{
v___y_2101_ = v___x_2108_;
goto v___jp_2100_;
}
v___jp_2100_:
{
if (v___y_2101_ == 0)
{
lean_object* v_ks_2102_; lean_object* v_vs_2103_; lean_object* v___x_2104_; lean_object* v___x_2105_; lean_object* v___x_2106_; 
v_ks_2102_ = lean_ctor_get(v_newNode_2099_, 0);
lean_inc_ref(v_ks_2102_);
v_vs_2103_ = lean_ctor_get(v_newNode_2099_, 1);
lean_inc_ref(v_vs_2103_);
lean_dec_ref(v_newNode_2099_);
v___x_2104_ = lean_unsigned_to_nat(0u);
v___x_2105_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3___redArg___closed__0);
v___x_2106_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3_spec__12___redArg(v_x_2043_, v_ks_2102_, v_vs_2103_, v___x_2104_, v___x_2105_);
lean_dec_ref(v_vs_2103_);
lean_dec_ref(v_ks_2102_);
return v___x_2106_;
}
else
{
return v_newNode_2099_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3_spec__12___redArg(size_t v_depth_2114_, lean_object* v_keys_2115_, lean_object* v_vals_2116_, lean_object* v_i_2117_, lean_object* v_entries_2118_){
_start:
{
lean_object* v___x_2119_; uint8_t v___x_2120_; 
v___x_2119_ = lean_array_get_size(v_keys_2115_);
v___x_2120_ = lean_nat_dec_lt(v_i_2117_, v___x_2119_);
if (v___x_2120_ == 0)
{
lean_dec(v_i_2117_);
return v_entries_2118_;
}
else
{
lean_object* v_k_2121_; lean_object* v_v_2122_; uint64_t v___x_2123_; size_t v_h_2124_; size_t v___x_2125_; lean_object* v___x_2126_; size_t v___x_2127_; size_t v___x_2128_; size_t v___x_2129_; size_t v_h_2130_; lean_object* v___x_2131_; lean_object* v___x_2132_; 
v_k_2121_ = lean_array_fget_borrowed(v_keys_2115_, v_i_2117_);
v_v_2122_ = lean_array_fget_borrowed(v_vals_2116_, v_i_2117_);
v___x_2123_ = l_Lean_instHashableMVarId_hash(v_k_2121_);
v_h_2124_ = lean_uint64_to_usize(v___x_2123_);
v___x_2125_ = ((size_t)5ULL);
v___x_2126_ = lean_unsigned_to_nat(1u);
v___x_2127_ = ((size_t)1ULL);
v___x_2128_ = lean_usize_sub(v_depth_2114_, v___x_2127_);
v___x_2129_ = lean_usize_mul(v___x_2125_, v___x_2128_);
v_h_2130_ = lean_usize_shift_right(v_h_2124_, v___x_2129_);
v___x_2131_ = lean_nat_add(v_i_2117_, v___x_2126_);
lean_dec(v_i_2117_);
lean_inc(v_v_2122_);
lean_inc(v_k_2121_);
v___x_2132_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3___redArg(v_entries_2118_, v_h_2130_, v_depth_2114_, v_k_2121_, v_v_2122_);
v_i_2117_ = v___x_2131_;
v_entries_2118_ = v___x_2132_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3_spec__12___redArg___boxed(lean_object* v_depth_2134_, lean_object* v_keys_2135_, lean_object* v_vals_2136_, lean_object* v_i_2137_, lean_object* v_entries_2138_){
_start:
{
size_t v_depth_boxed_2139_; lean_object* v_res_2140_; 
v_depth_boxed_2139_ = lean_unbox_usize(v_depth_2134_);
lean_dec(v_depth_2134_);
v_res_2140_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3_spec__12___redArg(v_depth_boxed_2139_, v_keys_2135_, v_vals_2136_, v_i_2137_, v_entries_2138_);
lean_dec_ref(v_vals_2136_);
lean_dec_ref(v_keys_2135_);
return v_res_2140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3___redArg___boxed(lean_object* v_x_2141_, lean_object* v_x_2142_, lean_object* v_x_2143_, lean_object* v_x_2144_, lean_object* v_x_2145_){
_start:
{
size_t v_x_68231__boxed_2146_; size_t v_x_68232__boxed_2147_; lean_object* v_res_2148_; 
v_x_68231__boxed_2146_ = lean_unbox_usize(v_x_2142_);
lean_dec(v_x_2142_);
v_x_68232__boxed_2147_ = lean_unbox_usize(v_x_2143_);
lean_dec(v_x_2143_);
v_res_2148_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3___redArg(v_x_2141_, v_x_68231__boxed_2146_, v_x_68232__boxed_2147_, v_x_2144_, v_x_2145_);
return v_res_2148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1___redArg(lean_object* v_x_2149_, lean_object* v_x_2150_, lean_object* v_x_2151_){
_start:
{
uint64_t v___x_2152_; size_t v___x_2153_; size_t v___x_2154_; lean_object* v___x_2155_; 
v___x_2152_ = l_Lean_instHashableMVarId_hash(v_x_2150_);
v___x_2153_ = lean_uint64_to_usize(v___x_2152_);
v___x_2154_ = ((size_t)1ULL);
v___x_2155_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3___redArg(v_x_2149_, v___x_2153_, v___x_2154_, v_x_2150_, v_x_2151_);
return v___x_2155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1___redArg(lean_object* v_mvarId_2156_, lean_object* v_val_2157_, lean_object* v___y_2158_){
_start:
{
lean_object* v___x_2160_; lean_object* v_mctx_2161_; lean_object* v_cache_2162_; lean_object* v_zetaDeltaFVarIds_2163_; lean_object* v_postponed_2164_; lean_object* v_diag_2165_; lean_object* v___x_2167_; uint8_t v_isShared_2168_; uint8_t v_isSharedCheck_2193_; 
v___x_2160_ = lean_st_ref_take(v___y_2158_);
v_mctx_2161_ = lean_ctor_get(v___x_2160_, 0);
v_cache_2162_ = lean_ctor_get(v___x_2160_, 1);
v_zetaDeltaFVarIds_2163_ = lean_ctor_get(v___x_2160_, 2);
v_postponed_2164_ = lean_ctor_get(v___x_2160_, 3);
v_diag_2165_ = lean_ctor_get(v___x_2160_, 4);
v_isSharedCheck_2193_ = !lean_is_exclusive(v___x_2160_);
if (v_isSharedCheck_2193_ == 0)
{
v___x_2167_ = v___x_2160_;
v_isShared_2168_ = v_isSharedCheck_2193_;
goto v_resetjp_2166_;
}
else
{
lean_inc(v_diag_2165_);
lean_inc(v_postponed_2164_);
lean_inc(v_zetaDeltaFVarIds_2163_);
lean_inc(v_cache_2162_);
lean_inc(v_mctx_2161_);
lean_dec(v___x_2160_);
v___x_2167_ = lean_box(0);
v_isShared_2168_ = v_isSharedCheck_2193_;
goto v_resetjp_2166_;
}
v_resetjp_2166_:
{
lean_object* v_depth_2169_; lean_object* v_levelAssignDepth_2170_; lean_object* v_lmvarCounter_2171_; lean_object* v_mvarCounter_2172_; lean_object* v_lDecls_2173_; lean_object* v_decls_2174_; lean_object* v_userNames_2175_; lean_object* v_lAssignment_2176_; lean_object* v_eAssignment_2177_; lean_object* v_dAssignment_2178_; lean_object* v___x_2180_; uint8_t v_isShared_2181_; uint8_t v_isSharedCheck_2192_; 
v_depth_2169_ = lean_ctor_get(v_mctx_2161_, 0);
v_levelAssignDepth_2170_ = lean_ctor_get(v_mctx_2161_, 1);
v_lmvarCounter_2171_ = lean_ctor_get(v_mctx_2161_, 2);
v_mvarCounter_2172_ = lean_ctor_get(v_mctx_2161_, 3);
v_lDecls_2173_ = lean_ctor_get(v_mctx_2161_, 4);
v_decls_2174_ = lean_ctor_get(v_mctx_2161_, 5);
v_userNames_2175_ = lean_ctor_get(v_mctx_2161_, 6);
v_lAssignment_2176_ = lean_ctor_get(v_mctx_2161_, 7);
v_eAssignment_2177_ = lean_ctor_get(v_mctx_2161_, 8);
v_dAssignment_2178_ = lean_ctor_get(v_mctx_2161_, 9);
v_isSharedCheck_2192_ = !lean_is_exclusive(v_mctx_2161_);
if (v_isSharedCheck_2192_ == 0)
{
v___x_2180_ = v_mctx_2161_;
v_isShared_2181_ = v_isSharedCheck_2192_;
goto v_resetjp_2179_;
}
else
{
lean_inc(v_dAssignment_2178_);
lean_inc(v_eAssignment_2177_);
lean_inc(v_lAssignment_2176_);
lean_inc(v_userNames_2175_);
lean_inc(v_decls_2174_);
lean_inc(v_lDecls_2173_);
lean_inc(v_mvarCounter_2172_);
lean_inc(v_lmvarCounter_2171_);
lean_inc(v_levelAssignDepth_2170_);
lean_inc(v_depth_2169_);
lean_dec(v_mctx_2161_);
v___x_2180_ = lean_box(0);
v_isShared_2181_ = v_isSharedCheck_2192_;
goto v_resetjp_2179_;
}
v_resetjp_2179_:
{
lean_object* v___x_2182_; lean_object* v___x_2184_; 
v___x_2182_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1___redArg(v_eAssignment_2177_, v_mvarId_2156_, v_val_2157_);
if (v_isShared_2181_ == 0)
{
lean_ctor_set(v___x_2180_, 8, v___x_2182_);
v___x_2184_ = v___x_2180_;
goto v_reusejp_2183_;
}
else
{
lean_object* v_reuseFailAlloc_2191_; 
v_reuseFailAlloc_2191_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_2191_, 0, v_depth_2169_);
lean_ctor_set(v_reuseFailAlloc_2191_, 1, v_levelAssignDepth_2170_);
lean_ctor_set(v_reuseFailAlloc_2191_, 2, v_lmvarCounter_2171_);
lean_ctor_set(v_reuseFailAlloc_2191_, 3, v_mvarCounter_2172_);
lean_ctor_set(v_reuseFailAlloc_2191_, 4, v_lDecls_2173_);
lean_ctor_set(v_reuseFailAlloc_2191_, 5, v_decls_2174_);
lean_ctor_set(v_reuseFailAlloc_2191_, 6, v_userNames_2175_);
lean_ctor_set(v_reuseFailAlloc_2191_, 7, v_lAssignment_2176_);
lean_ctor_set(v_reuseFailAlloc_2191_, 8, v___x_2182_);
lean_ctor_set(v_reuseFailAlloc_2191_, 9, v_dAssignment_2178_);
v___x_2184_ = v_reuseFailAlloc_2191_;
goto v_reusejp_2183_;
}
v_reusejp_2183_:
{
lean_object* v___x_2186_; 
if (v_isShared_2168_ == 0)
{
lean_ctor_set(v___x_2167_, 0, v___x_2184_);
v___x_2186_ = v___x_2167_;
goto v_reusejp_2185_;
}
else
{
lean_object* v_reuseFailAlloc_2190_; 
v_reuseFailAlloc_2190_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2190_, 0, v___x_2184_);
lean_ctor_set(v_reuseFailAlloc_2190_, 1, v_cache_2162_);
lean_ctor_set(v_reuseFailAlloc_2190_, 2, v_zetaDeltaFVarIds_2163_);
lean_ctor_set(v_reuseFailAlloc_2190_, 3, v_postponed_2164_);
lean_ctor_set(v_reuseFailAlloc_2190_, 4, v_diag_2165_);
v___x_2186_ = v_reuseFailAlloc_2190_;
goto v_reusejp_2185_;
}
v_reusejp_2185_:
{
lean_object* v___x_2187_; lean_object* v___x_2188_; lean_object* v___x_2189_; 
v___x_2187_ = lean_st_ref_set(v___y_2158_, v___x_2186_);
v___x_2188_ = lean_box(0);
v___x_2189_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2189_, 0, v___x_2188_);
return v___x_2189_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1___redArg___boxed(lean_object* v_mvarId_2194_, lean_object* v_val_2195_, lean_object* v___y_2196_, lean_object* v___y_2197_){
_start:
{
lean_object* v_res_2198_; 
v_res_2198_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1___redArg(v_mvarId_2194_, v_val_2195_, v___y_2196_);
lean_dec(v___y_2196_);
return v_res_2198_;
}
}
static lean_object* _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__7(void){
_start:
{
lean_object* v___x_2213_; lean_object* v___x_2214_; lean_object* v___x_2215_; 
v___x_2213_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__1_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_));
v___x_2214_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___lam__0___closed__1));
v___x_2215_ = l_Lean_Name_append(v___x_2214_, v___x_2213_);
return v___x_2215_;
}
}
static lean_object* _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__9(void){
_start:
{
lean_object* v___x_2217_; lean_object* v___x_2218_; 
v___x_2217_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__8));
v___x_2218_ = l_Lean_stringToMessageData(v___x_2217_);
return v___x_2218_;
}
}
static lean_object* _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__11(void){
_start:
{
lean_object* v___x_2220_; lean_object* v___x_2221_; 
v___x_2220_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__10));
v___x_2221_ = l_Lean_stringToMessageData(v___x_2220_);
return v___x_2221_;
}
}
static lean_object* _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__13(void){
_start:
{
lean_object* v___x_2223_; lean_object* v___x_2224_; 
v___x_2223_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__12));
v___x_2224_ = l_Lean_stringToMessageData(v___x_2223_);
return v___x_2224_;
}
}
static lean_object* _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__15(void){
_start:
{
lean_object* v___x_2226_; lean_object* v___x_2227_; 
v___x_2226_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__14));
v___x_2227_ = l_Lean_stringToMessageData(v___x_2226_);
return v___x_2227_;
}
}
static lean_object* _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__17(void){
_start:
{
lean_object* v___x_2229_; lean_object* v___x_2230_; 
v___x_2229_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__16));
v___x_2230_ = l_Lean_stringToMessageData(v___x_2229_);
return v___x_2230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5(lean_object* v_g_2234_, lean_object* v_a_2235_, lean_object* v_a_2236_, lean_object* v___y_2237_, lean_object* v___y_2238_, lean_object* v___y_2239_, lean_object* v___y_2240_, lean_object* v___y_2241_, lean_object* v___y_2242_){
_start:
{
if (lean_obj_tag(v_a_2235_) == 0)
{
lean_object* v___x_2244_; lean_object* v___x_2245_; 
lean_dec(v_g_2234_);
v___x_2244_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2244_, 0, v_a_2236_);
v___x_2245_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2245_, 0, v___x_2244_);
return v___x_2245_;
}
else
{
lean_object* v_key_2246_; lean_object* v_value_2247_; lean_object* v_tail_2248_; lean_object* v___x_2249_; 
lean_dec_ref(v_a_2236_);
v_key_2246_ = lean_ctor_get(v_a_2235_, 0);
lean_inc_n(v_key_2246_, 2);
v_value_2247_ = lean_ctor_get(v_a_2235_, 1);
lean_inc(v_value_2247_);
v_tail_2248_ = lean_ctor_get(v_a_2235_, 2);
lean_inc(v_tail_2248_);
lean_dec_ref_known(v_a_2235_, 3);
v___x_2249_ = lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance(v_key_2246_, v___y_2239_, v___y_2240_, v___y_2241_, v___y_2242_);
if (lean_obj_tag(v___x_2249_) == 0)
{
lean_object* v_a_2250_; lean_object* v___x_2252_; uint8_t v_isShared_2253_; uint8_t v_isSharedCheck_2531_; 
v_a_2250_ = lean_ctor_get(v___x_2249_, 0);
v_isSharedCheck_2531_ = !lean_is_exclusive(v___x_2249_);
if (v_isSharedCheck_2531_ == 0)
{
v___x_2252_ = v___x_2249_;
v_isShared_2253_ = v_isSharedCheck_2531_;
goto v_resetjp_2251_;
}
else
{
lean_inc(v_a_2250_);
lean_dec(v___x_2249_);
v___x_2252_ = lean_box(0);
v_isShared_2253_ = v_isSharedCheck_2531_;
goto v_resetjp_2251_;
}
v_resetjp_2251_:
{
lean_object* v___x_2254_; lean_object* v___y_2256_; uint8_t v___y_2257_; 
v___x_2254_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__0));
if (lean_obj_tag(v_a_2250_) == 1)
{
lean_object* v_val_2262_; lean_object* v___y_2264_; lean_object* v___y_2265_; lean_object* v___y_2266_; lean_object* v___y_2267_; lean_object* v___y_2268_; lean_object* v___y_2269_; lean_object* v___y_2270_; lean_object* v___y_2271_; lean_object* v_inheritedTraceOptions_2404_; lean_object* v___x_2405_; lean_object* v___x_2406_; lean_object* v_a_2407_; lean_object* v___x_2409_; uint8_t v_isShared_2410_; uint8_t v_isSharedCheck_2529_; 
v_val_2262_ = lean_ctor_get(v_a_2250_, 0);
lean_inc(v_val_2262_);
lean_dec_ref_known(v_a_2250_, 1);
v_inheritedTraceOptions_2404_ = lean_ctor_get(v___y_2241_, 13);
v___x_2405_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__1_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_));
v___x_2406_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___lam__0(v___x_2405_, v_inheritedTraceOptions_2404_, v___y_2237_, v___y_2238_, v___y_2239_, v___y_2240_, v___y_2241_, v___y_2242_);
v_a_2407_ = lean_ctor_get(v___x_2406_, 0);
v_isSharedCheck_2529_ = !lean_is_exclusive(v___x_2406_);
if (v_isSharedCheck_2529_ == 0)
{
v___x_2409_ = v___x_2406_;
v_isShared_2410_ = v_isSharedCheck_2529_;
goto v_resetjp_2408_;
}
else
{
lean_inc(v_a_2407_);
lean_dec(v___x_2406_);
v___x_2409_ = lean_box(0);
v_isShared_2410_ = v_isSharedCheck_2529_;
goto v_resetjp_2408_;
}
v___jp_2263_:
{
lean_object* v___x_2272_; 
v___x_2272_ = lp_mathlib_Mathlib_Tactic_Order_Graph_constructLeGraph(v___y_2265_, v___y_2268_, v___y_2269_, v___y_2270_, v___y_2271_);
if (lean_obj_tag(v___x_2272_) == 0)
{
lean_object* v_a_2273_; lean_object* v___x_2274_; 
v_a_2273_ = lean_ctor_get(v___x_2272_, 0);
lean_inc(v_a_2273_);
lean_dec_ref_known(v___x_2272_, 1);
v___x_2274_ = lp_mathlib_Mathlib_Tactic_Order_updateGraphWithNltInfSup(v_a_2273_, v___y_2265_, v___y_2266_, v___y_2267_, v___y_2268_, v___y_2269_, v___y_2270_, v___y_2271_);
if (lean_obj_tag(v___x_2274_) == 0)
{
lean_object* v_a_2275_; uint8_t v___x_2276_; uint8_t v___x_2277_; uint8_t v___x_2278_; 
v_a_2275_ = lean_ctor_get(v___x_2274_, 0);
lean_inc(v_a_2275_);
lean_dec_ref_known(v___x_2274_, 1);
v___x_2276_ = 2;
v___x_2277_ = lean_unbox(v_val_2262_);
v___x_2278_ = lp_mathlib_Mathlib_Tactic_Order_instBEqOrderType_beq(v___x_2277_, v___x_2276_);
if (v___x_2278_ == 0)
{
lean_object* v___x_2279_; 
v___x_2279_ = lp_mathlib_Mathlib_Tactic_Order_findContradictionWithNe(v_a_2275_, v___y_2265_, v___y_2266_, v___y_2267_, v___y_2268_, v___y_2269_, v___y_2270_, v___y_2271_);
lean_dec_ref(v___y_2265_);
lean_dec(v_a_2275_);
if (lean_obj_tag(v___x_2279_) == 0)
{
lean_object* v_a_2280_; 
v_a_2280_ = lean_ctor_get(v___x_2279_, 0);
lean_inc(v_a_2280_);
lean_dec_ref_known(v___x_2279_, 1);
if (lean_obj_tag(v_a_2280_) == 1)
{
lean_object* v_val_2281_; lean_object* v___x_2282_; lean_object* v___x_2284_; uint8_t v_isShared_2285_; uint8_t v_isSharedCheck_2290_; 
lean_dec_ref(v___y_2264_);
lean_dec(v_val_2262_);
lean_del_object(v___x_2252_);
lean_dec(v_tail_2248_);
lean_dec(v_key_2246_);
v_val_2281_ = lean_ctor_get(v_a_2280_, 0);
lean_inc(v_val_2281_);
lean_dec_ref_known(v_a_2280_, 1);
v___x_2282_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1___redArg(v_g_2234_, v_val_2281_, v___y_2269_);
v_isSharedCheck_2290_ = !lean_is_exclusive(v___x_2282_);
if (v_isSharedCheck_2290_ == 0)
{
lean_object* v_unused_2291_; 
v_unused_2291_ = lean_ctor_get(v___x_2282_, 0);
lean_dec(v_unused_2291_);
v___x_2284_ = v___x_2282_;
v_isShared_2285_ = v_isSharedCheck_2290_;
goto v_resetjp_2283_;
}
else
{
lean_dec(v___x_2282_);
v___x_2284_ = lean_box(0);
v_isShared_2285_ = v_isSharedCheck_2290_;
goto v_resetjp_2283_;
}
v_resetjp_2283_:
{
lean_object* v___x_2286_; lean_object* v___x_2288_; 
v___x_2286_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__3));
if (v_isShared_2285_ == 0)
{
lean_ctor_set(v___x_2284_, 0, v___x_2286_);
v___x_2288_ = v___x_2284_;
goto v_reusejp_2287_;
}
else
{
lean_object* v_reuseFailAlloc_2289_; 
v_reuseFailAlloc_2289_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2289_, 0, v___x_2286_);
v___x_2288_ = v_reuseFailAlloc_2289_;
goto v_reusejp_2287_;
}
v_reusejp_2287_:
{
return v___x_2288_;
}
}
}
else
{
uint8_t v___x_2292_; uint8_t v___x_2293_; uint8_t v___x_2294_; 
lean_dec(v_a_2280_);
v___x_2292_ = 0;
v___x_2293_ = lean_unbox(v_val_2262_);
lean_dec(v_val_2262_);
v___x_2294_ = lp_mathlib_Mathlib_Tactic_Order_instBEqOrderType_beq(v___x_2293_, v___x_2292_);
if (v___x_2294_ == 0)
{
lean_dec_ref(v___y_2264_);
lean_del_object(v___x_2252_);
lean_dec(v_key_2246_);
v_a_2235_ = v_tail_2248_;
v_a_2236_ = v___x_2254_;
goto _start;
}
else
{
lean_object* v___x_2296_; 
v___x_2296_ = lp_mathlib_Qq_getLevelQ_x27(v_key_2246_, v___y_2268_, v___y_2269_, v___y_2270_, v___y_2271_);
if (lean_obj_tag(v___x_2296_) == 0)
{
lean_object* v_a_2297_; lean_object* v_fst_2298_; lean_object* v_snd_2299_; lean_object* v___x_2301_; uint8_t v_isShared_2302_; uint8_t v_isSharedCheck_2349_; 
v_a_2297_ = lean_ctor_get(v___x_2296_, 0);
lean_inc(v_a_2297_);
lean_dec_ref_known(v___x_2296_, 1);
v_fst_2298_ = lean_ctor_get(v_a_2297_, 0);
v_snd_2299_ = lean_ctor_get(v_a_2297_, 1);
v_isSharedCheck_2349_ = !lean_is_exclusive(v_a_2297_);
if (v_isSharedCheck_2349_ == 0)
{
v___x_2301_ = v_a_2297_;
v_isShared_2302_ = v_isSharedCheck_2349_;
goto v_resetjp_2300_;
}
else
{
lean_inc(v_snd_2299_);
lean_inc(v_fst_2298_);
lean_dec(v_a_2297_);
v___x_2301_ = lean_box(0);
v_isShared_2302_ = v_isSharedCheck_2349_;
goto v_resetjp_2300_;
}
v_resetjp_2300_:
{
lean_object* v___x_2303_; lean_object* v___x_2304_; lean_object* v___x_2306_; 
v___x_2303_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__5));
v___x_2304_ = lean_box(0);
lean_inc(v_fst_2298_);
if (v_isShared_2302_ == 0)
{
lean_ctor_set_tag(v___x_2301_, 1);
lean_ctor_set(v___x_2301_, 1, v___x_2304_);
v___x_2306_ = v___x_2301_;
goto v_reusejp_2305_;
}
else
{
lean_object* v_reuseFailAlloc_2348_; 
v_reuseFailAlloc_2348_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2348_, 0, v_fst_2298_);
lean_ctor_set(v_reuseFailAlloc_2348_, 1, v___x_2304_);
v___x_2306_ = v_reuseFailAlloc_2348_;
goto v_reusejp_2305_;
}
v_reusejp_2305_:
{
lean_object* v___x_2307_; lean_object* v___x_2308_; lean_object* v___x_2309_; 
v___x_2307_ = l_Lean_Expr_const___override(v___x_2303_, v___x_2306_);
lean_inc(v_snd_2299_);
v___x_2308_ = l_Lean_Expr_app___override(v___x_2307_, v_snd_2299_);
v___x_2309_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_2308_, v___y_2268_, v___y_2269_, v___y_2270_, v___y_2271_);
if (lean_obj_tag(v___x_2309_) == 0)
{
lean_object* v_a_2310_; lean_object* v___x_2311_; 
v_a_2310_ = lean_ctor_get(v___x_2309_, 0);
lean_inc(v_a_2310_);
lean_dec_ref_known(v___x_2309_, 1);
v___x_2311_ = lp_mathlib_Mathlib_Tactic_Order_ToInt_translateToInt(v_fst_2298_, v_snd_2299_, v_a_2310_, v___y_2264_, v___y_2266_, v___y_2267_, v___y_2268_, v___y_2269_, v___y_2270_, v___y_2271_);
lean_dec_ref(v___y_2264_);
if (lean_obj_tag(v___x_2311_) == 0)
{
lean_object* v_a_2312_; lean_object* v_snd_2313_; lean_object* v___x_2314_; lean_object* v___x_2315_; lean_object* v___x_2316_; lean_object* v___x_2317_; lean_object* v___x_2318_; lean_object* v___x_2319_; 
v_a_2312_ = lean_ctor_get(v___x_2311_, 0);
lean_inc(v_a_2312_);
lean_dec_ref_known(v___x_2311_, 1);
v_snd_2313_ = lean_ctor_get(v_a_2312_, 1);
lean_inc(v_snd_2313_);
lean_dec(v_a_2312_);
v___x_2314_ = lean_unsigned_to_nat(0u);
v___x_2315_ = lean_array_get_size(v_snd_2313_);
v___x_2316_ = lp_mathlib_Array_filterMapM___at___00Mathlib_Tactic_Order_orderCoreImp_spec__2(v_snd_2313_, v___x_2314_, v___x_2315_);
lean_dec(v_snd_2313_);
v___x_2317_ = lean_array_to_list(v___x_2316_);
v___x_2318_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v___x_2318_, 0, v___x_2294_);
lean_ctor_set_uint8(v___x_2318_, 1, v___x_2294_);
lean_ctor_set_uint8(v___x_2318_, 2, v___x_2294_);
lean_ctor_set_uint8(v___x_2318_, 3, v___x_2294_);
lean_inc(v_g_2234_);
v___x_2319_ = l_Lean_Elab_Tactic_Omega_omega(v___x_2317_, v_g_2234_, v___x_2318_, v___y_2268_, v___y_2269_, v___y_2270_, v___y_2271_);
if (lean_obj_tag(v___x_2319_) == 0)
{
lean_object* v___x_2321_; uint8_t v_isShared_2322_; uint8_t v_isSharedCheck_2327_; 
lean_del_object(v___x_2252_);
lean_dec(v_tail_2248_);
lean_dec(v_g_2234_);
v_isSharedCheck_2327_ = !lean_is_exclusive(v___x_2319_);
if (v_isSharedCheck_2327_ == 0)
{
lean_object* v_unused_2328_; 
v_unused_2328_ = lean_ctor_get(v___x_2319_, 0);
lean_dec(v_unused_2328_);
v___x_2321_ = v___x_2319_;
v_isShared_2322_ = v_isSharedCheck_2327_;
goto v_resetjp_2320_;
}
else
{
lean_dec(v___x_2319_);
v___x_2321_ = lean_box(0);
v_isShared_2322_ = v_isSharedCheck_2327_;
goto v_resetjp_2320_;
}
v_resetjp_2320_:
{
lean_object* v___x_2323_; lean_object* v___x_2325_; 
v___x_2323_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__3));
if (v_isShared_2322_ == 0)
{
lean_ctor_set(v___x_2321_, 0, v___x_2323_);
v___x_2325_ = v___x_2321_;
goto v_reusejp_2324_;
}
else
{
lean_object* v_reuseFailAlloc_2326_; 
v_reuseFailAlloc_2326_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2326_, 0, v___x_2323_);
v___x_2325_ = v_reuseFailAlloc_2326_;
goto v_reusejp_2324_;
}
v_reusejp_2324_:
{
return v___x_2325_;
}
}
}
else
{
lean_object* v_a_2329_; uint8_t v___x_2330_; 
v_a_2329_ = lean_ctor_get(v___x_2319_, 0);
lean_inc(v_a_2329_);
lean_dec_ref_known(v___x_2319_, 1);
v___x_2330_ = l_Lean_Exception_isInterrupt(v_a_2329_);
if (v___x_2330_ == 0)
{
uint8_t v___x_2331_; 
lean_inc(v_a_2329_);
v___x_2331_ = l_Lean_Exception_isRuntime(v_a_2329_);
v___y_2256_ = v_a_2329_;
v___y_2257_ = v___x_2331_;
goto v___jp_2255_;
}
else
{
v___y_2256_ = v_a_2329_;
v___y_2257_ = v___x_2330_;
goto v___jp_2255_;
}
}
}
else
{
lean_object* v_a_2332_; lean_object* v___x_2334_; uint8_t v_isShared_2335_; uint8_t v_isSharedCheck_2339_; 
lean_del_object(v___x_2252_);
lean_dec(v_tail_2248_);
lean_dec(v_g_2234_);
v_a_2332_ = lean_ctor_get(v___x_2311_, 0);
v_isSharedCheck_2339_ = !lean_is_exclusive(v___x_2311_);
if (v_isSharedCheck_2339_ == 0)
{
v___x_2334_ = v___x_2311_;
v_isShared_2335_ = v_isSharedCheck_2339_;
goto v_resetjp_2333_;
}
else
{
lean_inc(v_a_2332_);
lean_dec(v___x_2311_);
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
lean_dec(v_snd_2299_);
lean_dec(v_fst_2298_);
lean_dec_ref(v___y_2264_);
lean_del_object(v___x_2252_);
lean_dec(v_tail_2248_);
lean_dec(v_g_2234_);
v_a_2340_ = lean_ctor_get(v___x_2309_, 0);
v_isSharedCheck_2347_ = !lean_is_exclusive(v___x_2309_);
if (v_isSharedCheck_2347_ == 0)
{
v___x_2342_ = v___x_2309_;
v_isShared_2343_ = v_isSharedCheck_2347_;
goto v_resetjp_2341_;
}
else
{
lean_inc(v_a_2340_);
lean_dec(v___x_2309_);
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
}
else
{
lean_object* v_a_2350_; lean_object* v___x_2352_; uint8_t v_isShared_2353_; uint8_t v_isSharedCheck_2357_; 
lean_dec_ref(v___y_2264_);
lean_del_object(v___x_2252_);
lean_dec(v_tail_2248_);
lean_dec(v_g_2234_);
v_a_2350_ = lean_ctor_get(v___x_2296_, 0);
v_isSharedCheck_2357_ = !lean_is_exclusive(v___x_2296_);
if (v_isSharedCheck_2357_ == 0)
{
v___x_2352_ = v___x_2296_;
v_isShared_2353_ = v_isSharedCheck_2357_;
goto v_resetjp_2351_;
}
else
{
lean_inc(v_a_2350_);
lean_dec(v___x_2296_);
v___x_2352_ = lean_box(0);
v_isShared_2353_ = v_isSharedCheck_2357_;
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
lean_object* v_reuseFailAlloc_2356_; 
v_reuseFailAlloc_2356_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2356_, 0, v_a_2350_);
v___x_2355_ = v_reuseFailAlloc_2356_;
goto v_reusejp_2354_;
}
v_reusejp_2354_:
{
return v___x_2355_;
}
}
}
}
}
}
else
{
lean_object* v_a_2358_; lean_object* v___x_2360_; uint8_t v_isShared_2361_; uint8_t v_isSharedCheck_2365_; 
lean_dec_ref(v___y_2264_);
lean_dec(v_val_2262_);
lean_del_object(v___x_2252_);
lean_dec(v_tail_2248_);
lean_dec(v_key_2246_);
lean_dec(v_g_2234_);
v_a_2358_ = lean_ctor_get(v___x_2279_, 0);
v_isSharedCheck_2365_ = !lean_is_exclusive(v___x_2279_);
if (v_isSharedCheck_2365_ == 0)
{
v___x_2360_ = v___x_2279_;
v_isShared_2361_ = v_isSharedCheck_2365_;
goto v_resetjp_2359_;
}
else
{
lean_inc(v_a_2358_);
lean_dec(v___x_2279_);
v___x_2360_ = lean_box(0);
v_isShared_2361_ = v_isSharedCheck_2365_;
goto v_resetjp_2359_;
}
v_resetjp_2359_:
{
lean_object* v___x_2363_; 
if (v_isShared_2361_ == 0)
{
v___x_2363_ = v___x_2360_;
goto v_reusejp_2362_;
}
else
{
lean_object* v_reuseFailAlloc_2364_; 
v_reuseFailAlloc_2364_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2364_, 0, v_a_2358_);
v___x_2363_ = v_reuseFailAlloc_2364_;
goto v_reusejp_2362_;
}
v_reusejp_2362_:
{
return v___x_2363_;
}
}
}
}
else
{
lean_object* v___x_2366_; 
lean_dec_ref(v___y_2264_);
lean_dec(v_val_2262_);
lean_del_object(v___x_2252_);
lean_dec(v_key_2246_);
v___x_2366_ = lp_mathlib_Mathlib_Tactic_Order_findContradictionWithNle(v_a_2275_, v___y_2265_, v___y_2266_, v___y_2267_, v___y_2268_, v___y_2269_, v___y_2270_, v___y_2271_);
lean_dec_ref(v___y_2265_);
lean_dec(v_a_2275_);
if (lean_obj_tag(v___x_2366_) == 0)
{
lean_object* v_a_2367_; 
v_a_2367_ = lean_ctor_get(v___x_2366_, 0);
lean_inc(v_a_2367_);
lean_dec_ref_known(v___x_2366_, 1);
if (lean_obj_tag(v_a_2367_) == 1)
{
lean_object* v_val_2368_; lean_object* v___x_2369_; lean_object* v___x_2371_; uint8_t v_isShared_2372_; uint8_t v_isSharedCheck_2377_; 
lean_dec(v_tail_2248_);
v_val_2368_ = lean_ctor_get(v_a_2367_, 0);
lean_inc(v_val_2368_);
lean_dec_ref_known(v_a_2367_, 1);
v___x_2369_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1___redArg(v_g_2234_, v_val_2368_, v___y_2269_);
v_isSharedCheck_2377_ = !lean_is_exclusive(v___x_2369_);
if (v_isSharedCheck_2377_ == 0)
{
lean_object* v_unused_2378_; 
v_unused_2378_ = lean_ctor_get(v___x_2369_, 0);
lean_dec(v_unused_2378_);
v___x_2371_ = v___x_2369_;
v_isShared_2372_ = v_isSharedCheck_2377_;
goto v_resetjp_2370_;
}
else
{
lean_dec(v___x_2369_);
v___x_2371_ = lean_box(0);
v_isShared_2372_ = v_isSharedCheck_2377_;
goto v_resetjp_2370_;
}
v_resetjp_2370_:
{
lean_object* v___x_2373_; lean_object* v___x_2375_; 
v___x_2373_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__3));
if (v_isShared_2372_ == 0)
{
lean_ctor_set(v___x_2371_, 0, v___x_2373_);
v___x_2375_ = v___x_2371_;
goto v_reusejp_2374_;
}
else
{
lean_object* v_reuseFailAlloc_2376_; 
v_reuseFailAlloc_2376_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2376_, 0, v___x_2373_);
v___x_2375_ = v_reuseFailAlloc_2376_;
goto v_reusejp_2374_;
}
v_reusejp_2374_:
{
return v___x_2375_;
}
}
}
else
{
lean_dec(v_a_2367_);
v_a_2235_ = v_tail_2248_;
v_a_2236_ = v___x_2254_;
goto _start;
}
}
else
{
lean_object* v_a_2380_; lean_object* v___x_2382_; uint8_t v_isShared_2383_; uint8_t v_isSharedCheck_2387_; 
lean_dec(v_tail_2248_);
lean_dec(v_g_2234_);
v_a_2380_ = lean_ctor_get(v___x_2366_, 0);
v_isSharedCheck_2387_ = !lean_is_exclusive(v___x_2366_);
if (v_isSharedCheck_2387_ == 0)
{
v___x_2382_ = v___x_2366_;
v_isShared_2383_ = v_isSharedCheck_2387_;
goto v_resetjp_2381_;
}
else
{
lean_inc(v_a_2380_);
lean_dec(v___x_2366_);
v___x_2382_ = lean_box(0);
v_isShared_2383_ = v_isSharedCheck_2387_;
goto v_resetjp_2381_;
}
v_resetjp_2381_:
{
lean_object* v___x_2385_; 
if (v_isShared_2383_ == 0)
{
v___x_2385_ = v___x_2382_;
goto v_reusejp_2384_;
}
else
{
lean_object* v_reuseFailAlloc_2386_; 
v_reuseFailAlloc_2386_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2386_, 0, v_a_2380_);
v___x_2385_ = v_reuseFailAlloc_2386_;
goto v_reusejp_2384_;
}
v_reusejp_2384_:
{
return v___x_2385_;
}
}
}
}
}
else
{
lean_object* v_a_2388_; lean_object* v___x_2390_; uint8_t v_isShared_2391_; uint8_t v_isSharedCheck_2395_; 
lean_dec_ref(v___y_2265_);
lean_dec_ref(v___y_2264_);
lean_dec(v_val_2262_);
lean_del_object(v___x_2252_);
lean_dec(v_tail_2248_);
lean_dec(v_key_2246_);
lean_dec(v_g_2234_);
v_a_2388_ = lean_ctor_get(v___x_2274_, 0);
v_isSharedCheck_2395_ = !lean_is_exclusive(v___x_2274_);
if (v_isSharedCheck_2395_ == 0)
{
v___x_2390_ = v___x_2274_;
v_isShared_2391_ = v_isSharedCheck_2395_;
goto v_resetjp_2389_;
}
else
{
lean_inc(v_a_2388_);
lean_dec(v___x_2274_);
v___x_2390_ = lean_box(0);
v_isShared_2391_ = v_isSharedCheck_2395_;
goto v_resetjp_2389_;
}
v_resetjp_2389_:
{
lean_object* v___x_2393_; 
if (v_isShared_2391_ == 0)
{
v___x_2393_ = v___x_2390_;
goto v_reusejp_2392_;
}
else
{
lean_object* v_reuseFailAlloc_2394_; 
v_reuseFailAlloc_2394_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2394_, 0, v_a_2388_);
v___x_2393_ = v_reuseFailAlloc_2394_;
goto v_reusejp_2392_;
}
v_reusejp_2392_:
{
return v___x_2393_;
}
}
}
}
else
{
lean_object* v_a_2396_; lean_object* v___x_2398_; uint8_t v_isShared_2399_; uint8_t v_isSharedCheck_2403_; 
lean_dec_ref(v___y_2265_);
lean_dec_ref(v___y_2264_);
lean_dec(v_val_2262_);
lean_del_object(v___x_2252_);
lean_dec(v_tail_2248_);
lean_dec(v_key_2246_);
lean_dec(v_g_2234_);
v_a_2396_ = lean_ctor_get(v___x_2272_, 0);
v_isSharedCheck_2403_ = !lean_is_exclusive(v___x_2272_);
if (v_isSharedCheck_2403_ == 0)
{
v___x_2398_ = v___x_2272_;
v_isShared_2399_ = v_isSharedCheck_2403_;
goto v_resetjp_2397_;
}
else
{
lean_inc(v_a_2396_);
lean_dec(v___x_2272_);
v___x_2398_ = lean_box(0);
v_isShared_2399_ = v_isSharedCheck_2403_;
goto v_resetjp_2397_;
}
v_resetjp_2397_:
{
lean_object* v___x_2401_; 
if (v_isShared_2399_ == 0)
{
v___x_2401_ = v___x_2398_;
goto v_reusejp_2400_;
}
else
{
lean_object* v_reuseFailAlloc_2402_; 
v_reuseFailAlloc_2402_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2402_, 0, v_a_2396_);
v___x_2401_ = v_reuseFailAlloc_2402_;
goto v_reusejp_2400_;
}
v_reusejp_2400_:
{
return v___x_2401_;
}
}
}
}
v_resetjp_2408_:
{
lean_object* v___x_2411_; lean_object* v___y_2413_; lean_object* v___y_2414_; lean_object* v___y_2415_; lean_object* v___y_2416_; lean_object* v___y_2417_; lean_object* v___y_2418_; lean_object* v___y_2464_; lean_object* v___y_2465_; lean_object* v___y_2466_; lean_object* v___y_2467_; lean_object* v___y_2468_; lean_object* v_inheritedTraceOptions_2469_; lean_object* v___y_2470_; uint8_t v___x_2491_; 
v___x_2411_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__6));
v___x_2491_ = lean_unbox(v_a_2407_);
lean_dec(v_a_2407_);
if (v___x_2491_ == 0)
{
lean_del_object(v___x_2409_);
v___y_2464_ = v___y_2237_;
v___y_2465_ = v___y_2238_;
v___y_2466_ = v___y_2239_;
v___y_2467_ = v___y_2240_;
v___y_2468_ = v___y_2241_;
v_inheritedTraceOptions_2469_ = v_inheritedTraceOptions_2404_;
v___y_2470_ = v___y_2242_;
goto v___jp_2463_;
}
else
{
lean_object* v___x_2492_; 
lean_inc(v_key_2246_);
v___x_2492_ = l_Lean_Meta_ppExpr(v_key_2246_, v___y_2239_, v___y_2240_, v___y_2241_, v___y_2242_);
if (lean_obj_tag(v___x_2492_) == 0)
{
lean_object* v_a_2493_; lean_object* v___x_2494_; lean_object* v___x_2495_; lean_object* v___x_2496_; lean_object* v___x_2497_; lean_object* v___x_2498_; lean_object* v___y_2500_; uint8_t v___x_2517_; 
v_a_2493_ = lean_ctor_get(v___x_2492_, 0);
lean_inc(v_a_2493_);
lean_dec_ref_known(v___x_2492_, 1);
v___x_2494_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__13, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__13_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__13);
v___x_2495_ = l_Lean_MessageData_ofFormat(v_a_2493_);
v___x_2496_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2496_, 0, v___x_2494_);
lean_ctor_set(v___x_2496_, 1, v___x_2495_);
v___x_2497_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__15, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__15_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__15);
v___x_2498_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2498_, 0, v___x_2496_);
lean_ctor_set(v___x_2498_, 1, v___x_2497_);
v___x_2517_ = lean_unbox(v_val_2262_);
switch(v___x_2517_)
{
case 0:
{
lean_object* v___x_2518_; 
v___x_2518_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__18));
v___y_2500_ = v___x_2518_;
goto v___jp_2499_;
}
case 1:
{
lean_object* v___x_2519_; 
v___x_2519_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__19));
v___y_2500_ = v___x_2519_;
goto v___jp_2499_;
}
default: 
{
lean_object* v___x_2520_; 
v___x_2520_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__20));
v___y_2500_ = v___x_2520_;
goto v___jp_2499_;
}
}
v___jp_2499_:
{
lean_object* v___x_2502_; 
lean_inc_ref(v___y_2500_);
if (v_isShared_2410_ == 0)
{
lean_ctor_set_tag(v___x_2409_, 3);
lean_ctor_set(v___x_2409_, 0, v___y_2500_);
v___x_2502_ = v___x_2409_;
goto v_reusejp_2501_;
}
else
{
lean_object* v_reuseFailAlloc_2516_; 
v_reuseFailAlloc_2516_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2516_, 0, v___y_2500_);
v___x_2502_ = v_reuseFailAlloc_2516_;
goto v_reusejp_2501_;
}
v_reusejp_2501_:
{
lean_object* v___x_2503_; lean_object* v___x_2504_; lean_object* v___x_2505_; lean_object* v___x_2506_; lean_object* v___x_2507_; 
v___x_2503_ = l_Lean_MessageData_ofFormat(v___x_2502_);
v___x_2504_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2504_, 0, v___x_2498_);
lean_ctor_set(v___x_2504_, 1, v___x_2503_);
v___x_2505_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__17, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__17_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__17);
v___x_2506_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2506_, 0, v___x_2504_);
lean_ctor_set(v___x_2506_, 1, v___x_2505_);
v___x_2507_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___redArg(v___x_2405_, v___x_2506_, v___y_2239_, v___y_2240_, v___y_2241_, v___y_2242_);
if (lean_obj_tag(v___x_2507_) == 0)
{
lean_dec_ref_known(v___x_2507_, 1);
v___y_2464_ = v___y_2237_;
v___y_2465_ = v___y_2238_;
v___y_2466_ = v___y_2239_;
v___y_2467_ = v___y_2240_;
v___y_2468_ = v___y_2241_;
v_inheritedTraceOptions_2469_ = v_inheritedTraceOptions_2404_;
v___y_2470_ = v___y_2242_;
goto v___jp_2463_;
}
else
{
lean_object* v_a_2508_; lean_object* v___x_2510_; uint8_t v_isShared_2511_; uint8_t v_isSharedCheck_2515_; 
lean_dec(v_val_2262_);
lean_del_object(v___x_2252_);
lean_dec(v_tail_2248_);
lean_dec(v_value_2247_);
lean_dec(v_key_2246_);
lean_dec(v_g_2234_);
v_a_2508_ = lean_ctor_get(v___x_2507_, 0);
v_isSharedCheck_2515_ = !lean_is_exclusive(v___x_2507_);
if (v_isSharedCheck_2515_ == 0)
{
v___x_2510_ = v___x_2507_;
v_isShared_2511_ = v_isSharedCheck_2515_;
goto v_resetjp_2509_;
}
else
{
lean_inc(v_a_2508_);
lean_dec(v___x_2507_);
v___x_2510_ = lean_box(0);
v_isShared_2511_ = v_isSharedCheck_2515_;
goto v_resetjp_2509_;
}
v_resetjp_2509_:
{
lean_object* v___x_2513_; 
if (v_isShared_2511_ == 0)
{
v___x_2513_ = v___x_2510_;
goto v_reusejp_2512_;
}
else
{
lean_object* v_reuseFailAlloc_2514_; 
v_reuseFailAlloc_2514_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2514_, 0, v_a_2508_);
v___x_2513_ = v_reuseFailAlloc_2514_;
goto v_reusejp_2512_;
}
v_reusejp_2512_:
{
return v___x_2513_;
}
}
}
}
}
}
else
{
lean_object* v_a_2521_; lean_object* v___x_2523_; uint8_t v_isShared_2524_; uint8_t v_isSharedCheck_2528_; 
lean_del_object(v___x_2409_);
lean_dec(v_val_2262_);
lean_del_object(v___x_2252_);
lean_dec(v_tail_2248_);
lean_dec(v_value_2247_);
lean_dec(v_key_2246_);
lean_dec(v_g_2234_);
v_a_2521_ = lean_ctor_get(v___x_2492_, 0);
v_isSharedCheck_2528_ = !lean_is_exclusive(v___x_2492_);
if (v_isSharedCheck_2528_ == 0)
{
v___x_2523_ = v___x_2492_;
v_isShared_2524_ = v_isSharedCheck_2528_;
goto v_resetjp_2522_;
}
else
{
lean_inc(v_a_2521_);
lean_dec(v___x_2492_);
v___x_2523_ = lean_box(0);
v_isShared_2524_ = v_isSharedCheck_2528_;
goto v_resetjp_2522_;
}
v_resetjp_2522_:
{
lean_object* v___x_2526_; 
if (v_isShared_2524_ == 0)
{
v___x_2526_ = v___x_2523_;
goto v_reusejp_2525_;
}
else
{
lean_object* v_reuseFailAlloc_2527_; 
v_reuseFailAlloc_2527_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2527_, 0, v_a_2521_);
v___x_2526_ = v_reuseFailAlloc_2527_;
goto v_reusejp_2525_;
}
v_reusejp_2525_:
{
return v___x_2526_;
}
}
}
}
v___jp_2412_:
{
lean_object* v___x_2419_; 
v___x_2419_ = lp_mathlib_Mathlib_Tactic_Order_replaceBotTop(v_value_2247_, v___y_2413_, v___y_2414_, v___y_2415_, v___y_2416_, v___y_2417_, v___y_2418_);
lean_dec(v_value_2247_);
if (lean_obj_tag(v___x_2419_) == 0)
{
lean_object* v_a_2420_; uint8_t v___x_2421_; lean_object* v___x_2422_; 
v_a_2420_ = lean_ctor_get(v___x_2419_, 0);
lean_inc(v_a_2420_);
lean_dec_ref_known(v___x_2419_, 1);
v___x_2421_ = lean_unbox(v_val_2262_);
v___x_2422_ = lp_mathlib_Mathlib_Tactic_Order_preprocessFacts(v_a_2420_, v___x_2421_, v___y_2413_, v___y_2414_, v___y_2415_, v___y_2416_, v___y_2417_, v___y_2418_);
if (lean_obj_tag(v___x_2422_) == 0)
{
lean_object* v_options_2423_; uint8_t v_hasTrace_2424_; 
v_options_2423_ = lean_ctor_get(v___y_2417_, 2);
v_hasTrace_2424_ = lean_ctor_get_uint8(v_options_2423_, sizeof(void*)*1);
if (v_hasTrace_2424_ == 0)
{
lean_object* v_a_2425_; 
v_a_2425_ = lean_ctor_get(v___x_2422_, 0);
lean_inc(v_a_2425_);
lean_dec_ref_known(v___x_2422_, 1);
v___y_2264_ = v_a_2420_;
v___y_2265_ = v_a_2425_;
v___y_2266_ = v___y_2413_;
v___y_2267_ = v___y_2414_;
v___y_2268_ = v___y_2415_;
v___y_2269_ = v___y_2416_;
v___y_2270_ = v___y_2417_;
v___y_2271_ = v___y_2418_;
goto v___jp_2263_;
}
else
{
lean_object* v_a_2426_; lean_object* v_inheritedTraceOptions_2427_; lean_object* v___x_2428_; uint8_t v___x_2429_; 
v_a_2426_ = lean_ctor_get(v___x_2422_, 0);
lean_inc(v_a_2426_);
lean_dec_ref_known(v___x_2422_, 1);
v_inheritedTraceOptions_2427_ = lean_ctor_get(v___y_2417_, 13);
v___x_2428_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__7, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__7_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__7);
v___x_2429_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2427_, v_options_2423_, v___x_2428_);
if (v___x_2429_ == 0)
{
v___y_2264_ = v_a_2420_;
v___y_2265_ = v_a_2426_;
v___y_2266_ = v___y_2413_;
v___y_2267_ = v___y_2414_;
v___y_2268_ = v___y_2415_;
v___y_2269_ = v___y_2416_;
v___y_2270_ = v___y_2417_;
v___y_2271_ = v___y_2418_;
goto v___jp_2263_;
}
else
{
size_t v_sz_2430_; size_t v___x_2431_; lean_object* v___x_2432_; lean_object* v___x_2433_; lean_object* v___x_2434_; lean_object* v___x_2435_; lean_object* v___x_2436_; lean_object* v___x_2437_; lean_object* v___x_2438_; 
v_sz_2430_ = lean_array_size(v_a_2426_);
v___x_2431_ = ((size_t)0ULL);
lean_inc(v_a_2426_);
v___x_2432_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3(v_sz_2430_, v___x_2431_, v_a_2426_);
v___x_2433_ = lean_array_to_list(v___x_2432_);
v___x_2434_ = l_String_intercalate(v___x_2411_, v___x_2433_);
v___x_2435_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__9, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__9_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__9);
v___x_2436_ = l_Lean_stringToMessageData(v___x_2434_);
v___x_2437_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2437_, 0, v___x_2435_);
lean_ctor_set(v___x_2437_, 1, v___x_2436_);
v___x_2438_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___redArg(v___x_2405_, v___x_2437_, v___y_2415_, v___y_2416_, v___y_2417_, v___y_2418_);
if (lean_obj_tag(v___x_2438_) == 0)
{
lean_dec_ref_known(v___x_2438_, 1);
v___y_2264_ = v_a_2420_;
v___y_2265_ = v_a_2426_;
v___y_2266_ = v___y_2413_;
v___y_2267_ = v___y_2414_;
v___y_2268_ = v___y_2415_;
v___y_2269_ = v___y_2416_;
v___y_2270_ = v___y_2417_;
v___y_2271_ = v___y_2418_;
goto v___jp_2263_;
}
else
{
lean_object* v_a_2439_; lean_object* v___x_2441_; uint8_t v_isShared_2442_; uint8_t v_isSharedCheck_2446_; 
lean_dec(v_a_2426_);
lean_dec(v_a_2420_);
lean_dec(v_val_2262_);
lean_del_object(v___x_2252_);
lean_dec(v_tail_2248_);
lean_dec(v_key_2246_);
lean_dec(v_g_2234_);
v_a_2439_ = lean_ctor_get(v___x_2438_, 0);
v_isSharedCheck_2446_ = !lean_is_exclusive(v___x_2438_);
if (v_isSharedCheck_2446_ == 0)
{
v___x_2441_ = v___x_2438_;
v_isShared_2442_ = v_isSharedCheck_2446_;
goto v_resetjp_2440_;
}
else
{
lean_inc(v_a_2439_);
lean_dec(v___x_2438_);
v___x_2441_ = lean_box(0);
v_isShared_2442_ = v_isSharedCheck_2446_;
goto v_resetjp_2440_;
}
v_resetjp_2440_:
{
lean_object* v___x_2444_; 
if (v_isShared_2442_ == 0)
{
v___x_2444_ = v___x_2441_;
goto v_reusejp_2443_;
}
else
{
lean_object* v_reuseFailAlloc_2445_; 
v_reuseFailAlloc_2445_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2445_, 0, v_a_2439_);
v___x_2444_ = v_reuseFailAlloc_2445_;
goto v_reusejp_2443_;
}
v_reusejp_2443_:
{
return v___x_2444_;
}
}
}
}
}
}
else
{
lean_object* v_a_2447_; lean_object* v___x_2449_; uint8_t v_isShared_2450_; uint8_t v_isSharedCheck_2454_; 
lean_dec(v_a_2420_);
lean_dec(v_val_2262_);
lean_del_object(v___x_2252_);
lean_dec(v_tail_2248_);
lean_dec(v_key_2246_);
lean_dec(v_g_2234_);
v_a_2447_ = lean_ctor_get(v___x_2422_, 0);
v_isSharedCheck_2454_ = !lean_is_exclusive(v___x_2422_);
if (v_isSharedCheck_2454_ == 0)
{
v___x_2449_ = v___x_2422_;
v_isShared_2450_ = v_isSharedCheck_2454_;
goto v_resetjp_2448_;
}
else
{
lean_inc(v_a_2447_);
lean_dec(v___x_2422_);
v___x_2449_ = lean_box(0);
v_isShared_2450_ = v_isSharedCheck_2454_;
goto v_resetjp_2448_;
}
v_resetjp_2448_:
{
lean_object* v___x_2452_; 
if (v_isShared_2450_ == 0)
{
v___x_2452_ = v___x_2449_;
goto v_reusejp_2451_;
}
else
{
lean_object* v_reuseFailAlloc_2453_; 
v_reuseFailAlloc_2453_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2453_, 0, v_a_2447_);
v___x_2452_ = v_reuseFailAlloc_2453_;
goto v_reusejp_2451_;
}
v_reusejp_2451_:
{
return v___x_2452_;
}
}
}
}
else
{
lean_object* v_a_2455_; lean_object* v___x_2457_; uint8_t v_isShared_2458_; uint8_t v_isSharedCheck_2462_; 
lean_dec(v_val_2262_);
lean_del_object(v___x_2252_);
lean_dec(v_tail_2248_);
lean_dec(v_key_2246_);
lean_dec(v_g_2234_);
v_a_2455_ = lean_ctor_get(v___x_2419_, 0);
v_isSharedCheck_2462_ = !lean_is_exclusive(v___x_2419_);
if (v_isSharedCheck_2462_ == 0)
{
v___x_2457_ = v___x_2419_;
v_isShared_2458_ = v_isSharedCheck_2462_;
goto v_resetjp_2456_;
}
else
{
lean_inc(v_a_2455_);
lean_dec(v___x_2419_);
v___x_2457_ = lean_box(0);
v_isShared_2458_ = v_isSharedCheck_2462_;
goto v_resetjp_2456_;
}
v_resetjp_2456_:
{
lean_object* v___x_2460_; 
if (v_isShared_2458_ == 0)
{
v___x_2460_ = v___x_2457_;
goto v_reusejp_2459_;
}
else
{
lean_object* v_reuseFailAlloc_2461_; 
v_reuseFailAlloc_2461_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2461_, 0, v_a_2455_);
v___x_2460_ = v_reuseFailAlloc_2461_;
goto v_reusejp_2459_;
}
v_reusejp_2459_:
{
return v___x_2460_;
}
}
}
}
v___jp_2463_:
{
lean_object* v___x_2471_; lean_object* v_a_2472_; uint8_t v___x_2473_; 
v___x_2471_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___lam__0(v___x_2405_, v_inheritedTraceOptions_2469_, v___y_2464_, v___y_2465_, v___y_2466_, v___y_2467_, v___y_2468_, v___y_2470_);
v_a_2472_ = lean_ctor_get(v___x_2471_, 0);
lean_inc(v_a_2472_);
lean_dec_ref(v___x_2471_);
v___x_2473_ = lean_unbox(v_a_2472_);
lean_dec(v_a_2472_);
if (v___x_2473_ == 0)
{
v___y_2413_ = v___y_2464_;
v___y_2414_ = v___y_2465_;
v___y_2415_ = v___y_2466_;
v___y_2416_ = v___y_2467_;
v___y_2417_ = v___y_2468_;
v___y_2418_ = v___y_2470_;
goto v___jp_2412_;
}
else
{
size_t v_sz_2474_; size_t v___x_2475_; lean_object* v___x_2476_; lean_object* v___x_2477_; lean_object* v___x_2478_; lean_object* v___x_2479_; lean_object* v___x_2480_; lean_object* v___x_2481_; lean_object* v___x_2482_; 
v_sz_2474_ = lean_array_size(v_value_2247_);
v___x_2475_ = ((size_t)0ULL);
lean_inc(v_value_2247_);
v___x_2476_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__3(v_sz_2474_, v___x_2475_, v_value_2247_);
v___x_2477_ = lean_array_to_list(v___x_2476_);
v___x_2478_ = l_String_intercalate(v___x_2411_, v___x_2477_);
v___x_2479_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__11, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__11_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__11);
v___x_2480_ = l_Lean_stringToMessageData(v___x_2478_);
v___x_2481_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2481_, 0, v___x_2479_);
lean_ctor_set(v___x_2481_, 1, v___x_2480_);
v___x_2482_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___redArg(v___x_2405_, v___x_2481_, v___y_2466_, v___y_2467_, v___y_2468_, v___y_2470_);
if (lean_obj_tag(v___x_2482_) == 0)
{
lean_dec_ref_known(v___x_2482_, 1);
v___y_2413_ = v___y_2464_;
v___y_2414_ = v___y_2465_;
v___y_2415_ = v___y_2466_;
v___y_2416_ = v___y_2467_;
v___y_2417_ = v___y_2468_;
v___y_2418_ = v___y_2470_;
goto v___jp_2412_;
}
else
{
lean_object* v_a_2483_; lean_object* v___x_2485_; uint8_t v_isShared_2486_; uint8_t v_isSharedCheck_2490_; 
lean_dec(v_val_2262_);
lean_del_object(v___x_2252_);
lean_dec(v_tail_2248_);
lean_dec(v_value_2247_);
lean_dec(v_key_2246_);
lean_dec(v_g_2234_);
v_a_2483_ = lean_ctor_get(v___x_2482_, 0);
v_isSharedCheck_2490_ = !lean_is_exclusive(v___x_2482_);
if (v_isSharedCheck_2490_ == 0)
{
v___x_2485_ = v___x_2482_;
v_isShared_2486_ = v_isSharedCheck_2490_;
goto v_resetjp_2484_;
}
else
{
lean_inc(v_a_2483_);
lean_dec(v___x_2482_);
v___x_2485_ = lean_box(0);
v_isShared_2486_ = v_isSharedCheck_2490_;
goto v_resetjp_2484_;
}
v_resetjp_2484_:
{
lean_object* v___x_2488_; 
if (v_isShared_2486_ == 0)
{
v___x_2488_ = v___x_2485_;
goto v_reusejp_2487_;
}
else
{
lean_object* v_reuseFailAlloc_2489_; 
v_reuseFailAlloc_2489_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2489_, 0, v_a_2483_);
v___x_2488_ = v_reuseFailAlloc_2489_;
goto v_reusejp_2487_;
}
v_reusejp_2487_:
{
return v___x_2488_;
}
}
}
}
}
}
}
else
{
lean_del_object(v___x_2252_);
lean_dec(v_a_2250_);
lean_dec(v_value_2247_);
lean_dec(v_key_2246_);
v_a_2235_ = v_tail_2248_;
v_a_2236_ = v___x_2254_;
goto _start;
}
v___jp_2255_:
{
if (v___y_2257_ == 0)
{
lean_dec_ref(v___y_2256_);
lean_del_object(v___x_2252_);
v_a_2235_ = v_tail_2248_;
v_a_2236_ = v___x_2254_;
goto _start;
}
else
{
lean_object* v___x_2260_; 
lean_dec(v_tail_2248_);
lean_dec(v_g_2234_);
if (v_isShared_2253_ == 0)
{
lean_ctor_set_tag(v___x_2252_, 1);
lean_ctor_set(v___x_2252_, 0, v___y_2256_);
v___x_2260_ = v___x_2252_;
goto v_reusejp_2259_;
}
else
{
lean_object* v_reuseFailAlloc_2261_; 
v_reuseFailAlloc_2261_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2261_, 0, v___y_2256_);
v___x_2260_ = v_reuseFailAlloc_2261_;
goto v_reusejp_2259_;
}
v_reusejp_2259_:
{
return v___x_2260_;
}
}
}
}
}
else
{
lean_object* v_a_2532_; lean_object* v___x_2534_; uint8_t v_isShared_2535_; uint8_t v_isSharedCheck_2539_; 
lean_dec(v_tail_2248_);
lean_dec(v_value_2247_);
lean_dec(v_key_2246_);
lean_dec(v_g_2234_);
v_a_2532_ = lean_ctor_get(v___x_2249_, 0);
v_isSharedCheck_2539_ = !lean_is_exclusive(v___x_2249_);
if (v_isSharedCheck_2539_ == 0)
{
v___x_2534_ = v___x_2249_;
v_isShared_2535_ = v_isSharedCheck_2539_;
goto v_resetjp_2533_;
}
else
{
lean_inc(v_a_2532_);
lean_dec(v___x_2249_);
v___x_2534_ = lean_box(0);
v_isShared_2535_ = v_isSharedCheck_2539_;
goto v_resetjp_2533_;
}
v_resetjp_2533_:
{
lean_object* v___x_2537_; 
if (v_isShared_2535_ == 0)
{
v___x_2537_ = v___x_2534_;
goto v_reusejp_2536_;
}
else
{
lean_object* v_reuseFailAlloc_2538_; 
v_reuseFailAlloc_2538_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2538_, 0, v_a_2532_);
v___x_2537_ = v_reuseFailAlloc_2538_;
goto v_reusejp_2536_;
}
v_reusejp_2536_:
{
return v___x_2537_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___boxed(lean_object* v_g_2540_, lean_object* v_a_2541_, lean_object* v_a_2542_, lean_object* v___y_2543_, lean_object* v___y_2544_, lean_object* v___y_2545_, lean_object* v___y_2546_, lean_object* v___y_2547_, lean_object* v___y_2548_, lean_object* v___y_2549_){
_start:
{
lean_object* v_res_2550_; 
v_res_2550_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5(v_g_2540_, v_a_2541_, v_a_2542_, v___y_2543_, v___y_2544_, v___y_2545_, v___y_2546_, v___y_2547_, v___y_2548_);
lean_dec(v___y_2548_);
lean_dec_ref(v___y_2547_);
lean_dec(v___y_2546_);
lean_dec_ref(v___y_2545_);
lean_dec(v___y_2544_);
lean_dec_ref(v___y_2543_);
return v_res_2550_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_orderCoreImp_spec__6(lean_object* v_g_2551_, lean_object* v_as_2552_, size_t v_sz_2553_, size_t v_i_2554_, lean_object* v_b_2555_, lean_object* v___y_2556_, lean_object* v___y_2557_, lean_object* v___y_2558_, lean_object* v___y_2559_, lean_object* v___y_2560_, lean_object* v___y_2561_){
_start:
{
uint8_t v___x_2563_; 
v___x_2563_ = lean_usize_dec_lt(v_i_2554_, v_sz_2553_);
if (v___x_2563_ == 0)
{
lean_object* v___x_2564_; 
lean_dec(v_g_2551_);
v___x_2564_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2564_, 0, v_b_2555_);
return v___x_2564_;
}
else
{
lean_object* v_a_2565_; lean_object* v___x_2566_; 
v_a_2565_ = lean_array_uget_borrowed(v_as_2552_, v_i_2554_);
lean_inc(v_a_2565_);
lean_inc(v_g_2551_);
v___x_2566_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5(v_g_2551_, v_a_2565_, v_b_2555_, v___y_2556_, v___y_2557_, v___y_2558_, v___y_2559_, v___y_2560_, v___y_2561_);
if (lean_obj_tag(v___x_2566_) == 0)
{
lean_object* v_a_2567_; lean_object* v___x_2569_; uint8_t v_isShared_2570_; uint8_t v_isSharedCheck_2579_; 
v_a_2567_ = lean_ctor_get(v___x_2566_, 0);
v_isSharedCheck_2579_ = !lean_is_exclusive(v___x_2566_);
if (v_isSharedCheck_2579_ == 0)
{
v___x_2569_ = v___x_2566_;
v_isShared_2570_ = v_isSharedCheck_2579_;
goto v_resetjp_2568_;
}
else
{
lean_inc(v_a_2567_);
lean_dec(v___x_2566_);
v___x_2569_ = lean_box(0);
v_isShared_2570_ = v_isSharedCheck_2579_;
goto v_resetjp_2568_;
}
v_resetjp_2568_:
{
if (lean_obj_tag(v_a_2567_) == 0)
{
lean_object* v_a_2571_; lean_object* v___x_2573_; 
lean_dec(v_g_2551_);
v_a_2571_ = lean_ctor_get(v_a_2567_, 0);
lean_inc(v_a_2571_);
lean_dec_ref_known(v_a_2567_, 1);
if (v_isShared_2570_ == 0)
{
lean_ctor_set(v___x_2569_, 0, v_a_2571_);
v___x_2573_ = v___x_2569_;
goto v_reusejp_2572_;
}
else
{
lean_object* v_reuseFailAlloc_2574_; 
v_reuseFailAlloc_2574_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2574_, 0, v_a_2571_);
v___x_2573_ = v_reuseFailAlloc_2574_;
goto v_reusejp_2572_;
}
v_reusejp_2572_:
{
return v___x_2573_;
}
}
else
{
lean_object* v_a_2575_; size_t v___x_2576_; size_t v___x_2577_; 
lean_del_object(v___x_2569_);
v_a_2575_ = lean_ctor_get(v_a_2567_, 0);
lean_inc(v_a_2575_);
lean_dec_ref_known(v_a_2567_, 1);
v___x_2576_ = ((size_t)1ULL);
v___x_2577_ = lean_usize_add(v_i_2554_, v___x_2576_);
v_i_2554_ = v___x_2577_;
v_b_2555_ = v_a_2575_;
goto _start;
}
}
}
else
{
lean_object* v_a_2580_; lean_object* v___x_2582_; uint8_t v_isShared_2583_; uint8_t v_isSharedCheck_2587_; 
lean_dec(v_g_2551_);
v_a_2580_ = lean_ctor_get(v___x_2566_, 0);
v_isSharedCheck_2587_ = !lean_is_exclusive(v___x_2566_);
if (v_isSharedCheck_2587_ == 0)
{
v___x_2582_ = v___x_2566_;
v_isShared_2583_ = v_isSharedCheck_2587_;
goto v_resetjp_2581_;
}
else
{
lean_inc(v_a_2580_);
lean_dec(v___x_2566_);
v___x_2582_ = lean_box(0);
v_isShared_2583_ = v_isSharedCheck_2587_;
goto v_resetjp_2581_;
}
v_resetjp_2581_:
{
lean_object* v___x_2585_; 
if (v_isShared_2583_ == 0)
{
v___x_2585_ = v___x_2582_;
goto v_reusejp_2584_;
}
else
{
lean_object* v_reuseFailAlloc_2586_; 
v_reuseFailAlloc_2586_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2586_, 0, v_a_2580_);
v___x_2585_ = v_reuseFailAlloc_2586_;
goto v_reusejp_2584_;
}
v_reusejp_2584_:
{
return v___x_2585_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_orderCoreImp_spec__6___boxed(lean_object* v_g_2588_, lean_object* v_as_2589_, lean_object* v_sz_2590_, lean_object* v_i_2591_, lean_object* v_b_2592_, lean_object* v___y_2593_, lean_object* v___y_2594_, lean_object* v___y_2595_, lean_object* v___y_2596_, lean_object* v___y_2597_, lean_object* v___y_2598_, lean_object* v___y_2599_){
_start:
{
size_t v_sz_boxed_2600_; size_t v_i_boxed_2601_; lean_object* v_res_2602_; 
v_sz_boxed_2600_ = lean_unbox_usize(v_sz_2590_);
lean_dec(v_sz_2590_);
v_i_boxed_2601_ = lean_unbox_usize(v_i_2591_);
lean_dec(v_i_2591_);
v_res_2602_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_orderCoreImp_spec__6(v_g_2588_, v_as_2589_, v_sz_boxed_2600_, v_i_boxed_2601_, v_b_2592_, v___y_2593_, v___y_2594_, v___y_2595_, v___y_2596_, v___y_2597_, v___y_2598_);
lean_dec(v___y_2598_);
lean_dec_ref(v___y_2597_);
lean_dec(v___y_2596_);
lean_dec_ref(v___y_2595_);
lean_dec(v___y_2594_);
lean_dec_ref(v___y_2593_);
lean_dec_ref(v_as_2589_);
return v_res_2602_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__2(void){
_start:
{
lean_object* v___x_2606_; lean_object* v___x_2607_; 
v___x_2606_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__1));
v___x_2607_ = l_Lean_MessageData_ofFormat(v___x_2606_);
return v___x_2607_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__5(void){
_start:
{
lean_object* v___x_2611_; lean_object* v___x_2612_; 
v___x_2611_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__4));
v___x_2612_ = l_Lean_MessageData_ofFormat(v___x_2611_);
return v___x_2612_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__6(void){
_start:
{
lean_object* v___x_2613_; lean_object* v___x_2614_; lean_object* v___x_2615_; 
v___x_2613_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__5, &lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__5);
v___x_2614_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__2, &lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__2);
v___x_2615_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2615_, 0, v___x_2614_);
lean_ctor_set(v___x_2615_, 1, v___x_2613_);
return v___x_2615_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__9(void){
_start:
{
lean_object* v___x_2619_; lean_object* v___x_2620_; 
v___x_2619_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__8));
v___x_2620_ = l_Lean_MessageData_ofFormat(v___x_2619_);
return v___x_2620_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__10(void){
_start:
{
lean_object* v___x_2621_; lean_object* v___x_2622_; lean_object* v___x_2623_; 
v___x_2621_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__9, &lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__9);
v___x_2622_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__6, &lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__6);
v___x_2623_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2623_, 0, v___x_2622_);
lean_ctor_set(v___x_2623_, 1, v___x_2621_);
return v___x_2623_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__12(void){
_start:
{
lean_object* v___x_2625_; lean_object* v___x_2626_; 
v___x_2625_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__11));
v___x_2626_ = l_Lean_stringToMessageData(v___x_2625_);
return v___x_2626_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0(uint8_t v_only_x3f_2627_, lean_object* v_hyps_2628_, lean_object* v_negGoal_2629_, lean_object* v_g_2630_, lean_object* v___y_2631_, lean_object* v___y_2632_, lean_object* v___y_2633_, lean_object* v___y_2634_, lean_object* v___y_2635_, lean_object* v___y_2636_){
_start:
{
lean_object* v___x_2638_; 
v___x_2638_ = lp_mathlib_Mathlib_Tactic_Order_collectFacts(v_only_x3f_2627_, v_hyps_2628_, v_negGoal_2629_, v___y_2631_, v___y_2632_, v___y_2633_, v___y_2634_, v___y_2635_, v___y_2636_);
if (lean_obj_tag(v___x_2638_) == 0)
{
lean_object* v_a_2639_; lean_object* v___x_2640_; size_t v_sz_2641_; size_t v___x_2642_; lean_object* v___y_2644_; lean_object* v___y_2645_; lean_object* v___y_2646_; lean_object* v___y_2647_; lean_object* v___y_2648_; lean_object* v___y_2649_; lean_object* v___x_2674_; 
v_a_2639_ = lean_ctor_get(v___x_2638_, 0);
lean_inc(v_a_2639_);
lean_dec_ref_known(v___x_2638_, 1);
v___x_2640_ = lean_st_ref_get(v___y_2632_);
v_sz_2641_ = lean_array_size(v___x_2640_);
v___x_2642_ = ((size_t)0ULL);
v___x_2674_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0___redArg(v_sz_2641_, v___x_2642_, v___x_2640_, v___y_2633_, v___y_2634_, v___y_2635_, v___y_2636_);
if (lean_obj_tag(v___x_2674_) == 0)
{
lean_object* v_options_2675_; uint8_t v_hasTrace_2676_; 
v_options_2675_ = lean_ctor_get(v___y_2635_, 2);
v_hasTrace_2676_ = lean_ctor_get_uint8(v_options_2675_, sizeof(void*)*1);
if (v_hasTrace_2676_ == 0)
{
lean_dec_ref_known(v___x_2674_, 1);
v___y_2644_ = v___y_2631_;
v___y_2645_ = v___y_2632_;
v___y_2646_ = v___y_2633_;
v___y_2647_ = v___y_2634_;
v___y_2648_ = v___y_2635_;
v___y_2649_ = v___y_2636_;
goto v___jp_2643_;
}
else
{
lean_object* v_a_2677_; lean_object* v_inheritedTraceOptions_2678_; lean_object* v___x_2679_; lean_object* v___x_2680_; uint8_t v___x_2681_; 
v_a_2677_ = lean_ctor_get(v___x_2674_, 0);
lean_inc(v_a_2677_);
lean_dec_ref_known(v___x_2674_, 1);
v_inheritedTraceOptions_2678_ = lean_ctor_get(v___y_2635_, 13);
v___x_2679_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__1_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_));
v___x_2680_ = lean_obj_once(&lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__7, &lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__7_once, _init_lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__7);
v___x_2681_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2678_, v_options_2675_, v___x_2680_);
if (v___x_2681_ == 0)
{
lean_dec(v_a_2677_);
v___y_2644_ = v___y_2631_;
v___y_2645_ = v___y_2632_;
v___y_2646_ = v___y_2633_;
v___y_2647_ = v___y_2634_;
v___y_2648_ = v___y_2635_;
v___y_2649_ = v___y_2636_;
goto v___jp_2643_;
}
else
{
lean_object* v___x_2682_; lean_object* v___x_2683_; lean_object* v___x_2684_; lean_object* v___x_2685_; lean_object* v___x_2686_; lean_object* v___x_2687_; lean_object* v___x_2688_; 
v___x_2682_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__6));
v___x_2683_ = lean_array_to_list(v_a_2677_);
v___x_2684_ = l_String_intercalate(v___x_2682_, v___x_2683_);
v___x_2685_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__12, &lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__12);
v___x_2686_ = l_Lean_stringToMessageData(v___x_2684_);
v___x_2687_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2687_, 0, v___x_2685_);
lean_ctor_set(v___x_2687_, 1, v___x_2686_);
v___x_2688_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___redArg(v___x_2679_, v___x_2687_, v___y_2633_, v___y_2634_, v___y_2635_, v___y_2636_);
if (lean_obj_tag(v___x_2688_) == 0)
{
lean_dec_ref_known(v___x_2688_, 1);
v___y_2644_ = v___y_2631_;
v___y_2645_ = v___y_2632_;
v___y_2646_ = v___y_2633_;
v___y_2647_ = v___y_2634_;
v___y_2648_ = v___y_2635_;
v___y_2649_ = v___y_2636_;
goto v___jp_2643_;
}
else
{
lean_dec(v_a_2639_);
lean_dec(v_g_2630_);
return v___x_2688_;
}
}
}
}
else
{
lean_object* v_a_2689_; lean_object* v___x_2691_; uint8_t v_isShared_2692_; uint8_t v_isSharedCheck_2696_; 
lean_dec(v_a_2639_);
lean_dec(v_g_2630_);
v_a_2689_ = lean_ctor_get(v___x_2674_, 0);
v_isSharedCheck_2696_ = !lean_is_exclusive(v___x_2674_);
if (v_isSharedCheck_2696_ == 0)
{
v___x_2691_ = v___x_2674_;
v_isShared_2692_ = v_isSharedCheck_2696_;
goto v_resetjp_2690_;
}
else
{
lean_inc(v_a_2689_);
lean_dec(v___x_2674_);
v___x_2691_ = lean_box(0);
v_isShared_2692_ = v_isSharedCheck_2696_;
goto v_resetjp_2690_;
}
v_resetjp_2690_:
{
lean_object* v___x_2694_; 
if (v_isShared_2692_ == 0)
{
v___x_2694_ = v___x_2691_;
goto v_reusejp_2693_;
}
else
{
lean_object* v_reuseFailAlloc_2695_; 
v_reuseFailAlloc_2695_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2695_, 0, v_a_2689_);
v___x_2694_ = v_reuseFailAlloc_2695_;
goto v_reusejp_2693_;
}
v_reusejp_2693_:
{
return v___x_2694_;
}
}
}
v___jp_2643_:
{
lean_object* v_buckets_2650_; lean_object* v___x_2651_; size_t v_sz_2652_; lean_object* v___x_2653_; 
v_buckets_2650_ = lean_ctor_get(v_a_2639_, 1);
lean_inc_ref(v_buckets_2650_);
lean_dec(v_a_2639_);
v___x_2651_ = ((lean_object*)(lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_orderCoreImp_spec__5___closed__0));
v_sz_2652_ = lean_array_size(v_buckets_2650_);
v___x_2653_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_orderCoreImp_spec__6(v_g_2630_, v_buckets_2650_, v_sz_2652_, v___x_2642_, v___x_2651_, v___y_2644_, v___y_2645_, v___y_2646_, v___y_2647_, v___y_2648_, v___y_2649_);
lean_dec_ref(v_buckets_2650_);
if (lean_obj_tag(v___x_2653_) == 0)
{
lean_object* v_a_2654_; lean_object* v___x_2656_; uint8_t v_isShared_2657_; uint8_t v_isSharedCheck_2665_; 
v_a_2654_ = lean_ctor_get(v___x_2653_, 0);
v_isSharedCheck_2665_ = !lean_is_exclusive(v___x_2653_);
if (v_isSharedCheck_2665_ == 0)
{
v___x_2656_ = v___x_2653_;
v_isShared_2657_ = v_isSharedCheck_2665_;
goto v_resetjp_2655_;
}
else
{
lean_inc(v_a_2654_);
lean_dec(v___x_2653_);
v___x_2656_ = lean_box(0);
v_isShared_2657_ = v_isSharedCheck_2665_;
goto v_resetjp_2655_;
}
v_resetjp_2655_:
{
lean_object* v_fst_2658_; 
v_fst_2658_ = lean_ctor_get(v_a_2654_, 0);
lean_inc(v_fst_2658_);
lean_dec(v_a_2654_);
if (lean_obj_tag(v_fst_2658_) == 0)
{
lean_object* v___x_2659_; lean_object* v___x_2660_; 
lean_del_object(v___x_2656_);
v___x_2659_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__10, &lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___closed__10);
v___x_2660_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Order_orderCoreImp_spec__7___redArg(v___x_2659_, v___y_2646_, v___y_2647_, v___y_2648_, v___y_2649_);
return v___x_2660_;
}
else
{
lean_object* v_val_2661_; lean_object* v___x_2663_; 
v_val_2661_ = lean_ctor_get(v_fst_2658_, 0);
lean_inc(v_val_2661_);
lean_dec_ref_known(v_fst_2658_, 1);
if (v_isShared_2657_ == 0)
{
lean_ctor_set(v___x_2656_, 0, v_val_2661_);
v___x_2663_ = v___x_2656_;
goto v_reusejp_2662_;
}
else
{
lean_object* v_reuseFailAlloc_2664_; 
v_reuseFailAlloc_2664_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2664_, 0, v_val_2661_);
v___x_2663_ = v_reuseFailAlloc_2664_;
goto v_reusejp_2662_;
}
v_reusejp_2662_:
{
return v___x_2663_;
}
}
}
}
else
{
lean_object* v_a_2666_; lean_object* v___x_2668_; uint8_t v_isShared_2669_; uint8_t v_isSharedCheck_2673_; 
v_a_2666_ = lean_ctor_get(v___x_2653_, 0);
v_isSharedCheck_2673_ = !lean_is_exclusive(v___x_2653_);
if (v_isSharedCheck_2673_ == 0)
{
v___x_2668_ = v___x_2653_;
v_isShared_2669_ = v_isSharedCheck_2673_;
goto v_resetjp_2667_;
}
else
{
lean_inc(v_a_2666_);
lean_dec(v___x_2653_);
v___x_2668_ = lean_box(0);
v_isShared_2669_ = v_isSharedCheck_2673_;
goto v_resetjp_2667_;
}
v_resetjp_2667_:
{
lean_object* v___x_2671_; 
if (v_isShared_2669_ == 0)
{
v___x_2671_ = v___x_2668_;
goto v_reusejp_2670_;
}
else
{
lean_object* v_reuseFailAlloc_2672_; 
v_reuseFailAlloc_2672_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2672_, 0, v_a_2666_);
v___x_2671_ = v_reuseFailAlloc_2672_;
goto v_reusejp_2670_;
}
v_reusejp_2670_:
{
return v___x_2671_;
}
}
}
}
}
else
{
lean_object* v_a_2697_; lean_object* v___x_2699_; uint8_t v_isShared_2700_; uint8_t v_isSharedCheck_2704_; 
lean_dec(v_g_2630_);
v_a_2697_ = lean_ctor_get(v___x_2638_, 0);
v_isSharedCheck_2704_ = !lean_is_exclusive(v___x_2638_);
if (v_isSharedCheck_2704_ == 0)
{
v___x_2699_ = v___x_2638_;
v_isShared_2700_ = v_isSharedCheck_2704_;
goto v_resetjp_2698_;
}
else
{
lean_inc(v_a_2697_);
lean_dec(v___x_2638_);
v___x_2699_ = lean_box(0);
v_isShared_2700_ = v_isSharedCheck_2704_;
goto v_resetjp_2698_;
}
v_resetjp_2698_:
{
lean_object* v___x_2702_; 
if (v_isShared_2700_ == 0)
{
v___x_2702_ = v___x_2699_;
goto v_reusejp_2701_;
}
else
{
lean_object* v_reuseFailAlloc_2703_; 
v_reuseFailAlloc_2703_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2703_, 0, v_a_2697_);
v___x_2702_ = v_reuseFailAlloc_2703_;
goto v_reusejp_2701_;
}
v_reusejp_2701_:
{
return v___x_2702_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___boxed(lean_object* v_only_x3f_2705_, lean_object* v_hyps_2706_, lean_object* v_negGoal_2707_, lean_object* v_g_2708_, lean_object* v___y_2709_, lean_object* v___y_2710_, lean_object* v___y_2711_, lean_object* v___y_2712_, lean_object* v___y_2713_, lean_object* v___y_2714_, lean_object* v___y_2715_){
_start:
{
uint8_t v_only_x3f_boxed_2716_; lean_object* v_res_2717_; 
v_only_x3f_boxed_2716_ = lean_unbox(v_only_x3f_2705_);
v_res_2717_ = lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0(v_only_x3f_boxed_2716_, v_hyps_2706_, v_negGoal_2707_, v_g_2708_, v___y_2709_, v___y_2710_, v___y_2711_, v___y_2712_, v___y_2713_, v___y_2714_);
lean_dec(v___y_2714_);
lean_dec_ref(v___y_2713_);
lean_dec(v___y_2712_);
lean_dec_ref(v___y_2711_);
lean_dec(v___y_2710_);
lean_dec_ref(v___y_2709_);
lean_dec_ref(v_hyps_2706_);
return v_res_2717_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCoreImp(uint8_t v_only_x3f_2718_, lean_object* v_hyps_2719_, lean_object* v_negGoal_2720_, lean_object* v_g_2721_, lean_object* v_a_2722_, lean_object* v_a_2723_, lean_object* v_a_2724_, lean_object* v_a_2725_, lean_object* v_a_2726_, lean_object* v_a_2727_){
_start:
{
lean_object* v___x_2729_; lean_object* v___f_2730_; lean_object* v___x_2731_; 
v___x_2729_ = lean_box(v_only_x3f_2718_);
lean_inc(v_g_2721_);
v___f_2730_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___lam__0___boxed), 11, 4);
lean_closure_set(v___f_2730_, 0, v___x_2729_);
lean_closure_set(v___f_2730_, 1, v_hyps_2719_);
lean_closure_set(v___f_2730_, 2, v_negGoal_2720_);
lean_closure_set(v___f_2730_, 3, v_g_2721_);
v___x_2731_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Order_orderCoreImp_spec__8___redArg(v_g_2721_, v___f_2730_, v_a_2722_, v_a_2723_, v_a_2724_, v_a_2725_, v_a_2726_, v_a_2727_);
return v___x_2731_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___boxed(lean_object* v_only_x3f_2732_, lean_object* v_hyps_2733_, lean_object* v_negGoal_2734_, lean_object* v_g_2735_, lean_object* v_a_2736_, lean_object* v_a_2737_, lean_object* v_a_2738_, lean_object* v_a_2739_, lean_object* v_a_2740_, lean_object* v_a_2741_, lean_object* v_a_2742_){
_start:
{
uint8_t v_only_x3f_boxed_2743_; lean_object* v_res_2744_; 
v_only_x3f_boxed_2743_ = lean_unbox(v_only_x3f_2732_);
v_res_2744_ = lp_mathlib_Mathlib_Tactic_Order_orderCoreImp(v_only_x3f_boxed_2743_, v_hyps_2733_, v_negGoal_2734_, v_g_2735_, v_a_2736_, v_a_2737_, v_a_2738_, v_a_2739_, v_a_2740_, v_a_2741_);
lean_dec(v_a_2741_);
lean_dec_ref(v_a_2740_);
lean_dec(v_a_2739_);
lean_dec_ref(v_a_2738_);
lean_dec(v_a_2737_);
lean_dec_ref(v_a_2736_);
return v_res_2744_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0(lean_object* v_as_2745_, size_t v_sz_2746_, size_t v_i_2747_, lean_object* v_bs_2748_, lean_object* v___y_2749_, lean_object* v___y_2750_, lean_object* v___y_2751_, lean_object* v___y_2752_, lean_object* v___y_2753_, lean_object* v___y_2754_){
_start:
{
lean_object* v___x_2756_; 
v___x_2756_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0___redArg(v_sz_2746_, v_i_2747_, v_bs_2748_, v___y_2751_, v___y_2752_, v___y_2753_, v___y_2754_);
return v___x_2756_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0___boxed(lean_object* v_as_2757_, lean_object* v_sz_2758_, lean_object* v_i_2759_, lean_object* v_bs_2760_, lean_object* v___y_2761_, lean_object* v___y_2762_, lean_object* v___y_2763_, lean_object* v___y_2764_, lean_object* v___y_2765_, lean_object* v___y_2766_, lean_object* v___y_2767_){
_start:
{
size_t v_sz_boxed_2768_; size_t v_i_boxed_2769_; lean_object* v_res_2770_; 
v_sz_boxed_2768_ = lean_unbox_usize(v_sz_2758_);
lean_dec(v_sz_2758_);
v_i_boxed_2769_ = lean_unbox_usize(v_i_2759_);
lean_dec(v_i_2759_);
v_res_2770_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00Mathlib_Tactic_Order_orderCoreImp_spec__0(v_as_2757_, v_sz_boxed_2768_, v_i_boxed_2769_, v_bs_2760_, v___y_2761_, v___y_2762_, v___y_2763_, v___y_2764_, v___y_2765_, v___y_2766_);
lean_dec(v___y_2766_);
lean_dec_ref(v___y_2765_);
lean_dec(v___y_2764_);
lean_dec_ref(v___y_2763_);
lean_dec(v___y_2762_);
lean_dec_ref(v___y_2761_);
lean_dec_ref(v_as_2757_);
return v_res_2770_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1(lean_object* v_mvarId_2771_, lean_object* v_val_2772_, lean_object* v___y_2773_, lean_object* v___y_2774_, lean_object* v___y_2775_, lean_object* v___y_2776_, lean_object* v___y_2777_, lean_object* v___y_2778_){
_start:
{
lean_object* v___x_2780_; 
v___x_2780_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1___redArg(v_mvarId_2771_, v_val_2772_, v___y_2776_);
return v___x_2780_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1___boxed(lean_object* v_mvarId_2781_, lean_object* v_val_2782_, lean_object* v___y_2783_, lean_object* v___y_2784_, lean_object* v___y_2785_, lean_object* v___y_2786_, lean_object* v___y_2787_, lean_object* v___y_2788_, lean_object* v___y_2789_){
_start:
{
lean_object* v_res_2790_; 
v_res_2790_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1(v_mvarId_2781_, v_val_2782_, v___y_2783_, v___y_2784_, v___y_2785_, v___y_2786_, v___y_2787_, v___y_2788_);
lean_dec(v___y_2788_);
lean_dec_ref(v___y_2787_);
lean_dec(v___y_2786_);
lean_dec_ref(v___y_2785_);
lean_dec(v___y_2784_);
lean_dec_ref(v___y_2783_);
return v_res_2790_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4(lean_object* v_cls_2791_, lean_object* v_msg_2792_, lean_object* v___y_2793_, lean_object* v___y_2794_, lean_object* v___y_2795_, lean_object* v___y_2796_, lean_object* v___y_2797_, lean_object* v___y_2798_){
_start:
{
lean_object* v___x_2800_; 
v___x_2800_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___redArg(v_cls_2791_, v_msg_2792_, v___y_2795_, v___y_2796_, v___y_2797_, v___y_2798_);
return v___x_2800_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4___boxed(lean_object* v_cls_2801_, lean_object* v_msg_2802_, lean_object* v___y_2803_, lean_object* v___y_2804_, lean_object* v___y_2805_, lean_object* v___y_2806_, lean_object* v___y_2807_, lean_object* v___y_2808_, lean_object* v___y_2809_){
_start:
{
lean_object* v_res_2810_; 
v_res_2810_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Tactic_Order_orderCoreImp_spec__4(v_cls_2801_, v_msg_2802_, v___y_2803_, v___y_2804_, v___y_2805_, v___y_2806_, v___y_2807_, v___y_2808_);
lean_dec(v___y_2808_);
lean_dec_ref(v___y_2807_);
lean_dec(v___y_2806_);
lean_dec_ref(v___y_2805_);
lean_dec(v___y_2804_);
lean_dec_ref(v___y_2803_);
return v_res_2810_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Order_orderCoreImp_spec__7(lean_object* v_00_u03b1_2811_, lean_object* v_msg_2812_, lean_object* v___y_2813_, lean_object* v___y_2814_, lean_object* v___y_2815_, lean_object* v___y_2816_, lean_object* v___y_2817_, lean_object* v___y_2818_){
_start:
{
lean_object* v___x_2820_; 
v___x_2820_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Order_orderCoreImp_spec__7___redArg(v_msg_2812_, v___y_2815_, v___y_2816_, v___y_2817_, v___y_2818_);
return v___x_2820_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Order_orderCoreImp_spec__7___boxed(lean_object* v_00_u03b1_2821_, lean_object* v_msg_2822_, lean_object* v___y_2823_, lean_object* v___y_2824_, lean_object* v___y_2825_, lean_object* v___y_2826_, lean_object* v___y_2827_, lean_object* v___y_2828_, lean_object* v___y_2829_){
_start:
{
lean_object* v_res_2830_; 
v_res_2830_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Order_orderCoreImp_spec__7(v_00_u03b1_2821_, v_msg_2822_, v___y_2823_, v___y_2824_, v___y_2825_, v___y_2826_, v___y_2827_, v___y_2828_);
lean_dec(v___y_2828_);
lean_dec_ref(v___y_2827_);
lean_dec(v___y_2826_);
lean_dec_ref(v___y_2825_);
lean_dec(v___y_2824_);
lean_dec_ref(v___y_2823_);
return v_res_2830_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1(lean_object* v_00_u03b2_2831_, lean_object* v_x_2832_, lean_object* v_x_2833_, lean_object* v_x_2834_){
_start:
{
lean_object* v___x_2835_; 
v___x_2835_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1___redArg(v_x_2832_, v_x_2833_, v_x_2834_);
return v___x_2835_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3(lean_object* v_00_u03b2_2836_, lean_object* v_x_2837_, size_t v_x_2838_, size_t v_x_2839_, lean_object* v_x_2840_, lean_object* v_x_2841_){
_start:
{
lean_object* v___x_2842_; 
v___x_2842_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3___redArg(v_x_2837_, v_x_2838_, v_x_2839_, v_x_2840_, v_x_2841_);
return v___x_2842_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3___boxed(lean_object* v_00_u03b2_2843_, lean_object* v_x_2844_, lean_object* v_x_2845_, lean_object* v_x_2846_, lean_object* v_x_2847_, lean_object* v_x_2848_){
_start:
{
size_t v_x_69592__boxed_2849_; size_t v_x_69593__boxed_2850_; lean_object* v_res_2851_; 
v_x_69592__boxed_2849_ = lean_unbox_usize(v_x_2845_);
lean_dec(v_x_2845_);
v_x_69593__boxed_2850_ = lean_unbox_usize(v_x_2846_);
lean_dec(v_x_2846_);
v_res_2851_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3(v_00_u03b2_2843_, v_x_2844_, v_x_69592__boxed_2849_, v_x_69593__boxed_2850_, v_x_2847_, v_x_2848_);
return v_res_2851_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3_spec__11(lean_object* v_00_u03b2_2852_, lean_object* v_n_2853_, lean_object* v_k_2854_, lean_object* v_v_2855_){
_start:
{
lean_object* v___x_2856_; 
v___x_2856_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3_spec__11___redArg(v_n_2853_, v_k_2854_, v_v_2855_);
return v___x_2856_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3_spec__12(lean_object* v_00_u03b2_2857_, size_t v_depth_2858_, lean_object* v_keys_2859_, lean_object* v_vals_2860_, lean_object* v_heq_2861_, lean_object* v_i_2862_, lean_object* v_entries_2863_){
_start:
{
lean_object* v___x_2864_; 
v___x_2864_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3_spec__12___redArg(v_depth_2858_, v_keys_2859_, v_vals_2860_, v_i_2862_, v_entries_2863_);
return v___x_2864_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3_spec__12___boxed(lean_object* v_00_u03b2_2865_, lean_object* v_depth_2866_, lean_object* v_keys_2867_, lean_object* v_vals_2868_, lean_object* v_heq_2869_, lean_object* v_i_2870_, lean_object* v_entries_2871_){
_start:
{
size_t v_depth_boxed_2872_; lean_object* v_res_2873_; 
v_depth_boxed_2872_ = lean_unbox_usize(v_depth_2866_);
lean_dec(v_depth_2866_);
v_res_2873_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3_spec__12(v_00_u03b2_2865_, v_depth_boxed_2872_, v_keys_2867_, v_vals_2868_, v_heq_2869_, v_i_2870_, v_entries_2871_);
lean_dec_ref(v_vals_2868_);
lean_dec_ref(v_keys_2867_);
return v_res_2873_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3_spec__11_spec__13(lean_object* v_00_u03b2_2874_, lean_object* v_x_2875_, lean_object* v_x_2876_, lean_object* v_x_2877_, lean_object* v_x_2878_){
_start:
{
lean_object* v___x_2879_; 
v___x_2879_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Tactic_Order_orderCoreImp_spec__1_spec__1_spec__3_spec__11_spec__13___redArg(v_x_2875_, v_x_2876_, v_x_2877_, v_x_2878_);
return v___x_2879_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCore___lam__0(lean_object* v_e_2880_, lean_object* v___y_2881_, lean_object* v___y_2882_, lean_object* v___y_2883_, lean_object* v___y_2884_){
_start:
{
lean_object* v___x_2886_; uint8_t v___x_2887_; lean_object* v___x_2888_; lean_object* v___x_2889_; 
v___x_2886_ = lean_box(0);
v___x_2887_ = 1;
v___x_2888_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_2888_, 0, v_e_2880_);
lean_ctor_set(v___x_2888_, 1, v___x_2886_);
lean_ctor_set_uint8(v___x_2888_, sizeof(void*)*2, v___x_2887_);
v___x_2889_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2889_, 0, v___x_2888_);
return v___x_2889_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCore___lam__0___boxed(lean_object* v_e_2890_, lean_object* v___y_2891_, lean_object* v___y_2892_, lean_object* v___y_2893_, lean_object* v___y_2894_, lean_object* v___y_2895_){
_start:
{
lean_object* v_res_2896_; 
v_res_2896_ = lp_mathlib_Mathlib_Tactic_Order_orderCore___lam__0(v_e_2890_, v___y_2891_, v___y_2892_, v___y_2893_, v___y_2894_);
lean_dec(v___y_2894_);
lean_dec_ref(v___y_2893_);
lean_dec(v___y_2892_);
lean_dec_ref(v___y_2891_);
return v_res_2896_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCore(uint8_t v_only_x3f_2898_, lean_object* v_hyps_2899_, lean_object* v_negGoal_2900_, lean_object* v_g_2901_, lean_object* v_a_2902_, lean_object* v_a_2903_, lean_object* v_a_2904_, lean_object* v_a_2905_){
_start:
{
lean_object* v___f_2907_; uint8_t v___x_2908_; lean_object* v___x_2909_; lean_object* v___x_2910_; lean_object* v___x_2911_; 
v___f_2907_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_orderCore___closed__0));
v___x_2908_ = 2;
v___x_2909_ = lean_box(v_only_x3f_2898_);
v___x_2910_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Order_orderCoreImp___boxed), 11, 4);
lean_closure_set(v___x_2910_, 0, v___x_2909_);
lean_closure_set(v___x_2910_, 1, v_hyps_2899_);
lean_closure_set(v___x_2910_, 2, v_negGoal_2900_);
lean_closure_set(v___x_2910_, 3, v_g_2901_);
v___x_2911_ = lp_mathlib_Mathlib_Tactic_AtomM_run___redArg(v___x_2908_, v___x_2910_, v___f_2907_, v_a_2902_, v_a_2903_, v_a_2904_, v_a_2905_);
return v___x_2911_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_orderCore___boxed(lean_object* v_only_x3f_2912_, lean_object* v_hyps_2913_, lean_object* v_negGoal_2914_, lean_object* v_g_2915_, lean_object* v_a_2916_, lean_object* v_a_2917_, lean_object* v_a_2918_, lean_object* v_a_2919_, lean_object* v_a_2920_){
_start:
{
uint8_t v_only_x3f_boxed_2921_; lean_object* v_res_2922_; 
v_only_x3f_boxed_2921_ = lean_unbox(v_only_x3f_2912_);
v_res_2922_ = lp_mathlib_Mathlib_Tactic_Order_orderCore(v_only_x3f_boxed_2921_, v_hyps_2913_, v_negGoal_2914_, v_g_2915_, v_a_2916_, v_a_2917_, v_a_2918_, v_a_2919_);
lean_dec(v_a_2919_);
lean_dec_ref(v_a_2918_);
lean_dec(v_a_2917_);
lean_dec_ref(v_a_2916_);
return v_res_2922_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_3008_; lean_object* v___x_3009_; lean_object* v___x_3010_; 
v___x_3008_ = lean_box(0);
v___x_3009_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_3010_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3010_, 0, v___x_3009_);
lean_ctor_set(v___x_3010_, 1, v___x_3008_);
return v___x_3010_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__0___redArg(){
_start:
{
lean_object* v___x_3012_; lean_object* v___x_3013_; 
v___x_3012_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__0___redArg___closed__0);
v___x_3013_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3013_, 0, v___x_3012_);
return v___x_3013_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__0___redArg___boxed(lean_object* v___y_3014_){
_start:
{
lean_object* v_res_3015_; 
v_res_3015_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__0___redArg();
return v_res_3015_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__0(lean_object* v_00_u03b1_3016_, lean_object* v___y_3017_, lean_object* v___y_3018_, lean_object* v___y_3019_, lean_object* v___y_3020_, lean_object* v___y_3021_, lean_object* v___y_3022_, lean_object* v___y_3023_, lean_object* v___y_3024_){
_start:
{
lean_object* v___x_3026_; 
v___x_3026_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__0___redArg();
return v___x_3026_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__0___boxed(lean_object* v_00_u03b1_3027_, lean_object* v___y_3028_, lean_object* v___y_3029_, lean_object* v___y_3030_, lean_object* v___y_3031_, lean_object* v___y_3032_, lean_object* v___y_3033_, lean_object* v___y_3034_, lean_object* v___y_3035_, lean_object* v___y_3036_){
_start:
{
lean_object* v_res_3037_; 
v_res_3037_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__0(v_00_u03b1_3027_, v___y_3028_, v___y_3029_, v___y_3030_, v___y_3031_, v___y_3032_, v___y_3033_, v___y_3034_, v___y_3035_);
lean_dec(v___y_3035_);
lean_dec_ref(v___y_3034_);
lean_dec(v___y_3033_);
lean_dec_ref(v___y_3032_);
lean_dec(v___y_3031_);
lean_dec_ref(v___y_3030_);
lean_dec(v___y_3029_);
lean_dec_ref(v___y_3028_);
return v_res_3037_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__2___redArg(lean_object* v_x_3038_, lean_object* v___y_3039_, lean_object* v___y_3040_, lean_object* v___y_3041_, lean_object* v___y_3042_, lean_object* v___y_3043_, lean_object* v___y_3044_, lean_object* v___y_3045_, lean_object* v___y_3046_){
_start:
{
lean_object* v___x_3048_; 
v___x_3048_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_3040_, v___y_3042_, v___y_3044_, v___y_3046_);
if (lean_obj_tag(v___x_3048_) == 0)
{
lean_object* v_a_3049_; lean_object* v___x_3050_; 
v_a_3049_ = lean_ctor_get(v___x_3048_, 0);
lean_inc(v_a_3049_);
lean_dec_ref_known(v___x_3048_, 1);
v___x_3050_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_3040_, v___y_3042_, v___y_3044_, v___y_3046_);
if (lean_obj_tag(v___x_3050_) == 0)
{
lean_object* v_a_3051_; lean_object* v___x_3052_; 
v_a_3051_ = lean_ctor_get(v___x_3050_, 0);
lean_inc(v_a_3051_);
lean_dec_ref_known(v___x_3050_, 1);
lean_inc(v___y_3046_);
lean_inc_ref(v___y_3045_);
lean_inc(v___y_3044_);
lean_inc_ref(v___y_3043_);
lean_inc(v___y_3042_);
lean_inc_ref(v___y_3041_);
lean_inc(v___y_3040_);
lean_inc_ref(v___y_3039_);
v___x_3052_ = lean_apply_9(v_x_3038_, v___y_3039_, v___y_3040_, v___y_3041_, v___y_3042_, v___y_3043_, v___y_3044_, v___y_3045_, v___y_3046_, lean_box(0));
if (lean_obj_tag(v___x_3052_) == 0)
{
lean_dec(v_a_3051_);
lean_dec(v_a_3049_);
return v___x_3052_;
}
else
{
lean_object* v_a_3053_; uint8_t v___y_3055_; uint8_t v___x_3082_; 
v_a_3053_ = lean_ctor_get(v___x_3052_, 0);
lean_inc(v_a_3053_);
v___x_3082_ = l_Lean_Exception_isInterrupt(v_a_3053_);
if (v___x_3082_ == 0)
{
uint8_t v___x_3083_; 
lean_inc(v_a_3053_);
v___x_3083_ = l_Lean_Exception_isRuntime(v_a_3053_);
v___y_3055_ = v___x_3083_;
goto v___jp_3054_;
}
else
{
v___y_3055_ = v___x_3082_;
goto v___jp_3054_;
}
v___jp_3054_:
{
if (v___y_3055_ == 0)
{
lean_object* v___x_3056_; 
lean_dec_ref_known(v___x_3052_, 1);
v___x_3056_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_3051_, v___y_3055_, v___y_3040_, v___y_3041_, v___y_3042_, v___y_3043_, v___y_3044_, v___y_3045_, v___y_3046_);
if (lean_obj_tag(v___x_3056_) == 0)
{
lean_object* v___x_3057_; 
lean_dec_ref_known(v___x_3056_, 1);
v___x_3057_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_3049_, v___y_3055_, v___y_3040_, v___y_3041_, v___y_3042_, v___y_3043_, v___y_3044_, v___y_3045_, v___y_3046_);
if (lean_obj_tag(v___x_3057_) == 0)
{
lean_object* v___x_3059_; uint8_t v_isShared_3060_; uint8_t v_isSharedCheck_3064_; 
v_isSharedCheck_3064_ = !lean_is_exclusive(v___x_3057_);
if (v_isSharedCheck_3064_ == 0)
{
lean_object* v_unused_3065_; 
v_unused_3065_ = lean_ctor_get(v___x_3057_, 0);
lean_dec(v_unused_3065_);
v___x_3059_ = v___x_3057_;
v_isShared_3060_ = v_isSharedCheck_3064_;
goto v_resetjp_3058_;
}
else
{
lean_dec(v___x_3057_);
v___x_3059_ = lean_box(0);
v_isShared_3060_ = v_isSharedCheck_3064_;
goto v_resetjp_3058_;
}
v_resetjp_3058_:
{
lean_object* v___x_3062_; 
if (v_isShared_3060_ == 0)
{
lean_ctor_set_tag(v___x_3059_, 1);
lean_ctor_set(v___x_3059_, 0, v_a_3053_);
v___x_3062_ = v___x_3059_;
goto v_reusejp_3061_;
}
else
{
lean_object* v_reuseFailAlloc_3063_; 
v_reuseFailAlloc_3063_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3063_, 0, v_a_3053_);
v___x_3062_ = v_reuseFailAlloc_3063_;
goto v_reusejp_3061_;
}
v_reusejp_3061_:
{
return v___x_3062_;
}
}
}
else
{
lean_object* v_a_3066_; lean_object* v___x_3068_; uint8_t v_isShared_3069_; uint8_t v_isSharedCheck_3073_; 
lean_dec(v_a_3053_);
v_a_3066_ = lean_ctor_get(v___x_3057_, 0);
v_isSharedCheck_3073_ = !lean_is_exclusive(v___x_3057_);
if (v_isSharedCheck_3073_ == 0)
{
v___x_3068_ = v___x_3057_;
v_isShared_3069_ = v_isSharedCheck_3073_;
goto v_resetjp_3067_;
}
else
{
lean_inc(v_a_3066_);
lean_dec(v___x_3057_);
v___x_3068_ = lean_box(0);
v_isShared_3069_ = v_isSharedCheck_3073_;
goto v_resetjp_3067_;
}
v_resetjp_3067_:
{
lean_object* v___x_3071_; 
if (v_isShared_3069_ == 0)
{
v___x_3071_ = v___x_3068_;
goto v_reusejp_3070_;
}
else
{
lean_object* v_reuseFailAlloc_3072_; 
v_reuseFailAlloc_3072_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3072_, 0, v_a_3066_);
v___x_3071_ = v_reuseFailAlloc_3072_;
goto v_reusejp_3070_;
}
v_reusejp_3070_:
{
return v___x_3071_;
}
}
}
}
else
{
lean_object* v_a_3074_; lean_object* v___x_3076_; uint8_t v_isShared_3077_; uint8_t v_isSharedCheck_3081_; 
lean_dec(v_a_3053_);
lean_dec(v_a_3049_);
v_a_3074_ = lean_ctor_get(v___x_3056_, 0);
v_isSharedCheck_3081_ = !lean_is_exclusive(v___x_3056_);
if (v_isSharedCheck_3081_ == 0)
{
v___x_3076_ = v___x_3056_;
v_isShared_3077_ = v_isSharedCheck_3081_;
goto v_resetjp_3075_;
}
else
{
lean_inc(v_a_3074_);
lean_dec(v___x_3056_);
v___x_3076_ = lean_box(0);
v_isShared_3077_ = v_isSharedCheck_3081_;
goto v_resetjp_3075_;
}
v_resetjp_3075_:
{
lean_object* v___x_3079_; 
if (v_isShared_3077_ == 0)
{
v___x_3079_ = v___x_3076_;
goto v_reusejp_3078_;
}
else
{
lean_object* v_reuseFailAlloc_3080_; 
v_reuseFailAlloc_3080_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3080_, 0, v_a_3074_);
v___x_3079_ = v_reuseFailAlloc_3080_;
goto v_reusejp_3078_;
}
v_reusejp_3078_:
{
return v___x_3079_;
}
}
}
}
else
{
lean_dec(v_a_3053_);
lean_dec(v_a_3051_);
lean_dec(v_a_3049_);
return v___x_3052_;
}
}
}
}
else
{
lean_object* v_a_3084_; lean_object* v___x_3086_; uint8_t v_isShared_3087_; uint8_t v_isSharedCheck_3091_; 
lean_dec(v_a_3049_);
lean_dec_ref(v_x_3038_);
v_a_3084_ = lean_ctor_get(v___x_3050_, 0);
v_isSharedCheck_3091_ = !lean_is_exclusive(v___x_3050_);
if (v_isSharedCheck_3091_ == 0)
{
v___x_3086_ = v___x_3050_;
v_isShared_3087_ = v_isSharedCheck_3091_;
goto v_resetjp_3085_;
}
else
{
lean_inc(v_a_3084_);
lean_dec(v___x_3050_);
v___x_3086_ = lean_box(0);
v_isShared_3087_ = v_isSharedCheck_3091_;
goto v_resetjp_3085_;
}
v_resetjp_3085_:
{
lean_object* v___x_3089_; 
if (v_isShared_3087_ == 0)
{
v___x_3089_ = v___x_3086_;
goto v_reusejp_3088_;
}
else
{
lean_object* v_reuseFailAlloc_3090_; 
v_reuseFailAlloc_3090_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3090_, 0, v_a_3084_);
v___x_3089_ = v_reuseFailAlloc_3090_;
goto v_reusejp_3088_;
}
v_reusejp_3088_:
{
return v___x_3089_;
}
}
}
}
else
{
lean_object* v_a_3092_; lean_object* v___x_3094_; uint8_t v_isShared_3095_; uint8_t v_isSharedCheck_3099_; 
lean_dec_ref(v_x_3038_);
v_a_3092_ = lean_ctor_get(v___x_3048_, 0);
v_isSharedCheck_3099_ = !lean_is_exclusive(v___x_3048_);
if (v_isSharedCheck_3099_ == 0)
{
v___x_3094_ = v___x_3048_;
v_isShared_3095_ = v_isSharedCheck_3099_;
goto v_resetjp_3093_;
}
else
{
lean_inc(v_a_3092_);
lean_dec(v___x_3048_);
v___x_3094_ = lean_box(0);
v_isShared_3095_ = v_isSharedCheck_3099_;
goto v_resetjp_3093_;
}
v_resetjp_3093_:
{
lean_object* v___x_3097_; 
if (v_isShared_3095_ == 0)
{
v___x_3097_ = v___x_3094_;
goto v_reusejp_3096_;
}
else
{
lean_object* v_reuseFailAlloc_3098_; 
v_reuseFailAlloc_3098_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3098_, 0, v_a_3092_);
v___x_3097_ = v_reuseFailAlloc_3098_;
goto v_reusejp_3096_;
}
v_reusejp_3096_:
{
return v___x_3097_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__2___redArg___boxed(lean_object* v_x_3100_, lean_object* v___y_3101_, lean_object* v___y_3102_, lean_object* v___y_3103_, lean_object* v___y_3104_, lean_object* v___y_3105_, lean_object* v___y_3106_, lean_object* v___y_3107_, lean_object* v___y_3108_, lean_object* v___y_3109_){
_start:
{
lean_object* v_res_3110_; 
v_res_3110_ = lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__2___redArg(v_x_3100_, v___y_3101_, v___y_3102_, v___y_3103_, v___y_3104_, v___y_3105_, v___y_3106_, v___y_3107_, v___y_3108_);
lean_dec(v___y_3108_);
lean_dec_ref(v___y_3107_);
lean_dec(v___y_3106_);
lean_dec_ref(v___y_3105_);
lean_dec(v___y_3104_);
lean_dec_ref(v___y_3103_);
lean_dec(v___y_3102_);
lean_dec_ref(v___y_3101_);
return v_res_3110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__2(lean_object* v_00_u03b1_3111_, lean_object* v_x_3112_, lean_object* v___y_3113_, lean_object* v___y_3114_, lean_object* v___y_3115_, lean_object* v___y_3116_, lean_object* v___y_3117_, lean_object* v___y_3118_, lean_object* v___y_3119_, lean_object* v___y_3120_){
_start:
{
lean_object* v___x_3122_; 
v___x_3122_ = lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__2___redArg(v_x_3112_, v___y_3113_, v___y_3114_, v___y_3115_, v___y_3116_, v___y_3117_, v___y_3118_, v___y_3119_, v___y_3120_);
return v___x_3122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__2___boxed(lean_object* v_00_u03b1_3123_, lean_object* v_x_3124_, lean_object* v___y_3125_, lean_object* v___y_3126_, lean_object* v___y_3127_, lean_object* v___y_3128_, lean_object* v___y_3129_, lean_object* v___y_3130_, lean_object* v___y_3131_, lean_object* v___y_3132_, lean_object* v___y_3133_){
_start:
{
lean_object* v_res_3134_; 
v_res_3134_ = lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__2(v_00_u03b1_3123_, v_x_3124_, v___y_3125_, v___y_3126_, v___y_3127_, v___y_3128_, v___y_3129_, v___y_3130_, v___y_3131_, v___y_3132_);
lean_dec(v___y_3132_);
lean_dec_ref(v___y_3131_);
lean_dec(v___y_3130_);
lean_dec_ref(v___y_3129_);
lean_dec(v___y_3128_);
lean_dec_ref(v___y_3127_);
lean_dec(v___y_3126_);
lean_dec_ref(v___y_3125_);
return v_res_3134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1___lam__0(uint8_t v___y_3135_, lean_object* v_a_3136_, lean_object* v_a_3137_, lean_object* v___y_3138_, lean_object* v___y_3139_, lean_object* v___y_3140_, lean_object* v___y_3141_, lean_object* v___y_3142_, lean_object* v___y_3143_, lean_object* v___y_3144_, lean_object* v___y_3145_){
_start:
{
lean_object* v___x_3147_; 
v___x_3147_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_3139_, v___y_3142_, v___y_3143_, v___y_3144_, v___y_3145_);
if (lean_obj_tag(v___x_3147_) == 0)
{
lean_object* v_a_3148_; lean_object* v___x_3149_; 
v_a_3148_ = lean_ctor_get(v___x_3147_, 0);
lean_inc(v_a_3148_);
lean_dec_ref_known(v___x_3147_, 1);
v___x_3149_ = lp_mathlib_Mathlib_Tactic_Order_orderCore(v___y_3135_, v_a_3136_, v_a_3137_, v_a_3148_, v___y_3142_, v___y_3143_, v___y_3144_, v___y_3145_);
if (lean_obj_tag(v___x_3149_) == 0)
{
lean_object* v___x_3150_; lean_object* v___x_3151_; 
lean_dec_ref_known(v___x_3149_, 1);
v___x_3150_ = lean_box(0);
v___x_3151_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_3150_, v___y_3139_, v___y_3142_, v___y_3143_, v___y_3144_, v___y_3145_);
if (lean_obj_tag(v___x_3151_) == 0)
{
lean_object* v___x_3153_; uint8_t v_isShared_3154_; uint8_t v_isSharedCheck_3159_; 
v_isSharedCheck_3159_ = !lean_is_exclusive(v___x_3151_);
if (v_isSharedCheck_3159_ == 0)
{
lean_object* v_unused_3160_; 
v_unused_3160_ = lean_ctor_get(v___x_3151_, 0);
lean_dec(v_unused_3160_);
v___x_3153_ = v___x_3151_;
v_isShared_3154_ = v_isSharedCheck_3159_;
goto v_resetjp_3152_;
}
else
{
lean_dec(v___x_3151_);
v___x_3153_ = lean_box(0);
v_isShared_3154_ = v_isSharedCheck_3159_;
goto v_resetjp_3152_;
}
v_resetjp_3152_:
{
lean_object* v___x_3155_; lean_object* v___x_3157_; 
v___x_3155_ = lean_box(0);
if (v_isShared_3154_ == 0)
{
lean_ctor_set(v___x_3153_, 0, v___x_3155_);
v___x_3157_ = v___x_3153_;
goto v_reusejp_3156_;
}
else
{
lean_object* v_reuseFailAlloc_3158_; 
v_reuseFailAlloc_3158_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3158_, 0, v___x_3155_);
v___x_3157_ = v_reuseFailAlloc_3158_;
goto v_reusejp_3156_;
}
v_reusejp_3156_:
{
return v___x_3157_;
}
}
}
else
{
return v___x_3151_;
}
}
else
{
return v___x_3149_;
}
}
else
{
lean_object* v_a_3161_; lean_object* v___x_3163_; uint8_t v_isShared_3164_; uint8_t v_isSharedCheck_3168_; 
lean_dec_ref(v_a_3137_);
lean_dec_ref(v_a_3136_);
v_a_3161_ = lean_ctor_get(v___x_3147_, 0);
v_isSharedCheck_3168_ = !lean_is_exclusive(v___x_3147_);
if (v_isSharedCheck_3168_ == 0)
{
v___x_3163_ = v___x_3147_;
v_isShared_3164_ = v_isSharedCheck_3168_;
goto v_resetjp_3162_;
}
else
{
lean_inc(v_a_3161_);
lean_dec(v___x_3147_);
v___x_3163_ = lean_box(0);
v_isShared_3164_ = v_isSharedCheck_3168_;
goto v_resetjp_3162_;
}
v_resetjp_3162_:
{
lean_object* v___x_3166_; 
if (v_isShared_3164_ == 0)
{
v___x_3166_ = v___x_3163_;
goto v_reusejp_3165_;
}
else
{
lean_object* v_reuseFailAlloc_3167_; 
v_reuseFailAlloc_3167_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3167_, 0, v_a_3161_);
v___x_3166_ = v_reuseFailAlloc_3167_;
goto v_reusejp_3165_;
}
v_reusejp_3165_:
{
return v___x_3166_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1___lam__0___boxed(lean_object* v___y_3169_, lean_object* v_a_3170_, lean_object* v_a_3171_, lean_object* v___y_3172_, lean_object* v___y_3173_, lean_object* v___y_3174_, lean_object* v___y_3175_, lean_object* v___y_3176_, lean_object* v___y_3177_, lean_object* v___y_3178_, lean_object* v___y_3179_, lean_object* v___y_3180_){
_start:
{
uint8_t v___y_2629__boxed_3181_; lean_object* v_res_3182_; 
v___y_2629__boxed_3181_ = lean_unbox(v___y_3169_);
v_res_3182_ = lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1___lam__0(v___y_2629__boxed_3181_, v_a_3170_, v_a_3171_, v___y_3172_, v___y_3173_, v___y_3174_, v___y_3175_, v___y_3176_, v___y_3177_, v___y_3178_, v___y_3179_);
lean_dec(v___y_3179_);
lean_dec_ref(v___y_3178_);
lean_dec(v___y_3177_);
lean_dec_ref(v___y_3176_);
lean_dec(v___y_3175_);
lean_dec_ref(v___y_3174_);
lean_dec(v___y_3173_);
lean_dec_ref(v___y_3172_);
return v_res_3182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1___lam__1(lean_object* v___f_3183_, lean_object* v___y_3184_, lean_object* v___y_3185_, lean_object* v___y_3186_, lean_object* v___y_3187_, lean_object* v___y_3188_, lean_object* v___y_3189_, lean_object* v___y_3190_, lean_object* v___y_3191_){
_start:
{
lean_object* v___x_3193_; 
v___x_3193_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_3183_, v___y_3184_, v___y_3185_, v___y_3186_, v___y_3187_, v___y_3188_, v___y_3189_, v___y_3190_, v___y_3191_);
return v___x_3193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1___lam__1___boxed(lean_object* v___f_3194_, lean_object* v___y_3195_, lean_object* v___y_3196_, lean_object* v___y_3197_, lean_object* v___y_3198_, lean_object* v___y_3199_, lean_object* v___y_3200_, lean_object* v___y_3201_, lean_object* v___y_3202_, lean_object* v___y_3203_){
_start:
{
lean_object* v_res_3204_; 
v_res_3204_ = lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1___lam__1(v___f_3194_, v___y_3195_, v___y_3196_, v___y_3197_, v___y_3198_, v___y_3199_, v___y_3200_, v___y_3201_, v___y_3202_);
lean_dec(v___y_3202_);
lean_dec_ref(v___y_3201_);
lean_dec(v___y_3200_);
lean_dec_ref(v___y_3199_);
lean_dec(v___y_3198_);
lean_dec_ref(v___y_3197_);
lean_dec(v___y_3196_);
lean_dec_ref(v___y_3195_);
return v_res_3204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__1(size_t v_sz_3205_, size_t v_i_3206_, lean_object* v_bs_3207_, lean_object* v___y_3208_, lean_object* v___y_3209_, lean_object* v___y_3210_, lean_object* v___y_3211_, lean_object* v___y_3212_, lean_object* v___y_3213_, lean_object* v___y_3214_, lean_object* v___y_3215_){
_start:
{
uint8_t v___x_3217_; 
v___x_3217_ = lean_usize_dec_lt(v_i_3206_, v_sz_3205_);
if (v___x_3217_ == 0)
{
lean_object* v___x_3218_; 
v___x_3218_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3218_, 0, v_bs_3207_);
return v___x_3218_;
}
else
{
lean_object* v___x_3219_; lean_object* v_v_3220_; lean_object* v___x_3221_; 
v___x_3219_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn___closed__1_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_));
v_v_3220_ = lean_array_uget_borrowed(v_bs_3207_, v_i_3206_);
lean_inc(v_v_3220_);
v___x_3221_ = lp_mathlib_elabTermWithoutNewMVars(v___x_3219_, v_v_3220_, v___y_3208_, v___y_3209_, v___y_3210_, v___y_3211_, v___y_3212_, v___y_3213_, v___y_3214_, v___y_3215_);
if (lean_obj_tag(v___x_3221_) == 0)
{
lean_object* v_a_3222_; lean_object* v___x_3223_; lean_object* v_bs_x27_3224_; size_t v___x_3225_; size_t v___x_3226_; lean_object* v___x_3227_; 
v_a_3222_ = lean_ctor_get(v___x_3221_, 0);
lean_inc(v_a_3222_);
lean_dec_ref_known(v___x_3221_, 1);
v___x_3223_ = lean_unsigned_to_nat(0u);
v_bs_x27_3224_ = lean_array_uset(v_bs_3207_, v_i_3206_, v___x_3223_);
v___x_3225_ = ((size_t)1ULL);
v___x_3226_ = lean_usize_add(v_i_3206_, v___x_3225_);
v___x_3227_ = lean_array_uset(v_bs_x27_3224_, v_i_3206_, v_a_3222_);
v_i_3206_ = v___x_3226_;
v_bs_3207_ = v___x_3227_;
goto _start;
}
else
{
lean_object* v_a_3229_; lean_object* v___x_3231_; uint8_t v_isShared_3232_; uint8_t v_isSharedCheck_3236_; 
lean_dec_ref(v_bs_3207_);
v_a_3229_ = lean_ctor_get(v___x_3221_, 0);
v_isSharedCheck_3236_ = !lean_is_exclusive(v___x_3221_);
if (v_isSharedCheck_3236_ == 0)
{
v___x_3231_ = v___x_3221_;
v_isShared_3232_ = v_isSharedCheck_3236_;
goto v_resetjp_3230_;
}
else
{
lean_inc(v_a_3229_);
lean_dec(v___x_3221_);
v___x_3231_ = lean_box(0);
v_isShared_3232_ = v_isSharedCheck_3236_;
goto v_resetjp_3230_;
}
v_resetjp_3230_:
{
lean_object* v___x_3234_; 
if (v_isShared_3232_ == 0)
{
v___x_3234_ = v___x_3231_;
goto v_reusejp_3233_;
}
else
{
lean_object* v_reuseFailAlloc_3235_; 
v_reuseFailAlloc_3235_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3235_, 0, v_a_3229_);
v___x_3234_ = v_reuseFailAlloc_3235_;
goto v_reusejp_3233_;
}
v_reusejp_3233_:
{
return v___x_3234_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__1___boxed(lean_object* v_sz_3237_, lean_object* v_i_3238_, lean_object* v_bs_3239_, lean_object* v___y_3240_, lean_object* v___y_3241_, lean_object* v___y_3242_, lean_object* v___y_3243_, lean_object* v___y_3244_, lean_object* v___y_3245_, lean_object* v___y_3246_, lean_object* v___y_3247_, lean_object* v___y_3248_){
_start:
{
size_t v_sz_boxed_3249_; size_t v_i_boxed_3250_; lean_object* v_res_3251_; 
v_sz_boxed_3249_ = lean_unbox_usize(v_sz_3237_);
lean_dec(v_sz_3237_);
v_i_boxed_3250_ = lean_unbox_usize(v_i_3238_);
lean_dec(v_i_3238_);
v_res_3251_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__1(v_sz_boxed_3249_, v_i_boxed_3250_, v_bs_3239_, v___y_3240_, v___y_3241_, v___y_3242_, v___y_3243_, v___y_3244_, v___y_3245_, v___y_3246_, v___y_3247_);
lean_dec(v___y_3247_);
lean_dec_ref(v___y_3246_);
lean_dec(v___y_3245_);
lean_dec_ref(v___y_3244_);
lean_dec(v___y_3243_);
lean_dec_ref(v___y_3242_);
lean_dec(v___y_3241_);
lean_dec_ref(v___y_3240_);
return v_res_3251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1___lam__2(lean_object* v___x_3252_, lean_object* v___x_3253_, uint8_t v___x_3254_, lean_object* v_o_3255_, uint8_t v___x_3256_, lean_object* v_args_3257_, lean_object* v___x_3258_, lean_object* v___y_3259_, lean_object* v___y_3260_, lean_object* v___y_3261_, lean_object* v___y_3262_, lean_object* v___y_3263_, lean_object* v___y_3264_, lean_object* v___y_3265_, lean_object* v___y_3266_){
_start:
{
lean_object* v___x_3268_; 
v___x_3268_ = l_Lean_Elab_Tactic_elabTerm(v___x_3252_, v___x_3253_, v___x_3254_, v___y_3259_, v___y_3260_, v___y_3261_, v___y_3262_, v___y_3263_, v___y_3264_, v___y_3265_, v___y_3266_);
if (lean_obj_tag(v___x_3268_) == 0)
{
lean_object* v_a_3269_; lean_object* v___y_3271_; uint8_t v___y_3272_; lean_object* v___y_3278_; 
v_a_3269_ = lean_ctor_get(v___x_3268_, 0);
lean_inc(v_a_3269_);
lean_dec_ref_known(v___x_3268_, 1);
if (lean_obj_tag(v_args_3257_) == 0)
{
lean_object* v___x_3292_; 
v___x_3292_ = lean_mk_empty_array_with_capacity(v___x_3258_);
v___y_3278_ = v___x_3292_;
goto v___jp_3277_;
}
else
{
lean_object* v_val_3293_; lean_object* v___x_3294_; 
v_val_3293_ = lean_ctor_get(v_args_3257_, 0);
v___x_3294_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_val_3293_);
v___y_3278_ = v___x_3294_;
goto v___jp_3277_;
}
v___jp_3270_:
{
lean_object* v___x_3273_; lean_object* v___f_3274_; lean_object* v___f_3275_; lean_object* v___x_3276_; 
v___x_3273_ = lean_box(v___y_3272_);
v___f_3274_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1___lam__0___boxed), 12, 3);
lean_closure_set(v___f_3274_, 0, v___x_3273_);
lean_closure_set(v___f_3274_, 1, v___y_3271_);
lean_closure_set(v___f_3274_, 2, v_a_3269_);
v___f_3275_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1___lam__1___boxed), 10, 1);
lean_closure_set(v___f_3275_, 0, v___f_3274_);
v___x_3276_ = lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__2___redArg(v___f_3275_, v___y_3259_, v___y_3260_, v___y_3261_, v___y_3262_, v___y_3263_, v___y_3264_, v___y_3265_, v___y_3266_);
return v___x_3276_;
}
v___jp_3277_:
{
size_t v_sz_3279_; size_t v___x_3280_; lean_object* v___x_3281_; 
v_sz_3279_ = lean_array_size(v___y_3278_);
v___x_3280_ = ((size_t)0ULL);
v___x_3281_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__1(v_sz_3279_, v___x_3280_, v___y_3278_, v___y_3259_, v___y_3260_, v___y_3261_, v___y_3262_, v___y_3263_, v___y_3264_, v___y_3265_, v___y_3266_);
if (lean_obj_tag(v___x_3281_) == 0)
{
if (lean_obj_tag(v_o_3255_) == 0)
{
lean_object* v_a_3282_; 
v_a_3282_ = lean_ctor_get(v___x_3281_, 0);
lean_inc(v_a_3282_);
lean_dec_ref_known(v___x_3281_, 1);
v___y_3271_ = v_a_3282_;
v___y_3272_ = v___x_3254_;
goto v___jp_3270_;
}
else
{
lean_object* v_a_3283_; 
v_a_3283_ = lean_ctor_get(v___x_3281_, 0);
lean_inc(v_a_3283_);
lean_dec_ref_known(v___x_3281_, 1);
v___y_3271_ = v_a_3283_;
v___y_3272_ = v___x_3256_;
goto v___jp_3270_;
}
}
else
{
lean_object* v_a_3284_; lean_object* v___x_3286_; uint8_t v_isShared_3287_; uint8_t v_isSharedCheck_3291_; 
lean_dec(v_a_3269_);
v_a_3284_ = lean_ctor_get(v___x_3281_, 0);
v_isSharedCheck_3291_ = !lean_is_exclusive(v___x_3281_);
if (v_isSharedCheck_3291_ == 0)
{
v___x_3286_ = v___x_3281_;
v_isShared_3287_ = v_isSharedCheck_3291_;
goto v_resetjp_3285_;
}
else
{
lean_inc(v_a_3284_);
lean_dec(v___x_3281_);
v___x_3286_ = lean_box(0);
v_isShared_3287_ = v_isSharedCheck_3291_;
goto v_resetjp_3285_;
}
v_resetjp_3285_:
{
lean_object* v___x_3289_; 
if (v_isShared_3287_ == 0)
{
v___x_3289_ = v___x_3286_;
goto v_reusejp_3288_;
}
else
{
lean_object* v_reuseFailAlloc_3290_; 
v_reuseFailAlloc_3290_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3290_, 0, v_a_3284_);
v___x_3289_ = v_reuseFailAlloc_3290_;
goto v_reusejp_3288_;
}
v_reusejp_3288_:
{
return v___x_3289_;
}
}
}
}
}
else
{
lean_object* v_a_3295_; lean_object* v___x_3297_; uint8_t v_isShared_3298_; uint8_t v_isSharedCheck_3302_; 
v_a_3295_ = lean_ctor_get(v___x_3268_, 0);
v_isSharedCheck_3302_ = !lean_is_exclusive(v___x_3268_);
if (v_isSharedCheck_3302_ == 0)
{
v___x_3297_ = v___x_3268_;
v_isShared_3298_ = v_isSharedCheck_3302_;
goto v_resetjp_3296_;
}
else
{
lean_inc(v_a_3295_);
lean_dec(v___x_3268_);
v___x_3297_ = lean_box(0);
v_isShared_3298_ = v_isSharedCheck_3302_;
goto v_resetjp_3296_;
}
v_resetjp_3296_:
{
lean_object* v___x_3300_; 
if (v_isShared_3298_ == 0)
{
v___x_3300_ = v___x_3297_;
goto v_reusejp_3299_;
}
else
{
lean_object* v_reuseFailAlloc_3301_; 
v_reuseFailAlloc_3301_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3301_, 0, v_a_3295_);
v___x_3300_ = v_reuseFailAlloc_3301_;
goto v_reusejp_3299_;
}
v_reusejp_3299_:
{
return v___x_3300_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1___lam__2___boxed(lean_object* v___x_3303_, lean_object* v___x_3304_, lean_object* v___x_3305_, lean_object* v_o_3306_, lean_object* v___x_3307_, lean_object* v_args_3308_, lean_object* v___x_3309_, lean_object* v___y_3310_, lean_object* v___y_3311_, lean_object* v___y_3312_, lean_object* v___y_3313_, lean_object* v___y_3314_, lean_object* v___y_3315_, lean_object* v___y_3316_, lean_object* v___y_3317_, lean_object* v___y_3318_){
_start:
{
uint8_t v___x_2816__boxed_3319_; uint8_t v___x_2817__boxed_3320_; lean_object* v_res_3321_; 
v___x_2816__boxed_3319_ = lean_unbox(v___x_3305_);
v___x_2817__boxed_3320_ = lean_unbox(v___x_3307_);
v_res_3321_ = lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1___lam__2(v___x_3303_, v___x_3304_, v___x_2816__boxed_3319_, v_o_3306_, v___x_2817__boxed_3320_, v_args_3308_, v___x_3309_, v___y_3310_, v___y_3311_, v___y_3312_, v___y_3313_, v___y_3314_, v___y_3315_, v___y_3316_, v___y_3317_);
lean_dec(v___y_3317_);
lean_dec_ref(v___y_3316_);
lean_dec(v___y_3315_);
lean_dec_ref(v___y_3314_);
lean_dec(v___y_3313_);
lean_dec_ref(v___y_3312_);
lean_dec(v___y_3311_);
lean_dec_ref(v___y_3310_);
lean_dec(v___x_3309_);
lean_dec(v_args_3308_);
lean_dec(v_o_3306_);
return v_res_3321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1(lean_object* v_x_3322_, lean_object* v_a_3323_, lean_object* v_a_3324_, lean_object* v_a_3325_, lean_object* v_a_3326_, lean_object* v_a_3327_, lean_object* v_a_3328_, lean_object* v_a_3329_, lean_object* v_a_3330_){
_start:
{
lean_object* v___x_3332_; lean_object* v___x_3333_; uint8_t v___x_3334_; 
v___x_3332_ = lean_unsigned_to_nat(0u);
v___x_3333_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__1));
lean_inc(v_x_3322_);
v___x_3334_ = l_Lean_Syntax_isOfKind(v_x_3322_, v___x_3333_);
if (v___x_3334_ == 0)
{
lean_object* v___x_3335_; 
lean_dec(v_x_3322_);
v___x_3335_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__0___redArg();
return v___x_3335_;
}
else
{
lean_object* v___x_3336_; lean_object* v___x_3337_; lean_object* v___x_3338_; uint8_t v___x_3339_; lean_object* v___y_3341_; lean_object* v___y_3342_; lean_object* v___y_3343_; lean_object* v___y_3344_; lean_object* v___y_3345_; lean_object* v___y_3346_; lean_object* v___y_3347_; lean_object* v___y_3348_; lean_object* v___y_3349_; lean_object* v_args_3350_; lean_object* v_o_3360_; lean_object* v___y_3361_; lean_object* v___y_3362_; lean_object* v___y_3363_; lean_object* v___y_3364_; lean_object* v___y_3365_; lean_object* v___y_3366_; lean_object* v___y_3367_; lean_object* v___y_3368_; 
v___x_3336_ = lean_unsigned_to_nat(1u);
v___x_3337_ = l_Lean_Syntax_getArg(v_x_3322_, v___x_3336_);
v___x_3338_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_orderArgs___closed__1));
lean_inc(v___x_3337_);
v___x_3339_ = l_Lean_Syntax_isOfKind(v___x_3337_, v___x_3338_);
if (v___x_3339_ == 0)
{
lean_object* v___x_3378_; 
lean_dec(v___x_3337_);
lean_dec(v_x_3322_);
v___x_3378_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__0___redArg();
return v___x_3378_;
}
else
{
lean_object* v___x_3379_; uint8_t v___x_3380_; 
v___x_3379_ = l_Lean_Syntax_getArg(v___x_3337_, v___x_3332_);
v___x_3380_ = l_Lean_Syntax_isNone(v___x_3379_);
if (v___x_3380_ == 0)
{
uint8_t v___x_3381_; 
lean_inc(v___x_3379_);
v___x_3381_ = l_Lean_Syntax_matchesNull(v___x_3379_, v___x_3336_);
if (v___x_3381_ == 0)
{
lean_object* v___x_3382_; 
lean_dec(v___x_3379_);
lean_dec(v___x_3337_);
lean_dec(v_x_3322_);
v___x_3382_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__0___redArg();
return v___x_3382_;
}
else
{
lean_object* v_o_3383_; lean_object* v___x_3384_; 
v_o_3383_ = l_Lean_Syntax_getArg(v___x_3379_, v___x_3332_);
lean_dec(v___x_3379_);
v___x_3384_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3384_, 0, v_o_3383_);
v_o_3360_ = v___x_3384_;
v___y_3361_ = v_a_3323_;
v___y_3362_ = v_a_3324_;
v___y_3363_ = v_a_3325_;
v___y_3364_ = v_a_3326_;
v___y_3365_ = v_a_3327_;
v___y_3366_ = v_a_3328_;
v___y_3367_ = v_a_3329_;
v___y_3368_ = v_a_3330_;
goto v___jp_3359_;
}
}
else
{
lean_object* v___x_3385_; 
lean_dec(v___x_3379_);
v___x_3385_ = lean_box(0);
v_o_3360_ = v___x_3385_;
v___y_3361_ = v_a_3323_;
v___y_3362_ = v_a_3324_;
v___y_3363_ = v_a_3325_;
v___y_3364_ = v_a_3326_;
v___y_3365_ = v_a_3327_;
v___y_3366_ = v_a_3328_;
v___y_3367_ = v_a_3329_;
v___y_3368_ = v_a_3330_;
goto v___jp_3359_;
}
}
v___jp_3340_:
{
lean_object* v___x_3351_; lean_object* v___x_3352_; lean_object* v___x_3353_; uint8_t v___x_3354_; lean_object* v___x_3355_; lean_object* v___x_3356_; lean_object* v___f_3357_; lean_object* v___x_3358_; 
v___x_3351_ = lean_unsigned_to_nat(2u);
v___x_3352_ = l_Lean_Syntax_getArg(v_x_3322_, v___x_3351_);
lean_dec(v_x_3322_);
v___x_3353_ = lean_box(0);
v___x_3354_ = 0;
v___x_3355_ = lean_box(v___x_3354_);
v___x_3356_ = lean_box(v___x_3339_);
v___f_3357_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1___lam__2___boxed), 16, 7);
lean_closure_set(v___f_3357_, 0, v___x_3352_);
lean_closure_set(v___f_3357_, 1, v___x_3353_);
lean_closure_set(v___f_3357_, 2, v___x_3355_);
lean_closure_set(v___f_3357_, 3, v___y_3347_);
lean_closure_set(v___f_3357_, 4, v___x_3356_);
lean_closure_set(v___f_3357_, 5, v_args_3350_);
lean_closure_set(v___f_3357_, 6, v___x_3332_);
v___x_3358_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_3357_, v___y_3348_, v___y_3342_, v___y_3341_, v___y_3346_, v___y_3345_, v___y_3344_, v___y_3343_, v___y_3349_);
return v___x_3358_;
}
v___jp_3359_:
{
lean_object* v___x_3369_; uint8_t v___x_3370_; 
v___x_3369_ = l_Lean_Syntax_getArg(v___x_3337_, v___x_3336_);
lean_dec(v___x_3337_);
v___x_3370_ = l_Lean_Syntax_isNone(v___x_3369_);
if (v___x_3370_ == 0)
{
lean_object* v___x_3371_; uint8_t v___x_3372_; 
v___x_3371_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_3369_);
v___x_3372_ = l_Lean_Syntax_matchesNull(v___x_3369_, v___x_3371_);
if (v___x_3372_ == 0)
{
lean_object* v___x_3373_; 
lean_dec(v___x_3369_);
lean_dec(v_o_3360_);
lean_dec(v_x_3322_);
v___x_3373_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1_spec__0___redArg();
return v___x_3373_;
}
else
{
lean_object* v___x_3374_; lean_object* v_args_3375_; lean_object* v___x_3376_; 
v___x_3374_ = l_Lean_Syntax_getArg(v___x_3369_, v___x_3336_);
lean_dec(v___x_3369_);
v_args_3375_ = l_Lean_Syntax_getArgs(v___x_3374_);
lean_dec(v___x_3374_);
v___x_3376_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3376_, 0, v_args_3375_);
v___y_3341_ = v___y_3363_;
v___y_3342_ = v___y_3362_;
v___y_3343_ = v___y_3367_;
v___y_3344_ = v___y_3366_;
v___y_3345_ = v___y_3365_;
v___y_3346_ = v___y_3364_;
v___y_3347_ = v_o_3360_;
v___y_3348_ = v___y_3361_;
v___y_3349_ = v___y_3368_;
v_args_3350_ = v___x_3376_;
goto v___jp_3340_;
}
}
else
{
lean_object* v___x_3377_; 
lean_dec(v___x_3369_);
v___x_3377_ = lean_box(0);
v___y_3341_ = v___y_3363_;
v___y_3342_ = v___y_3362_;
v___y_3343_ = v___y_3367_;
v___y_3344_ = v___y_3366_;
v___y_3345_ = v___y_3365_;
v___y_3346_ = v___y_3364_;
v___y_3347_ = v_o_3360_;
v___y_3348_ = v___y_3361_;
v___y_3349_ = v___y_3368_;
v_args_3350_ = v___x_3377_;
goto v___jp_3340_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1___boxed(lean_object* v_x_3386_, lean_object* v_a_3387_, lean_object* v_a_3388_, lean_object* v_a_3389_, lean_object* v_a_3390_, lean_object* v_a_3391_, lean_object* v_a_3392_, lean_object* v_a_3393_, lean_object* v_a_3394_, lean_object* v_a_3395_){
_start:
{
lean_object* v_res_3396_; 
v_res_3396_ = lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______elabRules____private__Mathlib__Tactic__Order__0__Mathlib__Tactic__Order__order__core__1(v_x_3386_, v_a_3387_, v_a_3388_, v_a_3389_, v_a_3390_, v_a_3391_, v_a_3392_, v_a_3393_, v_a_3394_);
lean_dec(v_a_3394_);
lean_dec_ref(v_a_3393_);
lean_dec(v_a_3392_);
lean_dec_ref(v_a_3391_);
lean_dec(v_a_3390_);
lean_dec_ref(v_a_3389_);
lean_dec(v_a_3388_);
lean_dec_ref(v_a_3387_);
return v_res_3396_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__15(void){
_start:
{
lean_object* v___x_3447_; 
v___x_3447_ = l_Array_mkArray0(lean_box(0));
return v___x_3447_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__28(void){
_start:
{
lean_object* v___x_3477_; lean_object* v___x_3478_; 
v___x_3477_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__27));
v___x_3478_ = l_String_toRawSubstring_x27(v___x_3477_);
return v___x_3478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1(lean_object* v_x_3481_, lean_object* v_a_3482_, lean_object* v_a_3483_){
_start:
{
lean_object* v___x_3484_; uint8_t v___x_3485_; 
v___x_3484_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_tacticOrder___00__closed__1));
lean_inc(v_x_3481_);
v___x_3485_ = l_Lean_Syntax_isOfKind(v_x_3481_, v___x_3484_);
if (v___x_3485_ == 0)
{
lean_object* v___x_3486_; lean_object* v___x_3487_; 
lean_dec(v_x_3481_);
v___x_3486_ = lean_box(1);
v___x_3487_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3487_, 0, v___x_3486_);
lean_ctor_set(v___x_3487_, 1, v_a_3483_);
return v___x_3487_;
}
else
{
lean_object* v_quotContext_3488_; lean_object* v_currMacroScope_3489_; lean_object* v_ref_3490_; lean_object* v___x_3491_; lean_object* v___x_3492_; uint8_t v___x_3493_; lean_object* v___x_3494_; lean_object* v___x_3495_; lean_object* v___x_3496_; lean_object* v___x_3497_; lean_object* v___x_3498_; lean_object* v___x_3499_; lean_object* v___x_3500_; lean_object* v___x_3501_; lean_object* v___x_3502_; lean_object* v___x_3503_; lean_object* v___x_3504_; lean_object* v___x_3505_; lean_object* v___x_3506_; lean_object* v___x_3507_; lean_object* v___x_3508_; lean_object* v___x_3509_; lean_object* v___x_3510_; lean_object* v___x_3511_; lean_object* v___x_3512_; lean_object* v___x_3513_; lean_object* v___x_3514_; lean_object* v___x_3515_; lean_object* v___x_3516_; lean_object* v___x_3517_; lean_object* v___x_3518_; lean_object* v___x_3519_; lean_object* v___x_3520_; lean_object* v___x_3521_; lean_object* v___x_3522_; lean_object* v___x_3523_; lean_object* v___x_3524_; lean_object* v___x_3525_; lean_object* v___x_3526_; lean_object* v___x_3527_; lean_object* v___x_3528_; lean_object* v___x_3529_; lean_object* v___x_3530_; lean_object* v___x_3531_; lean_object* v___x_3532_; lean_object* v___x_3533_; lean_object* v___x_3534_; 
v_quotContext_3488_ = lean_ctor_get(v_a_3482_, 1);
v_currMacroScope_3489_ = lean_ctor_get(v_a_3482_, 2);
v_ref_3490_ = lean_ctor_get(v_a_3482_, 5);
v___x_3491_ = lean_unsigned_to_nat(1u);
v___x_3492_ = l_Lean_Syntax_getArg(v_x_3481_, v___x_3491_);
lean_dec(v_x_3481_);
v___x_3493_ = 0;
v___x_3494_ = l_Lean_SourceInfo_fromRef(v_ref_3490_, v___x_3493_);
v___x_3495_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__2));
v___x_3496_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__4));
v___x_3497_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__5));
lean_inc_n(v___x_3494_, 18);
v___x_3498_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3498_, 0, v___x_3494_);
lean_ctor_set(v___x_3498_, 1, v___x_3497_);
v___x_3499_ = l_Lean_Syntax_node1(v___x_3494_, v___x_3496_, v___x_3498_);
v___x_3500_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__8));
v___x_3501_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__10));
v___x_3502_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__12));
v___x_3503_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__13));
v___x_3504_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__14));
v___x_3505_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3505_, 0, v___x_3494_);
lean_ctor_set(v___x_3505_, 1, v___x_3503_);
v___x_3506_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__15, &lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__15_once, _init_lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__15);
v___x_3507_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3507_, 0, v___x_3494_);
lean_ctor_set(v___x_3507_, 1, v___x_3502_);
lean_ctor_set(v___x_3507_, 2, v___x_3506_);
lean_inc_ref_n(v___x_3507_, 4);
v___x_3508_ = l_Lean_Syntax_node2(v___x_3494_, v___x_3504_, v___x_3505_, v___x_3507_);
v___x_3509_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__18));
v___x_3510_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__19));
v___x_3511_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3511_, 0, v___x_3494_);
lean_ctor_set(v___x_3511_, 1, v___x_3510_);
v___x_3512_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__21));
v___x_3513_ = l_Lean_Syntax_node1(v___x_3494_, v___x_3512_, v___x_3507_);
v___x_3514_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__23));
v___x_3515_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__26));
v___x_3516_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__28, &lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__28_once, _init_lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__28);
v___x_3517_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___closed__29));
lean_inc(v_currMacroScope_3489_);
lean_inc(v_quotContext_3488_);
v___x_3518_ = l_Lean_addMacroScope(v_quotContext_3488_, v___x_3517_, v_currMacroScope_3489_);
v___x_3519_ = lean_box(0);
v___x_3520_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_3520_, 0, v___x_3494_);
lean_ctor_set(v___x_3520_, 1, v___x_3516_);
lean_ctor_set(v___x_3520_, 2, v___x_3518_);
lean_ctor_set(v___x_3520_, 3, v___x_3519_);
lean_inc_ref(v___x_3520_);
v___x_3521_ = l_Lean_Syntax_node1(v___x_3494_, v___x_3515_, v___x_3520_);
v___x_3522_ = l_Lean_Syntax_node1(v___x_3494_, v___x_3502_, v___x_3521_);
v___x_3523_ = l_Lean_Syntax_node1(v___x_3494_, v___x_3514_, v___x_3522_);
v___x_3524_ = l_Lean_Syntax_node1(v___x_3494_, v___x_3502_, v___x_3523_);
v___x_3525_ = l_Lean_Syntax_node4(v___x_3494_, v___x_3509_, v___x_3511_, v___x_3513_, v___x_3524_, v___x_3507_);
v___x_3526_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__0));
v___x_3527_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_order__core___closed__1));
v___x_3528_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3528_, 0, v___x_3494_);
lean_ctor_set(v___x_3528_, 1, v___x_3526_);
v___x_3529_ = l_Lean_Syntax_node3(v___x_3494_, v___x_3527_, v___x_3528_, v___x_3492_, v___x_3520_);
v___x_3530_ = l_Lean_Syntax_node5(v___x_3494_, v___x_3502_, v___x_3508_, v___x_3507_, v___x_3525_, v___x_3507_, v___x_3529_);
v___x_3531_ = l_Lean_Syntax_node1(v___x_3494_, v___x_3501_, v___x_3530_);
v___x_3532_ = l_Lean_Syntax_node1(v___x_3494_, v___x_3500_, v___x_3531_);
v___x_3533_ = l_Lean_Syntax_node2(v___x_3494_, v___x_3495_, v___x_3499_, v___x_3532_);
v___x_3534_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3534_, 0, v___x_3533_);
lean_ctor_set(v___x_3534_, 1, v_a_3483_);
return v___x_3534_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1___boxed(lean_object* v_x_3535_, lean_object* v_a_3536_, lean_object* v_a_3537_){
_start:
{
lean_object* v_res_3538_; 
v_res_3538_ = lp_mathlib_Mathlib_Tactic_Order___aux__Mathlib__Tactic__Order______macroRules__Mathlib__Tactic__Order__tacticOrder____1(v_x_3535_, v_a_3536_, v_a_3537_);
lean_dec_ref(v_a_3536_);
return v_res_3538_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ByContra(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Order_Graph_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Order_Graph_Tarjan(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Order_Preprocessing(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Order_ToInt(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_ElabWithoutMVars(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Order(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ByContra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Order_Graph_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Order_Graph_Tarjan(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Order_Preprocessing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Order_ToInt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_ElabWithoutMVars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Omega(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Order_Graph_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Order(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Omega(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Order_Graph_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Order_0__Mathlib_Tactic_Order_initFn_00___x40_Mathlib_Tactic_Order_2841239826____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Omega(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Order_Graph_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ByContra(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Order_Graph_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Order_Graph_Tarjan(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Order_Preprocessing(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Order_ToInt(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_ElabWithoutMVars(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Order(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Omega(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Order_Graph_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ByContra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Order_Graph_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Order_Graph_Tarjan(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Order_Preprocessing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Order_ToInt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_ElabWithoutMVars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Order(builtin);
}
#ifdef __cplusplus
}
#endif
