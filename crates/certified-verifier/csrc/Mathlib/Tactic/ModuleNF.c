// Lean compiler output
// Module: Mathlib.Tactic.ModuleNF
// Imports: public import Init public meta import Init public import Mathlib.Tactic.Algebra.Basic public import Mathlib.Tactic.Module
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
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
extern lean_object* l_Lean_Parser_Tactic_location;
lean_object* l_Lean_FVarId_getType___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Meta_Simp_mkDefaultMethodsCore(lean_object*);
lean_object* l_Lean_Meta_Simp_main(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_Result_mkEqTrans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_empty(lean_object*);
lean_object* l_Lean_Meta_SimpTheorems_addConst(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_getSimpCongrTheorems___redArg(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_Meta_Simp_mkContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Qq_inferTypeQ_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* lp_Qq_Qq_synthInstanceQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Module_eval(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isApp(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* lean_whnf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Module_postprocessCtx(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_AtomM_recurse(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_transformAtNondepPropLocation(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_mkOptionalNode(lean_object*);
lean_object* l_Lean_Elab_Tactic_expandOptLocation(lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainTarget___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRings(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_foldl___at___00Array_appendList_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Expr_eqv___boxed(lean_object*, lean_object*);
lean_object* l_List_eraseDupsBy___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Qq_getLevelQ_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Level_ofNat(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Algebra_pickLargerRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_elabTerm(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_trySynthInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_List_eraseDups___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Expr_eqv___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_List_eraseDups___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__2___closed__0 = (const lean_object*)&lp_mathlib_List_eraseDups___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_eraseDups___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__2_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ModuleNF_inferBaseAtLocation_spec__0___redArg(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ModuleNF_inferBaseAtLocation_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation___lam__0___boxed, .m_arity = 10, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Elab_Tactic_getMainTarget___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ModuleNF_inferBaseAtLocation_spec__0(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ModuleNF_inferBaseAtLocation_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ModuleNF_evalExpr_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ModuleNF_evalExpr_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ModuleNF_evalExpr_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ModuleNF_evalExpr_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF_evalExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "AddCommMonoid"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_evalExpr___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_evalExpr___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_evalExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_evalExpr___closed__0_value),LEAN_SCALAR_PTR_LITERAL(159, 119, 180, 1, 34, 115, 27, 117)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_evalExpr___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_evalExpr___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF_evalExpr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_evalExpr___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_evalExpr___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ModuleNF_evalExpr___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_evalExpr___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_evalExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_evalExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ModuleNF_evalExpr_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ModuleNF_evalExpr_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__0___closed__0;
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__0(lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__1___closed__0;
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "one_smul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__6_value),LEAN_SCALAR_PTR_LITERAL(19, 96, 42, 60, 103, 53, 105, 117)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "zero_smul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__8_value),LEAN_SCALAR_PTR_LITERAL(192, 115, 222, 3, 254, 58, 217, 135)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "add_zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__10_value),LEAN_SCALAR_PTR_LITERAL(188, 217, 59, 250, 243, 223, 216, 213)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "zero_add"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__12_value),LEAN_SCALAR_PTR_LITERAL(81, 108, 255, 7, 24, 173, 4, 110)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "mul_one"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__14_value),LEAN_SCALAR_PTR_LITERAL(185, 178, 196, 247, 70, 46, 81, 207)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "one_mul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__16_value),LEAN_SCALAR_PTR_LITERAL(211, 55, 221, 12, 196, 98, 247, 238)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "neg_one_smul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__18_value),LEAN_SCALAR_PTR_LITERAL(211, 8, 60, 73, 159, 3, 104, 35)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "algebraMap_smul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__20_value),LEAN_SCALAR_PTR_LITERAL(128, 41, 46, 7, 239, 38, 2, 87)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__21_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__19_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__22_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__17_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__23_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__15_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__24_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__13_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__25_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__26_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__27_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__28_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 32, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(100000) << 1) | 1)),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 1, 0, 1, 1, 1, 0, 1),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 1, 0, 0, 0),LEAN_SCALAR_PTR_LITERAL(0, 1, 1, 1, 1, 1, 1, 1),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__30_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__6;
static const lean_array_object lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNFCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 8, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(3, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNFCore___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNFCore___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNFCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNFCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "ModuleNF"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "moduleNF"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__2_value),LEAN_SCALAR_PTR_LITERAL(174, 196, 226, 94, 16, 200, 184, 92)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__3_value),LEAN_SCALAR_PTR_LITERAL(181, 43, 140, 37, 9, 78, 112, 27)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__5_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "module_nf"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__7_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__9_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = " with "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__13_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__14_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__12_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__10_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__18_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__19;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__20;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__21;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Semiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(37, 127, 172, 14, 25, 240, 239, 179)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "module_nf failed: "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = " is not a semiring"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "location"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(124, 82, 43, 228, 241, 102, 135, 24)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_eraseDups___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__2(lean_object* v_as_2_){
_start:
{
lean_object* v___f_3_; lean_object* v___x_4_; 
v___f_3_ = ((lean_object*)(lp_mathlib_List_eraseDups___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__2___closed__0));
v___x_4_ = l_List_eraseDupsBy___redArg(v___f_3_, v_as_2_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__0(lean_object* v_x_5_, lean_object* v_x_6_, lean_object* v___y_7_, lean_object* v___y_8_, lean_object* v___y_9_, lean_object* v___y_10_){
_start:
{
if (lean_obj_tag(v_x_5_) == 0)
{
lean_object* v___x_12_; lean_object* v___x_13_; 
v___x_12_ = l_List_reverse___redArg(v_x_6_);
v___x_13_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_13_, 0, v___x_12_);
return v___x_13_;
}
else
{
lean_object* v_head_14_; lean_object* v_tail_15_; lean_object* v___x_17_; uint8_t v_isShared_18_; uint8_t v_isSharedCheck_33_; 
v_head_14_ = lean_ctor_get(v_x_5_, 0);
v_tail_15_ = lean_ctor_get(v_x_5_, 1);
v_isSharedCheck_33_ = !lean_is_exclusive(v_x_5_);
if (v_isSharedCheck_33_ == 0)
{
v___x_17_ = v_x_5_;
v_isShared_18_ = v_isSharedCheck_33_;
goto v_resetjp_16_;
}
else
{
lean_inc(v_tail_15_);
lean_inc(v_head_14_);
lean_dec(v_x_5_);
v___x_17_ = lean_box(0);
v_isShared_18_ = v_isSharedCheck_33_;
goto v_resetjp_16_;
}
v_resetjp_16_:
{
lean_object* v___x_19_; 
v___x_19_ = lp_mathlib_Mathlib_Tactic_Algebra_collectScalarRings(v_head_14_, v___y_7_, v___y_8_, v___y_9_, v___y_10_);
if (lean_obj_tag(v___x_19_) == 0)
{
lean_object* v_a_20_; lean_object* v___x_22_; 
v_a_20_ = lean_ctor_get(v___x_19_, 0);
lean_inc(v_a_20_);
lean_dec_ref_known(v___x_19_, 1);
if (v_isShared_18_ == 0)
{
lean_ctor_set(v___x_17_, 1, v_x_6_);
lean_ctor_set(v___x_17_, 0, v_a_20_);
v___x_22_ = v___x_17_;
goto v_reusejp_21_;
}
else
{
lean_object* v_reuseFailAlloc_24_; 
v_reuseFailAlloc_24_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_24_, 0, v_a_20_);
lean_ctor_set(v_reuseFailAlloc_24_, 1, v_x_6_);
v___x_22_ = v_reuseFailAlloc_24_;
goto v_reusejp_21_;
}
v_reusejp_21_:
{
v_x_5_ = v_tail_15_;
v_x_6_ = v___x_22_;
goto _start;
}
}
else
{
lean_object* v_a_25_; lean_object* v___x_27_; uint8_t v_isShared_28_; uint8_t v_isSharedCheck_32_; 
lean_del_object(v___x_17_);
lean_dec(v_tail_15_);
lean_dec(v_x_6_);
v_a_25_ = lean_ctor_get(v___x_19_, 0);
v_isSharedCheck_32_ = !lean_is_exclusive(v___x_19_);
if (v_isSharedCheck_32_ == 0)
{
v___x_27_ = v___x_19_;
v_isShared_28_ = v_isSharedCheck_32_;
goto v_resetjp_26_;
}
else
{
lean_inc(v_a_25_);
lean_dec(v___x_19_);
v___x_27_ = lean_box(0);
v_isShared_28_ = v_isSharedCheck_32_;
goto v_resetjp_26_;
}
v_resetjp_26_:
{
lean_object* v___x_30_; 
if (v_isShared_28_ == 0)
{
v___x_30_ = v___x_27_;
goto v_reusejp_29_;
}
else
{
lean_object* v_reuseFailAlloc_31_; 
v_reuseFailAlloc_31_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_31_, 0, v_a_25_);
v___x_30_ = v_reuseFailAlloc_31_;
goto v_reusejp_29_;
}
v_reusejp_29_:
{
return v___x_30_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__0___boxed(lean_object* v_x_34_, lean_object* v_x_35_, lean_object* v___y_36_, lean_object* v___y_37_, lean_object* v___y_38_, lean_object* v___y_39_, lean_object* v___y_40_){
_start:
{
lean_object* v_res_41_; 
v_res_41_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__0(v_x_34_, v_x_35_, v___y_36_, v___y_37_, v___y_38_, v___y_39_);
lean_dec(v___y_39_);
lean_dec_ref(v___y_38_);
lean_dec(v___y_37_);
lean_dec_ref(v___y_36_);
return v_res_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__1(lean_object* v_a_42_, lean_object* v_a_43_){
_start:
{
if (lean_obj_tag(v_a_42_) == 0)
{
lean_object* v___x_44_; 
v___x_44_ = lean_array_to_list(v_a_43_);
return v___x_44_;
}
else
{
lean_object* v_head_45_; lean_object* v_tail_46_; lean_object* v___x_47_; 
v_head_45_ = lean_ctor_get(v_a_42_, 0);
lean_inc(v_head_45_);
v_tail_46_ = lean_ctor_get(v_a_42_, 1);
lean_inc(v_tail_46_);
lean_dec_ref_known(v_a_42_, 2);
v___x_47_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_43_, v_head_45_);
v_a_42_ = v_tail_46_;
v_a_43_ = v___x_47_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__3(lean_object* v_x_49_, lean_object* v_x_50_, lean_object* v___y_51_, lean_object* v___y_52_, lean_object* v___y_53_, lean_object* v___y_54_){
_start:
{
if (lean_obj_tag(v_x_49_) == 0)
{
lean_object* v___x_56_; lean_object* v___x_57_; 
v___x_56_ = l_List_reverse___redArg(v_x_50_);
v___x_57_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_57_, 0, v___x_56_);
return v___x_57_;
}
else
{
lean_object* v_head_58_; lean_object* v_tail_59_; lean_object* v___x_61_; uint8_t v_isShared_62_; uint8_t v_isSharedCheck_77_; 
v_head_58_ = lean_ctor_get(v_x_49_, 0);
v_tail_59_ = lean_ctor_get(v_x_49_, 1);
v_isSharedCheck_77_ = !lean_is_exclusive(v_x_49_);
if (v_isSharedCheck_77_ == 0)
{
v___x_61_ = v_x_49_;
v_isShared_62_ = v_isSharedCheck_77_;
goto v_resetjp_60_;
}
else
{
lean_inc(v_tail_59_);
lean_inc(v_head_58_);
lean_dec(v_x_49_);
v___x_61_ = lean_box(0);
v_isShared_62_ = v_isSharedCheck_77_;
goto v_resetjp_60_;
}
v_resetjp_60_:
{
lean_object* v___x_63_; 
v___x_63_ = lp_mathlib_Qq_getLevelQ_x27(v_head_58_, v___y_51_, v___y_52_, v___y_53_, v___y_54_);
if (lean_obj_tag(v___x_63_) == 0)
{
lean_object* v_a_64_; lean_object* v___x_66_; 
v_a_64_ = lean_ctor_get(v___x_63_, 0);
lean_inc(v_a_64_);
lean_dec_ref_known(v___x_63_, 1);
if (v_isShared_62_ == 0)
{
lean_ctor_set(v___x_61_, 1, v_x_50_);
lean_ctor_set(v___x_61_, 0, v_a_64_);
v___x_66_ = v___x_61_;
goto v_reusejp_65_;
}
else
{
lean_object* v_reuseFailAlloc_68_; 
v_reuseFailAlloc_68_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_68_, 0, v_a_64_);
lean_ctor_set(v_reuseFailAlloc_68_, 1, v_x_50_);
v___x_66_ = v_reuseFailAlloc_68_;
goto v_reusejp_65_;
}
v_reusejp_65_:
{
v_x_49_ = v_tail_59_;
v_x_50_ = v___x_66_;
goto _start;
}
}
else
{
lean_object* v_a_69_; lean_object* v___x_71_; uint8_t v_isShared_72_; uint8_t v_isSharedCheck_76_; 
lean_del_object(v___x_61_);
lean_dec(v_tail_59_);
lean_dec(v_x_50_);
v_a_69_ = lean_ctor_get(v___x_63_, 0);
v_isSharedCheck_76_ = !lean_is_exclusive(v___x_63_);
if (v_isSharedCheck_76_ == 0)
{
v___x_71_ = v___x_63_;
v_isShared_72_ = v_isSharedCheck_76_;
goto v_resetjp_70_;
}
else
{
lean_inc(v_a_69_);
lean_dec(v___x_63_);
v___x_71_ = lean_box(0);
v_isShared_72_ = v_isSharedCheck_76_;
goto v_resetjp_70_;
}
v_resetjp_70_:
{
lean_object* v___x_74_; 
if (v_isShared_72_ == 0)
{
v___x_74_ = v___x_71_;
goto v_reusejp_73_;
}
else
{
lean_object* v_reuseFailAlloc_75_; 
v_reuseFailAlloc_75_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_75_, 0, v_a_69_);
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
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__3___boxed(lean_object* v_x_78_, lean_object* v_x_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_, lean_object* v___y_83_, lean_object* v___y_84_){
_start:
{
lean_object* v_res_85_; 
v_res_85_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__3(v_x_78_, v_x_79_, v___y_80_, v___y_81_, v___y_82_, v___y_83_);
lean_dec(v___y_83_);
lean_dec_ref(v___y_82_);
lean_dec(v___y_81_);
lean_dec_ref(v___y_80_);
return v_res_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__4(lean_object* v_x_86_, lean_object* v_x_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_){
_start:
{
if (lean_obj_tag(v_x_87_) == 0)
{
lean_object* v___x_93_; 
v___x_93_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_93_, 0, v_x_86_);
return v___x_93_;
}
else
{
lean_object* v_head_94_; lean_object* v_tail_95_; lean_object* v___x_96_; 
v_head_94_ = lean_ctor_get(v_x_87_, 0);
lean_inc(v_head_94_);
v_tail_95_ = lean_ctor_get(v_x_87_, 1);
lean_inc(v_tail_95_);
lean_dec_ref_known(v_x_87_, 2);
v___x_96_ = lp_mathlib_Mathlib_Tactic_Algebra_pickLargerRing(v_x_86_, v_head_94_, v___y_88_, v___y_89_, v___y_90_, v___y_91_);
if (lean_obj_tag(v___x_96_) == 0)
{
lean_object* v_a_97_; 
v_a_97_ = lean_ctor_get(v___x_96_, 0);
lean_inc(v_a_97_);
lean_dec_ref_known(v___x_96_, 1);
v_x_86_ = v_a_97_;
v_x_87_ = v_tail_95_;
goto _start;
}
else
{
lean_dec(v_tail_95_);
return v___x_96_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__4___boxed(lean_object* v_x_99_, lean_object* v_x_100_, lean_object* v___y_101_, lean_object* v___y_102_, lean_object* v___y_103_, lean_object* v___y_104_, lean_object* v___y_105_){
_start:
{
lean_object* v_res_106_; 
v_res_106_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__4(v_x_99_, v_x_100_, v___y_101_, v___y_102_, v___y_103_, v___y_104_);
lean_dec(v___y_104_);
lean_dec_ref(v___y_103_);
lean_dec(v___y_102_);
lean_dec_ref(v___y_101_);
return v_res_106_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__1(void){
_start:
{
lean_object* v___x_109_; lean_object* v___x_110_; 
v___x_109_ = lean_unsigned_to_nat(0u);
v___x_110_ = l_Lean_Level_ofNat(v___x_109_);
return v___x_110_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__4(void){
_start:
{
lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; 
v___x_114_ = lean_box(0);
v___x_115_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__3));
v___x_116_ = l_Lean_Expr_const___override(v___x_115_, v___x_114_);
return v___x_116_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__5(void){
_start:
{
lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; 
v___x_117_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__4, &lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__4);
v___x_118_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__1, &lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__1);
v___x_119_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_119_, 0, v___x_118_);
lean_ctor_set(v___x_119_, 1, v___x_117_);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase(lean_object* v_es_120_, lean_object* v_a_121_, lean_object* v_a_122_, lean_object* v_a_123_, lean_object* v_a_124_){
_start:
{
lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; 
v___x_126_ = lean_array_to_list(v_es_120_);
v___x_127_ = lean_box(0);
v___x_128_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__0(v___x_126_, v___x_127_, v_a_121_, v_a_122_, v_a_123_, v_a_124_);
if (lean_obj_tag(v___x_128_) == 0)
{
lean_object* v_a_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; 
v_a_129_ = lean_ctor_get(v___x_128_, 0);
lean_inc(v_a_129_);
lean_dec_ref_known(v___x_128_, 1);
v___x_130_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__0));
v___x_131_ = lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__1(v_a_129_, v___x_130_);
v___x_132_ = lp_mathlib_List_eraseDups___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__2(v___x_131_);
v___x_133_ = lp_mathlib_List_mapM_loop___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__3(v___x_132_, v___x_127_, v_a_121_, v_a_122_, v_a_123_, v_a_124_);
if (lean_obj_tag(v___x_133_) == 0)
{
lean_object* v_a_134_; lean_object* v___x_136_; uint8_t v_isShared_137_; uint8_t v_isSharedCheck_145_; 
v_a_134_ = lean_ctor_get(v___x_133_, 0);
v_isSharedCheck_145_ = !lean_is_exclusive(v___x_133_);
if (v_isSharedCheck_145_ == 0)
{
v___x_136_ = v___x_133_;
v_isShared_137_ = v_isSharedCheck_145_;
goto v_resetjp_135_;
}
else
{
lean_inc(v_a_134_);
lean_dec(v___x_133_);
v___x_136_ = lean_box(0);
v_isShared_137_ = v_isSharedCheck_145_;
goto v_resetjp_135_;
}
v_resetjp_135_:
{
if (lean_obj_tag(v_a_134_) == 0)
{
lean_object* v___x_138_; lean_object* v___x_140_; 
v___x_138_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__5, &lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___closed__5);
if (v_isShared_137_ == 0)
{
lean_ctor_set(v___x_136_, 0, v___x_138_);
v___x_140_ = v___x_136_;
goto v_reusejp_139_;
}
else
{
lean_object* v_reuseFailAlloc_141_; 
v_reuseFailAlloc_141_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_141_, 0, v___x_138_);
v___x_140_ = v_reuseFailAlloc_141_;
goto v_reusejp_139_;
}
v_reusejp_139_:
{
return v___x_140_;
}
}
else
{
lean_object* v_head_142_; lean_object* v_tail_143_; lean_object* v___x_144_; 
lean_del_object(v___x_136_);
v_head_142_ = lean_ctor_get(v_a_134_, 0);
lean_inc(v_head_142_);
v_tail_143_ = lean_ctor_get(v_a_134_, 1);
lean_inc(v_tail_143_);
lean_dec_ref_known(v_a_134_, 2);
v___x_144_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_ModuleNF_inferBase_spec__4(v_head_142_, v_tail_143_, v_a_121_, v_a_122_, v_a_123_, v_a_124_);
return v___x_144_;
}
}
}
else
{
lean_object* v_a_146_; lean_object* v___x_148_; uint8_t v_isShared_149_; uint8_t v_isSharedCheck_153_; 
v_a_146_ = lean_ctor_get(v___x_133_, 0);
v_isSharedCheck_153_ = !lean_is_exclusive(v___x_133_);
if (v_isSharedCheck_153_ == 0)
{
v___x_148_ = v___x_133_;
v_isShared_149_ = v_isSharedCheck_153_;
goto v_resetjp_147_;
}
else
{
lean_inc(v_a_146_);
lean_dec(v___x_133_);
v___x_148_ = lean_box(0);
v_isShared_149_ = v_isSharedCheck_153_;
goto v_resetjp_147_;
}
v_resetjp_147_:
{
lean_object* v___x_151_; 
if (v_isShared_149_ == 0)
{
v___x_151_ = v___x_148_;
goto v_reusejp_150_;
}
else
{
lean_object* v_reuseFailAlloc_152_; 
v_reuseFailAlloc_152_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_152_, 0, v_a_146_);
v___x_151_ = v_reuseFailAlloc_152_;
goto v_reusejp_150_;
}
v_reusejp_150_:
{
return v___x_151_;
}
}
}
}
else
{
lean_object* v_a_154_; lean_object* v___x_156_; uint8_t v_isShared_157_; uint8_t v_isSharedCheck_161_; 
v_a_154_ = lean_ctor_get(v___x_128_, 0);
v_isSharedCheck_161_ = !lean_is_exclusive(v___x_128_);
if (v_isSharedCheck_161_ == 0)
{
v___x_156_ = v___x_128_;
v_isShared_157_ = v_isSharedCheck_161_;
goto v_resetjp_155_;
}
else
{
lean_inc(v_a_154_);
lean_dec(v___x_128_);
v___x_156_ = lean_box(0);
v_isShared_157_ = v_isSharedCheck_161_;
goto v_resetjp_155_;
}
v_resetjp_155_:
{
lean_object* v___x_159_; 
if (v_isShared_157_ == 0)
{
v___x_159_ = v___x_156_;
goto v_reusejp_158_;
}
else
{
lean_object* v_reuseFailAlloc_160_; 
v_reuseFailAlloc_160_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_160_, 0, v_a_154_);
v___x_159_ = v_reuseFailAlloc_160_;
goto v_reusejp_158_;
}
v_reusejp_158_:
{
return v___x_159_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase___boxed(lean_object* v_es_162_, lean_object* v_a_163_, lean_object* v_a_164_, lean_object* v_a_165_, lean_object* v_a_166_, lean_object* v_a_167_){
_start:
{
lean_object* v_res_168_; 
v_res_168_ = lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase(v_es_162_, v_a_163_, v_a_164_, v_a_165_, v_a_166_);
lean_dec(v_a_166_);
lean_dec_ref(v_a_165_);
lean_dec(v_a_164_);
lean_dec_ref(v_a_163_);
return v_res_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation___lam__0(lean_object* v_fvarId_169_, lean_object* v___y_170_, lean_object* v___y_171_, lean_object* v___y_172_, lean_object* v___y_173_, lean_object* v___y_174_, lean_object* v___y_175_, lean_object* v___y_176_, lean_object* v___y_177_){
_start:
{
lean_object* v___x_179_; 
v___x_179_ = l_Lean_FVarId_getType___redArg(v_fvarId_169_, v___y_174_, v___y_176_, v___y_177_);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation___lam__0___boxed(lean_object* v_fvarId_180_, lean_object* v___y_181_, lean_object* v___y_182_, lean_object* v___y_183_, lean_object* v___y_184_, lean_object* v___y_185_, lean_object* v___y_186_, lean_object* v___y_187_, lean_object* v___y_188_, lean_object* v___y_189_){
_start:
{
lean_object* v_res_190_; 
v_res_190_ = lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation___lam__0(v_fvarId_180_, v___y_181_, v___y_182_, v___y_183_, v___y_184_, v___y_185_, v___y_186_, v___y_187_, v___y_188_);
lean_dec(v___y_188_);
lean_dec_ref(v___y_187_);
lean_dec(v___y_186_);
lean_dec_ref(v___y_185_);
lean_dec(v___y_184_);
lean_dec_ref(v___y_183_);
lean_dec(v___y_182_);
lean_dec_ref(v___y_181_);
return v_res_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ModuleNF_inferBaseAtLocation_spec__0___redArg(size_t v_sz_191_, size_t v_i_192_, lean_object* v_bs_193_, lean_object* v___y_194_, lean_object* v___y_195_, lean_object* v___y_196_, lean_object* v___y_197_){
_start:
{
uint8_t v___x_199_; 
v___x_199_ = lean_usize_dec_lt(v_i_192_, v_sz_191_);
if (v___x_199_ == 0)
{
lean_object* v___x_200_; 
v___x_200_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_200_, 0, v_bs_193_);
return v___x_200_;
}
else
{
lean_object* v_v_201_; lean_object* v___x_202_; 
v_v_201_ = lean_array_uget_borrowed(v_bs_193_, v_i_192_);
lean_inc(v___y_197_);
lean_inc_ref(v___y_196_);
lean_inc(v___y_195_);
lean_inc_ref(v___y_194_);
lean_inc(v_v_201_);
v___x_202_ = lean_whnf(v_v_201_, v___y_194_, v___y_195_, v___y_196_, v___y_197_);
if (lean_obj_tag(v___x_202_) == 0)
{
lean_object* v_a_203_; lean_object* v___x_204_; lean_object* v_bs_x27_205_; size_t v___x_206_; size_t v___x_207_; lean_object* v___x_208_; 
v_a_203_ = lean_ctor_get(v___x_202_, 0);
lean_inc(v_a_203_);
lean_dec_ref_known(v___x_202_, 1);
v___x_204_ = lean_unsigned_to_nat(0u);
v_bs_x27_205_ = lean_array_uset(v_bs_193_, v_i_192_, v___x_204_);
v___x_206_ = ((size_t)1ULL);
v___x_207_ = lean_usize_add(v_i_192_, v___x_206_);
v___x_208_ = lean_array_uset(v_bs_x27_205_, v_i_192_, v_a_203_);
v_i_192_ = v___x_207_;
v_bs_193_ = v___x_208_;
goto _start;
}
else
{
lean_object* v_a_210_; lean_object* v___x_212_; uint8_t v_isShared_213_; uint8_t v_isSharedCheck_217_; 
lean_dec_ref(v_bs_193_);
v_a_210_ = lean_ctor_get(v___x_202_, 0);
v_isSharedCheck_217_ = !lean_is_exclusive(v___x_202_);
if (v_isSharedCheck_217_ == 0)
{
v___x_212_ = v___x_202_;
v_isShared_213_ = v_isSharedCheck_217_;
goto v_resetjp_211_;
}
else
{
lean_inc(v_a_210_);
lean_dec(v___x_202_);
v___x_212_ = lean_box(0);
v_isShared_213_ = v_isSharedCheck_217_;
goto v_resetjp_211_;
}
v_resetjp_211_:
{
lean_object* v___x_215_; 
if (v_isShared_213_ == 0)
{
v___x_215_ = v___x_212_;
goto v_reusejp_214_;
}
else
{
lean_object* v_reuseFailAlloc_216_; 
v_reuseFailAlloc_216_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_216_, 0, v_a_210_);
v___x_215_ = v_reuseFailAlloc_216_;
goto v_reusejp_214_;
}
v_reusejp_214_:
{
return v___x_215_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ModuleNF_inferBaseAtLocation_spec__0___redArg___boxed(lean_object* v_sz_218_, lean_object* v_i_219_, lean_object* v_bs_220_, lean_object* v___y_221_, lean_object* v___y_222_, lean_object* v___y_223_, lean_object* v___y_224_, lean_object* v___y_225_){
_start:
{
size_t v_sz_boxed_226_; size_t v_i_boxed_227_; lean_object* v_res_228_; 
v_sz_boxed_226_ = lean_unbox_usize(v_sz_218_);
lean_dec(v_sz_218_);
v_i_boxed_227_ = lean_unbox_usize(v_i_219_);
lean_dec(v_i_219_);
v_res_228_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ModuleNF_inferBaseAtLocation_spec__0___redArg(v_sz_boxed_226_, v_i_boxed_227_, v_bs_220_, v___y_221_, v___y_222_, v___y_223_, v___y_224_);
lean_dec(v___y_224_);
lean_dec_ref(v___y_223_);
lean_dec(v___y_222_);
lean_dec_ref(v___y_221_);
return v_res_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation___lam__1(lean_object* v_loc_229_, lean_object* v___f_230_, lean_object* v___x_231_, lean_object* v___y_232_, lean_object* v___y_233_, lean_object* v___y_234_, lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_, lean_object* v___y_238_, lean_object* v___y_239_){
_start:
{
lean_object* v___x_241_; 
v___x_241_ = lp_mathlib_Lean_Elab_Tactic_mapNondepPropLocation___redArg(v_loc_229_, v___f_230_, v___x_231_, v___y_232_, v___y_233_, v___y_234_, v___y_235_, v___y_236_, v___y_237_, v___y_238_, v___y_239_);
if (lean_obj_tag(v___x_241_) == 0)
{
lean_object* v_a_242_; size_t v_sz_243_; size_t v___x_244_; lean_object* v___x_245_; 
v_a_242_ = lean_ctor_get(v___x_241_, 0);
lean_inc(v_a_242_);
lean_dec_ref_known(v___x_241_, 1);
v_sz_243_ = lean_array_size(v_a_242_);
v___x_244_ = ((size_t)0ULL);
v___x_245_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ModuleNF_inferBaseAtLocation_spec__0___redArg(v_sz_243_, v___x_244_, v_a_242_, v___y_236_, v___y_237_, v___y_238_, v___y_239_);
if (lean_obj_tag(v___x_245_) == 0)
{
lean_object* v_a_246_; lean_object* v___x_247_; 
v_a_246_ = lean_ctor_get(v___x_245_, 0);
lean_inc(v_a_246_);
lean_dec_ref_known(v___x_245_, 1);
v___x_247_ = lp_mathlib_Mathlib_Tactic_ModuleNF_inferBase(v_a_246_, v___y_236_, v___y_237_, v___y_238_, v___y_239_);
return v___x_247_;
}
else
{
lean_object* v_a_248_; lean_object* v___x_250_; uint8_t v_isShared_251_; uint8_t v_isSharedCheck_255_; 
v_a_248_ = lean_ctor_get(v___x_245_, 0);
v_isSharedCheck_255_ = !lean_is_exclusive(v___x_245_);
if (v_isSharedCheck_255_ == 0)
{
v___x_250_ = v___x_245_;
v_isShared_251_ = v_isSharedCheck_255_;
goto v_resetjp_249_;
}
else
{
lean_inc(v_a_248_);
lean_dec(v___x_245_);
v___x_250_ = lean_box(0);
v_isShared_251_ = v_isSharedCheck_255_;
goto v_resetjp_249_;
}
v_resetjp_249_:
{
lean_object* v___x_253_; 
if (v_isShared_251_ == 0)
{
v___x_253_ = v___x_250_;
goto v_reusejp_252_;
}
else
{
lean_object* v_reuseFailAlloc_254_; 
v_reuseFailAlloc_254_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_254_, 0, v_a_248_);
v___x_253_ = v_reuseFailAlloc_254_;
goto v_reusejp_252_;
}
v_reusejp_252_:
{
return v___x_253_;
}
}
}
}
else
{
lean_object* v_a_256_; lean_object* v___x_258_; uint8_t v_isShared_259_; uint8_t v_isSharedCheck_263_; 
v_a_256_ = lean_ctor_get(v___x_241_, 0);
v_isSharedCheck_263_ = !lean_is_exclusive(v___x_241_);
if (v_isSharedCheck_263_ == 0)
{
v___x_258_ = v___x_241_;
v_isShared_259_ = v_isSharedCheck_263_;
goto v_resetjp_257_;
}
else
{
lean_inc(v_a_256_);
lean_dec(v___x_241_);
v___x_258_ = lean_box(0);
v_isShared_259_ = v_isSharedCheck_263_;
goto v_resetjp_257_;
}
v_resetjp_257_:
{
lean_object* v___x_261_; 
if (v_isShared_259_ == 0)
{
v___x_261_ = v___x_258_;
goto v_reusejp_260_;
}
else
{
lean_object* v_reuseFailAlloc_262_; 
v_reuseFailAlloc_262_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_262_, 0, v_a_256_);
v___x_261_ = v_reuseFailAlloc_262_;
goto v_reusejp_260_;
}
v_reusejp_260_:
{
return v___x_261_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation___lam__1___boxed(lean_object* v_loc_264_, lean_object* v___f_265_, lean_object* v___x_266_, lean_object* v___y_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_, lean_object* v___y_272_, lean_object* v___y_273_, lean_object* v___y_274_, lean_object* v___y_275_){
_start:
{
lean_object* v_res_276_; 
v_res_276_ = lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation___lam__1(v_loc_264_, v___f_265_, v___x_266_, v___y_267_, v___y_268_, v___y_269_, v___y_270_, v___y_271_, v___y_272_, v___y_273_, v___y_274_);
lean_dec(v___y_274_);
lean_dec_ref(v___y_273_);
lean_dec(v___y_272_);
lean_dec_ref(v___y_271_);
lean_dec(v___y_270_);
lean_dec_ref(v___y_269_);
lean_dec(v___y_268_);
lean_dec_ref(v___y_267_);
lean_dec(v_loc_264_);
return v_res_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation(lean_object* v_loc_279_, lean_object* v_a_280_, lean_object* v_a_281_, lean_object* v_a_282_, lean_object* v_a_283_, lean_object* v_a_284_, lean_object* v_a_285_, lean_object* v_a_286_, lean_object* v_a_287_){
_start:
{
lean_object* v___f_289_; lean_object* v___x_290_; lean_object* v___f_291_; lean_object* v___x_292_; 
v___f_289_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation___closed__0));
v___x_290_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation___closed__1));
v___f_291_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation___lam__1___boxed), 12, 3);
lean_closure_set(v___f_291_, 0, v_loc_279_);
lean_closure_set(v___f_291_, 1, v___f_289_);
lean_closure_set(v___f_291_, 2, v___x_290_);
v___x_292_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_291_, v_a_280_, v_a_281_, v_a_282_, v_a_283_, v_a_284_, v_a_285_, v_a_286_, v_a_287_);
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation___boxed(lean_object* v_loc_293_, lean_object* v_a_294_, lean_object* v_a_295_, lean_object* v_a_296_, lean_object* v_a_297_, lean_object* v_a_298_, lean_object* v_a_299_, lean_object* v_a_300_, lean_object* v_a_301_, lean_object* v_a_302_){
_start:
{
lean_object* v_res_303_; 
v_res_303_ = lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation(v_loc_293_, v_a_294_, v_a_295_, v_a_296_, v_a_297_, v_a_298_, v_a_299_, v_a_300_, v_a_301_);
lean_dec(v_a_301_);
lean_dec_ref(v_a_300_);
lean_dec(v_a_299_);
lean_dec_ref(v_a_298_);
lean_dec(v_a_297_);
lean_dec_ref(v_a_296_);
lean_dec(v_a_295_);
lean_dec_ref(v_a_294_);
return v_res_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ModuleNF_inferBaseAtLocation_spec__0(size_t v_sz_304_, size_t v_i_305_, lean_object* v_bs_306_, lean_object* v___y_307_, lean_object* v___y_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_){
_start:
{
lean_object* v___x_316_; 
v___x_316_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ModuleNF_inferBaseAtLocation_spec__0___redArg(v_sz_304_, v_i_305_, v_bs_306_, v___y_311_, v___y_312_, v___y_313_, v___y_314_);
return v___x_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ModuleNF_inferBaseAtLocation_spec__0___boxed(lean_object* v_sz_317_, lean_object* v_i_318_, lean_object* v_bs_319_, lean_object* v___y_320_, lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_, lean_object* v___y_324_, lean_object* v___y_325_, lean_object* v___y_326_, lean_object* v___y_327_, lean_object* v___y_328_){
_start:
{
size_t v_sz_boxed_329_; size_t v_i_boxed_330_; lean_object* v_res_331_; 
v_sz_boxed_329_ = lean_unbox_usize(v_sz_317_);
lean_dec(v_sz_317_);
v_i_boxed_330_ = lean_unbox_usize(v_i_318_);
lean_dec(v_i_318_);
v_res_331_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_ModuleNF_inferBaseAtLocation_spec__0(v_sz_boxed_329_, v_i_boxed_330_, v_bs_319_, v___y_320_, v___y_321_, v___y_322_, v___y_323_, v___y_324_, v___y_325_, v___y_326_, v___y_327_);
lean_dec(v___y_327_);
lean_dec_ref(v___y_326_);
lean_dec(v___y_325_);
lean_dec_ref(v___y_324_);
lean_dec(v___y_323_);
lean_dec_ref(v___y_322_);
lean_dec(v___y_321_);
lean_dec_ref(v___y_320_);
return v_res_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ModuleNF_evalExpr_spec__0_spec__0(lean_object* v_msgData_332_, lean_object* v___y_333_, lean_object* v___y_334_, lean_object* v___y_335_, lean_object* v___y_336_){
_start:
{
lean_object* v___x_338_; lean_object* v_env_339_; lean_object* v___x_340_; lean_object* v_mctx_341_; lean_object* v_lctx_342_; lean_object* v_options_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; 
v___x_338_ = lean_st_ref_get(v___y_336_);
v_env_339_ = lean_ctor_get(v___x_338_, 0);
lean_inc_ref(v_env_339_);
lean_dec(v___x_338_);
v___x_340_ = lean_st_ref_get(v___y_334_);
v_mctx_341_ = lean_ctor_get(v___x_340_, 0);
lean_inc_ref(v_mctx_341_);
lean_dec(v___x_340_);
v_lctx_342_ = lean_ctor_get(v___y_333_, 2);
v_options_343_ = lean_ctor_get(v___y_335_, 2);
lean_inc_ref(v_options_343_);
lean_inc_ref(v_lctx_342_);
v___x_344_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_344_, 0, v_env_339_);
lean_ctor_set(v___x_344_, 1, v_mctx_341_);
lean_ctor_set(v___x_344_, 2, v_lctx_342_);
lean_ctor_set(v___x_344_, 3, v_options_343_);
v___x_345_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_345_, 0, v___x_344_);
lean_ctor_set(v___x_345_, 1, v_msgData_332_);
v___x_346_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_346_, 0, v___x_345_);
return v___x_346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ModuleNF_evalExpr_spec__0_spec__0___boxed(lean_object* v_msgData_347_, lean_object* v___y_348_, lean_object* v___y_349_, lean_object* v___y_350_, lean_object* v___y_351_, lean_object* v___y_352_){
_start:
{
lean_object* v_res_353_; 
v_res_353_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ModuleNF_evalExpr_spec__0_spec__0(v_msgData_347_, v___y_348_, v___y_349_, v___y_350_, v___y_351_);
lean_dec(v___y_351_);
lean_dec_ref(v___y_350_);
lean_dec(v___y_349_);
lean_dec_ref(v___y_348_);
return v_res_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ModuleNF_evalExpr_spec__0___redArg(lean_object* v_msg_354_, lean_object* v___y_355_, lean_object* v___y_356_, lean_object* v___y_357_, lean_object* v___y_358_){
_start:
{
lean_object* v_ref_360_; lean_object* v___x_361_; lean_object* v_a_362_; lean_object* v___x_364_; uint8_t v_isShared_365_; uint8_t v_isSharedCheck_370_; 
v_ref_360_ = lean_ctor_get(v___y_357_, 5);
v___x_361_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ModuleNF_evalExpr_spec__0_spec__0(v_msg_354_, v___y_355_, v___y_356_, v___y_357_, v___y_358_);
v_a_362_ = lean_ctor_get(v___x_361_, 0);
v_isSharedCheck_370_ = !lean_is_exclusive(v___x_361_);
if (v_isSharedCheck_370_ == 0)
{
v___x_364_ = v___x_361_;
v_isShared_365_ = v_isSharedCheck_370_;
goto v_resetjp_363_;
}
else
{
lean_inc(v_a_362_);
lean_dec(v___x_361_);
v___x_364_ = lean_box(0);
v_isShared_365_ = v_isSharedCheck_370_;
goto v_resetjp_363_;
}
v_resetjp_363_:
{
lean_object* v___x_366_; lean_object* v___x_368_; 
lean_inc(v_ref_360_);
v___x_366_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_366_, 0, v_ref_360_);
lean_ctor_set(v___x_366_, 1, v_a_362_);
if (v_isShared_365_ == 0)
{
lean_ctor_set_tag(v___x_364_, 1);
lean_ctor_set(v___x_364_, 0, v___x_366_);
v___x_368_ = v___x_364_;
goto v_reusejp_367_;
}
else
{
lean_object* v_reuseFailAlloc_369_; 
v_reuseFailAlloc_369_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_369_, 0, v___x_366_);
v___x_368_ = v_reuseFailAlloc_369_;
goto v_reusejp_367_;
}
v_reusejp_367_:
{
return v___x_368_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ModuleNF_evalExpr_spec__0___redArg___boxed(lean_object* v_msg_371_, lean_object* v___y_372_, lean_object* v___y_373_, lean_object* v___y_374_, lean_object* v___y_375_, lean_object* v___y_376_){
_start:
{
lean_object* v_res_377_; 
v_res_377_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ModuleNF_evalExpr_spec__0___redArg(v_msg_371_, v___y_372_, v___y_373_, v___y_374_, v___y_375_);
lean_dec(v___y_375_);
lean_dec_ref(v___y_374_);
lean_dec(v___y_373_);
lean_dec_ref(v___y_372_);
return v_res_377_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ModuleNF_evalExpr___closed__3(void){
_start:
{
lean_object* v___x_382_; lean_object* v___x_383_; 
v___x_382_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ModuleNF_evalExpr___closed__2));
v___x_383_ = l_Lean_stringToMessageData(v___x_382_);
return v___x_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_evalExpr(lean_object* v_base_384_, lean_object* v_postCtx_385_, lean_object* v_e_386_, lean_object* v_a_387_, lean_object* v_a_388_, lean_object* v_a_389_, lean_object* v_a_390_, lean_object* v_a_391_, lean_object* v_a_392_){
_start:
{
lean_object* v___y_395_; lean_object* v_a_434_; lean_object* v_keyedConfig_446_; uint8_t v_trackZetaDelta_447_; lean_object* v_zetaDeltaSet_448_; lean_object* v_lctx_449_; lean_object* v_localInstances_450_; lean_object* v_defEqCtx_x3f_451_; lean_object* v_synthPendingDepth_452_; lean_object* v_customCanUnfoldPredicate_x3f_453_; uint8_t v_univApprox_454_; uint8_t v_inTypeClassResolution_455_; uint8_t v_cacheInferType_456_; uint8_t v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; 
v_keyedConfig_446_ = lean_ctor_get(v_a_389_, 0);
v_trackZetaDelta_447_ = lean_ctor_get_uint8(v_a_389_, sizeof(void*)*7);
v_zetaDeltaSet_448_ = lean_ctor_get(v_a_389_, 1);
v_lctx_449_ = lean_ctor_get(v_a_389_, 2);
v_localInstances_450_ = lean_ctor_get(v_a_389_, 3);
v_defEqCtx_x3f_451_ = lean_ctor_get(v_a_389_, 4);
v_synthPendingDepth_452_ = lean_ctor_get(v_a_389_, 5);
v_customCanUnfoldPredicate_x3f_453_ = lean_ctor_get(v_a_389_, 6);
v_univApprox_454_ = lean_ctor_get_uint8(v_a_389_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_455_ = lean_ctor_get_uint8(v_a_389_, sizeof(void*)*7 + 2);
v_cacheInferType_456_ = lean_ctor_get_uint8(v_a_389_, sizeof(void*)*7 + 3);
v___x_457_ = 2;
lean_inc_ref(v_keyedConfig_446_);
v___x_458_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_457_, v_keyedConfig_446_);
lean_inc(v_customCanUnfoldPredicate_x3f_453_);
lean_inc(v_synthPendingDepth_452_);
lean_inc(v_defEqCtx_x3f_451_);
lean_inc_ref(v_localInstances_450_);
lean_inc_ref(v_lctx_449_);
lean_inc(v_zetaDeltaSet_448_);
v___x_459_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_459_, 0, v___x_458_);
lean_ctor_set(v___x_459_, 1, v_zetaDeltaSet_448_);
lean_ctor_set(v___x_459_, 2, v_lctx_449_);
lean_ctor_set(v___x_459_, 3, v_localInstances_450_);
lean_ctor_set(v___x_459_, 4, v_defEqCtx_x3f_451_);
lean_ctor_set(v___x_459_, 5, v_synthPendingDepth_452_);
lean_ctor_set(v___x_459_, 6, v_customCanUnfoldPredicate_x3f_453_);
lean_ctor_set_uint8(v___x_459_, sizeof(void*)*7, v_trackZetaDelta_447_);
lean_ctor_set_uint8(v___x_459_, sizeof(void*)*7 + 1, v_univApprox_454_);
lean_ctor_set_uint8(v___x_459_, sizeof(void*)*7 + 2, v_inTypeClassResolution_455_);
lean_ctor_set_uint8(v___x_459_, sizeof(void*)*7 + 3, v_cacheInferType_456_);
lean_inc(v_a_392_);
lean_inc_ref(v_a_391_);
lean_inc(v_a_390_);
v___x_460_ = lean_whnf(v_e_386_, v___x_459_, v_a_390_, v_a_391_, v_a_392_);
if (lean_obj_tag(v___x_460_) == 0)
{
lean_object* v_a_461_; 
v_a_461_ = lean_ctor_get(v___x_460_, 0);
lean_inc(v_a_461_);
lean_dec_ref_known(v___x_460_, 1);
v_a_434_ = v_a_461_;
goto v___jp_433_;
}
else
{
if (lean_obj_tag(v___x_460_) == 0)
{
lean_object* v_a_462_; 
v_a_462_ = lean_ctor_get(v___x_460_, 0);
lean_inc(v_a_462_);
lean_dec_ref_known(v___x_460_, 1);
v_a_434_ = v_a_462_;
goto v___jp_433_;
}
else
{
lean_object* v_a_463_; lean_object* v___x_465_; uint8_t v_isShared_466_; uint8_t v_isSharedCheck_470_; 
lean_dec_ref(v_postCtx_385_);
lean_dec_ref(v_base_384_);
v_a_463_ = lean_ctor_get(v___x_460_, 0);
v_isSharedCheck_470_ = !lean_is_exclusive(v___x_460_);
if (v_isSharedCheck_470_ == 0)
{
v___x_465_ = v___x_460_;
v_isShared_466_ = v_isSharedCheck_470_;
goto v_resetjp_464_;
}
else
{
lean_inc(v_a_463_);
lean_dec(v___x_460_);
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
v___jp_394_:
{
lean_object* v___x_396_; 
v___x_396_ = lp_mathlib_Qq_inferTypeQ_x27(v___y_395_, v_a_389_, v_a_390_, v_a_391_, v_a_392_);
if (lean_obj_tag(v___x_396_) == 0)
{
lean_object* v_a_397_; lean_object* v_snd_398_; lean_object* v_fst_399_; lean_object* v_fst_400_; lean_object* v_snd_401_; lean_object* v___x_403_; uint8_t v_isShared_404_; uint8_t v_isSharedCheck_424_; 
v_a_397_ = lean_ctor_get(v___x_396_, 0);
lean_inc(v_a_397_);
lean_dec_ref_known(v___x_396_, 1);
v_snd_398_ = lean_ctor_get(v_a_397_, 1);
lean_inc(v_snd_398_);
v_fst_399_ = lean_ctor_get(v_a_397_, 0);
lean_inc(v_fst_399_);
lean_dec(v_a_397_);
v_fst_400_ = lean_ctor_get(v_snd_398_, 0);
v_snd_401_ = lean_ctor_get(v_snd_398_, 1);
v_isSharedCheck_424_ = !lean_is_exclusive(v_snd_398_);
if (v_isSharedCheck_424_ == 0)
{
v___x_403_ = v_snd_398_;
v_isShared_404_ = v_isSharedCheck_424_;
goto v_resetjp_402_;
}
else
{
lean_inc(v_snd_401_);
lean_inc(v_fst_400_);
lean_dec(v_snd_398_);
v___x_403_ = lean_box(0);
v_isShared_404_ = v_isSharedCheck_424_;
goto v_resetjp_402_;
}
v_resetjp_402_:
{
lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_408_; 
v___x_405_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ModuleNF_evalExpr___closed__1));
v___x_406_ = lean_box(0);
lean_inc(v_fst_399_);
if (v_isShared_404_ == 0)
{
lean_ctor_set_tag(v___x_403_, 1);
lean_ctor_set(v___x_403_, 1, v___x_406_);
lean_ctor_set(v___x_403_, 0, v_fst_399_);
v___x_408_ = v___x_403_;
goto v_reusejp_407_;
}
else
{
lean_object* v_reuseFailAlloc_423_; 
v_reuseFailAlloc_423_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_423_, 0, v_fst_399_);
lean_ctor_set(v_reuseFailAlloc_423_, 1, v___x_406_);
v___x_408_ = v_reuseFailAlloc_423_;
goto v_reusejp_407_;
}
v_reusejp_407_:
{
lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; 
v___x_409_ = l_Lean_Expr_const___override(v___x_405_, v___x_408_);
lean_inc(v_fst_400_);
v___x_410_ = l_Lean_Expr_app___override(v___x_409_, v_fst_400_);
v___x_411_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_410_, v_a_389_, v_a_390_, v_a_391_, v_a_392_);
if (lean_obj_tag(v___x_411_) == 0)
{
lean_object* v_a_412_; lean_object* v___x_413_; lean_object* v___x_414_; 
v_a_412_ = lean_ctor_get(v___x_411_, 0);
lean_inc(v_a_412_);
lean_dec_ref_known(v___x_411_, 1);
v___x_413_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_413_, 0, v_base_384_);
v___x_414_ = lp_mathlib_Mathlib_Tactic_Module_eval(v_fst_399_, v_fst_400_, v_a_412_, v___x_413_, v_postCtx_385_, v_snd_401_, v_a_387_, v_a_388_, v_a_389_, v_a_390_, v_a_391_, v_a_392_);
return v___x_414_;
}
else
{
lean_object* v_a_415_; lean_object* v___x_417_; uint8_t v_isShared_418_; uint8_t v_isSharedCheck_422_; 
lean_dec(v_snd_401_);
lean_dec(v_fst_400_);
lean_dec(v_fst_399_);
lean_dec_ref(v_postCtx_385_);
lean_dec_ref(v_base_384_);
v_a_415_ = lean_ctor_get(v___x_411_, 0);
v_isSharedCheck_422_ = !lean_is_exclusive(v___x_411_);
if (v_isSharedCheck_422_ == 0)
{
v___x_417_ = v___x_411_;
v_isShared_418_ = v_isSharedCheck_422_;
goto v_resetjp_416_;
}
else
{
lean_inc(v_a_415_);
lean_dec(v___x_411_);
v___x_417_ = lean_box(0);
v_isShared_418_ = v_isSharedCheck_422_;
goto v_resetjp_416_;
}
v_resetjp_416_:
{
lean_object* v___x_420_; 
if (v_isShared_418_ == 0)
{
v___x_420_ = v___x_417_;
goto v_reusejp_419_;
}
else
{
lean_object* v_reuseFailAlloc_421_; 
v_reuseFailAlloc_421_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_421_, 0, v_a_415_);
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
}
}
else
{
lean_object* v_a_425_; lean_object* v___x_427_; uint8_t v_isShared_428_; uint8_t v_isSharedCheck_432_; 
lean_dec_ref(v_postCtx_385_);
lean_dec_ref(v_base_384_);
v_a_425_ = lean_ctor_get(v___x_396_, 0);
v_isSharedCheck_432_ = !lean_is_exclusive(v___x_396_);
if (v_isSharedCheck_432_ == 0)
{
v___x_427_ = v___x_396_;
v_isShared_428_ = v_isSharedCheck_432_;
goto v_resetjp_426_;
}
else
{
lean_inc(v_a_425_);
lean_dec(v___x_396_);
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
v___jp_433_:
{
uint8_t v___x_435_; 
v___x_435_ = l_Lean_Expr_isApp(v_a_434_);
if (v___x_435_ == 0)
{
lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v_a_438_; lean_object* v___x_440_; uint8_t v_isShared_441_; uint8_t v_isSharedCheck_445_; 
lean_dec_ref(v_a_434_);
lean_dec_ref(v_postCtx_385_);
lean_dec_ref(v_base_384_);
v___x_436_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ModuleNF_evalExpr___closed__3, &lp_mathlib_Mathlib_Tactic_ModuleNF_evalExpr___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_ModuleNF_evalExpr___closed__3);
v___x_437_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ModuleNF_evalExpr_spec__0___redArg(v___x_436_, v_a_389_, v_a_390_, v_a_391_, v_a_392_);
v_a_438_ = lean_ctor_get(v___x_437_, 0);
v_isSharedCheck_445_ = !lean_is_exclusive(v___x_437_);
if (v_isSharedCheck_445_ == 0)
{
v___x_440_ = v___x_437_;
v_isShared_441_ = v_isSharedCheck_445_;
goto v_resetjp_439_;
}
else
{
lean_inc(v_a_438_);
lean_dec(v___x_437_);
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
else
{
v___y_395_ = v_a_434_;
goto v___jp_394_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_evalExpr___boxed(lean_object* v_base_471_, lean_object* v_postCtx_472_, lean_object* v_e_473_, lean_object* v_a_474_, lean_object* v_a_475_, lean_object* v_a_476_, lean_object* v_a_477_, lean_object* v_a_478_, lean_object* v_a_479_, lean_object* v_a_480_){
_start:
{
lean_object* v_res_481_; 
v_res_481_ = lp_mathlib_Mathlib_Tactic_ModuleNF_evalExpr(v_base_471_, v_postCtx_472_, v_e_473_, v_a_474_, v_a_475_, v_a_476_, v_a_477_, v_a_478_, v_a_479_);
lean_dec(v_a_479_);
lean_dec_ref(v_a_478_);
lean_dec(v_a_477_);
lean_dec_ref(v_a_476_);
lean_dec(v_a_475_);
lean_dec_ref(v_a_474_);
return v_res_481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ModuleNF_evalExpr_spec__0(lean_object* v_00_u03b1_482_, lean_object* v_msg_483_, lean_object* v___y_484_, lean_object* v___y_485_, lean_object* v___y_486_, lean_object* v___y_487_){
_start:
{
lean_object* v___x_489_; 
v___x_489_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ModuleNF_evalExpr_spec__0___redArg(v_msg_483_, v___y_484_, v___y_485_, v___y_486_, v___y_487_);
return v___x_489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ModuleNF_evalExpr_spec__0___boxed(lean_object* v_00_u03b1_490_, lean_object* v_msg_491_, lean_object* v___y_492_, lean_object* v___y_493_, lean_object* v___y_494_, lean_object* v___y_495_, lean_object* v___y_496_){
_start:
{
lean_object* v_res_497_; 
v_res_497_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ModuleNF_evalExpr_spec__0(v_00_u03b1_490_, v_msg_491_, v___y_492_, v___y_493_, v___y_494_, v___y_495_);
lean_dec(v___y_495_);
lean_dec_ref(v___y_494_);
lean_dec(v___y_493_);
lean_dec_ref(v___y_492_);
return v_res_497_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__0___closed__0(void){
_start:
{
lean_object* v___x_498_; 
v___x_498_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_498_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__0___closed__1(void){
_start:
{
lean_object* v___x_499_; lean_object* v___x_500_; 
v___x_499_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__0___closed__0, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__0___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__0___closed__0);
v___x_500_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_500_, 0, v___x_499_);
return v___x_500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__0(lean_object* v_00_u03b2_501_){
_start:
{
lean_object* v___x_502_; 
v___x_502_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__0___closed__1, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__0___closed__1_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__0___closed__1);
return v___x_502_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__1___closed__0(void){
_start:
{
lean_object* v___x_503_; 
v___x_503_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_503_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__1___closed__1(void){
_start:
{
lean_object* v___x_504_; lean_object* v___x_505_; 
v___x_504_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__1___closed__0, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__1___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__1___closed__0);
v___x_505_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_505_, 0, v___x_504_);
return v___x_505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__1(lean_object* v_00_u03b2_506_){
_start:
{
lean_object* v___x_507_; 
v___x_507_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__1___closed__1, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__1___closed__1_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__1___closed__1);
return v___x_507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__2(lean_object* v_x_508_, lean_object* v_x_509_, lean_object* v___y_510_, lean_object* v___y_511_, lean_object* v___y_512_, lean_object* v___y_513_){
_start:
{
if (lean_obj_tag(v_x_509_) == 0)
{
lean_object* v___x_515_; 
v___x_515_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_515_, 0, v_x_508_);
return v___x_515_;
}
else
{
lean_object* v_head_516_; lean_object* v_tail_517_; uint8_t v___x_518_; uint8_t v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; 
v_head_516_ = lean_ctor_get(v_x_509_, 0);
lean_inc(v_head_516_);
v_tail_517_ = lean_ctor_get(v_x_509_, 1);
lean_inc(v_tail_517_);
lean_dec_ref_known(v_x_509_, 2);
v___x_518_ = 1;
v___x_519_ = 0;
v___x_520_ = lean_unsigned_to_nat(1000u);
v___x_521_ = l_Lean_Meta_SimpTheorems_addConst(v_x_508_, v_head_516_, v___x_518_, v___x_519_, v___x_520_, v___y_510_, v___y_511_, v___y_512_, v___y_513_);
if (lean_obj_tag(v___x_521_) == 0)
{
lean_object* v_a_522_; 
v_a_522_ = lean_ctor_get(v___x_521_, 0);
lean_inc(v_a_522_);
lean_dec_ref_known(v___x_521_, 1);
v_x_508_ = v_a_522_;
v_x_509_ = v_tail_517_;
goto _start;
}
else
{
lean_dec(v_tail_517_);
return v___x_521_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__2___boxed(lean_object* v_x_524_, lean_object* v_x_525_, lean_object* v___y_526_, lean_object* v___y_527_, lean_object* v___y_528_, lean_object* v___y_529_, lean_object* v___y_530_){
_start:
{
lean_object* v_res_531_; 
v_res_531_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__2(v_x_524_, v_x_525_, v___y_526_, v___y_527_, v___y_528_, v___y_529_);
lean_dec(v___y_529_);
lean_dec_ref(v___y_528_);
lean_dec(v___y_527_);
lean_dec_ref(v___y_526_);
return v_res_531_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__0(void){
_start:
{
lean_object* v___x_532_; 
v___x_532_ = l_Lean_Meta_DiscrTree_empty(lean_box(0));
return v___x_532_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__1(void){
_start:
{
lean_object* v___x_533_; 
v___x_533_ = lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__0(lean_box(0));
return v___x_533_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__2(void){
_start:
{
lean_object* v___x_534_; 
v___x_534_ = lp_mathlib_Lean_PersistentHashMap_empty___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__1(lean_box(0));
return v___x_534_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__3(void){
_start:
{
lean_object* v___x_535_; 
v___x_535_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_535_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__4(void){
_start:
{
lean_object* v___x_536_; lean_object* v___x_537_; 
v___x_536_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__3, &lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__3);
v___x_537_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_537_, 0, v___x_536_);
return v___x_537_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__5(void){
_start:
{
lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; 
v___x_538_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__4, &lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__4);
v___x_539_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__2, &lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__2);
v___x_540_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__1, &lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__1);
v___x_541_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__0, &lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__0);
v___x_542_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_542_, 0, v___x_541_);
lean_ctor_set(v___x_542_, 1, v___x_541_);
lean_ctor_set(v___x_542_, 2, v___x_540_);
lean_ctor_set(v___x_542_, 3, v___x_539_);
lean_ctor_set(v___x_542_, 4, v___x_540_);
lean_ctor_set(v___x_542_, 5, v___x_538_);
return v___x_542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx(lean_object* v_a_598_, lean_object* v_a_599_, lean_object* v_a_600_, lean_object* v_a_601_){
_start:
{
lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; 
v___x_603_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__5, &lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__5);
v___x_604_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__29));
v___x_605_ = lp_mathlib_List_foldlM___at___00Mathlib_Tactic_ModuleNF_cleanupCtx_spec__2(v___x_603_, v___x_604_, v_a_598_, v_a_599_, v_a_600_, v_a_601_);
if (lean_obj_tag(v___x_605_) == 0)
{
lean_object* v_a_606_; lean_object* v___x_607_; 
v_a_606_ = lean_ctor_get(v___x_605_, 0);
lean_inc(v_a_606_);
lean_dec_ref_known(v___x_605_, 1);
v___x_607_ = l_Lean_Meta_getSimpCongrTheorems___redArg(v_a_601_);
if (lean_obj_tag(v___x_607_) == 0)
{
lean_object* v_a_608_; lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; 
v_a_608_ = lean_ctor_get(v___x_607_, 0);
lean_inc(v_a_608_);
lean_dec_ref_known(v___x_607_, 1);
v___x_609_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___closed__30));
v___x_610_ = lean_unsigned_to_nat(1u);
v___x_611_ = lean_mk_empty_array_with_capacity(v___x_610_);
v___x_612_ = lean_array_push(v___x_611_, v_a_606_);
v___x_613_ = l_Lean_Options_empty;
v___x_614_ = l_Lean_Meta_Simp_mkContext___redArg(v___x_609_, v___x_612_, v_a_608_, v___x_613_, v_a_598_, v_a_600_, v_a_601_);
return v___x_614_;
}
else
{
lean_object* v_a_615_; lean_object* v___x_617_; uint8_t v_isShared_618_; uint8_t v_isSharedCheck_622_; 
lean_dec(v_a_606_);
v_a_615_ = lean_ctor_get(v___x_607_, 0);
v_isSharedCheck_622_ = !lean_is_exclusive(v___x_607_);
if (v_isSharedCheck_622_ == 0)
{
v___x_617_ = v___x_607_;
v_isShared_618_ = v_isSharedCheck_622_;
goto v_resetjp_616_;
}
else
{
lean_inc(v_a_615_);
lean_dec(v___x_607_);
v___x_617_ = lean_box(0);
v_isShared_618_ = v_isSharedCheck_622_;
goto v_resetjp_616_;
}
v_resetjp_616_:
{
lean_object* v___x_620_; 
if (v_isShared_618_ == 0)
{
v___x_620_ = v___x_617_;
goto v_reusejp_619_;
}
else
{
lean_object* v_reuseFailAlloc_621_; 
v_reuseFailAlloc_621_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_621_, 0, v_a_615_);
v___x_620_ = v_reuseFailAlloc_621_;
goto v_reusejp_619_;
}
v_reusejp_619_:
{
return v___x_620_;
}
}
}
}
else
{
lean_object* v_a_623_; lean_object* v___x_625_; uint8_t v_isShared_626_; uint8_t v_isSharedCheck_630_; 
v_a_623_ = lean_ctor_get(v___x_605_, 0);
v_isSharedCheck_630_ = !lean_is_exclusive(v___x_605_);
if (v_isSharedCheck_630_ == 0)
{
v___x_625_ = v___x_605_;
v_isShared_626_ = v_isSharedCheck_630_;
goto v_resetjp_624_;
}
else
{
lean_inc(v_a_623_);
lean_dec(v___x_605_);
v___x_625_ = lean_box(0);
v_isShared_626_ = v_isSharedCheck_630_;
goto v_resetjp_624_;
}
v_resetjp_624_:
{
lean_object* v___x_628_; 
if (v_isShared_626_ == 0)
{
v___x_628_ = v___x_625_;
goto v_reusejp_627_;
}
else
{
lean_object* v_reuseFailAlloc_629_; 
v_reuseFailAlloc_629_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_629_, 0, v_a_623_);
v___x_628_ = v_reuseFailAlloc_629_;
goto v_reusejp_627_;
}
v_reusejp_627_:
{
return v___x_628_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx___boxed(lean_object* v_a_631_, lean_object* v_a_632_, lean_object* v_a_633_, lean_object* v_a_634_, lean_object* v_a_635_){
_start:
{
lean_object* v_res_636_; 
v_res_636_ = lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx(v_a_631_, v_a_632_, v_a_633_, v_a_634_);
lean_dec(v_a_634_);
lean_dec_ref(v_a_633_);
lean_dec(v_a_632_);
lean_dec_ref(v_a_631_);
return v_res_636_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__0(void){
_start:
{
lean_object* v___x_637_; 
v___x_637_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_637_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__1(void){
_start:
{
lean_object* v___x_638_; lean_object* v___x_639_; 
v___x_638_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__0, &lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__0);
v___x_639_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_639_, 0, v___x_638_);
return v___x_639_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__2(void){
_start:
{
lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_642_; 
v___x_640_ = lean_unsigned_to_nat(0u);
v___x_641_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__1, &lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__1);
v___x_642_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_642_, 0, v___x_641_);
lean_ctor_set(v___x_642_, 1, v___x_640_);
return v___x_642_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__3(void){
_start:
{
lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; 
v___x_643_ = lean_unsigned_to_nat(32u);
v___x_644_ = lean_mk_empty_array_with_capacity(v___x_643_);
v___x_645_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_645_, 0, v___x_644_);
return v___x_645_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__4(void){
_start:
{
size_t v___x_646_; lean_object* v___x_647_; lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; 
v___x_646_ = ((size_t)5ULL);
v___x_647_ = lean_unsigned_to_nat(0u);
v___x_648_ = lean_unsigned_to_nat(32u);
v___x_649_ = lean_mk_empty_array_with_capacity(v___x_648_);
v___x_650_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__3, &lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__3);
v___x_651_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_651_, 0, v___x_650_);
lean_ctor_set(v___x_651_, 1, v___x_649_);
lean_ctor_set(v___x_651_, 2, v___x_647_);
lean_ctor_set(v___x_651_, 3, v___x_647_);
lean_ctor_set_usize(v___x_651_, 4, v___x_646_);
return v___x_651_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__5(void){
_start:
{
lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; 
v___x_652_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__4, &lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__4);
v___x_653_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__1, &lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__1);
v___x_654_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_654_, 0, v___x_653_);
lean_ctor_set(v___x_654_, 1, v___x_653_);
lean_ctor_set(v___x_654_, 2, v___x_653_);
lean_ctor_set(v___x_654_, 3, v___x_652_);
return v___x_654_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__6(void){
_start:
{
lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; 
v___x_655_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__5, &lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__5);
v___x_656_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__2, &lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__2);
v___x_657_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_657_, 0, v___x_656_);
lean_ctor_set(v___x_657_, 1, v___x_655_);
return v___x_657_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__8(void){
_start:
{
lean_object* v___x_660_; lean_object* v___x_661_; 
v___x_660_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__7));
v___x_661_ = l_Lean_Meta_Simp_mkDefaultMethodsCore(v___x_660_);
return v___x_661_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup(lean_object* v_ctx_662_, lean_object* v_r_663_, lean_object* v_a_664_, lean_object* v_a_665_, lean_object* v_a_666_, lean_object* v_a_667_){
_start:
{
lean_object* v_expr_669_; lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; 
v_expr_669_ = lean_ctor_get(v_r_663_, 0);
v___x_670_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__6, &lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__6);
v___x_671_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__8, &lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___closed__8);
lean_inc_ref(v_expr_669_);
v___x_672_ = l_Lean_Meta_Simp_main(v_expr_669_, v_ctx_662_, v___x_670_, v___x_671_, v_a_664_, v_a_665_, v_a_666_, v_a_667_);
if (lean_obj_tag(v___x_672_) == 0)
{
lean_object* v_a_673_; lean_object* v_fst_674_; lean_object* v___x_675_; 
v_a_673_ = lean_ctor_get(v___x_672_, 0);
lean_inc(v_a_673_);
lean_dec_ref_known(v___x_672_, 1);
v_fst_674_ = lean_ctor_get(v_a_673_, 0);
lean_inc(v_fst_674_);
lean_dec(v_a_673_);
v___x_675_ = l_Lean_Meta_Simp_Result_mkEqTrans(v_r_663_, v_fst_674_, v_a_664_, v_a_665_, v_a_666_, v_a_667_);
return v___x_675_;
}
else
{
lean_object* v_a_676_; lean_object* v___x_678_; uint8_t v_isShared_679_; uint8_t v_isSharedCheck_683_; 
lean_dec_ref(v_r_663_);
v_a_676_ = lean_ctor_get(v___x_672_, 0);
v_isSharedCheck_683_ = !lean_is_exclusive(v___x_672_);
if (v_isSharedCheck_683_ == 0)
{
v___x_678_ = v___x_672_;
v_isShared_679_ = v_isSharedCheck_683_;
goto v_resetjp_677_;
}
else
{
lean_inc(v_a_676_);
lean_dec(v___x_672_);
v___x_678_ = lean_box(0);
v_isShared_679_ = v_isSharedCheck_683_;
goto v_resetjp_677_;
}
v_resetjp_677_:
{
lean_object* v___x_681_; 
if (v_isShared_679_ == 0)
{
v___x_681_ = v___x_678_;
goto v_reusejp_680_;
}
else
{
lean_object* v_reuseFailAlloc_682_; 
v_reuseFailAlloc_682_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_682_, 0, v_a_676_);
v___x_681_ = v_reuseFailAlloc_682_;
goto v_reusejp_680_;
}
v_reusejp_680_:
{
return v___x_681_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___boxed(lean_object* v_ctx_684_, lean_object* v_r_685_, lean_object* v_a_686_, lean_object* v_a_687_, lean_object* v_a_688_, lean_object* v_a_689_, lean_object* v_a_690_){
_start:
{
lean_object* v_res_691_; 
v_res_691_ = lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup(v_ctx_684_, v_r_685_, v_a_686_, v_a_687_, v_a_688_, v_a_689_);
lean_dec(v_a_689_);
lean_dec_ref(v_a_688_);
lean_dec(v_a_687_);
lean_dec_ref(v_a_686_);
return v_res_691_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNFCore(lean_object* v_s_695_, lean_object* v_base_696_, lean_object* v_e_697_, lean_object* v_a_698_, lean_object* v_a_699_, lean_object* v_a_700_, lean_object* v_a_701_, lean_object* v_a_702_){
_start:
{
lean_object* v___x_704_; 
v___x_704_ = lp_mathlib_Mathlib_Tactic_ModuleNF_cleanupCtx(v_a_699_, v_a_700_, v_a_701_, v_a_702_);
if (lean_obj_tag(v___x_704_) == 0)
{
lean_object* v_a_705_; lean_object* v___x_706_; uint8_t v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; lean_object* v___x_710_; 
v_a_705_ = lean_ctor_get(v___x_704_, 0);
lean_inc(v_a_705_);
lean_dec_ref_known(v___x_704_, 1);
v___x_706_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNFCore___closed__0));
v___x_707_ = 1;
lean_inc_ref(v_a_698_);
v___x_708_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ModuleNF_evalExpr___boxed), 10, 2);
lean_closure_set(v___x_708_, 0, v_base_696_);
lean_closure_set(v___x_708_, 1, v_a_698_);
v___x_709_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ModuleNF_cleanup___boxed), 7, 1);
lean_closure_set(v___x_709_, 0, v_a_705_);
v___x_710_ = lp_mathlib_Mathlib_Tactic_AtomM_recurse(v_s_695_, v___x_706_, v___x_707_, v___x_708_, v___x_709_, v_e_697_, v_a_699_, v_a_700_, v_a_701_, v_a_702_);
return v___x_710_;
}
else
{
lean_object* v_a_711_; lean_object* v___x_713_; uint8_t v_isShared_714_; uint8_t v_isSharedCheck_718_; 
lean_dec_ref(v_e_697_);
lean_dec_ref(v_base_696_);
lean_dec(v_s_695_);
v_a_711_ = lean_ctor_get(v___x_704_, 0);
v_isSharedCheck_718_ = !lean_is_exclusive(v___x_704_);
if (v_isSharedCheck_718_ == 0)
{
v___x_713_ = v___x_704_;
v_isShared_714_ = v_isSharedCheck_718_;
goto v_resetjp_712_;
}
else
{
lean_inc(v_a_711_);
lean_dec(v___x_704_);
v___x_713_ = lean_box(0);
v_isShared_714_ = v_isSharedCheck_718_;
goto v_resetjp_712_;
}
v_resetjp_712_:
{
lean_object* v___x_716_; 
if (v_isShared_714_ == 0)
{
v___x_716_ = v___x_713_;
goto v_reusejp_715_;
}
else
{
lean_object* v_reuseFailAlloc_717_; 
v_reuseFailAlloc_717_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_717_, 0, v_a_711_);
v___x_716_ = v_reuseFailAlloc_717_;
goto v_reusejp_715_;
}
v_reusejp_715_:
{
return v___x_716_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNFCore___boxed(lean_object* v_s_719_, lean_object* v_base_720_, lean_object* v_e_721_, lean_object* v_a_722_, lean_object* v_a_723_, lean_object* v_a_724_, lean_object* v_a_725_, lean_object* v_a_726_, lean_object* v_a_727_){
_start:
{
lean_object* v_res_728_; 
v_res_728_ = lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNFCore(v_s_719_, v_base_720_, v_e_721_, v_a_722_, v_a_723_, v_a_724_, v_a_725_, v_a_726_);
lean_dec(v_a_726_);
lean_dec_ref(v_a_725_);
lean_dec(v_a_724_);
lean_dec_ref(v_a_723_);
lean_dec_ref(v_a_722_);
return v_res_728_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__19(void){
_start:
{
lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_770_; 
v___x_768_ = l_Lean_Parser_Tactic_location;
v___x_769_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__10));
v___x_770_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_770_, 0, v___x_769_);
lean_ctor_set(v___x_770_, 1, v___x_768_);
return v___x_770_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__20(void){
_start:
{
lean_object* v___x_771_; lean_object* v___x_772_; lean_object* v___x_773_; lean_object* v___x_774_; 
v___x_771_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__19, &lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__19_once, _init_lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__19);
v___x_772_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__18));
v___x_773_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__6));
v___x_774_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_774_, 0, v___x_773_);
lean_ctor_set(v___x_774_, 1, v___x_772_);
lean_ctor_set(v___x_774_, 2, v___x_771_);
return v___x_774_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__21(void){
_start:
{
lean_object* v___x_775_; lean_object* v___x_776_; lean_object* v___x_777_; lean_object* v___x_778_; 
v___x_775_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__20, &lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__20_once, _init_lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__20);
v___x_776_ = lean_unsigned_to_nat(1022u);
v___x_777_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__4));
v___x_778_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_778_, 0, v___x_777_);
lean_ctor_set(v___x_778_, 1, v___x_776_);
lean_ctor_set(v___x_778_, 2, v___x_775_);
return v___x_778_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF(void){
_start:
{
lean_object* v___x_779_; 
v___x_779_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__21, &lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__21_once, _init_lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__21);
return v___x_779_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_780_; lean_object* v___x_781_; lean_object* v___x_782_; 
v___x_780_ = lean_box(0);
v___x_781_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_782_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_782_, 0, v___x_781_);
lean_ctor_set(v___x_782_, 1, v___x_780_);
return v___x_782_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__0___redArg(){
_start:
{
lean_object* v___x_784_; lean_object* v___x_785_; 
v___x_784_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__0___redArg___closed__0);
v___x_785_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_785_, 0, v___x_784_);
return v___x_785_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__0___redArg___boxed(lean_object* v___y_786_){
_start:
{
lean_object* v_res_787_; 
v_res_787_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__0___redArg();
return v_res_787_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__0(lean_object* v_00_u03b1_788_, lean_object* v___y_789_, lean_object* v___y_790_, lean_object* v___y_791_, lean_object* v___y_792_, lean_object* v___y_793_, lean_object* v___y_794_, lean_object* v___y_795_, lean_object* v___y_796_){
_start:
{
lean_object* v___x_798_; 
v___x_798_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__0___redArg();
return v___x_798_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__0___boxed(lean_object* v_00_u03b1_799_, lean_object* v___y_800_, lean_object* v___y_801_, lean_object* v___y_802_, lean_object* v___y_803_, lean_object* v___y_804_, lean_object* v___y_805_, lean_object* v___y_806_, lean_object* v___y_807_, lean_object* v___y_808_){
_start:
{
lean_object* v_res_809_; 
v_res_809_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__0(v_00_u03b1_799_, v___y_800_, v___y_801_, v___y_802_, v___y_803_, v___y_804_, v___y_805_, v___y_806_, v___y_807_);
lean_dec(v___y_807_);
lean_dec_ref(v___y_806_);
lean_dec(v___y_805_);
lean_dec_ref(v___y_804_);
lean_dec(v___y_803_);
lean_dec_ref(v___y_802_);
lean_dec(v___y_801_);
lean_dec_ref(v___y_800_);
return v_res_809_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__0(lean_object* v___x_810_, lean_object* v_loc_811_, lean_object* v_base_812_, lean_object* v___y_813_, lean_object* v___y_814_, lean_object* v___y_815_, lean_object* v___y_816_, lean_object* v___y_817_, lean_object* v___y_818_, lean_object* v___y_819_, lean_object* v___y_820_){
_start:
{
lean_object* v___x_822_; lean_object* v___x_823_; lean_object* v___x_824_; 
v___x_822_ = lean_mk_empty_array_with_capacity(v___x_810_);
v___x_823_ = lean_st_mk_ref(v___x_822_);
v___x_824_ = lp_mathlib_Mathlib_Tactic_Module_postprocessCtx(v___y_817_, v___y_818_, v___y_819_, v___y_820_);
if (lean_obj_tag(v___x_824_) == 0)
{
lean_object* v_a_825_; lean_object* v___x_826_; lean_object* v___x_827_; uint8_t v___x_828_; uint8_t v___x_829_; lean_object* v___x_830_; 
v_a_825_ = lean_ctor_get(v___x_824_, 0);
lean_inc(v_a_825_);
lean_dec_ref_known(v___x_824_, 1);
v___x_826_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNFCore___boxed), 9, 2);
lean_closure_set(v___x_826_, 0, v___x_823_);
lean_closure_set(v___x_826_, 1, v_base_812_);
v___x_827_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__7));
v___x_828_ = 2;
v___x_829_ = 0;
v___x_830_ = lp_mathlib_Mathlib_Tactic_transformAtNondepPropLocation(v___x_826_, v___x_827_, v_loc_811_, v___x_828_, v___x_829_, v_a_825_, v___y_813_, v___y_814_, v___y_815_, v___y_816_, v___y_817_, v___y_818_, v___y_819_, v___y_820_);
return v___x_830_;
}
else
{
lean_object* v_a_831_; lean_object* v___x_833_; uint8_t v_isShared_834_; uint8_t v_isSharedCheck_838_; 
lean_dec(v___x_823_);
lean_dec_ref(v_base_812_);
v_a_831_ = lean_ctor_get(v___x_824_, 0);
v_isSharedCheck_838_ = !lean_is_exclusive(v___x_824_);
if (v_isSharedCheck_838_ == 0)
{
v___x_833_ = v___x_824_;
v_isShared_834_ = v_isSharedCheck_838_;
goto v_resetjp_832_;
}
else
{
lean_inc(v_a_831_);
lean_dec(v___x_824_);
v___x_833_ = lean_box(0);
v_isShared_834_ = v_isSharedCheck_838_;
goto v_resetjp_832_;
}
v_resetjp_832_:
{
lean_object* v___x_836_; 
if (v_isShared_834_ == 0)
{
v___x_836_ = v___x_833_;
goto v_reusejp_835_;
}
else
{
lean_object* v_reuseFailAlloc_837_; 
v_reuseFailAlloc_837_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_837_, 0, v_a_831_);
v___x_836_ = v_reuseFailAlloc_837_;
goto v_reusejp_835_;
}
v_reusejp_835_:
{
return v___x_836_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__0___boxed(lean_object* v___x_839_, lean_object* v_loc_840_, lean_object* v_base_841_, lean_object* v___y_842_, lean_object* v___y_843_, lean_object* v___y_844_, lean_object* v___y_845_, lean_object* v___y_846_, lean_object* v___y_847_, lean_object* v___y_848_, lean_object* v___y_849_, lean_object* v___y_850_){
_start:
{
lean_object* v_res_851_; 
v_res_851_ = lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__0(v___x_839_, v_loc_840_, v_base_841_, v___y_842_, v___y_843_, v___y_844_, v___y_845_, v___y_846_, v___y_847_, v___y_848_, v___y_849_);
lean_dec(v___y_849_);
lean_dec_ref(v___y_848_);
lean_dec(v___y_847_);
lean_dec_ref(v___y_846_);
lean_dec(v___y_845_);
lean_dec_ref(v___y_844_);
lean_dec(v___y_843_);
lean_dec_ref(v___y_842_);
lean_dec(v_loc_840_);
lean_dec(v___x_839_);
return v_res_851_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__1___redArg(lean_object* v_msg_852_, lean_object* v___y_853_, lean_object* v___y_854_, lean_object* v___y_855_, lean_object* v___y_856_){
_start:
{
lean_object* v_ref_858_; lean_object* v___x_859_; lean_object* v_a_860_; lean_object* v___x_862_; uint8_t v_isShared_863_; uint8_t v_isSharedCheck_868_; 
v_ref_858_ = lean_ctor_get(v___y_855_, 5);
v___x_859_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_ModuleNF_evalExpr_spec__0_spec__0(v_msg_852_, v___y_853_, v___y_854_, v___y_855_, v___y_856_);
v_a_860_ = lean_ctor_get(v___x_859_, 0);
v_isSharedCheck_868_ = !lean_is_exclusive(v___x_859_);
if (v_isSharedCheck_868_ == 0)
{
v___x_862_ = v___x_859_;
v_isShared_863_ = v_isSharedCheck_868_;
goto v_resetjp_861_;
}
else
{
lean_inc(v_a_860_);
lean_dec(v___x_859_);
v___x_862_ = lean_box(0);
v_isShared_863_ = v_isSharedCheck_868_;
goto v_resetjp_861_;
}
v_resetjp_861_:
{
lean_object* v___x_864_; lean_object* v___x_866_; 
lean_inc(v_ref_858_);
v___x_864_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_864_, 0, v_ref_858_);
lean_ctor_set(v___x_864_, 1, v_a_860_);
if (v_isShared_863_ == 0)
{
lean_ctor_set_tag(v___x_862_, 1);
lean_ctor_set(v___x_862_, 0, v___x_864_);
v___x_866_ = v___x_862_;
goto v_reusejp_865_;
}
else
{
lean_object* v_reuseFailAlloc_867_; 
v_reuseFailAlloc_867_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_867_, 0, v___x_864_);
v___x_866_ = v_reuseFailAlloc_867_;
goto v_reusejp_865_;
}
v_reusejp_865_:
{
return v___x_866_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__1___redArg___boxed(lean_object* v_msg_869_, lean_object* v___y_870_, lean_object* v___y_871_, lean_object* v___y_872_, lean_object* v___y_873_, lean_object* v___y_874_){
_start:
{
lean_object* v_res_875_; 
v_res_875_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__1___redArg(v_msg_869_, v___y_870_, v___y_871_, v___y_872_, v___y_873_);
lean_dec(v___y_873_);
lean_dec_ref(v___y_872_);
lean_dec(v___y_871_);
lean_dec_ref(v___y_870_);
return v_res_875_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__3(void){
_start:
{
lean_object* v___x_880_; lean_object* v___x_881_; 
v___x_880_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__2));
v___x_881_ = l_Lean_stringToMessageData(v___x_880_);
return v___x_881_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__5(void){
_start:
{
lean_object* v___x_883_; lean_object* v___x_884_; 
v___x_883_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__4));
v___x_884_ = l_Lean_stringToMessageData(v___x_883_);
return v___x_884_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1(lean_object* v_R_885_, lean_object* v_loc_886_, lean_object* v___f_887_, lean_object* v___y_888_, lean_object* v___y_889_, lean_object* v___y_890_, lean_object* v___y_891_, lean_object* v___y_892_, lean_object* v___y_893_, lean_object* v___y_894_, lean_object* v___y_895_){
_start:
{
if (lean_obj_tag(v_R_885_) == 0)
{
lean_object* v___x_897_; 
v___x_897_ = lp_mathlib_Mathlib_Tactic_ModuleNF_inferBaseAtLocation(v_loc_886_, v___y_888_, v___y_889_, v___y_890_, v___y_891_, v___y_892_, v___y_893_, v___y_894_, v___y_895_);
if (lean_obj_tag(v___x_897_) == 0)
{
lean_object* v_a_898_; lean_object* v___x_899_; 
v_a_898_ = lean_ctor_get(v___x_897_, 0);
lean_inc(v_a_898_);
lean_dec_ref_known(v___x_897_, 1);
lean_inc(v___y_895_);
lean_inc_ref(v___y_894_);
lean_inc(v___y_893_);
lean_inc_ref(v___y_892_);
lean_inc(v___y_891_);
lean_inc_ref(v___y_890_);
lean_inc(v___y_889_);
lean_inc_ref(v___y_888_);
v___x_899_ = lean_apply_10(v___f_887_, v_a_898_, v___y_888_, v___y_889_, v___y_890_, v___y_891_, v___y_892_, v___y_893_, v___y_894_, v___y_895_, lean_box(0));
return v___x_899_;
}
else
{
lean_object* v_a_900_; lean_object* v___x_902_; uint8_t v_isShared_903_; uint8_t v_isSharedCheck_907_; 
lean_dec_ref(v___f_887_);
v_a_900_ = lean_ctor_get(v___x_897_, 0);
v_isSharedCheck_907_ = !lean_is_exclusive(v___x_897_);
if (v_isSharedCheck_907_ == 0)
{
v___x_902_ = v___x_897_;
v_isShared_903_ = v_isSharedCheck_907_;
goto v_resetjp_901_;
}
else
{
lean_inc(v_a_900_);
lean_dec(v___x_897_);
v___x_902_ = lean_box(0);
v_isShared_903_ = v_isSharedCheck_907_;
goto v_resetjp_901_;
}
v_resetjp_901_:
{
lean_object* v___x_905_; 
if (v_isShared_903_ == 0)
{
v___x_905_ = v___x_902_;
goto v_reusejp_904_;
}
else
{
lean_object* v_reuseFailAlloc_906_; 
v_reuseFailAlloc_906_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_906_, 0, v_a_900_);
v___x_905_ = v_reuseFailAlloc_906_;
goto v_reusejp_904_;
}
v_reusejp_904_:
{
return v___x_905_;
}
}
}
}
else
{
lean_object* v_val_908_; lean_object* v___x_909_; uint8_t v___x_910_; lean_object* v___x_911_; 
lean_dec(v_loc_886_);
v_val_908_ = lean_ctor_get(v_R_885_, 0);
lean_inc(v_val_908_);
lean_dec_ref_known(v_R_885_, 1);
v___x_909_ = lean_box(0);
v___x_910_ = 0;
v___x_911_ = l_Lean_Elab_Tactic_elabTerm(v_val_908_, v___x_909_, v___x_910_, v___y_888_, v___y_889_, v___y_890_, v___y_891_, v___y_892_, v___y_893_, v___y_894_, v___y_895_);
if (lean_obj_tag(v___x_911_) == 0)
{
lean_object* v_a_912_; lean_object* v___x_913_; 
v_a_912_ = lean_ctor_get(v___x_911_, 0);
lean_inc(v_a_912_);
lean_dec_ref_known(v___x_911_, 1);
v___x_913_ = lp_mathlib_Qq_getLevelQ_x27(v_a_912_, v___y_892_, v___y_893_, v___y_894_, v___y_895_);
if (lean_obj_tag(v___x_913_) == 0)
{
lean_object* v_a_914_; lean_object* v_fst_915_; lean_object* v_snd_916_; lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v___x_921_; lean_object* v___x_922_; 
v_a_914_ = lean_ctor_get(v___x_913_, 0);
lean_inc(v_a_914_);
lean_dec_ref_known(v___x_913_, 1);
v_fst_915_ = lean_ctor_get(v_a_914_, 0);
v_snd_916_ = lean_ctor_get(v_a_914_, 1);
v___x_917_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__1));
v___x_918_ = lean_box(0);
lean_inc(v_fst_915_);
v___x_919_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_919_, 0, v_fst_915_);
lean_ctor_set(v___x_919_, 1, v___x_918_);
v___x_920_ = l_Lean_Expr_const___override(v___x_917_, v___x_919_);
lean_inc(v_snd_916_);
v___x_921_ = l_Lean_Expr_app___override(v___x_920_, v_snd_916_);
v___x_922_ = l_Lean_Meta_trySynthInstance(v___x_921_, v___x_909_, v___y_892_, v___y_893_, v___y_894_, v___y_895_);
if (lean_obj_tag(v___x_922_) == 0)
{
lean_object* v_a_923_; 
v_a_923_ = lean_ctor_get(v___x_922_, 0);
lean_inc(v_a_923_);
lean_dec_ref_known(v___x_922_, 1);
if (lean_obj_tag(v_a_923_) == 1)
{
lean_object* v___x_924_; 
lean_dec_ref_known(v_a_923_, 1);
lean_inc(v___y_895_);
lean_inc_ref(v___y_894_);
lean_inc(v___y_893_);
lean_inc_ref(v___y_892_);
lean_inc(v___y_891_);
lean_inc_ref(v___y_890_);
lean_inc(v___y_889_);
lean_inc_ref(v___y_888_);
v___x_924_ = lean_apply_10(v___f_887_, v_a_914_, v___y_888_, v___y_889_, v___y_890_, v___y_891_, v___y_892_, v___y_893_, v___y_894_, v___y_895_, lean_box(0));
return v___x_924_;
}
else
{
lean_object* v___x_926_; uint8_t v_isShared_927_; uint8_t v_isSharedCheck_936_; 
lean_inc(v_snd_916_);
lean_dec(v_a_923_);
lean_dec_ref(v___f_887_);
v_isSharedCheck_936_ = !lean_is_exclusive(v_a_914_);
if (v_isSharedCheck_936_ == 0)
{
lean_object* v_unused_937_; lean_object* v_unused_938_; 
v_unused_937_ = lean_ctor_get(v_a_914_, 1);
lean_dec(v_unused_937_);
v_unused_938_ = lean_ctor_get(v_a_914_, 0);
lean_dec(v_unused_938_);
v___x_926_ = v_a_914_;
v_isShared_927_ = v_isSharedCheck_936_;
goto v_resetjp_925_;
}
else
{
lean_dec(v_a_914_);
v___x_926_ = lean_box(0);
v_isShared_927_ = v_isSharedCheck_936_;
goto v_resetjp_925_;
}
v_resetjp_925_:
{
lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_931_; 
v___x_928_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__3, &lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__3);
v___x_929_ = l_Lean_MessageData_ofExpr(v_snd_916_);
if (v_isShared_927_ == 0)
{
lean_ctor_set_tag(v___x_926_, 7);
lean_ctor_set(v___x_926_, 1, v___x_929_);
lean_ctor_set(v___x_926_, 0, v___x_928_);
v___x_931_ = v___x_926_;
goto v_reusejp_930_;
}
else
{
lean_object* v_reuseFailAlloc_935_; 
v_reuseFailAlloc_935_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_935_, 0, v___x_928_);
lean_ctor_set(v_reuseFailAlloc_935_, 1, v___x_929_);
v___x_931_ = v_reuseFailAlloc_935_;
goto v_reusejp_930_;
}
v_reusejp_930_:
{
lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; 
v___x_932_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__5, &lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___closed__5);
v___x_933_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_933_, 0, v___x_931_);
lean_ctor_set(v___x_933_, 1, v___x_932_);
v___x_934_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__1___redArg(v___x_933_, v___y_892_, v___y_893_, v___y_894_, v___y_895_);
return v___x_934_;
}
}
}
}
else
{
lean_object* v_a_939_; lean_object* v___x_941_; uint8_t v_isShared_942_; uint8_t v_isSharedCheck_946_; 
lean_dec(v_a_914_);
lean_dec_ref(v___f_887_);
v_a_939_ = lean_ctor_get(v___x_922_, 0);
v_isSharedCheck_946_ = !lean_is_exclusive(v___x_922_);
if (v_isSharedCheck_946_ == 0)
{
v___x_941_ = v___x_922_;
v_isShared_942_ = v_isSharedCheck_946_;
goto v_resetjp_940_;
}
else
{
lean_inc(v_a_939_);
lean_dec(v___x_922_);
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
lean_dec_ref(v___f_887_);
v_a_947_ = lean_ctor_get(v___x_913_, 0);
v_isSharedCheck_954_ = !lean_is_exclusive(v___x_913_);
if (v_isSharedCheck_954_ == 0)
{
v___x_949_ = v___x_913_;
v_isShared_950_ = v_isSharedCheck_954_;
goto v_resetjp_948_;
}
else
{
lean_inc(v_a_947_);
lean_dec(v___x_913_);
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
lean_object* v_a_955_; lean_object* v___x_957_; uint8_t v_isShared_958_; uint8_t v_isSharedCheck_962_; 
lean_dec_ref(v___f_887_);
v_a_955_ = lean_ctor_get(v___x_911_, 0);
v_isSharedCheck_962_ = !lean_is_exclusive(v___x_911_);
if (v_isSharedCheck_962_ == 0)
{
v___x_957_ = v___x_911_;
v_isShared_958_ = v_isSharedCheck_962_;
goto v_resetjp_956_;
}
else
{
lean_inc(v_a_955_);
lean_dec(v___x_911_);
v___x_957_ = lean_box(0);
v_isShared_958_ = v_isSharedCheck_962_;
goto v_resetjp_956_;
}
v_resetjp_956_:
{
lean_object* v___x_960_; 
if (v_isShared_958_ == 0)
{
v___x_960_ = v___x_957_;
goto v_reusejp_959_;
}
else
{
lean_object* v_reuseFailAlloc_961_; 
v_reuseFailAlloc_961_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_961_, 0, v_a_955_);
v___x_960_ = v_reuseFailAlloc_961_;
goto v_reusejp_959_;
}
v_reusejp_959_:
{
return v___x_960_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___boxed(lean_object* v_R_963_, lean_object* v_loc_964_, lean_object* v___f_965_, lean_object* v___y_966_, lean_object* v___y_967_, lean_object* v___y_968_, lean_object* v___y_969_, lean_object* v___y_970_, lean_object* v___y_971_, lean_object* v___y_972_, lean_object* v___y_973_, lean_object* v___y_974_){
_start:
{
lean_object* v_res_975_; 
v_res_975_ = lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1(v_R_963_, v_loc_964_, v___f_965_, v___y_966_, v___y_967_, v___y_968_, v___y_969_, v___y_970_, v___y_971_, v___y_972_, v___y_973_);
lean_dec(v___y_973_);
lean_dec_ref(v___y_972_);
lean_dec(v___y_971_);
lean_dec_ref(v___y_970_);
lean_dec(v___y_969_);
lean_dec_ref(v___y_968_);
lean_dec(v___y_967_);
lean_dec_ref(v___y_966_);
return v_res_975_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1(lean_object* v_x_984_, lean_object* v_a_985_, lean_object* v_a_986_, lean_object* v_a_987_, lean_object* v_a_988_, lean_object* v_a_989_, lean_object* v_a_990_, lean_object* v_a_991_, lean_object* v_a_992_){
_start:
{
lean_object* v___x_994_; uint8_t v___x_995_; 
v___x_994_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF___closed__4));
lean_inc(v_x_984_);
v___x_995_ = l_Lean_Syntax_isOfKind(v_x_984_, v___x_994_);
if (v___x_995_ == 0)
{
lean_object* v___x_996_; 
lean_dec(v_x_984_);
v___x_996_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__0___redArg();
return v___x_996_;
}
else
{
lean_object* v___x_997_; lean_object* v___y_999_; lean_object* v___y_1000_; lean_object* v___y_1001_; lean_object* v___y_1002_; lean_object* v___y_1003_; lean_object* v___y_1004_; lean_object* v___y_1005_; lean_object* v___y_1006_; lean_object* v___y_1007_; lean_object* v___y_1008_; lean_object* v___x_1014_; lean_object* v_R_1016_; lean_object* v___y_1017_; lean_object* v___y_1018_; lean_object* v___y_1019_; lean_object* v___y_1020_; lean_object* v___y_1021_; lean_object* v___y_1022_; lean_object* v___y_1023_; lean_object* v___y_1024_; lean_object* v___x_1036_; uint8_t v___x_1037_; 
v___x_997_ = lean_unsigned_to_nat(0u);
v___x_1014_ = lean_unsigned_to_nat(1u);
v___x_1036_ = l_Lean_Syntax_getArg(v_x_984_, v___x_1014_);
v___x_1037_ = l_Lean_Syntax_isNone(v___x_1036_);
if (v___x_1037_ == 0)
{
lean_object* v___x_1038_; uint8_t v___x_1039_; 
v___x_1038_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_1036_);
v___x_1039_ = l_Lean_Syntax_matchesNull(v___x_1036_, v___x_1038_);
if (v___x_1039_ == 0)
{
lean_object* v___x_1040_; 
lean_dec(v___x_1036_);
lean_dec(v_x_984_);
v___x_1040_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__0___redArg();
return v___x_1040_;
}
else
{
lean_object* v_R_1041_; lean_object* v___x_1042_; 
v_R_1041_ = l_Lean_Syntax_getArg(v___x_1036_, v___x_1014_);
lean_dec(v___x_1036_);
v___x_1042_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1042_, 0, v_R_1041_);
v_R_1016_ = v___x_1042_;
v___y_1017_ = v_a_985_;
v___y_1018_ = v_a_986_;
v___y_1019_ = v_a_987_;
v___y_1020_ = v_a_988_;
v___y_1021_ = v_a_989_;
v___y_1022_ = v_a_990_;
v___y_1023_ = v_a_991_;
v___y_1024_ = v_a_992_;
goto v___jp_1015_;
}
}
else
{
lean_object* v___x_1043_; 
lean_dec(v___x_1036_);
v___x_1043_ = lean_box(0);
v_R_1016_ = v___x_1043_;
v___y_1017_ = v_a_985_;
v___y_1018_ = v_a_986_;
v___y_1019_ = v_a_987_;
v___y_1020_ = v_a_988_;
v___y_1021_ = v_a_989_;
v___y_1022_ = v_a_990_;
v___y_1023_ = v_a_991_;
v___y_1024_ = v_a_992_;
goto v___jp_1015_;
}
v___jp_998_:
{
lean_object* v___x_1009_; lean_object* v_loc_1010_; lean_object* v___f_1011_; lean_object* v___y_1012_; lean_object* v___x_1013_; 
v___x_1009_ = l_Lean_mkOptionalNode(v___y_1008_);
v_loc_1010_ = l_Lean_Elab_Tactic_expandOptLocation(v___x_1009_);
lean_dec(v___x_1009_);
lean_inc(v_loc_1010_);
v___f_1011_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__0___boxed), 12, 2);
lean_closure_set(v___f_1011_, 0, v___x_997_);
lean_closure_set(v___f_1011_, 1, v_loc_1010_);
v___y_1012_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___lam__1___boxed), 12, 3);
lean_closure_set(v___y_1012_, 0, v___y_999_);
lean_closure_set(v___y_1012_, 1, v_loc_1010_);
lean_closure_set(v___y_1012_, 2, v___f_1011_);
v___x_1013_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___y_1012_, v___y_1004_, v___y_1006_, v___y_1001_, v___y_1007_, v___y_1003_, v___y_1000_, v___y_1002_, v___y_1005_);
return v___x_1013_;
}
v___jp_1015_:
{
lean_object* v___x_1025_; lean_object* v___x_1026_; uint8_t v___x_1027_; 
v___x_1025_ = lean_unsigned_to_nat(2u);
v___x_1026_ = l_Lean_Syntax_getArg(v_x_984_, v___x_1025_);
lean_dec(v_x_984_);
v___x_1027_ = l_Lean_Syntax_isNone(v___x_1026_);
if (v___x_1027_ == 0)
{
uint8_t v___x_1028_; 
lean_inc(v___x_1026_);
v___x_1028_ = l_Lean_Syntax_matchesNull(v___x_1026_, v___x_1014_);
if (v___x_1028_ == 0)
{
lean_object* v___x_1029_; 
lean_dec(v___x_1026_);
lean_dec(v_R_1016_);
v___x_1029_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__0___redArg();
return v___x_1029_;
}
else
{
lean_object* v_loc_1030_; lean_object* v___x_1031_; uint8_t v___x_1032_; 
v_loc_1030_ = l_Lean_Syntax_getArg(v___x_1026_, v___x_997_);
lean_dec(v___x_1026_);
v___x_1031_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___closed__3));
lean_inc(v_loc_1030_);
v___x_1032_ = l_Lean_Syntax_isOfKind(v_loc_1030_, v___x_1031_);
if (v___x_1032_ == 0)
{
lean_object* v___x_1033_; 
lean_dec(v_loc_1030_);
lean_dec(v_R_1016_);
v___x_1033_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__0___redArg();
return v___x_1033_;
}
else
{
lean_object* v___x_1034_; 
v___x_1034_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1034_, 0, v_loc_1030_);
v___y_999_ = v_R_1016_;
v___y_1000_ = v___y_1022_;
v___y_1001_ = v___y_1019_;
v___y_1002_ = v___y_1023_;
v___y_1003_ = v___y_1021_;
v___y_1004_ = v___y_1017_;
v___y_1005_ = v___y_1024_;
v___y_1006_ = v___y_1018_;
v___y_1007_ = v___y_1020_;
v___y_1008_ = v___x_1034_;
goto v___jp_998_;
}
}
}
else
{
lean_object* v___x_1035_; 
lean_dec(v___x_1026_);
v___x_1035_ = lean_box(0);
v___y_999_ = v_R_1016_;
v___y_1000_ = v___y_1022_;
v___y_1001_ = v___y_1019_;
v___y_1002_ = v___y_1023_;
v___y_1003_ = v___y_1021_;
v___y_1004_ = v___y_1017_;
v___y_1005_ = v___y_1024_;
v___y_1006_ = v___y_1018_;
v___y_1007_ = v___y_1020_;
v___y_1008_ = v___x_1035_;
goto v___jp_998_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1___boxed(lean_object* v_x_1044_, lean_object* v_a_1045_, lean_object* v_a_1046_, lean_object* v_a_1047_, lean_object* v_a_1048_, lean_object* v_a_1049_, lean_object* v_a_1050_, lean_object* v_a_1051_, lean_object* v_a_1052_, lean_object* v_a_1053_){
_start:
{
lean_object* v_res_1054_; 
v_res_1054_ = lp_mathlib_Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1(v_x_1044_, v_a_1045_, v_a_1046_, v_a_1047_, v_a_1048_, v_a_1049_, v_a_1050_, v_a_1051_, v_a_1052_);
lean_dec(v_a_1052_);
lean_dec_ref(v_a_1051_);
lean_dec(v_a_1050_);
lean_dec_ref(v_a_1049_);
lean_dec(v_a_1048_);
lean_dec_ref(v_a_1047_);
lean_dec(v_a_1046_);
lean_dec_ref(v_a_1045_);
return v_res_1054_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__1(lean_object* v_00_u03b1_1055_, lean_object* v_msg_1056_, lean_object* v___y_1057_, lean_object* v___y_1058_, lean_object* v___y_1059_, lean_object* v___y_1060_, lean_object* v___y_1061_, lean_object* v___y_1062_, lean_object* v___y_1063_, lean_object* v___y_1064_){
_start:
{
lean_object* v___x_1066_; 
v___x_1066_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__1___redArg(v_msg_1056_, v___y_1061_, v___y_1062_, v___y_1063_, v___y_1064_);
return v___x_1066_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__1___boxed(lean_object* v_00_u03b1_1067_, lean_object* v_msg_1068_, lean_object* v___y_1069_, lean_object* v___y_1070_, lean_object* v___y_1071_, lean_object* v___y_1072_, lean_object* v___y_1073_, lean_object* v___y_1074_, lean_object* v___y_1075_, lean_object* v___y_1076_, lean_object* v___y_1077_){
_start:
{
lean_object* v_res_1078_; 
v_res_1078_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_ModuleNF___aux__Mathlib__Tactic__ModuleNF______elabRules__Mathlib__Tactic__ModuleNF__moduleNF__1_spec__1(v_00_u03b1_1067_, v_msg_1068_, v___y_1069_, v___y_1070_, v___y_1071_, v___y_1072_, v___y_1073_, v___y_1074_, v___y_1075_, v___y_1076_);
lean_dec(v___y_1076_);
lean_dec_ref(v___y_1075_);
lean_dec(v___y_1074_);
lean_dec_ref(v___y_1073_);
lean_dec(v___y_1072_);
lean_dec_ref(v___y_1071_);
lean_dec(v___y_1070_);
lean_dec_ref(v___y_1069_);
return v_res_1078_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Algebra_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Module(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ModuleNF(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Algebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_ModuleNF(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF = _init_lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ModuleNF_moduleNF);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Algebra_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Module(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_ModuleNF(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Algebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ModuleNF(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_ModuleNF(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_ModuleNF(builtin);
}
#ifdef __cplusplus
}
#endif
