// Lean compiler output
// Module: Plausible.Sampleable
// Imports: public import Init public meta import Init public meta import Lean.Elab.Command public meta import Lean.Meta.Eval public import Plausible.Arbitrary public import Plausible.Shrinkable
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
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lp_plausible_Plausible_Sigma_shrinkable___redArg___lam__2(lean_object*, lean_object*, lean_object*);
lean_object* lp_plausible_Plausible_Sigma_Arbitrary___redArg(lean_object*, lean_object*);
lean_object* l_Sigma_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Level_ofNat(lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Sum_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_plausible_Plausible_instShrinkableSum___redArg(lean_object*, lean_object*);
lean_object* lp_plausible_Plausible_Sum_Arbitrary___redArg(lean_object*, lean_object*);
lean_object* l_instReprTupleOfRepr___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Prod_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_plausible_Plausible_Prod_shrinkable___redArg___lam__2(lean_object*, lean_object*, lean_object*);
lean_object* lp_plausible_Plausible_Gen_prodOf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Prod_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTermAndSynthesize(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_whnf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshLevelMVar(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Meta_synthInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_instantiate_level_mvars(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAppB(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_cleanupAnnotations(lean_object*);
uint8_t l_Lean_Expr_isApp(lean_object*);
lean_object* l_Lean_Expr_appFnCleanup___redArg(lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_mkApp3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_evalExpr___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
extern lean_object* lp_plausible_Plausible_Bool_Arbitrary;
lean_object* lp_plausible_Plausible_Bool_shrinkable___lam__0___boxed(lean_object*);
lean_object* l_Bool_repr___boxed(lean_object*, lean_object*);
lean_object* l_Option_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_plausible_Plausible_Option_shrinkable___redArg(lean_object*);
lean_object* lp_plausible_Plausible_Option_Arbitrary___redArg(lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_runTermElabM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_plausible_Plausible_List_shrinkable___redArg___lam__4(lean_object*, lean_object*);
lean_object* lp_plausible_Plausible_Gen_listOf___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_mapTR(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_instRepr___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_plausible_Plausible_Array_shrinkable___redArg(lean_object*);
lean_object* lp_plausible_Plausible_Gen_arrayOf___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_map(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_plausible_Plausible_SampleableExt_selfContained___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_plausible_Plausible_SampleableExt_selfContained___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_SampleableExt_selfContained___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_SampleableExt_selfContained___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_SampleableExt_selfContained(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_SampleableExt_mkSelfContained___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_SampleableExt_mkSelfContained(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_SampleableExt_interpSample___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_SampleableExt_interpSample___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_SampleableExt_interpSample(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_SampleableExt_interpSample___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_arbitraryProxy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_arbitraryProxy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_arbitraryProxy(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_arbitraryProxy___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Sum_SampleableExt___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Sum_SampleableExt___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Sum_SampleableExt(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instSampleableExtSigma___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instSampleableExtSigma___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instSampleableExtSigma___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instSampleableExtSigma___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_instSampleableExtSigma(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Option_sampleableExt___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Option_sampleableExt___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Option_sampleableExt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Prod_sampleableExt___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Prod_sampleableExt(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_plausible_Plausible_Prop_sampleableExt___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Bool_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_Prop_sampleableExt___closed__0 = (const lean_object*)&lp_plausible_Plausible_Prop_sampleableExt___closed__0_value;
static const lean_closure_object lp_plausible_Plausible_Prop_sampleableExt___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_Bool_shrinkable___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_Prop_sampleableExt___closed__1 = (const lean_object*)&lp_plausible_Plausible_Prop_sampleableExt___closed__1_value;
static lean_once_cell_t lp_plausible_Plausible_Prop_sampleableExt___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Prop_sampleableExt___closed__2;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Prop_sampleableExt;
LEAN_EXPORT lean_object* lp_plausible_Plausible_List_sampleableExt___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_List_sampleableExt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_ULift_sampleableExt___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_ULift_sampleableExt___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_ULift_sampleableExt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Array_sampleableExt___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Array_sampleableExt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_mk___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_mk___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_mk(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_mk___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_get___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_get___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_get(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_get___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_inhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_inhabited___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_inhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_inhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_repr___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_repr___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_shrinkable___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_shrinkable___lam__0___boxed(lean_object*);
static const lean_closure_object lp_plausible_Plausible_NoShrink_shrinkable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_NoShrink_shrinkable___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_NoShrink_shrinkable___closed__0 = (const lean_object*)&lp_plausible_Plausible_NoShrink_shrinkable___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_shrinkable(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_arbitrary___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_arbitrary___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_arbitrary(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_arbitrary___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_sampleableExt___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_sampleableExt(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_sampleableExt___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_instantiateLevelMVars___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_instantiateLevelMVars___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_instantiateLevelMVars___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_instantiateLevelMVars___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Plausible"};
static const lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__0_value;
static const lean_string_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "SampleableExt"};
static const lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__1 = (const lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__1_value;
static const lean_ctor_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__2_value_aux_0),((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__1_value),LEAN_SCALAR_PTR_LITERAL(60, 134, 28, 181, 150, 194, 96, 231)}};
static const lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__2 = (const lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__2_value;
static const lean_string_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "proxyRepr"};
static const lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__3 = (const lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__3_value;
static const lean_ctor_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__4_value_aux_0),((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__1_value),LEAN_SCALAR_PTR_LITERAL(60, 134, 28, 181, 150, 194, 96, 231)}};
static const lean_ctor_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__4_value_aux_1),((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__3_value),LEAN_SCALAR_PTR_LITERAL(254, 29, 228, 106, 122, 160, 234, 102)}};
static const lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__4 = (const lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__4_value;
static const lean_string_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "sample"};
static const lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__5 = (const lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__5_value;
static const lean_ctor_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__6_value_aux_0),((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__1_value),LEAN_SCALAR_PTR_LITERAL(60, 134, 28, 181, 150, 194, 96, 231)}};
static const lean_ctor_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__6_value_aux_1),((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__5_value),LEAN_SCALAR_PTR_LITERAL(67, 194, 232, 62, 29, 120, 232, 201)}};
static const lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__6 = (const lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__6_value;
static const lean_string_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Arbitrary"};
static const lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__7 = (const lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__7_value;
static const lean_string_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "arbitrary"};
static const lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__8 = (const lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__8_value;
static const lean_ctor_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__9_value_aux_0),((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__7_value),LEAN_SCALAR_PTR_LITERAL(87, 65, 220, 66, 159, 24, 65, 122)}};
static const lean_ctor_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__9_value_aux_1),((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__8_value),LEAN_SCALAR_PTR_LITERAL(33, 81, 203, 107, 28, 10, 107, 113)}};
static const lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__9 = (const lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__9_value;
static const lean_string_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "proxy"};
static const lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__10 = (const lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__10_value;
static const lean_ctor_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__11_value_aux_0),((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__1_value),LEAN_SCALAR_PTR_LITERAL(60, 134, 28, 181, 150, 194, 96, 231)}};
static const lean_ctor_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__11_value_aux_1),((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__10_value),LEAN_SCALAR_PTR_LITERAL(203, 236, 56, 19, 198, 201, 6, 90)}};
static const lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__11 = (const lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__11_value;
static const lean_string_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Gen"};
static const lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__12 = (const lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__12_value;
static const lean_ctor_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__13_value_aux_0),((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__12_value),LEAN_SCALAR_PTR_LITERAL(59, 136, 107, 112, 1, 192, 26, 229)}};
static const lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__13 = (const lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__13_value;
static const lean_string_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Repr"};
static const lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__14 = (const lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__14_value;
static const lean_ctor_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__14_value),LEAN_SCALAR_PTR_LITERAL(192, 59, 131, 233, 62, 241, 250, 220)}};
static const lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__15 = (const lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__15_value;
static const lean_string_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = " is not a type with computational content"};
static const lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__16 = (const lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__16_value;
static lean_once_cell_t lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__17;
static const lean_string_object lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = " is not a type"};
static const lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__18 = (const lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__18_value;
static lean_once_cell_t lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__19;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_command_x23sample___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "command#sample_"};
static const lean_object* lp_plausible_Plausible_command_x23sample___00__closed__0 = (const lean_object*)&lp_plausible_Plausible_command_x23sample___00__closed__0_value;
static const lean_ctor_object lp_plausible_Plausible_command_x23sample___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible_Plausible_command_x23sample___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_command_x23sample___00__closed__1_value_aux_0),((lean_object*)&lp_plausible_Plausible_command_x23sample___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(49, 53, 239, 14, 146, 255, 5, 84)}};
static const lean_object* lp_plausible_Plausible_command_x23sample___00__closed__1 = (const lean_object*)&lp_plausible_Plausible_command_x23sample___00__closed__1_value;
static const lean_string_object lp_plausible_Plausible_command_x23sample___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_plausible_Plausible_command_x23sample___00__closed__2 = (const lean_object*)&lp_plausible_Plausible_command_x23sample___00__closed__2_value;
static const lean_ctor_object lp_plausible_Plausible_command_x23sample___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_command_x23sample___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_plausible_Plausible_command_x23sample___00__closed__3 = (const lean_object*)&lp_plausible_Plausible_command_x23sample___00__closed__3_value;
static const lean_string_object lp_plausible_Plausible_command_x23sample___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "#sample "};
static const lean_object* lp_plausible_Plausible_command_x23sample___00__closed__4 = (const lean_object*)&lp_plausible_Plausible_command_x23sample___00__closed__4_value;
static const lean_ctor_object lp_plausible_Plausible_command_x23sample___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_plausible_Plausible_command_x23sample___00__closed__4_value)}};
static const lean_object* lp_plausible_Plausible_command_x23sample___00__closed__5 = (const lean_object*)&lp_plausible_Plausible_command_x23sample___00__closed__5_value;
static const lean_string_object lp_plausible_Plausible_command_x23sample___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_plausible_Plausible_command_x23sample___00__closed__6 = (const lean_object*)&lp_plausible_Plausible_command_x23sample___00__closed__6_value;
static const lean_ctor_object lp_plausible_Plausible_command_x23sample___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_command_x23sample___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_plausible_Plausible_command_x23sample___00__closed__7 = (const lean_object*)&lp_plausible_Plausible_command_x23sample___00__closed__7_value;
static const lean_ctor_object lp_plausible_Plausible_command_x23sample___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_plausible_Plausible_command_x23sample___00__closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_plausible_Plausible_command_x23sample___00__closed__8 = (const lean_object*)&lp_plausible_Plausible_command_x23sample___00__closed__8_value;
static const lean_ctor_object lp_plausible_Plausible_command_x23sample___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_plausible_Plausible_command_x23sample___00__closed__3_value),((lean_object*)&lp_plausible_Plausible_command_x23sample___00__closed__5_value),((lean_object*)&lp_plausible_Plausible_command_x23sample___00__closed__8_value)}};
static const lean_object* lp_plausible_Plausible_command_x23sample___00__closed__9 = (const lean_object*)&lp_plausible_Plausible_command_x23sample___00__closed__9_value;
static const lean_ctor_object lp_plausible_Plausible_command_x23sample___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_plausible_Plausible_command_x23sample___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_command_x23sample___00__closed__9_value)}};
static const lean_object* lp_plausible_Plausible_command_x23sample___00__closed__10 = (const lean_object*)&lp_plausible_Plausible_command_x23sample___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_command_x23sample__ = (const lean_object*)&lp_plausible_Plausible_command_x23sample___00__closed__10_value;
static const lean_string_object lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "IO"};
static const lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__0_value;
static const lean_ctor_object lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(2, 76, 19, 202, 4, 69, 238, 60)}};
static const lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__1 = (const lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__1_value;
static lean_once_cell_t lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__2;
static const lean_string_object lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "PUnit"};
static const lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__3 = (const lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__3_value;
static const lean_ctor_object lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(23, 153, 158, 141, 176, 162, 235, 153)}};
static const lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__4 = (const lean_object*)&lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__4_value;
static lean_once_cell_t lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__5;
static lean_once_cell_t lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__6;
static lean_once_cell_t lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__7;
static lean_once_cell_t lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__8;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "printSamples"};
static const lean_object* lp_plausible_Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1___lam__0___closed__0 = (const lean_object*)&lp_plausible_Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_SampleableExt_selfContained___redArg(lean_object* v_inst_2_, lean_object* v_inst_3_, lean_object* v_inst_4_){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_5_ = ((lean_object*)(lp_plausible_Plausible_SampleableExt_selfContained___redArg___closed__0));
v___x_6_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_6_, 0, v_inst_2_);
lean_ctor_set(v___x_6_, 1, v_inst_3_);
lean_ctor_set(v___x_6_, 2, v_inst_4_);
lean_ctor_set(v___x_6_, 3, v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_SampleableExt_selfContained(lean_object* v_00_u03b1_7_, lean_object* v_inst_8_, lean_object* v_inst_9_, lean_object* v_inst_10_){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lp_plausible_Plausible_SampleableExt_selfContained___redArg(v_inst_8_, v_inst_9_, v_inst_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_SampleableExt_mkSelfContained___redArg(lean_object* v_inst_12_, lean_object* v_inst_13_, lean_object* v_g_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_plausible_Plausible_SampleableExt_selfContained___redArg(v_inst_12_, v_inst_13_, v_g_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_SampleableExt_mkSelfContained(lean_object* v_00_u03b1_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_g_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lp_plausible_Plausible_SampleableExt_selfContained___redArg(v_inst_17_, v_inst_18_, v_g_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_SampleableExt_interpSample___redArg(lean_object* v_inst_21_, lean_object* v_a_22_, lean_object* v_a_23_){
_start:
{
lean_object* v_sample_24_; lean_object* v_interp_25_; lean_object* v___x_26_; 
v_sample_24_ = lean_ctor_get(v_inst_21_, 2);
lean_inc_ref(v_sample_24_);
v_interp_25_ = lean_ctor_get(v_inst_21_, 3);
lean_inc(v_interp_25_);
lean_dec_ref(v_inst_21_);
lean_inc(v_a_23_);
v___x_26_ = lean_apply_2(v_sample_24_, v_a_22_, v_a_23_);
if (lean_obj_tag(v___x_26_) == 0)
{
lean_object* v_a_27_; lean_object* v___x_29_; uint8_t v_isShared_30_; uint8_t v_isSharedCheck_34_; 
lean_dec(v_interp_25_);
v_a_27_ = lean_ctor_get(v___x_26_, 0);
v_isSharedCheck_34_ = !lean_is_exclusive(v___x_26_);
if (v_isSharedCheck_34_ == 0)
{
v___x_29_ = v___x_26_;
v_isShared_30_ = v_isSharedCheck_34_;
goto v_resetjp_28_;
}
else
{
lean_inc(v_a_27_);
lean_dec(v___x_26_);
v___x_29_ = lean_box(0);
v_isShared_30_ = v_isSharedCheck_34_;
goto v_resetjp_28_;
}
v_resetjp_28_:
{
lean_object* v___x_32_; 
if (v_isShared_30_ == 0)
{
v___x_32_ = v___x_29_;
goto v_reusejp_31_;
}
else
{
lean_object* v_reuseFailAlloc_33_; 
v_reuseFailAlloc_33_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_33_, 0, v_a_27_);
v___x_32_ = v_reuseFailAlloc_33_;
goto v_reusejp_31_;
}
v_reusejp_31_:
{
return v___x_32_;
}
}
}
else
{
lean_object* v_a_35_; lean_object* v___x_37_; uint8_t v_isShared_38_; uint8_t v_isSharedCheck_52_; 
v_a_35_ = lean_ctor_get(v___x_26_, 0);
v_isSharedCheck_52_ = !lean_is_exclusive(v___x_26_);
if (v_isSharedCheck_52_ == 0)
{
v___x_37_ = v___x_26_;
v_isShared_38_ = v_isSharedCheck_52_;
goto v_resetjp_36_;
}
else
{
lean_inc(v_a_35_);
lean_dec(v___x_26_);
v___x_37_ = lean_box(0);
v_isShared_38_ = v_isSharedCheck_52_;
goto v_resetjp_36_;
}
v_resetjp_36_:
{
lean_object* v_fst_39_; lean_object* v_snd_40_; lean_object* v___x_42_; uint8_t v_isShared_43_; uint8_t v_isSharedCheck_51_; 
v_fst_39_ = lean_ctor_get(v_a_35_, 0);
v_snd_40_ = lean_ctor_get(v_a_35_, 1);
v_isSharedCheck_51_ = !lean_is_exclusive(v_a_35_);
if (v_isSharedCheck_51_ == 0)
{
v___x_42_ = v_a_35_;
v_isShared_43_ = v_isSharedCheck_51_;
goto v_resetjp_41_;
}
else
{
lean_inc(v_snd_40_);
lean_inc(v_fst_39_);
lean_dec(v_a_35_);
v___x_42_ = lean_box(0);
v_isShared_43_ = v_isSharedCheck_51_;
goto v_resetjp_41_;
}
v_resetjp_41_:
{
lean_object* v___x_44_; lean_object* v___x_46_; 
v___x_44_ = lean_apply_1(v_interp_25_, v_fst_39_);
if (v_isShared_43_ == 0)
{
lean_ctor_set(v___x_42_, 0, v___x_44_);
v___x_46_ = v___x_42_;
goto v_reusejp_45_;
}
else
{
lean_object* v_reuseFailAlloc_50_; 
v_reuseFailAlloc_50_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_50_, 0, v___x_44_);
lean_ctor_set(v_reuseFailAlloc_50_, 1, v_snd_40_);
v___x_46_ = v_reuseFailAlloc_50_;
goto v_reusejp_45_;
}
v_reusejp_45_:
{
lean_object* v___x_48_; 
if (v_isShared_38_ == 0)
{
lean_ctor_set(v___x_37_, 0, v___x_46_);
v___x_48_ = v___x_37_;
goto v_reusejp_47_;
}
else
{
lean_object* v_reuseFailAlloc_49_; 
v_reuseFailAlloc_49_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_49_, 0, v___x_46_);
v___x_48_ = v_reuseFailAlloc_49_;
goto v_reusejp_47_;
}
v_reusejp_47_:
{
return v___x_48_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_SampleableExt_interpSample___redArg___boxed(lean_object* v_inst_53_, lean_object* v_a_54_, lean_object* v_a_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_plausible_Plausible_SampleableExt_interpSample___redArg(v_inst_53_, v_a_54_, v_a_55_);
lean_dec(v_a_55_);
return v_res_56_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_SampleableExt_interpSample(lean_object* v_00_u03b1_57_, lean_object* v_inst_58_, lean_object* v_a_59_, lean_object* v_a_60_){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lp_plausible_Plausible_SampleableExt_interpSample___redArg(v_inst_58_, v_a_59_, v_a_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_SampleableExt_interpSample___boxed(lean_object* v_00_u03b1_62_, lean_object* v_inst_63_, lean_object* v_a_64_, lean_object* v_a_65_){
_start:
{
lean_object* v_res_66_; 
v_res_66_ = lp_plausible_Plausible_SampleableExt_interpSample(v_00_u03b1_62_, v_inst_63_, v_a_64_, v_a_65_);
lean_dec(v_a_65_);
return v_res_66_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_arbitraryProxy___redArg(lean_object* v_inst_67_){
_start:
{
lean_object* v_sample_68_; 
v_sample_68_ = lean_ctor_get(v_inst_67_, 2);
lean_inc_ref(v_sample_68_);
return v_sample_68_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_arbitraryProxy___redArg___boxed(lean_object* v_inst_69_){
_start:
{
lean_object* v_res_70_; 
v_res_70_ = lp_plausible_Plausible_arbitraryProxy___redArg(v_inst_69_);
lean_dec_ref(v_inst_69_);
return v_res_70_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_arbitraryProxy(lean_object* v_00_u03b1_71_, lean_object* v_inst_72_){
_start:
{
lean_object* v_sample_73_; 
v_sample_73_ = lean_ctor_get(v_inst_72_, 2);
lean_inc_ref(v_sample_73_);
return v_sample_73_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_arbitraryProxy___boxed(lean_object* v_00_u03b1_74_, lean_object* v_inst_75_){
_start:
{
lean_object* v_res_76_; 
v_res_76_ = lp_plausible_Plausible_arbitraryProxy(v_00_u03b1_74_, v_inst_75_);
lean_dec_ref(v_inst_75_);
return v_res_76_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Sum_SampleableExt___redArg___lam__0(lean_object* v_interp_77_, lean_object* v_interp_78_, lean_object* v_s_79_){
_start:
{
if (lean_obj_tag(v_s_79_) == 0)
{
lean_object* v_val_80_; lean_object* v___x_82_; uint8_t v_isShared_83_; uint8_t v_isSharedCheck_88_; 
lean_dec(v_interp_78_);
v_val_80_ = lean_ctor_get(v_s_79_, 0);
v_isSharedCheck_88_ = !lean_is_exclusive(v_s_79_);
if (v_isSharedCheck_88_ == 0)
{
v___x_82_ = v_s_79_;
v_isShared_83_ = v_isSharedCheck_88_;
goto v_resetjp_81_;
}
else
{
lean_inc(v_val_80_);
lean_dec(v_s_79_);
v___x_82_ = lean_box(0);
v_isShared_83_ = v_isSharedCheck_88_;
goto v_resetjp_81_;
}
v_resetjp_81_:
{
lean_object* v___x_84_; lean_object* v___x_86_; 
v___x_84_ = lean_apply_1(v_interp_77_, v_val_80_);
if (v_isShared_83_ == 0)
{
lean_ctor_set(v___x_82_, 0, v___x_84_);
v___x_86_ = v___x_82_;
goto v_reusejp_85_;
}
else
{
lean_object* v_reuseFailAlloc_87_; 
v_reuseFailAlloc_87_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_87_, 0, v___x_84_);
v___x_86_ = v_reuseFailAlloc_87_;
goto v_reusejp_85_;
}
v_reusejp_85_:
{
return v___x_86_;
}
}
}
else
{
lean_object* v_val_89_; lean_object* v___x_91_; uint8_t v_isShared_92_; uint8_t v_isSharedCheck_97_; 
lean_dec(v_interp_77_);
v_val_89_ = lean_ctor_get(v_s_79_, 0);
v_isSharedCheck_97_ = !lean_is_exclusive(v_s_79_);
if (v_isSharedCheck_97_ == 0)
{
v___x_91_ = v_s_79_;
v_isShared_92_ = v_isSharedCheck_97_;
goto v_resetjp_90_;
}
else
{
lean_inc(v_val_89_);
lean_dec(v_s_79_);
v___x_91_ = lean_box(0);
v_isShared_92_ = v_isSharedCheck_97_;
goto v_resetjp_90_;
}
v_resetjp_90_:
{
lean_object* v___x_93_; lean_object* v___x_95_; 
v___x_93_ = lean_apply_1(v_interp_78_, v_val_89_);
if (v_isShared_92_ == 0)
{
lean_ctor_set(v___x_91_, 0, v___x_93_);
v___x_95_ = v___x_91_;
goto v_reusejp_94_;
}
else
{
lean_object* v_reuseFailAlloc_96_; 
v_reuseFailAlloc_96_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_96_, 0, v___x_93_);
v___x_95_ = v_reuseFailAlloc_96_;
goto v_reusejp_94_;
}
v_reusejp_94_:
{
return v___x_95_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Sum_SampleableExt___redArg(lean_object* v_inst_98_, lean_object* v_inst_99_){
_start:
{
lean_object* v_proxyRepr_100_; lean_object* v_shrink_101_; lean_object* v_sample_102_; lean_object* v_interp_103_; lean_object* v_proxyRepr_104_; lean_object* v_shrink_105_; lean_object* v_sample_106_; lean_object* v_interp_107_; lean_object* v___x_109_; uint8_t v_isShared_110_; uint8_t v_isSharedCheck_118_; 
v_proxyRepr_100_ = lean_ctor_get(v_inst_98_, 0);
lean_inc_ref(v_proxyRepr_100_);
v_shrink_101_ = lean_ctor_get(v_inst_98_, 1);
lean_inc_ref(v_shrink_101_);
v_sample_102_ = lean_ctor_get(v_inst_98_, 2);
lean_inc_ref(v_sample_102_);
v_interp_103_ = lean_ctor_get(v_inst_98_, 3);
lean_inc(v_interp_103_);
lean_dec_ref(v_inst_98_);
v_proxyRepr_104_ = lean_ctor_get(v_inst_99_, 0);
v_shrink_105_ = lean_ctor_get(v_inst_99_, 1);
v_sample_106_ = lean_ctor_get(v_inst_99_, 2);
v_interp_107_ = lean_ctor_get(v_inst_99_, 3);
v_isSharedCheck_118_ = !lean_is_exclusive(v_inst_99_);
if (v_isSharedCheck_118_ == 0)
{
v___x_109_ = v_inst_99_;
v_isShared_110_ = v_isSharedCheck_118_;
goto v_resetjp_108_;
}
else
{
lean_inc(v_interp_107_);
lean_inc(v_sample_106_);
lean_inc(v_shrink_105_);
lean_inc(v_proxyRepr_104_);
lean_dec(v_inst_99_);
v___x_109_ = lean_box(0);
v_isShared_110_ = v_isSharedCheck_118_;
goto v_resetjp_108_;
}
v_resetjp_108_:
{
lean_object* v___f_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_116_; 
v___f_111_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Sum_SampleableExt___redArg___lam__0), 3, 2);
lean_closure_set(v___f_111_, 0, v_interp_103_);
lean_closure_set(v___f_111_, 1, v_interp_107_);
v___x_112_ = lean_alloc_closure((void*)(l_Sum_repr___boxed), 6, 4);
lean_closure_set(v___x_112_, 0, lean_box(0));
lean_closure_set(v___x_112_, 1, lean_box(0));
lean_closure_set(v___x_112_, 2, v_proxyRepr_100_);
lean_closure_set(v___x_112_, 3, v_proxyRepr_104_);
v___x_113_ = lp_plausible_Plausible_instShrinkableSum___redArg(v_shrink_101_, v_shrink_105_);
v___x_114_ = lp_plausible_Plausible_Sum_Arbitrary___redArg(v_sample_102_, v_sample_106_);
if (v_isShared_110_ == 0)
{
lean_ctor_set(v___x_109_, 3, v___f_111_);
lean_ctor_set(v___x_109_, 2, v___x_114_);
lean_ctor_set(v___x_109_, 1, v___x_113_);
lean_ctor_set(v___x_109_, 0, v___x_112_);
v___x_116_ = v___x_109_;
goto v_reusejp_115_;
}
else
{
lean_object* v_reuseFailAlloc_117_; 
v_reuseFailAlloc_117_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_117_, 0, v___x_112_);
lean_ctor_set(v_reuseFailAlloc_117_, 1, v___x_113_);
lean_ctor_set(v_reuseFailAlloc_117_, 2, v___x_114_);
lean_ctor_set(v_reuseFailAlloc_117_, 3, v___f_111_);
v___x_116_ = v_reuseFailAlloc_117_;
goto v_reusejp_115_;
}
v_reusejp_115_:
{
return v___x_116_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Sum_SampleableExt(lean_object* v_00_u03b1_119_, lean_object* v_00_u03b2_120_, lean_object* v_inst_121_, lean_object* v_inst_122_){
_start:
{
lean_object* v___x_123_; 
v___x_123_ = lp_plausible_Plausible_Sum_SampleableExt___redArg(v_inst_121_, v_inst_122_);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instSampleableExtSigma___redArg___lam__0(lean_object* v_interp_124_, lean_object* v_interp_125_, lean_object* v_s_126_){
_start:
{
lean_object* v_fst_127_; lean_object* v_snd_128_; lean_object* v___x_130_; uint8_t v_isShared_131_; uint8_t v_isSharedCheck_137_; 
v_fst_127_ = lean_ctor_get(v_s_126_, 0);
v_snd_128_ = lean_ctor_get(v_s_126_, 1);
v_isSharedCheck_137_ = !lean_is_exclusive(v_s_126_);
if (v_isSharedCheck_137_ == 0)
{
v___x_130_ = v_s_126_;
v_isShared_131_ = v_isSharedCheck_137_;
goto v_resetjp_129_;
}
else
{
lean_inc(v_snd_128_);
lean_inc(v_fst_127_);
lean_dec(v_s_126_);
v___x_130_ = lean_box(0);
v_isShared_131_ = v_isSharedCheck_137_;
goto v_resetjp_129_;
}
v_resetjp_129_:
{
lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_135_; 
v___x_132_ = lean_apply_1(v_interp_124_, v_fst_127_);
v___x_133_ = lean_apply_1(v_interp_125_, v_snd_128_);
if (v_isShared_131_ == 0)
{
lean_ctor_set(v___x_130_, 1, v___x_133_);
lean_ctor_set(v___x_130_, 0, v___x_132_);
v___x_135_ = v___x_130_;
goto v_reusejp_134_;
}
else
{
lean_object* v_reuseFailAlloc_136_; 
v_reuseFailAlloc_136_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_136_, 0, v___x_132_);
lean_ctor_set(v_reuseFailAlloc_136_, 1, v___x_133_);
v___x_135_ = v_reuseFailAlloc_136_;
goto v_reusejp_134_;
}
v_reusejp_134_:
{
return v___x_135_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instSampleableExtSigma___redArg___lam__1(lean_object* v_proxyRepr_138_, lean_object* v_x_139_, lean_object* v___y_140_, lean_object* v___y_141_){
_start:
{
lean_object* v___x_142_; 
v___x_142_ = lean_apply_2(v_proxyRepr_138_, v___y_140_, v___y_141_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instSampleableExtSigma___redArg___lam__1___boxed(lean_object* v_proxyRepr_143_, lean_object* v_x_144_, lean_object* v___y_145_, lean_object* v___y_146_){
_start:
{
lean_object* v_res_147_; 
v_res_147_ = lp_plausible_Plausible_instSampleableExtSigma___redArg___lam__1(v_proxyRepr_143_, v_x_144_, v___y_145_, v___y_146_);
lean_dec(v_x_144_);
return v_res_147_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instSampleableExtSigma___redArg(lean_object* v_inst_148_, lean_object* v_inst_149_){
_start:
{
lean_object* v_proxyRepr_150_; lean_object* v_shrink_151_; lean_object* v_sample_152_; lean_object* v_interp_153_; lean_object* v_proxyRepr_154_; lean_object* v_shrink_155_; lean_object* v_sample_156_; lean_object* v_interp_157_; lean_object* v___x_159_; uint8_t v_isShared_160_; uint8_t v_isSharedCheck_169_; 
v_proxyRepr_150_ = lean_ctor_get(v_inst_148_, 0);
lean_inc_ref(v_proxyRepr_150_);
v_shrink_151_ = lean_ctor_get(v_inst_148_, 1);
lean_inc_ref(v_shrink_151_);
v_sample_152_ = lean_ctor_get(v_inst_148_, 2);
lean_inc_ref(v_sample_152_);
v_interp_153_ = lean_ctor_get(v_inst_148_, 3);
lean_inc(v_interp_153_);
lean_dec_ref(v_inst_148_);
v_proxyRepr_154_ = lean_ctor_get(v_inst_149_, 0);
v_shrink_155_ = lean_ctor_get(v_inst_149_, 1);
v_sample_156_ = lean_ctor_get(v_inst_149_, 2);
v_interp_157_ = lean_ctor_get(v_inst_149_, 3);
v_isSharedCheck_169_ = !lean_is_exclusive(v_inst_149_);
if (v_isSharedCheck_169_ == 0)
{
v___x_159_ = v_inst_149_;
v_isShared_160_ = v_isSharedCheck_169_;
goto v_resetjp_158_;
}
else
{
lean_inc(v_interp_157_);
lean_inc(v_sample_156_);
lean_inc(v_shrink_155_);
lean_inc(v_proxyRepr_154_);
lean_dec(v_inst_149_);
v___x_159_ = lean_box(0);
v_isShared_160_ = v_isSharedCheck_169_;
goto v_resetjp_158_;
}
v_resetjp_158_:
{
lean_object* v___f_161_; lean_object* v___f_162_; lean_object* v___f_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_167_; 
v___f_161_ = lean_alloc_closure((void*)(lp_plausible_Plausible_instSampleableExtSigma___redArg___lam__0), 3, 2);
lean_closure_set(v___f_161_, 0, v_interp_153_);
lean_closure_set(v___f_161_, 1, v_interp_157_);
v___f_162_ = lean_alloc_closure((void*)(lp_plausible_Plausible_instSampleableExtSigma___redArg___lam__1___boxed), 4, 1);
lean_closure_set(v___f_162_, 0, v_proxyRepr_154_);
v___f_163_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Sigma_shrinkable___redArg___lam__2), 3, 2);
lean_closure_set(v___f_163_, 0, v_shrink_151_);
lean_closure_set(v___f_163_, 1, v_shrink_155_);
v___x_164_ = lp_plausible_Plausible_Sigma_Arbitrary___redArg(v_sample_152_, v_sample_156_);
v___x_165_ = lean_alloc_closure((void*)(l_Sigma_repr___boxed), 6, 4);
lean_closure_set(v___x_165_, 0, lean_box(0));
lean_closure_set(v___x_165_, 1, lean_box(0));
lean_closure_set(v___x_165_, 2, v_proxyRepr_150_);
lean_closure_set(v___x_165_, 3, v___f_162_);
if (v_isShared_160_ == 0)
{
lean_ctor_set(v___x_159_, 3, v___f_161_);
lean_ctor_set(v___x_159_, 2, v___x_164_);
lean_ctor_set(v___x_159_, 1, v___f_163_);
lean_ctor_set(v___x_159_, 0, v___x_165_);
v___x_167_ = v___x_159_;
goto v_reusejp_166_;
}
else
{
lean_object* v_reuseFailAlloc_168_; 
v_reuseFailAlloc_168_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_168_, 0, v___x_165_);
lean_ctor_set(v_reuseFailAlloc_168_, 1, v___f_163_);
lean_ctor_set(v_reuseFailAlloc_168_, 2, v___x_164_);
lean_ctor_set(v_reuseFailAlloc_168_, 3, v___f_161_);
v___x_167_ = v_reuseFailAlloc_168_;
goto v_reusejp_166_;
}
v_reusejp_166_:
{
return v___x_167_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instSampleableExtSigma(lean_object* v_00_u03b1_170_, lean_object* v_00_u03b2_171_, lean_object* v_inst_172_, lean_object* v_inst_173_){
_start:
{
lean_object* v___x_174_; 
v___x_174_ = lp_plausible_Plausible_instSampleableExtSigma___redArg(v_inst_172_, v_inst_173_);
return v___x_174_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Option_sampleableExt___redArg___lam__0(lean_object* v_interp_175_, lean_object* v_o_176_){
_start:
{
if (lean_obj_tag(v_o_176_) == 0)
{
lean_object* v___x_177_; 
lean_dec(v_interp_175_);
v___x_177_ = lean_box(0);
return v___x_177_;
}
else
{
lean_object* v_val_178_; lean_object* v___x_180_; uint8_t v_isShared_181_; uint8_t v_isSharedCheck_186_; 
v_val_178_ = lean_ctor_get(v_o_176_, 0);
v_isSharedCheck_186_ = !lean_is_exclusive(v_o_176_);
if (v_isSharedCheck_186_ == 0)
{
v___x_180_ = v_o_176_;
v_isShared_181_ = v_isSharedCheck_186_;
goto v_resetjp_179_;
}
else
{
lean_inc(v_val_178_);
lean_dec(v_o_176_);
v___x_180_ = lean_box(0);
v_isShared_181_ = v_isSharedCheck_186_;
goto v_resetjp_179_;
}
v_resetjp_179_:
{
lean_object* v___x_182_; lean_object* v___x_184_; 
v___x_182_ = lean_apply_1(v_interp_175_, v_val_178_);
if (v_isShared_181_ == 0)
{
lean_ctor_set(v___x_180_, 0, v___x_182_);
v___x_184_ = v___x_180_;
goto v_reusejp_183_;
}
else
{
lean_object* v_reuseFailAlloc_185_; 
v_reuseFailAlloc_185_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_185_, 0, v___x_182_);
v___x_184_ = v_reuseFailAlloc_185_;
goto v_reusejp_183_;
}
v_reusejp_183_:
{
return v___x_184_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Option_sampleableExt___redArg(lean_object* v_inst_187_){
_start:
{
lean_object* v_proxyRepr_188_; lean_object* v_shrink_189_; lean_object* v_sample_190_; lean_object* v_interp_191_; lean_object* v___x_193_; uint8_t v_isShared_194_; uint8_t v_isSharedCheck_202_; 
v_proxyRepr_188_ = lean_ctor_get(v_inst_187_, 0);
v_shrink_189_ = lean_ctor_get(v_inst_187_, 1);
v_sample_190_ = lean_ctor_get(v_inst_187_, 2);
v_interp_191_ = lean_ctor_get(v_inst_187_, 3);
v_isSharedCheck_202_ = !lean_is_exclusive(v_inst_187_);
if (v_isSharedCheck_202_ == 0)
{
v___x_193_ = v_inst_187_;
v_isShared_194_ = v_isSharedCheck_202_;
goto v_resetjp_192_;
}
else
{
lean_inc(v_interp_191_);
lean_inc(v_sample_190_);
lean_inc(v_shrink_189_);
lean_inc(v_proxyRepr_188_);
lean_dec(v_inst_187_);
v___x_193_ = lean_box(0);
v_isShared_194_ = v_isSharedCheck_202_;
goto v_resetjp_192_;
}
v_resetjp_192_:
{
lean_object* v___f_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_200_; 
v___f_195_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Option_sampleableExt___redArg___lam__0), 2, 1);
lean_closure_set(v___f_195_, 0, v_interp_191_);
v___x_196_ = lean_alloc_closure((void*)(l_Option_repr___boxed), 4, 2);
lean_closure_set(v___x_196_, 0, lean_box(0));
lean_closure_set(v___x_196_, 1, v_proxyRepr_188_);
v___x_197_ = lp_plausible_Plausible_Option_shrinkable___redArg(v_shrink_189_);
v___x_198_ = lp_plausible_Plausible_Option_Arbitrary___redArg(v_sample_190_);
if (v_isShared_194_ == 0)
{
lean_ctor_set(v___x_193_, 3, v___f_195_);
lean_ctor_set(v___x_193_, 2, v___x_198_);
lean_ctor_set(v___x_193_, 1, v___x_197_);
lean_ctor_set(v___x_193_, 0, v___x_196_);
v___x_200_ = v___x_193_;
goto v_reusejp_199_;
}
else
{
lean_object* v_reuseFailAlloc_201_; 
v_reuseFailAlloc_201_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_201_, 0, v___x_196_);
lean_ctor_set(v_reuseFailAlloc_201_, 1, v___x_197_);
lean_ctor_set(v_reuseFailAlloc_201_, 2, v___x_198_);
lean_ctor_set(v_reuseFailAlloc_201_, 3, v___f_195_);
v___x_200_ = v_reuseFailAlloc_201_;
goto v_reusejp_199_;
}
v_reusejp_199_:
{
return v___x_200_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Option_sampleableExt(lean_object* v_00_u03b1_203_, lean_object* v_inst_204_){
_start:
{
lean_object* v___x_205_; 
v___x_205_ = lp_plausible_Plausible_Option_sampleableExt___redArg(v_inst_204_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Prod_sampleableExt___redArg(lean_object* v_inst_206_, lean_object* v_inst_207_){
_start:
{
lean_object* v_proxyRepr_208_; lean_object* v_shrink_209_; lean_object* v_sample_210_; lean_object* v_interp_211_; lean_object* v_proxyRepr_212_; lean_object* v_shrink_213_; lean_object* v_sample_214_; lean_object* v_interp_215_; lean_object* v___x_217_; uint8_t v_isShared_218_; uint8_t v_isSharedCheck_227_; 
v_proxyRepr_208_ = lean_ctor_get(v_inst_206_, 0);
lean_inc_ref(v_proxyRepr_208_);
v_shrink_209_ = lean_ctor_get(v_inst_206_, 1);
lean_inc_ref(v_shrink_209_);
v_sample_210_ = lean_ctor_get(v_inst_206_, 2);
lean_inc_ref(v_sample_210_);
v_interp_211_ = lean_ctor_get(v_inst_206_, 3);
lean_inc(v_interp_211_);
lean_dec_ref(v_inst_206_);
v_proxyRepr_212_ = lean_ctor_get(v_inst_207_, 0);
v_shrink_213_ = lean_ctor_get(v_inst_207_, 1);
v_sample_214_ = lean_ctor_get(v_inst_207_, 2);
v_interp_215_ = lean_ctor_get(v_inst_207_, 3);
v_isSharedCheck_227_ = !lean_is_exclusive(v_inst_207_);
if (v_isSharedCheck_227_ == 0)
{
v___x_217_ = v_inst_207_;
v_isShared_218_ = v_isSharedCheck_227_;
goto v_resetjp_216_;
}
else
{
lean_inc(v_interp_215_);
lean_inc(v_sample_214_);
lean_inc(v_shrink_213_);
lean_inc(v_proxyRepr_212_);
lean_dec(v_inst_207_);
v___x_217_ = lean_box(0);
v_isShared_218_ = v_isSharedCheck_227_;
goto v_resetjp_216_;
}
v_resetjp_216_:
{
lean_object* v___f_219_; lean_object* v___x_220_; lean_object* v___f_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_225_; 
v___f_219_ = lean_alloc_closure((void*)(l_instReprTupleOfRepr___redArg___lam__0), 3, 1);
lean_closure_set(v___f_219_, 0, v_proxyRepr_212_);
v___x_220_ = lean_alloc_closure((void*)(l_Prod_repr___boxed), 6, 4);
lean_closure_set(v___x_220_, 0, lean_box(0));
lean_closure_set(v___x_220_, 1, lean_box(0));
lean_closure_set(v___x_220_, 2, v_proxyRepr_208_);
lean_closure_set(v___x_220_, 3, v___f_219_);
v___f_221_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Prod_shrinkable___redArg___lam__2), 3, 2);
lean_closure_set(v___f_221_, 0, v_shrink_209_);
lean_closure_set(v___f_221_, 1, v_shrink_213_);
v___x_222_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Gen_prodOf___boxed), 6, 4);
lean_closure_set(v___x_222_, 0, lean_box(0));
lean_closure_set(v___x_222_, 1, lean_box(0));
lean_closure_set(v___x_222_, 2, v_sample_210_);
lean_closure_set(v___x_222_, 3, v_sample_214_);
v___x_223_ = lean_alloc_closure((void*)(l_Prod_map), 7, 6);
lean_closure_set(v___x_223_, 0, lean_box(0));
lean_closure_set(v___x_223_, 1, lean_box(0));
lean_closure_set(v___x_223_, 2, lean_box(0));
lean_closure_set(v___x_223_, 3, lean_box(0));
lean_closure_set(v___x_223_, 4, v_interp_211_);
lean_closure_set(v___x_223_, 5, v_interp_215_);
if (v_isShared_218_ == 0)
{
lean_ctor_set(v___x_217_, 3, v___x_223_);
lean_ctor_set(v___x_217_, 2, v___x_222_);
lean_ctor_set(v___x_217_, 1, v___f_221_);
lean_ctor_set(v___x_217_, 0, v___x_220_);
v___x_225_ = v___x_217_;
goto v_reusejp_224_;
}
else
{
lean_object* v_reuseFailAlloc_226_; 
v_reuseFailAlloc_226_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_226_, 0, v___x_220_);
lean_ctor_set(v_reuseFailAlloc_226_, 1, v___f_221_);
lean_ctor_set(v_reuseFailAlloc_226_, 2, v___x_222_);
lean_ctor_set(v_reuseFailAlloc_226_, 3, v___x_223_);
v___x_225_ = v_reuseFailAlloc_226_;
goto v_reusejp_224_;
}
v_reusejp_224_:
{
return v___x_225_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Prod_sampleableExt(lean_object* v_00_u03b1_228_, lean_object* v_00_u03b2_229_, lean_object* v_inst_230_, lean_object* v_inst_231_){
_start:
{
lean_object* v___x_232_; 
v___x_232_ = lp_plausible_Plausible_Prod_sampleableExt___redArg(v_inst_230_, v_inst_231_);
return v___x_232_;
}
}
static lean_object* _init_lp_plausible_Plausible_Prop_sampleableExt___closed__2(void){
_start:
{
lean_object* v___x_235_; lean_object* v___f_236_; lean_object* v___x_237_; lean_object* v___x_238_; 
v___x_235_ = lp_plausible_Plausible_Bool_Arbitrary;
v___f_236_ = ((lean_object*)(lp_plausible_Plausible_Prop_sampleableExt___closed__1));
v___x_237_ = ((lean_object*)(lp_plausible_Plausible_Prop_sampleableExt___closed__0));
v___x_238_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_238_, 0, v___x_237_);
lean_ctor_set(v___x_238_, 1, v___f_236_);
lean_ctor_set(v___x_238_, 2, v___x_235_);
lean_ctor_set(v___x_238_, 3, lean_box(0));
return v___x_238_;
}
}
static lean_object* _init_lp_plausible_Plausible_Prop_sampleableExt(void){
_start:
{
lean_object* v___x_239_; 
v___x_239_ = lean_obj_once(&lp_plausible_Plausible_Prop_sampleableExt___closed__2, &lp_plausible_Plausible_Prop_sampleableExt___closed__2_once, _init_lp_plausible_Plausible_Prop_sampleableExt___closed__2);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_List_sampleableExt___redArg(lean_object* v_inst_240_){
_start:
{
lean_object* v_proxyRepr_241_; lean_object* v_shrink_242_; lean_object* v_sample_243_; lean_object* v_interp_244_; lean_object* v___x_246_; uint8_t v_isShared_247_; uint8_t v_isSharedCheck_255_; 
v_proxyRepr_241_ = lean_ctor_get(v_inst_240_, 0);
v_shrink_242_ = lean_ctor_get(v_inst_240_, 1);
v_sample_243_ = lean_ctor_get(v_inst_240_, 2);
v_interp_244_ = lean_ctor_get(v_inst_240_, 3);
v_isSharedCheck_255_ = !lean_is_exclusive(v_inst_240_);
if (v_isSharedCheck_255_ == 0)
{
v___x_246_ = v_inst_240_;
v_isShared_247_ = v_isSharedCheck_255_;
goto v_resetjp_245_;
}
else
{
lean_inc(v_interp_244_);
lean_inc(v_sample_243_);
lean_inc(v_shrink_242_);
lean_inc(v_proxyRepr_241_);
lean_dec(v_inst_240_);
v___x_246_ = lean_box(0);
v_isShared_247_ = v_isSharedCheck_255_;
goto v_resetjp_245_;
}
v_resetjp_245_:
{
lean_object* v___x_248_; lean_object* v___f_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_253_; 
v___x_248_ = lean_alloc_closure((void*)(l_List_repr___boxed), 4, 2);
lean_closure_set(v___x_248_, 0, lean_box(0));
lean_closure_set(v___x_248_, 1, v_proxyRepr_241_);
v___f_249_ = lean_alloc_closure((void*)(lp_plausible_Plausible_List_shrinkable___redArg___lam__4), 2, 1);
lean_closure_set(v___f_249_, 0, v_shrink_242_);
v___x_250_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Gen_listOf___boxed), 4, 2);
lean_closure_set(v___x_250_, 0, lean_box(0));
lean_closure_set(v___x_250_, 1, v_sample_243_);
v___x_251_ = lean_alloc_closure((void*)(l_List_mapTR), 4, 3);
lean_closure_set(v___x_251_, 0, lean_box(0));
lean_closure_set(v___x_251_, 1, lean_box(0));
lean_closure_set(v___x_251_, 2, v_interp_244_);
if (v_isShared_247_ == 0)
{
lean_ctor_set(v___x_246_, 3, v___x_251_);
lean_ctor_set(v___x_246_, 2, v___x_250_);
lean_ctor_set(v___x_246_, 1, v___f_249_);
lean_ctor_set(v___x_246_, 0, v___x_248_);
v___x_253_ = v___x_246_;
goto v_reusejp_252_;
}
else
{
lean_object* v_reuseFailAlloc_254_; 
v_reuseFailAlloc_254_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_254_, 0, v___x_248_);
lean_ctor_set(v_reuseFailAlloc_254_, 1, v___f_249_);
lean_ctor_set(v_reuseFailAlloc_254_, 2, v___x_250_);
lean_ctor_set(v_reuseFailAlloc_254_, 3, v___x_251_);
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
LEAN_EXPORT lean_object* lp_plausible_Plausible_List_sampleableExt(lean_object* v_00_u03b1_256_, lean_object* v_inst_257_){
_start:
{
lean_object* v___x_258_; 
v___x_258_ = lp_plausible_Plausible_List_sampleableExt___redArg(v_inst_257_);
return v___x_258_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_ULift_sampleableExt___redArg___lam__0(lean_object* v_interp_259_, lean_object* v_a_260_){
_start:
{
lean_object* v___x_261_; 
v___x_261_ = lean_apply_1(v_interp_259_, v_a_260_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_ULift_sampleableExt___redArg(lean_object* v_inst_262_){
_start:
{
lean_object* v_proxyRepr_263_; lean_object* v_shrink_264_; lean_object* v_sample_265_; lean_object* v_interp_266_; lean_object* v___x_268_; uint8_t v_isShared_269_; uint8_t v_isSharedCheck_274_; 
v_proxyRepr_263_ = lean_ctor_get(v_inst_262_, 0);
v_shrink_264_ = lean_ctor_get(v_inst_262_, 1);
v_sample_265_ = lean_ctor_get(v_inst_262_, 2);
v_interp_266_ = lean_ctor_get(v_inst_262_, 3);
v_isSharedCheck_274_ = !lean_is_exclusive(v_inst_262_);
if (v_isSharedCheck_274_ == 0)
{
v___x_268_ = v_inst_262_;
v_isShared_269_ = v_isSharedCheck_274_;
goto v_resetjp_267_;
}
else
{
lean_inc(v_interp_266_);
lean_inc(v_sample_265_);
lean_inc(v_shrink_264_);
lean_inc(v_proxyRepr_263_);
lean_dec(v_inst_262_);
v___x_268_ = lean_box(0);
v_isShared_269_ = v_isSharedCheck_274_;
goto v_resetjp_267_;
}
v_resetjp_267_:
{
lean_object* v___f_270_; lean_object* v___x_272_; 
v___f_270_ = lean_alloc_closure((void*)(lp_plausible_Plausible_ULift_sampleableExt___redArg___lam__0), 2, 1);
lean_closure_set(v___f_270_, 0, v_interp_266_);
if (v_isShared_269_ == 0)
{
lean_ctor_set(v___x_268_, 3, v___f_270_);
v___x_272_ = v___x_268_;
goto v_reusejp_271_;
}
else
{
lean_object* v_reuseFailAlloc_273_; 
v_reuseFailAlloc_273_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_273_, 0, v_proxyRepr_263_);
lean_ctor_set(v_reuseFailAlloc_273_, 1, v_shrink_264_);
lean_ctor_set(v_reuseFailAlloc_273_, 2, v_sample_265_);
lean_ctor_set(v_reuseFailAlloc_273_, 3, v___f_270_);
v___x_272_ = v_reuseFailAlloc_273_;
goto v_reusejp_271_;
}
v_reusejp_271_:
{
return v___x_272_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_ULift_sampleableExt(lean_object* v_00_u03b1_275_, lean_object* v_inst_276_){
_start:
{
lean_object* v___x_277_; 
v___x_277_ = lp_plausible_Plausible_ULift_sampleableExt___redArg(v_inst_276_);
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Array_sampleableExt___redArg(lean_object* v_inst_278_){
_start:
{
lean_object* v_proxyRepr_279_; lean_object* v_shrink_280_; lean_object* v_sample_281_; lean_object* v_interp_282_; lean_object* v___x_284_; uint8_t v_isShared_285_; uint8_t v_isSharedCheck_293_; 
v_proxyRepr_279_ = lean_ctor_get(v_inst_278_, 0);
v_shrink_280_ = lean_ctor_get(v_inst_278_, 1);
v_sample_281_ = lean_ctor_get(v_inst_278_, 2);
v_interp_282_ = lean_ctor_get(v_inst_278_, 3);
v_isSharedCheck_293_ = !lean_is_exclusive(v_inst_278_);
if (v_isSharedCheck_293_ == 0)
{
v___x_284_ = v_inst_278_;
v_isShared_285_ = v_isSharedCheck_293_;
goto v_resetjp_283_;
}
else
{
lean_inc(v_interp_282_);
lean_inc(v_sample_281_);
lean_inc(v_shrink_280_);
lean_inc(v_proxyRepr_279_);
lean_dec(v_inst_278_);
v___x_284_ = lean_box(0);
v_isShared_285_ = v_isSharedCheck_293_;
goto v_resetjp_283_;
}
v_resetjp_283_:
{
lean_object* v___f_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_291_; 
v___f_286_ = lean_alloc_closure((void*)(l_Array_instRepr___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_286_, 0, v_proxyRepr_279_);
v___x_287_ = lp_plausible_Plausible_Array_shrinkable___redArg(v_shrink_280_);
v___x_288_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Gen_arrayOf___boxed), 4, 2);
lean_closure_set(v___x_288_, 0, lean_box(0));
lean_closure_set(v___x_288_, 1, v_sample_281_);
v___x_289_ = lean_alloc_closure((void*)(l_Array_map), 4, 3);
lean_closure_set(v___x_289_, 0, lean_box(0));
lean_closure_set(v___x_289_, 1, lean_box(0));
lean_closure_set(v___x_289_, 2, v_interp_282_);
if (v_isShared_285_ == 0)
{
lean_ctor_set(v___x_284_, 3, v___x_289_);
lean_ctor_set(v___x_284_, 2, v___x_288_);
lean_ctor_set(v___x_284_, 1, v___x_287_);
lean_ctor_set(v___x_284_, 0, v___f_286_);
v___x_291_ = v___x_284_;
goto v_reusejp_290_;
}
else
{
lean_object* v_reuseFailAlloc_292_; 
v_reuseFailAlloc_292_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_292_, 0, v___f_286_);
lean_ctor_set(v_reuseFailAlloc_292_, 1, v___x_287_);
lean_ctor_set(v_reuseFailAlloc_292_, 2, v___x_288_);
lean_ctor_set(v_reuseFailAlloc_292_, 3, v___x_289_);
v___x_291_ = v_reuseFailAlloc_292_;
goto v_reusejp_290_;
}
v_reusejp_290_:
{
return v___x_291_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Array_sampleableExt(lean_object* v_00_u03b1_294_, lean_object* v_inst_295_){
_start:
{
lean_object* v___x_296_; 
v___x_296_ = lp_plausible_Plausible_Array_sampleableExt___redArg(v_inst_295_);
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_mk___redArg(lean_object* v_x_297_){
_start:
{
lean_inc(v_x_297_);
return v_x_297_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_mk___redArg___boxed(lean_object* v_x_298_){
_start:
{
lean_object* v_res_299_; 
v_res_299_ = lp_plausible_Plausible_NoShrink_mk___redArg(v_x_298_);
lean_dec(v_x_298_);
return v_res_299_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_mk(lean_object* v_00_u03b1_300_, lean_object* v_x_301_){
_start:
{
lean_inc(v_x_301_);
return v_x_301_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_mk___boxed(lean_object* v_00_u03b1_302_, lean_object* v_x_303_){
_start:
{
lean_object* v_res_304_; 
v_res_304_ = lp_plausible_Plausible_NoShrink_mk(v_00_u03b1_302_, v_x_303_);
lean_dec(v_x_303_);
return v_res_304_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_get___redArg(lean_object* v_x_305_){
_start:
{
lean_inc(v_x_305_);
return v_x_305_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_get___redArg___boxed(lean_object* v_x_306_){
_start:
{
lean_object* v_res_307_; 
v_res_307_ = lp_plausible_Plausible_NoShrink_get___redArg(v_x_306_);
lean_dec(v_x_306_);
return v_res_307_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_get(lean_object* v_00_u03b1_308_, lean_object* v_x_309_){
_start:
{
lean_inc(v_x_309_);
return v_x_309_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_get___boxed(lean_object* v_00_u03b1_310_, lean_object* v_x_311_){
_start:
{
lean_object* v_res_312_; 
v_res_312_ = lp_plausible_Plausible_NoShrink_get(v_00_u03b1_310_, v_x_311_);
lean_dec(v_x_311_);
return v_res_312_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_inhabited___redArg(lean_object* v_inst_313_){
_start:
{
lean_inc(v_inst_313_);
return v_inst_313_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_inhabited___redArg___boxed(lean_object* v_inst_314_){
_start:
{
lean_object* v_res_315_; 
v_res_315_ = lp_plausible_Plausible_NoShrink_inhabited___redArg(v_inst_314_);
lean_dec(v_inst_314_);
return v_res_315_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_inhabited(lean_object* v_00_u03b1_316_, lean_object* v_inst_317_){
_start:
{
lean_inc(v_inst_317_);
return v_inst_317_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_inhabited___boxed(lean_object* v_00_u03b1_318_, lean_object* v_inst_319_){
_start:
{
lean_object* v_res_320_; 
v_res_320_ = lp_plausible_Plausible_NoShrink_inhabited(v_00_u03b1_318_, v_inst_319_);
lean_dec(v_inst_319_);
return v_res_320_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_repr___redArg(lean_object* v_inst_321_){
_start:
{
lean_inc_ref(v_inst_321_);
return v_inst_321_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_repr___redArg___boxed(lean_object* v_inst_322_){
_start:
{
lean_object* v_res_323_; 
v_res_323_ = lp_plausible_Plausible_NoShrink_repr___redArg(v_inst_322_);
lean_dec_ref(v_inst_322_);
return v_res_323_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_repr(lean_object* v_00_u03b1_324_, lean_object* v_inst_325_){
_start:
{
lean_inc_ref(v_inst_325_);
return v_inst_325_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_repr___boxed(lean_object* v_00_u03b1_326_, lean_object* v_inst_327_){
_start:
{
lean_object* v_res_328_; 
v_res_328_ = lp_plausible_Plausible_NoShrink_repr(v_00_u03b1_326_, v_inst_327_);
lean_dec_ref(v_inst_327_);
return v_res_328_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_shrinkable___lam__0(lean_object* v_x_329_){
_start:
{
lean_object* v___x_330_; 
v___x_330_ = lean_box(0);
return v___x_330_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_shrinkable___lam__0___boxed(lean_object* v_x_331_){
_start:
{
lean_object* v_res_332_; 
v_res_332_ = lp_plausible_Plausible_NoShrink_shrinkable___lam__0(v_x_331_);
lean_dec(v_x_331_);
return v_res_332_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_shrinkable(lean_object* v_00_u03b1_334_){
_start:
{
lean_object* v___f_335_; 
v___f_335_ = ((lean_object*)(lp_plausible_Plausible_NoShrink_shrinkable___closed__0));
return v___f_335_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_arbitrary___redArg(lean_object* v_arb_336_){
_start:
{
lean_inc_ref(v_arb_336_);
return v_arb_336_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_arbitrary___redArg___boxed(lean_object* v_arb_337_){
_start:
{
lean_object* v_res_338_; 
v_res_338_ = lp_plausible_Plausible_NoShrink_arbitrary___redArg(v_arb_337_);
lean_dec_ref(v_arb_337_);
return v_res_338_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_arbitrary(lean_object* v_00_u03b1_339_, lean_object* v_arb_340_){
_start:
{
lean_inc_ref(v_arb_340_);
return v_arb_340_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_arbitrary___boxed(lean_object* v_00_u03b1_341_, lean_object* v_arb_342_){
_start:
{
lean_object* v_res_343_; 
v_res_343_ = lp_plausible_Plausible_NoShrink_arbitrary(v_00_u03b1_341_, v_arb_342_);
lean_dec_ref(v_arb_342_);
return v_res_343_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_sampleableExt___redArg(lean_object* v_inst_344_){
_start:
{
lean_object* v_proxyRepr_345_; lean_object* v_sample_346_; lean_object* v_interp_347_; lean_object* v___x_349_; uint8_t v_isShared_350_; uint8_t v_isSharedCheck_355_; 
v_proxyRepr_345_ = lean_ctor_get(v_inst_344_, 0);
v_sample_346_ = lean_ctor_get(v_inst_344_, 2);
v_interp_347_ = lean_ctor_get(v_inst_344_, 3);
v_isSharedCheck_355_ = !lean_is_exclusive(v_inst_344_);
if (v_isSharedCheck_355_ == 0)
{
lean_object* v_unused_356_; 
v_unused_356_ = lean_ctor_get(v_inst_344_, 1);
lean_dec(v_unused_356_);
v___x_349_ = v_inst_344_;
v_isShared_350_ = v_isSharedCheck_355_;
goto v_resetjp_348_;
}
else
{
lean_inc(v_interp_347_);
lean_inc(v_sample_346_);
lean_inc(v_proxyRepr_345_);
lean_dec(v_inst_344_);
v___x_349_ = lean_box(0);
v_isShared_350_ = v_isSharedCheck_355_;
goto v_resetjp_348_;
}
v_resetjp_348_:
{
lean_object* v___f_351_; lean_object* v___x_353_; 
v___f_351_ = ((lean_object*)(lp_plausible_Plausible_NoShrink_shrinkable___closed__0));
if (v_isShared_350_ == 0)
{
lean_ctor_set(v___x_349_, 1, v___f_351_);
v___x_353_ = v___x_349_;
goto v_reusejp_352_;
}
else
{
lean_object* v_reuseFailAlloc_354_; 
v_reuseFailAlloc_354_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_354_, 0, v_proxyRepr_345_);
lean_ctor_set(v_reuseFailAlloc_354_, 1, v___f_351_);
lean_ctor_set(v_reuseFailAlloc_354_, 2, v_sample_346_);
lean_ctor_set(v_reuseFailAlloc_354_, 3, v_interp_347_);
v___x_353_ = v_reuseFailAlloc_354_;
goto v_reusejp_352_;
}
v_reusejp_352_:
{
return v___x_353_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_sampleableExt(lean_object* v_00_u03b1_357_, lean_object* v_inst_358_, lean_object* v_inst_359_){
_start:
{
lean_object* v___x_360_; 
v___x_360_ = lp_plausible_Plausible_NoShrink_sampleableExt___redArg(v_inst_358_);
return v___x_360_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_NoShrink_sampleableExt___boxed(lean_object* v_00_u03b1_361_, lean_object* v_inst_362_, lean_object* v_inst_363_){
_start:
{
lean_object* v_res_364_; 
v_res_364_ = lp_plausible_Plausible_NoShrink_sampleableExt(v_00_u03b1_361_, v_inst_362_, v_inst_363_);
lean_dec_ref(v_inst_363_);
return v_res_364_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_instantiateLevelMVars___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__0___redArg(lean_object* v_l_365_, lean_object* v___y_366_){
_start:
{
lean_object* v___x_368_; lean_object* v_mctx_369_; lean_object* v___x_370_; lean_object* v_fst_371_; lean_object* v_snd_372_; lean_object* v___x_373_; lean_object* v_cache_374_; lean_object* v_zetaDeltaFVarIds_375_; lean_object* v_postponed_376_; lean_object* v_diag_377_; lean_object* v___x_379_; uint8_t v_isShared_380_; uint8_t v_isSharedCheck_386_; 
v___x_368_ = lean_st_ref_get(v___y_366_);
v_mctx_369_ = lean_ctor_get(v___x_368_, 0);
lean_inc_ref(v_mctx_369_);
lean_dec(v___x_368_);
v___x_370_ = lean_instantiate_level_mvars(v_mctx_369_, v_l_365_);
v_fst_371_ = lean_ctor_get(v___x_370_, 0);
lean_inc(v_fst_371_);
v_snd_372_ = lean_ctor_get(v___x_370_, 1);
lean_inc(v_snd_372_);
lean_dec_ref(v___x_370_);
v___x_373_ = lean_st_ref_take(v___y_366_);
v_cache_374_ = lean_ctor_get(v___x_373_, 1);
v_zetaDeltaFVarIds_375_ = lean_ctor_get(v___x_373_, 2);
v_postponed_376_ = lean_ctor_get(v___x_373_, 3);
v_diag_377_ = lean_ctor_get(v___x_373_, 4);
v_isSharedCheck_386_ = !lean_is_exclusive(v___x_373_);
if (v_isSharedCheck_386_ == 0)
{
lean_object* v_unused_387_; 
v_unused_387_ = lean_ctor_get(v___x_373_, 0);
lean_dec(v_unused_387_);
v___x_379_ = v___x_373_;
v_isShared_380_ = v_isSharedCheck_386_;
goto v_resetjp_378_;
}
else
{
lean_inc(v_diag_377_);
lean_inc(v_postponed_376_);
lean_inc(v_zetaDeltaFVarIds_375_);
lean_inc(v_cache_374_);
lean_dec(v___x_373_);
v___x_379_ = lean_box(0);
v_isShared_380_ = v_isSharedCheck_386_;
goto v_resetjp_378_;
}
v_resetjp_378_:
{
lean_object* v___x_382_; 
if (v_isShared_380_ == 0)
{
lean_ctor_set(v___x_379_, 0, v_fst_371_);
v___x_382_ = v___x_379_;
goto v_reusejp_381_;
}
else
{
lean_object* v_reuseFailAlloc_385_; 
v_reuseFailAlloc_385_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_385_, 0, v_fst_371_);
lean_ctor_set(v_reuseFailAlloc_385_, 1, v_cache_374_);
lean_ctor_set(v_reuseFailAlloc_385_, 2, v_zetaDeltaFVarIds_375_);
lean_ctor_set(v_reuseFailAlloc_385_, 3, v_postponed_376_);
lean_ctor_set(v_reuseFailAlloc_385_, 4, v_diag_377_);
v___x_382_ = v_reuseFailAlloc_385_;
goto v_reusejp_381_;
}
v_reusejp_381_:
{
lean_object* v___x_383_; lean_object* v___x_384_; 
v___x_383_ = lean_st_ref_set(v___y_366_, v___x_382_);
v___x_384_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_384_, 0, v_snd_372_);
return v___x_384_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_instantiateLevelMVars___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__0___redArg___boxed(lean_object* v_l_388_, lean_object* v___y_389_, lean_object* v___y_390_){
_start:
{
lean_object* v_res_391_; 
v_res_391_ = lp_plausible_Lean_instantiateLevelMVars___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__0___redArg(v_l_388_, v___y_389_);
lean_dec(v___y_389_);
return v_res_391_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_instantiateLevelMVars___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__0(lean_object* v_l_392_, lean_object* v___y_393_, lean_object* v___y_394_, lean_object* v___y_395_, lean_object* v___y_396_){
_start:
{
lean_object* v___x_398_; 
v___x_398_ = lp_plausible_Lean_instantiateLevelMVars___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__0___redArg(v_l_392_, v___y_394_);
return v___x_398_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_instantiateLevelMVars___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__0___boxed(lean_object* v_l_399_, lean_object* v___y_400_, lean_object* v___y_401_, lean_object* v___y_402_, lean_object* v___y_403_, lean_object* v___y_404_){
_start:
{
lean_object* v_res_405_; 
v_res_405_ = lp_plausible_Lean_instantiateLevelMVars___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__0(v_l_399_, v___y_400_, v___y_401_, v___y_402_, v___y_403_);
lean_dec(v___y_403_);
lean_dec_ref(v___y_402_);
lean_dec(v___y_401_);
lean_dec_ref(v___y_400_);
return v_res_405_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__1_spec__1(lean_object* v_msgData_406_, lean_object* v___y_407_, lean_object* v___y_408_, lean_object* v___y_409_, lean_object* v___y_410_){
_start:
{
lean_object* v___x_412_; lean_object* v_env_413_; lean_object* v___x_414_; lean_object* v_mctx_415_; lean_object* v_lctx_416_; lean_object* v_options_417_; lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; 
v___x_412_ = lean_st_ref_get(v___y_410_);
v_env_413_ = lean_ctor_get(v___x_412_, 0);
lean_inc_ref(v_env_413_);
lean_dec(v___x_412_);
v___x_414_ = lean_st_ref_get(v___y_408_);
v_mctx_415_ = lean_ctor_get(v___x_414_, 0);
lean_inc_ref(v_mctx_415_);
lean_dec(v___x_414_);
v_lctx_416_ = lean_ctor_get(v___y_407_, 2);
v_options_417_ = lean_ctor_get(v___y_409_, 2);
lean_inc_ref(v_options_417_);
lean_inc_ref(v_lctx_416_);
v___x_418_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_418_, 0, v_env_413_);
lean_ctor_set(v___x_418_, 1, v_mctx_415_);
lean_ctor_set(v___x_418_, 2, v_lctx_416_);
lean_ctor_set(v___x_418_, 3, v_options_417_);
v___x_419_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_419_, 0, v___x_418_);
lean_ctor_set(v___x_419_, 1, v_msgData_406_);
v___x_420_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_420_, 0, v___x_419_);
return v___x_420_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__1_spec__1___boxed(lean_object* v_msgData_421_, lean_object* v___y_422_, lean_object* v___y_423_, lean_object* v___y_424_, lean_object* v___y_425_, lean_object* v___y_426_){
_start:
{
lean_object* v_res_427_; 
v_res_427_ = lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__1_spec__1(v_msgData_421_, v___y_422_, v___y_423_, v___y_424_, v___y_425_);
lean_dec(v___y_425_);
lean_dec_ref(v___y_424_);
lean_dec(v___y_423_);
lean_dec_ref(v___y_422_);
return v_res_427_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__1___redArg(lean_object* v_msg_428_, lean_object* v___y_429_, lean_object* v___y_430_, lean_object* v___y_431_, lean_object* v___y_432_){
_start:
{
lean_object* v_ref_434_; lean_object* v___x_435_; lean_object* v_a_436_; lean_object* v___x_438_; uint8_t v_isShared_439_; uint8_t v_isSharedCheck_444_; 
v_ref_434_ = lean_ctor_get(v___y_431_, 5);
v___x_435_ = lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__1_spec__1(v_msg_428_, v___y_429_, v___y_430_, v___y_431_, v___y_432_);
v_a_436_ = lean_ctor_get(v___x_435_, 0);
v_isSharedCheck_444_ = !lean_is_exclusive(v___x_435_);
if (v_isSharedCheck_444_ == 0)
{
v___x_438_ = v___x_435_;
v_isShared_439_ = v_isSharedCheck_444_;
goto v_resetjp_437_;
}
else
{
lean_inc(v_a_436_);
lean_dec(v___x_435_);
v___x_438_ = lean_box(0);
v_isShared_439_ = v_isSharedCheck_444_;
goto v_resetjp_437_;
}
v_resetjp_437_:
{
lean_object* v___x_440_; lean_object* v___x_442_; 
lean_inc(v_ref_434_);
v___x_440_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_440_, 0, v_ref_434_);
lean_ctor_set(v___x_440_, 1, v_a_436_);
if (v_isShared_439_ == 0)
{
lean_ctor_set_tag(v___x_438_, 1);
lean_ctor_set(v___x_438_, 0, v___x_440_);
v___x_442_ = v___x_438_;
goto v_reusejp_441_;
}
else
{
lean_object* v_reuseFailAlloc_443_; 
v_reuseFailAlloc_443_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_443_, 0, v___x_440_);
v___x_442_ = v_reuseFailAlloc_443_;
goto v_reusejp_441_;
}
v_reusejp_441_:
{
return v___x_442_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__1___redArg___boxed(lean_object* v_msg_445_, lean_object* v___y_446_, lean_object* v___y_447_, lean_object* v___y_448_, lean_object* v___y_449_, lean_object* v___y_450_){
_start:
{
lean_object* v_res_451_; 
v_res_451_ = lp_plausible_Lean_throwError___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__1___redArg(v_msg_445_, v___y_446_, v___y_447_, v___y_448_, v___y_449_);
lean_dec(v___y_449_);
lean_dec_ref(v___y_448_);
lean_dec(v___y_447_);
lean_dec_ref(v___y_446_);
return v_res_451_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__17(void){
_start:
{
lean_object* v___x_486_; lean_object* v___x_487_; 
v___x_486_ = ((lean_object*)(lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__16));
v___x_487_ = l_Lean_stringToMessageData(v___x_486_);
return v___x_487_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__19(void){
_start:
{
lean_object* v___x_489_; lean_object* v___x_490_; 
v___x_489_ = ((lean_object*)(lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__18));
v___x_490_ = l_Lean_stringToMessageData(v___x_489_);
return v___x_490_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator(lean_object* v_e_491_, lean_object* v_a_492_, lean_object* v_a_493_, lean_object* v_a_494_, lean_object* v_a_495_){
_start:
{
lean_object* v___x_497_; 
lean_inc(v_a_495_);
lean_inc_ref(v_a_494_);
lean_inc(v_a_493_);
lean_inc_ref(v_a_492_);
lean_inc_ref(v_e_491_);
v___x_497_ = lean_infer_type(v_e_491_, v_a_492_, v_a_493_, v_a_494_, v_a_495_);
if (lean_obj_tag(v___x_497_) == 0)
{
lean_object* v_a_498_; lean_object* v___x_499_; 
v_a_498_ = lean_ctor_get(v___x_497_, 0);
lean_inc_n(v_a_498_, 2);
lean_dec_ref_known(v___x_497_, 1);
lean_inc(v_a_495_);
lean_inc_ref(v_a_494_);
lean_inc(v_a_493_);
lean_inc_ref(v_a_492_);
v___x_499_ = lean_infer_type(v_a_498_, v_a_492_, v_a_493_, v_a_494_, v_a_495_);
if (lean_obj_tag(v___x_499_) == 0)
{
lean_object* v_a_500_; lean_object* v___x_501_; 
v_a_500_ = lean_ctor_get(v___x_499_, 0);
lean_inc(v_a_500_);
lean_dec_ref_known(v___x_499_, 1);
lean_inc(v_a_495_);
lean_inc_ref(v_a_494_);
lean_inc(v_a_493_);
lean_inc_ref(v_a_492_);
v___x_501_ = lean_whnf(v_a_500_, v_a_492_, v_a_493_, v_a_494_, v_a_495_);
if (lean_obj_tag(v___x_501_) == 0)
{
lean_object* v_a_502_; 
v_a_502_ = lean_ctor_get(v___x_501_, 0);
lean_inc(v_a_502_);
lean_dec_ref_known(v___x_501_, 1);
if (lean_obj_tag(v_a_502_) == 3)
{
lean_object* v_u_503_; 
v_u_503_ = lean_ctor_get(v_a_502_, 0);
lean_inc(v_u_503_);
lean_dec_ref_known(v_a_502_, 1);
if (lean_obj_tag(v_u_503_) == 1)
{
lean_object* v_a_504_; lean_object* v___x_505_; 
v_a_504_ = lean_ctor_get(v_u_503_, 0);
lean_inc(v_a_504_);
lean_dec_ref_known(v_u_503_, 1);
v___x_505_ = l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(v_a_498_, v_a_493_);
if (lean_obj_tag(v___x_505_) == 0)
{
lean_object* v_a_506_; lean_object* v___y_508_; lean_object* v___y_509_; lean_object* v___y_510_; lean_object* v___y_511_; lean_object* v___x_565_; uint8_t v___x_566_; 
v_a_506_ = lean_ctor_get(v___x_505_, 0);
lean_inc(v_a_506_);
lean_dec_ref_known(v___x_505_, 1);
v___x_565_ = l_Lean_Expr_cleanupAnnotations(v_a_506_);
v___x_566_ = l_Lean_Expr_isApp(v___x_565_);
if (v___x_566_ == 0)
{
lean_dec_ref(v___x_565_);
v___y_508_ = v_a_492_;
v___y_509_ = v_a_493_;
v___y_510_ = v_a_494_;
v___y_511_ = v_a_495_;
goto v___jp_507_;
}
else
{
lean_object* v_arg_567_; lean_object* v___x_568_; lean_object* v___x_569_; uint8_t v___x_570_; 
v_arg_567_ = lean_ctor_get(v___x_565_, 1);
lean_inc_ref(v_arg_567_);
v___x_568_ = l_Lean_Expr_appFnCleanup___redArg(v___x_565_);
v___x_569_ = ((lean_object*)(lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__13));
v___x_570_ = l_Lean_Expr_isConstOf(v___x_568_, v___x_569_);
lean_dec_ref(v___x_568_);
if (v___x_570_ == 0)
{
lean_dec_ref(v_arg_567_);
v___y_508_ = v_a_492_;
v___y_509_ = v_a_493_;
v___y_510_ = v_a_494_;
v___y_511_ = v_a_495_;
goto v___jp_507_;
}
else
{
lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; 
v___x_571_ = ((lean_object*)(lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__15));
v___x_572_ = lean_box(0);
lean_inc(v_a_504_);
v___x_573_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_573_, 0, v_a_504_);
lean_ctor_set(v___x_573_, 1, v___x_572_);
v___x_574_ = l_Lean_mkConst(v___x_571_, v___x_573_);
lean_inc_ref(v_arg_567_);
v___x_575_ = l_Lean_Expr_app___override(v___x_574_, v_arg_567_);
v___x_576_ = lean_box(0);
v___x_577_ = l_Lean_Meta_synthInstance(v___x_575_, v___x_576_, v_a_492_, v_a_493_, v_a_494_, v_a_495_);
if (lean_obj_tag(v___x_577_) == 0)
{
lean_object* v_a_578_; lean_object* v___x_580_; uint8_t v_isShared_581_; uint8_t v_isSharedCheck_588_; 
v_a_578_ = lean_ctor_get(v___x_577_, 0);
v_isSharedCheck_588_ = !lean_is_exclusive(v___x_577_);
if (v_isSharedCheck_588_ == 0)
{
v___x_580_ = v___x_577_;
v_isShared_581_ = v_isSharedCheck_588_;
goto v_resetjp_579_;
}
else
{
lean_inc(v_a_578_);
lean_dec(v___x_577_);
v___x_580_ = lean_box(0);
v_isShared_581_ = v_isSharedCheck_588_;
goto v_resetjp_579_;
}
v_resetjp_579_:
{
lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_586_; 
v___x_582_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_582_, 0, v_a_578_);
lean_ctor_set(v___x_582_, 1, v_e_491_);
v___x_583_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_583_, 0, v_arg_567_);
lean_ctor_set(v___x_583_, 1, v___x_582_);
v___x_584_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_584_, 0, v_a_504_);
lean_ctor_set(v___x_584_, 1, v___x_583_);
if (v_isShared_581_ == 0)
{
lean_ctor_set(v___x_580_, 0, v___x_584_);
v___x_586_ = v___x_580_;
goto v_reusejp_585_;
}
else
{
lean_object* v_reuseFailAlloc_587_; 
v_reuseFailAlloc_587_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_587_, 0, v___x_584_);
v___x_586_ = v_reuseFailAlloc_587_;
goto v_reusejp_585_;
}
v_reusejp_585_:
{
return v___x_586_;
}
}
}
else
{
lean_object* v_a_589_; lean_object* v___x_591_; uint8_t v_isShared_592_; uint8_t v_isSharedCheck_596_; 
lean_dec_ref(v_arg_567_);
lean_dec(v_a_504_);
lean_dec_ref(v_e_491_);
v_a_589_ = lean_ctor_get(v___x_577_, 0);
v_isSharedCheck_596_ = !lean_is_exclusive(v___x_577_);
if (v_isSharedCheck_596_ == 0)
{
v___x_591_ = v___x_577_;
v_isShared_592_ = v_isSharedCheck_596_;
goto v_resetjp_590_;
}
else
{
lean_inc(v_a_589_);
lean_dec(v___x_577_);
v___x_591_ = lean_box(0);
v_isShared_592_ = v_isSharedCheck_596_;
goto v_resetjp_590_;
}
v_resetjp_590_:
{
lean_object* v___x_594_; 
if (v_isShared_592_ == 0)
{
v___x_594_ = v___x_591_;
goto v_reusejp_593_;
}
else
{
lean_object* v_reuseFailAlloc_595_; 
v_reuseFailAlloc_595_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_595_, 0, v_a_589_);
v___x_594_ = v_reuseFailAlloc_595_;
goto v_reusejp_593_;
}
v_reusejp_593_:
{
return v___x_594_;
}
}
}
}
}
v___jp_507_:
{
lean_object* v___x_512_; 
v___x_512_ = l_Lean_Meta_mkFreshLevelMVar(v___y_508_, v___y_509_, v___y_510_, v___y_511_);
if (lean_obj_tag(v___x_512_) == 0)
{
lean_object* v_a_513_; lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; 
v_a_513_ = lean_ctor_get(v___x_512_, 0);
lean_inc_n(v_a_513_, 2);
lean_dec_ref_known(v___x_512_, 1);
v___x_514_ = ((lean_object*)(lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__2));
v___x_515_ = lean_box(0);
v___x_516_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_516_, 0, v_a_513_);
lean_ctor_set(v___x_516_, 1, v___x_515_);
lean_inc(v_a_504_);
v___x_517_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_517_, 0, v_a_504_);
lean_ctor_set(v___x_517_, 1, v___x_516_);
v___x_518_ = l_Lean_mkConst(v___x_514_, v___x_517_);
lean_inc_ref(v_e_491_);
v___x_519_ = l_Lean_Expr_app___override(v___x_518_, v_e_491_);
v___x_520_ = lean_box(0);
v___x_521_ = l_Lean_Meta_synthInstance(v___x_519_, v___x_520_, v___y_508_, v___y_509_, v___y_510_, v___y_511_);
if (lean_obj_tag(v___x_521_) == 0)
{
lean_object* v_a_522_; lean_object* v___x_523_; lean_object* v_a_524_; lean_object* v___x_526_; uint8_t v_isShared_527_; uint8_t v_isSharedCheck_548_; 
v_a_522_ = lean_ctor_get(v___x_521_, 0);
lean_inc(v_a_522_);
lean_dec_ref_known(v___x_521_, 1);
v___x_523_ = lp_plausible_Lean_instantiateLevelMVars___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__0___redArg(v_a_513_, v___y_509_);
v_a_524_ = lean_ctor_get(v___x_523_, 0);
v_isSharedCheck_548_ = !lean_is_exclusive(v___x_523_);
if (v_isSharedCheck_548_ == 0)
{
v___x_526_ = v___x_523_;
v_isShared_527_ = v_isSharedCheck_548_;
goto v_resetjp_525_;
}
else
{
lean_inc(v_a_524_);
lean_dec(v___x_523_);
v___x_526_ = lean_box(0);
v_isShared_527_ = v_isSharedCheck_548_;
goto v_resetjp_525_;
}
v_resetjp_525_:
{
lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_546_; 
v___x_528_ = ((lean_object*)(lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__4));
lean_inc(v_a_524_);
v___x_529_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_529_, 0, v_a_524_);
lean_ctor_set(v___x_529_, 1, v___x_515_);
lean_inc_ref(v___x_529_);
v___x_530_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_530_, 0, v_a_504_);
lean_ctor_set(v___x_530_, 1, v___x_529_);
lean_inc_ref_n(v___x_530_, 2);
v___x_531_ = l_Lean_mkConst(v___x_528_, v___x_530_);
lean_inc_n(v_a_522_, 2);
lean_inc_ref_n(v_e_491_, 3);
v___x_532_ = l_Lean_mkAppB(v___x_531_, v_e_491_, v_a_522_);
v___x_533_ = ((lean_object*)(lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__6));
v___x_534_ = l_Lean_mkConst(v___x_533_, v___x_530_);
v___x_535_ = l_Lean_mkAppB(v___x_534_, v_e_491_, v_a_522_);
v___x_536_ = ((lean_object*)(lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__9));
v___x_537_ = l_Lean_mkConst(v___x_536_, v___x_529_);
v___x_538_ = l_Lean_mkAppB(v___x_537_, v_e_491_, v___x_535_);
v___x_539_ = ((lean_object*)(lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__11));
v___x_540_ = l_Lean_mkConst(v___x_539_, v___x_530_);
v___x_541_ = l_Lean_mkAppB(v___x_540_, v_e_491_, v_a_522_);
v___x_542_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_542_, 0, v___x_532_);
lean_ctor_set(v___x_542_, 1, v___x_538_);
v___x_543_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_543_, 0, v___x_541_);
lean_ctor_set(v___x_543_, 1, v___x_542_);
v___x_544_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_544_, 0, v_a_524_);
lean_ctor_set(v___x_544_, 1, v___x_543_);
if (v_isShared_527_ == 0)
{
lean_ctor_set(v___x_526_, 0, v___x_544_);
v___x_546_ = v___x_526_;
goto v_reusejp_545_;
}
else
{
lean_object* v_reuseFailAlloc_547_; 
v_reuseFailAlloc_547_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_547_, 0, v___x_544_);
v___x_546_ = v_reuseFailAlloc_547_;
goto v_reusejp_545_;
}
v_reusejp_545_:
{
return v___x_546_;
}
}
}
else
{
lean_object* v_a_549_; lean_object* v___x_551_; uint8_t v_isShared_552_; uint8_t v_isSharedCheck_556_; 
lean_dec(v_a_513_);
lean_dec(v_a_504_);
lean_dec_ref(v_e_491_);
v_a_549_ = lean_ctor_get(v___x_521_, 0);
v_isSharedCheck_556_ = !lean_is_exclusive(v___x_521_);
if (v_isSharedCheck_556_ == 0)
{
v___x_551_ = v___x_521_;
v_isShared_552_ = v_isSharedCheck_556_;
goto v_resetjp_550_;
}
else
{
lean_inc(v_a_549_);
lean_dec(v___x_521_);
v___x_551_ = lean_box(0);
v_isShared_552_ = v_isSharedCheck_556_;
goto v_resetjp_550_;
}
v_resetjp_550_:
{
lean_object* v___x_554_; 
if (v_isShared_552_ == 0)
{
v___x_554_ = v___x_551_;
goto v_reusejp_553_;
}
else
{
lean_object* v_reuseFailAlloc_555_; 
v_reuseFailAlloc_555_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_555_, 0, v_a_549_);
v___x_554_ = v_reuseFailAlloc_555_;
goto v_reusejp_553_;
}
v_reusejp_553_:
{
return v___x_554_;
}
}
}
}
else
{
lean_object* v_a_557_; lean_object* v___x_559_; uint8_t v_isShared_560_; uint8_t v_isSharedCheck_564_; 
lean_dec(v_a_504_);
lean_dec_ref(v_e_491_);
v_a_557_ = lean_ctor_get(v___x_512_, 0);
v_isSharedCheck_564_ = !lean_is_exclusive(v___x_512_);
if (v_isSharedCheck_564_ == 0)
{
v___x_559_ = v___x_512_;
v_isShared_560_ = v_isSharedCheck_564_;
goto v_resetjp_558_;
}
else
{
lean_inc(v_a_557_);
lean_dec(v___x_512_);
v___x_559_ = lean_box(0);
v_isShared_560_ = v_isSharedCheck_564_;
goto v_resetjp_558_;
}
v_resetjp_558_:
{
lean_object* v___x_562_; 
if (v_isShared_560_ == 0)
{
v___x_562_ = v___x_559_;
goto v_reusejp_561_;
}
else
{
lean_object* v_reuseFailAlloc_563_; 
v_reuseFailAlloc_563_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_563_, 0, v_a_557_);
v___x_562_ = v_reuseFailAlloc_563_;
goto v_reusejp_561_;
}
v_reusejp_561_:
{
return v___x_562_;
}
}
}
}
}
else
{
lean_object* v_a_597_; lean_object* v___x_599_; uint8_t v_isShared_600_; uint8_t v_isSharedCheck_604_; 
lean_dec(v_a_504_);
lean_dec_ref(v_e_491_);
v_a_597_ = lean_ctor_get(v___x_505_, 0);
v_isSharedCheck_604_ = !lean_is_exclusive(v___x_505_);
if (v_isSharedCheck_604_ == 0)
{
v___x_599_ = v___x_505_;
v_isShared_600_ = v_isSharedCheck_604_;
goto v_resetjp_598_;
}
else
{
lean_inc(v_a_597_);
lean_dec(v___x_505_);
v___x_599_ = lean_box(0);
v_isShared_600_ = v_isSharedCheck_604_;
goto v_resetjp_598_;
}
v_resetjp_598_:
{
lean_object* v___x_602_; 
if (v_isShared_600_ == 0)
{
v___x_602_ = v___x_599_;
goto v_reusejp_601_;
}
else
{
lean_object* v_reuseFailAlloc_603_; 
v_reuseFailAlloc_603_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_603_, 0, v_a_597_);
v___x_602_ = v_reuseFailAlloc_603_;
goto v_reusejp_601_;
}
v_reusejp_601_:
{
return v___x_602_;
}
}
}
}
else
{
lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; 
lean_dec(v_u_503_);
lean_dec_ref(v_e_491_);
v___x_605_ = l_Lean_MessageData_ofExpr(v_a_498_);
v___x_606_ = lean_obj_once(&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__17, &lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__17_once, _init_lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__17);
v___x_607_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_607_, 0, v___x_605_);
lean_ctor_set(v___x_607_, 1, v___x_606_);
v___x_608_ = lp_plausible_Lean_throwError___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__1___redArg(v___x_607_, v_a_492_, v_a_493_, v_a_494_, v_a_495_);
return v___x_608_;
}
}
else
{
lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v___x_612_; 
lean_dec(v_a_502_);
lean_dec_ref(v_e_491_);
v___x_609_ = l_Lean_MessageData_ofExpr(v_a_498_);
v___x_610_ = lean_obj_once(&lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__19, &lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__19_once, _init_lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__19);
v___x_611_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_611_, 0, v___x_609_);
lean_ctor_set(v___x_611_, 1, v___x_610_);
v___x_612_ = lp_plausible_Lean_throwError___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__1___redArg(v___x_611_, v_a_492_, v_a_493_, v_a_494_, v_a_495_);
return v___x_612_;
}
}
else
{
lean_object* v_a_613_; lean_object* v___x_615_; uint8_t v_isShared_616_; uint8_t v_isSharedCheck_620_; 
lean_dec(v_a_498_);
lean_dec_ref(v_e_491_);
v_a_613_ = lean_ctor_get(v___x_501_, 0);
v_isSharedCheck_620_ = !lean_is_exclusive(v___x_501_);
if (v_isSharedCheck_620_ == 0)
{
v___x_615_ = v___x_501_;
v_isShared_616_ = v_isSharedCheck_620_;
goto v_resetjp_614_;
}
else
{
lean_inc(v_a_613_);
lean_dec(v___x_501_);
v___x_615_ = lean_box(0);
v_isShared_616_ = v_isSharedCheck_620_;
goto v_resetjp_614_;
}
v_resetjp_614_:
{
lean_object* v___x_618_; 
if (v_isShared_616_ == 0)
{
v___x_618_ = v___x_615_;
goto v_reusejp_617_;
}
else
{
lean_object* v_reuseFailAlloc_619_; 
v_reuseFailAlloc_619_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_619_, 0, v_a_613_);
v___x_618_ = v_reuseFailAlloc_619_;
goto v_reusejp_617_;
}
v_reusejp_617_:
{
return v___x_618_;
}
}
}
}
else
{
lean_object* v_a_621_; lean_object* v___x_623_; uint8_t v_isShared_624_; uint8_t v_isSharedCheck_628_; 
lean_dec(v_a_498_);
lean_dec_ref(v_e_491_);
v_a_621_ = lean_ctor_get(v___x_499_, 0);
v_isSharedCheck_628_ = !lean_is_exclusive(v___x_499_);
if (v_isSharedCheck_628_ == 0)
{
v___x_623_ = v___x_499_;
v_isShared_624_ = v_isSharedCheck_628_;
goto v_resetjp_622_;
}
else
{
lean_inc(v_a_621_);
lean_dec(v___x_499_);
v___x_623_ = lean_box(0);
v_isShared_624_ = v_isSharedCheck_628_;
goto v_resetjp_622_;
}
v_resetjp_622_:
{
lean_object* v___x_626_; 
if (v_isShared_624_ == 0)
{
v___x_626_ = v___x_623_;
goto v_reusejp_625_;
}
else
{
lean_object* v_reuseFailAlloc_627_; 
v_reuseFailAlloc_627_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_627_, 0, v_a_621_);
v___x_626_ = v_reuseFailAlloc_627_;
goto v_reusejp_625_;
}
v_reusejp_625_:
{
return v___x_626_;
}
}
}
}
else
{
lean_object* v_a_629_; lean_object* v___x_631_; uint8_t v_isShared_632_; uint8_t v_isSharedCheck_636_; 
lean_dec_ref(v_e_491_);
v_a_629_ = lean_ctor_get(v___x_497_, 0);
v_isSharedCheck_636_ = !lean_is_exclusive(v___x_497_);
if (v_isSharedCheck_636_ == 0)
{
v___x_631_ = v___x_497_;
v_isShared_632_ = v_isSharedCheck_636_;
goto v_resetjp_630_;
}
else
{
lean_inc(v_a_629_);
lean_dec(v___x_497_);
v___x_631_ = lean_box(0);
v_isShared_632_ = v_isSharedCheck_636_;
goto v_resetjp_630_;
}
v_resetjp_630_:
{
lean_object* v___x_634_; 
if (v_isShared_632_ == 0)
{
v___x_634_ = v___x_631_;
goto v_reusejp_633_;
}
else
{
lean_object* v_reuseFailAlloc_635_; 
v_reuseFailAlloc_635_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_635_, 0, v_a_629_);
v___x_634_ = v_reuseFailAlloc_635_;
goto v_reusejp_633_;
}
v_reusejp_633_:
{
return v___x_634_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___boxed(lean_object* v_e_637_, lean_object* v_a_638_, lean_object* v_a_639_, lean_object* v_a_640_, lean_object* v_a_641_, lean_object* v_a_642_){
_start:
{
lean_object* v_res_643_; 
v_res_643_ = lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator(v_e_637_, v_a_638_, v_a_639_, v_a_640_, v_a_641_);
lean_dec(v_a_641_);
lean_dec_ref(v_a_640_);
lean_dec(v_a_639_);
lean_dec_ref(v_a_638_);
return v_res_643_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__1(lean_object* v_00_u03b1_644_, lean_object* v_msg_645_, lean_object* v___y_646_, lean_object* v___y_647_, lean_object* v___y_648_, lean_object* v___y_649_){
_start:
{
lean_object* v___x_651_; 
v___x_651_ = lp_plausible_Lean_throwError___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__1___redArg(v_msg_645_, v___y_646_, v___y_647_, v___y_648_, v___y_649_);
return v___x_651_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__1___boxed(lean_object* v_00_u03b1_652_, lean_object* v_msg_653_, lean_object* v___y_654_, lean_object* v___y_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_){
_start:
{
lean_object* v_res_659_; 
v_res_659_ = lp_plausible_Lean_throwError___at___00__private_Plausible_Sampleable_0__Plausible_mkGenerator_spec__1(v_00_u03b1_652_, v_msg_653_, v___y_654_, v___y_655_, v___y_656_, v___y_657_);
lean_dec(v___y_657_);
lean_dec_ref(v___y_656_);
lean_dec(v___y_655_);
lean_dec_ref(v___y_654_);
return v_res_659_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__2(void){
_start:
{
lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v___x_690_; 
v___x_688_ = lean_box(0);
v___x_689_ = ((lean_object*)(lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__1));
v___x_690_ = l_Lean_mkConst(v___x_689_, v___x_688_);
return v___x_690_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__5(void){
_start:
{
lean_object* v___x_694_; lean_object* v___x_695_; 
v___x_694_ = lean_unsigned_to_nat(1u);
v___x_695_ = l_Lean_Level_ofNat(v___x_694_);
return v___x_695_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__6(void){
_start:
{
lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; 
v___x_696_ = lean_box(0);
v___x_697_ = lean_obj_once(&lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__5, &lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__5_once, _init_lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__5);
v___x_698_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_698_, 0, v___x_697_);
lean_ctor_set(v___x_698_, 1, v___x_696_);
return v___x_698_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__7(void){
_start:
{
lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v___x_701_; 
v___x_699_ = lean_obj_once(&lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__6, &lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__6_once, _init_lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__6);
v___x_700_ = ((lean_object*)(lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__4));
v___x_701_ = l_Lean_mkConst(v___x_700_, v___x_699_);
return v___x_701_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__8(void){
_start:
{
lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; 
v___x_702_ = lean_obj_once(&lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__7, &lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__7_once, _init_lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__7);
v___x_703_ = lean_obj_once(&lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__2, &lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__2_once, _init_lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__2);
v___x_704_ = l_Lean_Expr_app___override(v___x_703_, v___x_702_);
return v___x_704_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1(lean_object* v_printSamples_705_, lean_object* v_a_706_, lean_object* v_a_707_, lean_object* v_a_708_, lean_object* v_a_709_){
_start:
{
lean_object* v___x_711_; uint8_t v___x_712_; uint8_t v___x_713_; lean_object* v___x_714_; 
v___x_711_ = lean_obj_once(&lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__8, &lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__8_once, _init_lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__8);
v___x_712_ = 1;
v___x_713_ = 1;
v___x_714_ = l_Lean_Meta_evalExpr___redArg(v___x_711_, v_printSamples_705_, v___x_712_, v___x_713_, v_a_706_, v_a_707_, v_a_708_, v_a_709_);
return v___x_714_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___boxed(lean_object* v_printSamples_715_, lean_object* v_a_716_, lean_object* v_a_717_, lean_object* v_a_718_, lean_object* v_a_719_, lean_object* v_a_720_){
_start:
{
lean_object* v_res_721_; 
v_res_721_ = lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1(v_printSamples_715_, v_a_716_, v_a_717_, v_a_718_, v_a_719_);
lean_dec(v_a_719_);
lean_dec_ref(v_a_718_);
lean_dec(v_a_717_);
lean_dec_ref(v_a_716_);
return v_res_721_;
}
}
static lean_object* _init_lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_722_; lean_object* v___x_723_; lean_object* v___x_724_; 
v___x_722_ = lean_box(0);
v___x_723_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_724_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_724_, 0, v___x_723_);
lean_ctor_set(v___x_724_, 1, v___x_722_);
return v___x_724_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_spec__0___redArg(){
_start:
{
lean_object* v___x_726_; lean_object* v___x_727_; 
v___x_726_ = lean_obj_once(&lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_spec__0___redArg___closed__0, &lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_spec__0___redArg___closed__0_once, _init_lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_spec__0___redArg___closed__0);
v___x_727_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_727_, 0, v___x_726_);
return v___x_727_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_spec__0___redArg___boxed(lean_object* v___y_728_){
_start:
{
lean_object* v_res_729_; 
v_res_729_ = lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_spec__0___redArg();
return v_res_729_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_spec__0(lean_object* v_00_u03b1_730_, lean_object* v___y_731_, lean_object* v___y_732_){
_start:
{
lean_object* v___x_734_; 
v___x_734_ = lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_spec__0___redArg();
return v___x_734_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_spec__0___boxed(lean_object* v_00_u03b1_735_, lean_object* v___y_736_, lean_object* v___y_737_, lean_object* v___y_738_){
_start:
{
lean_object* v_res_739_; 
v_res_739_ = lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_spec__0(v_00_u03b1_735_, v___y_736_, v___y_737_);
lean_dec(v___y_737_);
lean_dec_ref(v___y_736_);
return v_res_739_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1___lam__0(lean_object* v___x_741_, lean_object* v___x_742_, lean_object* v___x_743_, uint8_t v___x_744_, lean_object* v_x_745_, lean_object* v___y_746_, lean_object* v___y_747_, lean_object* v___y_748_, lean_object* v___y_749_, lean_object* v___y_750_, lean_object* v___y_751_){
_start:
{
lean_object* v___x_753_; lean_object* v___x_754_; 
v___x_753_ = lean_box(0);
v___x_754_ = l_Lean_Elab_Term_elabTermAndSynthesize(v___x_741_, v___x_753_, v___y_746_, v___y_747_, v___y_748_, v___y_749_, v___y_750_, v___y_751_);
if (lean_obj_tag(v___x_754_) == 0)
{
lean_object* v_a_755_; lean_object* v___x_756_; 
v_a_755_ = lean_ctor_get(v___x_754_, 0);
lean_inc(v_a_755_);
lean_dec_ref_known(v___x_754_, 1);
v___x_756_ = lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator(v_a_755_, v___y_748_, v___y_749_, v___y_750_, v___y_751_);
if (lean_obj_tag(v___x_756_) == 0)
{
lean_object* v_a_757_; lean_object* v_snd_758_; lean_object* v_snd_759_; lean_object* v_fst_760_; lean_object* v___x_762_; uint8_t v_isShared_763_; uint8_t v_isSharedCheck_820_; 
v_a_757_ = lean_ctor_get(v___x_756_, 0);
lean_inc(v_a_757_);
lean_dec_ref_known(v___x_756_, 1);
v_snd_758_ = lean_ctor_get(v_a_757_, 1);
lean_inc(v_snd_758_);
lean_dec(v_a_757_);
v_snd_759_ = lean_ctor_get(v_snd_758_, 1);
v_fst_760_ = lean_ctor_get(v_snd_758_, 0);
v_isSharedCheck_820_ = !lean_is_exclusive(v_snd_758_);
if (v_isSharedCheck_820_ == 0)
{
v___x_762_ = v_snd_758_;
v_isShared_763_ = v_isSharedCheck_820_;
goto v_resetjp_761_;
}
else
{
lean_inc(v_snd_759_);
lean_inc(v_fst_760_);
lean_dec(v_snd_758_);
v___x_762_ = lean_box(0);
v_isShared_763_ = v_isSharedCheck_820_;
goto v_resetjp_761_;
}
v_resetjp_761_:
{
lean_object* v_fst_764_; lean_object* v_snd_765_; lean_object* v___x_767_; uint8_t v_isShared_768_; uint8_t v_isSharedCheck_819_; 
v_fst_764_ = lean_ctor_get(v_snd_759_, 0);
v_snd_765_ = lean_ctor_get(v_snd_759_, 1);
v_isSharedCheck_819_ = !lean_is_exclusive(v_snd_759_);
if (v_isSharedCheck_819_ == 0)
{
v___x_767_ = v_snd_759_;
v_isShared_768_ = v_isSharedCheck_819_;
goto v_resetjp_766_;
}
else
{
lean_inc(v_snd_765_);
lean_inc(v_fst_764_);
lean_dec(v_snd_759_);
v___x_767_ = lean_box(0);
v_isShared_768_ = v_isSharedCheck_819_;
goto v_resetjp_766_;
}
v_resetjp_766_:
{
lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_771_; lean_object* v___x_772_; lean_object* v___x_773_; lean_object* v___x_774_; lean_object* v___x_775_; lean_object* v___x_776_; lean_object* v___x_777_; lean_object* v___x_779_; 
v___x_769_ = ((lean_object*)(lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__12));
v___x_770_ = ((lean_object*)(lp_plausible_Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1___lam__0___closed__0));
v___x_771_ = l_Lean_Name_mkStr3(v___x_742_, v___x_769_, v___x_770_);
v___x_772_ = lean_box(0);
v___x_773_ = l_Lean_mkConst(v___x_771_, v___x_772_);
v___x_774_ = l_Lean_mkApp3(v___x_773_, v_fst_760_, v_fst_764_, v_snd_765_);
v___x_775_ = lean_obj_once(&lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__2, &lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__2_once, _init_lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__2);
v___x_776_ = ((lean_object*)(lp_plausible___private_Plausible_Sampleable_0__Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_unsafe__1___closed__4));
v___x_777_ = l_Lean_Level_ofNat(v___x_743_);
if (v_isShared_768_ == 0)
{
lean_ctor_set_tag(v___x_767_, 1);
lean_ctor_set(v___x_767_, 1, v___x_772_);
lean_ctor_set(v___x_767_, 0, v___x_777_);
v___x_779_ = v___x_767_;
goto v_reusejp_778_;
}
else
{
lean_object* v_reuseFailAlloc_818_; 
v_reuseFailAlloc_818_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_818_, 0, v___x_777_);
lean_ctor_set(v_reuseFailAlloc_818_, 1, v___x_772_);
v___x_779_ = v_reuseFailAlloc_818_;
goto v_reusejp_778_;
}
v_reusejp_778_:
{
lean_object* v___x_780_; lean_object* v___x_781_; uint8_t v___x_782_; lean_object* v___x_783_; 
v___x_780_ = l_Lean_mkConst(v___x_776_, v___x_779_);
v___x_781_ = l_Lean_Expr_app___override(v___x_775_, v___x_780_);
v___x_782_ = 1;
v___x_783_ = l_Lean_Meta_evalExpr___redArg(v___x_781_, v___x_774_, v___x_782_, v___x_744_, v___y_748_, v___y_749_, v___y_750_, v___y_751_);
if (lean_obj_tag(v___x_783_) == 0)
{
lean_object* v_a_784_; lean_object* v___x_785_; 
v_a_784_ = lean_ctor_get(v___x_783_, 0);
lean_inc(v_a_784_);
lean_dec_ref_known(v___x_783_, 1);
v___x_785_ = lean_apply_1(v_a_784_, lean_box(0));
if (lean_obj_tag(v___x_785_) == 0)
{
lean_object* v___x_787_; uint8_t v_isShared_788_; uint8_t v_isSharedCheck_793_; 
lean_del_object(v___x_762_);
v_isSharedCheck_793_ = !lean_is_exclusive(v___x_785_);
if (v_isSharedCheck_793_ == 0)
{
lean_object* v_unused_794_; 
v_unused_794_ = lean_ctor_get(v___x_785_, 0);
lean_dec(v_unused_794_);
v___x_787_ = v___x_785_;
v_isShared_788_ = v_isSharedCheck_793_;
goto v_resetjp_786_;
}
else
{
lean_dec(v___x_785_);
v___x_787_ = lean_box(0);
v_isShared_788_ = v_isSharedCheck_793_;
goto v_resetjp_786_;
}
v_resetjp_786_:
{
lean_object* v___x_789_; lean_object* v___x_791_; 
v___x_789_ = lean_box(0);
if (v_isShared_788_ == 0)
{
lean_ctor_set(v___x_787_, 0, v___x_789_);
v___x_791_ = v___x_787_;
goto v_reusejp_790_;
}
else
{
lean_object* v_reuseFailAlloc_792_; 
v_reuseFailAlloc_792_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_792_, 0, v___x_789_);
v___x_791_ = v_reuseFailAlloc_792_;
goto v_reusejp_790_;
}
v_reusejp_790_:
{
return v___x_791_;
}
}
}
else
{
lean_object* v_a_795_; lean_object* v___x_797_; uint8_t v_isShared_798_; uint8_t v_isSharedCheck_809_; 
v_a_795_ = lean_ctor_get(v___x_785_, 0);
v_isSharedCheck_809_ = !lean_is_exclusive(v___x_785_);
if (v_isSharedCheck_809_ == 0)
{
v___x_797_ = v___x_785_;
v_isShared_798_ = v_isSharedCheck_809_;
goto v_resetjp_796_;
}
else
{
lean_inc(v_a_795_);
lean_dec(v___x_785_);
v___x_797_ = lean_box(0);
v_isShared_798_ = v_isSharedCheck_809_;
goto v_resetjp_796_;
}
v_resetjp_796_:
{
lean_object* v_ref_799_; lean_object* v___x_800_; lean_object* v___x_801_; lean_object* v___x_802_; lean_object* v___x_804_; 
v_ref_799_ = lean_ctor_get(v___y_750_, 5);
v___x_800_ = lean_io_error_to_string(v_a_795_);
v___x_801_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_801_, 0, v___x_800_);
v___x_802_ = l_Lean_MessageData_ofFormat(v___x_801_);
lean_inc(v_ref_799_);
if (v_isShared_763_ == 0)
{
lean_ctor_set(v___x_762_, 1, v___x_802_);
lean_ctor_set(v___x_762_, 0, v_ref_799_);
v___x_804_ = v___x_762_;
goto v_reusejp_803_;
}
else
{
lean_object* v_reuseFailAlloc_808_; 
v_reuseFailAlloc_808_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_808_, 0, v_ref_799_);
lean_ctor_set(v_reuseFailAlloc_808_, 1, v___x_802_);
v___x_804_ = v_reuseFailAlloc_808_;
goto v_reusejp_803_;
}
v_reusejp_803_:
{
lean_object* v___x_806_; 
if (v_isShared_798_ == 0)
{
lean_ctor_set(v___x_797_, 0, v___x_804_);
v___x_806_ = v___x_797_;
goto v_reusejp_805_;
}
else
{
lean_object* v_reuseFailAlloc_807_; 
v_reuseFailAlloc_807_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_807_, 0, v___x_804_);
v___x_806_ = v_reuseFailAlloc_807_;
goto v_reusejp_805_;
}
v_reusejp_805_:
{
return v___x_806_;
}
}
}
}
}
else
{
lean_object* v_a_810_; lean_object* v___x_812_; uint8_t v_isShared_813_; uint8_t v_isSharedCheck_817_; 
lean_del_object(v___x_762_);
v_a_810_ = lean_ctor_get(v___x_783_, 0);
v_isSharedCheck_817_ = !lean_is_exclusive(v___x_783_);
if (v_isSharedCheck_817_ == 0)
{
v___x_812_ = v___x_783_;
v_isShared_813_ = v_isSharedCheck_817_;
goto v_resetjp_811_;
}
else
{
lean_inc(v_a_810_);
lean_dec(v___x_783_);
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
}
}
}
else
{
lean_object* v_a_821_; lean_object* v___x_823_; uint8_t v_isShared_824_; uint8_t v_isSharedCheck_828_; 
lean_dec_ref(v___x_742_);
v_a_821_ = lean_ctor_get(v___x_756_, 0);
v_isSharedCheck_828_ = !lean_is_exclusive(v___x_756_);
if (v_isSharedCheck_828_ == 0)
{
v___x_823_ = v___x_756_;
v_isShared_824_ = v_isSharedCheck_828_;
goto v_resetjp_822_;
}
else
{
lean_inc(v_a_821_);
lean_dec(v___x_756_);
v___x_823_ = lean_box(0);
v_isShared_824_ = v_isSharedCheck_828_;
goto v_resetjp_822_;
}
v_resetjp_822_:
{
lean_object* v___x_826_; 
if (v_isShared_824_ == 0)
{
v___x_826_ = v___x_823_;
goto v_reusejp_825_;
}
else
{
lean_object* v_reuseFailAlloc_827_; 
v_reuseFailAlloc_827_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_827_, 0, v_a_821_);
v___x_826_ = v_reuseFailAlloc_827_;
goto v_reusejp_825_;
}
v_reusejp_825_:
{
return v___x_826_;
}
}
}
}
else
{
lean_object* v_a_829_; lean_object* v___x_831_; uint8_t v_isShared_832_; uint8_t v_isSharedCheck_836_; 
lean_dec_ref(v___x_742_);
v_a_829_ = lean_ctor_get(v___x_754_, 0);
v_isSharedCheck_836_ = !lean_is_exclusive(v___x_754_);
if (v_isSharedCheck_836_ == 0)
{
v___x_831_ = v___x_754_;
v_isShared_832_ = v_isSharedCheck_836_;
goto v_resetjp_830_;
}
else
{
lean_inc(v_a_829_);
lean_dec(v___x_754_);
v___x_831_ = lean_box(0);
v_isShared_832_ = v_isSharedCheck_836_;
goto v_resetjp_830_;
}
v_resetjp_830_:
{
lean_object* v___x_834_; 
if (v_isShared_832_ == 0)
{
v___x_834_ = v___x_831_;
goto v_reusejp_833_;
}
else
{
lean_object* v_reuseFailAlloc_835_; 
v_reuseFailAlloc_835_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_835_, 0, v_a_829_);
v___x_834_ = v_reuseFailAlloc_835_;
goto v_reusejp_833_;
}
v_reusejp_833_:
{
return v___x_834_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1___lam__0___boxed(lean_object* v___x_837_, lean_object* v___x_838_, lean_object* v___x_839_, lean_object* v___x_840_, lean_object* v_x_841_, lean_object* v___y_842_, lean_object* v___y_843_, lean_object* v___y_844_, lean_object* v___y_845_, lean_object* v___y_846_, lean_object* v___y_847_, lean_object* v___y_848_){
_start:
{
uint8_t v___x_1435__boxed_849_; lean_object* v_res_850_; 
v___x_1435__boxed_849_ = lean_unbox(v___x_840_);
v_res_850_ = lp_plausible_Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1___lam__0(v___x_837_, v___x_838_, v___x_839_, v___x_1435__boxed_849_, v_x_841_, v___y_842_, v___y_843_, v___y_844_, v___y_845_, v___y_846_, v___y_847_);
lean_dec(v___y_847_);
lean_dec_ref(v___y_846_);
lean_dec(v___y_845_);
lean_dec_ref(v___y_844_);
lean_dec(v___y_843_);
lean_dec_ref(v___y_842_);
lean_dec_ref(v_x_841_);
lean_dec(v___x_839_);
return v_res_850_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1(lean_object* v_x_851_, lean_object* v_a_852_, lean_object* v_a_853_){
_start:
{
lean_object* v___x_855_; lean_object* v___x_856_; uint8_t v___x_857_; 
v___x_855_ = ((lean_object*)(lp_plausible___private_Plausible_Sampleable_0__Plausible_mkGenerator___closed__0));
v___x_856_ = ((lean_object*)(lp_plausible_Plausible_command_x23sample___00__closed__1));
lean_inc(v_x_851_);
v___x_857_ = l_Lean_Syntax_isOfKind(v_x_851_, v___x_856_);
if (v___x_857_ == 0)
{
lean_object* v___x_858_; 
lean_dec(v_x_851_);
v___x_858_ = lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1_spec__0___redArg();
return v___x_858_;
}
else
{
lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_861_; lean_object* v___f_862_; lean_object* v___x_863_; 
v___x_859_ = lean_unsigned_to_nat(1u);
v___x_860_ = l_Lean_Syntax_getArg(v_x_851_, v___x_859_);
lean_dec(v_x_851_);
v___x_861_ = lean_box(v___x_857_);
v___f_862_ = lean_alloc_closure((void*)(lp_plausible_Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1___lam__0___boxed), 12, 4);
lean_closure_set(v___f_862_, 0, v___x_860_);
lean_closure_set(v___f_862_, 1, v___x_855_);
lean_closure_set(v___f_862_, 2, v___x_859_);
lean_closure_set(v___f_862_, 3, v___x_861_);
v___x_863_ = l_Lean_Elab_Command_runTermElabM___redArg(v___f_862_, v_a_852_, v_a_853_);
return v___x_863_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1___boxed(lean_object* v_x_864_, lean_object* v_a_865_, lean_object* v_a_866_, lean_object* v_a_867_){
_start:
{
lean_object* v_res_868_; 
v_res_868_ = lp_plausible_Plausible___aux__Plausible__Sampleable______elabRules__Plausible__command_x23sample____1(v_x_864_, v_a_865_, v_a_866_);
lean_dec(v_a_866_);
lean_dec_ref(v_a_865_);
return v_res_868_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_plausible_Plausible_Arbitrary(uint8_t builtin);
lean_object* runtime_initialize_plausible_Plausible_Shrinkable(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_plausible_Plausible_Sampleable(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_plausible_Plausible_Arbitrary(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_plausible_Plausible_Shrinkable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_plausible_Plausible_Prop_sampleableExt = _init_lp_plausible_Plausible_Prop_sampleableExt();
lean_mark_persistent(lp_plausible_Plausible_Prop_sampleableExt);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Eval(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_plausible_Plausible_Sampleable(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Eval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* initialize_Lean_Meta_Eval(uint8_t builtin);
lean_object* initialize_plausible_Plausible_Arbitrary(uint8_t builtin);
lean_object* initialize_plausible_Plausible_Shrinkable(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_plausible_Plausible_Sampleable(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Eval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_plausible_Plausible_Arbitrary(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_plausible_Plausible_Shrinkable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_plausible_Plausible_Sampleable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_plausible_Plausible_Sampleable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_plausible_Plausible_Sampleable(builtin);
}
#ifdef __cplusplus
}
#endif
