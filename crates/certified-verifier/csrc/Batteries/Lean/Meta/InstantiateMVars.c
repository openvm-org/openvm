// Lean compiler output
// Module: Batteries.Lean.Meta.InstantiateMVars
// Imports: public import Init public meta import Init public import Batteries.Lean.Meta.Basic
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
lean_object* l_Lean_instantiateMVarDeclMVars___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Lean_MetavarContext_declareExprMVar(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instBEqFVarId_beq___boxed(lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVars___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_set___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_local_ctx_find(lean_object*, lean_object*);
lean_object* l_Lean_instHashableFVarId_hash___boxed(lean_object*);
lean_object* l_Lean_PersistentHashMap_insert___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_instantiateLocalDeclMVars___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_throwError___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instantiateLCtxMVars___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instBEqFVarId_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__2___closed__0 = (const lean_object*)&lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__2___closed__0_value;
static const lean_closure_object lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instHashableFVarId_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__2___closed__1 = (const lean_object*)&lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__2___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "unknown fvar '"};
static const lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__0 = (const lean_object*)&lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__0_value;
static lean_once_cell_t lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__1;
static const lean_string_object lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "' (in local context of mvar '\?"};
static const lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__2 = (const lean_object*)&lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__2_value;
static lean_once_cell_t lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__3;
static const lean_string_object lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "')"};
static const lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__4 = (const lean_object*)&lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__4_value;
static lean_once_cell_t lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__5;
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalContext___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalContext___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalContext___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalContext___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalContext___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalContext___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalContext(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVars___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVars___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVars___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVars___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg___lam__0(lean_object* v_toApplicative_1_, lean_object* v_type_2_, lean_object* v_____r_3_){
_start:
{
lean_object* v_toPure_4_; lean_object* v___x_5_; 
v_toPure_4_ = lean_ctor_get(v_toApplicative_1_, 1);
lean_inc(v_toPure_4_);
lean_dec_ref(v_toApplicative_1_);
v___x_5_ = lean_apply_2(v_toPure_4_, lean_box(0), v_type_2_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg___lam__1(lean_object* v_mvarId_6_, lean_object* v_mdecl_7_, lean_object* v_x_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lp_batteries_Lean_MetavarContext_declareExprMVar(v_x_8_, v_mvarId_6_, v_mdecl_7_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg___lam__2(lean_object* v_toApplicative_10_, lean_object* v_userName_11_, lean_object* v_lctx_12_, lean_object* v_depth_13_, lean_object* v_localInstances_14_, uint8_t v_kind_15_, lean_object* v_numScopeArgs_16_, lean_object* v_index_17_, lean_object* v_mvarId_18_, lean_object* v_modifyMCtx_19_, lean_object* v_toBind_20_, lean_object* v_type_21_){
_start:
{
lean_object* v___f_22_; lean_object* v_mdecl_23_; lean_object* v___f_24_; lean_object* v___x_25_; lean_object* v___x_26_; 
lean_inc_ref(v_type_21_);
v___f_22_ = lean_alloc_closure((void*)(lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg___lam__0), 3, 2);
lean_closure_set(v___f_22_, 0, v_toApplicative_10_);
lean_closure_set(v___f_22_, 1, v_type_21_);
v_mdecl_23_ = lean_alloc_ctor(0, 7, 1);
lean_ctor_set(v_mdecl_23_, 0, v_userName_11_);
lean_ctor_set(v_mdecl_23_, 1, v_lctx_12_);
lean_ctor_set(v_mdecl_23_, 2, v_type_21_);
lean_ctor_set(v_mdecl_23_, 3, v_depth_13_);
lean_ctor_set(v_mdecl_23_, 4, v_localInstances_14_);
lean_ctor_set(v_mdecl_23_, 5, v_numScopeArgs_16_);
lean_ctor_set(v_mdecl_23_, 6, v_index_17_);
lean_ctor_set_uint8(v_mdecl_23_, sizeof(void*)*7, v_kind_15_);
v___f_24_ = lean_alloc_closure((void*)(lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg___lam__1), 3, 2);
lean_closure_set(v___f_24_, 0, v_mvarId_18_);
lean_closure_set(v___f_24_, 1, v_mdecl_23_);
v___x_25_ = lean_apply_1(v_modifyMCtx_19_, v___f_24_);
v___x_26_ = lean_apply_4(v_toBind_20_, lean_box(0), lean_box(0), v___x_25_, v___f_22_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg___lam__2___boxed(lean_object* v_toApplicative_27_, lean_object* v_userName_28_, lean_object* v_lctx_29_, lean_object* v_depth_30_, lean_object* v_localInstances_31_, lean_object* v_kind_32_, lean_object* v_numScopeArgs_33_, lean_object* v_index_34_, lean_object* v_mvarId_35_, lean_object* v_modifyMCtx_36_, lean_object* v_toBind_37_, lean_object* v_type_38_){
_start:
{
uint8_t v_kind_boxed_39_; lean_object* v_res_40_; 
v_kind_boxed_39_ = lean_unbox(v_kind_32_);
v_res_40_ = lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg___lam__2(v_toApplicative_27_, v_userName_28_, v_lctx_29_, v_depth_30_, v_localInstances_31_, v_kind_boxed_39_, v_numScopeArgs_33_, v_index_34_, v_mvarId_35_, v_modifyMCtx_36_, v_toBind_37_, v_type_38_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg___lam__3(lean_object* v_toApplicative_41_, lean_object* v_mvarId_42_, lean_object* v_modifyMCtx_43_, lean_object* v_toBind_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_mdecl_47_){
_start:
{
lean_object* v_userName_48_; lean_object* v_lctx_49_; lean_object* v_type_50_; lean_object* v_depth_51_; lean_object* v_localInstances_52_; uint8_t v_kind_53_; lean_object* v_numScopeArgs_54_; lean_object* v_index_55_; uint8_t v___x_56_; 
v_userName_48_ = lean_ctor_get(v_mdecl_47_, 0);
lean_inc(v_userName_48_);
v_lctx_49_ = lean_ctor_get(v_mdecl_47_, 1);
lean_inc_ref(v_lctx_49_);
v_type_50_ = lean_ctor_get(v_mdecl_47_, 2);
lean_inc_ref(v_type_50_);
v_depth_51_ = lean_ctor_get(v_mdecl_47_, 3);
lean_inc(v_depth_51_);
v_localInstances_52_ = lean_ctor_get(v_mdecl_47_, 4);
lean_inc_ref(v_localInstances_52_);
v_kind_53_ = lean_ctor_get_uint8(v_mdecl_47_, sizeof(void*)*7);
v_numScopeArgs_54_ = lean_ctor_get(v_mdecl_47_, 5);
lean_inc(v_numScopeArgs_54_);
v_index_55_ = lean_ctor_get(v_mdecl_47_, 6);
lean_inc(v_index_55_);
lean_dec_ref(v_mdecl_47_);
v___x_56_ = l_Lean_Expr_hasMVar(v_type_50_);
if (v___x_56_ == 0)
{
lean_object* v_toPure_57_; lean_object* v___x_58_; 
lean_dec(v_index_55_);
lean_dec(v_numScopeArgs_54_);
lean_dec_ref(v_localInstances_52_);
lean_dec(v_depth_51_);
lean_dec_ref(v_lctx_49_);
lean_dec(v_userName_48_);
lean_dec_ref(v_inst_46_);
lean_dec_ref(v_inst_45_);
lean_dec(v_toBind_44_);
lean_dec(v_modifyMCtx_43_);
lean_dec(v_mvarId_42_);
v_toPure_57_ = lean_ctor_get(v_toApplicative_41_, 1);
lean_inc(v_toPure_57_);
lean_dec_ref(v_toApplicative_41_);
v___x_58_ = lean_apply_2(v_toPure_57_, lean_box(0), v_type_50_);
return v___x_58_;
}
else
{
lean_object* v___x_59_; lean_object* v___f_60_; lean_object* v___x_61_; lean_object* v___x_62_; 
v___x_59_ = lean_box(v_kind_53_);
lean_inc(v_toBind_44_);
v___f_60_ = lean_alloc_closure((void*)(lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg___lam__2___boxed), 12, 11);
lean_closure_set(v___f_60_, 0, v_toApplicative_41_);
lean_closure_set(v___f_60_, 1, v_userName_48_);
lean_closure_set(v___f_60_, 2, v_lctx_49_);
lean_closure_set(v___f_60_, 3, v_depth_51_);
lean_closure_set(v___f_60_, 4, v_localInstances_52_);
lean_closure_set(v___f_60_, 5, v___x_59_);
lean_closure_set(v___f_60_, 6, v_numScopeArgs_54_);
lean_closure_set(v___f_60_, 7, v_index_55_);
lean_closure_set(v___f_60_, 8, v_mvarId_42_);
lean_closure_set(v___f_60_, 9, v_modifyMCtx_43_);
lean_closure_set(v___f_60_, 10, v_toBind_44_);
v___x_61_ = l_Lean_instantiateMVars___redArg(v_inst_45_, v_inst_46_, v_type_50_);
v___x_62_ = lean_apply_4(v_toBind_44_, lean_box(0), lean_box(0), v___x_61_, v___f_60_);
return v___x_62_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg___lam__4(lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_mvarId_65_, lean_object* v_toBind_66_, lean_object* v___f_67_, lean_object* v_____do__lift_68_){
_start:
{
lean_object* v___x_69_; lean_object* v___x_70_; 
v___x_69_ = lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg(v_inst_63_, v_inst_64_, v_____do__lift_68_, v_mvarId_65_);
v___x_70_ = lean_apply_4(v_toBind_66_, lean_box(0), lean_box(0), v___x_69_, v___f_67_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg___lam__4___boxed(lean_object* v_inst_71_, lean_object* v_inst_72_, lean_object* v_mvarId_73_, lean_object* v_toBind_74_, lean_object* v___f_75_, lean_object* v_____do__lift_76_){
_start:
{
lean_object* v_res_77_; 
v_res_77_ = lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg___lam__4(v_inst_71_, v_inst_72_, v_mvarId_73_, v_toBind_74_, v___f_75_, v_____do__lift_76_);
lean_dec_ref(v_____do__lift_76_);
return v_res_77_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg(lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_inst_80_, lean_object* v_mvarId_81_){
_start:
{
lean_object* v_toApplicative_82_; lean_object* v_toBind_83_; lean_object* v_getMCtx_84_; lean_object* v_modifyMCtx_85_; lean_object* v___f_86_; lean_object* v___f_87_; lean_object* v___x_88_; 
v_toApplicative_82_ = lean_ctor_get(v_inst_78_, 0);
v_toBind_83_ = lean_ctor_get(v_inst_78_, 1);
lean_inc_n(v_toBind_83_, 3);
v_getMCtx_84_ = lean_ctor_get(v_inst_79_, 0);
lean_inc(v_getMCtx_84_);
v_modifyMCtx_85_ = lean_ctor_get(v_inst_79_, 1);
lean_inc(v_modifyMCtx_85_);
lean_inc_ref(v_inst_78_);
lean_inc(v_mvarId_81_);
lean_inc_ref(v_toApplicative_82_);
v___f_86_ = lean_alloc_closure((void*)(lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg___lam__3), 7, 6);
lean_closure_set(v___f_86_, 0, v_toApplicative_82_);
lean_closure_set(v___f_86_, 1, v_mvarId_81_);
lean_closure_set(v___f_86_, 2, v_modifyMCtx_85_);
lean_closure_set(v___f_86_, 3, v_toBind_83_);
lean_closure_set(v___f_86_, 4, v_inst_78_);
lean_closure_set(v___f_86_, 5, v_inst_79_);
v___f_87_ = lean_alloc_closure((void*)(lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg___lam__4___boxed), 6, 5);
lean_closure_set(v___f_87_, 0, v_inst_78_);
lean_closure_set(v___f_87_, 1, v_inst_80_);
lean_closure_set(v___f_87_, 2, v_mvarId_81_);
lean_closure_set(v___f_87_, 3, v_toBind_83_);
lean_closure_set(v___f_87_, 4, v___f_86_);
v___x_88_ = lean_apply_4(v_toBind_83_, lean_box(0), lean_box(0), v_getMCtx_84_, v___f_87_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInType(lean_object* v_m_89_, lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_inst_92_, lean_object* v_mvarId_93_){
_start:
{
lean_object* v___x_94_; 
v___x_94_ = lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg(v_inst_90_, v_inst_91_, v_inst_92_, v_mvarId_93_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__0(lean_object* v_toPure_95_, lean_object* v_ldecl_96_, lean_object* v_____r_97_){
_start:
{
lean_object* v___x_98_; 
v___x_98_ = lean_apply_2(v_toPure_95_, lean_box(0), v_ldecl_96_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__2(lean_object* v_lctx_101_, lean_object* v_toPure_102_, lean_object* v_userName_103_, lean_object* v_type_104_, lean_object* v_depth_105_, lean_object* v_localInstances_106_, uint8_t v_kind_107_, lean_object* v_numScopeArgs_108_, lean_object* v_index_109_, lean_object* v_mvarId_110_, lean_object* v_modifyMCtx_111_, lean_object* v_toBind_112_, lean_object* v_fvarId_113_, lean_object* v_ldecl_114_){
_start:
{
lean_object* v_fvarIdToDecl_115_; lean_object* v_decls_116_; lean_object* v_auxDeclToFullName_117_; lean_object* v___f_118_; lean_object* v___y_120_; lean_object* v___y_126_; lean_object* v___y_127_; lean_object* v___x_131_; 
v_fvarIdToDecl_115_ = lean_ctor_get(v_lctx_101_, 0);
v_decls_116_ = lean_ctor_get(v_lctx_101_, 1);
v_auxDeclToFullName_117_ = lean_ctor_get(v_lctx_101_, 2);
lean_inc_ref(v_ldecl_114_);
v___f_118_ = lean_alloc_closure((void*)(lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__0), 3, 2);
lean_closure_set(v___f_118_, 0, v_toPure_102_);
lean_closure_set(v___f_118_, 1, v_ldecl_114_);
lean_inc_ref(v_lctx_101_);
v___x_131_ = lean_local_ctx_find(v_lctx_101_, v_fvarId_113_);
if (lean_obj_tag(v___x_131_) == 0)
{
lean_dec_ref(v_ldecl_114_);
v___y_120_ = v_lctx_101_;
goto v___jp_119_;
}
else
{
lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___y_135_; lean_object* v_fvarId_138_; 
lean_inc(v_auxDeclToFullName_117_);
lean_inc_ref(v_decls_116_);
lean_inc_ref(v_fvarIdToDecl_115_);
lean_dec_ref_known(v___x_131_, 1);
lean_dec_ref(v_lctx_101_);
v___x_132_ = ((lean_object*)(lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__2___closed__0));
v___x_133_ = ((lean_object*)(lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__2___closed__1));
v_fvarId_138_ = lean_ctor_get(v_ldecl_114_, 1);
lean_inc(v_fvarId_138_);
v___y_135_ = v_fvarId_138_;
goto v___jp_134_;
v___jp_134_:
{
lean_object* v___x_136_; lean_object* v_index_137_; 
lean_inc_ref(v_ldecl_114_);
v___x_136_ = l_Lean_PersistentHashMap_insert___redArg(v___x_132_, v___x_133_, v_fvarIdToDecl_115_, v___y_135_, v_ldecl_114_);
v_index_137_ = lean_ctor_get(v_ldecl_114_, 0);
lean_inc(v_index_137_);
v___y_126_ = v___x_136_;
v___y_127_ = v_index_137_;
goto v___jp_125_;
}
}
v___jp_119_:
{
lean_object* v_mdecl_121_; lean_object* v___f_122_; lean_object* v___x_123_; lean_object* v___x_124_; 
v_mdecl_121_ = lean_alloc_ctor(0, 7, 1);
lean_ctor_set(v_mdecl_121_, 0, v_userName_103_);
lean_ctor_set(v_mdecl_121_, 1, v___y_120_);
lean_ctor_set(v_mdecl_121_, 2, v_type_104_);
lean_ctor_set(v_mdecl_121_, 3, v_depth_105_);
lean_ctor_set(v_mdecl_121_, 4, v_localInstances_106_);
lean_ctor_set(v_mdecl_121_, 5, v_numScopeArgs_108_);
lean_ctor_set(v_mdecl_121_, 6, v_index_109_);
lean_ctor_set_uint8(v_mdecl_121_, sizeof(void*)*7, v_kind_107_);
v___f_122_ = lean_alloc_closure((void*)(lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg___lam__1), 3, 2);
lean_closure_set(v___f_122_, 0, v_mvarId_110_);
lean_closure_set(v___f_122_, 1, v_mdecl_121_);
v___x_123_ = lean_apply_1(v_modifyMCtx_111_, v___f_122_);
v___x_124_ = lean_apply_4(v_toBind_112_, lean_box(0), lean_box(0), v___x_123_, v___f_118_);
return v___x_124_;
}
v___jp_125_:
{
lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; 
v___x_128_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_128_, 0, v_ldecl_114_);
v___x_129_ = l_Lean_PersistentArray_set___redArg(v_decls_116_, v___y_127_, v___x_128_);
lean_dec(v___y_127_);
v___x_130_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_130_, 0, v___y_126_);
lean_ctor_set(v___x_130_, 1, v___x_129_);
lean_ctor_set(v___x_130_, 2, v_auxDeclToFullName_117_);
v___y_120_ = v___x_130_;
goto v___jp_119_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__2___boxed(lean_object* v_lctx_139_, lean_object* v_toPure_140_, lean_object* v_userName_141_, lean_object* v_type_142_, lean_object* v_depth_143_, lean_object* v_localInstances_144_, lean_object* v_kind_145_, lean_object* v_numScopeArgs_146_, lean_object* v_index_147_, lean_object* v_mvarId_148_, lean_object* v_modifyMCtx_149_, lean_object* v_toBind_150_, lean_object* v_fvarId_151_, lean_object* v_ldecl_152_){
_start:
{
uint8_t v_kind_boxed_153_; lean_object* v_res_154_; 
v_kind_boxed_153_ = lean_unbox(v_kind_145_);
v_res_154_ = lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__2(v_lctx_139_, v_toPure_140_, v_userName_141_, v_type_142_, v_depth_143_, v_localInstances_144_, v_kind_boxed_153_, v_numScopeArgs_146_, v_index_147_, v_mvarId_148_, v_modifyMCtx_149_, v_toBind_150_, v_fvarId_151_, v_ldecl_152_);
return v_res_154_;
}
}
static lean_object* _init_lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__1(void){
_start:
{
lean_object* v___x_156_; lean_object* v___x_157_; 
v___x_156_ = ((lean_object*)(lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__0));
v___x_157_ = l_Lean_stringToMessageData(v___x_156_);
return v___x_157_;
}
}
static lean_object* _init_lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__3(void){
_start:
{
lean_object* v___x_159_; lean_object* v___x_160_; 
v___x_159_ = ((lean_object*)(lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__2));
v___x_160_ = l_Lean_stringToMessageData(v___x_159_);
return v___x_160_;
}
}
static lean_object* _init_lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__5(void){
_start:
{
lean_object* v___x_162_; lean_object* v___x_163_; 
v___x_162_ = ((lean_object*)(lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__4));
v___x_163_ = l_Lean_stringToMessageData(v___x_162_);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1(lean_object* v_fvarId_164_, lean_object* v_toPure_165_, lean_object* v_mvarId_166_, lean_object* v_modifyMCtx_167_, lean_object* v_toBind_168_, lean_object* v_inst_169_, lean_object* v_inst_170_, lean_object* v_inst_171_, lean_object* v_mdecl_172_){
_start:
{
lean_object* v_userName_173_; lean_object* v_lctx_174_; lean_object* v_type_175_; lean_object* v_depth_176_; lean_object* v_localInstances_177_; uint8_t v_kind_178_; lean_object* v_numScopeArgs_179_; lean_object* v_index_180_; lean_object* v___x_181_; 
v_userName_173_ = lean_ctor_get(v_mdecl_172_, 0);
lean_inc(v_userName_173_);
v_lctx_174_ = lean_ctor_get(v_mdecl_172_, 1);
lean_inc_ref_n(v_lctx_174_, 2);
v_type_175_ = lean_ctor_get(v_mdecl_172_, 2);
lean_inc_ref(v_type_175_);
v_depth_176_ = lean_ctor_get(v_mdecl_172_, 3);
lean_inc(v_depth_176_);
v_localInstances_177_ = lean_ctor_get(v_mdecl_172_, 4);
lean_inc_ref(v_localInstances_177_);
v_kind_178_ = lean_ctor_get_uint8(v_mdecl_172_, sizeof(void*)*7);
v_numScopeArgs_179_ = lean_ctor_get(v_mdecl_172_, 5);
lean_inc(v_numScopeArgs_179_);
v_index_180_ = lean_ctor_get(v_mdecl_172_, 6);
lean_inc(v_index_180_);
lean_dec_ref(v_mdecl_172_);
lean_inc(v_fvarId_164_);
v___x_181_ = lean_local_ctx_find(v_lctx_174_, v_fvarId_164_);
if (lean_obj_tag(v___x_181_) == 1)
{
lean_object* v_val_182_; lean_object* v___x_183_; lean_object* v___f_184_; lean_object* v___x_185_; lean_object* v___x_186_; 
lean_dec_ref(v_inst_171_);
v_val_182_ = lean_ctor_get(v___x_181_, 0);
lean_inc(v_val_182_);
lean_dec_ref_known(v___x_181_, 1);
v___x_183_ = lean_box(v_kind_178_);
lean_inc(v_toBind_168_);
v___f_184_ = lean_alloc_closure((void*)(lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__2___boxed), 14, 13);
lean_closure_set(v___f_184_, 0, v_lctx_174_);
lean_closure_set(v___f_184_, 1, v_toPure_165_);
lean_closure_set(v___f_184_, 2, v_userName_173_);
lean_closure_set(v___f_184_, 3, v_type_175_);
lean_closure_set(v___f_184_, 4, v_depth_176_);
lean_closure_set(v___f_184_, 5, v_localInstances_177_);
lean_closure_set(v___f_184_, 6, v___x_183_);
lean_closure_set(v___f_184_, 7, v_numScopeArgs_179_);
lean_closure_set(v___f_184_, 8, v_index_180_);
lean_closure_set(v___f_184_, 9, v_mvarId_166_);
lean_closure_set(v___f_184_, 10, v_modifyMCtx_167_);
lean_closure_set(v___f_184_, 11, v_toBind_168_);
lean_closure_set(v___f_184_, 12, v_fvarId_164_);
v___x_185_ = l_Lean_instantiateLocalDeclMVars___redArg(v_inst_169_, v_inst_170_, v_val_182_);
v___x_186_ = lean_apply_4(v_toBind_168_, lean_box(0), lean_box(0), v___x_185_, v___f_184_);
return v___x_186_;
}
else
{
lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; 
lean_dec(v___x_181_);
lean_dec(v_index_180_);
lean_dec(v_numScopeArgs_179_);
lean_dec_ref(v_localInstances_177_);
lean_dec(v_depth_176_);
lean_dec_ref(v_type_175_);
lean_dec_ref(v_lctx_174_);
lean_dec(v_userName_173_);
lean_dec_ref(v_inst_170_);
lean_dec(v_toBind_168_);
lean_dec(v_modifyMCtx_167_);
lean_dec(v_toPure_165_);
v___x_187_ = lean_obj_once(&lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__1, &lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__1_once, _init_lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__1);
v___x_188_ = l_Lean_MessageData_ofName(v_fvarId_164_);
v___x_189_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_189_, 0, v___x_187_);
lean_ctor_set(v___x_189_, 1, v___x_188_);
v___x_190_ = lean_obj_once(&lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__3, &lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__3_once, _init_lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__3);
v___x_191_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_191_, 0, v___x_189_);
lean_ctor_set(v___x_191_, 1, v___x_190_);
v___x_192_ = l_Lean_MessageData_ofName(v_mvarId_166_);
v___x_193_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_193_, 0, v___x_191_);
lean_ctor_set(v___x_193_, 1, v___x_192_);
v___x_194_ = lean_obj_once(&lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__5, &lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__5_once, _init_lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1___closed__5);
v___x_195_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_195_, 0, v___x_193_);
lean_ctor_set(v___x_195_, 1, v___x_194_);
v___x_196_ = l_Lean_throwError___redArg(v_inst_169_, v_inst_171_, v___x_195_);
return v___x_196_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg(lean_object* v_inst_197_, lean_object* v_inst_198_, lean_object* v_inst_199_, lean_object* v_mvarId_200_, lean_object* v_fvarId_201_){
_start:
{
lean_object* v_toApplicative_202_; lean_object* v_toBind_203_; lean_object* v_getMCtx_204_; lean_object* v_modifyMCtx_205_; lean_object* v_toPure_206_; lean_object* v___f_207_; lean_object* v___f_208_; lean_object* v___x_209_; 
v_toApplicative_202_ = lean_ctor_get(v_inst_197_, 0);
v_toBind_203_ = lean_ctor_get(v_inst_197_, 1);
lean_inc_n(v_toBind_203_, 3);
v_getMCtx_204_ = lean_ctor_get(v_inst_198_, 0);
lean_inc(v_getMCtx_204_);
v_modifyMCtx_205_ = lean_ctor_get(v_inst_198_, 1);
lean_inc(v_modifyMCtx_205_);
v_toPure_206_ = lean_ctor_get(v_toApplicative_202_, 1);
lean_inc_ref(v_inst_199_);
lean_inc_ref(v_inst_197_);
lean_inc(v_mvarId_200_);
lean_inc(v_toPure_206_);
v___f_207_ = lean_alloc_closure((void*)(lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg___lam__1), 9, 8);
lean_closure_set(v___f_207_, 0, v_fvarId_201_);
lean_closure_set(v___f_207_, 1, v_toPure_206_);
lean_closure_set(v___f_207_, 2, v_mvarId_200_);
lean_closure_set(v___f_207_, 3, v_modifyMCtx_205_);
lean_closure_set(v___f_207_, 4, v_toBind_203_);
lean_closure_set(v___f_207_, 5, v_inst_197_);
lean_closure_set(v___f_207_, 6, v_inst_198_);
lean_closure_set(v___f_207_, 7, v_inst_199_);
v___f_208_ = lean_alloc_closure((void*)(lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg___lam__4___boxed), 6, 5);
lean_closure_set(v___f_208_, 0, v_inst_197_);
lean_closure_set(v___f_208_, 1, v_inst_199_);
lean_closure_set(v___f_208_, 2, v_mvarId_200_);
lean_closure_set(v___f_208_, 3, v_toBind_203_);
lean_closure_set(v___f_208_, 4, v___f_207_);
v___x_209_ = lean_apply_4(v_toBind_203_, lean_box(0), lean_box(0), v_getMCtx_204_, v___f_208_);
return v___x_209_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl(lean_object* v_m_210_, lean_object* v_inst_211_, lean_object* v_inst_212_, lean_object* v_inst_213_, lean_object* v_mvarId_214_, lean_object* v_fvarId_215_){
_start:
{
lean_object* v___x_216_; 
v___x_216_ = lp_batteries_Lean_MVarId_instantiateMVarsInLocalDecl___redArg(v_inst_211_, v_inst_212_, v_inst_213_, v_mvarId_214_, v_fvarId_215_);
return v___x_216_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalContext___redArg___lam__0(lean_object* v_userName_217_, lean_object* v_lctx_218_, lean_object* v_type_219_, lean_object* v_depth_220_, lean_object* v_localInstances_221_, uint8_t v_kind_222_, lean_object* v_numScopeArgs_223_, lean_object* v_index_224_, lean_object* v_mvarId_225_, lean_object* v_x_226_){
_start:
{
lean_object* v___x_227_; lean_object* v___x_228_; 
v___x_227_ = lean_alloc_ctor(0, 7, 1);
lean_ctor_set(v___x_227_, 0, v_userName_217_);
lean_ctor_set(v___x_227_, 1, v_lctx_218_);
lean_ctor_set(v___x_227_, 2, v_type_219_);
lean_ctor_set(v___x_227_, 3, v_depth_220_);
lean_ctor_set(v___x_227_, 4, v_localInstances_221_);
lean_ctor_set(v___x_227_, 5, v_numScopeArgs_223_);
lean_ctor_set(v___x_227_, 6, v_index_224_);
lean_ctor_set_uint8(v___x_227_, sizeof(void*)*7, v_kind_222_);
v___x_228_ = lp_batteries_Lean_MetavarContext_declareExprMVar(v_x_226_, v_mvarId_225_, v___x_227_);
return v___x_228_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalContext___redArg___lam__0___boxed(lean_object* v_userName_229_, lean_object* v_lctx_230_, lean_object* v_type_231_, lean_object* v_depth_232_, lean_object* v_localInstances_233_, lean_object* v_kind_234_, lean_object* v_numScopeArgs_235_, lean_object* v_index_236_, lean_object* v_mvarId_237_, lean_object* v_x_238_){
_start:
{
uint8_t v_kind_boxed_239_; lean_object* v_res_240_; 
v_kind_boxed_239_ = lean_unbox(v_kind_234_);
v_res_240_ = lp_batteries_Lean_MVarId_instantiateMVarsInLocalContext___redArg___lam__0(v_userName_229_, v_lctx_230_, v_type_231_, v_depth_232_, v_localInstances_233_, v_kind_boxed_239_, v_numScopeArgs_235_, v_index_236_, v_mvarId_237_, v_x_238_);
return v_res_240_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalContext___redArg___lam__1(lean_object* v_toPure_241_, lean_object* v_lctx_242_, lean_object* v_____r_243_){
_start:
{
lean_object* v___x_244_; 
v___x_244_ = lean_apply_2(v_toPure_241_, lean_box(0), v_lctx_242_);
return v___x_244_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalContext___redArg___lam__2(lean_object* v_userName_245_, lean_object* v_type_246_, lean_object* v_depth_247_, lean_object* v_localInstances_248_, uint8_t v_kind_249_, lean_object* v_numScopeArgs_250_, lean_object* v_index_251_, lean_object* v_mvarId_252_, lean_object* v_toPure_253_, lean_object* v_modifyMCtx_254_, lean_object* v_toBind_255_, lean_object* v_lctx_256_){
_start:
{
lean_object* v___x_257_; lean_object* v___f_258_; lean_object* v___f_259_; lean_object* v___x_260_; lean_object* v___x_261_; 
v___x_257_ = lean_box(v_kind_249_);
lean_inc_ref(v_lctx_256_);
v___f_258_ = lean_alloc_closure((void*)(lp_batteries_Lean_MVarId_instantiateMVarsInLocalContext___redArg___lam__0___boxed), 10, 9);
lean_closure_set(v___f_258_, 0, v_userName_245_);
lean_closure_set(v___f_258_, 1, v_lctx_256_);
lean_closure_set(v___f_258_, 2, v_type_246_);
lean_closure_set(v___f_258_, 3, v_depth_247_);
lean_closure_set(v___f_258_, 4, v_localInstances_248_);
lean_closure_set(v___f_258_, 5, v___x_257_);
lean_closure_set(v___f_258_, 6, v_numScopeArgs_250_);
lean_closure_set(v___f_258_, 7, v_index_251_);
lean_closure_set(v___f_258_, 8, v_mvarId_252_);
v___f_259_ = lean_alloc_closure((void*)(lp_batteries_Lean_MVarId_instantiateMVarsInLocalContext___redArg___lam__1), 3, 2);
lean_closure_set(v___f_259_, 0, v_toPure_253_);
lean_closure_set(v___f_259_, 1, v_lctx_256_);
v___x_260_ = lean_apply_1(v_modifyMCtx_254_, v___f_258_);
v___x_261_ = lean_apply_4(v_toBind_255_, lean_box(0), lean_box(0), v___x_260_, v___f_259_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalContext___redArg___lam__2___boxed(lean_object* v_userName_262_, lean_object* v_type_263_, lean_object* v_depth_264_, lean_object* v_localInstances_265_, lean_object* v_kind_266_, lean_object* v_numScopeArgs_267_, lean_object* v_index_268_, lean_object* v_mvarId_269_, lean_object* v_toPure_270_, lean_object* v_modifyMCtx_271_, lean_object* v_toBind_272_, lean_object* v_lctx_273_){
_start:
{
uint8_t v_kind_boxed_274_; lean_object* v_res_275_; 
v_kind_boxed_274_ = lean_unbox(v_kind_266_);
v_res_275_ = lp_batteries_Lean_MVarId_instantiateMVarsInLocalContext___redArg___lam__2(v_userName_262_, v_type_263_, v_depth_264_, v_localInstances_265_, v_kind_boxed_274_, v_numScopeArgs_267_, v_index_268_, v_mvarId_269_, v_toPure_270_, v_modifyMCtx_271_, v_toBind_272_, v_lctx_273_);
return v_res_275_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalContext___redArg___lam__3(lean_object* v_mvarId_276_, lean_object* v_toPure_277_, lean_object* v_modifyMCtx_278_, lean_object* v_toBind_279_, lean_object* v_inst_280_, lean_object* v_inst_281_, lean_object* v_mdecl_282_){
_start:
{
lean_object* v_userName_283_; lean_object* v_lctx_284_; lean_object* v_type_285_; lean_object* v_depth_286_; lean_object* v_localInstances_287_; uint8_t v_kind_288_; lean_object* v_numScopeArgs_289_; lean_object* v_index_290_; lean_object* v___x_291_; lean_object* v___f_292_; lean_object* v___x_293_; lean_object* v___x_294_; 
v_userName_283_ = lean_ctor_get(v_mdecl_282_, 0);
lean_inc(v_userName_283_);
v_lctx_284_ = lean_ctor_get(v_mdecl_282_, 1);
lean_inc_ref(v_lctx_284_);
v_type_285_ = lean_ctor_get(v_mdecl_282_, 2);
lean_inc_ref(v_type_285_);
v_depth_286_ = lean_ctor_get(v_mdecl_282_, 3);
lean_inc(v_depth_286_);
v_localInstances_287_ = lean_ctor_get(v_mdecl_282_, 4);
lean_inc_ref(v_localInstances_287_);
v_kind_288_ = lean_ctor_get_uint8(v_mdecl_282_, sizeof(void*)*7);
v_numScopeArgs_289_ = lean_ctor_get(v_mdecl_282_, 5);
lean_inc(v_numScopeArgs_289_);
v_index_290_ = lean_ctor_get(v_mdecl_282_, 6);
lean_inc(v_index_290_);
lean_dec_ref(v_mdecl_282_);
v___x_291_ = lean_box(v_kind_288_);
lean_inc(v_toBind_279_);
v___f_292_ = lean_alloc_closure((void*)(lp_batteries_Lean_MVarId_instantiateMVarsInLocalContext___redArg___lam__2___boxed), 12, 11);
lean_closure_set(v___f_292_, 0, v_userName_283_);
lean_closure_set(v___f_292_, 1, v_type_285_);
lean_closure_set(v___f_292_, 2, v_depth_286_);
lean_closure_set(v___f_292_, 3, v_localInstances_287_);
lean_closure_set(v___f_292_, 4, v___x_291_);
lean_closure_set(v___f_292_, 5, v_numScopeArgs_289_);
lean_closure_set(v___f_292_, 6, v_index_290_);
lean_closure_set(v___f_292_, 7, v_mvarId_276_);
lean_closure_set(v___f_292_, 8, v_toPure_277_);
lean_closure_set(v___f_292_, 9, v_modifyMCtx_278_);
lean_closure_set(v___f_292_, 10, v_toBind_279_);
v___x_293_ = l_Lean_instantiateLCtxMVars___redArg(v_inst_280_, v_inst_281_, v_lctx_284_);
v___x_294_ = lean_apply_4(v_toBind_279_, lean_box(0), lean_box(0), v___x_293_, v___f_292_);
return v___x_294_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalContext___redArg(lean_object* v_inst_295_, lean_object* v_inst_296_, lean_object* v_inst_297_, lean_object* v_mvarId_298_){
_start:
{
lean_object* v_toApplicative_299_; lean_object* v_toBind_300_; lean_object* v_getMCtx_301_; lean_object* v_modifyMCtx_302_; lean_object* v_toPure_303_; lean_object* v___f_304_; lean_object* v___f_305_; lean_object* v___x_306_; 
v_toApplicative_299_ = lean_ctor_get(v_inst_295_, 0);
v_toBind_300_ = lean_ctor_get(v_inst_295_, 1);
lean_inc_n(v_toBind_300_, 3);
v_getMCtx_301_ = lean_ctor_get(v_inst_296_, 0);
lean_inc(v_getMCtx_301_);
v_modifyMCtx_302_ = lean_ctor_get(v_inst_296_, 1);
lean_inc(v_modifyMCtx_302_);
v_toPure_303_ = lean_ctor_get(v_toApplicative_299_, 1);
lean_inc_ref(v_inst_295_);
lean_inc(v_toPure_303_);
lean_inc(v_mvarId_298_);
v___f_304_ = lean_alloc_closure((void*)(lp_batteries_Lean_MVarId_instantiateMVarsInLocalContext___redArg___lam__3), 7, 6);
lean_closure_set(v___f_304_, 0, v_mvarId_298_);
lean_closure_set(v___f_304_, 1, v_toPure_303_);
lean_closure_set(v___f_304_, 2, v_modifyMCtx_302_);
lean_closure_set(v___f_304_, 3, v_toBind_300_);
lean_closure_set(v___f_304_, 4, v_inst_295_);
lean_closure_set(v___f_304_, 5, v_inst_296_);
v___f_305_ = lean_alloc_closure((void*)(lp_batteries_Lean_MVarId_instantiateMVarsInType___redArg___lam__4___boxed), 6, 5);
lean_closure_set(v___f_305_, 0, v_inst_295_);
lean_closure_set(v___f_305_, 1, v_inst_297_);
lean_closure_set(v___f_305_, 2, v_mvarId_298_);
lean_closure_set(v___f_305_, 3, v_toBind_300_);
lean_closure_set(v___f_305_, 4, v___f_304_);
v___x_306_ = lean_apply_4(v_toBind_300_, lean_box(0), lean_box(0), v_getMCtx_301_, v___f_305_);
return v___x_306_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVarsInLocalContext(lean_object* v_m_307_, lean_object* v_inst_308_, lean_object* v_inst_309_, lean_object* v_inst_310_, lean_object* v_mvarId_311_){
_start:
{
lean_object* v___x_312_; 
v___x_312_ = lp_batteries_Lean_MVarId_instantiateMVarsInLocalContext___redArg(v_inst_308_, v_inst_309_, v_inst_310_, v_mvarId_311_);
return v___x_312_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVars___redArg___lam__0(lean_object* v_inst_313_, lean_object* v_inst_314_, lean_object* v_mvarId_315_, lean_object* v_____r_316_){
_start:
{
lean_object* v___x_317_; 
v___x_317_ = l_Lean_instantiateMVarDeclMVars___redArg(v_inst_313_, v_inst_314_, v_mvarId_315_);
return v___x_317_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVars___redArg___lam__1(lean_object* v_toFunctor_318_, lean_object* v_inst_319_, lean_object* v_inst_320_, lean_object* v_mvarId_321_, lean_object* v_toBind_322_, lean_object* v___f_323_, lean_object* v_____do__lift_324_){
_start:
{
lean_object* v_mapConst_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; 
v_mapConst_325_ = lean_ctor_get(v_toFunctor_318_, 1);
lean_inc(v_mapConst_325_);
lean_dec_ref(v_toFunctor_318_);
v___x_326_ = lp_batteries_Lean_MetavarContext_getExprMVarDecl___redArg(v_inst_319_, v_inst_320_, v_____do__lift_324_, v_mvarId_321_);
v___x_327_ = lean_box(0);
v___x_328_ = lean_apply_4(v_mapConst_325_, lean_box(0), lean_box(0), v___x_327_, v___x_326_);
v___x_329_ = lean_apply_4(v_toBind_322_, lean_box(0), lean_box(0), v___x_328_, v___f_323_);
return v___x_329_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVars___redArg___lam__1___boxed(lean_object* v_toFunctor_330_, lean_object* v_inst_331_, lean_object* v_inst_332_, lean_object* v_mvarId_333_, lean_object* v_toBind_334_, lean_object* v___f_335_, lean_object* v_____do__lift_336_){
_start:
{
lean_object* v_res_337_; 
v_res_337_ = lp_batteries_Lean_MVarId_instantiateMVars___redArg___lam__1(v_toFunctor_330_, v_inst_331_, v_inst_332_, v_mvarId_333_, v_toBind_334_, v___f_335_, v_____do__lift_336_);
lean_dec_ref(v_____do__lift_336_);
return v_res_337_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVars___redArg(lean_object* v_inst_338_, lean_object* v_inst_339_, lean_object* v_inst_340_, lean_object* v_mvarId_341_){
_start:
{
lean_object* v_toApplicative_342_; lean_object* v_toBind_343_; lean_object* v_getMCtx_344_; lean_object* v_toFunctor_345_; lean_object* v___f_346_; lean_object* v___f_347_; lean_object* v___x_348_; 
v_toApplicative_342_ = lean_ctor_get(v_inst_338_, 0);
v_toBind_343_ = lean_ctor_get(v_inst_338_, 1);
lean_inc_n(v_toBind_343_, 2);
v_getMCtx_344_ = lean_ctor_get(v_inst_339_, 0);
lean_inc(v_getMCtx_344_);
v_toFunctor_345_ = lean_ctor_get(v_toApplicative_342_, 0);
lean_inc_ref(v_toFunctor_345_);
lean_inc(v_mvarId_341_);
lean_inc_ref(v_inst_338_);
v___f_346_ = lean_alloc_closure((void*)(lp_batteries_Lean_MVarId_instantiateMVars___redArg___lam__0), 4, 3);
lean_closure_set(v___f_346_, 0, v_inst_338_);
lean_closure_set(v___f_346_, 1, v_inst_339_);
lean_closure_set(v___f_346_, 2, v_mvarId_341_);
v___f_347_ = lean_alloc_closure((void*)(lp_batteries_Lean_MVarId_instantiateMVars___redArg___lam__1___boxed), 7, 6);
lean_closure_set(v___f_347_, 0, v_toFunctor_345_);
lean_closure_set(v___f_347_, 1, v_inst_338_);
lean_closure_set(v___f_347_, 2, v_inst_340_);
lean_closure_set(v___f_347_, 3, v_mvarId_341_);
lean_closure_set(v___f_347_, 4, v_toBind_343_);
lean_closure_set(v___f_347_, 5, v___f_346_);
v___x_348_ = lean_apply_4(v_toBind_343_, lean_box(0), lean_box(0), v_getMCtx_344_, v___f_347_);
return v___x_348_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_instantiateMVars(lean_object* v_m_349_, lean_object* v_inst_350_, lean_object* v_inst_351_, lean_object* v_inst_352_, lean_object* v_mvarId_353_){
_start:
{
lean_object* v___x_354_; 
v___x_354_ = lp_batteries_Lean_MVarId_instantiateMVars___redArg(v_inst_350_, v_inst_351_, v_inst_352_, v_mvarId_353_);
return v___x_354_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_InstantiateMVars(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Lean_Meta_InstantiateMVars(uint8_t builtin) {
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
lean_object* initialize_batteries_Batteries_Lean_Meta_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Lean_Meta_InstantiateMVars(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_InstantiateMVars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Lean_Meta_InstantiateMVars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Lean_Meta_InstantiateMVars(builtin);
}
#ifdef __cplusplus
}
#endif
