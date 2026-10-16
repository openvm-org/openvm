// Lean compiler output
// Module: Mathlib.Util.Tactic
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.MetavarContext
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
lean_object* l_Lean_instBEqMVarId_beq___boxed(lean_object*, lean_object*);
lean_object* l_Lean_instHashableMVarId_hash___boxed(lean_object*);
lean_object* l_Lean_PersistentHashMap_find_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_insert___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_local_ctx_find(lean_object*, lean_object*);
lean_object* l_Lean_instBEqFVarId_beq___boxed(lean_object*, lean_object*);
lean_object* l_Lean_instHashableFVarId_hash___boxed(lean_object*);
lean_object* l_Lean_PersistentArray_set___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instBEqMVarId_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instHashableMVarId_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyMetavarDecl(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyTarget___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyTarget___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyTarget(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalContext___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalContext___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalContext(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instBEqFVarId_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instHashableFVarId_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___lam__0(lean_object* v___x_1_, lean_object* v___x_2_, lean_object* v_mvarId_3_, lean_object* v_f_4_, lean_object* v_mctx_5_){
_start:
{
lean_object* v_depth_6_; lean_object* v_levelAssignDepth_7_; lean_object* v_lmvarCounter_8_; lean_object* v_mvarCounter_9_; lean_object* v_lDecls_10_; lean_object* v_decls_11_; lean_object* v_userNames_12_; lean_object* v_lAssignment_13_; lean_object* v_eAssignment_14_; lean_object* v_dAssignment_15_; lean_object* v___x_16_; 
v_depth_6_ = lean_ctor_get(v_mctx_5_, 0);
v_levelAssignDepth_7_ = lean_ctor_get(v_mctx_5_, 1);
v_lmvarCounter_8_ = lean_ctor_get(v_mctx_5_, 2);
v_mvarCounter_9_ = lean_ctor_get(v_mctx_5_, 3);
v_lDecls_10_ = lean_ctor_get(v_mctx_5_, 4);
v_decls_11_ = lean_ctor_get(v_mctx_5_, 5);
v_userNames_12_ = lean_ctor_get(v_mctx_5_, 6);
v_lAssignment_13_ = lean_ctor_get(v_mctx_5_, 7);
v_eAssignment_14_ = lean_ctor_get(v_mctx_5_, 8);
v_dAssignment_15_ = lean_ctor_get(v_mctx_5_, 9);
lean_inc(v_mvarId_3_);
lean_inc_ref(v___x_2_);
lean_inc_ref(v___x_1_);
v___x_16_ = l_Lean_PersistentHashMap_find_x3f___redArg(v___x_1_, v___x_2_, v_decls_11_, v_mvarId_3_);
if (lean_obj_tag(v___x_16_) == 0)
{
lean_dec_ref(v_f_4_);
lean_dec(v_mvarId_3_);
lean_dec_ref(v___x_2_);
lean_dec_ref(v___x_1_);
return v_mctx_5_;
}
else
{
lean_object* v___x_18_; uint8_t v_isShared_19_; uint8_t v_isSharedCheck_26_; 
lean_inc_ref(v_dAssignment_15_);
lean_inc_ref(v_eAssignment_14_);
lean_inc_ref(v_lAssignment_13_);
lean_inc_ref(v_userNames_12_);
lean_inc_ref(v_decls_11_);
lean_inc_ref(v_lDecls_10_);
lean_inc(v_mvarCounter_9_);
lean_inc(v_lmvarCounter_8_);
lean_inc(v_levelAssignDepth_7_);
lean_inc(v_depth_6_);
v_isSharedCheck_26_ = !lean_is_exclusive(v_mctx_5_);
if (v_isSharedCheck_26_ == 0)
{
lean_object* v_unused_27_; lean_object* v_unused_28_; lean_object* v_unused_29_; lean_object* v_unused_30_; lean_object* v_unused_31_; lean_object* v_unused_32_; lean_object* v_unused_33_; lean_object* v_unused_34_; lean_object* v_unused_35_; lean_object* v_unused_36_; 
v_unused_27_ = lean_ctor_get(v_mctx_5_, 9);
lean_dec(v_unused_27_);
v_unused_28_ = lean_ctor_get(v_mctx_5_, 8);
lean_dec(v_unused_28_);
v_unused_29_ = lean_ctor_get(v_mctx_5_, 7);
lean_dec(v_unused_29_);
v_unused_30_ = lean_ctor_get(v_mctx_5_, 6);
lean_dec(v_unused_30_);
v_unused_31_ = lean_ctor_get(v_mctx_5_, 5);
lean_dec(v_unused_31_);
v_unused_32_ = lean_ctor_get(v_mctx_5_, 4);
lean_dec(v_unused_32_);
v_unused_33_ = lean_ctor_get(v_mctx_5_, 3);
lean_dec(v_unused_33_);
v_unused_34_ = lean_ctor_get(v_mctx_5_, 2);
lean_dec(v_unused_34_);
v_unused_35_ = lean_ctor_get(v_mctx_5_, 1);
lean_dec(v_unused_35_);
v_unused_36_ = lean_ctor_get(v_mctx_5_, 0);
lean_dec(v_unused_36_);
v___x_18_ = v_mctx_5_;
v_isShared_19_ = v_isSharedCheck_26_;
goto v_resetjp_17_;
}
else
{
lean_dec(v_mctx_5_);
v___x_18_ = lean_box(0);
v_isShared_19_ = v_isSharedCheck_26_;
goto v_resetjp_17_;
}
v_resetjp_17_:
{
lean_object* v_val_20_; lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_24_; 
v_val_20_ = lean_ctor_get(v___x_16_, 0);
lean_inc(v_val_20_);
lean_dec_ref_known(v___x_16_, 1);
v___x_21_ = lean_apply_1(v_f_4_, v_val_20_);
v___x_22_ = l_Lean_PersistentHashMap_insert___redArg(v___x_1_, v___x_2_, v_decls_11_, v_mvarId_3_, v___x_21_);
if (v_isShared_19_ == 0)
{
lean_ctor_set(v___x_18_, 5, v___x_22_);
v___x_24_ = v___x_18_;
goto v_reusejp_23_;
}
else
{
lean_object* v_reuseFailAlloc_25_; 
v_reuseFailAlloc_25_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_25_, 0, v_depth_6_);
lean_ctor_set(v_reuseFailAlloc_25_, 1, v_levelAssignDepth_7_);
lean_ctor_set(v_reuseFailAlloc_25_, 2, v_lmvarCounter_8_);
lean_ctor_set(v_reuseFailAlloc_25_, 3, v_mvarCounter_9_);
lean_ctor_set(v_reuseFailAlloc_25_, 4, v_lDecls_10_);
lean_ctor_set(v_reuseFailAlloc_25_, 5, v___x_22_);
lean_ctor_set(v_reuseFailAlloc_25_, 6, v_userNames_12_);
lean_ctor_set(v_reuseFailAlloc_25_, 7, v_lAssignment_13_);
lean_ctor_set(v_reuseFailAlloc_25_, 8, v_eAssignment_14_);
lean_ctor_set(v_reuseFailAlloc_25_, 9, v_dAssignment_15_);
v___x_24_ = v_reuseFailAlloc_25_;
goto v_reusejp_23_;
}
v_reusejp_23_:
{
return v___x_24_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg(lean_object* v_inst_39_, lean_object* v_mvarId_40_, lean_object* v_f_41_){
_start:
{
lean_object* v_modifyMCtx_42_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___f_45_; lean_object* v___x_46_; 
v_modifyMCtx_42_ = lean_ctor_get(v_inst_39_, 1);
lean_inc(v_modifyMCtx_42_);
lean_dec_ref(v_inst_39_);
v___x_43_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___closed__0));
v___x_44_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___closed__1));
v___f_45_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___lam__0), 5, 4);
lean_closure_set(v___f_45_, 0, v___x_43_);
lean_closure_set(v___f_45_, 1, v___x_44_);
lean_closure_set(v___f_45_, 2, v_mvarId_40_);
lean_closure_set(v___f_45_, 3, v_f_41_);
v___x_46_ = lean_apply_1(v_modifyMCtx_42_, v___f_45_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyMetavarDecl(lean_object* v_m_47_, lean_object* v_inst_48_, lean_object* v_mvarId_49_, lean_object* v_f_50_){
_start:
{
lean_object* v___x_51_; 
v___x_51_ = lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg(v_inst_48_, v_mvarId_49_, v_f_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyTarget___redArg___lam__0(lean_object* v_f_52_, lean_object* v_mdecl_53_){
_start:
{
lean_object* v_userName_54_; lean_object* v_lctx_55_; lean_object* v_type_56_; lean_object* v_depth_57_; lean_object* v_localInstances_58_; uint8_t v_kind_59_; lean_object* v_numScopeArgs_60_; lean_object* v_index_61_; lean_object* v___x_63_; uint8_t v_isShared_64_; uint8_t v_isSharedCheck_69_; 
v_userName_54_ = lean_ctor_get(v_mdecl_53_, 0);
v_lctx_55_ = lean_ctor_get(v_mdecl_53_, 1);
v_type_56_ = lean_ctor_get(v_mdecl_53_, 2);
v_depth_57_ = lean_ctor_get(v_mdecl_53_, 3);
v_localInstances_58_ = lean_ctor_get(v_mdecl_53_, 4);
v_kind_59_ = lean_ctor_get_uint8(v_mdecl_53_, sizeof(void*)*7);
v_numScopeArgs_60_ = lean_ctor_get(v_mdecl_53_, 5);
v_index_61_ = lean_ctor_get(v_mdecl_53_, 6);
v_isSharedCheck_69_ = !lean_is_exclusive(v_mdecl_53_);
if (v_isSharedCheck_69_ == 0)
{
v___x_63_ = v_mdecl_53_;
v_isShared_64_ = v_isSharedCheck_69_;
goto v_resetjp_62_;
}
else
{
lean_inc(v_index_61_);
lean_inc(v_numScopeArgs_60_);
lean_inc(v_localInstances_58_);
lean_inc(v_depth_57_);
lean_inc(v_type_56_);
lean_inc(v_lctx_55_);
lean_inc(v_userName_54_);
lean_dec(v_mdecl_53_);
v___x_63_ = lean_box(0);
v_isShared_64_ = v_isSharedCheck_69_;
goto v_resetjp_62_;
}
v_resetjp_62_:
{
lean_object* v___x_65_; lean_object* v___x_67_; 
v___x_65_ = lean_apply_1(v_f_52_, v_type_56_);
if (v_isShared_64_ == 0)
{
lean_ctor_set(v___x_63_, 2, v___x_65_);
v___x_67_ = v___x_63_;
goto v_reusejp_66_;
}
else
{
lean_object* v_reuseFailAlloc_68_; 
v_reuseFailAlloc_68_ = lean_alloc_ctor(0, 7, 1);
lean_ctor_set(v_reuseFailAlloc_68_, 0, v_userName_54_);
lean_ctor_set(v_reuseFailAlloc_68_, 1, v_lctx_55_);
lean_ctor_set(v_reuseFailAlloc_68_, 2, v___x_65_);
lean_ctor_set(v_reuseFailAlloc_68_, 3, v_depth_57_);
lean_ctor_set(v_reuseFailAlloc_68_, 4, v_localInstances_58_);
lean_ctor_set(v_reuseFailAlloc_68_, 5, v_numScopeArgs_60_);
lean_ctor_set(v_reuseFailAlloc_68_, 6, v_index_61_);
lean_ctor_set_uint8(v_reuseFailAlloc_68_, sizeof(void*)*7, v_kind_59_);
v___x_67_ = v_reuseFailAlloc_68_;
goto v_reusejp_66_;
}
v_reusejp_66_:
{
return v___x_67_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyTarget___redArg(lean_object* v_inst_70_, lean_object* v_mvarId_71_, lean_object* v_f_72_){
_start:
{
lean_object* v___f_73_; lean_object* v___x_74_; 
v___f_73_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_modifyTarget___redArg___lam__0), 2, 1);
lean_closure_set(v___f_73_, 0, v_f_72_);
v___x_74_ = lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg(v_inst_70_, v_mvarId_71_, v___f_73_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyTarget(lean_object* v_m_75_, lean_object* v_inst_76_, lean_object* v_mvarId_77_, lean_object* v_f_78_){
_start:
{
lean_object* v___x_79_; 
v___x_79_ = lp_mathlib_Mathlib_Tactic_modifyTarget___redArg(v_inst_76_, v_mvarId_77_, v_f_78_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalContext___redArg___lam__0(lean_object* v_f_80_, lean_object* v_mdecl_81_){
_start:
{
lean_object* v_userName_82_; lean_object* v_lctx_83_; lean_object* v_type_84_; lean_object* v_depth_85_; lean_object* v_localInstances_86_; uint8_t v_kind_87_; lean_object* v_numScopeArgs_88_; lean_object* v_index_89_; lean_object* v___x_91_; uint8_t v_isShared_92_; uint8_t v_isSharedCheck_97_; 
v_userName_82_ = lean_ctor_get(v_mdecl_81_, 0);
v_lctx_83_ = lean_ctor_get(v_mdecl_81_, 1);
v_type_84_ = lean_ctor_get(v_mdecl_81_, 2);
v_depth_85_ = lean_ctor_get(v_mdecl_81_, 3);
v_localInstances_86_ = lean_ctor_get(v_mdecl_81_, 4);
v_kind_87_ = lean_ctor_get_uint8(v_mdecl_81_, sizeof(void*)*7);
v_numScopeArgs_88_ = lean_ctor_get(v_mdecl_81_, 5);
v_index_89_ = lean_ctor_get(v_mdecl_81_, 6);
v_isSharedCheck_97_ = !lean_is_exclusive(v_mdecl_81_);
if (v_isSharedCheck_97_ == 0)
{
v___x_91_ = v_mdecl_81_;
v_isShared_92_ = v_isSharedCheck_97_;
goto v_resetjp_90_;
}
else
{
lean_inc(v_index_89_);
lean_inc(v_numScopeArgs_88_);
lean_inc(v_localInstances_86_);
lean_inc(v_depth_85_);
lean_inc(v_type_84_);
lean_inc(v_lctx_83_);
lean_inc(v_userName_82_);
lean_dec(v_mdecl_81_);
v___x_91_ = lean_box(0);
v_isShared_92_ = v_isSharedCheck_97_;
goto v_resetjp_90_;
}
v_resetjp_90_:
{
lean_object* v___x_93_; lean_object* v___x_95_; 
v___x_93_ = lean_apply_1(v_f_80_, v_lctx_83_);
if (v_isShared_92_ == 0)
{
lean_ctor_set(v___x_91_, 1, v___x_93_);
v___x_95_ = v___x_91_;
goto v_reusejp_94_;
}
else
{
lean_object* v_reuseFailAlloc_96_; 
v_reuseFailAlloc_96_ = lean_alloc_ctor(0, 7, 1);
lean_ctor_set(v_reuseFailAlloc_96_, 0, v_userName_82_);
lean_ctor_set(v_reuseFailAlloc_96_, 1, v___x_93_);
lean_ctor_set(v_reuseFailAlloc_96_, 2, v_type_84_);
lean_ctor_set(v_reuseFailAlloc_96_, 3, v_depth_85_);
lean_ctor_set(v_reuseFailAlloc_96_, 4, v_localInstances_86_);
lean_ctor_set(v_reuseFailAlloc_96_, 5, v_numScopeArgs_88_);
lean_ctor_set(v_reuseFailAlloc_96_, 6, v_index_89_);
lean_ctor_set_uint8(v_reuseFailAlloc_96_, sizeof(void*)*7, v_kind_87_);
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
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalContext___redArg(lean_object* v_inst_98_, lean_object* v_mvarId_99_, lean_object* v_f_100_){
_start:
{
lean_object* v___f_101_; lean_object* v___x_102_; 
v___f_101_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_modifyLocalContext___redArg___lam__0), 2, 1);
lean_closure_set(v___f_101_, 0, v_f_100_);
v___x_102_ = lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg(v_inst_98_, v_mvarId_99_, v___f_101_);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalContext(lean_object* v_m_103_, lean_object* v_inst_104_, lean_object* v_mvarId_105_, lean_object* v_f_106_){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = lp_mathlib_Mathlib_Tactic_modifyLocalContext___redArg(v_inst_104_, v_mvarId_105_, v_f_106_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0(lean_object* v_fvarId_110_, lean_object* v_f_111_, lean_object* v_lctx_112_){
_start:
{
lean_object* v_fvarIdToDecl_113_; lean_object* v_decls_114_; lean_object* v_auxDeclToFullName_115_; lean_object* v___x_116_; 
v_fvarIdToDecl_113_ = lean_ctor_get(v_lctx_112_, 0);
v_decls_114_ = lean_ctor_get(v_lctx_112_, 1);
v_auxDeclToFullName_115_ = lean_ctor_get(v_lctx_112_, 2);
lean_inc_ref(v_lctx_112_);
v___x_116_ = lean_local_ctx_find(v_lctx_112_, v_fvarId_110_);
if (lean_obj_tag(v___x_116_) == 0)
{
lean_dec_ref(v_f_111_);
return v_lctx_112_;
}
else
{
lean_object* v___x_118_; uint8_t v_isShared_119_; uint8_t v_isSharedCheck_143_; 
lean_inc(v_auxDeclToFullName_115_);
lean_inc_ref(v_decls_114_);
lean_inc_ref(v_fvarIdToDecl_113_);
v_isSharedCheck_143_ = !lean_is_exclusive(v_lctx_112_);
if (v_isSharedCheck_143_ == 0)
{
lean_object* v_unused_144_; lean_object* v_unused_145_; lean_object* v_unused_146_; 
v_unused_144_ = lean_ctor_get(v_lctx_112_, 2);
lean_dec(v_unused_144_);
v_unused_145_ = lean_ctor_get(v_lctx_112_, 1);
lean_dec(v_unused_145_);
v_unused_146_ = lean_ctor_get(v_lctx_112_, 0);
lean_dec(v_unused_146_);
v___x_118_ = v_lctx_112_;
v_isShared_119_ = v_isSharedCheck_143_;
goto v_resetjp_117_;
}
else
{
lean_dec(v_lctx_112_);
v___x_118_ = lean_box(0);
v_isShared_119_ = v_isSharedCheck_143_;
goto v_resetjp_117_;
}
v_resetjp_117_:
{
lean_object* v_val_120_; lean_object* v___x_122_; uint8_t v_isShared_123_; uint8_t v_isSharedCheck_142_; 
v_val_120_ = lean_ctor_get(v___x_116_, 0);
v_isSharedCheck_142_ = !lean_is_exclusive(v___x_116_);
if (v_isSharedCheck_142_ == 0)
{
v___x_122_ = v___x_116_;
v_isShared_123_ = v_isSharedCheck_142_;
goto v_resetjp_121_;
}
else
{
lean_inc(v_val_120_);
lean_dec(v___x_116_);
v___x_122_ = lean_box(0);
v_isShared_123_ = v_isSharedCheck_142_;
goto v_resetjp_121_;
}
v_resetjp_121_:
{
lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v_decl_126_; lean_object* v___y_128_; lean_object* v___y_129_; lean_object* v___y_138_; lean_object* v_fvarId_141_; 
v___x_124_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0___closed__0));
v___x_125_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0___closed__1));
v_decl_126_ = lean_apply_1(v_f_111_, v_val_120_);
v_fvarId_141_ = lean_ctor_get(v_decl_126_, 1);
lean_inc(v_fvarId_141_);
v___y_138_ = v_fvarId_141_;
goto v___jp_137_;
v___jp_127_:
{
lean_object* v___x_131_; 
if (v_isShared_123_ == 0)
{
lean_ctor_set(v___x_122_, 0, v_decl_126_);
v___x_131_ = v___x_122_;
goto v_reusejp_130_;
}
else
{
lean_object* v_reuseFailAlloc_136_; 
v_reuseFailAlloc_136_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_136_, 0, v_decl_126_);
v___x_131_ = v_reuseFailAlloc_136_;
goto v_reusejp_130_;
}
v_reusejp_130_:
{
lean_object* v___x_132_; lean_object* v___x_134_; 
v___x_132_ = l_Lean_PersistentArray_set___redArg(v_decls_114_, v___y_129_, v___x_131_);
lean_dec(v___y_129_);
if (v_isShared_119_ == 0)
{
lean_ctor_set(v___x_118_, 1, v___x_132_);
lean_ctor_set(v___x_118_, 0, v___y_128_);
v___x_134_ = v___x_118_;
goto v_reusejp_133_;
}
else
{
lean_object* v_reuseFailAlloc_135_; 
v_reuseFailAlloc_135_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_135_, 0, v___y_128_);
lean_ctor_set(v_reuseFailAlloc_135_, 1, v___x_132_);
lean_ctor_set(v_reuseFailAlloc_135_, 2, v_auxDeclToFullName_115_);
v___x_134_ = v_reuseFailAlloc_135_;
goto v_reusejp_133_;
}
v_reusejp_133_:
{
return v___x_134_;
}
}
}
v___jp_137_:
{
lean_object* v___x_139_; lean_object* v_index_140_; 
lean_inc_ref(v_decl_126_);
v___x_139_ = l_Lean_PersistentHashMap_insert___redArg(v___x_124_, v___x_125_, v_fvarIdToDecl_113_, v___y_138_, v_decl_126_);
v_index_140_ = lean_ctor_get(v_decl_126_, 0);
lean_inc(v_index_140_);
v___y_128_ = v___x_139_;
v___y_129_ = v_index_140_;
goto v___jp_127_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg(lean_object* v_inst_147_, lean_object* v_mvarId_148_, lean_object* v_fvarId_149_, lean_object* v_f_150_){
_start:
{
lean_object* v___f_151_; lean_object* v___x_152_; 
v___f_151_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0), 3, 2);
lean_closure_set(v___f_151_, 0, v_fvarId_149_);
lean_closure_set(v___f_151_, 1, v_f_150_);
v___x_152_ = lp_mathlib_Mathlib_Tactic_modifyLocalContext___redArg(v_inst_147_, v_mvarId_148_, v___f_151_);
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalDecl(lean_object* v_m_153_, lean_object* v_inst_154_, lean_object* v_mvarId_155_, lean_object* v_fvarId_156_, lean_object* v_f_157_){
_start:
{
lean_object* v___x_158_; 
v___x_158_ = lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg(v_inst_154_, v_mvarId_155_, v_fvarId_156_, v_f_157_);
return v___x_158_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Util_Tactic(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_MetavarContext(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Util_Tactic(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_MetavarContext(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_MetavarContext(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Util_Tactic(uint8_t builtin) {
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
res = initialize_Lean_MetavarContext(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Util_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Util_Tactic(builtin);
}
#ifdef __cplusplus
}
#endif
