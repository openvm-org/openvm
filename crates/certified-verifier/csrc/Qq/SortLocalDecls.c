// Lean compiler output
// Module: Qq.SortLocalDecls
// Imports: public import Init public meta import Init public import Lean.Meta.Basic
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
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
extern lean_object* l_Lean_NameSet_empty;
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
uint8_t l_Lean_NameSet_contains(lean_object*, lean_object*);
lean_object* l_Lean_NameSet_insert(lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_type(lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
uint8_t l_Lean_Expr_isMVar(lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_value_x3f(lean_object*, uint8_t);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_SortLocalDecls_visitExpr_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_SortLocalDecls_visitExpr_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_SortLocalDecls_visitExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_SortLocalDecls_visitLocalDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_SortLocalDecls_visitLocalDecl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_SortLocalDecls_visitExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_SortLocalDecls_visitExpr_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_SortLocalDecls_visitExpr_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_Qq_Qq_sortLocalDecls___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_Qq_Qq_sortLocalDecls___closed__0 = (const lean_object*)&lp_Qq_Qq_sortLocalDecls___closed__0_value;
static lean_once_cell_t lp_Qq_Qq_sortLocalDecls___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_sortLocalDecls___closed__1;
LEAN_EXPORT lean_object* lp_Qq_Qq_sortLocalDecls(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_sortLocalDecls___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_SortLocalDecls_visitExpr_spec__0___redArg(lean_object* v_e_1_, lean_object* v___y_2_){
_start:
{
uint8_t v___x_4_; 
v___x_4_ = l_Lean_Expr_hasMVar(v_e_1_);
if (v___x_4_ == 0)
{
lean_object* v___x_5_; 
v___x_5_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5_, 0, v_e_1_);
return v___x_5_;
}
else
{
lean_object* v___x_6_; lean_object* v_mctx_7_; lean_object* v___x_8_; lean_object* v_fst_9_; lean_object* v_snd_10_; lean_object* v___x_11_; lean_object* v_cache_12_; lean_object* v_zetaDeltaFVarIds_13_; lean_object* v_postponed_14_; lean_object* v_diag_15_; lean_object* v___x_17_; uint8_t v_isShared_18_; uint8_t v_isSharedCheck_24_; 
v___x_6_ = lean_st_ref_get(v___y_2_);
v_mctx_7_ = lean_ctor_get(v___x_6_, 0);
lean_inc_ref(v_mctx_7_);
lean_dec(v___x_6_);
v___x_8_ = l_Lean_instantiateMVarsCore(v_mctx_7_, v_e_1_);
v_fst_9_ = lean_ctor_get(v___x_8_, 0);
lean_inc(v_fst_9_);
v_snd_10_ = lean_ctor_get(v___x_8_, 1);
lean_inc(v_snd_10_);
lean_dec_ref(v___x_8_);
v___x_11_ = lean_st_ref_take(v___y_2_);
v_cache_12_ = lean_ctor_get(v___x_11_, 1);
v_zetaDeltaFVarIds_13_ = lean_ctor_get(v___x_11_, 2);
v_postponed_14_ = lean_ctor_get(v___x_11_, 3);
v_diag_15_ = lean_ctor_get(v___x_11_, 4);
v_isSharedCheck_24_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_24_ == 0)
{
lean_object* v_unused_25_; 
v_unused_25_ = lean_ctor_get(v___x_11_, 0);
lean_dec(v_unused_25_);
v___x_17_ = v___x_11_;
v_isShared_18_ = v_isSharedCheck_24_;
goto v_resetjp_16_;
}
else
{
lean_inc(v_diag_15_);
lean_inc(v_postponed_14_);
lean_inc(v_zetaDeltaFVarIds_13_);
lean_inc(v_cache_12_);
lean_dec(v___x_11_);
v___x_17_ = lean_box(0);
v_isShared_18_ = v_isSharedCheck_24_;
goto v_resetjp_16_;
}
v_resetjp_16_:
{
lean_object* v___x_20_; 
if (v_isShared_18_ == 0)
{
lean_ctor_set(v___x_17_, 0, v_snd_10_);
v___x_20_ = v___x_17_;
goto v_reusejp_19_;
}
else
{
lean_object* v_reuseFailAlloc_23_; 
v_reuseFailAlloc_23_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_23_, 0, v_snd_10_);
lean_ctor_set(v_reuseFailAlloc_23_, 1, v_cache_12_);
lean_ctor_set(v_reuseFailAlloc_23_, 2, v_zetaDeltaFVarIds_13_);
lean_ctor_set(v_reuseFailAlloc_23_, 3, v_postponed_14_);
lean_ctor_set(v_reuseFailAlloc_23_, 4, v_diag_15_);
v___x_20_ = v_reuseFailAlloc_23_;
goto v_reusejp_19_;
}
v_reusejp_19_:
{
lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_21_ = lean_st_ref_set(v___y_2_, v___x_20_);
v___x_22_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_22_, 0, v_fst_9_);
return v___x_22_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_SortLocalDecls_visitExpr_spec__0___redArg___boxed(lean_object* v_e_26_, lean_object* v___y_27_, lean_object* v___y_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_Qq_Lean_instantiateMVars___at___00Qq_SortLocalDecls_visitExpr_spec__0___redArg(v_e_26_, v___y_27_);
lean_dec(v___y_27_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_SortLocalDecls_visitExpr(lean_object* v_e_30_, lean_object* v_a_31_, lean_object* v_a_32_, lean_object* v_a_33_, lean_object* v_a_34_, lean_object* v_a_35_, lean_object* v_a_36_){
_start:
{
lean_object* v_d_39_; lean_object* v_b_40_; lean_object* v___y_41_; lean_object* v___y_42_; lean_object* v___y_43_; lean_object* v___y_44_; lean_object* v___y_45_; lean_object* v___y_46_; 
switch(lean_obj_tag(v_e_30_))
{
case 11:
{
lean_object* v_struct_49_; 
v_struct_49_ = lean_ctor_get(v_e_30_, 2);
lean_inc_ref(v_struct_49_);
lean_dec_ref_known(v_e_30_, 3);
v_e_30_ = v_struct_49_;
goto _start;
}
case 7:
{
lean_object* v_binderType_51_; lean_object* v_body_52_; 
v_binderType_51_ = lean_ctor_get(v_e_30_, 1);
lean_inc_ref(v_binderType_51_);
v_body_52_ = lean_ctor_get(v_e_30_, 2);
lean_inc_ref(v_body_52_);
lean_dec_ref_known(v_e_30_, 3);
v_d_39_ = v_binderType_51_;
v_b_40_ = v_body_52_;
v___y_41_ = v_a_31_;
v___y_42_ = v_a_32_;
v___y_43_ = v_a_33_;
v___y_44_ = v_a_34_;
v___y_45_ = v_a_35_;
v___y_46_ = v_a_36_;
goto v___jp_38_;
}
case 6:
{
lean_object* v_binderType_53_; lean_object* v_body_54_; 
v_binderType_53_ = lean_ctor_get(v_e_30_, 1);
lean_inc_ref(v_binderType_53_);
v_body_54_ = lean_ctor_get(v_e_30_, 2);
lean_inc_ref(v_body_54_);
lean_dec_ref_known(v_e_30_, 3);
v_d_39_ = v_binderType_53_;
v_b_40_ = v_body_54_;
v___y_41_ = v_a_31_;
v___y_42_ = v_a_32_;
v___y_43_ = v_a_33_;
v___y_44_ = v_a_34_;
v___y_45_ = v_a_35_;
v___y_46_ = v_a_36_;
goto v___jp_38_;
}
case 8:
{
lean_object* v_type_55_; lean_object* v_value_56_; lean_object* v_body_57_; lean_object* v___x_58_; 
v_type_55_ = lean_ctor_get(v_e_30_, 1);
lean_inc_ref(v_type_55_);
v_value_56_ = lean_ctor_get(v_e_30_, 2);
lean_inc_ref(v_value_56_);
v_body_57_ = lean_ctor_get(v_e_30_, 3);
lean_inc_ref(v_body_57_);
lean_dec_ref_known(v_e_30_, 4);
v___x_58_ = lp_Qq_Qq_SortLocalDecls_visitExpr(v_type_55_, v_a_31_, v_a_32_, v_a_33_, v_a_34_, v_a_35_, v_a_36_);
if (lean_obj_tag(v___x_58_) == 0)
{
lean_object* v___x_59_; 
lean_dec_ref_known(v___x_58_, 1);
v___x_59_ = lp_Qq_Qq_SortLocalDecls_visitExpr(v_value_56_, v_a_31_, v_a_32_, v_a_33_, v_a_34_, v_a_35_, v_a_36_);
if (lean_obj_tag(v___x_59_) == 0)
{
lean_dec_ref_known(v___x_59_, 1);
v_e_30_ = v_body_57_;
goto _start;
}
else
{
lean_dec_ref(v_body_57_);
return v___x_59_;
}
}
else
{
lean_dec_ref(v_body_57_);
lean_dec_ref(v_value_56_);
return v___x_58_;
}
}
case 5:
{
lean_object* v_fn_61_; lean_object* v_arg_62_; lean_object* v___x_63_; 
v_fn_61_ = lean_ctor_get(v_e_30_, 0);
lean_inc_ref(v_fn_61_);
v_arg_62_ = lean_ctor_get(v_e_30_, 1);
lean_inc_ref(v_arg_62_);
lean_dec_ref_known(v_e_30_, 2);
v___x_63_ = lp_Qq_Qq_SortLocalDecls_visitExpr(v_fn_61_, v_a_31_, v_a_32_, v_a_33_, v_a_34_, v_a_35_, v_a_36_);
if (lean_obj_tag(v___x_63_) == 0)
{
lean_dec_ref_known(v___x_63_, 1);
v_e_30_ = v_arg_62_;
goto _start;
}
else
{
lean_dec_ref(v_arg_62_);
return v___x_63_;
}
}
case 10:
{
lean_object* v_expr_65_; 
v_expr_65_ = lean_ctor_get(v_e_30_, 1);
lean_inc_ref(v_expr_65_);
lean_dec_ref_known(v_e_30_, 2);
v_e_30_ = v_expr_65_;
goto _start;
}
case 2:
{
lean_object* v___x_67_; 
v___x_67_ = lp_Qq_Lean_instantiateMVars___at___00Qq_SortLocalDecls_visitExpr_spec__0___redArg(v_e_30_, v_a_34_);
if (lean_obj_tag(v___x_67_) == 0)
{
lean_object* v_a_68_; lean_object* v___x_70_; uint8_t v_isShared_71_; uint8_t v_isSharedCheck_78_; 
v_a_68_ = lean_ctor_get(v___x_67_, 0);
v_isSharedCheck_78_ = !lean_is_exclusive(v___x_67_);
if (v_isSharedCheck_78_ == 0)
{
v___x_70_ = v___x_67_;
v_isShared_71_ = v_isSharedCheck_78_;
goto v_resetjp_69_;
}
else
{
lean_inc(v_a_68_);
lean_dec(v___x_67_);
v___x_70_ = lean_box(0);
v_isShared_71_ = v_isSharedCheck_78_;
goto v_resetjp_69_;
}
v_resetjp_69_:
{
uint8_t v___x_72_; 
v___x_72_ = l_Lean_Expr_isMVar(v_a_68_);
if (v___x_72_ == 0)
{
lean_del_object(v___x_70_);
v_e_30_ = v_a_68_;
goto _start;
}
else
{
lean_object* v___x_74_; lean_object* v___x_76_; 
lean_dec(v_a_68_);
v___x_74_ = lean_box(0);
if (v_isShared_71_ == 0)
{
lean_ctor_set(v___x_70_, 0, v___x_74_);
v___x_76_ = v___x_70_;
goto v_reusejp_75_;
}
else
{
lean_object* v_reuseFailAlloc_77_; 
v_reuseFailAlloc_77_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_77_, 0, v___x_74_);
v___x_76_ = v_reuseFailAlloc_77_;
goto v_reusejp_75_;
}
v_reusejp_75_:
{
return v___x_76_;
}
}
}
}
else
{
lean_object* v_a_79_; lean_object* v___x_81_; uint8_t v_isShared_82_; uint8_t v_isSharedCheck_86_; 
v_a_79_ = lean_ctor_get(v___x_67_, 0);
v_isSharedCheck_86_ = !lean_is_exclusive(v___x_67_);
if (v_isSharedCheck_86_ == 0)
{
v___x_81_ = v___x_67_;
v_isShared_82_ = v_isSharedCheck_86_;
goto v_resetjp_80_;
}
else
{
lean_inc(v_a_79_);
lean_dec(v___x_67_);
v___x_81_ = lean_box(0);
v_isShared_82_ = v_isSharedCheck_86_;
goto v_resetjp_80_;
}
v_resetjp_80_:
{
lean_object* v___x_84_; 
if (v_isShared_82_ == 0)
{
v___x_84_ = v___x_81_;
goto v_reusejp_83_;
}
else
{
lean_object* v_reuseFailAlloc_85_; 
v_reuseFailAlloc_85_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_85_, 0, v_a_79_);
v___x_84_ = v_reuseFailAlloc_85_;
goto v_reusejp_83_;
}
v_reusejp_83_:
{
return v___x_84_;
}
}
}
}
case 1:
{
lean_object* v_fvarId_87_; lean_object* v___x_88_; 
v_fvarId_87_ = lean_ctor_get(v_e_30_, 0);
lean_inc(v_fvarId_87_);
lean_dec_ref_known(v_e_30_, 1);
v___x_88_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_a_31_, v_fvarId_87_);
lean_dec(v_fvarId_87_);
if (lean_obj_tag(v___x_88_) == 1)
{
lean_object* v_val_89_; lean_object* v___x_90_; 
v_val_89_ = lean_ctor_get(v___x_88_, 0);
lean_inc(v_val_89_);
lean_dec_ref_known(v___x_88_, 1);
v___x_90_ = lp_Qq_Qq_SortLocalDecls_visitLocalDecl(v_val_89_, v_a_31_, v_a_32_, v_a_33_, v_a_34_, v_a_35_, v_a_36_);
return v___x_90_;
}
else
{
lean_object* v___x_91_; lean_object* v___x_92_; 
lean_dec(v___x_88_);
v___x_91_ = lean_box(0);
v___x_92_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_92_, 0, v___x_91_);
return v___x_92_;
}
}
default: 
{
lean_object* v___x_93_; lean_object* v___x_94_; 
lean_dec_ref(v_e_30_);
v___x_93_ = lean_box(0);
v___x_94_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_94_, 0, v___x_93_);
return v___x_94_;
}
}
v___jp_38_:
{
lean_object* v___x_47_; 
v___x_47_ = lp_Qq_Qq_SortLocalDecls_visitExpr(v_d_39_, v___y_41_, v___y_42_, v___y_43_, v___y_44_, v___y_45_, v___y_46_);
if (lean_obj_tag(v___x_47_) == 0)
{
lean_dec_ref_known(v___x_47_, 1);
v_e_30_ = v_b_40_;
v_a_31_ = v___y_41_;
v_a_32_ = v___y_42_;
v_a_33_ = v___y_43_;
v_a_34_ = v___y_44_;
v_a_35_ = v___y_45_;
v_a_36_ = v___y_46_;
goto _start;
}
else
{
lean_dec_ref(v_b_40_);
return v___x_47_;
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_SortLocalDecls_visitLocalDecl(lean_object* v_localDecl_95_, lean_object* v_a_96_, lean_object* v_a_97_, lean_object* v_a_98_, lean_object* v_a_99_, lean_object* v_a_100_, lean_object* v_a_101_){
_start:
{
lean_object* v___y_104_; lean_object* v___x_119_; lean_object* v_visited_120_; lean_object* v___x_121_; uint8_t v___x_122_; 
v___x_119_ = lean_st_ref_get(v_a_97_);
v_visited_120_ = lean_ctor_get(v___x_119_, 0);
lean_inc(v_visited_120_);
lean_dec(v___x_119_);
v___x_121_ = l_Lean_LocalDecl_fvarId(v_localDecl_95_);
v___x_122_ = l_Lean_NameSet_contains(v_visited_120_, v___x_121_);
lean_dec(v_visited_120_);
if (v___x_122_ == 0)
{
lean_object* v___x_123_; lean_object* v_visited_124_; lean_object* v_result_125_; lean_object* v___x_127_; uint8_t v_isShared_128_; uint8_t v_isSharedCheck_139_; 
v___x_123_ = lean_st_ref_take(v_a_97_);
v_visited_124_ = lean_ctor_get(v___x_123_, 0);
v_result_125_ = lean_ctor_get(v___x_123_, 1);
v_isSharedCheck_139_ = !lean_is_exclusive(v___x_123_);
if (v_isSharedCheck_139_ == 0)
{
v___x_127_ = v___x_123_;
v_isShared_128_ = v_isSharedCheck_139_;
goto v_resetjp_126_;
}
else
{
lean_inc(v_result_125_);
lean_inc(v_visited_124_);
lean_dec(v___x_123_);
v___x_127_ = lean_box(0);
v_isShared_128_ = v_isSharedCheck_139_;
goto v_resetjp_126_;
}
v_resetjp_126_:
{
lean_object* v___x_129_; lean_object* v___x_131_; 
v___x_129_ = l_Lean_NameSet_insert(v_visited_124_, v___x_121_);
if (v_isShared_128_ == 0)
{
lean_ctor_set(v___x_127_, 0, v___x_129_);
v___x_131_ = v___x_127_;
goto v_reusejp_130_;
}
else
{
lean_object* v_reuseFailAlloc_138_; 
v_reuseFailAlloc_138_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_138_, 0, v___x_129_);
lean_ctor_set(v_reuseFailAlloc_138_, 1, v_result_125_);
v___x_131_ = v_reuseFailAlloc_138_;
goto v_reusejp_130_;
}
v_reusejp_130_:
{
lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; 
v___x_132_ = lean_st_ref_set(v_a_97_, v___x_131_);
v___x_133_ = l_Lean_LocalDecl_type(v_localDecl_95_);
v___x_134_ = lp_Qq_Qq_SortLocalDecls_visitExpr(v___x_133_, v_a_96_, v_a_97_, v_a_98_, v_a_99_, v_a_100_, v_a_101_);
if (lean_obj_tag(v___x_134_) == 0)
{
lean_object* v___x_135_; 
lean_dec_ref_known(v___x_134_, 1);
v___x_135_ = l_Lean_LocalDecl_value_x3f(v_localDecl_95_, v___x_122_);
if (lean_obj_tag(v___x_135_) == 1)
{
lean_object* v_val_136_; lean_object* v___x_137_; 
v_val_136_ = lean_ctor_get(v___x_135_, 0);
lean_inc(v_val_136_);
lean_dec_ref_known(v___x_135_, 1);
v___x_137_ = lp_Qq_Qq_SortLocalDecls_visitExpr(v_val_136_, v_a_96_, v_a_97_, v_a_98_, v_a_99_, v_a_100_, v_a_101_);
if (lean_obj_tag(v___x_137_) == 0)
{
lean_dec_ref_known(v___x_137_, 1);
v___y_104_ = v_a_97_;
goto v___jp_103_;
}
else
{
lean_dec_ref(v_localDecl_95_);
return v___x_137_;
}
}
else
{
lean_dec(v___x_135_);
v___y_104_ = v_a_97_;
goto v___jp_103_;
}
}
else
{
lean_dec_ref(v_localDecl_95_);
return v___x_134_;
}
}
}
}
else
{
lean_object* v___x_140_; lean_object* v___x_141_; 
lean_dec(v___x_121_);
lean_dec_ref(v_localDecl_95_);
v___x_140_ = lean_box(0);
v___x_141_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_141_, 0, v___x_140_);
return v___x_141_;
}
v___jp_103_:
{
lean_object* v___x_105_; lean_object* v_visited_106_; lean_object* v_result_107_; lean_object* v___x_109_; uint8_t v_isShared_110_; uint8_t v_isSharedCheck_118_; 
v___x_105_ = lean_st_ref_take(v___y_104_);
v_visited_106_ = lean_ctor_get(v___x_105_, 0);
v_result_107_ = lean_ctor_get(v___x_105_, 1);
v_isSharedCheck_118_ = !lean_is_exclusive(v___x_105_);
if (v_isSharedCheck_118_ == 0)
{
v___x_109_ = v___x_105_;
v_isShared_110_ = v_isSharedCheck_118_;
goto v_resetjp_108_;
}
else
{
lean_inc(v_result_107_);
lean_inc(v_visited_106_);
lean_dec(v___x_105_);
v___x_109_ = lean_box(0);
v_isShared_110_ = v_isSharedCheck_118_;
goto v_resetjp_108_;
}
v_resetjp_108_:
{
lean_object* v___x_111_; lean_object* v___x_113_; 
v___x_111_ = lean_array_push(v_result_107_, v_localDecl_95_);
if (v_isShared_110_ == 0)
{
lean_ctor_set(v___x_109_, 1, v___x_111_);
v___x_113_ = v___x_109_;
goto v_reusejp_112_;
}
else
{
lean_object* v_reuseFailAlloc_117_; 
v_reuseFailAlloc_117_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_117_, 0, v_visited_106_);
lean_ctor_set(v_reuseFailAlloc_117_, 1, v___x_111_);
v___x_113_ = v_reuseFailAlloc_117_;
goto v_reusejp_112_;
}
v_reusejp_112_:
{
lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; 
v___x_114_ = lean_st_ref_set(v___y_104_, v___x_113_);
v___x_115_ = lean_box(0);
v___x_116_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_116_, 0, v___x_115_);
return v___x_116_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_SortLocalDecls_visitLocalDecl___boxed(lean_object* v_localDecl_142_, lean_object* v_a_143_, lean_object* v_a_144_, lean_object* v_a_145_, lean_object* v_a_146_, lean_object* v_a_147_, lean_object* v_a_148_, lean_object* v_a_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_Qq_Qq_SortLocalDecls_visitLocalDecl(v_localDecl_142_, v_a_143_, v_a_144_, v_a_145_, v_a_146_, v_a_147_, v_a_148_);
lean_dec(v_a_148_);
lean_dec_ref(v_a_147_);
lean_dec(v_a_146_);
lean_dec_ref(v_a_145_);
lean_dec(v_a_144_);
lean_dec(v_a_143_);
return v_res_150_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_SortLocalDecls_visitExpr___boxed(lean_object* v_e_151_, lean_object* v_a_152_, lean_object* v_a_153_, lean_object* v_a_154_, lean_object* v_a_155_, lean_object* v_a_156_, lean_object* v_a_157_, lean_object* v_a_158_){
_start:
{
lean_object* v_res_159_; 
v_res_159_ = lp_Qq_Qq_SortLocalDecls_visitExpr(v_e_151_, v_a_152_, v_a_153_, v_a_154_, v_a_155_, v_a_156_, v_a_157_);
lean_dec(v_a_157_);
lean_dec_ref(v_a_156_);
lean_dec(v_a_155_);
lean_dec_ref(v_a_154_);
lean_dec(v_a_153_);
lean_dec(v_a_152_);
return v_res_159_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_SortLocalDecls_visitExpr_spec__0(lean_object* v_e_160_, lean_object* v___y_161_, lean_object* v___y_162_, lean_object* v___y_163_, lean_object* v___y_164_, lean_object* v___y_165_, lean_object* v___y_166_){
_start:
{
lean_object* v___x_168_; 
v___x_168_ = lp_Qq_Lean_instantiateMVars___at___00Qq_SortLocalDecls_visitExpr_spec__0___redArg(v_e_160_, v___y_164_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_SortLocalDecls_visitExpr_spec__0___boxed(lean_object* v_e_169_, lean_object* v___y_170_, lean_object* v___y_171_, lean_object* v___y_172_, lean_object* v___y_173_, lean_object* v___y_174_, lean_object* v___y_175_, lean_object* v___y_176_){
_start:
{
lean_object* v_res_177_; 
v_res_177_ = lp_Qq_Lean_instantiateMVars___at___00Qq_SortLocalDecls_visitExpr_spec__0(v_e_169_, v___y_170_, v___y_171_, v___y_172_, v___y_173_, v___y_174_, v___y_175_);
lean_dec(v___y_175_);
lean_dec_ref(v___y_174_);
lean_dec(v___y_173_);
lean_dec_ref(v___y_172_);
lean_dec(v___y_171_);
lean_dec(v___y_170_);
return v_res_177_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__1(lean_object* v_as_178_, size_t v_i_179_, size_t v_stop_180_, lean_object* v_b_181_){
_start:
{
uint8_t v___x_182_; 
v___x_182_ = lean_usize_dec_eq(v_i_179_, v_stop_180_);
if (v___x_182_ == 0)
{
lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; size_t v___x_186_; size_t v___x_187_; 
v___x_183_ = lean_array_uget_borrowed(v_as_178_, v_i_179_);
v___x_184_ = l_Lean_LocalDecl_fvarId(v___x_183_);
lean_inc(v___x_183_);
v___x_185_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v___x_184_, v___x_183_, v_b_181_);
v___x_186_ = ((size_t)1ULL);
v___x_187_ = lean_usize_add(v_i_179_, v___x_186_);
v_i_179_ = v___x_187_;
v_b_181_ = v___x_185_;
goto _start;
}
else
{
return v_b_181_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__1___boxed(lean_object* v_as_189_, lean_object* v_i_190_, lean_object* v_stop_191_, lean_object* v_b_192_){
_start:
{
size_t v_i_boxed_193_; size_t v_stop_boxed_194_; lean_object* v_res_195_; 
v_i_boxed_193_ = lean_unbox_usize(v_i_190_);
lean_dec(v_i_190_);
v_stop_boxed_194_ = lean_unbox_usize(v_stop_191_);
lean_dec(v_stop_191_);
v_res_195_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__1(v_as_189_, v_i_boxed_193_, v_stop_boxed_194_, v_b_192_);
lean_dec_ref(v_as_189_);
return v_res_195_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__0(lean_object* v_as_196_, size_t v_i_197_, size_t v_stop_198_, lean_object* v_b_199_, lean_object* v___y_200_, lean_object* v___y_201_, lean_object* v___y_202_, lean_object* v___y_203_, lean_object* v___y_204_, lean_object* v___y_205_){
_start:
{
uint8_t v___x_207_; 
v___x_207_ = lean_usize_dec_eq(v_i_197_, v_stop_198_);
if (v___x_207_ == 0)
{
lean_object* v___x_208_; lean_object* v___x_209_; 
v___x_208_ = lean_array_uget_borrowed(v_as_196_, v_i_197_);
lean_inc(v___x_208_);
v___x_209_ = lp_Qq_Qq_SortLocalDecls_visitLocalDecl(v___x_208_, v___y_200_, v___y_201_, v___y_202_, v___y_203_, v___y_204_, v___y_205_);
if (lean_obj_tag(v___x_209_) == 0)
{
lean_object* v_a_210_; size_t v___x_211_; size_t v___x_212_; 
v_a_210_ = lean_ctor_get(v___x_209_, 0);
lean_inc(v_a_210_);
lean_dec_ref_known(v___x_209_, 1);
v___x_211_ = ((size_t)1ULL);
v___x_212_ = lean_usize_add(v_i_197_, v___x_211_);
v_i_197_ = v___x_212_;
v_b_199_ = v_a_210_;
goto _start;
}
else
{
return v___x_209_;
}
}
else
{
lean_object* v___x_214_; 
v___x_214_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_214_, 0, v_b_199_);
return v___x_214_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__0___boxed(lean_object* v_as_215_, lean_object* v_i_216_, lean_object* v_stop_217_, lean_object* v_b_218_, lean_object* v___y_219_, lean_object* v___y_220_, lean_object* v___y_221_, lean_object* v___y_222_, lean_object* v___y_223_, lean_object* v___y_224_, lean_object* v___y_225_){
_start:
{
size_t v_i_boxed_226_; size_t v_stop_boxed_227_; lean_object* v_res_228_; 
v_i_boxed_226_ = lean_unbox_usize(v_i_216_);
lean_dec(v_i_216_);
v_stop_boxed_227_ = lean_unbox_usize(v_stop_217_);
lean_dec(v_stop_217_);
v_res_228_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__0(v_as_215_, v_i_boxed_226_, v_stop_boxed_227_, v_b_218_, v___y_219_, v___y_220_, v___y_221_, v___y_222_, v___y_223_, v___y_224_);
lean_dec(v___y_224_);
lean_dec_ref(v___y_223_);
lean_dec(v___y_222_);
lean_dec_ref(v___y_221_);
lean_dec(v___y_220_);
lean_dec(v___y_219_);
lean_dec_ref(v_as_215_);
return v_res_228_;
}
}
static lean_object* _init_lp_Qq_Qq_sortLocalDecls___closed__1(void){
_start:
{
lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; 
v___x_231_ = ((lean_object*)(lp_Qq_Qq_sortLocalDecls___closed__0));
v___x_232_ = l_Lean_NameSet_empty;
v___x_233_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_233_, 0, v___x_232_);
lean_ctor_set(v___x_233_, 1, v___x_231_);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_sortLocalDecls(lean_object* v_localDecls_234_, lean_object* v_a_235_, lean_object* v_a_236_, lean_object* v_a_237_, lean_object* v_a_238_){
_start:
{
lean_object* v___y_241_; lean_object* v___y_247_; lean_object* v___y_248_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___y_260_; lean_object* v___x_272_; uint8_t v___x_273_; 
v___x_257_ = lean_unsigned_to_nat(0u);
v___x_258_ = lean_array_get_size(v_localDecls_234_);
v___x_272_ = lean_box(1);
v___x_273_ = lean_nat_dec_lt(v___x_257_, v___x_258_);
if (v___x_273_ == 0)
{
v___y_260_ = v___x_272_;
goto v___jp_259_;
}
else
{
uint8_t v___x_274_; 
v___x_274_ = lean_nat_dec_le(v___x_258_, v___x_258_);
if (v___x_274_ == 0)
{
if (v___x_273_ == 0)
{
v___y_260_ = v___x_272_;
goto v___jp_259_;
}
else
{
size_t v___x_275_; size_t v___x_276_; lean_object* v___x_277_; 
v___x_275_ = ((size_t)0ULL);
v___x_276_ = lean_usize_of_nat(v___x_258_);
v___x_277_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__1(v_localDecls_234_, v___x_275_, v___x_276_, v___x_272_);
v___y_260_ = v___x_277_;
goto v___jp_259_;
}
}
else
{
size_t v___x_278_; size_t v___x_279_; lean_object* v___x_280_; 
v___x_278_ = ((size_t)0ULL);
v___x_279_ = lean_usize_of_nat(v___x_258_);
v___x_280_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__1(v_localDecls_234_, v___x_278_, v___x_279_, v___x_272_);
v___y_260_ = v___x_280_;
goto v___jp_259_;
}
}
v___jp_240_:
{
lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v_result_244_; lean_object* v___x_245_; 
v___x_242_ = lean_st_ref_get(v___y_241_);
v___x_243_ = lean_st_ref_get(v___y_241_);
lean_dec(v___y_241_);
lean_dec(v___x_243_);
v_result_244_ = lean_ctor_get(v___x_242_, 1);
lean_inc_ref(v_result_244_);
lean_dec(v___x_242_);
v___x_245_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_245_, 0, v_result_244_);
return v___x_245_;
}
v___jp_246_:
{
if (lean_obj_tag(v___y_248_) == 0)
{
lean_dec_ref_known(v___y_248_, 1);
v___y_241_ = v___y_247_;
goto v___jp_240_;
}
else
{
lean_object* v_a_249_; lean_object* v___x_251_; uint8_t v_isShared_252_; uint8_t v_isSharedCheck_256_; 
lean_dec(v___y_247_);
v_a_249_ = lean_ctor_get(v___y_248_, 0);
v_isSharedCheck_256_ = !lean_is_exclusive(v___y_248_);
if (v_isSharedCheck_256_ == 0)
{
v___x_251_ = v___y_248_;
v_isShared_252_ = v_isSharedCheck_256_;
goto v_resetjp_250_;
}
else
{
lean_inc(v_a_249_);
lean_dec(v___y_248_);
v___x_251_ = lean_box(0);
v_isShared_252_ = v_isSharedCheck_256_;
goto v_resetjp_250_;
}
v_resetjp_250_:
{
lean_object* v___x_254_; 
if (v_isShared_252_ == 0)
{
v___x_254_ = v___x_251_;
goto v_reusejp_253_;
}
else
{
lean_object* v_reuseFailAlloc_255_; 
v_reuseFailAlloc_255_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_255_, 0, v_a_249_);
v___x_254_ = v_reuseFailAlloc_255_;
goto v_reusejp_253_;
}
v_reusejp_253_:
{
return v___x_254_;
}
}
}
}
v___jp_259_:
{
lean_object* v___x_261_; lean_object* v___x_262_; uint8_t v___x_263_; 
v___x_261_ = lean_obj_once(&lp_Qq_Qq_sortLocalDecls___closed__1, &lp_Qq_Qq_sortLocalDecls___closed__1_once, _init_lp_Qq_Qq_sortLocalDecls___closed__1);
v___x_262_ = lean_st_mk_ref(v___x_261_);
v___x_263_ = lean_nat_dec_lt(v___x_257_, v___x_258_);
if (v___x_263_ == 0)
{
lean_dec(v___y_260_);
v___y_241_ = v___x_262_;
goto v___jp_240_;
}
else
{
lean_object* v___x_264_; uint8_t v___x_265_; 
v___x_264_ = lean_box(0);
v___x_265_ = lean_nat_dec_le(v___x_258_, v___x_258_);
if (v___x_265_ == 0)
{
if (v___x_263_ == 0)
{
lean_dec(v___y_260_);
v___y_241_ = v___x_262_;
goto v___jp_240_;
}
else
{
size_t v___x_266_; size_t v___x_267_; lean_object* v___x_268_; 
v___x_266_ = ((size_t)0ULL);
v___x_267_ = lean_usize_of_nat(v___x_258_);
v___x_268_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__0(v_localDecls_234_, v___x_266_, v___x_267_, v___x_264_, v___y_260_, v___x_262_, v_a_235_, v_a_236_, v_a_237_, v_a_238_);
lean_dec(v___y_260_);
v___y_247_ = v___x_262_;
v___y_248_ = v___x_268_;
goto v___jp_246_;
}
}
else
{
size_t v___x_269_; size_t v___x_270_; lean_object* v___x_271_; 
v___x_269_ = ((size_t)0ULL);
v___x_270_ = lean_usize_of_nat(v___x_258_);
v___x_271_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__0(v_localDecls_234_, v___x_269_, v___x_270_, v___x_264_, v___y_260_, v___x_262_, v_a_235_, v_a_236_, v_a_237_, v_a_238_);
lean_dec(v___y_260_);
v___y_247_ = v___x_262_;
v___y_248_ = v___x_271_;
goto v___jp_246_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_sortLocalDecls___boxed(lean_object* v_localDecls_281_, lean_object* v_a_282_, lean_object* v_a_283_, lean_object* v_a_284_, lean_object* v_a_285_, lean_object* v_a_286_){
_start:
{
lean_object* v_res_287_; 
v_res_287_ = lp_Qq_Qq_sortLocalDecls(v_localDecls_281_, v_a_282_, v_a_283_, v_a_284_, v_a_285_);
lean_dec(v_a_285_);
lean_dec_ref(v_a_284_);
lean_dec(v_a_283_);
lean_dec_ref(v_a_282_);
lean_dec_ref(v_localDecls_281_);
return v_res_287_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_Qq_Qq_SortLocalDecls(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_Qq_Qq_SortLocalDecls(uint8_t builtin) {
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
lean_object* initialize_Lean_Meta_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_Qq_Qq_SortLocalDecls(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_SortLocalDecls(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_Qq_Qq_SortLocalDecls(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_Qq_Qq_SortLocalDecls(builtin);
}
#ifdef __cplusplus
}
#endif
