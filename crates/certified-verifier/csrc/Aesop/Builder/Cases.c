// Lean compiler output
// Module: Aesop.Builder.Cases
// Imports: public import Init public meta import Init public import Aesop.Builder.Basic public import Aesop.RuleTac.Cases import Batteries.Lean.Expr
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
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Meta_instBEqTransparencyMode_beq(uint8_t, uint8_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lp_aesop_Aesop_CasesPattern_toExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_mkDiscrTreePath(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lp_aesop_Aesop_IndexingMode_hypsMatchingConst(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
uint8_t lp_batteries_Lean_Expr_isAppOf_x27(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* lean_expr_dbg_to_string(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* lp_aesop_Aesop_mkCtorNames(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_PhaseSpec_toRule(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lp_aesop_Aesop_elabInductiveRuleIdent(uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_PhaseSpec_phase(lean_object*);
uint8_t lp_aesop_Aesop_instBEqPhaseName_beq(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesPattern_check_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesPattern_check_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesPattern_check_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesPattern_check_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_CasesPattern_check_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_CasesPattern_check_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_CasesPattern_check_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_CasesPattern_check_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_CasesPattern_check___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "expected pattern '"};
static const lean_object* lp_aesop_Aesop_CasesPattern_check___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_CasesPattern_check___lam__0___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_CasesPattern_check___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_CasesPattern_check___lam__0___closed__1;
static const lean_string_object lp_aesop_Aesop_CasesPattern_check___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "' ("};
static const lean_object* lp_aesop_Aesop_CasesPattern_check___lam__0___closed__2 = (const lean_object*)&lp_aesop_Aesop_CasesPattern_check___lam__0___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_CasesPattern_check___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_CasesPattern_check___lam__0___closed__3;
static const lean_string_object lp_aesop_Aesop_CasesPattern_check___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = ") to be an application of '"};
static const lean_object* lp_aesop_Aesop_CasesPattern_check___lam__0___closed__4 = (const lean_object*)&lp_aesop_Aesop_CasesPattern_check___lam__0___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_CasesPattern_check___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_CasesPattern_check___lam__0___closed__5;
static const lean_string_object lp_aesop_Aesop_CasesPattern_check___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_aesop_Aesop_CasesPattern_check___lam__0___closed__6 = (const lean_object*)&lp_aesop_Aesop_CasesPattern_check___lam__0___closed__6_value;
static lean_once_cell_t lp_aesop_Aesop_CasesPattern_check___lam__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_CasesPattern_check___lam__0___closed__7;
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesPattern_check___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesPattern_check___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesPattern_check(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesPattern_check___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_CasesPattern_check_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_CasesPattern_check_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesPattern_toIndexingMode___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesPattern_toIndexingMode___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesPattern_toIndexingMode(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesPattern_toIndexingMode___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleBuilderOptions_casesTransparency(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilderOptions_casesTransparency___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleBuilderOptions_casesIndexTransparency(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilderOptions_casesIndexTransparency___boxed(lean_object*);
static const lean_array_object lp_aesop_Aesop_RuleBuilderOptions_casesPatterns___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_RuleBuilderOptions_casesPatterns___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleBuilderOptions_casesPatterns___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilderOptions_casesPatterns(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilderOptions_casesPatterns___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_mkCasesTarget(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleBuilder_getCasesIndexingMode_spec__0(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleBuilder_getCasesIndexingMode_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getCasesIndexingMode(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getCasesIndexingMode___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleBuilder_casesCore_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleBuilder_casesCore_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_casesCore(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_casesCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_cases_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_cases_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_RuleBuilder_cases___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 62, .m_capacity = 62, .m_length = 61, .m_data = "aesop: cases builder cannot currently be used for norm rules."};
static const lean_object* lp_aesop_Aesop_RuleBuilder_cases___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleBuilder_cases___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_RuleBuilder_cases___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleBuilder_cases___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_cases(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_cases___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_cases_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_cases_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesPattern_check_spec__1___redArg(lean_object* v_x_1_, lean_object* v___y_2_, lean_object* v___y_3_, lean_object* v___y_4_, lean_object* v___y_5_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = l_Lean_Meta_saveState___redArg(v___y_3_, v___y_5_);
if (lean_obj_tag(v___x_7_) == 0)
{
lean_object* v_a_8_; lean_object* v_r_9_; 
v_a_8_ = lean_ctor_get(v___x_7_, 0);
lean_inc(v_a_8_);
lean_dec_ref_known(v___x_7_, 1);
lean_inc(v___y_5_);
lean_inc_ref(v___y_4_);
lean_inc(v___y_3_);
lean_inc_ref(v___y_2_);
v_r_9_ = lean_apply_5(v_x_1_, v___y_2_, v___y_3_, v___y_4_, v___y_5_, lean_box(0));
if (lean_obj_tag(v_r_9_) == 0)
{
lean_object* v_a_10_; lean_object* v___x_11_; 
v_a_10_ = lean_ctor_get(v_r_9_, 0);
lean_inc(v_a_10_);
lean_dec_ref_known(v_r_9_, 1);
v___x_11_ = l_Lean_Meta_SavedState_restore___redArg(v_a_8_, v___y_3_, v___y_5_);
lean_dec(v_a_8_);
if (lean_obj_tag(v___x_11_) == 0)
{
lean_object* v___x_13_; uint8_t v_isShared_14_; uint8_t v_isSharedCheck_18_; 
v_isSharedCheck_18_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_18_ == 0)
{
lean_object* v_unused_19_; 
v_unused_19_ = lean_ctor_get(v___x_11_, 0);
lean_dec(v_unused_19_);
v___x_13_ = v___x_11_;
v_isShared_14_ = v_isSharedCheck_18_;
goto v_resetjp_12_;
}
else
{
lean_dec(v___x_11_);
v___x_13_ = lean_box(0);
v_isShared_14_ = v_isSharedCheck_18_;
goto v_resetjp_12_;
}
v_resetjp_12_:
{
lean_object* v___x_16_; 
if (v_isShared_14_ == 0)
{
lean_ctor_set(v___x_13_, 0, v_a_10_);
v___x_16_ = v___x_13_;
goto v_reusejp_15_;
}
else
{
lean_object* v_reuseFailAlloc_17_; 
v_reuseFailAlloc_17_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_17_, 0, v_a_10_);
v___x_16_ = v_reuseFailAlloc_17_;
goto v_reusejp_15_;
}
v_reusejp_15_:
{
return v___x_16_;
}
}
}
else
{
lean_object* v_a_20_; lean_object* v___x_22_; uint8_t v_isShared_23_; uint8_t v_isSharedCheck_27_; 
lean_dec(v_a_10_);
v_a_20_ = lean_ctor_get(v___x_11_, 0);
v_isSharedCheck_27_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_27_ == 0)
{
v___x_22_ = v___x_11_;
v_isShared_23_ = v_isSharedCheck_27_;
goto v_resetjp_21_;
}
else
{
lean_inc(v_a_20_);
lean_dec(v___x_11_);
v___x_22_ = lean_box(0);
v_isShared_23_ = v_isSharedCheck_27_;
goto v_resetjp_21_;
}
v_resetjp_21_:
{
lean_object* v___x_25_; 
if (v_isShared_23_ == 0)
{
v___x_25_ = v___x_22_;
goto v_reusejp_24_;
}
else
{
lean_object* v_reuseFailAlloc_26_; 
v_reuseFailAlloc_26_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_26_, 0, v_a_20_);
v___x_25_ = v_reuseFailAlloc_26_;
goto v_reusejp_24_;
}
v_reusejp_24_:
{
return v___x_25_;
}
}
}
}
else
{
lean_object* v_a_28_; lean_object* v___x_29_; 
v_a_28_ = lean_ctor_get(v_r_9_, 0);
lean_inc(v_a_28_);
lean_dec_ref_known(v_r_9_, 1);
v___x_29_ = l_Lean_Meta_SavedState_restore___redArg(v_a_8_, v___y_3_, v___y_5_);
lean_dec(v_a_8_);
if (lean_obj_tag(v___x_29_) == 0)
{
lean_object* v___x_31_; uint8_t v_isShared_32_; uint8_t v_isSharedCheck_36_; 
v_isSharedCheck_36_ = !lean_is_exclusive(v___x_29_);
if (v_isSharedCheck_36_ == 0)
{
lean_object* v_unused_37_; 
v_unused_37_ = lean_ctor_get(v___x_29_, 0);
lean_dec(v_unused_37_);
v___x_31_ = v___x_29_;
v_isShared_32_ = v_isSharedCheck_36_;
goto v_resetjp_30_;
}
else
{
lean_dec(v___x_29_);
v___x_31_ = lean_box(0);
v_isShared_32_ = v_isSharedCheck_36_;
goto v_resetjp_30_;
}
v_resetjp_30_:
{
lean_object* v___x_34_; 
if (v_isShared_32_ == 0)
{
lean_ctor_set_tag(v___x_31_, 1);
lean_ctor_set(v___x_31_, 0, v_a_28_);
v___x_34_ = v___x_31_;
goto v_reusejp_33_;
}
else
{
lean_object* v_reuseFailAlloc_35_; 
v_reuseFailAlloc_35_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_35_, 0, v_a_28_);
v___x_34_ = v_reuseFailAlloc_35_;
goto v_reusejp_33_;
}
v_reusejp_33_:
{
return v___x_34_;
}
}
}
else
{
lean_object* v_a_38_; lean_object* v___x_40_; uint8_t v_isShared_41_; uint8_t v_isSharedCheck_45_; 
lean_dec(v_a_28_);
v_a_38_ = lean_ctor_get(v___x_29_, 0);
v_isSharedCheck_45_ = !lean_is_exclusive(v___x_29_);
if (v_isSharedCheck_45_ == 0)
{
v___x_40_ = v___x_29_;
v_isShared_41_ = v_isSharedCheck_45_;
goto v_resetjp_39_;
}
else
{
lean_inc(v_a_38_);
lean_dec(v___x_29_);
v___x_40_ = lean_box(0);
v_isShared_41_ = v_isSharedCheck_45_;
goto v_resetjp_39_;
}
v_resetjp_39_:
{
lean_object* v___x_43_; 
if (v_isShared_41_ == 0)
{
v___x_43_ = v___x_40_;
goto v_reusejp_42_;
}
else
{
lean_object* v_reuseFailAlloc_44_; 
v_reuseFailAlloc_44_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_44_, 0, v_a_38_);
v___x_43_ = v_reuseFailAlloc_44_;
goto v_reusejp_42_;
}
v_reusejp_42_:
{
return v___x_43_;
}
}
}
}
}
else
{
lean_object* v_a_46_; lean_object* v___x_48_; uint8_t v_isShared_49_; uint8_t v_isSharedCheck_53_; 
lean_dec_ref(v_x_1_);
v_a_46_ = lean_ctor_get(v___x_7_, 0);
v_isSharedCheck_53_ = !lean_is_exclusive(v___x_7_);
if (v_isSharedCheck_53_ == 0)
{
v___x_48_ = v___x_7_;
v_isShared_49_ = v_isSharedCheck_53_;
goto v_resetjp_47_;
}
else
{
lean_inc(v_a_46_);
lean_dec(v___x_7_);
v___x_48_ = lean_box(0);
v_isShared_49_ = v_isSharedCheck_53_;
goto v_resetjp_47_;
}
v_resetjp_47_:
{
lean_object* v___x_51_; 
if (v_isShared_49_ == 0)
{
v___x_51_ = v___x_48_;
goto v_reusejp_50_;
}
else
{
lean_object* v_reuseFailAlloc_52_; 
v_reuseFailAlloc_52_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_52_, 0, v_a_46_);
v___x_51_ = v_reuseFailAlloc_52_;
goto v_reusejp_50_;
}
v_reusejp_50_:
{
return v___x_51_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesPattern_check_spec__1___redArg___boxed(lean_object* v_x_54_, lean_object* v___y_55_, lean_object* v___y_56_, lean_object* v___y_57_, lean_object* v___y_58_, lean_object* v___y_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesPattern_check_spec__1___redArg(v_x_54_, v___y_55_, v___y_56_, v___y_57_, v___y_58_);
lean_dec(v___y_58_);
lean_dec_ref(v___y_57_);
lean_dec(v___y_56_);
lean_dec_ref(v___y_55_);
return v_res_60_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesPattern_check_spec__1(lean_object* v_00_u03b1_61_, lean_object* v_x_62_, lean_object* v___y_63_, lean_object* v___y_64_, lean_object* v___y_65_, lean_object* v___y_66_){
_start:
{
lean_object* v___x_68_; 
v___x_68_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesPattern_check_spec__1___redArg(v_x_62_, v___y_63_, v___y_64_, v___y_65_, v___y_66_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesPattern_check_spec__1___boxed(lean_object* v_00_u03b1_69_, lean_object* v_x_70_, lean_object* v___y_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_, lean_object* v___y_75_){
_start:
{
lean_object* v_res_76_; 
v_res_76_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesPattern_check_spec__1(v_00_u03b1_69_, v_x_70_, v___y_71_, v___y_72_, v___y_73_, v___y_74_);
lean_dec(v___y_74_);
lean_dec_ref(v___y_73_);
lean_dec(v___y_72_);
lean_dec_ref(v___y_71_);
return v_res_76_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_CasesPattern_check_spec__0_spec__0(lean_object* v_msgData_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_){
_start:
{
lean_object* v___x_83_; lean_object* v_env_84_; lean_object* v___x_85_; lean_object* v_mctx_86_; lean_object* v_lctx_87_; lean_object* v_options_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; 
v___x_83_ = lean_st_ref_get(v___y_81_);
v_env_84_ = lean_ctor_get(v___x_83_, 0);
lean_inc_ref(v_env_84_);
lean_dec(v___x_83_);
v___x_85_ = lean_st_ref_get(v___y_79_);
v_mctx_86_ = lean_ctor_get(v___x_85_, 0);
lean_inc_ref(v_mctx_86_);
lean_dec(v___x_85_);
v_lctx_87_ = lean_ctor_get(v___y_78_, 2);
v_options_88_ = lean_ctor_get(v___y_80_, 2);
lean_inc_ref(v_options_88_);
lean_inc_ref(v_lctx_87_);
v___x_89_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_89_, 0, v_env_84_);
lean_ctor_set(v___x_89_, 1, v_mctx_86_);
lean_ctor_set(v___x_89_, 2, v_lctx_87_);
lean_ctor_set(v___x_89_, 3, v_options_88_);
v___x_90_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_90_, 0, v___x_89_);
lean_ctor_set(v___x_90_, 1, v_msgData_77_);
v___x_91_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_91_, 0, v___x_90_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_CasesPattern_check_spec__0_spec__0___boxed(lean_object* v_msgData_92_, lean_object* v___y_93_, lean_object* v___y_94_, lean_object* v___y_95_, lean_object* v___y_96_, lean_object* v___y_97_){
_start:
{
lean_object* v_res_98_; 
v_res_98_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_CasesPattern_check_spec__0_spec__0(v_msgData_92_, v___y_93_, v___y_94_, v___y_95_, v___y_96_);
lean_dec(v___y_96_);
lean_dec_ref(v___y_95_);
lean_dec(v___y_94_);
lean_dec_ref(v___y_93_);
return v_res_98_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_CasesPattern_check_spec__0___redArg(lean_object* v_msg_99_, lean_object* v___y_100_, lean_object* v___y_101_, lean_object* v___y_102_, lean_object* v___y_103_){
_start:
{
lean_object* v_ref_105_; lean_object* v___x_106_; lean_object* v_a_107_; lean_object* v___x_109_; uint8_t v_isShared_110_; uint8_t v_isSharedCheck_115_; 
v_ref_105_ = lean_ctor_get(v___y_102_, 5);
v___x_106_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_CasesPattern_check_spec__0_spec__0(v_msg_99_, v___y_100_, v___y_101_, v___y_102_, v___y_103_);
v_a_107_ = lean_ctor_get(v___x_106_, 0);
v_isSharedCheck_115_ = !lean_is_exclusive(v___x_106_);
if (v_isSharedCheck_115_ == 0)
{
v___x_109_ = v___x_106_;
v_isShared_110_ = v_isSharedCheck_115_;
goto v_resetjp_108_;
}
else
{
lean_inc(v_a_107_);
lean_dec(v___x_106_);
v___x_109_ = lean_box(0);
v_isShared_110_ = v_isSharedCheck_115_;
goto v_resetjp_108_;
}
v_resetjp_108_:
{
lean_object* v___x_111_; lean_object* v___x_113_; 
lean_inc(v_ref_105_);
v___x_111_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_111_, 0, v_ref_105_);
lean_ctor_set(v___x_111_, 1, v_a_107_);
if (v_isShared_110_ == 0)
{
lean_ctor_set_tag(v___x_109_, 1);
lean_ctor_set(v___x_109_, 0, v___x_111_);
v___x_113_ = v___x_109_;
goto v_reusejp_112_;
}
else
{
lean_object* v_reuseFailAlloc_114_; 
v_reuseFailAlloc_114_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_114_, 0, v___x_111_);
v___x_113_ = v_reuseFailAlloc_114_;
goto v_reusejp_112_;
}
v_reusejp_112_:
{
return v___x_113_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_CasesPattern_check_spec__0___redArg___boxed(lean_object* v_msg_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_, lean_object* v___y_121_){
_start:
{
lean_object* v_res_122_; 
v_res_122_ = lp_aesop_Lean_throwError___at___00Aesop_CasesPattern_check_spec__0___redArg(v_msg_116_, v___y_117_, v___y_118_, v___y_119_, v___y_120_);
lean_dec(v___y_120_);
lean_dec_ref(v___y_119_);
lean_dec(v___y_118_);
lean_dec_ref(v___y_117_);
return v_res_122_;
}
}
static lean_object* _init_lp_aesop_Aesop_CasesPattern_check___lam__0___closed__1(void){
_start:
{
lean_object* v___x_124_; lean_object* v___x_125_; 
v___x_124_ = ((lean_object*)(lp_aesop_Aesop_CasesPattern_check___lam__0___closed__0));
v___x_125_ = l_Lean_stringToMessageData(v___x_124_);
return v___x_125_;
}
}
static lean_object* _init_lp_aesop_Aesop_CasesPattern_check___lam__0___closed__3(void){
_start:
{
lean_object* v___x_127_; lean_object* v___x_128_; 
v___x_127_ = ((lean_object*)(lp_aesop_Aesop_CasesPattern_check___lam__0___closed__2));
v___x_128_ = l_Lean_stringToMessageData(v___x_127_);
return v___x_128_;
}
}
static lean_object* _init_lp_aesop_Aesop_CasesPattern_check___lam__0___closed__5(void){
_start:
{
lean_object* v___x_130_; lean_object* v___x_131_; 
v___x_130_ = ((lean_object*)(lp_aesop_Aesop_CasesPattern_check___lam__0___closed__4));
v___x_131_ = l_Lean_stringToMessageData(v___x_130_);
return v___x_131_;
}
}
static lean_object* _init_lp_aesop_Aesop_CasesPattern_check___lam__0___closed__7(void){
_start:
{
lean_object* v___x_133_; lean_object* v___x_134_; 
v___x_133_ = ((lean_object*)(lp_aesop_Aesop_CasesPattern_check___lam__0___closed__6));
v___x_134_ = l_Lean_stringToMessageData(v___x_133_);
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesPattern_check___lam__0(lean_object* v_p_135_, lean_object* v_decl_136_, lean_object* v___y_137_, lean_object* v___y_138_, lean_object* v___y_139_, lean_object* v___y_140_){
_start:
{
lean_object* v___x_142_; 
v___x_142_ = lp_aesop_Aesop_CasesPattern_toExpr(v_p_135_, v___y_137_, v___y_138_, v___y_139_, v___y_140_);
if (lean_obj_tag(v___x_142_) == 0)
{
lean_object* v_a_143_; lean_object* v___x_145_; uint8_t v_isShared_146_; uint8_t v_isSharedCheck_167_; 
v_a_143_ = lean_ctor_get(v___x_142_, 0);
v_isSharedCheck_167_ = !lean_is_exclusive(v___x_142_);
if (v_isSharedCheck_167_ == 0)
{
v___x_145_ = v___x_142_;
v_isShared_146_ = v_isSharedCheck_167_;
goto v_resetjp_144_;
}
else
{
lean_inc(v_a_143_);
lean_dec(v___x_142_);
v___x_145_ = lean_box(0);
v_isShared_146_ = v_isSharedCheck_167_;
goto v_resetjp_144_;
}
v_resetjp_144_:
{
uint8_t v___x_147_; 
v___x_147_ = lp_batteries_Lean_Expr_isAppOf_x27(v_a_143_, v_decl_136_);
if (v___x_147_ == 0)
{
lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; 
lean_del_object(v___x_145_);
v___x_148_ = lean_obj_once(&lp_aesop_Aesop_CasesPattern_check___lam__0___closed__1, &lp_aesop_Aesop_CasesPattern_check___lam__0___closed__1_once, _init_lp_aesop_Aesop_CasesPattern_check___lam__0___closed__1);
lean_inc(v_a_143_);
v___x_149_ = l_Lean_MessageData_ofExpr(v_a_143_);
v___x_150_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_150_, 0, v___x_148_);
lean_ctor_set(v___x_150_, 1, v___x_149_);
v___x_151_ = lean_obj_once(&lp_aesop_Aesop_CasesPattern_check___lam__0___closed__3, &lp_aesop_Aesop_CasesPattern_check___lam__0___closed__3_once, _init_lp_aesop_Aesop_CasesPattern_check___lam__0___closed__3);
v___x_152_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_152_, 0, v___x_150_);
lean_ctor_set(v___x_152_, 1, v___x_151_);
v___x_153_ = lean_expr_dbg_to_string(v_a_143_);
lean_dec(v_a_143_);
v___x_154_ = l_Lean_stringToMessageData(v___x_153_);
v___x_155_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_155_, 0, v___x_152_);
lean_ctor_set(v___x_155_, 1, v___x_154_);
v___x_156_ = lean_obj_once(&lp_aesop_Aesop_CasesPattern_check___lam__0___closed__5, &lp_aesop_Aesop_CasesPattern_check___lam__0___closed__5_once, _init_lp_aesop_Aesop_CasesPattern_check___lam__0___closed__5);
v___x_157_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_157_, 0, v___x_155_);
lean_ctor_set(v___x_157_, 1, v___x_156_);
v___x_158_ = l_Lean_MessageData_ofName(v_decl_136_);
v___x_159_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_159_, 0, v___x_157_);
lean_ctor_set(v___x_159_, 1, v___x_158_);
v___x_160_ = lean_obj_once(&lp_aesop_Aesop_CasesPattern_check___lam__0___closed__7, &lp_aesop_Aesop_CasesPattern_check___lam__0___closed__7_once, _init_lp_aesop_Aesop_CasesPattern_check___lam__0___closed__7);
v___x_161_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_161_, 0, v___x_159_);
lean_ctor_set(v___x_161_, 1, v___x_160_);
v___x_162_ = lp_aesop_Lean_throwError___at___00Aesop_CasesPattern_check_spec__0___redArg(v___x_161_, v___y_137_, v___y_138_, v___y_139_, v___y_140_);
return v___x_162_;
}
else
{
lean_object* v___x_163_; lean_object* v___x_165_; 
lean_dec(v_a_143_);
lean_dec(v_decl_136_);
v___x_163_ = lean_box(0);
if (v_isShared_146_ == 0)
{
lean_ctor_set(v___x_145_, 0, v___x_163_);
v___x_165_ = v___x_145_;
goto v_reusejp_164_;
}
else
{
lean_object* v_reuseFailAlloc_166_; 
v_reuseFailAlloc_166_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_166_, 0, v___x_163_);
v___x_165_ = v_reuseFailAlloc_166_;
goto v_reusejp_164_;
}
v_reusejp_164_:
{
return v___x_165_;
}
}
}
}
else
{
lean_object* v_a_168_; lean_object* v___x_170_; uint8_t v_isShared_171_; uint8_t v_isSharedCheck_175_; 
lean_dec(v_decl_136_);
v_a_168_ = lean_ctor_get(v___x_142_, 0);
v_isSharedCheck_175_ = !lean_is_exclusive(v___x_142_);
if (v_isSharedCheck_175_ == 0)
{
v___x_170_ = v___x_142_;
v_isShared_171_ = v_isSharedCheck_175_;
goto v_resetjp_169_;
}
else
{
lean_inc(v_a_168_);
lean_dec(v___x_142_);
v___x_170_ = lean_box(0);
v_isShared_171_ = v_isSharedCheck_175_;
goto v_resetjp_169_;
}
v_resetjp_169_:
{
lean_object* v___x_173_; 
if (v_isShared_171_ == 0)
{
v___x_173_ = v___x_170_;
goto v_reusejp_172_;
}
else
{
lean_object* v_reuseFailAlloc_174_; 
v_reuseFailAlloc_174_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_174_, 0, v_a_168_);
v___x_173_ = v_reuseFailAlloc_174_;
goto v_reusejp_172_;
}
v_reusejp_172_:
{
return v___x_173_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesPattern_check___lam__0___boxed(lean_object* v_p_176_, lean_object* v_decl_177_, lean_object* v___y_178_, lean_object* v___y_179_, lean_object* v___y_180_, lean_object* v___y_181_, lean_object* v___y_182_){
_start:
{
lean_object* v_res_183_; 
v_res_183_ = lp_aesop_Aesop_CasesPattern_check___lam__0(v_p_176_, v_decl_177_, v___y_178_, v___y_179_, v___y_180_, v___y_181_);
lean_dec(v___y_181_);
lean_dec_ref(v___y_180_);
lean_dec(v___y_179_);
lean_dec_ref(v___y_178_);
return v_res_183_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesPattern_check(lean_object* v_decl_184_, lean_object* v_p_185_, lean_object* v_a_186_, lean_object* v_a_187_, lean_object* v_a_188_, lean_object* v_a_189_){
_start:
{
lean_object* v___f_191_; lean_object* v___x_192_; 
v___f_191_ = lean_alloc_closure((void*)(lp_aesop_Aesop_CasesPattern_check___lam__0___boxed), 7, 2);
lean_closure_set(v___f_191_, 0, v_p_185_);
lean_closure_set(v___f_191_, 1, v_decl_184_);
v___x_192_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesPattern_check_spec__1___redArg(v___f_191_, v_a_186_, v_a_187_, v_a_188_, v_a_189_);
return v___x_192_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesPattern_check___boxed(lean_object* v_decl_193_, lean_object* v_p_194_, lean_object* v_a_195_, lean_object* v_a_196_, lean_object* v_a_197_, lean_object* v_a_198_, lean_object* v_a_199_){
_start:
{
lean_object* v_res_200_; 
v_res_200_ = lp_aesop_Aesop_CasesPattern_check(v_decl_193_, v_p_194_, v_a_195_, v_a_196_, v_a_197_, v_a_198_);
lean_dec(v_a_198_);
lean_dec_ref(v_a_197_);
lean_dec(v_a_196_);
lean_dec_ref(v_a_195_);
return v_res_200_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_CasesPattern_check_spec__0(lean_object* v_00_u03b1_201_, lean_object* v_msg_202_, lean_object* v___y_203_, lean_object* v___y_204_, lean_object* v___y_205_, lean_object* v___y_206_){
_start:
{
lean_object* v___x_208_; 
v___x_208_ = lp_aesop_Lean_throwError___at___00Aesop_CasesPattern_check_spec__0___redArg(v_msg_202_, v___y_203_, v___y_204_, v___y_205_, v___y_206_);
return v___x_208_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_CasesPattern_check_spec__0___boxed(lean_object* v_00_u03b1_209_, lean_object* v_msg_210_, lean_object* v___y_211_, lean_object* v___y_212_, lean_object* v___y_213_, lean_object* v___y_214_, lean_object* v___y_215_){
_start:
{
lean_object* v_res_216_; 
v_res_216_ = lp_aesop_Lean_throwError___at___00Aesop_CasesPattern_check_spec__0(v_00_u03b1_209_, v_msg_210_, v___y_211_, v___y_212_, v___y_213_, v___y_214_);
lean_dec(v___y_214_);
lean_dec_ref(v___y_213_);
lean_dec(v___y_212_);
lean_dec_ref(v___y_211_);
return v_res_216_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesPattern_toIndexingMode___lam__0(lean_object* v_p_217_, lean_object* v___y_218_, lean_object* v___y_219_, lean_object* v___y_220_, lean_object* v___y_221_){
_start:
{
lean_object* v___x_223_; 
v___x_223_ = lp_aesop_Aesop_CasesPattern_toExpr(v_p_217_, v___y_218_, v___y_219_, v___y_220_, v___y_221_);
if (lean_obj_tag(v___x_223_) == 0)
{
lean_object* v_a_224_; lean_object* v___x_225_; 
v_a_224_ = lean_ctor_get(v___x_223_, 0);
lean_inc(v_a_224_);
lean_dec_ref_known(v___x_223_, 1);
v___x_225_ = lp_aesop_Aesop_mkDiscrTreePath(v_a_224_, v___y_218_, v___y_219_, v___y_220_, v___y_221_);
if (lean_obj_tag(v___x_225_) == 0)
{
lean_object* v_a_226_; lean_object* v___x_228_; uint8_t v_isShared_229_; uint8_t v_isSharedCheck_234_; 
v_a_226_ = lean_ctor_get(v___x_225_, 0);
v_isSharedCheck_234_ = !lean_is_exclusive(v___x_225_);
if (v_isSharedCheck_234_ == 0)
{
v___x_228_ = v___x_225_;
v_isShared_229_ = v_isSharedCheck_234_;
goto v_resetjp_227_;
}
else
{
lean_inc(v_a_226_);
lean_dec(v___x_225_);
v___x_228_ = lean_box(0);
v_isShared_229_ = v_isSharedCheck_234_;
goto v_resetjp_227_;
}
v_resetjp_227_:
{
lean_object* v___x_230_; lean_object* v___x_232_; 
v___x_230_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_230_, 0, v_a_226_);
if (v_isShared_229_ == 0)
{
lean_ctor_set(v___x_228_, 0, v___x_230_);
v___x_232_ = v___x_228_;
goto v_reusejp_231_;
}
else
{
lean_object* v_reuseFailAlloc_233_; 
v_reuseFailAlloc_233_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_233_, 0, v___x_230_);
v___x_232_ = v_reuseFailAlloc_233_;
goto v_reusejp_231_;
}
v_reusejp_231_:
{
return v___x_232_;
}
}
}
else
{
lean_object* v_a_235_; lean_object* v___x_237_; uint8_t v_isShared_238_; uint8_t v_isSharedCheck_242_; 
v_a_235_ = lean_ctor_get(v___x_225_, 0);
v_isSharedCheck_242_ = !lean_is_exclusive(v___x_225_);
if (v_isSharedCheck_242_ == 0)
{
v___x_237_ = v___x_225_;
v_isShared_238_ = v_isSharedCheck_242_;
goto v_resetjp_236_;
}
else
{
lean_inc(v_a_235_);
lean_dec(v___x_225_);
v___x_237_ = lean_box(0);
v_isShared_238_ = v_isSharedCheck_242_;
goto v_resetjp_236_;
}
v_resetjp_236_:
{
lean_object* v___x_240_; 
if (v_isShared_238_ == 0)
{
v___x_240_ = v___x_237_;
goto v_reusejp_239_;
}
else
{
lean_object* v_reuseFailAlloc_241_; 
v_reuseFailAlloc_241_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_241_, 0, v_a_235_);
v___x_240_ = v_reuseFailAlloc_241_;
goto v_reusejp_239_;
}
v_reusejp_239_:
{
return v___x_240_;
}
}
}
}
else
{
lean_object* v_a_243_; lean_object* v___x_245_; uint8_t v_isShared_246_; uint8_t v_isSharedCheck_250_; 
v_a_243_ = lean_ctor_get(v___x_223_, 0);
v_isSharedCheck_250_ = !lean_is_exclusive(v___x_223_);
if (v_isSharedCheck_250_ == 0)
{
v___x_245_ = v___x_223_;
v_isShared_246_ = v_isSharedCheck_250_;
goto v_resetjp_244_;
}
else
{
lean_inc(v_a_243_);
lean_dec(v___x_223_);
v___x_245_ = lean_box(0);
v_isShared_246_ = v_isSharedCheck_250_;
goto v_resetjp_244_;
}
v_resetjp_244_:
{
lean_object* v___x_248_; 
if (v_isShared_246_ == 0)
{
v___x_248_ = v___x_245_;
goto v_reusejp_247_;
}
else
{
lean_object* v_reuseFailAlloc_249_; 
v_reuseFailAlloc_249_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_249_, 0, v_a_243_);
v___x_248_ = v_reuseFailAlloc_249_;
goto v_reusejp_247_;
}
v_reusejp_247_:
{
return v___x_248_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesPattern_toIndexingMode___lam__0___boxed(lean_object* v_p_251_, lean_object* v___y_252_, lean_object* v___y_253_, lean_object* v___y_254_, lean_object* v___y_255_, lean_object* v___y_256_){
_start:
{
lean_object* v_res_257_; 
v_res_257_ = lp_aesop_Aesop_CasesPattern_toIndexingMode___lam__0(v_p_251_, v___y_252_, v___y_253_, v___y_254_, v___y_255_);
lean_dec(v___y_255_);
lean_dec_ref(v___y_254_);
lean_dec(v___y_253_);
lean_dec_ref(v___y_252_);
return v_res_257_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesPattern_toIndexingMode(lean_object* v_p_258_, lean_object* v_a_259_, lean_object* v_a_260_, lean_object* v_a_261_, lean_object* v_a_262_){
_start:
{
lean_object* v___f_264_; lean_object* v___x_265_; 
v___f_264_ = lean_alloc_closure((void*)(lp_aesop_Aesop_CasesPattern_toIndexingMode___lam__0___boxed), 6, 1);
lean_closure_set(v___f_264_, 0, v_p_258_);
v___x_265_ = lp_aesop_Lean_withoutModifyingState___at___00Aesop_CasesPattern_check_spec__1___redArg(v___f_264_, v_a_259_, v_a_260_, v_a_261_, v_a_262_);
return v___x_265_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CasesPattern_toIndexingMode___boxed(lean_object* v_p_266_, lean_object* v_a_267_, lean_object* v_a_268_, lean_object* v_a_269_, lean_object* v_a_270_, lean_object* v_a_271_){
_start:
{
lean_object* v_res_272_; 
v_res_272_ = lp_aesop_Aesop_CasesPattern_toIndexingMode(v_p_266_, v_a_267_, v_a_268_, v_a_269_, v_a_270_);
lean_dec(v_a_270_);
lean_dec_ref(v_a_269_);
lean_dec(v_a_268_);
lean_dec_ref(v_a_267_);
return v_res_272_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleBuilderOptions_casesTransparency(lean_object* v_opts_273_){
_start:
{
lean_object* v_transparency_x3f_274_; 
v_transparency_x3f_274_ = lean_ctor_get(v_opts_273_, 4);
if (lean_obj_tag(v_transparency_x3f_274_) == 0)
{
uint8_t v___x_275_; 
v___x_275_ = 2;
return v___x_275_;
}
else
{
lean_object* v_val_276_; uint8_t v___x_277_; 
v_val_276_ = lean_ctor_get(v_transparency_x3f_274_, 0);
v___x_277_ = lean_unbox(v_val_276_);
return v___x_277_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilderOptions_casesTransparency___boxed(lean_object* v_opts_278_){
_start:
{
uint8_t v_res_279_; lean_object* v_r_280_; 
v_res_279_ = lp_aesop_Aesop_RuleBuilderOptions_casesTransparency(v_opts_278_);
lean_dec_ref(v_opts_278_);
v_r_280_ = lean_box(v_res_279_);
return v_r_280_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleBuilderOptions_casesIndexTransparency(lean_object* v_opts_281_){
_start:
{
lean_object* v_indexTransparency_x3f_282_; 
v_indexTransparency_x3f_282_ = lean_ctor_get(v_opts_281_, 5);
if (lean_obj_tag(v_indexTransparency_x3f_282_) == 0)
{
uint8_t v___x_283_; 
v___x_283_ = 2;
return v___x_283_;
}
else
{
lean_object* v_val_284_; uint8_t v___x_285_; 
v_val_284_ = lean_ctor_get(v_indexTransparency_x3f_282_, 0);
v___x_285_ = lean_unbox(v_val_284_);
return v___x_285_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilderOptions_casesIndexTransparency___boxed(lean_object* v_opts_286_){
_start:
{
uint8_t v_res_287_; lean_object* v_r_288_; 
v_res_287_ = lp_aesop_Aesop_RuleBuilderOptions_casesIndexTransparency(v_opts_286_);
lean_dec_ref(v_opts_286_);
v_r_288_ = lean_box(v_res_287_);
return v_r_288_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilderOptions_casesPatterns(lean_object* v_opts_291_){
_start:
{
lean_object* v_casesPatterns_x3f_292_; 
v_casesPatterns_x3f_292_ = lean_ctor_get(v_opts_291_, 2);
if (lean_obj_tag(v_casesPatterns_x3f_292_) == 0)
{
lean_object* v___x_293_; 
v___x_293_ = ((lean_object*)(lp_aesop_Aesop_RuleBuilderOptions_casesPatterns___closed__0));
return v___x_293_;
}
else
{
lean_object* v_val_294_; 
v_val_294_ = lean_ctor_get(v_casesPatterns_x3f_292_, 0);
lean_inc(v_val_294_);
return v_val_294_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilderOptions_casesPatterns___boxed(lean_object* v_opts_295_){
_start:
{
lean_object* v_res_296_; 
v_res_296_ = lp_aesop_Aesop_RuleBuilderOptions_casesPatterns(v_opts_295_);
lean_dec_ref(v_opts_295_);
return v_res_296_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_mkCasesTarget(lean_object* v_decl_297_, lean_object* v_casesPatterns_298_){
_start:
{
lean_object* v___x_299_; lean_object* v___x_300_; uint8_t v___x_301_; 
v___x_299_ = lean_array_get_size(v_casesPatterns_298_);
v___x_300_ = lean_unsigned_to_nat(0u);
v___x_301_ = lean_nat_dec_eq(v___x_299_, v___x_300_);
if (v___x_301_ == 0)
{
lean_object* v___x_302_; 
lean_dec(v_decl_297_);
v___x_302_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_302_, 0, v_casesPatterns_298_);
return v___x_302_;
}
else
{
lean_object* v___x_303_; 
lean_dec_ref(v_casesPatterns_298_);
v___x_303_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_303_, 0, v_decl_297_);
return v___x_303_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleBuilder_getCasesIndexingMode_spec__0(size_t v_sz_304_, size_t v_i_305_, lean_object* v_bs_306_, lean_object* v___y_307_, lean_object* v___y_308_, lean_object* v___y_309_, lean_object* v___y_310_){
_start:
{
uint8_t v___x_312_; 
v___x_312_ = lean_usize_dec_lt(v_i_305_, v_sz_304_);
if (v___x_312_ == 0)
{
lean_object* v___x_313_; 
v___x_313_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_313_, 0, v_bs_306_);
return v___x_313_;
}
else
{
lean_object* v_v_314_; lean_object* v___x_315_; 
v_v_314_ = lean_array_uget_borrowed(v_bs_306_, v_i_305_);
lean_inc(v_v_314_);
v___x_315_ = lp_aesop_Aesop_CasesPattern_toIndexingMode(v_v_314_, v___y_307_, v___y_308_, v___y_309_, v___y_310_);
if (lean_obj_tag(v___x_315_) == 0)
{
lean_object* v_a_316_; lean_object* v___x_317_; lean_object* v_bs_x27_318_; size_t v___x_319_; size_t v___x_320_; lean_object* v___x_321_; 
v_a_316_ = lean_ctor_get(v___x_315_, 0);
lean_inc(v_a_316_);
lean_dec_ref_known(v___x_315_, 1);
v___x_317_ = lean_unsigned_to_nat(0u);
v_bs_x27_318_ = lean_array_uset(v_bs_306_, v_i_305_, v___x_317_);
v___x_319_ = ((size_t)1ULL);
v___x_320_ = lean_usize_add(v_i_305_, v___x_319_);
v___x_321_ = lean_array_uset(v_bs_x27_318_, v_i_305_, v_a_316_);
v_i_305_ = v___x_320_;
v_bs_306_ = v___x_321_;
goto _start;
}
else
{
lean_object* v_a_323_; lean_object* v___x_325_; uint8_t v_isShared_326_; uint8_t v_isSharedCheck_330_; 
lean_dec_ref(v_bs_306_);
v_a_323_ = lean_ctor_get(v___x_315_, 0);
v_isSharedCheck_330_ = !lean_is_exclusive(v___x_315_);
if (v_isSharedCheck_330_ == 0)
{
v___x_325_ = v___x_315_;
v_isShared_326_ = v_isSharedCheck_330_;
goto v_resetjp_324_;
}
else
{
lean_inc(v_a_323_);
lean_dec(v___x_315_);
v___x_325_ = lean_box(0);
v_isShared_326_ = v_isSharedCheck_330_;
goto v_resetjp_324_;
}
v_resetjp_324_:
{
lean_object* v___x_328_; 
if (v_isShared_326_ == 0)
{
v___x_328_ = v___x_325_;
goto v_reusejp_327_;
}
else
{
lean_object* v_reuseFailAlloc_329_; 
v_reuseFailAlloc_329_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_329_, 0, v_a_323_);
v___x_328_ = v_reuseFailAlloc_329_;
goto v_reusejp_327_;
}
v_reusejp_327_:
{
return v___x_328_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleBuilder_getCasesIndexingMode_spec__0___boxed(lean_object* v_sz_331_, lean_object* v_i_332_, lean_object* v_bs_333_, lean_object* v___y_334_, lean_object* v___y_335_, lean_object* v___y_336_, lean_object* v___y_337_, lean_object* v___y_338_){
_start:
{
size_t v_sz_boxed_339_; size_t v_i_boxed_340_; lean_object* v_res_341_; 
v_sz_boxed_339_ = lean_unbox_usize(v_sz_331_);
lean_dec(v_sz_331_);
v_i_boxed_340_ = lean_unbox_usize(v_i_332_);
lean_dec(v_i_332_);
v_res_341_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleBuilder_getCasesIndexingMode_spec__0(v_sz_boxed_339_, v_i_boxed_340_, v_bs_333_, v___y_334_, v___y_335_, v___y_336_, v___y_337_);
lean_dec(v___y_337_);
lean_dec_ref(v___y_336_);
lean_dec(v___y_335_);
lean_dec_ref(v___y_334_);
return v_res_341_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getCasesIndexingMode(lean_object* v_decl_342_, uint8_t v_indexMd_343_, lean_object* v_casesPatterns_344_, lean_object* v_a_345_, lean_object* v_a_346_, lean_object* v_a_347_, lean_object* v_a_348_){
_start:
{
uint8_t v___x_350_; uint8_t v___x_351_; 
v___x_350_ = 2;
v___x_351_ = l_Lean_Meta_instBEqTransparencyMode_beq(v_indexMd_343_, v___x_350_);
if (v___x_351_ == 0)
{
lean_object* v___x_352_; lean_object* v___x_353_; 
lean_dec_ref(v_casesPatterns_344_);
lean_dec(v_decl_342_);
v___x_352_ = lean_box(0);
v___x_353_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_353_, 0, v___x_352_);
return v___x_353_;
}
else
{
lean_object* v___x_354_; lean_object* v___x_355_; uint8_t v___x_356_; 
v___x_354_ = lean_array_get_size(v_casesPatterns_344_);
v___x_355_ = lean_unsigned_to_nat(0u);
v___x_356_ = lean_nat_dec_eq(v___x_354_, v___x_355_);
if (v___x_356_ == 0)
{
size_t v_sz_357_; size_t v___x_358_; lean_object* v___x_359_; 
lean_dec(v_decl_342_);
v_sz_357_ = lean_array_size(v_casesPatterns_344_);
v___x_358_ = ((size_t)0ULL);
v___x_359_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleBuilder_getCasesIndexingMode_spec__0(v_sz_357_, v___x_358_, v_casesPatterns_344_, v_a_345_, v_a_346_, v_a_347_, v_a_348_);
if (lean_obj_tag(v___x_359_) == 0)
{
lean_object* v_a_360_; lean_object* v___x_362_; uint8_t v_isShared_363_; uint8_t v_isSharedCheck_368_; 
v_a_360_ = lean_ctor_get(v___x_359_, 0);
v_isSharedCheck_368_ = !lean_is_exclusive(v___x_359_);
if (v_isSharedCheck_368_ == 0)
{
v___x_362_ = v___x_359_;
v_isShared_363_ = v_isSharedCheck_368_;
goto v_resetjp_361_;
}
else
{
lean_inc(v_a_360_);
lean_dec(v___x_359_);
v___x_362_ = lean_box(0);
v_isShared_363_ = v_isSharedCheck_368_;
goto v_resetjp_361_;
}
v_resetjp_361_:
{
lean_object* v___x_364_; lean_object* v___x_366_; 
v___x_364_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_364_, 0, v_a_360_);
if (v_isShared_363_ == 0)
{
lean_ctor_set(v___x_362_, 0, v___x_364_);
v___x_366_ = v___x_362_;
goto v_reusejp_365_;
}
else
{
lean_object* v_reuseFailAlloc_367_; 
v_reuseFailAlloc_367_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_367_, 0, v___x_364_);
v___x_366_ = v_reuseFailAlloc_367_;
goto v_reusejp_365_;
}
v_reusejp_365_:
{
return v___x_366_;
}
}
}
else
{
lean_object* v_a_369_; lean_object* v___x_371_; uint8_t v_isShared_372_; uint8_t v_isSharedCheck_376_; 
v_a_369_ = lean_ctor_get(v___x_359_, 0);
v_isSharedCheck_376_ = !lean_is_exclusive(v___x_359_);
if (v_isSharedCheck_376_ == 0)
{
v___x_371_ = v___x_359_;
v_isShared_372_ = v_isSharedCheck_376_;
goto v_resetjp_370_;
}
else
{
lean_inc(v_a_369_);
lean_dec(v___x_359_);
v___x_371_ = lean_box(0);
v_isShared_372_ = v_isSharedCheck_376_;
goto v_resetjp_370_;
}
v_resetjp_370_:
{
lean_object* v___x_374_; 
if (v_isShared_372_ == 0)
{
v___x_374_ = v___x_371_;
goto v_reusejp_373_;
}
else
{
lean_object* v_reuseFailAlloc_375_; 
v_reuseFailAlloc_375_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_375_, 0, v_a_369_);
v___x_374_ = v_reuseFailAlloc_375_;
goto v_reusejp_373_;
}
v_reusejp_373_:
{
return v___x_374_;
}
}
}
}
else
{
lean_object* v___x_377_; 
lean_dec_ref(v_casesPatterns_344_);
v___x_377_ = lp_aesop_Aesop_IndexingMode_hypsMatchingConst(v_decl_342_, v_a_345_, v_a_346_, v_a_347_, v_a_348_);
return v___x_377_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_getCasesIndexingMode___boxed(lean_object* v_decl_378_, lean_object* v_indexMd_379_, lean_object* v_casesPatterns_380_, lean_object* v_a_381_, lean_object* v_a_382_, lean_object* v_a_383_, lean_object* v_a_384_, lean_object* v_a_385_){
_start:
{
uint8_t v_indexMd_boxed_386_; lean_object* v_res_387_; 
v_indexMd_boxed_386_ = lean_unbox(v_indexMd_379_);
v_res_387_ = lp_aesop_Aesop_RuleBuilder_getCasesIndexingMode(v_decl_378_, v_indexMd_boxed_386_, v_casesPatterns_380_, v_a_381_, v_a_382_, v_a_383_, v_a_384_);
lean_dec(v_a_384_);
lean_dec_ref(v_a_383_);
lean_dec(v_a_382_);
lean_dec_ref(v_a_381_);
return v_res_387_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleBuilder_casesCore_spec__0(lean_object* v_decl_388_, lean_object* v_as_389_, size_t v_i_390_, size_t v_stop_391_, lean_object* v_b_392_, lean_object* v___y_393_, lean_object* v___y_394_, lean_object* v___y_395_, lean_object* v___y_396_){
_start:
{
uint8_t v___x_398_; 
v___x_398_ = lean_usize_dec_eq(v_i_390_, v_stop_391_);
if (v___x_398_ == 0)
{
lean_object* v___x_399_; lean_object* v___x_400_; 
v___x_399_ = lean_array_uget_borrowed(v_as_389_, v_i_390_);
lean_inc(v___x_399_);
lean_inc(v_decl_388_);
v___x_400_ = lp_aesop_Aesop_CasesPattern_check(v_decl_388_, v___x_399_, v___y_393_, v___y_394_, v___y_395_, v___y_396_);
if (lean_obj_tag(v___x_400_) == 0)
{
lean_object* v_a_401_; size_t v___x_402_; size_t v___x_403_; 
v_a_401_ = lean_ctor_get(v___x_400_, 0);
lean_inc(v_a_401_);
lean_dec_ref_known(v___x_400_, 1);
v___x_402_ = ((size_t)1ULL);
v___x_403_ = lean_usize_add(v_i_390_, v___x_402_);
v_i_390_ = v___x_403_;
v_b_392_ = v_a_401_;
goto _start;
}
else
{
lean_dec(v_decl_388_);
return v___x_400_;
}
}
else
{
lean_object* v___x_405_; 
lean_dec(v_decl_388_);
v___x_405_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_405_, 0, v_b_392_);
return v___x_405_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleBuilder_casesCore_spec__0___boxed(lean_object* v_decl_406_, lean_object* v_as_407_, lean_object* v_i_408_, lean_object* v_stop_409_, lean_object* v_b_410_, lean_object* v___y_411_, lean_object* v___y_412_, lean_object* v___y_413_, lean_object* v___y_414_, lean_object* v___y_415_){
_start:
{
size_t v_i_boxed_416_; size_t v_stop_boxed_417_; lean_object* v_res_418_; 
v_i_boxed_416_ = lean_unbox_usize(v_i_408_);
lean_dec(v_i_408_);
v_stop_boxed_417_ = lean_unbox_usize(v_stop_409_);
lean_dec(v_stop_409_);
v_res_418_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleBuilder_casesCore_spec__0(v_decl_406_, v_as_407_, v_i_boxed_416_, v_stop_boxed_417_, v_b_410_, v___y_411_, v___y_412_, v___y_413_, v___y_414_);
lean_dec(v___y_414_);
lean_dec_ref(v___y_413_);
lean_dec(v___y_412_);
lean_dec_ref(v___y_411_);
lean_dec_ref(v_as_407_);
return v_res_418_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_casesCore(lean_object* v_decl_419_, lean_object* v_info_420_, lean_object* v_pats_421_, lean_object* v_imode_x3f_422_, uint8_t v_md_423_, uint8_t v_indexMd_424_, lean_object* v_phase_425_, lean_object* v_a_426_, lean_object* v_a_427_, lean_object* v_a_428_, lean_object* v_a_429_){
_start:
{
lean_object* v_a_432_; lean_object* v___y_472_; lean_object* v___x_481_; lean_object* v___x_482_; uint8_t v___x_483_; 
v___x_481_ = lean_unsigned_to_nat(0u);
v___x_482_ = lean_array_get_size(v_pats_421_);
v___x_483_ = lean_nat_dec_lt(v___x_481_, v___x_482_);
if (v___x_483_ == 0)
{
goto v___jp_459_;
}
else
{
lean_object* v___x_484_; uint8_t v___x_485_; 
v___x_484_ = lean_box(0);
v___x_485_ = lean_nat_dec_le(v___x_482_, v___x_482_);
if (v___x_485_ == 0)
{
if (v___x_483_ == 0)
{
goto v___jp_459_;
}
else
{
size_t v___x_486_; size_t v___x_487_; lean_object* v___x_488_; 
v___x_486_ = ((size_t)0ULL);
v___x_487_ = lean_usize_of_nat(v___x_482_);
lean_inc(v_decl_419_);
v___x_488_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleBuilder_casesCore_spec__0(v_decl_419_, v_pats_421_, v___x_486_, v___x_487_, v___x_484_, v_a_426_, v_a_427_, v_a_428_, v_a_429_);
v___y_472_ = v___x_488_;
goto v___jp_471_;
}
}
else
{
size_t v___x_489_; size_t v___x_490_; lean_object* v___x_491_; 
v___x_489_ = ((size_t)0ULL);
v___x_490_ = lean_usize_of_nat(v___x_482_);
lean_inc(v_decl_419_);
v___x_491_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleBuilder_casesCore_spec__0(v_decl_419_, v_pats_421_, v___x_489_, v___x_490_, v___x_484_, v_a_426_, v_a_427_, v_a_428_, v_a_429_);
v___y_472_ = v___x_491_;
goto v___jp_471_;
}
}
v___jp_431_:
{
lean_object* v___x_433_; 
lean_inc_ref(v_info_420_);
v___x_433_ = lp_aesop_Aesop_mkCtorNames(v_info_420_, v_a_428_, v_a_429_);
if (lean_obj_tag(v___x_433_) == 0)
{
lean_object* v_a_434_; lean_object* v___x_436_; uint8_t v_isShared_437_; uint8_t v_isSharedCheck_450_; 
v_a_434_ = lean_ctor_get(v___x_433_, 0);
v_isSharedCheck_450_ = !lean_is_exclusive(v___x_433_);
if (v_isSharedCheck_450_ == 0)
{
v___x_436_ = v___x_433_;
v_isShared_437_ = v_isSharedCheck_450_;
goto v_resetjp_435_;
}
else
{
lean_inc(v_a_434_);
lean_dec(v___x_433_);
v___x_436_ = lean_box(0);
v_isShared_437_ = v_isSharedCheck_450_;
goto v_resetjp_435_;
}
v_resetjp_435_:
{
uint8_t v_isRec_438_; lean_object* v___x_439_; lean_object* v___x_440_; uint8_t v___x_441_; uint8_t v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_448_; 
v_isRec_438_ = lean_ctor_get_uint8(v_info_420_, sizeof(void*)*6);
lean_dec_ref(v_info_420_);
lean_inc(v_decl_419_);
v___x_439_ = lp_aesop_Aesop_RuleBuilder_mkCasesTarget(v_decl_419_, v_pats_421_);
v___x_440_ = lean_alloc_ctor(3, 2, 2);
lean_ctor_set(v___x_440_, 0, v___x_439_);
lean_ctor_set(v___x_440_, 1, v_a_434_);
lean_ctor_set_uint8(v___x_440_, sizeof(void*)*2, v_md_423_);
lean_ctor_set_uint8(v___x_440_, sizeof(void*)*2 + 1, v_isRec_438_);
v___x_441_ = 1;
v___x_442_ = 0;
v___x_443_ = lean_box(0);
v___x_444_ = lp_aesop_Aesop_PhaseSpec_toRule(v_phase_425_, v_decl_419_, v___x_441_, v___x_442_, v___x_440_, v_a_432_, v___x_443_);
v___x_445_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_445_, 0, v___x_444_);
v___x_446_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_446_, 0, v___x_445_);
if (v_isShared_437_ == 0)
{
lean_ctor_set(v___x_436_, 0, v___x_446_);
v___x_448_ = v___x_436_;
goto v_reusejp_447_;
}
else
{
lean_object* v_reuseFailAlloc_449_; 
v_reuseFailAlloc_449_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_449_, 0, v___x_446_);
v___x_448_ = v_reuseFailAlloc_449_;
goto v_reusejp_447_;
}
v_reusejp_447_:
{
return v___x_448_;
}
}
}
else
{
lean_object* v_a_451_; lean_object* v___x_453_; uint8_t v_isShared_454_; uint8_t v_isSharedCheck_458_; 
lean_dec(v_a_432_);
lean_dec_ref(v_phase_425_);
lean_dec_ref(v_pats_421_);
lean_dec_ref(v_info_420_);
lean_dec(v_decl_419_);
v_a_451_ = lean_ctor_get(v___x_433_, 0);
v_isSharedCheck_458_ = !lean_is_exclusive(v___x_433_);
if (v_isSharedCheck_458_ == 0)
{
v___x_453_ = v___x_433_;
v_isShared_454_ = v_isSharedCheck_458_;
goto v_resetjp_452_;
}
else
{
lean_inc(v_a_451_);
lean_dec(v___x_433_);
v___x_453_ = lean_box(0);
v_isShared_454_ = v_isSharedCheck_458_;
goto v_resetjp_452_;
}
v_resetjp_452_:
{
lean_object* v___x_456_; 
if (v_isShared_454_ == 0)
{
v___x_456_ = v___x_453_;
goto v_reusejp_455_;
}
else
{
lean_object* v_reuseFailAlloc_457_; 
v_reuseFailAlloc_457_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_457_, 0, v_a_451_);
v___x_456_ = v_reuseFailAlloc_457_;
goto v_reusejp_455_;
}
v_reusejp_455_:
{
return v___x_456_;
}
}
}
}
v___jp_459_:
{
if (lean_obj_tag(v_imode_x3f_422_) == 0)
{
lean_object* v___x_460_; 
lean_inc_ref(v_pats_421_);
lean_inc(v_decl_419_);
v___x_460_ = lp_aesop_Aesop_RuleBuilder_getCasesIndexingMode(v_decl_419_, v_indexMd_424_, v_pats_421_, v_a_426_, v_a_427_, v_a_428_, v_a_429_);
if (lean_obj_tag(v___x_460_) == 0)
{
lean_object* v_a_461_; 
v_a_461_ = lean_ctor_get(v___x_460_, 0);
lean_inc(v_a_461_);
lean_dec_ref_known(v___x_460_, 1);
v_a_432_ = v_a_461_;
goto v___jp_431_;
}
else
{
lean_object* v_a_462_; lean_object* v___x_464_; uint8_t v_isShared_465_; uint8_t v_isSharedCheck_469_; 
lean_dec_ref(v_phase_425_);
lean_dec_ref(v_pats_421_);
lean_dec_ref(v_info_420_);
lean_dec(v_decl_419_);
v_a_462_ = lean_ctor_get(v___x_460_, 0);
v_isSharedCheck_469_ = !lean_is_exclusive(v___x_460_);
if (v_isSharedCheck_469_ == 0)
{
v___x_464_ = v___x_460_;
v_isShared_465_ = v_isSharedCheck_469_;
goto v_resetjp_463_;
}
else
{
lean_inc(v_a_462_);
lean_dec(v___x_460_);
v___x_464_ = lean_box(0);
v_isShared_465_ = v_isSharedCheck_469_;
goto v_resetjp_463_;
}
v_resetjp_463_:
{
lean_object* v___x_467_; 
if (v_isShared_465_ == 0)
{
v___x_467_ = v___x_464_;
goto v_reusejp_466_;
}
else
{
lean_object* v_reuseFailAlloc_468_; 
v_reuseFailAlloc_468_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_468_, 0, v_a_462_);
v___x_467_ = v_reuseFailAlloc_468_;
goto v_reusejp_466_;
}
v_reusejp_466_:
{
return v___x_467_;
}
}
}
}
else
{
lean_object* v_val_470_; 
v_val_470_ = lean_ctor_get(v_imode_x3f_422_, 0);
lean_inc(v_val_470_);
lean_dec_ref_known(v_imode_x3f_422_, 1);
v_a_432_ = v_val_470_;
goto v___jp_431_;
}
}
v___jp_471_:
{
if (lean_obj_tag(v___y_472_) == 0)
{
lean_dec_ref_known(v___y_472_, 1);
goto v___jp_459_;
}
else
{
lean_object* v_a_473_; lean_object* v___x_475_; uint8_t v_isShared_476_; uint8_t v_isSharedCheck_480_; 
lean_dec_ref(v_phase_425_);
lean_dec(v_imode_x3f_422_);
lean_dec_ref(v_pats_421_);
lean_dec_ref(v_info_420_);
lean_dec(v_decl_419_);
v_a_473_ = lean_ctor_get(v___y_472_, 0);
v_isSharedCheck_480_ = !lean_is_exclusive(v___y_472_);
if (v_isSharedCheck_480_ == 0)
{
v___x_475_ = v___y_472_;
v_isShared_476_ = v_isSharedCheck_480_;
goto v_resetjp_474_;
}
else
{
lean_inc(v_a_473_);
lean_dec(v___y_472_);
v___x_475_ = lean_box(0);
v_isShared_476_ = v_isSharedCheck_480_;
goto v_resetjp_474_;
}
v_resetjp_474_:
{
lean_object* v___x_478_; 
if (v_isShared_476_ == 0)
{
v___x_478_ = v___x_475_;
goto v_reusejp_477_;
}
else
{
lean_object* v_reuseFailAlloc_479_; 
v_reuseFailAlloc_479_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_479_, 0, v_a_473_);
v___x_478_ = v_reuseFailAlloc_479_;
goto v_reusejp_477_;
}
v_reusejp_477_:
{
return v___x_478_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_casesCore___boxed(lean_object* v_decl_492_, lean_object* v_info_493_, lean_object* v_pats_494_, lean_object* v_imode_x3f_495_, lean_object* v_md_496_, lean_object* v_indexMd_497_, lean_object* v_phase_498_, lean_object* v_a_499_, lean_object* v_a_500_, lean_object* v_a_501_, lean_object* v_a_502_, lean_object* v_a_503_){
_start:
{
uint8_t v_md_boxed_504_; uint8_t v_indexMd_boxed_505_; lean_object* v_res_506_; 
v_md_boxed_504_ = lean_unbox(v_md_496_);
v_indexMd_boxed_505_ = lean_unbox(v_indexMd_497_);
v_res_506_ = lp_aesop_Aesop_RuleBuilder_casesCore(v_decl_492_, v_info_493_, v_pats_494_, v_imode_x3f_495_, v_md_boxed_504_, v_indexMd_boxed_505_, v_phase_498_, v_a_499_, v_a_500_, v_a_501_, v_a_502_);
lean_dec(v_a_502_);
lean_dec_ref(v_a_501_);
lean_dec(v_a_500_);
lean_dec_ref(v_a_499_);
return v_res_506_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_cases_spec__0___redArg(lean_object* v_msg_507_, lean_object* v___y_508_, lean_object* v___y_509_, lean_object* v___y_510_, lean_object* v___y_511_){
_start:
{
lean_object* v_ref_513_; lean_object* v___x_514_; lean_object* v_a_515_; lean_object* v___x_517_; uint8_t v_isShared_518_; uint8_t v_isSharedCheck_523_; 
v_ref_513_ = lean_ctor_get(v___y_510_, 5);
v___x_514_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00Aesop_CasesPattern_check_spec__0_spec__0(v_msg_507_, v___y_508_, v___y_509_, v___y_510_, v___y_511_);
v_a_515_ = lean_ctor_get(v___x_514_, 0);
v_isSharedCheck_523_ = !lean_is_exclusive(v___x_514_);
if (v_isSharedCheck_523_ == 0)
{
v___x_517_ = v___x_514_;
v_isShared_518_ = v_isSharedCheck_523_;
goto v_resetjp_516_;
}
else
{
lean_inc(v_a_515_);
lean_dec(v___x_514_);
v___x_517_ = lean_box(0);
v_isShared_518_ = v_isSharedCheck_523_;
goto v_resetjp_516_;
}
v_resetjp_516_:
{
lean_object* v___x_519_; lean_object* v___x_521_; 
lean_inc(v_ref_513_);
v___x_519_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_519_, 0, v_ref_513_);
lean_ctor_set(v___x_519_, 1, v_a_515_);
if (v_isShared_518_ == 0)
{
lean_ctor_set_tag(v___x_517_, 1);
lean_ctor_set(v___x_517_, 0, v___x_519_);
v___x_521_ = v___x_517_;
goto v_reusejp_520_;
}
else
{
lean_object* v_reuseFailAlloc_522_; 
v_reuseFailAlloc_522_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_522_, 0, v___x_519_);
v___x_521_ = v_reuseFailAlloc_522_;
goto v_reusejp_520_;
}
v_reusejp_520_:
{
return v___x_521_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_cases_spec__0___redArg___boxed(lean_object* v_msg_524_, lean_object* v___y_525_, lean_object* v___y_526_, lean_object* v___y_527_, lean_object* v___y_528_, lean_object* v___y_529_){
_start:
{
lean_object* v_res_530_; 
v_res_530_ = lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_cases_spec__0___redArg(v_msg_524_, v___y_525_, v___y_526_, v___y_527_, v___y_528_);
lean_dec(v___y_528_);
lean_dec_ref(v___y_527_);
lean_dec(v___y_526_);
lean_dec_ref(v___y_525_);
return v_res_530_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleBuilder_cases___closed__1(void){
_start:
{
lean_object* v___x_532_; lean_object* v___x_533_; 
v___x_532_ = ((lean_object*)(lp_aesop_Aesop_RuleBuilder_cases___closed__0));
v___x_533_ = l_Lean_stringToMessageData(v___x_532_);
return v___x_533_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_cases(lean_object* v_input_534_, lean_object* v_a_535_, lean_object* v_a_536_, lean_object* v_a_537_, lean_object* v_a_538_, lean_object* v_a_539_, lean_object* v_a_540_, lean_object* v_a_541_){
_start:
{
lean_object* v_term_543_; lean_object* v_options_544_; lean_object* v_phase_545_; lean_object* v___y_547_; lean_object* v___y_548_; lean_object* v___y_549_; lean_object* v___y_550_; lean_object* v___y_551_; lean_object* v___y_552_; uint8_t v___x_571_; uint8_t v___x_572_; uint8_t v___x_573_; 
v_term_543_ = lean_ctor_get(v_input_534_, 0);
lean_inc(v_term_543_);
v_options_544_ = lean_ctor_get(v_input_534_, 1);
lean_inc_ref(v_options_544_);
v_phase_545_ = lean_ctor_get(v_input_534_, 2);
lean_inc_ref(v_phase_545_);
lean_dec_ref(v_input_534_);
v___x_571_ = lp_aesop_Aesop_PhaseSpec_phase(v_phase_545_);
v___x_572_ = 0;
v___x_573_ = lp_aesop_Aesop_instBEqPhaseName_beq(v___x_571_, v___x_572_);
if (v___x_573_ == 0)
{
v___y_547_ = v_a_536_;
v___y_548_ = v_a_537_;
v___y_549_ = v_a_538_;
v___y_550_ = v_a_539_;
v___y_551_ = v_a_540_;
v___y_552_ = v_a_541_;
goto v___jp_546_;
}
else
{
lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v_a_576_; lean_object* v___x_578_; uint8_t v_isShared_579_; uint8_t v_isSharedCheck_583_; 
lean_dec_ref(v_phase_545_);
lean_dec_ref(v_options_544_);
lean_dec(v_term_543_);
v___x_574_ = lean_obj_once(&lp_aesop_Aesop_RuleBuilder_cases___closed__1, &lp_aesop_Aesop_RuleBuilder_cases___closed__1_once, _init_lp_aesop_Aesop_RuleBuilder_cases___closed__1);
v___x_575_ = lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_cases_spec__0___redArg(v___x_574_, v_a_538_, v_a_539_, v_a_540_, v_a_541_);
v_a_576_ = lean_ctor_get(v___x_575_, 0);
v_isSharedCheck_583_ = !lean_is_exclusive(v___x_575_);
if (v_isSharedCheck_583_ == 0)
{
v___x_578_ = v___x_575_;
v_isShared_579_ = v_isSharedCheck_583_;
goto v_resetjp_577_;
}
else
{
lean_inc(v_a_576_);
lean_dec(v___x_575_);
v___x_578_ = lean_box(0);
v_isShared_579_ = v_isSharedCheck_583_;
goto v_resetjp_577_;
}
v_resetjp_577_:
{
lean_object* v___x_581_; 
if (v_isShared_579_ == 0)
{
v___x_581_ = v___x_578_;
goto v_reusejp_580_;
}
else
{
lean_object* v_reuseFailAlloc_582_; 
v_reuseFailAlloc_582_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_582_, 0, v_a_576_);
v___x_581_ = v_reuseFailAlloc_582_;
goto v_reusejp_580_;
}
v_reusejp_580_:
{
return v___x_581_;
}
}
}
v___jp_546_:
{
uint8_t v___x_553_; uint8_t v___x_554_; lean_object* v___x_555_; 
v___x_553_ = 1;
v___x_554_ = lp_aesop_Aesop_RuleBuilderOptions_casesTransparency(v_options_544_);
v___x_555_ = lp_aesop_Aesop_elabInductiveRuleIdent(v___x_553_, v_term_543_, v___x_554_, v___y_547_, v___y_548_, v___y_549_, v___y_550_, v___y_551_, v___y_552_);
if (lean_obj_tag(v___x_555_) == 0)
{
lean_object* v_a_556_; lean_object* v_fst_557_; lean_object* v_snd_558_; lean_object* v_indexingMode_x3f_559_; lean_object* v___x_560_; uint8_t v___x_561_; lean_object* v___x_562_; 
v_a_556_ = lean_ctor_get(v___x_555_, 0);
lean_inc(v_a_556_);
lean_dec_ref_known(v___x_555_, 1);
v_fst_557_ = lean_ctor_get(v_a_556_, 0);
lean_inc(v_fst_557_);
v_snd_558_ = lean_ctor_get(v_a_556_, 1);
lean_inc(v_snd_558_);
lean_dec(v_a_556_);
v_indexingMode_x3f_559_ = lean_ctor_get(v_options_544_, 1);
lean_inc(v_indexingMode_x3f_559_);
v___x_560_ = lp_aesop_Aesop_RuleBuilderOptions_casesPatterns(v_options_544_);
v___x_561_ = lp_aesop_Aesop_RuleBuilderOptions_casesIndexTransparency(v_options_544_);
lean_dec_ref(v_options_544_);
v___x_562_ = lp_aesop_Aesop_RuleBuilder_casesCore(v_fst_557_, v_snd_558_, v___x_560_, v_indexingMode_x3f_559_, v___x_554_, v___x_561_, v_phase_545_, v___y_549_, v___y_550_, v___y_551_, v___y_552_);
return v___x_562_;
}
else
{
lean_object* v_a_563_; lean_object* v___x_565_; uint8_t v_isShared_566_; uint8_t v_isSharedCheck_570_; 
lean_dec_ref(v_phase_545_);
lean_dec_ref(v_options_544_);
v_a_563_ = lean_ctor_get(v___x_555_, 0);
v_isSharedCheck_570_ = !lean_is_exclusive(v___x_555_);
if (v_isSharedCheck_570_ == 0)
{
v___x_565_ = v___x_555_;
v_isShared_566_ = v_isSharedCheck_570_;
goto v_resetjp_564_;
}
else
{
lean_inc(v_a_563_);
lean_dec(v___x_555_);
v___x_565_ = lean_box(0);
v_isShared_566_ = v_isSharedCheck_570_;
goto v_resetjp_564_;
}
v_resetjp_564_:
{
lean_object* v___x_568_; 
if (v_isShared_566_ == 0)
{
v___x_568_ = v___x_565_;
goto v_reusejp_567_;
}
else
{
lean_object* v_reuseFailAlloc_569_; 
v_reuseFailAlloc_569_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_569_, 0, v_a_563_);
v___x_568_ = v_reuseFailAlloc_569_;
goto v_reusejp_567_;
}
v_reusejp_567_:
{
return v___x_568_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_cases___boxed(lean_object* v_input_584_, lean_object* v_a_585_, lean_object* v_a_586_, lean_object* v_a_587_, lean_object* v_a_588_, lean_object* v_a_589_, lean_object* v_a_590_, lean_object* v_a_591_, lean_object* v_a_592_){
_start:
{
lean_object* v_res_593_; 
v_res_593_ = lp_aesop_Aesop_RuleBuilder_cases(v_input_584_, v_a_585_, v_a_586_, v_a_587_, v_a_588_, v_a_589_, v_a_590_, v_a_591_);
lean_dec(v_a_591_);
lean_dec_ref(v_a_590_);
lean_dec(v_a_589_);
lean_dec_ref(v_a_588_);
lean_dec(v_a_587_);
lean_dec_ref(v_a_586_);
lean_dec_ref(v_a_585_);
return v_res_593_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_cases_spec__0(lean_object* v_00_u03b1_594_, lean_object* v_msg_595_, lean_object* v___y_596_, lean_object* v___y_597_, lean_object* v___y_598_, lean_object* v___y_599_, lean_object* v___y_600_, lean_object* v___y_601_, lean_object* v___y_602_){
_start:
{
lean_object* v___x_604_; 
v___x_604_ = lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_cases_spec__0___redArg(v_msg_595_, v___y_599_, v___y_600_, v___y_601_, v___y_602_);
return v___x_604_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_cases_spec__0___boxed(lean_object* v_00_u03b1_605_, lean_object* v_msg_606_, lean_object* v___y_607_, lean_object* v___y_608_, lean_object* v___y_609_, lean_object* v___y_610_, lean_object* v___y_611_, lean_object* v___y_612_, lean_object* v___y_613_, lean_object* v___y_614_){
_start:
{
lean_object* v_res_615_; 
v_res_615_ = lp_aesop_Lean_throwError___at___00Aesop_RuleBuilder_cases_spec__0(v_00_u03b1_605_, v_msg_606_, v___y_607_, v___y_608_, v___y_609_, v___y_610_, v___y_611_, v___y_612_, v___y_613_);
lean_dec(v___y_613_);
lean_dec_ref(v___y_612_);
lean_dec(v___y_611_);
lean_dec_ref(v___y_610_);
lean_dec(v___y_609_);
lean_dec_ref(v___y_608_);
lean_dec_ref(v___y_607_);
return v_res_615_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Builder_Basic(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleTac_Cases(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Expr(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Builder_Cases(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Builder_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_Cases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Expr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Builder_Cases(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Builder_Basic(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleTac_Cases(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_Expr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Builder_Cases(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Builder_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_Cases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Expr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Builder_Cases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Builder_Cases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Builder_Cases(builtin);
}
#ifdef __cplusplus
}
#endif
