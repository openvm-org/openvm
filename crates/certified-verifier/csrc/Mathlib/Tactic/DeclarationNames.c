// Lean compiler output
// Module: Mathlib.Tactic.DeclarationNames
// Imports: public import Init public meta import Init public meta import Lean.DeclarationRange public meta import Lean.ResolveName public meta import Mathlib.Tactic.Linter.Header public import Lean.Message
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
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
lean_object* l_Lean_logWarningAt___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_FileMap_ofPosition(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_ofRange(lean_object*, uint8_t);
lean_object* l_Lean_mkIdentFrom(lean_object*, lean_object*, uint8_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_forInStep___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getRange_x3f(lean_object*, uint8_t);
extern lean_object* l_Lean_Syntax_instInhabitedRange_default;
size_t lean_array_size(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
extern lean_object* l_Lean_declRangeExt;
lean_object* l_Lean_PersistentEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getNamesFrom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getNamesFrom___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getNamesFrom___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getNamesFrom___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Linter_getNamesFrom___redArg___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Linter_getNamesFrom___redArg___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_getNamesFrom___redArg___lam__3___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getNamesFrom___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getNamesFrom___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getNamesFrom___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getNamesFrom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___lam__1(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "export"};
static const lean_object* lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(6, 73, 228, 195, 89, 60, 49, 127)}};
static const lean_object* lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getAliasSyntax(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Linter_logLint0Disable___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "This linter can be disabled with `set_option "};
static const lean_object* lp_mathlib_Mathlib_Linter_logLint0Disable___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Linter_logLint0Disable___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_logLint0Disable___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_logLint0Disable___redArg___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Linter_logLint0Disable___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " 0`"};
static const lean_object* lp_mathlib_Mathlib_Linter_logLint0Disable___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Linter_logLint0Disable___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Linter_logLint0Disable___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Linter_logLint0Disable___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_logLint0Disable___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_logLint0Disable(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getNamesFrom___redArg___lam__0(lean_object* v_toPure_1_, lean_object* v_____do__lift_2_){
_start:
{
lean_object* v_a_3_; lean_object* v___x_4_; 
v_a_3_ = lean_ctor_get(v_____do__lift_2_, 0);
lean_inc(v_a_3_);
lean_dec_ref(v_____do__lift_2_);
v___x_4_ = lean_apply_2(v_toPure_1_, lean_box(0), v_a_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getNamesFrom___redArg___lam__1(lean_object* v_toPure_5_, lean_object* v_____s_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lean_apply_2(v_toPure_5_, lean_box(0), v_____s_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getNamesFrom___redArg___lam__2(lean_object* v_fm_8_, lean_object* v_pos_9_, lean_object* v_toPure_10_, lean_object* v_a_11_, lean_object* v_b_12_, lean_object* v_c_13_){
_start:
{
lean_object* v_range_14_; lean_object* v_selectionRange_15_; lean_object* v___x_17_; uint8_t v_isShared_18_; uint8_t v_isSharedCheck_37_; 
v_range_14_ = lean_ctor_get(v_b_12_, 0);
v_selectionRange_15_ = lean_ctor_get(v_b_12_, 1);
v_isSharedCheck_37_ = !lean_is_exclusive(v_b_12_);
if (v_isSharedCheck_37_ == 0)
{
v___x_17_ = v_b_12_;
v_isShared_18_ = v_isSharedCheck_37_;
goto v_resetjp_16_;
}
else
{
lean_inc(v_selectionRange_15_);
lean_inc(v_range_14_);
lean_dec(v_b_12_);
v___x_17_ = lean_box(0);
v_isShared_18_ = v_isSharedCheck_37_;
goto v_resetjp_16_;
}
v_resetjp_16_:
{
lean_object* v_pos_19_; lean_object* v___x_20_; uint8_t v___x_21_; 
v_pos_19_ = lean_ctor_get(v_range_14_, 0);
lean_inc_ref(v_pos_19_);
lean_dec_ref(v_range_14_);
v___x_20_ = l_Lean_FileMap_ofPosition(v_fm_8_, v_pos_19_);
v___x_21_ = lean_nat_dec_le(v_pos_9_, v___x_20_);
lean_dec(v___x_20_);
if (v___x_21_ == 0)
{
lean_object* v___x_22_; lean_object* v___x_23_; 
lean_del_object(v___x_17_);
lean_dec_ref(v_selectionRange_15_);
lean_dec(v_a_11_);
v___x_22_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_22_, 0, v_c_13_);
v___x_23_ = lean_apply_2(v_toPure_10_, lean_box(0), v___x_22_);
return v___x_23_;
}
else
{
lean_object* v_pos_24_; lean_object* v_endPos_25_; lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_29_; 
v_pos_24_ = lean_ctor_get(v_selectionRange_15_, 0);
lean_inc_ref(v_pos_24_);
v_endPos_25_ = lean_ctor_get(v_selectionRange_15_, 2);
lean_inc_ref(v_endPos_25_);
lean_dec_ref(v_selectionRange_15_);
v___x_26_ = l_Lean_FileMap_ofPosition(v_fm_8_, v_pos_24_);
v___x_27_ = l_Lean_FileMap_ofPosition(v_fm_8_, v_endPos_25_);
if (v_isShared_18_ == 0)
{
lean_ctor_set(v___x_17_, 1, v___x_27_);
lean_ctor_set(v___x_17_, 0, v___x_26_);
v___x_29_ = v___x_17_;
goto v_reusejp_28_;
}
else
{
lean_object* v_reuseFailAlloc_36_; 
v_reuseFailAlloc_36_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_36_, 0, v___x_26_);
lean_ctor_set(v_reuseFailAlloc_36_, 1, v___x_27_);
v___x_29_ = v_reuseFailAlloc_36_;
goto v_reusejp_28_;
}
v_reusejp_28_:
{
lean_object* v___x_30_; uint8_t v___x_31_; lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; 
v___x_30_ = l_Lean_Syntax_ofRange(v___x_29_, v___x_21_);
v___x_31_ = 0;
v___x_32_ = l_Lean_mkIdentFrom(v___x_30_, v_a_11_, v___x_31_);
lean_dec(v___x_30_);
v___x_33_ = lean_array_push(v_c_13_, v___x_32_);
v___x_34_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_34_, 0, v___x_33_);
v___x_35_ = lean_apply_2(v_toPure_10_, lean_box(0), v___x_34_);
return v___x_35_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getNamesFrom___redArg___lam__2___boxed(lean_object* v_fm_38_, lean_object* v_pos_39_, lean_object* v_toPure_40_, lean_object* v_a_41_, lean_object* v_b_42_, lean_object* v_c_43_){
_start:
{
lean_object* v_res_44_; 
v_res_44_ = lp_mathlib_Mathlib_Linter_getNamesFrom___redArg___lam__2(v_fm_38_, v_pos_39_, v_toPure_40_, v_a_41_, v_b_42_, v_c_43_);
lean_dec(v_pos_39_);
lean_dec_ref(v_fm_38_);
return v_res_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getNamesFrom___redArg___lam__3(lean_object* v_pos_47_, lean_object* v_toPure_48_, lean_object* v_inst_49_, lean_object* v_drs_50_, lean_object* v_toBind_51_, lean_object* v___f_52_, lean_object* v___f_53_, lean_object* v_fm_54_){
_start:
{
lean_object* v___f_55_; lean_object* v_nms_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; 
v___f_55_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Linter_getNamesFrom___redArg___lam__2___boxed), 6, 3);
lean_closure_set(v___f_55_, 0, v_fm_54_);
lean_closure_set(v___f_55_, 1, v_pos_47_);
lean_closure_set(v___f_55_, 2, v_toPure_48_);
v_nms_56_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_getNamesFrom___redArg___lam__3___closed__0));
v___x_57_ = l_Std_DTreeMap_Internal_Impl_forInStep___redArg(v_inst_49_, v___f_55_, v_nms_56_, v_drs_50_);
lean_inc(v_toBind_51_);
v___x_58_ = lean_apply_4(v_toBind_51_, lean_box(0), lean_box(0), v___x_57_, v___f_52_);
v___x_59_ = lean_apply_4(v_toBind_51_, lean_box(0), lean_box(0), v___x_58_, v___f_53_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getNamesFrom___redArg___lam__4(lean_object* v___x_60_, lean_object* v_pos_61_, lean_object* v_toPure_62_, lean_object* v_inst_63_, lean_object* v_toBind_64_, lean_object* v___f_65_, lean_object* v___f_66_, lean_object* v_inst_67_, lean_object* v_____do__lift_68_){
_start:
{
lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v_drs_72_; lean_object* v___f_73_; lean_object* v___x_74_; 
v___x_69_ = l_Lean_declRangeExt;
v___x_70_ = lean_box(1);
v___x_71_ = lean_box(0);
v_drs_72_ = l_Lean_PersistentEnvExtension_getState___redArg(v___x_60_, v___x_69_, v_____do__lift_68_, v___x_70_, v___x_71_);
lean_inc(v_toBind_64_);
v___f_73_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Linter_getNamesFrom___redArg___lam__3), 8, 7);
lean_closure_set(v___f_73_, 0, v_pos_61_);
lean_closure_set(v___f_73_, 1, v_toPure_62_);
lean_closure_set(v___f_73_, 2, v_inst_63_);
lean_closure_set(v___f_73_, 3, v_drs_72_);
lean_closure_set(v___f_73_, 4, v_toBind_64_);
lean_closure_set(v___f_73_, 5, v___f_65_);
lean_closure_set(v___f_73_, 6, v___f_66_);
v___x_74_ = lean_apply_4(v_toBind_64_, lean_box(0), lean_box(0), v_inst_67_, v___f_73_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getNamesFrom___redArg(lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_pos_78_){
_start:
{
lean_object* v_toApplicative_79_; lean_object* v_toBind_80_; lean_object* v_getEnv_81_; lean_object* v_toPure_82_; lean_object* v___x_83_; lean_object* v___f_84_; lean_object* v___f_85_; lean_object* v___f_86_; lean_object* v___x_87_; 
v_toApplicative_79_ = lean_ctor_get(v_inst_75_, 0);
v_toBind_80_ = lean_ctor_get(v_inst_75_, 1);
lean_inc_n(v_toBind_80_, 2);
v_getEnv_81_ = lean_ctor_get(v_inst_76_, 0);
lean_inc(v_getEnv_81_);
lean_dec_ref(v_inst_76_);
v_toPure_82_ = lean_ctor_get(v_toApplicative_79_, 1);
lean_inc_n(v_toPure_82_, 3);
v___x_83_ = lean_box(1);
v___f_84_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Linter_getNamesFrom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_84_, 0, v_toPure_82_);
v___f_85_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Linter_getNamesFrom___redArg___lam__1), 2, 1);
lean_closure_set(v___f_85_, 0, v_toPure_82_);
v___f_86_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Linter_getNamesFrom___redArg___lam__4), 9, 8);
lean_closure_set(v___f_86_, 0, v___x_83_);
lean_closure_set(v___f_86_, 1, v_pos_78_);
lean_closure_set(v___f_86_, 2, v_toPure_82_);
lean_closure_set(v___f_86_, 3, v_inst_75_);
lean_closure_set(v___f_86_, 4, v_toBind_80_);
lean_closure_set(v___f_86_, 5, v___f_84_);
lean_closure_set(v___f_86_, 6, v___f_85_);
lean_closure_set(v___f_86_, 7, v_inst_77_);
v___x_87_ = lean_apply_4(v_toBind_80_, lean_box(0), lean_box(0), v_getEnv_81_, v___f_86_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getNamesFrom(lean_object* v_m_88_, lean_object* v_inst_89_, lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_pos_92_){
_start:
{
lean_object* v___x_93_; 
v___x_93_ = lp_mathlib_Mathlib_Linter_getNamesFrom___redArg(v_inst_89_, v_inst_90_, v_inst_91_, v_pos_92_);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___lam__1(uint8_t v___x_94_, lean_object* v_currNamespace_95_, lean_object* v_toPure_96_, lean_object* v_a_97_, lean_object* v_x_98_, lean_object* v___y_99_){
_start:
{
lean_object* v___x_100_; uint8_t v___x_101_; lean_object* v___y_103_; lean_object* v___x_110_; 
v___x_100_ = l_Lean_TSyntax_getId(v_a_97_);
v___x_101_ = 0;
v___x_110_ = l_Lean_Syntax_getRange_x3f(v_a_97_, v___x_101_);
if (lean_obj_tag(v___x_110_) == 0)
{
lean_object* v___x_111_; 
v___x_111_ = l_Lean_Syntax_instInhabitedRange_default;
v___y_103_ = v___x_111_;
goto v___jp_102_;
}
else
{
lean_object* v_val_112_; 
v_val_112_ = lean_ctor_get(v___x_110_, 0);
lean_inc(v_val_112_);
lean_dec_ref_known(v___x_110_, 1);
v___y_103_ = v_val_112_;
goto v___jp_102_;
}
v___jp_102_:
{
lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; 
v___x_104_ = l_Lean_Syntax_ofRange(v___y_103_, v___x_94_);
v___x_105_ = l_Lean_Name_append(v_currNamespace_95_, v___x_100_);
v___x_106_ = l_Lean_mkIdentFrom(v___x_104_, v___x_105_, v___x_101_);
lean_dec(v___x_104_);
v___x_107_ = lean_array_push(v___y_99_, v___x_106_);
v___x_108_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_108_, 0, v___x_107_);
v___x_109_ = lean_apply_2(v_toPure_96_, lean_box(0), v___x_108_);
return v___x_109_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___lam__1___boxed(lean_object* v___x_113_, lean_object* v_currNamespace_114_, lean_object* v_toPure_115_, lean_object* v_a_116_, lean_object* v_x_117_, lean_object* v___y_118_){
_start:
{
uint8_t v___x_301__boxed_119_; lean_object* v_res_120_; 
v___x_301__boxed_119_ = lean_unbox(v___x_113_);
v_res_120_ = lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___lam__1(v___x_301__boxed_119_, v_currNamespace_114_, v_toPure_115_, v_a_116_, v_x_117_, v___y_118_);
lean_dec(v_a_116_);
return v_res_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___lam__0(uint8_t v___x_121_, lean_object* v_toPure_122_, lean_object* v_ids_123_, lean_object* v_inst_124_, lean_object* v_aliases_125_, lean_object* v_toBind_126_, lean_object* v___f_127_, lean_object* v_currNamespace_128_){
_start:
{
lean_object* v___x_129_; lean_object* v___f_130_; size_t v_sz_131_; size_t v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; 
v___x_129_ = lean_box(v___x_121_);
v___f_130_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___lam__1___boxed), 6, 3);
lean_closure_set(v___f_130_, 0, v___x_129_);
lean_closure_set(v___f_130_, 1, v_currNamespace_128_);
lean_closure_set(v___f_130_, 2, v_toPure_122_);
v_sz_131_ = lean_array_size(v_ids_123_);
v___x_132_ = ((size_t)0ULL);
v___x_133_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v_inst_124_, v_ids_123_, v___f_130_, v_sz_131_, v___x_132_, v_aliases_125_);
v___x_134_ = lean_apply_4(v_toBind_126_, lean_box(0), lean_box(0), v___x_133_, v___f_127_);
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___lam__0___boxed(lean_object* v___x_135_, lean_object* v_toPure_136_, lean_object* v_ids_137_, lean_object* v_inst_138_, lean_object* v_aliases_139_, lean_object* v_toBind_140_, lean_object* v___f_141_, lean_object* v_currNamespace_142_){
_start:
{
uint8_t v___x_336__boxed_143_; lean_object* v_res_144_; 
v___x_336__boxed_143_ = lean_unbox(v___x_135_);
v_res_144_ = lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___lam__0(v___x_336__boxed_143_, v_toPure_136_, v_ids_137_, v_inst_138_, v_aliases_139_, v_toBind_140_, v___f_141_, v_currNamespace_142_);
return v_res_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg(lean_object* v_inst_154_, lean_object* v_inst_155_, lean_object* v_stx_156_){
_start:
{
lean_object* v_toApplicative_157_; lean_object* v_toBind_158_; lean_object* v_toPure_159_; lean_object* v_aliases_160_; lean_object* v___x_161_; uint8_t v___x_162_; 
v_toApplicative_157_ = lean_ctor_get(v_inst_154_, 0);
v_toBind_158_ = lean_ctor_get(v_inst_154_, 1);
lean_inc(v_toBind_158_);
v_toPure_159_ = lean_ctor_get(v_toApplicative_157_, 1);
lean_inc(v_toPure_159_);
v_aliases_160_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_getNamesFrom___redArg___lam__3___closed__0));
v___x_161_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___closed__4));
lean_inc(v_stx_156_);
v___x_162_ = l_Lean_Syntax_isOfKind(v_stx_156_, v___x_161_);
if (v___x_162_ == 0)
{
lean_object* v___x_163_; 
lean_dec(v_toBind_158_);
lean_dec(v_stx_156_);
lean_dec_ref(v_inst_155_);
lean_dec_ref(v_inst_154_);
v___x_163_ = lean_apply_2(v_toPure_159_, lean_box(0), v_aliases_160_);
return v___x_163_;
}
else
{
lean_object* v_getCurrNamespace_164_; lean_object* v___f_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v_ids_168_; lean_object* v___x_169_; lean_object* v___f_170_; lean_object* v___x_171_; 
v_getCurrNamespace_164_ = lean_ctor_get(v_inst_155_, 0);
lean_inc(v_getCurrNamespace_164_);
lean_dec_ref(v_inst_155_);
lean_inc(v_toPure_159_);
v___f_165_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Linter_getNamesFrom___redArg___lam__1), 2, 1);
lean_closure_set(v___f_165_, 0, v_toPure_159_);
v___x_166_ = lean_unsigned_to_nat(3u);
v___x_167_ = l_Lean_Syntax_getArg(v_stx_156_, v___x_166_);
lean_dec(v_stx_156_);
v_ids_168_ = l_Lean_Syntax_getArgs(v___x_167_);
lean_dec(v___x_167_);
v___x_169_ = lean_box(v___x_162_);
lean_inc(v_toBind_158_);
v___f_170_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg___lam__0___boxed), 8, 7);
lean_closure_set(v___f_170_, 0, v___x_169_);
lean_closure_set(v___f_170_, 1, v_toPure_159_);
lean_closure_set(v___f_170_, 2, v_ids_168_);
lean_closure_set(v___f_170_, 3, v_inst_154_);
lean_closure_set(v___f_170_, 4, v_aliases_160_);
lean_closure_set(v___f_170_, 5, v_toBind_158_);
lean_closure_set(v___f_170_, 6, v___f_165_);
v___x_171_ = lean_apply_4(v_toBind_158_, lean_box(0), lean_box(0), v_getCurrNamespace_164_, v___f_170_);
return v___x_171_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_getAliasSyntax(lean_object* v_m_172_, lean_object* v_inst_173_, lean_object* v_inst_174_, lean_object* v_stx_175_){
_start:
{
lean_object* v___x_176_; 
v___x_176_ = lp_mathlib_Mathlib_Linter_getAliasSyntax___redArg(v_inst_173_, v_inst_174_, v_stx_175_);
return v___x_176_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_logLint0Disable___redArg___closed__1(void){
_start:
{
lean_object* v___x_178_; lean_object* v___x_179_; 
v___x_178_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_logLint0Disable___redArg___closed__0));
v___x_179_ = l_Lean_stringToMessageData(v___x_178_);
return v___x_179_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Linter_logLint0Disable___redArg___closed__3(void){
_start:
{
lean_object* v___x_181_; lean_object* v___x_182_; 
v___x_181_ = ((lean_object*)(lp_mathlib_Mathlib_Linter_logLint0Disable___redArg___closed__2));
v___x_182_ = l_Lean_stringToMessageData(v___x_181_);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_logLint0Disable___redArg(lean_object* v_inst_183_, lean_object* v_inst_184_, lean_object* v_inst_185_, lean_object* v_inst_186_, lean_object* v_linterOption_187_, lean_object* v_stx_188_, lean_object* v_msg_189_){
_start:
{
lean_object* v_name_190_; lean_object* v___x_192_; uint8_t v_isShared_193_; uint8_t v_isSharedCheck_205_; 
v_name_190_ = lean_ctor_get(v_linterOption_187_, 0);
v_isSharedCheck_205_ = !lean_is_exclusive(v_linterOption_187_);
if (v_isSharedCheck_205_ == 0)
{
lean_object* v_unused_206_; 
v_unused_206_ = lean_ctor_get(v_linterOption_187_, 1);
lean_dec(v_unused_206_);
v___x_192_ = v_linterOption_187_;
v_isShared_193_ = v_isSharedCheck_205_;
goto v_resetjp_191_;
}
else
{
lean_inc(v_name_190_);
lean_dec(v_linterOption_187_);
v___x_192_ = lean_box(0);
v_isShared_193_ = v_isSharedCheck_205_;
goto v_resetjp_191_;
}
v_resetjp_191_:
{
lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_197_; 
v___x_194_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_logLint0Disable___redArg___closed__1, &lp_mathlib_Mathlib_Linter_logLint0Disable___redArg___closed__1_once, _init_lp_mathlib_Mathlib_Linter_logLint0Disable___redArg___closed__1);
lean_inc(v_name_190_);
v___x_195_ = l_Lean_MessageData_ofName(v_name_190_);
if (v_isShared_193_ == 0)
{
lean_ctor_set_tag(v___x_192_, 7);
lean_ctor_set(v___x_192_, 1, v___x_195_);
lean_ctor_set(v___x_192_, 0, v___x_194_);
v___x_197_ = v___x_192_;
goto v_reusejp_196_;
}
else
{
lean_object* v_reuseFailAlloc_204_; 
v_reuseFailAlloc_204_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_204_, 0, v___x_194_);
lean_ctor_set(v_reuseFailAlloc_204_, 1, v___x_195_);
v___x_197_ = v_reuseFailAlloc_204_;
goto v_reusejp_196_;
}
v_reusejp_196_:
{
lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v_disable_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; 
v___x_198_ = lean_obj_once(&lp_mathlib_Mathlib_Linter_logLint0Disable___redArg___closed__3, &lp_mathlib_Mathlib_Linter_logLint0Disable___redArg___closed__3_once, _init_lp_mathlib_Mathlib_Linter_logLint0Disable___redArg___closed__3);
v___x_199_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_199_, 0, v___x_197_);
lean_ctor_set(v___x_199_, 1, v___x_198_);
v_disable_200_ = l_Lean_MessageData_note(v___x_199_);
v___x_201_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_201_, 0, v_msg_189_);
lean_ctor_set(v___x_201_, 1, v_disable_200_);
v___x_202_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_202_, 0, v_name_190_);
lean_ctor_set(v___x_202_, 1, v___x_201_);
v___x_203_ = l_Lean_logWarningAt___redArg(v_inst_183_, v_inst_184_, v_inst_185_, v_inst_186_, v_stx_188_, v___x_202_);
return v___x_203_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Linter_logLint0Disable(lean_object* v_m_207_, lean_object* v_inst_208_, lean_object* v_inst_209_, lean_object* v_inst_210_, lean_object* v_inst_211_, lean_object* v_linterOption_212_, lean_object* v_stx_213_, lean_object* v_msg_214_){
_start:
{
lean_object* v___x_215_; 
v___x_215_ = lp_mathlib_Mathlib_Linter_logLint0Disable___redArg(v_inst_208_, v_inst_209_, v_inst_210_, v_inst_211_, v_linterOption_212_, v_stx_213_, v_msg_214_);
return v___x_215_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Message(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_DeclarationNames(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Message(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_DeclarationRange(uint8_t builtin);
lean_object* runtime_initialize_Lean_ResolveName(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_DeclarationNames(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_DeclarationRange(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_ResolveName(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_DeclarationRange(uint8_t builtin);
lean_object* initialize_Lean_ResolveName(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
lean_object* initialize_Lean_Message(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_DeclarationNames(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_DeclarationRange(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_ResolveName(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Message(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_DeclarationNames(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_DeclarationNames(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_DeclarationNames(builtin);
}
#ifdef __cplusplus
}
#endif
