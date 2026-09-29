// Lean compiler output
// Module: Aesop.Builder.Default
// Imports: public import Init public meta import Init public import Aesop.Builder.Constructors public import Aesop.Builder.NormSimp public import Aesop.Builder.Tactic public import Aesop.Builder.Apply
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
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleBuilder_apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* lp_aesop_Aesop_RuleBuilder_simp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_PhaseSpec_phase(lean_object*);
lean_object* lp_aesop_Aesop_RuleBuilder_constructors___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleBuilder_tactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "aesop: Unable to interpret '"};
static const lean_object* lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__1;
static const lean_string_object lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "' as "};
static const lean_object* lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__2_value;
static lean_once_cell_t lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__3;
static const lean_string_object lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = " rule. Try specifying a builder."};
static const lean_object* lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__4_value;
static lean_once_cell_t lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__5;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_RuleBuilder_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "an unsafe"};
static const lean_object* lp_aesop_Aesop_RuleBuilder_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleBuilder_default___closed__0_value;
static const lean_string_object lp_aesop_Aesop_RuleBuilder_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "a norm"};
static const lean_object* lp_aesop_Aesop_RuleBuilder_default___closed__1 = (const lean_object*)&lp_aesop_Aesop_RuleBuilder_default___closed__1_value;
static const lean_string_object lp_aesop_Aesop_RuleBuilder_default___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "a safe"};
static const lean_object* lp_aesop_Aesop_RuleBuilder_default___closed__2 = (const lean_object*)&lp_aesop_Aesop_RuleBuilder_default___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_default(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_default___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0_spec__0(lean_object* v_msgData_1_, lean_object* v___y_2_, lean_object* v___y_3_, lean_object* v___y_4_, lean_object* v___y_5_){
_start:
{
lean_object* v___x_7_; lean_object* v_env_8_; lean_object* v___x_9_; lean_object* v_mctx_10_; lean_object* v_lctx_11_; lean_object* v_options_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_7_ = lean_st_ref_get(v___y_5_);
v_env_8_ = lean_ctor_get(v___x_7_, 0);
lean_inc_ref(v_env_8_);
lean_dec(v___x_7_);
v___x_9_ = lean_st_ref_get(v___y_3_);
v_mctx_10_ = lean_ctor_get(v___x_9_, 0);
lean_inc_ref(v_mctx_10_);
lean_dec(v___x_9_);
v_lctx_11_ = lean_ctor_get(v___y_2_, 2);
v_options_12_ = lean_ctor_get(v___y_4_, 2);
lean_inc_ref(v_options_12_);
lean_inc_ref(v_lctx_11_);
v___x_13_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_13_, 0, v_env_8_);
lean_ctor_set(v___x_13_, 1, v_mctx_10_);
lean_ctor_set(v___x_13_, 2, v_lctx_11_);
lean_ctor_set(v___x_13_, 3, v_options_12_);
v___x_14_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_14_, 0, v___x_13_);
lean_ctor_set(v___x_14_, 1, v_msgData_1_);
v___x_15_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_15_, 0, v___x_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0_spec__0___boxed(lean_object* v_msgData_16_, lean_object* v___y_17_, lean_object* v___y_18_, lean_object* v___y_19_, lean_object* v___y_20_, lean_object* v___y_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0_spec__0(v_msgData_16_, v___y_17_, v___y_18_, v___y_19_, v___y_20_);
lean_dec(v___y_20_);
lean_dec_ref(v___y_19_);
lean_dec(v___y_18_);
lean_dec_ref(v___y_17_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0___redArg(lean_object* v_msg_23_, lean_object* v___y_24_, lean_object* v___y_25_, lean_object* v___y_26_, lean_object* v___y_27_){
_start:
{
lean_object* v_ref_29_; lean_object* v___x_30_; lean_object* v_a_31_; lean_object* v___x_33_; uint8_t v_isShared_34_; uint8_t v_isSharedCheck_39_; 
v_ref_29_ = lean_ctor_get(v___y_26_, 5);
v___x_30_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0_spec__0(v_msg_23_, v___y_24_, v___y_25_, v___y_26_, v___y_27_);
v_a_31_ = lean_ctor_get(v___x_30_, 0);
v_isSharedCheck_39_ = !lean_is_exclusive(v___x_30_);
if (v_isSharedCheck_39_ == 0)
{
v___x_33_ = v___x_30_;
v_isShared_34_ = v_isSharedCheck_39_;
goto v_resetjp_32_;
}
else
{
lean_inc(v_a_31_);
lean_dec(v___x_30_);
v___x_33_ = lean_box(0);
v_isShared_34_ = v_isSharedCheck_39_;
goto v_resetjp_32_;
}
v_resetjp_32_:
{
lean_object* v___x_35_; lean_object* v___x_37_; 
lean_inc(v_ref_29_);
v___x_35_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_35_, 0, v_ref_29_);
lean_ctor_set(v___x_35_, 1, v_a_31_);
if (v_isShared_34_ == 0)
{
lean_ctor_set_tag(v___x_33_, 1);
lean_ctor_set(v___x_33_, 0, v___x_35_);
v___x_37_ = v___x_33_;
goto v_reusejp_36_;
}
else
{
lean_object* v_reuseFailAlloc_38_; 
v_reuseFailAlloc_38_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_38_, 0, v___x_35_);
v___x_37_ = v_reuseFailAlloc_38_;
goto v_reusejp_36_;
}
v_reusejp_36_:
{
return v___x_37_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0___redArg___boxed(lean_object* v_msg_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_aesop_Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0___redArg(v_msg_40_, v___y_41_, v___y_42_, v___y_43_, v___y_44_);
lean_dec(v___y_44_);
lean_dec_ref(v___y_43_);
lean_dec(v___y_42_);
lean_dec_ref(v___y_41_);
return v_res_46_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__1(void){
_start:
{
lean_object* v___x_48_; lean_object* v___x_49_; 
v___x_48_ = ((lean_object*)(lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__0));
v___x_49_ = l_Lean_stringToMessageData(v___x_48_);
return v___x_49_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__3(void){
_start:
{
lean_object* v___x_51_; lean_object* v___x_52_; 
v___x_51_ = ((lean_object*)(lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__2));
v___x_52_ = l_Lean_stringToMessageData(v___x_51_);
return v___x_52_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__5(void){
_start:
{
lean_object* v___x_54_; lean_object* v___x_55_; 
v___x_54_ = ((lean_object*)(lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__4));
v___x_55_ = l_Lean_stringToMessageData(v___x_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err(lean_object* v_ruleType_56_, lean_object* v_a_57_, lean_object* v_a_58_, lean_object* v_a_59_, lean_object* v_a_60_, lean_object* v_a_61_, lean_object* v_a_62_, lean_object* v_a_63_, lean_object* v_a_64_){
_start:
{
lean_object* v_term_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; 
v_term_66_ = lean_ctor_get(v_a_57_, 0);
lean_inc(v_term_66_);
lean_dec_ref(v_a_57_);
v___x_67_ = lean_obj_once(&lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__1, &lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__1_once, _init_lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__1);
v___x_68_ = l_Lean_MessageData_ofSyntax(v_term_66_);
v___x_69_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_69_, 0, v___x_67_);
lean_ctor_set(v___x_69_, 1, v___x_68_);
v___x_70_ = lean_obj_once(&lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__3, &lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__3_once, _init_lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__3);
v___x_71_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_71_, 0, v___x_69_);
lean_ctor_set(v___x_71_, 1, v___x_70_);
v___x_72_ = l_Lean_stringToMessageData(v_ruleType_56_);
v___x_73_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_73_, 0, v___x_71_);
lean_ctor_set(v___x_73_, 1, v___x_72_);
v___x_74_ = lean_obj_once(&lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__5, &lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__5_once, _init_lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___closed__5);
v___x_75_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_75_, 0, v___x_73_);
lean_ctor_set(v___x_75_, 1, v___x_74_);
v___x_76_ = lp_aesop_Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0___redArg(v___x_75_, v_a_61_, v_a_62_, v_a_63_, v_a_64_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err___boxed(lean_object* v_ruleType_77_, lean_object* v_a_78_, lean_object* v_a_79_, lean_object* v_a_80_, lean_object* v_a_81_, lean_object* v_a_82_, lean_object* v_a_83_, lean_object* v_a_84_, lean_object* v_a_85_, lean_object* v_a_86_){
_start:
{
lean_object* v_res_87_; 
v_res_87_ = lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err(v_ruleType_77_, v_a_78_, v_a_79_, v_a_80_, v_a_81_, v_a_82_, v_a_83_, v_a_84_, v_a_85_);
lean_dec(v_a_85_);
lean_dec_ref(v_a_84_);
lean_dec(v_a_83_);
lean_dec_ref(v_a_82_);
lean_dec(v_a_81_);
lean_dec_ref(v_a_80_);
lean_dec_ref(v_a_79_);
return v_res_87_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0(lean_object* v_00_u03b1_88_, lean_object* v_msg_89_, lean_object* v___y_90_, lean_object* v___y_91_, lean_object* v___y_92_, lean_object* v___y_93_, lean_object* v___y_94_, lean_object* v___y_95_, lean_object* v___y_96_){
_start:
{
lean_object* v___x_98_; 
v___x_98_ = lp_aesop_Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0___redArg(v_msg_89_, v___y_93_, v___y_94_, v___y_95_, v___y_96_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0___boxed(lean_object* v_00_u03b1_99_, lean_object* v_msg_100_, lean_object* v___y_101_, lean_object* v___y_102_, lean_object* v___y_103_, lean_object* v___y_104_, lean_object* v___y_105_, lean_object* v___y_106_, lean_object* v___y_107_, lean_object* v___y_108_){
_start:
{
lean_object* v_res_109_; 
v_res_109_ = lp_aesop_Lean_throwError___at___00__private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err_spec__0(v_00_u03b1_99_, v_msg_100_, v___y_101_, v___y_102_, v___y_103_, v___y_104_, v___y_105_, v___y_106_, v___y_107_);
lean_dec(v___y_107_);
lean_dec_ref(v___y_106_);
lean_dec(v___y_105_);
lean_dec_ref(v___y_104_);
lean_dec(v___y_103_);
lean_dec_ref(v___y_102_);
lean_dec_ref(v___y_101_);
return v_res_109_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_default(lean_object* v_input_113_, lean_object* v_a_114_, lean_object* v_a_115_, lean_object* v_a_116_, lean_object* v_a_117_, lean_object* v_a_118_, lean_object* v_a_119_, lean_object* v_a_120_){
_start:
{
lean_object* v___y_123_; lean_object* v___y_124_; uint8_t v___y_125_; lean_object* v___y_138_; lean_object* v___y_139_; uint8_t v___y_140_; lean_object* v___y_165_; lean_object* v___y_166_; uint8_t v___y_167_; lean_object* v___y_180_; lean_object* v___y_181_; uint8_t v___y_182_; lean_object* v___y_207_; lean_object* v___y_208_; uint8_t v___y_209_; lean_object* v___y_234_; lean_object* v___y_235_; uint8_t v___y_236_; lean_object* v___y_249_; lean_object* v___y_250_; uint8_t v___y_251_; lean_object* v_phase_275_; uint8_t v___x_276_; 
v_phase_275_ = lean_ctor_get(v_input_113_, 2);
v___x_276_ = lp_aesop_Aesop_PhaseSpec_phase(v_phase_275_);
switch(v___x_276_)
{
case 0:
{
lean_object* v___x_277_; 
v___x_277_ = l_Lean_Meta_saveState___redArg(v_a_118_, v_a_120_);
if (lean_obj_tag(v___x_277_) == 0)
{
lean_object* v_a_278_; lean_object* v___x_279_; 
v_a_278_ = lean_ctor_get(v___x_277_, 0);
lean_inc(v_a_278_);
lean_dec_ref_known(v___x_277_, 1);
lean_inc_ref(v_input_113_);
v___x_279_ = lp_aesop_Aesop_RuleBuilder_constructors___redArg(v_input_113_, v_a_115_, v_a_116_, v_a_117_, v_a_118_, v_a_119_, v_a_120_);
if (lean_obj_tag(v___x_279_) == 0)
{
lean_dec(v_a_278_);
lean_dec_ref(v_input_113_);
return v___x_279_;
}
else
{
lean_object* v_a_280_; uint8_t v___y_282_; uint8_t v___x_306_; 
v_a_280_ = lean_ctor_get(v___x_279_, 0);
lean_inc(v_a_280_);
v___x_306_ = l_Lean_Exception_isInterrupt(v_a_280_);
if (v___x_306_ == 0)
{
uint8_t v___x_307_; 
v___x_307_ = l_Lean_Exception_isRuntime(v_a_280_);
v___y_282_ = v___x_307_;
goto v___jp_281_;
}
else
{
lean_dec(v_a_280_);
v___y_282_ = v___x_306_;
goto v___jp_281_;
}
v___jp_281_:
{
if (v___y_282_ == 0)
{
lean_object* v___x_283_; 
lean_dec_ref_known(v___x_279_, 1);
v___x_283_ = l_Lean_Meta_SavedState_restore___redArg(v_a_278_, v_a_118_, v_a_120_);
lean_dec(v_a_278_);
if (lean_obj_tag(v___x_283_) == 0)
{
lean_object* v___x_284_; 
lean_dec_ref_known(v___x_283_, 1);
v___x_284_ = l_Lean_Meta_saveState___redArg(v_a_118_, v_a_120_);
if (lean_obj_tag(v___x_284_) == 0)
{
lean_object* v_a_285_; lean_object* v___x_286_; 
v_a_285_ = lean_ctor_get(v___x_284_, 0);
lean_inc(v_a_285_);
lean_dec_ref_known(v___x_284_, 1);
lean_inc_ref(v_input_113_);
v___x_286_ = lp_aesop_Aesop_RuleBuilder_tactic(v_input_113_, v_a_114_, v_a_115_, v_a_116_, v_a_117_, v_a_118_, v_a_119_, v_a_120_);
if (lean_obj_tag(v___x_286_) == 0)
{
lean_dec(v_a_285_);
lean_dec_ref(v_input_113_);
return v___x_286_;
}
else
{
lean_object* v_a_287_; uint8_t v___x_288_; 
v_a_287_ = lean_ctor_get(v___x_286_, 0);
lean_inc(v_a_287_);
v___x_288_ = l_Lean_Exception_isInterrupt(v_a_287_);
if (v___x_288_ == 0)
{
uint8_t v___x_289_; 
v___x_289_ = l_Lean_Exception_isRuntime(v_a_287_);
v___y_207_ = v_a_285_;
v___y_208_ = v___x_286_;
v___y_209_ = v___x_289_;
goto v___jp_206_;
}
else
{
lean_dec(v_a_287_);
v___y_207_ = v_a_285_;
v___y_208_ = v___x_286_;
v___y_209_ = v___x_288_;
goto v___jp_206_;
}
}
}
else
{
lean_object* v_a_290_; lean_object* v___x_292_; uint8_t v_isShared_293_; uint8_t v_isSharedCheck_297_; 
lean_dec_ref(v_input_113_);
v_a_290_ = lean_ctor_get(v___x_284_, 0);
v_isSharedCheck_297_ = !lean_is_exclusive(v___x_284_);
if (v_isSharedCheck_297_ == 0)
{
v___x_292_ = v___x_284_;
v_isShared_293_ = v_isSharedCheck_297_;
goto v_resetjp_291_;
}
else
{
lean_inc(v_a_290_);
lean_dec(v___x_284_);
v___x_292_ = lean_box(0);
v_isShared_293_ = v_isSharedCheck_297_;
goto v_resetjp_291_;
}
v_resetjp_291_:
{
lean_object* v___x_295_; 
if (v_isShared_293_ == 0)
{
v___x_295_ = v___x_292_;
goto v_reusejp_294_;
}
else
{
lean_object* v_reuseFailAlloc_296_; 
v_reuseFailAlloc_296_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_296_, 0, v_a_290_);
v___x_295_ = v_reuseFailAlloc_296_;
goto v_reusejp_294_;
}
v_reusejp_294_:
{
return v___x_295_;
}
}
}
}
else
{
lean_object* v_a_298_; lean_object* v___x_300_; uint8_t v_isShared_301_; uint8_t v_isSharedCheck_305_; 
lean_dec_ref(v_input_113_);
v_a_298_ = lean_ctor_get(v___x_283_, 0);
v_isSharedCheck_305_ = !lean_is_exclusive(v___x_283_);
if (v_isSharedCheck_305_ == 0)
{
v___x_300_ = v___x_283_;
v_isShared_301_ = v_isSharedCheck_305_;
goto v_resetjp_299_;
}
else
{
lean_inc(v_a_298_);
lean_dec(v___x_283_);
v___x_300_ = lean_box(0);
v_isShared_301_ = v_isSharedCheck_305_;
goto v_resetjp_299_;
}
v_resetjp_299_:
{
lean_object* v___x_303_; 
if (v_isShared_301_ == 0)
{
v___x_303_ = v___x_300_;
goto v_reusejp_302_;
}
else
{
lean_object* v_reuseFailAlloc_304_; 
v_reuseFailAlloc_304_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_304_, 0, v_a_298_);
v___x_303_ = v_reuseFailAlloc_304_;
goto v_reusejp_302_;
}
v_reusejp_302_:
{
return v___x_303_;
}
}
}
}
else
{
lean_dec(v_a_278_);
lean_dec_ref(v_input_113_);
return v___x_279_;
}
}
}
}
else
{
lean_object* v_a_308_; lean_object* v___x_310_; uint8_t v_isShared_311_; uint8_t v_isSharedCheck_315_; 
lean_dec_ref(v_input_113_);
v_a_308_ = lean_ctor_get(v___x_277_, 0);
v_isSharedCheck_315_ = !lean_is_exclusive(v___x_277_);
if (v_isSharedCheck_315_ == 0)
{
v___x_310_ = v___x_277_;
v_isShared_311_ = v_isSharedCheck_315_;
goto v_resetjp_309_;
}
else
{
lean_inc(v_a_308_);
lean_dec(v___x_277_);
v___x_310_ = lean_box(0);
v_isShared_311_ = v_isSharedCheck_315_;
goto v_resetjp_309_;
}
v_resetjp_309_:
{
lean_object* v___x_313_; 
if (v_isShared_311_ == 0)
{
v___x_313_ = v___x_310_;
goto v_reusejp_312_;
}
else
{
lean_object* v_reuseFailAlloc_314_; 
v_reuseFailAlloc_314_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_314_, 0, v_a_308_);
v___x_313_ = v_reuseFailAlloc_314_;
goto v_reusejp_312_;
}
v_reusejp_312_:
{
return v___x_313_;
}
}
}
}
case 1:
{
lean_object* v___x_316_; 
v___x_316_ = l_Lean_Meta_saveState___redArg(v_a_118_, v_a_120_);
if (lean_obj_tag(v___x_316_) == 0)
{
lean_object* v_a_317_; lean_object* v___x_318_; 
v_a_317_ = lean_ctor_get(v___x_316_, 0);
lean_inc(v_a_317_);
lean_dec_ref_known(v___x_316_, 1);
lean_inc_ref(v_input_113_);
v___x_318_ = lp_aesop_Aesop_RuleBuilder_constructors___redArg(v_input_113_, v_a_115_, v_a_116_, v_a_117_, v_a_118_, v_a_119_, v_a_120_);
if (lean_obj_tag(v___x_318_) == 0)
{
lean_dec(v_a_317_);
lean_dec_ref(v_input_113_);
return v___x_318_;
}
else
{
lean_object* v_a_319_; uint8_t v___y_321_; uint8_t v___x_345_; 
v_a_319_ = lean_ctor_get(v___x_318_, 0);
lean_inc(v_a_319_);
v___x_345_ = l_Lean_Exception_isInterrupt(v_a_319_);
if (v___x_345_ == 0)
{
uint8_t v___x_346_; 
v___x_346_ = l_Lean_Exception_isRuntime(v_a_319_);
v___y_321_ = v___x_346_;
goto v___jp_320_;
}
else
{
lean_dec(v_a_319_);
v___y_321_ = v___x_345_;
goto v___jp_320_;
}
v___jp_320_:
{
if (v___y_321_ == 0)
{
lean_object* v___x_322_; 
lean_dec_ref_known(v___x_318_, 1);
v___x_322_ = l_Lean_Meta_SavedState_restore___redArg(v_a_317_, v_a_118_, v_a_120_);
lean_dec(v_a_317_);
if (lean_obj_tag(v___x_322_) == 0)
{
lean_object* v___x_323_; 
lean_dec_ref_known(v___x_322_, 1);
v___x_323_ = l_Lean_Meta_saveState___redArg(v_a_118_, v_a_120_);
if (lean_obj_tag(v___x_323_) == 0)
{
lean_object* v_a_324_; lean_object* v___x_325_; 
v_a_324_ = lean_ctor_get(v___x_323_, 0);
lean_inc(v_a_324_);
lean_dec_ref_known(v___x_323_, 1);
lean_inc_ref(v_input_113_);
v___x_325_ = lp_aesop_Aesop_RuleBuilder_tactic(v_input_113_, v_a_114_, v_a_115_, v_a_116_, v_a_117_, v_a_118_, v_a_119_, v_a_120_);
if (lean_obj_tag(v___x_325_) == 0)
{
lean_dec(v_a_324_);
lean_dec_ref(v_input_113_);
return v___x_325_;
}
else
{
lean_object* v_a_326_; uint8_t v___x_327_; 
v_a_326_ = lean_ctor_get(v___x_325_, 0);
lean_inc(v_a_326_);
v___x_327_ = l_Lean_Exception_isInterrupt(v_a_326_);
if (v___x_327_ == 0)
{
uint8_t v___x_328_; 
v___x_328_ = l_Lean_Exception_isRuntime(v_a_326_);
v___y_249_ = v_a_324_;
v___y_250_ = v___x_325_;
v___y_251_ = v___x_328_;
goto v___jp_248_;
}
else
{
lean_dec(v_a_326_);
v___y_249_ = v_a_324_;
v___y_250_ = v___x_325_;
v___y_251_ = v___x_327_;
goto v___jp_248_;
}
}
}
else
{
lean_object* v_a_329_; lean_object* v___x_331_; uint8_t v_isShared_332_; uint8_t v_isSharedCheck_336_; 
lean_dec_ref(v_input_113_);
v_a_329_ = lean_ctor_get(v___x_323_, 0);
v_isSharedCheck_336_ = !lean_is_exclusive(v___x_323_);
if (v_isSharedCheck_336_ == 0)
{
v___x_331_ = v___x_323_;
v_isShared_332_ = v_isSharedCheck_336_;
goto v_resetjp_330_;
}
else
{
lean_inc(v_a_329_);
lean_dec(v___x_323_);
v___x_331_ = lean_box(0);
v_isShared_332_ = v_isSharedCheck_336_;
goto v_resetjp_330_;
}
v_resetjp_330_:
{
lean_object* v___x_334_; 
if (v_isShared_332_ == 0)
{
v___x_334_ = v___x_331_;
goto v_reusejp_333_;
}
else
{
lean_object* v_reuseFailAlloc_335_; 
v_reuseFailAlloc_335_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_335_, 0, v_a_329_);
v___x_334_ = v_reuseFailAlloc_335_;
goto v_reusejp_333_;
}
v_reusejp_333_:
{
return v___x_334_;
}
}
}
}
else
{
lean_object* v_a_337_; lean_object* v___x_339_; uint8_t v_isShared_340_; uint8_t v_isSharedCheck_344_; 
lean_dec_ref(v_input_113_);
v_a_337_ = lean_ctor_get(v___x_322_, 0);
v_isSharedCheck_344_ = !lean_is_exclusive(v___x_322_);
if (v_isSharedCheck_344_ == 0)
{
v___x_339_ = v___x_322_;
v_isShared_340_ = v_isSharedCheck_344_;
goto v_resetjp_338_;
}
else
{
lean_inc(v_a_337_);
lean_dec(v___x_322_);
v___x_339_ = lean_box(0);
v_isShared_340_ = v_isSharedCheck_344_;
goto v_resetjp_338_;
}
v_resetjp_338_:
{
lean_object* v___x_342_; 
if (v_isShared_340_ == 0)
{
v___x_342_ = v___x_339_;
goto v_reusejp_341_;
}
else
{
lean_object* v_reuseFailAlloc_343_; 
v_reuseFailAlloc_343_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_343_, 0, v_a_337_);
v___x_342_ = v_reuseFailAlloc_343_;
goto v_reusejp_341_;
}
v_reusejp_341_:
{
return v___x_342_;
}
}
}
}
else
{
lean_dec(v_a_317_);
lean_dec_ref(v_input_113_);
return v___x_318_;
}
}
}
}
else
{
lean_object* v_a_347_; lean_object* v___x_349_; uint8_t v_isShared_350_; uint8_t v_isSharedCheck_354_; 
lean_dec_ref(v_input_113_);
v_a_347_ = lean_ctor_get(v___x_316_, 0);
v_isSharedCheck_354_ = !lean_is_exclusive(v___x_316_);
if (v_isSharedCheck_354_ == 0)
{
v___x_349_ = v___x_316_;
v_isShared_350_ = v_isSharedCheck_354_;
goto v_resetjp_348_;
}
else
{
lean_inc(v_a_347_);
lean_dec(v___x_316_);
v___x_349_ = lean_box(0);
v_isShared_350_ = v_isSharedCheck_354_;
goto v_resetjp_348_;
}
v_resetjp_348_:
{
lean_object* v___x_352_; 
if (v_isShared_350_ == 0)
{
v___x_352_ = v___x_349_;
goto v_reusejp_351_;
}
else
{
lean_object* v_reuseFailAlloc_353_; 
v_reuseFailAlloc_353_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_353_, 0, v_a_347_);
v___x_352_ = v_reuseFailAlloc_353_;
goto v_reusejp_351_;
}
v_reusejp_351_:
{
return v___x_352_;
}
}
}
}
default: 
{
lean_object* v___x_355_; 
v___x_355_ = l_Lean_Meta_saveState___redArg(v_a_118_, v_a_120_);
if (lean_obj_tag(v___x_355_) == 0)
{
lean_object* v_a_356_; lean_object* v___x_357_; 
v_a_356_ = lean_ctor_get(v___x_355_, 0);
lean_inc(v_a_356_);
lean_dec_ref_known(v___x_355_, 1);
lean_inc_ref(v_input_113_);
v___x_357_ = lp_aesop_Aesop_RuleBuilder_constructors___redArg(v_input_113_, v_a_115_, v_a_116_, v_a_117_, v_a_118_, v_a_119_, v_a_120_);
if (lean_obj_tag(v___x_357_) == 0)
{
lean_dec(v_a_356_);
lean_dec_ref(v_input_113_);
return v___x_357_;
}
else
{
lean_object* v_a_358_; uint8_t v___y_360_; uint8_t v___x_384_; 
v_a_358_ = lean_ctor_get(v___x_357_, 0);
lean_inc(v_a_358_);
v___x_384_ = l_Lean_Exception_isInterrupt(v_a_358_);
if (v___x_384_ == 0)
{
uint8_t v___x_385_; 
v___x_385_ = l_Lean_Exception_isRuntime(v_a_358_);
v___y_360_ = v___x_385_;
goto v___jp_359_;
}
else
{
lean_dec(v_a_358_);
v___y_360_ = v___x_384_;
goto v___jp_359_;
}
v___jp_359_:
{
if (v___y_360_ == 0)
{
lean_object* v___x_361_; 
lean_dec_ref_known(v___x_357_, 1);
v___x_361_ = l_Lean_Meta_SavedState_restore___redArg(v_a_356_, v_a_118_, v_a_120_);
lean_dec(v_a_356_);
if (lean_obj_tag(v___x_361_) == 0)
{
lean_object* v___x_362_; 
lean_dec_ref_known(v___x_361_, 1);
v___x_362_ = l_Lean_Meta_saveState___redArg(v_a_118_, v_a_120_);
if (lean_obj_tag(v___x_362_) == 0)
{
lean_object* v_a_363_; lean_object* v___x_364_; 
v_a_363_ = lean_ctor_get(v___x_362_, 0);
lean_inc(v_a_363_);
lean_dec_ref_known(v___x_362_, 1);
lean_inc_ref(v_input_113_);
v___x_364_ = lp_aesop_Aesop_RuleBuilder_tactic(v_input_113_, v_a_114_, v_a_115_, v_a_116_, v_a_117_, v_a_118_, v_a_119_, v_a_120_);
if (lean_obj_tag(v___x_364_) == 0)
{
lean_dec(v_a_363_);
lean_dec_ref(v_input_113_);
return v___x_364_;
}
else
{
lean_object* v_a_365_; uint8_t v___x_366_; 
v_a_365_ = lean_ctor_get(v___x_364_, 0);
lean_inc(v_a_365_);
v___x_366_ = l_Lean_Exception_isInterrupt(v_a_365_);
if (v___x_366_ == 0)
{
uint8_t v___x_367_; 
v___x_367_ = l_Lean_Exception_isRuntime(v_a_365_);
v___y_138_ = v___x_364_;
v___y_139_ = v_a_363_;
v___y_140_ = v___x_367_;
goto v___jp_137_;
}
else
{
lean_dec(v_a_365_);
v___y_138_ = v___x_364_;
v___y_139_ = v_a_363_;
v___y_140_ = v___x_366_;
goto v___jp_137_;
}
}
}
else
{
lean_object* v_a_368_; lean_object* v___x_370_; uint8_t v_isShared_371_; uint8_t v_isSharedCheck_375_; 
lean_dec_ref(v_input_113_);
v_a_368_ = lean_ctor_get(v___x_362_, 0);
v_isSharedCheck_375_ = !lean_is_exclusive(v___x_362_);
if (v_isSharedCheck_375_ == 0)
{
v___x_370_ = v___x_362_;
v_isShared_371_ = v_isSharedCheck_375_;
goto v_resetjp_369_;
}
else
{
lean_inc(v_a_368_);
lean_dec(v___x_362_);
v___x_370_ = lean_box(0);
v_isShared_371_ = v_isSharedCheck_375_;
goto v_resetjp_369_;
}
v_resetjp_369_:
{
lean_object* v___x_373_; 
if (v_isShared_371_ == 0)
{
v___x_373_ = v___x_370_;
goto v_reusejp_372_;
}
else
{
lean_object* v_reuseFailAlloc_374_; 
v_reuseFailAlloc_374_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_374_, 0, v_a_368_);
v___x_373_ = v_reuseFailAlloc_374_;
goto v_reusejp_372_;
}
v_reusejp_372_:
{
return v___x_373_;
}
}
}
}
else
{
lean_object* v_a_376_; lean_object* v___x_378_; uint8_t v_isShared_379_; uint8_t v_isSharedCheck_383_; 
lean_dec_ref(v_input_113_);
v_a_376_ = lean_ctor_get(v___x_361_, 0);
v_isSharedCheck_383_ = !lean_is_exclusive(v___x_361_);
if (v_isSharedCheck_383_ == 0)
{
v___x_378_ = v___x_361_;
v_isShared_379_ = v_isSharedCheck_383_;
goto v_resetjp_377_;
}
else
{
lean_inc(v_a_376_);
lean_dec(v___x_361_);
v___x_378_ = lean_box(0);
v_isShared_379_ = v_isSharedCheck_383_;
goto v_resetjp_377_;
}
v_resetjp_377_:
{
lean_object* v___x_381_; 
if (v_isShared_379_ == 0)
{
v___x_381_ = v___x_378_;
goto v_reusejp_380_;
}
else
{
lean_object* v_reuseFailAlloc_382_; 
v_reuseFailAlloc_382_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_382_, 0, v_a_376_);
v___x_381_ = v_reuseFailAlloc_382_;
goto v_reusejp_380_;
}
v_reusejp_380_:
{
return v___x_381_;
}
}
}
}
else
{
lean_dec(v_a_356_);
lean_dec_ref(v_input_113_);
return v___x_357_;
}
}
}
}
else
{
lean_object* v_a_386_; lean_object* v___x_388_; uint8_t v_isShared_389_; uint8_t v_isSharedCheck_393_; 
lean_dec_ref(v_input_113_);
v_a_386_ = lean_ctor_get(v___x_355_, 0);
v_isSharedCheck_393_ = !lean_is_exclusive(v___x_355_);
if (v_isSharedCheck_393_ == 0)
{
v___x_388_ = v___x_355_;
v_isShared_389_ = v_isSharedCheck_393_;
goto v_resetjp_387_;
}
else
{
lean_inc(v_a_386_);
lean_dec(v___x_355_);
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
v___jp_122_:
{
if (v___y_125_ == 0)
{
lean_object* v___x_126_; 
lean_dec_ref(v___y_124_);
v___x_126_ = l_Lean_Meta_SavedState_restore___redArg(v___y_123_, v_a_118_, v_a_120_);
lean_dec_ref(v___y_123_);
if (lean_obj_tag(v___x_126_) == 0)
{
lean_object* v___x_127_; lean_object* v___x_128_; 
lean_dec_ref_known(v___x_126_, 1);
v___x_127_ = ((lean_object*)(lp_aesop_Aesop_RuleBuilder_default___closed__0));
v___x_128_ = lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err(v___x_127_, v_input_113_, v_a_114_, v_a_115_, v_a_116_, v_a_117_, v_a_118_, v_a_119_, v_a_120_);
return v___x_128_;
}
else
{
lean_object* v_a_129_; lean_object* v___x_131_; uint8_t v_isShared_132_; uint8_t v_isSharedCheck_136_; 
lean_dec_ref(v_input_113_);
v_a_129_ = lean_ctor_get(v___x_126_, 0);
v_isSharedCheck_136_ = !lean_is_exclusive(v___x_126_);
if (v_isSharedCheck_136_ == 0)
{
v___x_131_ = v___x_126_;
v_isShared_132_ = v_isSharedCheck_136_;
goto v_resetjp_130_;
}
else
{
lean_inc(v_a_129_);
lean_dec(v___x_126_);
v___x_131_ = lean_box(0);
v_isShared_132_ = v_isSharedCheck_136_;
goto v_resetjp_130_;
}
v_resetjp_130_:
{
lean_object* v___x_134_; 
if (v_isShared_132_ == 0)
{
v___x_134_ = v___x_131_;
goto v_reusejp_133_;
}
else
{
lean_object* v_reuseFailAlloc_135_; 
v_reuseFailAlloc_135_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_135_, 0, v_a_129_);
v___x_134_ = v_reuseFailAlloc_135_;
goto v_reusejp_133_;
}
v_reusejp_133_:
{
return v___x_134_;
}
}
}
}
else
{
lean_dec_ref(v___y_123_);
lean_dec_ref(v_input_113_);
return v___y_124_;
}
}
v___jp_137_:
{
if (v___y_140_ == 0)
{
lean_object* v___x_141_; 
lean_dec_ref(v___y_138_);
v___x_141_ = l_Lean_Meta_SavedState_restore___redArg(v___y_139_, v_a_118_, v_a_120_);
lean_dec_ref(v___y_139_);
if (lean_obj_tag(v___x_141_) == 0)
{
lean_object* v___x_142_; 
lean_dec_ref_known(v___x_141_, 1);
v___x_142_ = l_Lean_Meta_saveState___redArg(v_a_118_, v_a_120_);
if (lean_obj_tag(v___x_142_) == 0)
{
lean_object* v_a_143_; lean_object* v___x_144_; 
v_a_143_ = lean_ctor_get(v___x_142_, 0);
lean_inc(v_a_143_);
lean_dec_ref_known(v___x_142_, 1);
lean_inc_ref(v_input_113_);
v___x_144_ = lp_aesop_Aesop_RuleBuilder_apply(v_input_113_, v_a_114_, v_a_115_, v_a_116_, v_a_117_, v_a_118_, v_a_119_, v_a_120_);
if (lean_obj_tag(v___x_144_) == 0)
{
lean_dec(v_a_143_);
lean_dec_ref(v_input_113_);
return v___x_144_;
}
else
{
lean_object* v_a_145_; uint8_t v___x_146_; 
v_a_145_ = lean_ctor_get(v___x_144_, 0);
lean_inc(v_a_145_);
v___x_146_ = l_Lean_Exception_isInterrupt(v_a_145_);
if (v___x_146_ == 0)
{
uint8_t v___x_147_; 
v___x_147_ = l_Lean_Exception_isRuntime(v_a_145_);
v___y_123_ = v_a_143_;
v___y_124_ = v___x_144_;
v___y_125_ = v___x_147_;
goto v___jp_122_;
}
else
{
lean_dec(v_a_145_);
v___y_123_ = v_a_143_;
v___y_124_ = v___x_144_;
v___y_125_ = v___x_146_;
goto v___jp_122_;
}
}
}
else
{
lean_object* v_a_148_; lean_object* v___x_150_; uint8_t v_isShared_151_; uint8_t v_isSharedCheck_155_; 
lean_dec_ref(v_input_113_);
v_a_148_ = lean_ctor_get(v___x_142_, 0);
v_isSharedCheck_155_ = !lean_is_exclusive(v___x_142_);
if (v_isSharedCheck_155_ == 0)
{
v___x_150_ = v___x_142_;
v_isShared_151_ = v_isSharedCheck_155_;
goto v_resetjp_149_;
}
else
{
lean_inc(v_a_148_);
lean_dec(v___x_142_);
v___x_150_ = lean_box(0);
v_isShared_151_ = v_isSharedCheck_155_;
goto v_resetjp_149_;
}
v_resetjp_149_:
{
lean_object* v___x_153_; 
if (v_isShared_151_ == 0)
{
v___x_153_ = v___x_150_;
goto v_reusejp_152_;
}
else
{
lean_object* v_reuseFailAlloc_154_; 
v_reuseFailAlloc_154_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_154_, 0, v_a_148_);
v___x_153_ = v_reuseFailAlloc_154_;
goto v_reusejp_152_;
}
v_reusejp_152_:
{
return v___x_153_;
}
}
}
}
else
{
lean_object* v_a_156_; lean_object* v___x_158_; uint8_t v_isShared_159_; uint8_t v_isSharedCheck_163_; 
lean_dec_ref(v_input_113_);
v_a_156_ = lean_ctor_get(v___x_141_, 0);
v_isSharedCheck_163_ = !lean_is_exclusive(v___x_141_);
if (v_isSharedCheck_163_ == 0)
{
v___x_158_ = v___x_141_;
v_isShared_159_ = v_isSharedCheck_163_;
goto v_resetjp_157_;
}
else
{
lean_inc(v_a_156_);
lean_dec(v___x_141_);
v___x_158_ = lean_box(0);
v_isShared_159_ = v_isSharedCheck_163_;
goto v_resetjp_157_;
}
v_resetjp_157_:
{
lean_object* v___x_161_; 
if (v_isShared_159_ == 0)
{
v___x_161_ = v___x_158_;
goto v_reusejp_160_;
}
else
{
lean_object* v_reuseFailAlloc_162_; 
v_reuseFailAlloc_162_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_162_, 0, v_a_156_);
v___x_161_ = v_reuseFailAlloc_162_;
goto v_reusejp_160_;
}
v_reusejp_160_:
{
return v___x_161_;
}
}
}
}
else
{
lean_dec_ref(v___y_139_);
lean_dec_ref(v_input_113_);
return v___y_138_;
}
}
v___jp_164_:
{
if (v___y_167_ == 0)
{
lean_object* v___x_168_; 
lean_dec_ref(v___y_165_);
v___x_168_ = l_Lean_Meta_SavedState_restore___redArg(v___y_166_, v_a_118_, v_a_120_);
lean_dec_ref(v___y_166_);
if (lean_obj_tag(v___x_168_) == 0)
{
lean_object* v___x_169_; lean_object* v___x_170_; 
lean_dec_ref_known(v___x_168_, 1);
v___x_169_ = ((lean_object*)(lp_aesop_Aesop_RuleBuilder_default___closed__1));
v___x_170_ = lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err(v___x_169_, v_input_113_, v_a_114_, v_a_115_, v_a_116_, v_a_117_, v_a_118_, v_a_119_, v_a_120_);
return v___x_170_;
}
else
{
lean_object* v_a_171_; lean_object* v___x_173_; uint8_t v_isShared_174_; uint8_t v_isSharedCheck_178_; 
lean_dec_ref(v_input_113_);
v_a_171_ = lean_ctor_get(v___x_168_, 0);
v_isSharedCheck_178_ = !lean_is_exclusive(v___x_168_);
if (v_isSharedCheck_178_ == 0)
{
v___x_173_ = v___x_168_;
v_isShared_174_ = v_isSharedCheck_178_;
goto v_resetjp_172_;
}
else
{
lean_inc(v_a_171_);
lean_dec(v___x_168_);
v___x_173_ = lean_box(0);
v_isShared_174_ = v_isSharedCheck_178_;
goto v_resetjp_172_;
}
v_resetjp_172_:
{
lean_object* v___x_176_; 
if (v_isShared_174_ == 0)
{
v___x_176_ = v___x_173_;
goto v_reusejp_175_;
}
else
{
lean_object* v_reuseFailAlloc_177_; 
v_reuseFailAlloc_177_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_177_, 0, v_a_171_);
v___x_176_ = v_reuseFailAlloc_177_;
goto v_reusejp_175_;
}
v_reusejp_175_:
{
return v___x_176_;
}
}
}
}
else
{
lean_dec_ref(v___y_166_);
lean_dec_ref(v_input_113_);
return v___y_165_;
}
}
v___jp_179_:
{
if (v___y_182_ == 0)
{
lean_object* v___x_183_; 
lean_dec_ref(v___y_180_);
v___x_183_ = l_Lean_Meta_SavedState_restore___redArg(v___y_181_, v_a_118_, v_a_120_);
lean_dec_ref(v___y_181_);
if (lean_obj_tag(v___x_183_) == 0)
{
lean_object* v___x_184_; 
lean_dec_ref_known(v___x_183_, 1);
v___x_184_ = l_Lean_Meta_saveState___redArg(v_a_118_, v_a_120_);
if (lean_obj_tag(v___x_184_) == 0)
{
lean_object* v_a_185_; lean_object* v___x_186_; 
v_a_185_ = lean_ctor_get(v___x_184_, 0);
lean_inc(v_a_185_);
lean_dec_ref_known(v___x_184_, 1);
lean_inc_ref(v_input_113_);
v___x_186_ = lp_aesop_Aesop_RuleBuilder_apply(v_input_113_, v_a_114_, v_a_115_, v_a_116_, v_a_117_, v_a_118_, v_a_119_, v_a_120_);
if (lean_obj_tag(v___x_186_) == 0)
{
lean_dec(v_a_185_);
lean_dec_ref(v_input_113_);
return v___x_186_;
}
else
{
lean_object* v_a_187_; uint8_t v___x_188_; 
v_a_187_ = lean_ctor_get(v___x_186_, 0);
lean_inc(v_a_187_);
v___x_188_ = l_Lean_Exception_isInterrupt(v_a_187_);
if (v___x_188_ == 0)
{
uint8_t v___x_189_; 
v___x_189_ = l_Lean_Exception_isRuntime(v_a_187_);
v___y_165_ = v___x_186_;
v___y_166_ = v_a_185_;
v___y_167_ = v___x_189_;
goto v___jp_164_;
}
else
{
lean_dec(v_a_187_);
v___y_165_ = v___x_186_;
v___y_166_ = v_a_185_;
v___y_167_ = v___x_188_;
goto v___jp_164_;
}
}
}
else
{
lean_object* v_a_190_; lean_object* v___x_192_; uint8_t v_isShared_193_; uint8_t v_isSharedCheck_197_; 
lean_dec_ref(v_input_113_);
v_a_190_ = lean_ctor_get(v___x_184_, 0);
v_isSharedCheck_197_ = !lean_is_exclusive(v___x_184_);
if (v_isSharedCheck_197_ == 0)
{
v___x_192_ = v___x_184_;
v_isShared_193_ = v_isSharedCheck_197_;
goto v_resetjp_191_;
}
else
{
lean_inc(v_a_190_);
lean_dec(v___x_184_);
v___x_192_ = lean_box(0);
v_isShared_193_ = v_isSharedCheck_197_;
goto v_resetjp_191_;
}
v_resetjp_191_:
{
lean_object* v___x_195_; 
if (v_isShared_193_ == 0)
{
v___x_195_ = v___x_192_;
goto v_reusejp_194_;
}
else
{
lean_object* v_reuseFailAlloc_196_; 
v_reuseFailAlloc_196_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_196_, 0, v_a_190_);
v___x_195_ = v_reuseFailAlloc_196_;
goto v_reusejp_194_;
}
v_reusejp_194_:
{
return v___x_195_;
}
}
}
}
else
{
lean_object* v_a_198_; lean_object* v___x_200_; uint8_t v_isShared_201_; uint8_t v_isSharedCheck_205_; 
lean_dec_ref(v_input_113_);
v_a_198_ = lean_ctor_get(v___x_183_, 0);
v_isSharedCheck_205_ = !lean_is_exclusive(v___x_183_);
if (v_isSharedCheck_205_ == 0)
{
v___x_200_ = v___x_183_;
v_isShared_201_ = v_isSharedCheck_205_;
goto v_resetjp_199_;
}
else
{
lean_inc(v_a_198_);
lean_dec(v___x_183_);
v___x_200_ = lean_box(0);
v_isShared_201_ = v_isSharedCheck_205_;
goto v_resetjp_199_;
}
v_resetjp_199_:
{
lean_object* v___x_203_; 
if (v_isShared_201_ == 0)
{
v___x_203_ = v___x_200_;
goto v_reusejp_202_;
}
else
{
lean_object* v_reuseFailAlloc_204_; 
v_reuseFailAlloc_204_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_204_, 0, v_a_198_);
v___x_203_ = v_reuseFailAlloc_204_;
goto v_reusejp_202_;
}
v_reusejp_202_:
{
return v___x_203_;
}
}
}
}
else
{
lean_dec_ref(v___y_181_);
lean_dec_ref(v_input_113_);
return v___y_180_;
}
}
v___jp_206_:
{
if (v___y_209_ == 0)
{
lean_object* v___x_210_; 
lean_dec_ref(v___y_208_);
v___x_210_ = l_Lean_Meta_SavedState_restore___redArg(v___y_207_, v_a_118_, v_a_120_);
lean_dec_ref(v___y_207_);
if (lean_obj_tag(v___x_210_) == 0)
{
lean_object* v___x_211_; 
lean_dec_ref_known(v___x_210_, 1);
v___x_211_ = l_Lean_Meta_saveState___redArg(v_a_118_, v_a_120_);
if (lean_obj_tag(v___x_211_) == 0)
{
lean_object* v_a_212_; lean_object* v___x_213_; 
v_a_212_ = lean_ctor_get(v___x_211_, 0);
lean_inc(v_a_212_);
lean_dec_ref_known(v___x_211_, 1);
lean_inc_ref(v_input_113_);
v___x_213_ = lp_aesop_Aesop_RuleBuilder_simp(v_input_113_, v_a_114_, v_a_115_, v_a_116_, v_a_117_, v_a_118_, v_a_119_, v_a_120_);
if (lean_obj_tag(v___x_213_) == 0)
{
lean_dec(v_a_212_);
lean_dec_ref(v_input_113_);
return v___x_213_;
}
else
{
lean_object* v_a_214_; uint8_t v___x_215_; 
v_a_214_ = lean_ctor_get(v___x_213_, 0);
lean_inc(v_a_214_);
v___x_215_ = l_Lean_Exception_isInterrupt(v_a_214_);
if (v___x_215_ == 0)
{
uint8_t v___x_216_; 
v___x_216_ = l_Lean_Exception_isRuntime(v_a_214_);
v___y_180_ = v___x_213_;
v___y_181_ = v_a_212_;
v___y_182_ = v___x_216_;
goto v___jp_179_;
}
else
{
lean_dec(v_a_214_);
v___y_180_ = v___x_213_;
v___y_181_ = v_a_212_;
v___y_182_ = v___x_215_;
goto v___jp_179_;
}
}
}
else
{
lean_object* v_a_217_; lean_object* v___x_219_; uint8_t v_isShared_220_; uint8_t v_isSharedCheck_224_; 
lean_dec_ref(v_input_113_);
v_a_217_ = lean_ctor_get(v___x_211_, 0);
v_isSharedCheck_224_ = !lean_is_exclusive(v___x_211_);
if (v_isSharedCheck_224_ == 0)
{
v___x_219_ = v___x_211_;
v_isShared_220_ = v_isSharedCheck_224_;
goto v_resetjp_218_;
}
else
{
lean_inc(v_a_217_);
lean_dec(v___x_211_);
v___x_219_ = lean_box(0);
v_isShared_220_ = v_isSharedCheck_224_;
goto v_resetjp_218_;
}
v_resetjp_218_:
{
lean_object* v___x_222_; 
if (v_isShared_220_ == 0)
{
v___x_222_ = v___x_219_;
goto v_reusejp_221_;
}
else
{
lean_object* v_reuseFailAlloc_223_; 
v_reuseFailAlloc_223_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_223_, 0, v_a_217_);
v___x_222_ = v_reuseFailAlloc_223_;
goto v_reusejp_221_;
}
v_reusejp_221_:
{
return v___x_222_;
}
}
}
}
else
{
lean_object* v_a_225_; lean_object* v___x_227_; uint8_t v_isShared_228_; uint8_t v_isSharedCheck_232_; 
lean_dec_ref(v_input_113_);
v_a_225_ = lean_ctor_get(v___x_210_, 0);
v_isSharedCheck_232_ = !lean_is_exclusive(v___x_210_);
if (v_isSharedCheck_232_ == 0)
{
v___x_227_ = v___x_210_;
v_isShared_228_ = v_isSharedCheck_232_;
goto v_resetjp_226_;
}
else
{
lean_inc(v_a_225_);
lean_dec(v___x_210_);
v___x_227_ = lean_box(0);
v_isShared_228_ = v_isSharedCheck_232_;
goto v_resetjp_226_;
}
v_resetjp_226_:
{
lean_object* v___x_230_; 
if (v_isShared_228_ == 0)
{
v___x_230_ = v___x_227_;
goto v_reusejp_229_;
}
else
{
lean_object* v_reuseFailAlloc_231_; 
v_reuseFailAlloc_231_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_231_, 0, v_a_225_);
v___x_230_ = v_reuseFailAlloc_231_;
goto v_reusejp_229_;
}
v_reusejp_229_:
{
return v___x_230_;
}
}
}
}
else
{
lean_dec_ref(v___y_207_);
lean_dec_ref(v_input_113_);
return v___y_208_;
}
}
v___jp_233_:
{
if (v___y_236_ == 0)
{
lean_object* v___x_237_; 
lean_dec_ref(v___y_235_);
v___x_237_ = l_Lean_Meta_SavedState_restore___redArg(v___y_234_, v_a_118_, v_a_120_);
lean_dec_ref(v___y_234_);
if (lean_obj_tag(v___x_237_) == 0)
{
lean_object* v___x_238_; lean_object* v___x_239_; 
lean_dec_ref_known(v___x_237_, 1);
v___x_238_ = ((lean_object*)(lp_aesop_Aesop_RuleBuilder_default___closed__2));
v___x_239_ = lp_aesop___private_Aesop_Builder_Default_0__Aesop_RuleBuilder_default_err(v___x_238_, v_input_113_, v_a_114_, v_a_115_, v_a_116_, v_a_117_, v_a_118_, v_a_119_, v_a_120_);
return v___x_239_;
}
else
{
lean_object* v_a_240_; lean_object* v___x_242_; uint8_t v_isShared_243_; uint8_t v_isSharedCheck_247_; 
lean_dec_ref(v_input_113_);
v_a_240_ = lean_ctor_get(v___x_237_, 0);
v_isSharedCheck_247_ = !lean_is_exclusive(v___x_237_);
if (v_isSharedCheck_247_ == 0)
{
v___x_242_ = v___x_237_;
v_isShared_243_ = v_isSharedCheck_247_;
goto v_resetjp_241_;
}
else
{
lean_inc(v_a_240_);
lean_dec(v___x_237_);
v___x_242_ = lean_box(0);
v_isShared_243_ = v_isSharedCheck_247_;
goto v_resetjp_241_;
}
v_resetjp_241_:
{
lean_object* v___x_245_; 
if (v_isShared_243_ == 0)
{
v___x_245_ = v___x_242_;
goto v_reusejp_244_;
}
else
{
lean_object* v_reuseFailAlloc_246_; 
v_reuseFailAlloc_246_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_246_, 0, v_a_240_);
v___x_245_ = v_reuseFailAlloc_246_;
goto v_reusejp_244_;
}
v_reusejp_244_:
{
return v___x_245_;
}
}
}
}
else
{
lean_dec_ref(v___y_234_);
lean_dec_ref(v_input_113_);
return v___y_235_;
}
}
v___jp_248_:
{
if (v___y_251_ == 0)
{
lean_object* v___x_252_; 
lean_dec_ref(v___y_250_);
v___x_252_ = l_Lean_Meta_SavedState_restore___redArg(v___y_249_, v_a_118_, v_a_120_);
lean_dec_ref(v___y_249_);
if (lean_obj_tag(v___x_252_) == 0)
{
lean_object* v___x_253_; 
lean_dec_ref_known(v___x_252_, 1);
v___x_253_ = l_Lean_Meta_saveState___redArg(v_a_118_, v_a_120_);
if (lean_obj_tag(v___x_253_) == 0)
{
lean_object* v_a_254_; lean_object* v___x_255_; 
v_a_254_ = lean_ctor_get(v___x_253_, 0);
lean_inc(v_a_254_);
lean_dec_ref_known(v___x_253_, 1);
lean_inc_ref(v_input_113_);
v___x_255_ = lp_aesop_Aesop_RuleBuilder_apply(v_input_113_, v_a_114_, v_a_115_, v_a_116_, v_a_117_, v_a_118_, v_a_119_, v_a_120_);
if (lean_obj_tag(v___x_255_) == 0)
{
lean_dec(v_a_254_);
lean_dec_ref(v_input_113_);
return v___x_255_;
}
else
{
lean_object* v_a_256_; uint8_t v___x_257_; 
v_a_256_ = lean_ctor_get(v___x_255_, 0);
lean_inc(v_a_256_);
v___x_257_ = l_Lean_Exception_isInterrupt(v_a_256_);
if (v___x_257_ == 0)
{
uint8_t v___x_258_; 
v___x_258_ = l_Lean_Exception_isRuntime(v_a_256_);
v___y_234_ = v_a_254_;
v___y_235_ = v___x_255_;
v___y_236_ = v___x_258_;
goto v___jp_233_;
}
else
{
lean_dec(v_a_256_);
v___y_234_ = v_a_254_;
v___y_235_ = v___x_255_;
v___y_236_ = v___x_257_;
goto v___jp_233_;
}
}
}
else
{
lean_object* v_a_259_; lean_object* v___x_261_; uint8_t v_isShared_262_; uint8_t v_isSharedCheck_266_; 
lean_dec_ref(v_input_113_);
v_a_259_ = lean_ctor_get(v___x_253_, 0);
v_isSharedCheck_266_ = !lean_is_exclusive(v___x_253_);
if (v_isSharedCheck_266_ == 0)
{
v___x_261_ = v___x_253_;
v_isShared_262_ = v_isSharedCheck_266_;
goto v_resetjp_260_;
}
else
{
lean_inc(v_a_259_);
lean_dec(v___x_253_);
v___x_261_ = lean_box(0);
v_isShared_262_ = v_isSharedCheck_266_;
goto v_resetjp_260_;
}
v_resetjp_260_:
{
lean_object* v___x_264_; 
if (v_isShared_262_ == 0)
{
v___x_264_ = v___x_261_;
goto v_reusejp_263_;
}
else
{
lean_object* v_reuseFailAlloc_265_; 
v_reuseFailAlloc_265_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_265_, 0, v_a_259_);
v___x_264_ = v_reuseFailAlloc_265_;
goto v_reusejp_263_;
}
v_reusejp_263_:
{
return v___x_264_;
}
}
}
}
else
{
lean_object* v_a_267_; lean_object* v___x_269_; uint8_t v_isShared_270_; uint8_t v_isSharedCheck_274_; 
lean_dec_ref(v_input_113_);
v_a_267_ = lean_ctor_get(v___x_252_, 0);
v_isSharedCheck_274_ = !lean_is_exclusive(v___x_252_);
if (v_isSharedCheck_274_ == 0)
{
v___x_269_ = v___x_252_;
v_isShared_270_ = v_isSharedCheck_274_;
goto v_resetjp_268_;
}
else
{
lean_inc(v_a_267_);
lean_dec(v___x_252_);
v___x_269_ = lean_box(0);
v_isShared_270_ = v_isSharedCheck_274_;
goto v_resetjp_268_;
}
v_resetjp_268_:
{
lean_object* v___x_272_; 
if (v_isShared_270_ == 0)
{
v___x_272_ = v___x_269_;
goto v_reusejp_271_;
}
else
{
lean_object* v_reuseFailAlloc_273_; 
v_reuseFailAlloc_273_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_273_, 0, v_a_267_);
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
else
{
lean_dec_ref(v___y_249_);
lean_dec_ref(v_input_113_);
return v___y_250_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleBuilder_default___boxed(lean_object* v_input_394_, lean_object* v_a_395_, lean_object* v_a_396_, lean_object* v_a_397_, lean_object* v_a_398_, lean_object* v_a_399_, lean_object* v_a_400_, lean_object* v_a_401_, lean_object* v_a_402_){
_start:
{
lean_object* v_res_403_; 
v_res_403_ = lp_aesop_Aesop_RuleBuilder_default(v_input_394_, v_a_395_, v_a_396_, v_a_397_, v_a_398_, v_a_399_, v_a_400_, v_a_401_);
lean_dec(v_a_401_);
lean_dec_ref(v_a_400_);
lean_dec(v_a_399_);
lean_dec_ref(v_a_398_);
lean_dec(v_a_397_);
lean_dec_ref(v_a_396_);
lean_dec_ref(v_a_395_);
return v_res_403_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Builder_Constructors(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Builder_NormSimp(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Builder_Tactic(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Builder_Apply(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Builder_Default(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Builder_Constructors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Builder_NormSimp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Builder_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Builder_Apply(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Builder_Default(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Builder_Constructors(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Builder_NormSimp(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Builder_Tactic(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Builder_Apply(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Builder_Default(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Builder_Constructors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Builder_NormSimp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Builder_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Builder_Apply(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Builder_Default(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Builder_Default(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Builder_Default(builtin);
}
#ifdef __cplusplus
}
#endif
