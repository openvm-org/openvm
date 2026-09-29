// Lean compiler output
// Module: Qq.ForLean.Do
// Imports: public import Init public meta import Init public import Lean.Elab.Do.Legacy
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
lean_object* l_Lean_Meta_getDecLevel(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* l_Lean_Meta_unfoldDefinition_x3f(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
uint8_t l_Lean_Expr_isMVar(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
static const lean_string_object lp_Qq_Lean_Elab_Term_mkIdBindFor___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Id"};
static const lean_object* lp_Qq_Lean_Elab_Term_mkIdBindFor___redArg___closed__0 = (const lean_object*)&lp_Qq_Lean_Elab_Term_mkIdBindFor___redArg___closed__0_value;
static const lean_ctor_object lp_Qq_Lean_Elab_Term_mkIdBindFor___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Lean_Elab_Term_mkIdBindFor___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(243, 233, 154, 254, 135, 150, 160, 105)}};
static const lean_object* lp_Qq_Lean_Elab_Term_mkIdBindFor___redArg___closed__1 = (const lean_object*)&lp_Qq_Lean_Elab_Term_mkIdBindFor___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_Term_mkIdBindFor___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_Term_mkIdBindFor___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_Term_mkIdBindFor(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_Term_mkIdBindFor___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00__private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00__private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq___private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "invalid 'do' notation, expected type is not available"};
static const lean_object* lp_Qq___private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f___closed__0 = (const lean_object*)&lp_Qq___private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f___closed__0_value;
static lean_once_cell_t lp_Qq___private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq___private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f___closed__1;
LEAN_EXPORT lean_object* lp_Qq___private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00__private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00__private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_Term_extractBind___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_Term_extractBind___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_Qq_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2___closed__0;
static const lean_string_object lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2___closed__1 = (const lean_object*)&lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2___closed__1_value;
static const lean_ctor_object lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2___closed__1_value)}};
static const lean_object* lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2___closed__2 = (const lean_object*)&lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2___closed__2_value;
static lean_once_cell_t lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2___closed__3;
LEAN_EXPORT lean_object* lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2(lean_object*, lean_object*);
static const lean_string_object lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0___redArg___closed__0 = (const lean_object*)&lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0___redArg___closed__0_value)}};
static const lean_object* lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0___redArg___closed__1 = (const lean_object*)&lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0___redArg___closed__1_value;
static lean_once_cell_t lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0___redArg___closed__2;
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_Term_extractBind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_Term_extractBind___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_Term_mkIdBindFor___redArg(lean_object* v_type_4_, lean_object* v_a_5_, lean_object* v_a_6_, lean_object* v_a_7_, lean_object* v_a_8_){
_start:
{
lean_object* v___x_10_; 
lean_inc_ref(v_type_4_);
v___x_10_ = l_Lean_Meta_getDecLevel(v_type_4_, v_a_5_, v_a_6_, v_a_7_, v_a_8_);
if (lean_obj_tag(v___x_10_) == 0)
{
lean_object* v_a_11_; lean_object* v___x_13_; uint8_t v_isShared_14_; uint8_t v_isSharedCheck_24_; 
v_a_11_ = lean_ctor_get(v___x_10_, 0);
v_isSharedCheck_24_ = !lean_is_exclusive(v___x_10_);
if (v_isSharedCheck_24_ == 0)
{
v___x_13_ = v___x_10_;
v_isShared_14_ = v_isSharedCheck_24_;
goto v_resetjp_12_;
}
else
{
lean_inc(v_a_11_);
lean_dec(v___x_10_);
v___x_13_ = lean_box(0);
v_isShared_14_ = v_isSharedCheck_24_;
goto v_resetjp_12_;
}
v_resetjp_12_:
{
lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_22_; 
v___x_15_ = ((lean_object*)(lp_Qq_Lean_Elab_Term_mkIdBindFor___redArg___closed__1));
v___x_16_ = lean_box(0);
v___x_17_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_17_, 0, v_a_11_);
lean_ctor_set(v___x_17_, 1, v___x_16_);
v___x_18_ = l_Lean_mkConst(v___x_15_, v___x_17_);
lean_inc_ref(v_type_4_);
lean_inc_ref(v___x_18_);
v___x_19_ = l_Lean_Expr_app___override(v___x_18_, v_type_4_);
v___x_20_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_20_, 0, v___x_18_);
lean_ctor_set(v___x_20_, 1, v_type_4_);
lean_ctor_set(v___x_20_, 2, v___x_19_);
if (v_isShared_14_ == 0)
{
lean_ctor_set(v___x_13_, 0, v___x_20_);
v___x_22_ = v___x_13_;
goto v_reusejp_21_;
}
else
{
lean_object* v_reuseFailAlloc_23_; 
v_reuseFailAlloc_23_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_23_, 0, v___x_20_);
v___x_22_ = v_reuseFailAlloc_23_;
goto v_reusejp_21_;
}
v_reusejp_21_:
{
return v___x_22_;
}
}
}
else
{
lean_object* v_a_25_; lean_object* v___x_27_; uint8_t v_isShared_28_; uint8_t v_isSharedCheck_32_; 
lean_dec_ref(v_type_4_);
v_a_25_ = lean_ctor_get(v___x_10_, 0);
v_isSharedCheck_32_ = !lean_is_exclusive(v___x_10_);
if (v_isSharedCheck_32_ == 0)
{
v___x_27_ = v___x_10_;
v_isShared_28_ = v_isSharedCheck_32_;
goto v_resetjp_26_;
}
else
{
lean_inc(v_a_25_);
lean_dec(v___x_10_);
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
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_Term_mkIdBindFor___redArg___boxed(lean_object* v_type_33_, lean_object* v_a_34_, lean_object* v_a_35_, lean_object* v_a_36_, lean_object* v_a_37_, lean_object* v_a_38_){
_start:
{
lean_object* v_res_39_; 
v_res_39_ = lp_Qq_Lean_Elab_Term_mkIdBindFor___redArg(v_type_33_, v_a_34_, v_a_35_, v_a_36_, v_a_37_);
lean_dec(v_a_37_);
lean_dec_ref(v_a_36_);
lean_dec(v_a_35_);
lean_dec_ref(v_a_34_);
return v_res_39_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_Term_mkIdBindFor(lean_object* v_type_40_, lean_object* v_a_41_, lean_object* v_a_42_, lean_object* v_a_43_, lean_object* v_a_44_, lean_object* v_a_45_, lean_object* v_a_46_){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lp_Qq_Lean_Elab_Term_mkIdBindFor___redArg(v_type_40_, v_a_43_, v_a_44_, v_a_45_, v_a_46_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_Term_mkIdBindFor___boxed(lean_object* v_type_49_, lean_object* v_a_50_, lean_object* v_a_51_, lean_object* v_a_52_, lean_object* v_a_53_, lean_object* v_a_54_, lean_object* v_a_55_, lean_object* v_a_56_){
_start:
{
lean_object* v_res_57_; 
v_res_57_ = lp_Qq_Lean_Elab_Term_mkIdBindFor(v_type_49_, v_a_50_, v_a_51_, v_a_52_, v_a_53_, v_a_54_, v_a_55_);
lean_dec(v_a_55_);
lean_dec_ref(v_a_54_);
lean_dec(v_a_53_);
lean_dec_ref(v_a_52_);
lean_dec(v_a_51_);
lean_dec_ref(v_a_50_);
return v_res_57_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f_spec__0_spec__0(lean_object* v_msgData_58_, lean_object* v___y_59_, lean_object* v___y_60_, lean_object* v___y_61_, lean_object* v___y_62_){
_start:
{
lean_object* v___x_64_; lean_object* v_env_65_; lean_object* v___x_66_; lean_object* v_mctx_67_; lean_object* v_lctx_68_; lean_object* v_options_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; 
v___x_64_ = lean_st_ref_get(v___y_62_);
v_env_65_ = lean_ctor_get(v___x_64_, 0);
lean_inc_ref(v_env_65_);
lean_dec(v___x_64_);
v___x_66_ = lean_st_ref_get(v___y_60_);
v_mctx_67_ = lean_ctor_get(v___x_66_, 0);
lean_inc_ref(v_mctx_67_);
lean_dec(v___x_66_);
v_lctx_68_ = lean_ctor_get(v___y_59_, 2);
v_options_69_ = lean_ctor_get(v___y_61_, 2);
lean_inc_ref(v_options_69_);
lean_inc_ref(v_lctx_68_);
v___x_70_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_70_, 0, v_env_65_);
lean_ctor_set(v___x_70_, 1, v_mctx_67_);
lean_ctor_set(v___x_70_, 2, v_lctx_68_);
lean_ctor_set(v___x_70_, 3, v_options_69_);
v___x_71_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_71_, 0, v___x_70_);
lean_ctor_set(v___x_71_, 1, v_msgData_58_);
v___x_72_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_72_, 0, v___x_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f_spec__0_spec__0___boxed(lean_object* v_msgData_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_){
_start:
{
lean_object* v_res_79_; 
v_res_79_ = lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f_spec__0_spec__0(v_msgData_73_, v___y_74_, v___y_75_, v___y_76_, v___y_77_);
lean_dec(v___y_77_);
lean_dec_ref(v___y_76_);
lean_dec(v___y_75_);
lean_dec_ref(v___y_74_);
return v_res_79_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00__private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f_spec__0___redArg(lean_object* v_msg_80_, lean_object* v___y_81_, lean_object* v___y_82_, lean_object* v___y_83_, lean_object* v___y_84_){
_start:
{
lean_object* v_ref_86_; lean_object* v___x_87_; lean_object* v_a_88_; lean_object* v___x_90_; uint8_t v_isShared_91_; uint8_t v_isSharedCheck_96_; 
v_ref_86_ = lean_ctor_get(v___y_83_, 5);
v___x_87_ = lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f_spec__0_spec__0(v_msg_80_, v___y_81_, v___y_82_, v___y_83_, v___y_84_);
v_a_88_ = lean_ctor_get(v___x_87_, 0);
v_isSharedCheck_96_ = !lean_is_exclusive(v___x_87_);
if (v_isSharedCheck_96_ == 0)
{
v___x_90_ = v___x_87_;
v_isShared_91_ = v_isSharedCheck_96_;
goto v_resetjp_89_;
}
else
{
lean_inc(v_a_88_);
lean_dec(v___x_87_);
v___x_90_ = lean_box(0);
v_isShared_91_ = v_isSharedCheck_96_;
goto v_resetjp_89_;
}
v_resetjp_89_:
{
lean_object* v___x_92_; lean_object* v___x_94_; 
lean_inc(v_ref_86_);
v___x_92_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_92_, 0, v_ref_86_);
lean_ctor_set(v___x_92_, 1, v_a_88_);
if (v_isShared_91_ == 0)
{
lean_ctor_set_tag(v___x_90_, 1);
lean_ctor_set(v___x_90_, 0, v___x_92_);
v___x_94_ = v___x_90_;
goto v_reusejp_93_;
}
else
{
lean_object* v_reuseFailAlloc_95_; 
v_reuseFailAlloc_95_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_95_, 0, v___x_92_);
v___x_94_ = v_reuseFailAlloc_95_;
goto v_reusejp_93_;
}
v_reusejp_93_:
{
return v___x_94_;
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00__private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f_spec__0___redArg___boxed(lean_object* v_msg_97_, lean_object* v___y_98_, lean_object* v___y_99_, lean_object* v___y_100_, lean_object* v___y_101_, lean_object* v___y_102_){
_start:
{
lean_object* v_res_103_; 
v_res_103_ = lp_Qq_Lean_throwError___at___00__private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f_spec__0___redArg(v_msg_97_, v___y_98_, v___y_99_, v___y_100_, v___y_101_);
lean_dec(v___y_101_);
lean_dec_ref(v___y_100_);
lean_dec(v___y_99_);
lean_dec_ref(v___y_98_);
return v_res_103_;
}
}
static lean_object* _init_lp_Qq___private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f___closed__1(void){
_start:
{
lean_object* v___x_105_; lean_object* v___x_106_; 
v___x_105_ = ((lean_object*)(lp_Qq___private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f___closed__0));
v___x_106_ = l_Lean_stringToMessageData(v___x_105_);
return v___x_106_;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f(lean_object* v_extractStep_x3f_107_, lean_object* v_type_108_, lean_object* v_a_109_, lean_object* v_a_110_, lean_object* v_a_111_, lean_object* v_a_112_){
_start:
{
lean_object* v___x_114_; 
lean_inc_ref(v_extractStep_x3f_107_);
lean_inc(v_a_112_);
lean_inc_ref(v_a_111_);
lean_inc(v_a_110_);
lean_inc_ref(v_a_109_);
lean_inc_ref(v_type_108_);
v___x_114_ = lean_apply_6(v_extractStep_x3f_107_, v_type_108_, v_a_109_, v_a_110_, v_a_111_, v_a_112_, lean_box(0));
if (lean_obj_tag(v___x_114_) == 0)
{
lean_object* v_a_115_; 
v_a_115_ = lean_ctor_get(v___x_114_, 0);
lean_inc(v_a_115_);
if (lean_obj_tag(v_a_115_) == 0)
{
lean_object* v___x_116_; 
lean_dec_ref_known(v___x_114_, 1);
lean_inc_ref(v_type_108_);
v___x_116_ = l_Lean_Meta_whnfCore(v_type_108_, v_a_109_, v_a_110_, v_a_111_, v_a_112_);
if (lean_obj_tag(v___x_116_) == 0)
{
lean_object* v_a_117_; uint8_t v___x_118_; 
v_a_117_ = lean_ctor_get(v___x_116_, 0);
lean_inc(v_a_117_);
lean_dec_ref_known(v___x_116_, 1);
v___x_118_ = lean_expr_eqv(v_a_117_, v_type_108_);
lean_dec_ref(v_type_108_);
if (v___x_118_ == 0)
{
v_type_108_ = v_a_117_;
goto _start;
}
else
{
uint8_t v___x_120_; lean_object* v___y_122_; lean_object* v___y_123_; lean_object* v___y_124_; lean_object* v___y_125_; lean_object* v___x_145_; uint8_t v___x_146_; 
v___x_120_ = 0;
v___x_145_ = l_Lean_Expr_getAppFn(v_a_117_);
v___x_146_ = l_Lean_Expr_isMVar(v___x_145_);
lean_dec_ref(v___x_145_);
if (v___x_146_ == 0)
{
v___y_122_ = v_a_109_;
v___y_123_ = v_a_110_;
v___y_124_ = v_a_111_;
v___y_125_ = v_a_112_;
goto v___jp_121_;
}
else
{
lean_object* v___x_147_; lean_object* v___x_148_; 
v___x_147_ = lean_obj_once(&lp_Qq___private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f___closed__1, &lp_Qq___private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f___closed__1_once, _init_lp_Qq___private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f___closed__1);
v___x_148_ = lp_Qq_Lean_throwError___at___00__private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f_spec__0___redArg(v___x_147_, v_a_109_, v_a_110_, v_a_111_, v_a_112_);
if (lean_obj_tag(v___x_148_) == 0)
{
lean_dec_ref_known(v___x_148_, 1);
v___y_122_ = v_a_109_;
v___y_123_ = v_a_110_;
v___y_124_ = v_a_111_;
v___y_125_ = v_a_112_;
goto v___jp_121_;
}
else
{
lean_object* v_a_149_; lean_object* v___x_151_; uint8_t v_isShared_152_; uint8_t v_isSharedCheck_156_; 
lean_dec(v_a_117_);
lean_dec_ref(v_extractStep_x3f_107_);
v_a_149_ = lean_ctor_get(v___x_148_, 0);
v_isSharedCheck_156_ = !lean_is_exclusive(v___x_148_);
if (v_isSharedCheck_156_ == 0)
{
v___x_151_ = v___x_148_;
v_isShared_152_ = v_isSharedCheck_156_;
goto v_resetjp_150_;
}
else
{
lean_inc(v_a_149_);
lean_dec(v___x_148_);
v___x_151_ = lean_box(0);
v_isShared_152_ = v_isSharedCheck_156_;
goto v_resetjp_150_;
}
v_resetjp_150_:
{
lean_object* v___x_154_; 
if (v_isShared_152_ == 0)
{
v___x_154_ = v___x_151_;
goto v_reusejp_153_;
}
else
{
lean_object* v_reuseFailAlloc_155_; 
v_reuseFailAlloc_155_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_155_, 0, v_a_149_);
v___x_154_ = v_reuseFailAlloc_155_;
goto v_reusejp_153_;
}
v_reusejp_153_:
{
return v___x_154_;
}
}
}
}
v___jp_121_:
{
lean_object* v___x_126_; 
v___x_126_ = l_Lean_Meta_unfoldDefinition_x3f(v_a_117_, v___x_120_, v___y_122_, v___y_123_, v___y_124_, v___y_125_);
if (lean_obj_tag(v___x_126_) == 0)
{
lean_object* v_a_127_; lean_object* v___x_129_; uint8_t v_isShared_130_; uint8_t v_isSharedCheck_136_; 
v_a_127_ = lean_ctor_get(v___x_126_, 0);
v_isSharedCheck_136_ = !lean_is_exclusive(v___x_126_);
if (v_isSharedCheck_136_ == 0)
{
v___x_129_ = v___x_126_;
v_isShared_130_ = v_isSharedCheck_136_;
goto v_resetjp_128_;
}
else
{
lean_inc(v_a_127_);
lean_dec(v___x_126_);
v___x_129_ = lean_box(0);
v_isShared_130_ = v_isSharedCheck_136_;
goto v_resetjp_128_;
}
v_resetjp_128_:
{
if (lean_obj_tag(v_a_127_) == 0)
{
lean_object* v___x_132_; 
lean_dec_ref(v_extractStep_x3f_107_);
if (v_isShared_130_ == 0)
{
lean_ctor_set(v___x_129_, 0, v_a_115_);
v___x_132_ = v___x_129_;
goto v_reusejp_131_;
}
else
{
lean_object* v_reuseFailAlloc_133_; 
v_reuseFailAlloc_133_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_133_, 0, v_a_115_);
v___x_132_ = v_reuseFailAlloc_133_;
goto v_reusejp_131_;
}
v_reusejp_131_:
{
return v___x_132_;
}
}
else
{
lean_object* v_val_134_; 
lean_del_object(v___x_129_);
v_val_134_ = lean_ctor_get(v_a_127_, 0);
lean_inc(v_val_134_);
lean_dec_ref_known(v_a_127_, 1);
v_type_108_ = v_val_134_;
v_a_109_ = v___y_122_;
v_a_110_ = v___y_123_;
v_a_111_ = v___y_124_;
v_a_112_ = v___y_125_;
goto _start;
}
}
}
else
{
lean_object* v_a_137_; lean_object* v___x_139_; uint8_t v_isShared_140_; uint8_t v_isSharedCheck_144_; 
lean_dec_ref(v_extractStep_x3f_107_);
v_a_137_ = lean_ctor_get(v___x_126_, 0);
v_isSharedCheck_144_ = !lean_is_exclusive(v___x_126_);
if (v_isSharedCheck_144_ == 0)
{
v___x_139_ = v___x_126_;
v_isShared_140_ = v_isSharedCheck_144_;
goto v_resetjp_138_;
}
else
{
lean_inc(v_a_137_);
lean_dec(v___x_126_);
v___x_139_ = lean_box(0);
v_isShared_140_ = v_isSharedCheck_144_;
goto v_resetjp_138_;
}
v_resetjp_138_:
{
lean_object* v___x_142_; 
if (v_isShared_140_ == 0)
{
v___x_142_ = v___x_139_;
goto v_reusejp_141_;
}
else
{
lean_object* v_reuseFailAlloc_143_; 
v_reuseFailAlloc_143_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_143_, 0, v_a_137_);
v___x_142_ = v_reuseFailAlloc_143_;
goto v_reusejp_141_;
}
v_reusejp_141_:
{
return v___x_142_;
}
}
}
}
}
}
else
{
lean_object* v_a_157_; lean_object* v___x_159_; uint8_t v_isShared_160_; uint8_t v_isSharedCheck_164_; 
lean_dec_ref(v_type_108_);
lean_dec_ref(v_extractStep_x3f_107_);
v_a_157_ = lean_ctor_get(v___x_116_, 0);
v_isSharedCheck_164_ = !lean_is_exclusive(v___x_116_);
if (v_isSharedCheck_164_ == 0)
{
v___x_159_ = v___x_116_;
v_isShared_160_ = v_isSharedCheck_164_;
goto v_resetjp_158_;
}
else
{
lean_inc(v_a_157_);
lean_dec(v___x_116_);
v___x_159_ = lean_box(0);
v_isShared_160_ = v_isSharedCheck_164_;
goto v_resetjp_158_;
}
v_resetjp_158_:
{
lean_object* v___x_162_; 
if (v_isShared_160_ == 0)
{
v___x_162_ = v___x_159_;
goto v_reusejp_161_;
}
else
{
lean_object* v_reuseFailAlloc_163_; 
v_reuseFailAlloc_163_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_163_, 0, v_a_157_);
v___x_162_ = v_reuseFailAlloc_163_;
goto v_reusejp_161_;
}
v_reusejp_161_:
{
return v___x_162_;
}
}
}
}
else
{
lean_dec_ref_known(v_a_115_, 1);
lean_dec_ref(v_type_108_);
lean_dec_ref(v_extractStep_x3f_107_);
return v___x_114_;
}
}
else
{
lean_dec_ref(v_type_108_);
lean_dec_ref(v_extractStep_x3f_107_);
return v___x_114_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f___boxed(lean_object* v_extractStep_x3f_165_, lean_object* v_type_166_, lean_object* v_a_167_, lean_object* v_a_168_, lean_object* v_a_169_, lean_object* v_a_170_, lean_object* v_a_171_){
_start:
{
lean_object* v_res_172_; 
v_res_172_ = lp_Qq___private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f(v_extractStep_x3f_165_, v_type_166_, v_a_167_, v_a_168_, v_a_169_, v_a_170_);
lean_dec(v_a_170_);
lean_dec_ref(v_a_169_);
lean_dec(v_a_168_);
lean_dec_ref(v_a_167_);
return v_res_172_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00__private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f_spec__0(lean_object* v_00_u03b1_173_, lean_object* v_msg_174_, lean_object* v___y_175_, lean_object* v___y_176_, lean_object* v___y_177_, lean_object* v___y_178_){
_start:
{
lean_object* v___x_180_; 
v___x_180_ = lp_Qq_Lean_throwError___at___00__private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f_spec__0___redArg(v_msg_174_, v___y_175_, v___y_176_, v___y_177_, v___y_178_);
return v___x_180_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00__private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f_spec__0___boxed(lean_object* v_00_u03b1_181_, lean_object* v_msg_182_, lean_object* v___y_183_, lean_object* v___y_184_, lean_object* v___y_185_, lean_object* v___y_186_, lean_object* v___y_187_){
_start:
{
lean_object* v_res_188_; 
v_res_188_ = lp_Qq_Lean_throwError___at___00__private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f_spec__0(v_00_u03b1_181_, v_msg_182_, v___y_183_, v___y_184_, v___y_185_, v___y_186_);
lean_dec(v___y_186_);
lean_dec_ref(v___y_185_);
lean_dec(v___y_184_);
lean_dec_ref(v___y_183_);
return v_res_188_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_Term_extractBind___lam__0(lean_object* v_val_189_, lean_object* v_type_190_, lean_object* v___y_191_, lean_object* v___y_192_, lean_object* v___y_193_, lean_object* v___y_194_){
_start:
{
if (lean_obj_tag(v_type_190_) == 5)
{
lean_object* v_fn_196_; lean_object* v_arg_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; 
v_fn_196_ = lean_ctor_get(v_type_190_, 0);
v_arg_197_ = lean_ctor_get(v_type_190_, 1);
lean_inc_ref(v_arg_197_);
lean_inc_ref(v_fn_196_);
v___x_198_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_198_, 0, v_fn_196_);
lean_ctor_set(v___x_198_, 1, v_arg_197_);
lean_ctor_set(v___x_198_, 2, v_val_189_);
v___x_199_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_199_, 0, v___x_198_);
v___x_200_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_200_, 0, v___x_199_);
return v___x_200_;
}
else
{
lean_object* v___x_201_; lean_object* v___x_202_; 
lean_dec_ref(v_val_189_);
v___x_201_ = lean_box(0);
v___x_202_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_202_, 0, v___x_201_);
return v___x_202_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_Term_extractBind___lam__0___boxed(lean_object* v_val_203_, lean_object* v_type_204_, lean_object* v___y_205_, lean_object* v___y_206_, lean_object* v___y_207_, lean_object* v___y_208_, lean_object* v___y_209_){
_start:
{
lean_object* v_res_210_; 
v_res_210_ = lp_Qq_Lean_Elab_Term_extractBind___lam__0(v_val_203_, v_type_204_, v___y_205_, v___y_206_, v___y_207_, v___y_208_);
lean_dec(v___y_208_);
lean_dec_ref(v___y_207_);
lean_dec(v___y_206_);
lean_dec_ref(v___y_205_);
lean_dec_ref(v_type_204_);
return v_res_210_;
}
}
LEAN_EXPORT uint8_t lp_Qq_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__1(lean_object* v_opts_211_, lean_object* v_opt_212_){
_start:
{
lean_object* v_name_213_; lean_object* v_defValue_214_; lean_object* v_map_215_; lean_object* v___x_216_; 
v_name_213_ = lean_ctor_get(v_opt_212_, 0);
v_defValue_214_ = lean_ctor_get(v_opt_212_, 1);
v_map_215_ = lean_ctor_get(v_opts_211_, 0);
v___x_216_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_215_, v_name_213_);
if (lean_obj_tag(v___x_216_) == 0)
{
uint8_t v___x_217_; 
v___x_217_ = lean_unbox(v_defValue_214_);
return v___x_217_;
}
else
{
lean_object* v_val_218_; 
v_val_218_ = lean_ctor_get(v___x_216_, 0);
lean_inc(v_val_218_);
lean_dec_ref_known(v___x_216_, 1);
if (lean_obj_tag(v_val_218_) == 1)
{
uint8_t v_v_219_; 
v_v_219_ = lean_ctor_get_uint8(v_val_218_, 0);
lean_dec_ref_known(v_val_218_, 0);
return v_v_219_;
}
else
{
uint8_t v___x_220_; 
lean_dec(v_val_218_);
v___x_220_ = lean_unbox(v_defValue_214_);
return v___x_220_;
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__1___boxed(lean_object* v_opts_221_, lean_object* v_opt_222_){
_start:
{
uint8_t v_res_223_; lean_object* v_r_224_; 
v_res_223_ = lp_Qq_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__1(v_opts_221_, v_opt_222_);
lean_dec_ref(v_opt_222_);
lean_dec_ref(v_opts_221_);
v_r_224_ = lean_box(v_res_223_);
return v_r_224_;
}
}
static lean_object* _init_lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2___closed__0(void){
_start:
{
lean_object* v___x_225_; lean_object* v___x_226_; 
v___x_225_ = lean_box(1);
v___x_226_ = l_Lean_MessageData_ofFormat(v___x_225_);
return v___x_226_;
}
}
static lean_object* _init_lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2___closed__3(void){
_start:
{
lean_object* v___x_230_; lean_object* v___x_231_; 
v___x_230_ = ((lean_object*)(lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2___closed__2));
v___x_231_ = l_Lean_MessageData_ofFormat(v___x_230_);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2(lean_object* v_x_232_, lean_object* v_x_233_){
_start:
{
if (lean_obj_tag(v_x_233_) == 0)
{
return v_x_232_;
}
else
{
lean_object* v_head_234_; lean_object* v_tail_235_; lean_object* v___x_237_; uint8_t v_isShared_238_; uint8_t v_isSharedCheck_257_; 
v_head_234_ = lean_ctor_get(v_x_233_, 0);
v_tail_235_ = lean_ctor_get(v_x_233_, 1);
v_isSharedCheck_257_ = !lean_is_exclusive(v_x_233_);
if (v_isSharedCheck_257_ == 0)
{
v___x_237_ = v_x_233_;
v_isShared_238_ = v_isSharedCheck_257_;
goto v_resetjp_236_;
}
else
{
lean_inc(v_tail_235_);
lean_inc(v_head_234_);
lean_dec(v_x_233_);
v___x_237_ = lean_box(0);
v_isShared_238_ = v_isSharedCheck_257_;
goto v_resetjp_236_;
}
v_resetjp_236_:
{
lean_object* v_before_239_; lean_object* v___x_241_; uint8_t v_isShared_242_; uint8_t v_isSharedCheck_255_; 
v_before_239_ = lean_ctor_get(v_head_234_, 0);
v_isSharedCheck_255_ = !lean_is_exclusive(v_head_234_);
if (v_isSharedCheck_255_ == 0)
{
lean_object* v_unused_256_; 
v_unused_256_ = lean_ctor_get(v_head_234_, 1);
lean_dec(v_unused_256_);
v___x_241_ = v_head_234_;
v_isShared_242_ = v_isSharedCheck_255_;
goto v_resetjp_240_;
}
else
{
lean_inc(v_before_239_);
lean_dec(v_head_234_);
v___x_241_ = lean_box(0);
v_isShared_242_ = v_isSharedCheck_255_;
goto v_resetjp_240_;
}
v_resetjp_240_:
{
lean_object* v___x_243_; lean_object* v___x_245_; 
v___x_243_ = lean_obj_once(&lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2___closed__0, &lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2___closed__0_once, _init_lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2___closed__0);
if (v_isShared_242_ == 0)
{
lean_ctor_set_tag(v___x_241_, 7);
lean_ctor_set(v___x_241_, 1, v___x_243_);
lean_ctor_set(v___x_241_, 0, v_x_232_);
v___x_245_ = v___x_241_;
goto v_reusejp_244_;
}
else
{
lean_object* v_reuseFailAlloc_254_; 
v_reuseFailAlloc_254_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_254_, 0, v_x_232_);
lean_ctor_set(v_reuseFailAlloc_254_, 1, v___x_243_);
v___x_245_ = v_reuseFailAlloc_254_;
goto v_reusejp_244_;
}
v_reusejp_244_:
{
lean_object* v___x_246_; lean_object* v___x_248_; 
v___x_246_ = lean_obj_once(&lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2___closed__3, &lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2___closed__3_once, _init_lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2___closed__3);
if (v_isShared_238_ == 0)
{
lean_ctor_set_tag(v___x_237_, 7);
lean_ctor_set(v___x_237_, 1, v___x_246_);
lean_ctor_set(v___x_237_, 0, v___x_245_);
v___x_248_ = v___x_237_;
goto v_reusejp_247_;
}
else
{
lean_object* v_reuseFailAlloc_253_; 
v_reuseFailAlloc_253_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_253_, 0, v___x_245_);
lean_ctor_set(v_reuseFailAlloc_253_, 1, v___x_246_);
v___x_248_ = v_reuseFailAlloc_253_;
goto v_reusejp_247_;
}
v_reusejp_247_:
{
lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; 
v___x_249_ = l_Lean_MessageData_ofSyntax(v_before_239_);
v___x_250_ = l_Lean_indentD(v___x_249_);
v___x_251_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_251_, 0, v___x_248_);
lean_ctor_set(v___x_251_, 1, v___x_250_);
v_x_232_ = v___x_251_;
v_x_233_ = v_tail_235_;
goto _start;
}
}
}
}
}
}
}
static lean_object* _init_lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0___redArg___closed__2(void){
_start:
{
lean_object* v___x_261_; lean_object* v___x_262_; 
v___x_261_ = ((lean_object*)(lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0___redArg___closed__1));
v___x_262_ = l_Lean_MessageData_ofFormat(v___x_261_);
return v___x_262_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0___redArg(lean_object* v_msgData_263_, lean_object* v_macroStack_264_, lean_object* v___y_265_){
_start:
{
lean_object* v_options_267_; lean_object* v___x_268_; uint8_t v___x_269_; 
v_options_267_ = lean_ctor_get(v___y_265_, 2);
v___x_268_ = l_Lean_Elab_pp_macroStack;
v___x_269_ = lp_Qq_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__1(v_options_267_, v___x_268_);
if (v___x_269_ == 0)
{
lean_object* v___x_270_; 
lean_dec(v_macroStack_264_);
v___x_270_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_270_, 0, v_msgData_263_);
return v___x_270_;
}
else
{
if (lean_obj_tag(v_macroStack_264_) == 0)
{
lean_object* v___x_271_; 
v___x_271_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_271_, 0, v_msgData_263_);
return v___x_271_;
}
else
{
lean_object* v_head_272_; lean_object* v_after_273_; lean_object* v___x_275_; uint8_t v_isShared_276_; uint8_t v_isSharedCheck_288_; 
v_head_272_ = lean_ctor_get(v_macroStack_264_, 0);
lean_inc(v_head_272_);
v_after_273_ = lean_ctor_get(v_head_272_, 1);
v_isSharedCheck_288_ = !lean_is_exclusive(v_head_272_);
if (v_isSharedCheck_288_ == 0)
{
lean_object* v_unused_289_; 
v_unused_289_ = lean_ctor_get(v_head_272_, 0);
lean_dec(v_unused_289_);
v___x_275_ = v_head_272_;
v_isShared_276_ = v_isSharedCheck_288_;
goto v_resetjp_274_;
}
else
{
lean_inc(v_after_273_);
lean_dec(v_head_272_);
v___x_275_ = lean_box(0);
v_isShared_276_ = v_isSharedCheck_288_;
goto v_resetjp_274_;
}
v_resetjp_274_:
{
lean_object* v___x_277_; lean_object* v___x_279_; 
v___x_277_ = lean_obj_once(&lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2___closed__0, &lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2___closed__0_once, _init_lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2___closed__0);
if (v_isShared_276_ == 0)
{
lean_ctor_set_tag(v___x_275_, 7);
lean_ctor_set(v___x_275_, 1, v___x_277_);
lean_ctor_set(v___x_275_, 0, v_msgData_263_);
v___x_279_ = v___x_275_;
goto v_reusejp_278_;
}
else
{
lean_object* v_reuseFailAlloc_287_; 
v_reuseFailAlloc_287_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_287_, 0, v_msgData_263_);
lean_ctor_set(v_reuseFailAlloc_287_, 1, v___x_277_);
v___x_279_ = v_reuseFailAlloc_287_;
goto v_reusejp_278_;
}
v_reusejp_278_:
{
lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v_msgData_284_; lean_object* v___x_285_; lean_object* v___x_286_; 
v___x_280_ = lean_obj_once(&lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0___redArg___closed__2, &lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0___redArg___closed__2_once, _init_lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0___redArg___closed__2);
v___x_281_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_281_, 0, v___x_279_);
lean_ctor_set(v___x_281_, 1, v___x_280_);
v___x_282_ = l_Lean_MessageData_ofSyntax(v_after_273_);
v___x_283_ = l_Lean_indentD(v___x_282_);
v_msgData_284_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_284_, 0, v___x_281_);
lean_ctor_set(v_msgData_284_, 1, v___x_283_);
v___x_285_ = lp_Qq_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0_spec__2(v_msgData_284_, v_macroStack_264_);
v___x_286_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_286_, 0, v___x_285_);
return v___x_286_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0___redArg___boxed(lean_object* v_msgData_290_, lean_object* v_macroStack_291_, lean_object* v___y_292_, lean_object* v___y_293_){
_start:
{
lean_object* v_res_294_; 
v_res_294_ = lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0___redArg(v_msgData_290_, v_macroStack_291_, v___y_292_);
lean_dec_ref(v___y_292_);
return v_res_294_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0___redArg(lean_object* v_msg_295_, lean_object* v___y_296_, lean_object* v___y_297_, lean_object* v___y_298_, lean_object* v___y_299_, lean_object* v___y_300_, lean_object* v___y_301_){
_start:
{
lean_object* v_ref_303_; lean_object* v___x_304_; lean_object* v_a_305_; lean_object* v_macroStack_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v_a_309_; lean_object* v___x_311_; uint8_t v_isShared_312_; uint8_t v_isSharedCheck_317_; 
v_ref_303_ = lean_ctor_get(v___y_300_, 5);
v___x_304_ = lp_Qq_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f_spec__0_spec__0(v_msg_295_, v___y_298_, v___y_299_, v___y_300_, v___y_301_);
v_a_305_ = lean_ctor_get(v___x_304_, 0);
lean_inc(v_a_305_);
lean_dec_ref(v___x_304_);
v_macroStack_306_ = lean_ctor_get(v___y_296_, 1);
v___x_307_ = l_Lean_Elab_getBetterRef(v_ref_303_, v_macroStack_306_);
lean_inc(v_macroStack_306_);
v___x_308_ = lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0___redArg(v_a_305_, v_macroStack_306_, v___y_300_);
v_a_309_ = lean_ctor_get(v___x_308_, 0);
v_isSharedCheck_317_ = !lean_is_exclusive(v___x_308_);
if (v_isSharedCheck_317_ == 0)
{
v___x_311_ = v___x_308_;
v_isShared_312_ = v_isSharedCheck_317_;
goto v_resetjp_310_;
}
else
{
lean_inc(v_a_309_);
lean_dec(v___x_308_);
v___x_311_ = lean_box(0);
v_isShared_312_ = v_isSharedCheck_317_;
goto v_resetjp_310_;
}
v_resetjp_310_:
{
lean_object* v___x_313_; lean_object* v___x_315_; 
v___x_313_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_313_, 0, v___x_307_);
lean_ctor_set(v___x_313_, 1, v_a_309_);
if (v_isShared_312_ == 0)
{
lean_ctor_set_tag(v___x_311_, 1);
lean_ctor_set(v___x_311_, 0, v___x_313_);
v___x_315_ = v___x_311_;
goto v_reusejp_314_;
}
else
{
lean_object* v_reuseFailAlloc_316_; 
v_reuseFailAlloc_316_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_316_, 0, v___x_313_);
v___x_315_ = v_reuseFailAlloc_316_;
goto v_reusejp_314_;
}
v_reusejp_314_:
{
return v___x_315_;
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0___redArg___boxed(lean_object* v_msg_318_, lean_object* v___y_319_, lean_object* v___y_320_, lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_, lean_object* v___y_324_, lean_object* v___y_325_){
_start:
{
lean_object* v_res_326_; 
v_res_326_ = lp_Qq_Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0___redArg(v_msg_318_, v___y_319_, v___y_320_, v___y_321_, v___y_322_, v___y_323_, v___y_324_);
lean_dec(v___y_324_);
lean_dec_ref(v___y_323_);
lean_dec(v___y_322_);
lean_dec_ref(v___y_321_);
lean_dec(v___y_320_);
lean_dec_ref(v___y_319_);
return v_res_326_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_Term_extractBind(lean_object* v_expectedType_x3f_327_, lean_object* v_a_328_, lean_object* v_a_329_, lean_object* v_a_330_, lean_object* v_a_331_, lean_object* v_a_332_, lean_object* v_a_333_){
_start:
{
if (lean_obj_tag(v_expectedType_x3f_327_) == 0)
{
lean_object* v___x_335_; lean_object* v___x_336_; 
v___x_335_ = lean_obj_once(&lp_Qq___private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f___closed__1, &lp_Qq___private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f___closed__1_once, _init_lp_Qq___private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f___closed__1);
v___x_336_ = lp_Qq_Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0___redArg(v___x_335_, v_a_328_, v_a_329_, v_a_330_, v_a_331_, v_a_332_, v_a_333_);
return v___x_336_;
}
else
{
lean_object* v_val_337_; lean_object* v_extractStep_x3f_338_; lean_object* v___x_339_; 
v_val_337_ = lean_ctor_get(v_expectedType_x3f_327_, 0);
lean_inc_n(v_val_337_, 3);
lean_dec_ref_known(v_expectedType_x3f_327_, 1);
v_extractStep_x3f_338_ = lean_alloc_closure((void*)(lp_Qq_Lean_Elab_Term_extractBind___lam__0___boxed), 7, 1);
lean_closure_set(v_extractStep_x3f_338_, 0, v_val_337_);
v___x_339_ = lp_Qq___private_Qq_ForLean_Do_0__Lean_Elab_Term_extractBind_extract_x3f(v_extractStep_x3f_338_, v_val_337_, v_a_330_, v_a_331_, v_a_332_, v_a_333_);
if (lean_obj_tag(v___x_339_) == 0)
{
lean_object* v_a_340_; lean_object* v___x_342_; uint8_t v_isShared_343_; uint8_t v_isSharedCheck_349_; 
v_a_340_ = lean_ctor_get(v___x_339_, 0);
v_isSharedCheck_349_ = !lean_is_exclusive(v___x_339_);
if (v_isSharedCheck_349_ == 0)
{
v___x_342_ = v___x_339_;
v_isShared_343_ = v_isSharedCheck_349_;
goto v_resetjp_341_;
}
else
{
lean_inc(v_a_340_);
lean_dec(v___x_339_);
v___x_342_ = lean_box(0);
v_isShared_343_ = v_isSharedCheck_349_;
goto v_resetjp_341_;
}
v_resetjp_341_:
{
if (lean_obj_tag(v_a_340_) == 0)
{
lean_object* v___x_344_; 
lean_del_object(v___x_342_);
v___x_344_ = lp_Qq_Lean_Elab_Term_mkIdBindFor___redArg(v_val_337_, v_a_330_, v_a_331_, v_a_332_, v_a_333_);
return v___x_344_;
}
else
{
lean_object* v_val_345_; lean_object* v___x_347_; 
lean_dec(v_val_337_);
v_val_345_ = lean_ctor_get(v_a_340_, 0);
lean_inc(v_val_345_);
lean_dec_ref_known(v_a_340_, 1);
if (v_isShared_343_ == 0)
{
lean_ctor_set(v___x_342_, 0, v_val_345_);
v___x_347_ = v___x_342_;
goto v_reusejp_346_;
}
else
{
lean_object* v_reuseFailAlloc_348_; 
v_reuseFailAlloc_348_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_348_, 0, v_val_345_);
v___x_347_ = v_reuseFailAlloc_348_;
goto v_reusejp_346_;
}
v_reusejp_346_:
{
return v___x_347_;
}
}
}
}
else
{
lean_object* v_a_350_; lean_object* v___x_352_; uint8_t v_isShared_353_; uint8_t v_isSharedCheck_357_; 
lean_dec(v_val_337_);
v_a_350_ = lean_ctor_get(v___x_339_, 0);
v_isSharedCheck_357_ = !lean_is_exclusive(v___x_339_);
if (v_isSharedCheck_357_ == 0)
{
v___x_352_ = v___x_339_;
v_isShared_353_ = v_isSharedCheck_357_;
goto v_resetjp_351_;
}
else
{
lean_inc(v_a_350_);
lean_dec(v___x_339_);
v___x_352_ = lean_box(0);
v_isShared_353_ = v_isSharedCheck_357_;
goto v_resetjp_351_;
}
v_resetjp_351_:
{
lean_object* v___x_355_; 
if (v_isShared_353_ == 0)
{
v___x_355_ = v___x_352_;
goto v_reusejp_354_;
}
else
{
lean_object* v_reuseFailAlloc_356_; 
v_reuseFailAlloc_356_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_356_, 0, v_a_350_);
v___x_355_ = v_reuseFailAlloc_356_;
goto v_reusejp_354_;
}
v_reusejp_354_:
{
return v___x_355_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_Term_extractBind___boxed(lean_object* v_expectedType_x3f_358_, lean_object* v_a_359_, lean_object* v_a_360_, lean_object* v_a_361_, lean_object* v_a_362_, lean_object* v_a_363_, lean_object* v_a_364_, lean_object* v_a_365_){
_start:
{
lean_object* v_res_366_; 
v_res_366_ = lp_Qq_Lean_Elab_Term_extractBind(v_expectedType_x3f_358_, v_a_359_, v_a_360_, v_a_361_, v_a_362_, v_a_363_, v_a_364_);
lean_dec(v_a_364_);
lean_dec_ref(v_a_363_);
lean_dec(v_a_362_);
lean_dec_ref(v_a_361_);
lean_dec(v_a_360_);
lean_dec_ref(v_a_359_);
return v_res_366_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0(lean_object* v_00_u03b1_367_, lean_object* v_msg_368_, lean_object* v___y_369_, lean_object* v___y_370_, lean_object* v___y_371_, lean_object* v___y_372_, lean_object* v___y_373_, lean_object* v___y_374_){
_start:
{
lean_object* v___x_376_; 
v___x_376_ = lp_Qq_Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0___redArg(v_msg_368_, v___y_369_, v___y_370_, v___y_371_, v___y_372_, v___y_373_, v___y_374_);
return v___x_376_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0___boxed(lean_object* v_00_u03b1_377_, lean_object* v_msg_378_, lean_object* v___y_379_, lean_object* v___y_380_, lean_object* v___y_381_, lean_object* v___y_382_, lean_object* v___y_383_, lean_object* v___y_384_, lean_object* v___y_385_){
_start:
{
lean_object* v_res_386_; 
v_res_386_ = lp_Qq_Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0(v_00_u03b1_377_, v_msg_378_, v___y_379_, v___y_380_, v___y_381_, v___y_382_, v___y_383_, v___y_384_);
lean_dec(v___y_384_);
lean_dec_ref(v___y_383_);
lean_dec(v___y_382_);
lean_dec_ref(v___y_381_);
lean_dec(v___y_380_);
lean_dec_ref(v___y_379_);
return v_res_386_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0(lean_object* v_msgData_387_, lean_object* v_macroStack_388_, lean_object* v___y_389_, lean_object* v___y_390_, lean_object* v___y_391_, lean_object* v___y_392_, lean_object* v___y_393_, lean_object* v___y_394_){
_start:
{
lean_object* v___x_396_; 
v___x_396_ = lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0___redArg(v_msgData_387_, v_macroStack_388_, v___y_393_);
return v___x_396_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0___boxed(lean_object* v_msgData_397_, lean_object* v_macroStack_398_, lean_object* v___y_399_, lean_object* v___y_400_, lean_object* v___y_401_, lean_object* v___y_402_, lean_object* v___y_403_, lean_object* v___y_404_, lean_object* v___y_405_){
_start:
{
lean_object* v_res_406_; 
v_res_406_ = lp_Qq_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_Term_extractBind_spec__0_spec__0(v_msgData_397_, v_macroStack_398_, v___y_399_, v___y_400_, v___y_401_, v___y_402_, v___y_403_, v___y_404_);
lean_dec(v___y_404_);
lean_dec_ref(v___y_403_);
lean_dec(v___y_402_);
lean_dec_ref(v___y_401_);
lean_dec(v___y_400_);
lean_dec_ref(v___y_399_);
return v_res_406_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Do_Legacy(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_Qq_Qq_ForLean_Do(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Do_Legacy(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_Qq_Qq_ForLean_Do(uint8_t builtin) {
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
lean_object* initialize_Lean_Elab_Do_Legacy(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_Qq_Qq_ForLean_Do(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Do_Legacy(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_ForLean_Do(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_Qq_Qq_ForLean_Do(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_Qq_Qq_ForLean_Do(builtin);
}
#ifdef __cplusplus
}
#endif
