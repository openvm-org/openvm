// Lean compiler output
// Module: Mathlib.Lean.Expr.Rat
// Imports: public import Init public meta import Init public import Mathlib.Init public import Lean.ToExpr
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
lean_object* l_Rat_ofInt(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_int_x3f(lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* l_Lean_Expr_nat_x3f(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Expr_appFn_x21(lean_object*);
lean_object* l_mkRat(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_cast___at___00Lean_Expr_rat_x3f_spec__0(lean_object*);
static const lean_string_object lp_mathlib_Lean_Expr_rat_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Div"};
static const lean_object* lp_mathlib_Lean_Expr_rat_x3f___closed__0 = (const lean_object*)&lp_mathlib_Lean_Expr_rat_x3f___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Expr_rat_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "div"};
static const lean_object* lp_mathlib_Lean_Expr_rat_x3f___closed__1 = (const lean_object*)&lp_mathlib_Lean_Expr_rat_x3f___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_rat_x3f___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_rat_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(153, 247, 56, 19, 64, 245, 190, 87)}};
static const lean_ctor_object lp_mathlib_Lean_Expr_rat_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Expr_rat_x3f___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Lean_Expr_rat_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(25, 78, 24, 213, 240, 238, 239, 80)}};
static const lean_object* lp_mathlib_Lean_Expr_rat_x3f___closed__2 = (const lean_object*)&lp_mathlib_Lean_Expr_rat_x3f___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_rat_x3f(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Expr_isExplicitNumber(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_isExplicitNumber___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_cast___at___00Lean_Expr_rat_x3f_spec__0(lean_object* v_a_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = l_Rat_ofInt(v_a_1_);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_rat_x3f(lean_object* v_e_8_){
_start:
{
lean_object* v___x_9_; lean_object* v___x_10_; uint8_t v___x_11_; 
v___x_9_ = ((lean_object*)(lp_mathlib_Lean_Expr_rat_x3f___closed__2));
v___x_10_ = lean_unsigned_to_nat(4u);
v___x_11_ = l_Lean_Expr_isAppOfArity(v_e_8_, v___x_9_, v___x_10_);
if (v___x_11_ == 0)
{
lean_object* v___x_12_; 
v___x_12_ = l_Lean_Expr_int_x3f(v_e_8_);
if (lean_obj_tag(v___x_12_) == 0)
{
lean_object* v___x_13_; 
v___x_13_ = lean_box(0);
return v___x_13_;
}
else
{
lean_object* v_val_14_; lean_object* v___x_16_; uint8_t v_isShared_17_; uint8_t v_isSharedCheck_22_; 
v_val_14_ = lean_ctor_get(v___x_12_, 0);
v_isSharedCheck_22_ = !lean_is_exclusive(v___x_12_);
if (v_isSharedCheck_22_ == 0)
{
v___x_16_ = v___x_12_;
v_isShared_17_ = v_isSharedCheck_22_;
goto v_resetjp_15_;
}
else
{
lean_inc(v_val_14_);
lean_dec(v___x_12_);
v___x_16_ = lean_box(0);
v_isShared_17_ = v_isSharedCheck_22_;
goto v_resetjp_15_;
}
v_resetjp_15_:
{
lean_object* v___x_18_; lean_object* v___x_20_; 
v___x_18_ = l_Rat_ofInt(v_val_14_);
if (v_isShared_17_ == 0)
{
lean_ctor_set(v___x_16_, 0, v___x_18_);
v___x_20_ = v___x_16_;
goto v_reusejp_19_;
}
else
{
lean_object* v_reuseFailAlloc_21_; 
v_reuseFailAlloc_21_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_21_, 0, v___x_18_);
v___x_20_ = v_reuseFailAlloc_21_;
goto v_reusejp_19_;
}
v_reusejp_19_:
{
return v___x_20_;
}
}
}
}
else
{
lean_object* v___x_23_; lean_object* v___x_24_; 
v___x_23_ = l_Lean_Expr_appArg_x21(v_e_8_);
v___x_24_ = l_Lean_Expr_nat_x3f(v___x_23_);
if (lean_obj_tag(v___x_24_) == 0)
{
lean_object* v___x_25_; 
lean_dec_ref(v_e_8_);
v___x_25_ = lean_box(0);
return v___x_25_;
}
else
{
lean_object* v_val_26_; lean_object* v___x_27_; uint8_t v___x_28_; 
v_val_26_ = lean_ctor_get(v___x_24_, 0);
lean_inc(v_val_26_);
lean_dec_ref_known(v___x_24_, 1);
v___x_27_ = lean_unsigned_to_nat(1u);
v___x_28_ = lean_nat_dec_eq(v_val_26_, v___x_27_);
if (v___x_28_ == 0)
{
if (v___x_11_ == 0)
{
lean_object* v___x_29_; 
lean_dec(v_val_26_);
lean_dec_ref(v_e_8_);
v___x_29_ = lean_box(0);
return v___x_29_;
}
else
{
lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; 
v___x_30_ = l_Lean_Expr_appFn_x21(v_e_8_);
lean_dec_ref(v_e_8_);
v___x_31_ = l_Lean_Expr_appArg_x21(v___x_30_);
lean_dec_ref(v___x_30_);
v___x_32_ = l_Lean_Expr_int_x3f(v___x_31_);
if (lean_obj_tag(v___x_32_) == 0)
{
lean_object* v___x_33_; 
lean_dec(v_val_26_);
v___x_33_ = lean_box(0);
return v___x_33_;
}
else
{
lean_object* v_val_34_; lean_object* v___x_36_; uint8_t v_isShared_37_; uint8_t v_isSharedCheck_45_; 
v_val_34_ = lean_ctor_get(v___x_32_, 0);
v_isSharedCheck_45_ = !lean_is_exclusive(v___x_32_);
if (v_isSharedCheck_45_ == 0)
{
v___x_36_ = v___x_32_;
v_isShared_37_ = v_isSharedCheck_45_;
goto v_resetjp_35_;
}
else
{
lean_inc(v_val_34_);
lean_dec(v___x_32_);
v___x_36_ = lean_box(0);
v_isShared_37_ = v_isSharedCheck_45_;
goto v_resetjp_35_;
}
v_resetjp_35_:
{
lean_object* v___x_38_; lean_object* v_den_39_; uint8_t v___x_40_; 
lean_inc(v_val_26_);
v___x_38_ = l_mkRat(v_val_34_, v_val_26_);
v_den_39_ = lean_ctor_get(v___x_38_, 1);
lean_inc(v_den_39_);
v___x_40_ = lean_nat_dec_eq(v_den_39_, v_val_26_);
lean_dec(v_val_26_);
lean_dec(v_den_39_);
if (v___x_40_ == 0)
{
lean_object* v___x_41_; 
lean_dec_ref(v___x_38_);
lean_del_object(v___x_36_);
v___x_41_ = lean_box(0);
return v___x_41_;
}
else
{
lean_object* v___x_43_; 
if (v_isShared_37_ == 0)
{
lean_ctor_set(v___x_36_, 0, v___x_38_);
v___x_43_ = v___x_36_;
goto v_reusejp_42_;
}
else
{
lean_object* v_reuseFailAlloc_44_; 
v_reuseFailAlloc_44_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_44_, 0, v___x_38_);
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
}
else
{
lean_object* v___x_46_; 
lean_dec(v_val_26_);
lean_dec_ref(v_e_8_);
v___x_46_ = lean_box(0);
return v___x_46_;
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Expr_isExplicitNumber(lean_object* v_x_47_){
_start:
{
switch(lean_obj_tag(v_x_47_))
{
case 9:
{
uint8_t v___x_48_; 
lean_dec_ref_known(v_x_47_, 1);
v___x_48_ = 1;
return v___x_48_;
}
case 10:
{
lean_object* v_expr_49_; 
v_expr_49_ = lean_ctor_get(v_x_47_, 1);
lean_inc_ref(v_expr_49_);
lean_dec_ref_known(v_x_47_, 2);
v_x_47_ = v_expr_49_;
goto _start;
}
default: 
{
lean_object* v___x_51_; 
v___x_51_ = lp_mathlib_Lean_Expr_rat_x3f(v_x_47_);
if (lean_obj_tag(v___x_51_) == 0)
{
uint8_t v___x_52_; 
v___x_52_ = 0;
return v___x_52_;
}
else
{
uint8_t v___x_53_; 
lean_dec_ref_known(v___x_51_, 1);
v___x_53_ = 1;
return v___x_53_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_isExplicitNumber___boxed(lean_object* v_x_54_){
_start:
{
uint8_t v_res_55_; lean_object* v_r_56_; 
v_res_55_ = lp_mathlib_Lean_Expr_isExplicitNumber(v_x_54_);
v_r_56_ = lean_box(v_res_55_);
return v_r_56_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_ToExpr(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Lean_Expr_Rat(uint8_t builtin) {
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
res = runtime_initialize_Lean_ToExpr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Lean_Expr_Rat(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_ToExpr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Lean_Expr_Rat(uint8_t builtin) {
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
res = initialize_Lean_ToExpr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Expr_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Lean_Expr_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Lean_Expr_Rat(builtin);
}
#ifdef __cplusplus
}
#endif
