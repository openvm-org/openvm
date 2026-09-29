// Lean compiler output
// Module: Mathlib.Lean.Expr.ExtraRecognizers
// Imports: public import Init public meta import Init public import Mathlib.Data.Set.CoeSort
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
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Set"};
static const lean_object* lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__0 = (const lean_object*)&lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elem"};
static const lean_object* lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__1 = (const lean_object*)&lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(20, 190, 239, 85, 165, 199, 80, 79)}};
static const lean_object* lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__2 = (const lean_object*)&lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Subtype"};
static const lean_object* lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__3 = (const lean_object*)&lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__3_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__3_value),LEAN_SCALAR_PTR_LITERAL(30, 108, 3, 75, 185, 102, 103, 84)}};
static const lean_object* lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__4 = (const lean_object*)&lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__4_value;
static const lean_string_object lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Membership"};
static const lean_object* lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__5 = (const lean_object*)&lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__5_value;
static const lean_string_object lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "mem"};
static const lean_object* lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__6 = (const lean_object*)&lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__6_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__5_value),LEAN_SCALAR_PTR_LITERAL(205, 217, 109, 94, 255, 55, 82, 109)}};
static const lean_ctor_object lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__6_value),LEAN_SCALAR_PTR_LITERAL(224, 90, 126, 237, 128, 148, 153, 69)}};
static const lean_object* lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__7 = (const lean_object*)&lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__7_value;
static lean_once_cell_t lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__8;
static const lean_string_object lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "instMembership"};
static const lean_object* lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__9 = (const lean_object*)&lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__9_value;
static const lean_ctor_object lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__9_value),LEAN_SCALAR_PTR_LITERAL(42, 66, 71, 72, 45, 15, 111, 224)}};
static const lean_object* lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__10 = (const lean_object*)&lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_coeTypeSet_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_coeTypeSet_x3f___boxed(lean_object*);
static lean_object* _init_lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__8(void){
_start:
{
lean_object* v___x_14_; lean_object* v_dummy_15_; 
v___x_14_ = lean_box(0);
v_dummy_15_ = l_Lean_Expr_sort___override(v___x_14_);
return v_dummy_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_coeTypeSet_x3f(lean_object* v_e_20_){
_start:
{
lean_object* v___x_21_; lean_object* v___x_22_; uint8_t v___x_23_; 
v___x_21_ = ((lean_object*)(lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__2));
v___x_22_ = lean_unsigned_to_nat(2u);
v___x_23_ = l_Lean_Expr_isAppOfArity(v_e_20_, v___x_21_, v___x_22_);
if (v___x_23_ == 0)
{
lean_object* v___x_24_; uint8_t v___x_25_; 
v___x_24_ = ((lean_object*)(lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__4));
v___x_25_ = l_Lean_Expr_isAppOfArity(v_e_20_, v___x_24_, v___x_22_);
if (v___x_25_ == 0)
{
lean_object* v___x_26_; 
v___x_26_ = lean_box(0);
return v___x_26_;
}
else
{
lean_object* v___x_27_; 
v___x_27_ = l_Lean_Expr_appArg_x21(v_e_20_);
if (lean_obj_tag(v___x_27_) == 6)
{
lean_object* v_body_28_; lean_object* v___x_29_; lean_object* v___x_30_; uint8_t v___x_31_; 
v_body_28_ = lean_ctor_get(v___x_27_, 2);
lean_inc_ref(v_body_28_);
lean_dec_ref_known(v___x_27_, 3);
v___x_29_ = ((lean_object*)(lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__7));
v___x_30_ = lean_unsigned_to_nat(5u);
v___x_31_ = l_Lean_Expr_isAppOfArity(v_body_28_, v___x_29_, v___x_30_);
if (v___x_31_ == 0)
{
lean_object* v___x_32_; 
lean_dec_ref(v_body_28_);
v___x_32_ = lean_box(0);
return v___x_32_;
}
else
{
lean_object* v_dummy_33_; lean_object* v_nargs_34_; lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; uint8_t v___x_40_; 
v_dummy_33_ = lean_obj_once(&lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__8, &lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__8_once, _init_lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__8);
v_nargs_34_ = l_Lean_Expr_getAppNumArgs(v_body_28_);
lean_inc(v_nargs_34_);
v___x_35_ = lean_mk_array(v_nargs_34_, v_dummy_33_);
v___x_36_ = lean_unsigned_to_nat(1u);
v___x_37_ = lean_nat_sub(v_nargs_34_, v___x_36_);
lean_dec(v_nargs_34_);
v___x_38_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_body_28_, v___x_35_, v___x_37_);
v___x_39_ = lean_array_get_size(v___x_38_);
v___x_40_ = lean_nat_dec_eq(v___x_39_, v___x_30_);
if (v___x_40_ == 0)
{
lean_object* v___x_41_; 
lean_dec_ref(v___x_38_);
v___x_41_ = lean_box(0);
return v___x_41_;
}
else
{
lean_object* v___x_42_; lean_object* v___x_43_; 
v___x_42_ = lean_unsigned_to_nat(3u);
v___x_43_ = lean_array_fget(v___x_38_, v___x_42_);
if (lean_obj_tag(v___x_43_) == 0)
{
lean_object* v_deBruijnIndex_44_; lean_object* v___x_45_; uint8_t v___x_46_; 
v_deBruijnIndex_44_ = lean_ctor_get(v___x_43_, 0);
lean_inc(v_deBruijnIndex_44_);
lean_dec_ref_known(v___x_43_, 1);
v___x_45_ = lean_unsigned_to_nat(0u);
v___x_46_ = lean_nat_dec_eq(v_deBruijnIndex_44_, v___x_45_);
lean_dec(v_deBruijnIndex_44_);
if (v___x_46_ == 0)
{
lean_object* v___x_47_; 
lean_dec_ref(v___x_38_);
v___x_47_ = lean_box(0);
return v___x_47_;
}
else
{
lean_object* v___x_48_; lean_object* v___x_49_; uint8_t v___x_50_; 
v___x_48_ = lean_array_fget(v___x_38_, v___x_22_);
v___x_49_ = ((lean_object*)(lp_mathlib_Lean_Expr_coeTypeSet_x3f___closed__10));
v___x_50_ = l_Lean_Expr_isAppOfArity(v___x_48_, v___x_49_, v___x_36_);
lean_dec(v___x_48_);
if (v___x_50_ == 0)
{
lean_object* v___x_51_; 
lean_dec_ref(v___x_38_);
v___x_51_ = lean_box(0);
return v___x_51_;
}
else
{
lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; 
v___x_52_ = lean_unsigned_to_nat(4u);
v___x_53_ = lean_array_fget(v___x_38_, v___x_52_);
lean_dec_ref(v___x_38_);
v___x_54_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_54_, 0, v___x_53_);
return v___x_54_;
}
}
}
else
{
lean_object* v___x_55_; 
lean_dec(v___x_43_);
lean_dec_ref(v___x_38_);
v___x_55_ = lean_box(0);
return v___x_55_;
}
}
}
}
else
{
lean_object* v___x_56_; 
lean_dec_ref(v___x_27_);
v___x_56_ = lean_box(0);
return v___x_56_;
}
}
}
else
{
lean_object* v___x_57_; lean_object* v___x_58_; 
v___x_57_ = l_Lean_Expr_appArg_x21(v_e_20_);
v___x_58_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_58_, 0, v___x_57_);
return v___x_58_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_coeTypeSet_x3f___boxed(lean_object* v_e_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_mathlib_Lean_Expr_coeTypeSet_x3f(v_e_59_);
lean_dec_ref(v_e_59_);
return v_res_60_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_CoeSort(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Lean_Expr_ExtraRecognizers(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_CoeSort(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Lean_Expr_ExtraRecognizers(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Set_CoeSort(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Lean_Expr_ExtraRecognizers(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_CoeSort(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Expr_ExtraRecognizers(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Lean_Expr_ExtraRecognizers(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Lean_Expr_ExtraRecognizers(builtin);
}
#ifdef __cplusplus
}
#endif
