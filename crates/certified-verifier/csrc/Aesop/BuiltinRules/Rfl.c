// Lean compiler output
// Module: Aesop.BuiltinRules.Rfl
// Imports: public import Init public meta import Init public import Aesop.Frontend.Attribute
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
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_RuleTac_ofTacticSyntax(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__0_value;
static const lean_string_object lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__1 = (const lean_object*)&lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__1_value;
static const lean_string_object lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__2 = (const lean_object*)&lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__2_value;
static const lean_string_object lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticRfl"};
static const lean_object* lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__3 = (const lean_object*)&lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__3_value;
static const lean_ctor_object lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__4_value_aux_0),((lean_object*)&lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__4_value_aux_1),((lean_object*)&lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__4_value_aux_2),((lean_object*)&lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(201, 188, 173, 198, 169, 252, 183, 45)}};
static const lean_object* lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__4 = (const lean_object*)&lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__4_value;
static const lean_string_object lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rfl"};
static const lean_object* lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__5 = (const lean_object*)&lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__5_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_rfl___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_rfl___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_BuiltinRules_rfl___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_BuiltinRules_rfl___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_BuiltinRules_rfl___closed__0 = (const lean_object*)&lp_aesop_Aesop_BuiltinRules_rfl___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_rfl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_rfl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_rfl___lam__0(lean_object* v_x_11_, lean_object* v___y_12_, lean_object* v___y_13_, lean_object* v___y_14_, lean_object* v___y_15_){
_start:
{
lean_object* v_ref_17_; uint8_t v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___x_24_; 
v_ref_17_ = lean_ctor_get(v___y_14_, 5);
v___x_18_ = 0;
v___x_19_ = l_Lean_SourceInfo_fromRef(v_ref_17_, v___x_18_);
v___x_20_ = ((lean_object*)(lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__4));
v___x_21_ = ((lean_object*)(lp_aesop_Aesop_BuiltinRules_rfl___lam__0___closed__5));
lean_inc(v___x_19_);
v___x_22_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_22_, 0, v___x_19_);
lean_ctor_set(v___x_22_, 1, v___x_21_);
v___x_23_ = l_Lean_Syntax_node1(v___x_19_, v___x_20_, v___x_22_);
v___x_24_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_24_, 0, v___x_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_rfl___lam__0___boxed(lean_object* v_x_25_, lean_object* v___y_26_, lean_object* v___y_27_, lean_object* v___y_28_, lean_object* v___y_29_, lean_object* v___y_30_){
_start:
{
lean_object* v_res_31_; 
v_res_31_ = lp_aesop_Aesop_BuiltinRules_rfl___lam__0(v_x_25_, v___y_26_, v___y_27_, v___y_28_, v___y_29_);
lean_dec(v___y_29_);
lean_dec_ref(v___y_28_);
lean_dec(v___y_27_);
lean_dec_ref(v___y_26_);
lean_dec_ref(v_x_25_);
return v_res_31_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_rfl(lean_object* v_a_33_, lean_object* v_a_34_, lean_object* v_a_35_, lean_object* v_a_36_, lean_object* v_a_37_, lean_object* v_a_38_){
_start:
{
lean_object* v___f_40_; lean_object* v___x_41_; 
v___f_40_ = ((lean_object*)(lp_aesop_Aesop_BuiltinRules_rfl___closed__0));
v___x_41_ = lp_aesop_Aesop_RuleTac_ofTacticSyntax(v___f_40_, v_a_33_, v_a_34_, v_a_35_, v_a_36_, v_a_37_, v_a_38_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BuiltinRules_rfl___boxed(lean_object* v_a_42_, lean_object* v_a_43_, lean_object* v_a_44_, lean_object* v_a_45_, lean_object* v_a_46_, lean_object* v_a_47_, lean_object* v_a_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_aesop_Aesop_BuiltinRules_rfl(v_a_42_, v_a_43_, v_a_44_, v_a_45_, v_a_46_, v_a_47_);
lean_dec(v_a_47_);
lean_dec_ref(v_a_46_);
lean_dec(v_a_45_);
lean_dec_ref(v_a_44_);
lean_dec(v_a_43_);
return v_res_49_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Frontend_Attribute(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_BuiltinRules_Rfl(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Frontend_Attribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_BuiltinRules_Rfl(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Frontend_Attribute(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_BuiltinRules_Rfl(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Frontend_Attribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_BuiltinRules_Rfl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_BuiltinRules_Rfl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_BuiltinRules_Rfl(builtin);
}
#ifdef __cplusplus
}
#endif
