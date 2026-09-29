// Lean compiler output
// Module: Mathlib.Data.Nat.Notation
// Imports: public import Init public meta import Init public import Mathlib.Init
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term_u2115___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 5, .m_data = "termℕ"};
static const lean_object* lp_mathlib_term_u2115___closed__0 = (const lean_object*)&lp_mathlib_term_u2115___closed__0_value;
static const lean_ctor_object lp_mathlib_term_u2115___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_u2115___closed__0_value),LEAN_SCALAR_PTR_LITERAL(182, 70, 148, 59, 158, 17, 147, 87)}};
static const lean_object* lp_mathlib_term_u2115___closed__1 = (const lean_object*)&lp_mathlib_term_u2115___closed__1_value;
static const lean_string_object lp_mathlib_term_u2115___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "ℕ"};
static const lean_object* lp_mathlib_term_u2115___closed__2 = (const lean_object*)&lp_mathlib_term_u2115___closed__2_value;
static const lean_ctor_object lp_mathlib_term_u2115___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term_u2115___closed__2_value)}};
static const lean_object* lp_mathlib_term_u2115___closed__3 = (const lean_object*)&lp_mathlib_term_u2115___closed__3_value;
static const lean_ctor_object lp_mathlib_term_u2115___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_term_u2115___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_term_u2115___closed__3_value)}};
static const lean_object* lp_mathlib_term_u2115___closed__4 = (const lean_object*)&lp_mathlib_term_u2115___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_term_u2115 = (const lean_object*)&lp_mathlib_term_u2115___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__2_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__3_value),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__5_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Nat__Notation______unexpand__Nat__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Nat__Notation______unexpand__Nat__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Nat__Notation______unexpand__Nat__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Nat__Notation______unexpand__Nat__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Nat__Notation______unexpand__Nat__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Nat__Notation______unexpand__Nat__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Nat__Notation______unexpand__Nat__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Nat__Notation______unexpand__Nat__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Nat__Notation______unexpand__Nat__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__1(void){
_start:
{
lean_object* v___x_13_; lean_object* v___x_14_; 
v___x_13_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__0));
v___x_14_ = l_String_toRawSubstring_x27(v___x_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1(lean_object* v_x_28_, lean_object* v_a_29_, lean_object* v_a_30_){
_start:
{
lean_object* v___x_31_; uint8_t v___x_32_; 
v___x_31_ = ((lean_object*)(lp_mathlib_term_u2115___closed__1));
v___x_32_ = l_Lean_Syntax_isOfKind(v_x_28_, v___x_31_);
if (v___x_32_ == 0)
{
lean_object* v___x_33_; lean_object* v___x_34_; 
v___x_33_ = lean_box(1);
v___x_34_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_34_, 0, v___x_33_);
lean_ctor_set(v___x_34_, 1, v_a_30_);
return v___x_34_;
}
else
{
lean_object* v_quotContext_35_; lean_object* v_currMacroScope_36_; lean_object* v_ref_37_; uint8_t v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; 
v_quotContext_35_ = lean_ctor_get(v_a_29_, 1);
v_currMacroScope_36_ = lean_ctor_get(v_a_29_, 2);
v_ref_37_ = lean_ctor_get(v_a_29_, 5);
v___x_38_ = 0;
v___x_39_ = l_Lean_SourceInfo_fromRef(v_ref_37_, v___x_38_);
v___x_40_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__1, &lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__1);
v___x_41_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__2));
lean_inc(v_currMacroScope_36_);
lean_inc(v_quotContext_35_);
v___x_42_ = l_Lean_addMacroScope(v_quotContext_35_, v___x_41_, v_currMacroScope_36_);
v___x_43_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___closed__6));
v___x_44_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_44_, 0, v___x_39_);
lean_ctor_set(v___x_44_, 1, v___x_40_);
lean_ctor_set(v___x_44_, 2, v___x_42_);
lean_ctor_set(v___x_44_, 3, v___x_43_);
v___x_45_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_45_, 0, v___x_44_);
lean_ctor_set(v___x_45_, 1, v_a_30_);
return v___x_45_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1___boxed(lean_object* v_x_46_, lean_object* v_a_47_, lean_object* v_a_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_mathlib___aux__Mathlib__Data__Nat__Notation______macroRules__term_u2115__1(v_x_46_, v_a_47_, v_a_48_);
lean_dec_ref(v_a_47_);
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Nat__Notation______unexpand__Nat__1(lean_object* v_x_53_, lean_object* v_a_54_, lean_object* v_a_55_){
_start:
{
lean_object* v___x_56_; uint8_t v___x_57_; 
v___x_56_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Nat__Notation______unexpand__Nat__1___closed__1));
lean_inc(v_x_53_);
v___x_57_ = l_Lean_Syntax_isOfKind(v_x_53_, v___x_56_);
if (v___x_57_ == 0)
{
lean_object* v___x_58_; lean_object* v___x_59_; 
lean_dec(v_x_53_);
v___x_58_ = lean_box(0);
v___x_59_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_59_, 0, v___x_58_);
lean_ctor_set(v___x_59_, 1, v_a_55_);
return v___x_59_;
}
else
{
lean_object* v_ref_60_; uint8_t v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; 
v_ref_60_ = l_Lean_replaceRef(v_x_53_, v_a_54_);
lean_dec(v_x_53_);
v___x_61_ = 0;
v___x_62_ = l_Lean_SourceInfo_fromRef(v_ref_60_, v___x_61_);
lean_dec(v_ref_60_);
v___x_63_ = ((lean_object*)(lp_mathlib_term_u2115___closed__1));
v___x_64_ = ((lean_object*)(lp_mathlib_term_u2115___closed__2));
lean_inc(v___x_62_);
v___x_65_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_65_, 0, v___x_62_);
lean_ctor_set(v___x_65_, 1, v___x_64_);
v___x_66_ = l_Lean_Syntax_node1(v___x_62_, v___x_63_, v___x_65_);
v___x_67_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_67_, 0, v___x_66_);
lean_ctor_set(v___x_67_, 1, v_a_55_);
return v___x_67_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Nat__Notation______unexpand__Nat__1___boxed(lean_object* v_x_68_, lean_object* v_a_69_, lean_object* v_a_70_){
_start:
{
lean_object* v_res_71_; 
v_res_71_ = lp_mathlib___aux__Mathlib__Data__Nat__Notation______unexpand__Nat__1(v_x_68_, v_a_69_, v_a_70_);
lean_dec(v_a_69_);
return v_res_71_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin) {
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
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin) {
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
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
}
#ifdef __cplusplus
}
#endif
