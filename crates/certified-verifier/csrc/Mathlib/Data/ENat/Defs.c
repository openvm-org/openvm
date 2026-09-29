// Lean compiler output
// Module: Mathlib.Data.ENat.Defs
// Imports: public import Init public meta import Init public import Batteries.Tactic.Alias public import Mathlib.Data.Nat.Notation public import Mathlib.Order.TypeTags
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
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_some(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTopENat___aux__1;
LEAN_EXPORT lean_object* lp_mathlib_instTopENat;
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedENat___aux__1;
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedENat;
static const lean_string_object lp_mathlib_term_u2115_u221e___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 6, .m_data = "termℕ∞"};
static const lean_object* lp_mathlib_term_u2115_u221e___closed__0 = (const lean_object*)&lp_mathlib_term_u2115_u221e___closed__0_value;
static const lean_ctor_object lp_mathlib_term_u2115_u221e___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_u2115_u221e___closed__0_value),LEAN_SCALAR_PTR_LITERAL(249, 167, 24, 203, 93, 182, 115, 9)}};
static const lean_object* lp_mathlib_term_u2115_u221e___closed__1 = (const lean_object*)&lp_mathlib_term_u2115_u221e___closed__1_value;
static const lean_string_object lp_mathlib_term_u2115_u221e___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 2, .m_data = "ℕ∞"};
static const lean_object* lp_mathlib_term_u2115_u221e___closed__2 = (const lean_object*)&lp_mathlib_term_u2115_u221e___closed__2_value;
static const lean_ctor_object lp_mathlib_term_u2115_u221e___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term_u2115_u221e___closed__2_value)}};
static const lean_object* lp_mathlib_term_u2115_u221e___closed__3 = (const lean_object*)&lp_mathlib_term_u2115_u221e___closed__3_value;
static const lean_ctor_object lp_mathlib_term_u2115_u221e___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_term_u2115_u221e___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_term_u2115_u221e___closed__3_value)}};
static const lean_object* lp_mathlib_term_u2115_u221e___closed__4 = (const lean_object*)&lp_mathlib_term_u2115_u221e___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_term_u2115_u221e = (const lean_object*)&lp_mathlib_term_u2115_u221e___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "ENat"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(54, 10, 120, 91, 211, 186, 213, 159)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_ENat_instNatCast___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithTop_some, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_ENat_instNatCast___closed__0 = (const lean_object*)&lp_mathlib_ENat_instNatCast___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_ENat_instNatCast = (const lean_object*)&lp_mathlib_ENat_instNatCast___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_ENat_recTopCoe___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ENat_recTopCoe___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ENat_recTopCoe(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ENat_recTopCoe___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_instTopENat___aux__1(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lean_box(0);
return v___x_1_;
}
}
static lean_object* _init_lp_mathlib_instTopENat(void){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_box(0);
return v___x_2_;
}
}
static lean_object* _init_lp_mathlib_instInhabitedENat___aux__1(void){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
}
static lean_object* _init_lp_mathlib_instInhabitedENat(void){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_box(0);
return v___x_4_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__1(void){
_start:
{
lean_object* v___x_17_; lean_object* v___x_18_; 
v___x_17_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__0));
v___x_18_ = l_String_toRawSubstring_x27(v___x_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1(lean_object* v_x_27_, lean_object* v_a_28_, lean_object* v_a_29_){
_start:
{
lean_object* v___x_30_; uint8_t v___x_31_; 
v___x_30_ = ((lean_object*)(lp_mathlib_term_u2115_u221e___closed__1));
v___x_31_ = l_Lean_Syntax_isOfKind(v_x_27_, v___x_30_);
if (v___x_31_ == 0)
{
lean_object* v___x_32_; lean_object* v___x_33_; 
v___x_32_ = lean_box(1);
v___x_33_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_33_, 0, v___x_32_);
lean_ctor_set(v___x_33_, 1, v_a_29_);
return v___x_33_;
}
else
{
lean_object* v_quotContext_34_; lean_object* v_currMacroScope_35_; lean_object* v_ref_36_; uint8_t v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; 
v_quotContext_34_ = lean_ctor_get(v_a_28_, 1);
v_currMacroScope_35_ = lean_ctor_get(v_a_28_, 2);
v_ref_36_ = lean_ctor_get(v_a_28_, 5);
v___x_37_ = 0;
v___x_38_ = l_Lean_SourceInfo_fromRef(v_ref_36_, v___x_37_);
v___x_39_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__1, &lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__1);
v___x_40_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__2));
lean_inc(v_currMacroScope_35_);
lean_inc(v_quotContext_34_);
v___x_41_ = l_Lean_addMacroScope(v_quotContext_34_, v___x_40_, v_currMacroScope_35_);
v___x_42_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___closed__4));
v___x_43_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_43_, 0, v___x_38_);
lean_ctor_set(v___x_43_, 1, v___x_39_);
lean_ctor_set(v___x_43_, 2, v___x_41_);
lean_ctor_set(v___x_43_, 3, v___x_42_);
v___x_44_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_44_, 0, v___x_43_);
lean_ctor_set(v___x_44_, 1, v_a_29_);
return v___x_44_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1___boxed(lean_object* v_x_45_, lean_object* v_a_46_, lean_object* v_a_47_){
_start:
{
lean_object* v_res_48_; 
v_res_48_ = lp_mathlib___aux__Mathlib__Data__ENat__Defs______macroRules__term_u2115_u221e__1(v_x_45_, v_a_46_, v_a_47_);
lean_dec_ref(v_a_46_);
return v_res_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1(lean_object* v_x_52_, lean_object* v_a_53_, lean_object* v_a_54_){
_start:
{
lean_object* v___x_55_; uint8_t v___x_56_; 
v___x_55_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1___closed__1));
lean_inc(v_x_52_);
v___x_56_ = l_Lean_Syntax_isOfKind(v_x_52_, v___x_55_);
if (v___x_56_ == 0)
{
lean_object* v___x_57_; lean_object* v___x_58_; 
lean_dec(v_x_52_);
v___x_57_ = lean_box(0);
v___x_58_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_58_, 0, v___x_57_);
lean_ctor_set(v___x_58_, 1, v_a_54_);
return v___x_58_;
}
else
{
lean_object* v_ref_59_; uint8_t v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; 
v_ref_59_ = l_Lean_replaceRef(v_x_52_, v_a_53_);
lean_dec(v_x_52_);
v___x_60_ = 0;
v___x_61_ = l_Lean_SourceInfo_fromRef(v_ref_59_, v___x_60_);
lean_dec(v_ref_59_);
v___x_62_ = ((lean_object*)(lp_mathlib_term_u2115_u221e___closed__1));
v___x_63_ = ((lean_object*)(lp_mathlib_term_u2115_u221e___closed__2));
lean_inc(v___x_61_);
v___x_64_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_64_, 0, v___x_61_);
lean_ctor_set(v___x_64_, 1, v___x_63_);
v___x_65_ = l_Lean_Syntax_node1(v___x_61_, v___x_62_, v___x_64_);
v___x_66_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_66_, 0, v___x_65_);
lean_ctor_set(v___x_66_, 1, v_a_54_);
return v___x_66_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1___boxed(lean_object* v_x_67_, lean_object* v_a_68_, lean_object* v_a_69_){
_start:
{
lean_object* v_res_70_; 
v_res_70_ = lp_mathlib___aux__Mathlib__Data__ENat__Defs______unexpand__ENat__1(v_x_67_, v_a_68_, v_a_69_);
lean_dec(v_a_68_);
return v_res_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ENat_recTopCoe___redArg(lean_object* v_top_73_, lean_object* v_coe_74_, lean_object* v_x_75_){
_start:
{
if (lean_obj_tag(v_x_75_) == 0)
{
lean_dec(v_coe_74_);
lean_inc(v_top_73_);
return v_top_73_;
}
else
{
lean_object* v_val_76_; lean_object* v___x_77_; 
v_val_76_ = lean_ctor_get(v_x_75_, 0);
lean_inc(v_val_76_);
lean_dec_ref_known(v_x_75_, 1);
v___x_77_ = lean_apply_1(v_coe_74_, v_val_76_);
return v___x_77_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ENat_recTopCoe___redArg___boxed(lean_object* v_top_78_, lean_object* v_coe_79_, lean_object* v_x_80_){
_start:
{
lean_object* v_res_81_; 
v_res_81_ = lp_mathlib_ENat_recTopCoe___redArg(v_top_78_, v_coe_79_, v_x_80_);
lean_dec(v_top_78_);
return v_res_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ENat_recTopCoe(lean_object* v_C_82_, lean_object* v_top_83_, lean_object* v_coe_84_, lean_object* v_x_85_){
_start:
{
lean_object* v___x_86_; 
v___x_86_ = lp_mathlib_ENat_recTopCoe___redArg(v_top_83_, v_coe_84_, v_x_85_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ENat_recTopCoe___boxed(lean_object* v_C_87_, lean_object* v_top_88_, lean_object* v_coe_89_, lean_object* v_x_90_){
_start:
{
lean_object* v_res_91_; 
v_res_91_ = lp_mathlib_ENat_recTopCoe(v_C_87_, v_top_88_, v_coe_89_, v_x_90_);
lean_dec(v_top_88_);
return v_res_91_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_TypeTags(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_ENat_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Alias(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_TypeTags(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_instTopENat___aux__1 = _init_lp_mathlib_instTopENat___aux__1();
lean_mark_persistent(lp_mathlib_instTopENat___aux__1);
lp_mathlib_instTopENat = _init_lp_mathlib_instTopENat();
lean_mark_persistent(lp_mathlib_instTopENat);
lp_mathlib_instInhabitedENat___aux__1 = _init_lp_mathlib_instInhabitedENat___aux__1();
lean_mark_persistent(lp_mathlib_instInhabitedENat___aux__1);
lp_mathlib_instInhabitedENat = _init_lp_mathlib_instInhabitedENat();
lean_mark_persistent(lp_mathlib_instInhabitedENat);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_ENat_Defs(uint8_t builtin) {
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
lean_object* initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_TypeTags(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_ENat_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Alias(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_TypeTags(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_ENat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_ENat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_ENat_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
