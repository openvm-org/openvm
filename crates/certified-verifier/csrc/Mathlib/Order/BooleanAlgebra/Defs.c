// Lean compiler output
// Module: Mathlib.Order.BooleanAlgebra.Defs
// Imports: public import Init public meta import Init public import Aesop public import Mathlib.Order.Heyting.Basic
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
extern lean_object* lp_mathlib_PUnit_instBiheytingAlgebra;
extern lean_object* lp_mathlib_Prop_instHeytingAlgebra;
lean_object* l_Bool_Internal_not___boxed(lean_object*);
extern lean_object* lp_mathlib_Bool_instDistribLattice;
static const lean_string_object lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__0 = (const lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__0_value;
static const lean_string_object lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__1 = (const lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__1_value;
static const lean_string_object lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__2 = (const lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__2_value;
static const lean_string_object lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__3 = (const lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__3_value;
static const lean_ctor_object lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__4_value_aux_0),((lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__4_value_aux_1),((lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__4_value_aux_2),((lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__4 = (const lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__4_value;
static const lean_array_object lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__5 = (const lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__5_value;
static const lean_string_object lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__6 = (const lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__6_value;
static const lean_ctor_object lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__7_value_aux_0),((lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__7_value_aux_1),((lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__7_value_aux_2),((lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__7 = (const lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__7_value;
static const lean_string_object lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__8 = (const lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__8_value;
static const lean_ctor_object lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__9 = (const lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__9_value;
static const lean_string_object lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Aesop"};
static const lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__10 = (const lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__10_value;
static const lean_string_object lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Frontend"};
static const lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__11 = (const lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__11_value;
static const lean_string_object lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "aesopTactic"};
static const lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__12 = (const lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__12_value;
static const lean_ctor_object lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__10_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__13_value_aux_0),((lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__11_value),LEAN_SCALAR_PTR_LITERAL(212, 61, 86, 210, 90, 71, 197, 0)}};
static const lean_ctor_object lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__13_value_aux_1),((lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(13, 69, 189, 207, 58, 198, 186, 160)}};
static const lean_ctor_object lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__13_value_aux_2),((lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__12_value),LEAN_SCALAR_PTR_LITERAL(54, 142, 162, 195, 161, 101, 248, 175)}};
static const lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__13 = (const lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__13_value;
static const lean_string_object lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "aesop"};
static const lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__14 = (const lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__14_value;
static lean_once_cell_t lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__15;
static lean_once_cell_t lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__16;
static const lean_ctor_object lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__9_value),((lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__5_value)}};
static const lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__17 = (const lean_object*)&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__17_value;
static lean_once_cell_t lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__18;
static lean_once_cell_t lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__19;
static lean_once_cell_t lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__20;
static lean_once_cell_t lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__21;
static lean_once_cell_t lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__22;
static lean_once_cell_t lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__23;
static lean_once_cell_t lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__24;
static lean_once_cell_t lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__25;
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_himp__eq___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_toBoundedOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_toBoundedOrder___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_toBoundedOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_toBoundedOrder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prop_instBooleanAlgebra;
LEAN_EXPORT uint8_t lp_mathlib_Bool_instBooleanAlgebra___lam__0(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Bool_instBooleanAlgebra___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Bool_instBooleanAlgebra___lam__1(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Bool_instBooleanAlgebra___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Bool_instBooleanAlgebra___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Bool_instBooleanAlgebra___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Bool_instBooleanAlgebra___closed__0 = (const lean_object*)&lp_mathlib_Bool_instBooleanAlgebra___closed__0_value;
static const lean_closure_object lp_mathlib_Bool_instBooleanAlgebra___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Bool_instBooleanAlgebra___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Bool_instBooleanAlgebra___closed__1 = (const lean_object*)&lp_mathlib_Bool_instBooleanAlgebra___closed__1_value;
static const lean_closure_object lp_mathlib_Bool_instBooleanAlgebra___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Bool_Internal_not___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Bool_instBooleanAlgebra___closed__2 = (const lean_object*)&lp_mathlib_Bool_instBooleanAlgebra___closed__2_value;
static lean_once_cell_t lp_mathlib_Bool_instBooleanAlgebra___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Bool_instBooleanAlgebra___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Bool_instBooleanAlgebra;
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instBooleanAlgebra;
static lean_object* _init_lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__15(void){
_start:
{
lean_object* v___x_30_; lean_object* v___x_31_; 
v___x_30_ = ((lean_object*)(lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__14));
v___x_31_ = l_Lean_mkAtom(v___x_30_);
return v___x_31_;
}
}
static lean_object* _init_lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__16(void){
_start:
{
lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; 
v___x_32_ = lean_obj_once(&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__15, &lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__15_once, _init_lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__15);
v___x_33_ = ((lean_object*)(lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__5));
v___x_34_ = lean_array_push(v___x_33_, v___x_32_);
return v___x_34_;
}
}
static lean_object* _init_lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__18(void){
_start:
{
lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; 
v___x_39_ = ((lean_object*)(lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__17));
v___x_40_ = lean_obj_once(&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__16, &lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__16_once, _init_lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__16);
v___x_41_ = lean_array_push(v___x_40_, v___x_39_);
return v___x_41_;
}
}
static lean_object* _init_lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__19(void){
_start:
{
lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; 
v___x_42_ = lean_obj_once(&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__18, &lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__18_once, _init_lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__18);
v___x_43_ = ((lean_object*)(lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__13));
v___x_44_ = lean_box(2);
v___x_45_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_45_, 0, v___x_44_);
lean_ctor_set(v___x_45_, 1, v___x_43_);
lean_ctor_set(v___x_45_, 2, v___x_42_);
return v___x_45_;
}
}
static lean_object* _init_lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__20(void){
_start:
{
lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; 
v___x_46_ = lean_obj_once(&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__19, &lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__19_once, _init_lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__19);
v___x_47_ = ((lean_object*)(lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__5));
v___x_48_ = lean_array_push(v___x_47_, v___x_46_);
return v___x_48_;
}
}
static lean_object* _init_lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__21(void){
_start:
{
lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; 
v___x_49_ = lean_obj_once(&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__20, &lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__20_once, _init_lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__20);
v___x_50_ = ((lean_object*)(lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__9));
v___x_51_ = lean_box(2);
v___x_52_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_52_, 0, v___x_51_);
lean_ctor_set(v___x_52_, 1, v___x_50_);
lean_ctor_set(v___x_52_, 2, v___x_49_);
return v___x_52_;
}
}
static lean_object* _init_lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__22(void){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; 
v___x_53_ = lean_obj_once(&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__21, &lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__21_once, _init_lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__21);
v___x_54_ = ((lean_object*)(lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__5));
v___x_55_ = lean_array_push(v___x_54_, v___x_53_);
return v___x_55_;
}
}
static lean_object* _init_lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__23(void){
_start:
{
lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; 
v___x_56_ = lean_obj_once(&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__22, &lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__22_once, _init_lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__22);
v___x_57_ = ((lean_object*)(lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__7));
v___x_58_ = lean_box(2);
v___x_59_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_59_, 0, v___x_58_);
lean_ctor_set(v___x_59_, 1, v___x_57_);
lean_ctor_set(v___x_59_, 2, v___x_56_);
return v___x_59_;
}
}
static lean_object* _init_lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__24(void){
_start:
{
lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; 
v___x_60_ = lean_obj_once(&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__23, &lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__23_once, _init_lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__23);
v___x_61_ = ((lean_object*)(lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__5));
v___x_62_ = lean_array_push(v___x_61_, v___x_60_);
return v___x_62_;
}
}
static lean_object* _init_lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__25(void){
_start:
{
lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; 
v___x_63_ = lean_obj_once(&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__24, &lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__24_once, _init_lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__24);
v___x_64_ = ((lean_object*)(lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__4));
v___x_65_ = lean_box(2);
v___x_66_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_66_, 0, v___x_65_);
lean_ctor_set(v___x_66_, 1, v___x_64_);
lean_ctor_set(v___x_66_, 2, v___x_63_);
return v___x_66_;
}
}
static lean_object* _init_lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam(void){
_start:
{
lean_object* v___x_67_; 
v___x_67_ = lean_obj_once(&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__25, &lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__25_once, _init_lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__25);
return v___x_67_;
}
}
static lean_object* _init_lp_mathlib_BooleanAlgebra_himp__eq___autoParam(void){
_start:
{
lean_object* v___x_68_; 
v___x_68_ = lean_obj_once(&lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__25, &lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__25_once, _init_lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam___closed__25);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_toBoundedOrder___redArg(lean_object* v_h_69_){
_start:
{
lean_object* v_toTop_70_; lean_object* v_toBot_71_; lean_object* v___x_72_; 
v_toTop_70_ = lean_ctor_get(v_h_69_, 4);
v_toBot_71_ = lean_ctor_get(v_h_69_, 5);
lean_inc(v_toBot_71_);
lean_inc(v_toTop_70_);
v___x_72_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_72_, 0, v_toTop_70_);
lean_ctor_set(v___x_72_, 1, v_toBot_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_toBoundedOrder___redArg___boxed(lean_object* v_h_73_){
_start:
{
lean_object* v_res_74_; 
v_res_74_ = lp_mathlib_BooleanAlgebra_toBoundedOrder___redArg(v_h_73_);
lean_dec_ref(v_h_73_);
return v_res_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_toBoundedOrder(lean_object* v_00_u03b1_75_, lean_object* v_h_76_){
_start:
{
lean_object* v___x_77_; 
v___x_77_ = lp_mathlib_BooleanAlgebra_toBoundedOrder___redArg(v_h_76_);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BooleanAlgebra_toBoundedOrder___boxed(lean_object* v_00_u03b1_78_, lean_object* v_h_79_){
_start:
{
lean_object* v_res_80_; 
v_res_80_ = lp_mathlib_BooleanAlgebra_toBoundedOrder(v_00_u03b1_78_, v_h_79_);
lean_dec_ref(v_h_79_);
return v_res_80_;
}
}
static lean_object* _init_lp_mathlib_Prop_instBooleanAlgebra(void){
_start:
{
lean_object* v___x_81_; lean_object* v_toGeneralizedHeytingAlgebra_82_; lean_object* v_toLattice_83_; lean_object* v___x_84_; 
v___x_81_ = lp_mathlib_Prop_instHeytingAlgebra;
v_toGeneralizedHeytingAlgebra_82_ = lean_ctor_get(v___x_81_, 0);
v_toLattice_83_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_82_, 0);
lean_inc_ref(v_toLattice_83_);
v___x_84_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_84_, 0, v_toLattice_83_);
lean_ctor_set(v___x_84_, 1, lean_box(0));
lean_ctor_set(v___x_84_, 2, lean_box(0));
lean_ctor_set(v___x_84_, 3, lean_box(0));
lean_ctor_set(v___x_84_, 4, lean_box(0));
lean_ctor_set(v___x_84_, 5, lean_box(0));
return v___x_84_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Bool_instBooleanAlgebra___lam__0(uint8_t v_x_85_, uint8_t v_y_86_){
_start:
{
if (v_y_86_ == 0)
{
return v_x_85_;
}
else
{
if (v_x_85_ == 0)
{
return v_x_85_;
}
else
{
uint8_t v___x_87_; 
v___x_87_ = 0;
return v___x_87_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bool_instBooleanAlgebra___lam__0___boxed(lean_object* v_x_88_, lean_object* v_y_89_){
_start:
{
uint8_t v_x_boxed_90_; uint8_t v_y_boxed_91_; uint8_t v_res_92_; lean_object* v_r_93_; 
v_x_boxed_90_ = lean_unbox(v_x_88_);
v_y_boxed_91_ = lean_unbox(v_y_89_);
v_res_92_ = lp_mathlib_Bool_instBooleanAlgebra___lam__0(v_x_boxed_90_, v_y_boxed_91_);
v_r_93_ = lean_box(v_res_92_);
return v_r_93_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Bool_instBooleanAlgebra___lam__1(uint8_t v_x_94_, uint8_t v_y_95_){
_start:
{
if (v_x_94_ == 0)
{
if (v_y_95_ == 0)
{
uint8_t v___x_96_; 
v___x_96_ = 1;
return v___x_96_;
}
else
{
return v_y_95_;
}
}
else
{
return v_y_95_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Bool_instBooleanAlgebra___lam__1___boxed(lean_object* v_x_97_, lean_object* v_y_98_){
_start:
{
uint8_t v_x_boxed_99_; uint8_t v_y_boxed_100_; uint8_t v_res_101_; lean_object* v_r_102_; 
v_x_boxed_99_ = lean_unbox(v_x_97_);
v_y_boxed_100_ = lean_unbox(v_y_98_);
v_res_101_ = lp_mathlib_Bool_instBooleanAlgebra___lam__1(v_x_boxed_99_, v_y_boxed_100_);
v_r_102_ = lean_box(v_res_101_);
return v_r_102_;
}
}
static lean_object* _init_lp_mathlib_Bool_instBooleanAlgebra___closed__3(void){
_start:
{
uint8_t v___x_106_; uint8_t v___x_107_; lean_object* v___f_108_; lean_object* v___f_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; 
v___x_106_ = 0;
v___x_107_ = 1;
v___f_108_ = ((lean_object*)(lp_mathlib_Bool_instBooleanAlgebra___closed__1));
v___f_109_ = ((lean_object*)(lp_mathlib_Bool_instBooleanAlgebra___closed__0));
v___x_110_ = ((lean_object*)(lp_mathlib_Bool_instBooleanAlgebra___closed__2));
v___x_111_ = lp_mathlib_Bool_instDistribLattice;
v___x_112_ = lean_box(v___x_107_);
v___x_113_ = lean_box(v___x_106_);
v___x_114_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_114_, 0, v___x_111_);
lean_ctor_set(v___x_114_, 1, v___x_110_);
lean_ctor_set(v___x_114_, 2, v___f_109_);
lean_ctor_set(v___x_114_, 3, v___f_108_);
lean_ctor_set(v___x_114_, 4, v___x_112_);
lean_ctor_set(v___x_114_, 5, v___x_113_);
return v___x_114_;
}
}
static lean_object* _init_lp_mathlib_Bool_instBooleanAlgebra(void){
_start:
{
lean_object* v___x_115_; 
v___x_115_ = lean_obj_once(&lp_mathlib_Bool_instBooleanAlgebra___closed__3, &lp_mathlib_Bool_instBooleanAlgebra___closed__3_once, _init_lp_mathlib_Bool_instBooleanAlgebra___closed__3);
return v___x_115_;
}
}
static lean_object* _init_lp_mathlib_PUnit_instBooleanAlgebra(void){
_start:
{
lean_object* v___x_116_; lean_object* v_toHeytingAlgebra_117_; lean_object* v_toGeneralizedHeytingAlgebra_118_; lean_object* v_toSDiff_119_; lean_object* v_toOrderBot_120_; lean_object* v_toCompl_121_; lean_object* v_toLattice_122_; lean_object* v_toOrderTop_123_; lean_object* v_toHImp_124_; lean_object* v___x_125_; 
v___x_116_ = lp_mathlib_PUnit_instBiheytingAlgebra;
v_toHeytingAlgebra_117_ = lean_ctor_get(v___x_116_, 0);
v_toGeneralizedHeytingAlgebra_118_ = lean_ctor_get(v_toHeytingAlgebra_117_, 0);
v_toSDiff_119_ = lean_ctor_get(v___x_116_, 1);
v_toOrderBot_120_ = lean_ctor_get(v_toHeytingAlgebra_117_, 1);
v_toCompl_121_ = lean_ctor_get(v_toHeytingAlgebra_117_, 2);
v_toLattice_122_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_118_, 0);
v_toOrderTop_123_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_118_, 1);
v_toHImp_124_ = lean_ctor_get(v_toGeneralizedHeytingAlgebra_118_, 2);
lean_inc(v_toOrderBot_120_);
lean_inc(v_toOrderTop_123_);
lean_inc(v_toHImp_124_);
lean_inc(v_toSDiff_119_);
lean_inc(v_toCompl_121_);
lean_inc_ref(v_toLattice_122_);
v___x_125_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_125_, 0, v_toLattice_122_);
lean_ctor_set(v___x_125_, 1, v_toCompl_121_);
lean_ctor_set(v___x_125_, 2, v_toSDiff_119_);
lean_ctor_set(v___x_125_, 3, v_toHImp_124_);
lean_ctor_set(v___x_125_, 4, v_toOrderTop_123_);
lean_ctor_set(v___x_125_, 5, v_toOrderBot_120_);
return v___x_125_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Heyting_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_BooleanAlgebra_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Heyting_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Prop_instBooleanAlgebra = _init_lp_mathlib_Prop_instBooleanAlgebra();
lean_mark_persistent(lp_mathlib_Prop_instBooleanAlgebra);
lp_mathlib_Bool_instBooleanAlgebra = _init_lp_mathlib_Bool_instBooleanAlgebra();
lean_mark_persistent(lp_mathlib_Bool_instBooleanAlgebra);
lp_mathlib_PUnit_instBooleanAlgebra = _init_lp_mathlib_PUnit_instBooleanAlgebra();
lean_mark_persistent(lp_mathlib_PUnit_instBooleanAlgebra);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_BooleanAlgebra_Defs(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam = _init_lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam();
lean_mark_persistent(lp_mathlib_BooleanAlgebra_sdiff__eq___autoParam);
lp_mathlib_BooleanAlgebra_himp__eq___autoParam = _init_lp_mathlib_BooleanAlgebra_himp__eq___autoParam();
lean_mark_persistent(lp_mathlib_BooleanAlgebra_himp__eq___autoParam);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Heyting_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_BooleanAlgebra_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Heyting_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_BooleanAlgebra_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_BooleanAlgebra_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_BooleanAlgebra_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
