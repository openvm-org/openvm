// Lean compiler output
// Module: Mathlib.Tactic.Attr.Register
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.LabelAttribute public import Lean.LabelAttribute public import Lean.Meta.Tactic.Simp
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
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Meta_registerSimpAttr(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_simpPost;
extern lean_object* l_Lean_Parser_Tactic_simpPre;
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_registerSimprocAttr(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_registerLabelAttr(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "functor_norm"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(231, 168, 30, 17, 22, 99, 221, 74)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "Simp set for `functor_norm` "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "ext"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(241, 12, 90, 240, 78, 252, 149, 89)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(92, 126, 46, 149, 51, 125, 16, 48)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__7_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__7_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__7_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__8_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__7_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(237, 143, 4, 212, 196, 7, 65, 73)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__8_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__8_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__10_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__8_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(84, 15, 245, 124, 246, 224, 156, 196)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__10_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__10_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Attr"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__12_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__10_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(248, 197, 98, 97, 188, 57, 150, 89)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__12_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__12_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__13_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Register"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__13_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__13_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__14_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__12_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__13_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(25, 11, 249, 139, 134, 11, 215, 90)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__14_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__14_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__15_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__15_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__17_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__17_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__19_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__19_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__20_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__20_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_3_;
static const lean_string_object lp_mathlib_Parser_Attr_functor__norm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Parser_Attr_functor__norm___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_functor__norm___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_functor__norm___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_functor__norm___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(194, 81, 13, 229, 29, 153, 199, 112)}};
static const lean_object* lp_mathlib_Parser_Attr_functor__norm___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__1_value;
static const lean_string_object lp_mathlib_Parser_Attr_functor__norm___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Parser_Attr_functor__norm___closed__2 = (const lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__2_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_functor__norm___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Parser_Attr_functor__norm___closed__3 = (const lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__3_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_functor__norm___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_functor__norm___closed__4 = (const lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__4_value;
static const lean_string_object lp_mathlib_Parser_Attr_functor__norm___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Parser_Attr_functor__norm___closed__5 = (const lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__5_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_functor__norm___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__5_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Parser_Attr_functor__norm___closed__6 = (const lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__6_value;
static const lean_string_object lp_mathlib_Parser_Attr_functor__norm___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "orelse"};
static const lean_object* lp_mathlib_Parser_Attr_functor__norm___closed__7 = (const lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__7_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_functor__norm___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__7_value),LEAN_SCALAR_PTR_LITERAL(78, 76, 4, 51, 251, 212, 116, 5)}};
static const lean_object* lp_mathlib_Parser_Attr_functor__norm___closed__8 = (const lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__8_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_functor__norm___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_functor__norm___closed__9;
static lean_once_cell_t lp_mathlib_Parser_Attr_functor__norm___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_functor__norm___closed__10;
static lean_once_cell_t lp_mathlib_Parser_Attr_functor__norm___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_functor__norm___closed__11;
static const lean_string_object lp_mathlib_Parser_Attr_functor__norm___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = " ←"};
static const lean_object* lp_mathlib_Parser_Attr_functor__norm___closed__12 = (const lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__12_value;
static const lean_string_object lp_mathlib_Parser_Attr_functor__norm___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " <-"};
static const lean_object* lp_mathlib_Parser_Attr_functor__norm___closed__13 = (const lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__13_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_functor__norm___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 12}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__12_value),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__13_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_functor__norm___closed__14 = (const lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__14_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_functor__norm___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__6_value),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__14_value)}};
static const lean_object* lp_mathlib_Parser_Attr_functor__norm___closed__15 = (const lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__15_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_functor__norm___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_functor__norm___closed__16;
static const lean_string_object lp_mathlib_Parser_Attr_functor__norm___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "prio"};
static const lean_object* lp_mathlib_Parser_Attr_functor__norm___closed__17 = (const lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__17_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_functor__norm___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__17_value),LEAN_SCALAR_PTR_LITERAL(122, 247, 65, 238, 243, 154, 137, 247)}};
static const lean_object* lp_mathlib_Parser_Attr_functor__norm___closed__18 = (const lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__18_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_functor__norm___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__18_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Parser_Attr_functor__norm___closed__19 = (const lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__19_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_functor__norm___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__6_value),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__19_value)}};
static const lean_object* lp_mathlib_Parser_Attr_functor__norm___closed__20 = (const lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__20_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_functor__norm___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_functor__norm___closed__21;
static lean_once_cell_t lp_mathlib_Parser_Attr_functor__norm___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_functor__norm___closed__22;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_functor__norm;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "functor_norm_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(230, 195, 176, 41, 180, 210, 41, 123)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "simproc set for functor_norm_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "extProc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(153, 229, 121, 159, 4, 151, 74, 120)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(212, 140, 237, 18, 250, 244, 167, 98)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__7_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(197, 62, 250, 110, 78, 137, 47, 111)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__7_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(156, 105, 51, 109, 106, 142, 39, 103)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__7_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__7_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__8_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__7_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(16, 71, 226, 87, 19, 78, 148, 3)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__8_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__8_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__8_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__13_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(97, 224, 167, 168, 122, 132, 15, 160)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__10_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__10_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__12_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__12_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__13_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__13_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_functor__norm__proc___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_functor__norm__proc___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_functor__norm__proc___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_functor__norm__proc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_functor__norm__proc___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(235, 109, 19, 222, 94, 13, 94, 20)}};
static const lean_object* lp_mathlib_Parser_Attr_functor__norm__proc___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_functor__norm__proc___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_functor__norm__proc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_functor__norm__proc___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_functor__norm__proc___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_functor__norm__proc___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_functor__norm__proc___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_functor__norm__proc___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_functor__norm__proc___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_functor__norm__proc;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "monad_norm"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(17, 18, 66, 45, 151, 225, 61, 91)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_monad__norm___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_monad__norm___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_monad__norm___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_monad__norm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_monad__norm___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(236, 110, 10, 51, 227, 231, 194, 76)}};
static const lean_object* lp_mathlib_Parser_Attr_monad__norm___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_monad__norm___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_monad__norm___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_monad__norm___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_monad__norm___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_monad__norm___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_monad__norm___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_monad__norm___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_monad__norm___closed__3;
static lean_once_cell_t lp_mathlib_Parser_Attr_monad__norm___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_monad__norm___closed__4;
static lean_once_cell_t lp_mathlib_Parser_Attr_monad__norm___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_monad__norm___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_monad__norm;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "monad_norm_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(254, 138, 210, 9, 2, 140, 98, 210)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "simproc set for monad_norm_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_monad__norm__proc___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_monad__norm__proc___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_monad__norm__proc___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_monad__norm__proc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_monad__norm__proc___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(3, 99, 219, 179, 99, 99, 251, 165)}};
static const lean_object* lp_mathlib_Parser_Attr_monad__norm__proc___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_monad__norm__proc___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_monad__norm__proc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_monad__norm__proc___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_monad__norm__proc___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_monad__norm__proc___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_monad__norm__proc___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_monad__norm__proc___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_monad__norm__proc___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_monad__norm__proc;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "parity_simps"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(50, 224, 107, 174, 201, 74, 18, 50)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 40, .m_capacity = 40, .m_length = 39, .m_data = "Simp attribute for lemmas about `Even` "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__14_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),((lean_object*)(((size_t)(1568842782) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(37, 22, 198, 43, 163, 70, 40, 124)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(246, 1, 126, 12, 182, 9, 84, 45)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(74, 80, 107, 59, 245, 60, 86, 21)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(215, 233, 160, 182, 116, 96, 156, 16)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_parity__simps___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_parity__simps___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_parity__simps___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_parity__simps___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_parity__simps___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(247, 36, 26, 83, 89, 225, 192, 192)}};
static const lean_object* lp_mathlib_Parser_Attr_parity__simps___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_parity__simps___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_parity__simps___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_parity__simps___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_parity__simps___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_parity__simps___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_parity__simps___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_parity__simps___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_parity__simps___closed__3;
static lean_once_cell_t lp_mathlib_Parser_Attr_parity__simps___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_parity__simps___closed__4;
static lean_once_cell_t lp_mathlib_Parser_Attr_parity__simps___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_parity__simps___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_parity__simps;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "parity_simps_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(239, 84, 131, 242, 127, 74, 253, 141)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "simproc set for parity_simps_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value),((lean_object*)(((size_t)(1568842782) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(221, 73, 228, 28, 233, 94, 219, 6)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(94, 196, 224, 182, 172, 82, 151, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(210, 72, 253, 217, 59, 132, 163, 137)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(95, 49, 54, 171, 157, 48, 206, 109)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_parity__simps__proc___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_parity__simps__proc___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_parity__simps__proc___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_parity__simps__proc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_parity__simps__proc___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(234, 86, 180, 12, 99, 28, 46, 216)}};
static const lean_object* lp_mathlib_Parser_Attr_parity__simps__proc___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_parity__simps__proc___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_parity__simps__proc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_parity__simps__proc___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_parity__simps__proc___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_parity__simps__proc___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_parity__simps__proc___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_parity__simps__proc___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_parity__simps__proc___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_parity__simps__proc;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "rclike_simps"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(31, 32, 222, 207, 9, 100, 103, 250)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 44, .m_capacity = 44, .m_length = 43, .m_data = "\"Simp attribute for lemmas about `RCLike`\" "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_rclike__simps___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_rclike__simps___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_rclike__simps___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_rclike__simps___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_rclike__simps___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(186, 107, 50, 42, 210, 217, 209, 246)}};
static const lean_object* lp_mathlib_Parser_Attr_rclike__simps___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_rclike__simps___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_rclike__simps___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_rclike__simps___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_rclike__simps___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_rclike__simps___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_rclike__simps___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_rclike__simps___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_rclike__simps___closed__3;
static lean_once_cell_t lp_mathlib_Parser_Attr_rclike__simps___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_rclike__simps___closed__4;
static lean_once_cell_t lp_mathlib_Parser_Attr_rclike__simps___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_rclike__simps___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_rclike__simps;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "rclike_simps_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(223, 93, 198, 118, 206, 99, 28, 146)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "simproc set for rclike_simps_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_rclike__simps__proc___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_rclike__simps__proc___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_rclike__simps__proc___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_rclike__simps__proc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_rclike__simps__proc___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(122, 35, 135, 243, 60, 170, 99, 136)}};
static const lean_object* lp_mathlib_Parser_Attr_rclike__simps__proc___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_rclike__simps__proc___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_rclike__simps__proc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_rclike__simps__proc___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_rclike__simps__proc___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_rclike__simps__proc___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_rclike__simps__proc___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_rclike__simps__proc___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_rclike__simps__proc___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_rclike__simps__proc;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "rify_simps"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(95, 204, 117, 112, 29, 134, 158, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 113, .m_capacity = 113, .m_length = 104, .m_data = "The simpset `rify_simps` is used by the tactic `rify` to move expressions from `ℕ`, `ℤ`, or\n`ℚ` to `ℝ`. "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__14_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),((lean_object*)(((size_t)(612238087) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(251, 223, 44, 17, 79, 121, 61, 49)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(64, 106, 186, 177, 106, 31, 79, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(164, 6, 137, 34, 100, 132, 68, 123)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(121, 146, 133, 80, 224, 96, 93, 5)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_rify__simps___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_rify__simps___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_rify__simps___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_rify__simps___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_rify__simps___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(250, 84, 138, 1, 93, 36, 81, 63)}};
static const lean_object* lp_mathlib_Parser_Attr_rify__simps___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_rify__simps___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_rify__simps___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_rify__simps___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_rify__simps___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_rify__simps___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_rify__simps___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_rify__simps___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_rify__simps___closed__3;
static lean_once_cell_t lp_mathlib_Parser_Attr_rify__simps___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_rify__simps___closed__4;
static lean_once_cell_t lp_mathlib_Parser_Attr_rify__simps___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_rify__simps___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_rify__simps;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "rify_simps_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(37, 131, 101, 10, 72, 151, 40, 72)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "simproc set for rify_simps_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value),((lean_object*)(((size_t)(612238087) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(3, 115, 190, 13, 137, 235, 205, 154)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(24, 196, 15, 183, 103, 85, 174, 200)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(60, 129, 140, 101, 3, 199, 35, 22)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(81, 251, 77, 131, 141, 66, 75, 37)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_rify__simps__proc___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_rify__simps__proc___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_rify__simps__proc___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_rify__simps__proc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_rify__simps__proc___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(96, 66, 58, 42, 54, 224, 119, 88)}};
static const lean_object* lp_mathlib_Parser_Attr_rify__simps__proc___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_rify__simps__proc___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_rify__simps__proc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_rify__simps__proc___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_rify__simps__proc___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_rify__simps__proc___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_rify__simps__proc___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_rify__simps__proc___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_rify__simps__proc___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_rify__simps__proc;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "qify_simps"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(126, 72, 43, 208, 215, 37, 114, 190)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 141, .m_capacity = 141, .m_length = 134, .m_data = "The simpset `qify_simps` is used by the tactic `qify` to move expressions from `ℕ` or `ℤ` to `ℚ`\nwhich gives a well-behaved division. "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__14_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),((lean_object*)(((size_t)(1503578496) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(254, 225, 46, 139, 10, 237, 149, 254)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(41, 48, 248, 204, 184, 191, 215, 223)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(193, 18, 204, 159, 185, 198, 237, 191)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(240, 43, 70, 235, 59, 59, 201, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_qify__simps___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_qify__simps___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_qify__simps___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_qify__simps___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_qify__simps___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(131, 5, 145, 201, 202, 70, 159, 203)}};
static const lean_object* lp_mathlib_Parser_Attr_qify__simps___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_qify__simps___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_qify__simps___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_qify__simps___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_qify__simps___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_qify__simps___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_qify__simps___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_qify__simps___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_qify__simps___closed__3;
static lean_once_cell_t lp_mathlib_Parser_Attr_qify__simps___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_qify__simps___closed__4;
static lean_once_cell_t lp_mathlib_Parser_Attr_qify__simps___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_qify__simps___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_qify__simps;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "qify_simps_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(26, 41, 170, 253, 230, 32, 124, 180)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "simproc set for qify_simps_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value),((lean_object*)(((size_t)(1503578496) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(214, 146, 165, 232, 146, 62, 38, 188)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(113, 32, 204, 211, 25, 104, 147, 1)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(137, 119, 174, 21, 179, 166, 4, 39)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(216, 230, 145, 89, 119, 58, 209, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_qify__simps__proc___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_qify__simps__proc___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_qify__simps__proc___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_qify__simps__proc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_qify__simps__proc___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(63, 120, 97, 214, 217, 234, 213, 116)}};
static const lean_object* lp_mathlib_Parser_Attr_qify__simps__proc___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_qify__simps__proc___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_qify__simps__proc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_qify__simps__proc___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_qify__simps__proc___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_qify__simps__proc___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_qify__simps__proc___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_qify__simps__proc___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_qify__simps__proc___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_qify__simps__proc;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "zify_simps"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(83, 133, 36, 222, 110, 71, 247, 20)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 135, .m_capacity = 135, .m_length = 130, .m_data = "The simpset `zify_simps` is used by the tactic `zify` to move expressions from `ℕ` to `ℤ`\nwhich gives a well-behaved subtraction. "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_zify__simps___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_zify__simps___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_zify__simps___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_zify__simps___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_zify__simps___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(46, 115, 99, 166, 132, 223, 64, 152)}};
static const lean_object* lp_mathlib_Parser_Attr_zify__simps___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_zify__simps___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_zify__simps___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_zify__simps___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_zify__simps___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_zify__simps___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_zify__simps___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_zify__simps___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_zify__simps___closed__3;
static lean_once_cell_t lp_mathlib_Parser_Attr_zify__simps___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_zify__simps___closed__4;
static lean_once_cell_t lp_mathlib_Parser_Attr_zify__simps___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_zify__simps___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_zify__simps;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "zify_simps_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(254, 71, 143, 219, 57, 87, 228, 109)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "simproc set for zify_simps_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_zify__simps__proc___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_zify__simps__proc___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_zify__simps__proc___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_zify__simps__proc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_zify__simps__proc___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(3, 70, 190, 73, 167, 115, 62, 37)}};
static const lean_object* lp_mathlib_Parser_Attr_zify__simps__proc___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_zify__simps__proc___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_zify__simps__proc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_zify__simps__proc___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_zify__simps__proc___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_zify__simps__proc___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_zify__simps__proc___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_zify__simps__proc___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_zify__simps__proc___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_zify__simps__proc;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "pull_end"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(49, 233, 153, 170, 209, 15, 209, 87)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 246, .m_capacity = 246, .m_length = 242, .m_data = "The simpset `pull_end` translates algebraic formulations of endomorphisms into the standard\nformulation of homomorphisms, so for example `1 : Equiv α α` becomes `Equiv.refl α` and\n`a * b` becomes `b.trans a`.\n\nThe dual simpset is `push_end`.\n"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__14_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),((lean_object*)(((size_t)(1133190711) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(143, 157, 210, 109, 253, 234, 126, 38)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(68, 225, 113, 184, 11, 46, 65, 222)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(80, 121, 198, 204, 119, 178, 76, 224)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(181, 205, 225, 247, 9, 173, 103, 248)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_pull__end___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_pull__end___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_pull__end___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_pull__end___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_pull__end___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(204, 151, 138, 200, 247, 61, 239, 12)}};
static const lean_object* lp_mathlib_Parser_Attr_pull__end___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_pull__end___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_pull__end___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_pull__end___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_pull__end___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_pull__end___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_pull__end___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_pull__end___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_pull__end___closed__3;
static lean_once_cell_t lp_mathlib_Parser_Attr_pull__end___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_pull__end___closed__4;
static lean_once_cell_t lp_mathlib_Parser_Attr_pull__end___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_pull__end___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_pull__end;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "pull_end_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(93, 104, 22, 236, 3, 205, 82, 117)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "simproc set for pull_end_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value),((lean_object*)(((size_t)(1133190711) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(199, 181, 201, 151, 139, 33, 97, 29)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(44, 155, 8, 60, 60, 249, 247, 72)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(216, 159, 123, 242, 83, 215, 36, 110)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(221, 217, 221, 207, 68, 99, 138, 35)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_pull__end__proc___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_pull__end__proc___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_pull__end__proc___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_pull__end__proc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_pull__end__proc___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(248, 171, 64, 177, 182, 190, 166, 159)}};
static const lean_object* lp_mathlib_Parser_Attr_pull__end__proc___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_pull__end__proc___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_pull__end__proc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_pull__end__proc___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_pull__end__proc___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_pull__end__proc___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_pull__end__proc___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_pull__end__proc___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_pull__end__proc___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_pull__end__proc;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "push_end"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(0, 151, 23, 234, 41, 153, 42, 248)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 231, .m_capacity = 231, .m_length = 227, .m_data = "The simpset `push_end` translates the standard formulations of endomorphisms to the\nalgebraic formulation, so for example `Equiv.refl α` becomes `1 : Equiv α α` and\n`b.trans a` becomes `a * b`.\n\nThe dual simpset is `pull_end`.\n"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__14_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),((lean_object*)(((size_t)(1988518680) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(229, 47, 154, 122, 203, 230, 56, 234)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(182, 153, 16, 210, 8, 240, 116, 189)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(138, 45, 59, 42, 193, 239, 248, 219)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(151, 5, 223, 21, 156, 136, 95, 68)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_push__end___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_push__end___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_push__end___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_push__end___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_push__end___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(5, 4, 202, 129, 118, 230, 24, 8)}};
static const lean_object* lp_mathlib_Parser_Attr_push__end___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_push__end___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_push__end___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_push__end___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_push__end___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_push__end___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_push__end___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_push__end___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_push__end___closed__3;
static lean_once_cell_t lp_mathlib_Parser_Attr_push__end___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_push__end___closed__4;
static lean_once_cell_t lp_mathlib_Parser_Attr_push__end___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_push__end___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_push__end;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "push_end_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(38, 189, 35, 120, 98, 137, 192, 125)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "simproc set for push_end_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value),((lean_object*)(((size_t)(1988518680) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(29, 136, 45, 98, 99, 57, 182, 243)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(158, 84, 78, 183, 83, 194, 152, 113)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(146, 72, 22, 249, 56, 62, 128, 208)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(159, 150, 125, 116, 12, 247, 148, 134)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_push__end__proc___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_push__end__proc___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_push__end__proc___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_push__end__proc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_push__end__proc___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(43, 42, 118, 29, 114, 239, 3, 220)}};
static const lean_object* lp_mathlib_Parser_Attr_push__end__proc___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_push__end__proc___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_push__end__proc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_push__end__proc___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_push__end__proc___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_push__end__proc___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_push__end__proc___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_push__end__proc___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_push__end__proc___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_push__end__proc;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "mfld_simps"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(236, 187, 93, 29, 15, 239, 5, 138)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 607, .m_capacity = 607, .m_length = 606, .m_data = "The simpset `mfld_simps` records several simp lemmas that are\nespecially useful in manifolds. It is a subset of the whole set of simp lemmas, but it makes it\npossible to have quicker proofs (when used with `squeeze_simp` or `simp only`) while retaining\nreadability.\n\nThe typical use case is the following, in a file on manifolds:\nIf `simp [foo, bar]` is slow, replace it with `squeeze_simp [foo, bar, mfld_simps]` and paste\nits output. The list of lemmas should be reasonable (contrary to the output of\n`squeeze_simp [foo, bar]` which might contain tens of lemmas), and the outcome should be quick\nenough.\n"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__14_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),((lean_object*)(((size_t)(479538640) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(212, 36, 113, 192, 178, 40, 198, 237)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(155, 89, 234, 131, 104, 51, 8, 219)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(123, 151, 120, 146, 131, 38, 78, 10)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(82, 179, 91, 238, 232, 129, 191, 4)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_mfld__simps___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_mfld__simps___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_mfld__simps___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_mfld__simps___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_mfld__simps___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(17, 229, 174, 104, 64, 58, 104, 248)}};
static const lean_object* lp_mathlib_Parser_Attr_mfld__simps___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_mfld__simps___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_mfld__simps___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_mfld__simps___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_mfld__simps___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_mfld__simps___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_mfld__simps___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_mfld__simps___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_mfld__simps___closed__3;
static lean_once_cell_t lp_mathlib_Parser_Attr_mfld__simps___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_mfld__simps___closed__4;
static lean_once_cell_t lp_mathlib_Parser_Attr_mfld__simps___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_mfld__simps___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_mfld__simps;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "mfld_simps_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(216, 136, 149, 78, 227, 169, 117, 244)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "simproc set for mfld_simps_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value),((lean_object*)(((size_t)(479538640) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(172, 169, 27, 100, 47, 160, 191, 90)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(99, 207, 49, 29, 109, 221, 84, 84)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(3, 149, 20, 129, 254, 126, 185, 252)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(218, 193, 217, 212, 23, 43, 87, 175)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_mfld__simps__proc___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_mfld__simps__proc___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_mfld__simps__proc___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_mfld__simps__proc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_mfld__simps__proc___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(253, 181, 131, 124, 145, 191, 63, 254)}};
static const lean_object* lp_mathlib_Parser_Attr_mfld__simps__proc___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_mfld__simps__proc___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_mfld__simps__proc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_mfld__simps__proc___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_mfld__simps__proc___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_mfld__simps__proc___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_mfld__simps__proc___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_mfld__simps__proc___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_mfld__simps__proc___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_mfld__simps__proc;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "integral_simps"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(186, 40, 77, 14, 31, 71, 14, 87)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "Simp set for integral rules. "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_integral__simps___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_integral__simps___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_integral__simps___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_integral__simps___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_integral__simps___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(31, 99, 2, 12, 30, 142, 20, 50)}};
static const lean_object* lp_mathlib_Parser_Attr_integral__simps___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_integral__simps___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_integral__simps___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_integral__simps___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_integral__simps___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_integral__simps___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_integral__simps___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_integral__simps___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_integral__simps___closed__3;
static lean_once_cell_t lp_mathlib_Parser_Attr_integral__simps___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_integral__simps___closed__4;
static lean_once_cell_t lp_mathlib_Parser_Attr_integral__simps___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_integral__simps___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_integral__simps;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "integral_simps_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(216, 25, 141, 225, 200, 226, 97, 8)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 36, .m_capacity = 36, .m_length = 35, .m_data = "simproc set for integral_simps_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_integral__simps__proc___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_integral__simps__proc___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_integral__simps__proc___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_integral__simps__proc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_integral__simps__proc___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(253, 196, 62, 199, 71, 36, 30, 185)}};
static const lean_object* lp_mathlib_Parser_Attr_integral__simps__proc___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_integral__simps__proc___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_integral__simps__proc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_integral__simps__proc___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_integral__simps__proc___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_integral__simps__proc___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_integral__simps__proc___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_integral__simps__proc___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_integral__simps__proc___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_integral__simps__proc;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "typevec"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(68, 54, 241, 254, 99, 188, 78, 35)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 64, .m_capacity = 64, .m_length = 63, .m_data = "simp set for the manipulation of typevec and arrow expressions "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__14_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),((lean_object*)(((size_t)(1131139975) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(87, 44, 68, 205, 191, 209, 112, 13)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(92, 191, 131, 108, 98, 229, 108, 116)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(168, 98, 235, 13, 25, 51, 38, 164)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(205, 4, 89, 61, 141, 229, 27, 123)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_typevec___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_typevec___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_typevec___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_typevec___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_typevec___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(105, 49, 242, 182, 34, 101, 97, 38)}};
static const lean_object* lp_mathlib_Parser_Attr_typevec___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_typevec___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_typevec___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_typevec___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_typevec___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_typevec___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_typevec___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_typevec___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_typevec___closed__3;
static lean_once_cell_t lp_mathlib_Parser_Attr_typevec___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_typevec___closed__4;
static lean_once_cell_t lp_mathlib_Parser_Attr_typevec___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_typevec___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_typevec;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "typevec_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(92, 20, 62, 115, 51, 48, 142, 229)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "simproc set for typevec_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value),((lean_object*)(((size_t)(1131139975) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(255, 49, 97, 142, 48, 48, 4, 169)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(148, 86, 129, 208, 90, 12, 117, 215)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(192, 82, 59, 114, 228, 219, 174, 47)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(133, 213, 181, 154, 52, 33, 9, 28)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_typevec__proc___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_typevec__proc___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_typevec__proc___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_typevec__proc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_typevec__proc___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(33, 215, 223, 195, 106, 204, 93, 32)}};
static const lean_object* lp_mathlib_Parser_Attr_typevec__proc___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_typevec__proc___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_typevec__proc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_typevec__proc___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_typevec__proc___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_typevec__proc___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_typevec__proc___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_typevec__proc___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_typevec__proc___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_typevec__proc;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "ghost_simps"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(89, 76, 236, 99, 141, 6, 241, 151)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "Simplification rules for ghost equations. "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__14_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),((lean_object*)(((size_t)(1086225606) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(225, 159, 27, 169, 133, 145, 58, 111)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(98, 99, 116, 150, 78, 100, 135, 184)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(110, 207, 117, 8, 227, 231, 255, 86)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(139, 148, 173, 254, 189, 223, 20, 129)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_ghost__simps___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_ghost__simps___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_ghost__simps___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_ghost__simps___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_ghost__simps___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(84, 59, 74, 123, 173, 101, 94, 53)}};
static const lean_object* lp_mathlib_Parser_Attr_ghost__simps___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_ghost__simps___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_ghost__simps___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_ghost__simps___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_ghost__simps___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_ghost__simps___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_ghost__simps___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_ghost__simps___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_ghost__simps___closed__3;
static lean_once_cell_t lp_mathlib_Parser_Attr_ghost__simps___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_ghost__simps___closed__4;
static lean_once_cell_t lp_mathlib_Parser_Attr_ghost__simps___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_ghost__simps___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_ghost__simps;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "ghost_simps_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(193, 124, 87, 3, 251, 149, 94, 99)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "simproc set for ghost_simps_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value),((lean_object*)(((size_t)(1086225606) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(201, 196, 8, 157, 103, 103, 97, 204)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(90, 246, 74, 64, 111, 252, 244, 41)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(38, 9, 214, 75, 233, 188, 237, 49)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(35, 67, 170, 176, 191, 177, 106, 232)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_ghost__simps__proc___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_ghost__simps__proc___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_ghost__simps__proc___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_ghost__simps__proc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_ghost__simps__proc___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(188, 206, 193, 46, 169, 125, 71, 118)}};
static const lean_object* lp_mathlib_Parser_Attr_ghost__simps__proc___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_ghost__simps__proc___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_ghost__simps__proc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_ghost__simps__proc___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_ghost__simps__proc___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_ghost__simps__proc___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_ghost__simps__proc___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_ghost__simps__proc___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_ghost__simps__proc___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_ghost__simps__proc;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "nontriviality"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(39, 78, 96, 91, 103, 133, 39, 103)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 217, .m_capacity = 217, .m_length = 215, .m_data = "The `@[nontriviality]` simp set is used by the `nontriviality` tactic to automatically\ndischarge theorems about the trivial case (where we know `Subsingleton α` and many theorems\nin e.g. groups are trivially true). "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__14_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),((lean_object*)(((size_t)(2133673922) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(252, 12, 255, 114, 214, 88, 111, 31)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(83, 210, 177, 244, 127, 164, 89, 14)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(19, 178, 200, 245, 236, 78, 92, 28)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(138, 117, 145, 21, 182, 71, 64, 254)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_nontriviality___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_nontriviality___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_nontriviality___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_nontriviality___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_nontriviality___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(2, 247, 142, 133, 140, 53, 99, 24)}};
static const lean_object* lp_mathlib_Parser_Attr_nontriviality___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_nontriviality___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_nontriviality___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_nontriviality___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_nontriviality___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_nontriviality___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_nontriviality___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_nontriviality___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_nontriviality___closed__3;
static lean_once_cell_t lp_mathlib_Parser_Attr_nontriviality___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_nontriviality___closed__4;
static lean_once_cell_t lp_mathlib_Parser_Attr_nontriviality___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_nontriviality___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_nontriviality;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "nontriviality_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(185, 253, 8, 201, 18, 32, 5, 123)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "simproc set for nontriviality_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value),((lean_object*)(((size_t)(2133673922) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(132, 36, 180, 8, 232, 45, 123, 172)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(171, 141, 101, 111, 216, 188, 143, 138)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(107, 167, 30, 183, 130, 113, 112, 100)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(162, 252, 22, 16, 216, 48, 206, 141)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_nontriviality__proc___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_nontriviality__proc___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_nontriviality__proc___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_nontriviality__proc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_nontriviality__proc___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(244, 73, 93, 179, 229, 253, 24, 105)}};
static const lean_object* lp_mathlib_Parser_Attr_nontriviality__proc___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_nontriviality__proc___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_nontriviality__proc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_nontriviality__proc___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_nontriviality__proc___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_nontriviality__proc___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_nontriviality__proc___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_nontriviality__proc___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_nontriviality__proc___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_nontriviality__proc;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2811593030____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "is_poly"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2811593030____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2811593030____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2811593030____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2811593030____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(127, 188, 188, 140, 250, 121, 185, 2)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2811593030____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2811593030____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2811593030____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "A stub attribute for `is_poly`. "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2811593030____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2811593030____hygCtx___hyg_5__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2811593030____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2811593030____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_2811593030____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_is__poly___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_is__poly___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_is__poly___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_is__poly___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_is__poly___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2811593030____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(90, 143, 110, 128, 171, 215, 102, 176)}};
static const lean_object* lp_mathlib_Parser_Attr_is__poly___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_is__poly___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_is__poly___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2811593030____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_is__poly___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_is__poly___closed__1_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_is__poly___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_is__poly___closed__0_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_is__poly___closed__1_value)}};
static const lean_object* lp_mathlib_Parser_Attr_is__poly___closed__2 = (const lean_object*)&lp_mathlib_Parser_Attr_is__poly___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Parser_Attr_is__poly = (const lean_object*)&lp_mathlib_Parser_Attr_is__poly___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "fin_omega"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(180, 240, 18, 190, 174, 37, 227, 30)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 56, .m_capacity = 56, .m_length = 55, .m_data = "A simp set for the `fin_omega` wrapper around `omega`. "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_fin__omega___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_fin__omega___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_fin__omega___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_fin__omega___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_fin__omega___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(121, 225, 58, 136, 192, 67, 230, 48)}};
static const lean_object* lp_mathlib_Parser_Attr_fin__omega___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_fin__omega___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_fin__omega___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_fin__omega___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_fin__omega___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_fin__omega___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_fin__omega___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_fin__omega___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_fin__omega___closed__3;
static lean_once_cell_t lp_mathlib_Parser_Attr_fin__omega___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_fin__omega___closed__4;
static lean_once_cell_t lp_mathlib_Parser_Attr_fin__omega___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_fin__omega___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_fin__omega;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "fin_omega_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(34, 165, 185, 41, 194, 214, 11, 11)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "simproc set for fin_omega_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_fin__omega__proc___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_fin__omega__proc___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_fin__omega__proc___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_fin__omega__proc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_fin__omega__proc___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(135, 181, 235, 144, 252, 22, 217, 139)}};
static const lean_object* lp_mathlib_Parser_Attr_fin__omega__proc___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_fin__omega__proc___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_fin__omega__proc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_fin__omega__proc___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_fin__omega__proc___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_fin__omega__proc___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_fin__omega__proc___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_fin__omega__proc___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_fin__omega__proc___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_fin__omega__proc;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "enat_to_nat_top"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(166, 234, 49, 40, 135, 241, 40, 197)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 74, .m_capacity = 74, .m_length = 71, .m_data = "A simp set for simplifying expressions involving `⊤` in `enat_to_nat`. "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__14_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),((lean_object*)(((size_t)(2016062075) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(156, 230, 139, 64, 10, 171, 124, 46)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(179, 189, 4, 194, 238, 226, 126, 120)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(115, 177, 49, 55, 37, 103, 17, 244)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(170, 156, 115, 111, 230, 158, 30, 240)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_enat__to__nat__top___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_enat__to__nat__top___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_enat__to__nat__top___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_enat__to__nat__top___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_enat__to__nat__top___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(171, 60, 33, 54, 214, 100, 91, 227)}};
static const lean_object* lp_mathlib_Parser_Attr_enat__to__nat__top___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_enat__to__nat__top___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_enat__to__nat__top___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_enat__to__nat__top___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_enat__to__nat__top___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_enat__to__nat__top___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_enat__to__nat__top___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_enat__to__nat__top___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_enat__to__nat__top___closed__3;
static lean_once_cell_t lp_mathlib_Parser_Attr_enat__to__nat__top___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_enat__to__nat__top___closed__4;
static lean_once_cell_t lp_mathlib_Parser_Attr_enat__to__nat__top___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_enat__to__nat__top___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_enat__to__nat__top;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "enat_to_nat_top_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(203, 225, 238, 244, 45, 111, 127, 191)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "simproc set for enat_to_nat_top_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value),((lean_object*)(((size_t)(2016062075) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(228, 242, 9, 98, 2, 17, 157, 114)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(75, 122, 133, 107, 250, 15, 90, 33)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(11, 96, 100, 7, 233, 220, 149, 186)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(130, 21, 245, 140, 137, 149, 90, 158)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_enat__to__nat__top__proc___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_enat__to__nat__top__proc___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_enat__to__nat__top__proc___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_enat__to__nat__top__proc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_enat__to__nat__top__proc___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(6, 176, 227, 66, 200, 202, 43, 117)}};
static const lean_object* lp_mathlib_Parser_Attr_enat__to__nat__top__proc___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_enat__to__nat__top__proc___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_enat__to__nat__top__proc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_enat__to__nat__top__proc___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_enat__to__nat__top__proc___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_enat__to__nat__top__proc___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_enat__to__nat__top__proc___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_enat__to__nat__top__proc___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_enat__to__nat__top__proc___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_enat__to__nat__top__proc;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "enat_to_nat_coe"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(186, 150, 144, 238, 199, 17, 9, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 75, .m_capacity = 75, .m_length = 68, .m_data = "A simp set for pushing coercions from `ℕ` to `ℕ∞` in `enat_to_nat`. "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__14_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),((lean_object*)(((size_t)(668879677) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(152, 215, 188, 153, 222, 85, 187, 164)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(95, 156, 15, 129, 229, 63, 51, 44)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(103, 234, 192, 110, 176, 254, 60, 8)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(78, 213, 24, 215, 171, 124, 155, 170)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(31, 117, 53, 34, 99, 241, 69, 41)}};
static const lean_object* lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__3;
static lean_once_cell_t lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__4;
static lean_once_cell_t lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_enat__to__nat__coe;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "enat_to_nat_coe_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(116, 249, 182, 71, 121, 241, 129, 35)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "simproc set for enat_to_nat_coe_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value),((lean_object*)(((size_t)(668879677) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(64, 109, 244, 205, 183, 107, 71, 35)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(23, 196, 132, 193, 176, 130, 218, 148)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(255, 74, 110, 144, 91, 9, 22, 252)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(38, 222, 239, 236, 31, 225, 66, 156)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_enat__to__nat__coe__proc___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_enat__to__nat__coe__proc___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_enat__to__nat__coe__proc___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_enat__to__nat__coe__proc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_enat__to__nat__coe__proc___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(57, 238, 181, 31, 108, 188, 69, 208)}};
static const lean_object* lp_mathlib_Parser_Attr_enat__to__nat__coe__proc___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_enat__to__nat__coe__proc___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_enat__to__nat__coe__proc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_enat__to__nat__coe__proc___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_enat__to__nat__coe__proc___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_enat__to__nat__coe__proc___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_enat__to__nat__coe__proc___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_enat__to__nat__coe__proc___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_enat__to__nat__coe__proc___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_enat__to__nat__coe__proc;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "pnat_to_nat_coe"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(33, 143, 136, 94, 87, 187, 105, 7)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "A simp set for the `pnat_to_nat` tactic. "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(92, 28, 218, 251, 234, 255, 40, 218)}};
static const lean_object* lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__3;
static lean_once_cell_t lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__4;
static lean_once_cell_t lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_pnat__to__nat__coe;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "pnat_to_nat_coe_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(106, 111, 75, 129, 149, 242, 209, 42)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "simproc set for pnat_to_nat_coe_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(111, 220, 183, 9, 118, 113, 180, 240)}};
static const lean_object* lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "mon_tauto"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(94, 146, 56, 118, 247, 4, 212, 219)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2127, .m_capacity = 2127, .m_length = 2020, .m_data = "`mon_tauto` is a simp set to prove tautologies about morphisms from some (tensor) power of `M`\nto `M`, where `M` is a (commutative) monoid object in a (braided) monoidal category.\n\n**This `simp` set is incompatible with the standard simp set.**\nIf you want to use it, make sure to add the following to your simp call to disable the problematic\ndefault simp lemmas:\n```\n-MonoidalCategory.whiskerLeft_id, -MonoidalCategory.id_whiskerRight,\n-MonoidalCategory.tensor_comp, -MonoidalCategory.tensor_comp_assoc,\n-MonObj.mul_assoc, -MonObj.mul_assoc_assoc\n```\n\nThe general algorithm it follows is to push the associators `α_` and commutators `β_` inwards until\nthey cancel against the right sequence of multiplications.\n\nThis approach is justified by the fact that a tautology in the language of (commutative) monoid\nobjects \"remembers\" how it was proved: Every use of a (commutative) monoid object axiom inserts a\nunitor, associator or commutator, and proving a tautology simply amounts to undoing those moves as\nprescribed by the presence of unitors, associators and commutators in its expression.\n\nThis simp set is opinionated about its normal form, which is why it cannot be used concurrently with\nsome of the simp lemmas in the standard simp set:\n* It eliminates all mentions of whiskers by rewriting them to tensored homs,\n  which goes against `whiskerLeft_id` and `id_whiskerRight`:\n  `X ◁ f = 𝟙 X ⊗ₘ f`, `f ▷ X = 𝟙 X ⊗ₘ f`.\n  This goes against `whiskerLeft_id` and `id_whiskerRight` in the standard simp set.\n* It collapses compositions of tensored homs to the tensored hom of the compositions,\n  which goes against `tensor_comp`:\n  `(f₁ ⊗ₘ g₁) ≫ (f₂ ⊗ₘ g₂) = (f₁ ≫ f₂) ⊗ₘ (g₁ ≫ g₂)`. TODO: Isn't this direction Just Better\?\n* It cancels the associators against multiplications,\n  which goes against `mul_assoc`:\n  `(α_ M M M).hom ≫ (𝟙 M ⊗ₘ μ) ≫ μ = (μ ⊗ₘ 𝟙 M) ≫ μ`,\n  `(α_ M M M).inv ≫ (μ ⊗ₘ 𝟙 M) ≫ μ = (𝟙 M ⊗ₘ μ) ≫ μ`\n* It unfolds non-primitive coherence isomorphisms, like the tensor strengths `tensorμ`, `tensorδ`.\n"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_mon__tauto___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_mon__tauto___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_mon__tauto___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_mon__tauto___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_mon__tauto___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(35, 70, 42, 90, 254, 95, 186, 92)}};
static const lean_object* lp_mathlib_Parser_Attr_mon__tauto___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_mon__tauto___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_mon__tauto___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_mon__tauto___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_mon__tauto___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_mon__tauto___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_mon__tauto___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_mon__tauto___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_mon__tauto___closed__3;
static lean_once_cell_t lp_mathlib_Parser_Attr_mon__tauto___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_mon__tauto___closed__4;
static lean_once_cell_t lp_mathlib_Parser_Attr_mon__tauto___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_mon__tauto___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_mon__tauto;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "mon_tauto_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(154, 119, 177, 45, 92, 107, 3, 178)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "simproc set for mon_tauto_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_mon__tauto__proc___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_mon__tauto__proc___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_mon__tauto__proc___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_mon__tauto__proc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_mon__tauto__proc___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(191, 137, 146, 253, 145, 12, 134, 75)}};
static const lean_object* lp_mathlib_Parser_Attr_mon__tauto__proc___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_mon__tauto__proc___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_mon__tauto__proc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_mon__tauto__proc___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_mon__tauto__proc___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_mon__tauto__proc___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_mon__tauto__proc___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_mon__tauto__proc___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_mon__tauto__proc___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_mon__tauto__proc;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "coassoc_simps"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(145, 1, 188, 172, 112, 28, 178, 39)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1318, .m_capacity = 1318, .m_length = 1231, .m_data = "`coassoc_simps` is a simp set useful to prove tautologies on coalgebras.\n\nThe general algorithm it follows is to push the associators `TensorProduct.assoc` and\ncommutators `TensorProduct.comm` inwards (to the right) until they cancel against\nco-multiplications.\n\nThe simp set makes the following choice of normal form\n* It regards `TensorProduct.map`, `TensorProduct.assoc`, `TensorProduct.comm` as the primitive\n  constructions and rewrites everything else such as `lTensor`, `leftComm` using them.\n* It rewrites both sides into a right associated composition of linear maps.\n  In particular `LinearMap.comp_assoc` and `LinearEquiv.coe_trans` are tagged.\n* It rewrites `(f₂ ⊗ g₂) ∘ (f₁ ⊗ g₁)` into `(f₂ ∘ f₁) ⊗ (g₂ ∘ g₁)`.\n\n## Notes\n\n- It is not confluent with `(ε ⊗ₘ id) ∘ₗ δ = λ⁻¹`.\n  It is often useful to `trans` (or `calc`) with a term containing\n  `(ε ⊗ₘ _) ∘ₗ δ` or `(_ ⊗ₘ ε) ∘ₗ δ`,\n  and use one of `map_counit_comp_comul_left` `map_counit_comp_comul_right`\n  `map_counit_comp_comul_left_assoc` `map_counit_comp_comul_right_assoc` to continue.\n\n- Some lemmas (e.g. `lid_comp_map : λ ∘ₗ (f ⊗ₘ g) = g ∘ₗ λ ∘ₗ (f ⊗ₘ id)`) loops when tagged as simp,\n  so we wrap it inside a rudimentary simproc that only fires when `g ≠ id`.\n"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__14_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),((lean_object*)(((size_t)(85691930) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(135, 229, 245, 114, 173, 168, 121, 95)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(236, 165, 147, 26, 66, 95, 63, 209)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(24, 215, 210, 167, 128, 83, 205, 44)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(157, 33, 126, 174, 148, 122, 244, 199)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_coassoc__simps___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_coassoc__simps___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_coassoc__simps___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_coassoc__simps___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_coassoc__simps___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(108, 63, 106, 43, 150, 221, 226, 118)}};
static const lean_object* lp_mathlib_Parser_Attr_coassoc__simps___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_coassoc__simps___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_coassoc__simps___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_coassoc__simps___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_coassoc__simps___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_coassoc__simps___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_coassoc__simps___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_coassoc__simps___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_coassoc__simps___closed__3;
static lean_once_cell_t lp_mathlib_Parser_Attr_coassoc__simps___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_coassoc__simps___closed__4;
static lean_once_cell_t lp_mathlib_Parser_Attr_coassoc__simps___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_coassoc__simps___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_coassoc__simps;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "coassoc_simps_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(12, 19, 24, 19, 6, 187, 158, 162)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "simproc set for coassoc_simps_proc"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__value),((lean_object*)(((size_t)(85691930) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(207, 109, 208, 202, 227, 208, 154, 85)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(132, 150, 198, 195, 173, 186, 250, 82)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(16, 44, 125, 121, 225, 183, 222, 22)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28__value),((lean_object*)(((size_t)(3) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(245, 129, 90, 64, 48, 111, 66, 250)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_3_;
static const lean_ctor_object lp_mathlib_Parser_Attr_coassoc__simps__proc___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Parser_Attr_functor__norm___closed__0_value),LEAN_SCALAR_PTR_LITERAL(46, 201, 23, 171, 41, 77, 220, 95)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_coassoc__simps__proc___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_coassoc__simps__proc___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(10, 9, 185, 250, 127, 107, 245, 225)}};
static const lean_ctor_object lp_mathlib_Parser_Attr_coassoc__simps__proc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Parser_Attr_coassoc__simps__proc___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(113, 184, 29, 166, 251, 1, 161, 248)}};
static const lean_object* lp_mathlib_Parser_Attr_coassoc__simps__proc___closed__0 = (const lean_object*)&lp_mathlib_Parser_Attr_coassoc__simps__proc___closed__0_value;
static const lean_ctor_object lp_mathlib_Parser_Attr_coassoc__simps__proc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__0_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Parser_Attr_coassoc__simps__proc___closed__1 = (const lean_object*)&lp_mathlib_Parser_Attr_coassoc__simps__proc___closed__1_value;
static lean_once_cell_t lp_mathlib_Parser_Attr_coassoc__simps__proc___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_coassoc__simps__proc___closed__2;
static lean_once_cell_t lp_mathlib_Parser_Attr_coassoc__simps__proc___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Parser_Attr_coassoc__simps__proc___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Parser_Attr_coassoc__simps__proc;
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__15_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; 
v___x_29_ = lean_unsigned_to_nat(2336186397u);
v___x_30_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__14_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_31_ = l_Lean_Name_num___override(v___x_30_, v___x_29_);
return v___x_31_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__17_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; 
v___x_33_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_34_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__15_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__15_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__15_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_);
v___x_35_ = l_Lean_Name_str___override(v___x_34_, v___x_33_);
return v___x_35_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__19_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; 
v___x_37_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_38_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__17_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__17_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__17_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_);
v___x_39_ = l_Lean_Name_str___override(v___x_38_, v___x_37_);
return v___x_39_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__20_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; 
v___x_40_ = lean_unsigned_to_nat(3u);
v___x_41_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__19_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__19_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__19_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_);
v___x_42_ = l_Lean_Name_num___override(v___x_41_, v___x_40_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; 
v___x_44_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_45_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_46_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__20_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__20_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__20_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_);
v___x_47_ = l_Lean_Meta_registerSimpAttr(v___x_44_, v___x_45_, v___x_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5____boxed(lean_object* v_a_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_();
return v_res_49_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_functor__norm___closed__9(void){
_start:
{
lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; 
v___x_67_ = l_Lean_Parser_Tactic_simpPost;
v___x_68_ = l_Lean_Parser_Tactic_simpPre;
v___x_69_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__8));
v___x_70_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_70_, 0, v___x_69_);
lean_ctor_set(v___x_70_, 1, v___x_68_);
lean_ctor_set(v___x_70_, 2, v___x_67_);
return v___x_70_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_functor__norm___closed__10(void){
_start:
{
lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; 
v___x_71_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__9, &lp_mathlib_Parser_Attr_functor__norm___closed__9_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__9);
v___x_72_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__6));
v___x_73_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_73_, 0, v___x_72_);
lean_ctor_set(v___x_73_, 1, v___x_71_);
return v___x_73_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_functor__norm___closed__11(void){
_start:
{
lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; 
v___x_74_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_75_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__4));
v___x_76_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_77_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_77_, 0, v___x_76_);
lean_ctor_set(v___x_77_, 1, v___x_75_);
lean_ctor_set(v___x_77_, 2, v___x_74_);
return v___x_77_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_functor__norm___closed__16(void){
_start:
{
lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; 
v___x_87_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__15));
v___x_88_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__11, &lp_mathlib_Parser_Attr_functor__norm___closed__11_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__11);
v___x_89_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_90_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_90_, 0, v___x_89_);
lean_ctor_set(v___x_90_, 1, v___x_88_);
lean_ctor_set(v___x_90_, 2, v___x_87_);
return v___x_90_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_functor__norm___closed__21(void){
_start:
{
lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; 
v___x_100_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__20));
v___x_101_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__16, &lp_mathlib_Parser_Attr_functor__norm___closed__16_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__16);
v___x_102_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_103_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_103_, 0, v___x_102_);
lean_ctor_set(v___x_103_, 1, v___x_101_);
lean_ctor_set(v___x_103_, 2, v___x_100_);
return v___x_103_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_functor__norm___closed__22(void){
_start:
{
lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; 
v___x_104_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__21, &lp_mathlib_Parser_Attr_functor__norm___closed__21_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__21);
v___x_105_ = lean_unsigned_to_nat(1022u);
v___x_106_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__1));
v___x_107_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_107_, 0, v___x_106_);
lean_ctor_set(v___x_107_, 1, v___x_105_);
lean_ctor_set(v___x_107_, 2, v___x_104_);
return v___x_107_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_functor__norm(void){
_start:
{
lean_object* v___x_108_; 
v___x_108_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__22, &lp_mathlib_Parser_Attr_functor__norm___closed__22_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__22);
return v___x_108_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__10_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; 
v___x_132_ = lean_unsigned_to_nat(2336186397u);
v___x_133_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_));
v___x_134_ = l_Lean_Name_num___override(v___x_133_, v___x_132_);
return v___x_134_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; 
v___x_135_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_136_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__10_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__10_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__10_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_);
v___x_137_ = l_Lean_Name_str___override(v___x_136_, v___x_135_);
return v___x_137_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__12_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; 
v___x_138_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_139_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__11_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_);
v___x_140_ = l_Lean_Name_str___override(v___x_139_, v___x_138_);
return v___x_140_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__13_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; 
v___x_141_ = lean_unsigned_to_nat(3u);
v___x_142_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__12_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__12_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__12_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_);
v___x_143_ = l_Lean_Name_num___override(v___x_142_, v___x_141_);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_(){
_start:
{
lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; 
v___x_145_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_));
v___x_146_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_));
v___x_147_ = lean_box(0);
v___x_148_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__13_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__13_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__13_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_);
v___x_149_ = l_Lean_Meta_Simp_registerSimprocAttr(v___x_145_, v___x_146_, v___x_147_, v___x_148_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28____boxed(lean_object* v_a_150_){
_start:
{
lean_object* v_res_151_; 
v_res_151_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_();
return v_res_151_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_functor__norm__proc___closed__2(void){
_start:
{
lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; 
v___x_159_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_160_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm__proc___closed__1));
v___x_161_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_162_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_162_, 0, v___x_161_);
lean_ctor_set(v___x_162_, 1, v___x_160_);
lean_ctor_set(v___x_162_, 2, v___x_159_);
return v___x_162_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_functor__norm__proc___closed__3(void){
_start:
{
lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; 
v___x_163_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm__proc___closed__2, &lp_mathlib_Parser_Attr_functor__norm__proc___closed__2_once, _init_lp_mathlib_Parser_Attr_functor__norm__proc___closed__2);
v___x_164_ = lean_unsigned_to_nat(1022u);
v___x_165_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm__proc___closed__0));
v___x_166_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_166_, 0, v___x_165_);
lean_ctor_set(v___x_166_, 1, v___x_164_);
lean_ctor_set(v___x_166_, 2, v___x_163_);
return v___x_166_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_functor__norm__proc(void){
_start:
{
lean_object* v___x_167_; 
v___x_167_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm__proc___closed__3, &lp_mathlib_Parser_Attr_functor__norm__proc___closed__3_once, _init_lp_mathlib_Parser_Attr_functor__norm__proc___closed__3);
return v___x_167_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; 
v___x_171_ = lean_unsigned_to_nat(3215534580u);
v___x_172_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__14_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_173_ = l_Lean_Name_num___override(v___x_172_, v___x_171_);
return v___x_173_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; 
v___x_174_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_175_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5_);
v___x_176_ = l_Lean_Name_str___override(v___x_175_, v___x_174_);
return v___x_176_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; 
v___x_177_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_178_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5_);
v___x_179_ = l_Lean_Name_str___override(v___x_178_, v___x_177_);
return v___x_179_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; 
v___x_180_ = lean_unsigned_to_nat(3u);
v___x_181_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5_);
v___x_182_ = l_Lean_Name_num___override(v___x_181_, v___x_180_);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; 
v___x_184_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5_));
v___x_185_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_186_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5_);
v___x_187_ = l_Lean_Meta_registerSimpAttr(v___x_184_, v___x_185_, v___x_186_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5____boxed(lean_object* v_a_188_){
_start:
{
lean_object* v_res_189_; 
v_res_189_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5_();
return v_res_189_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_monad__norm___closed__2(void){
_start:
{
lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; 
v___x_197_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_198_ = ((lean_object*)(lp_mathlib_Parser_Attr_monad__norm___closed__1));
v___x_199_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_200_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_200_, 0, v___x_199_);
lean_ctor_set(v___x_200_, 1, v___x_198_);
lean_ctor_set(v___x_200_, 2, v___x_197_);
return v___x_200_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_monad__norm___closed__3(void){
_start:
{
lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; 
v___x_201_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__15));
v___x_202_ = lean_obj_once(&lp_mathlib_Parser_Attr_monad__norm___closed__2, &lp_mathlib_Parser_Attr_monad__norm___closed__2_once, _init_lp_mathlib_Parser_Attr_monad__norm___closed__2);
v___x_203_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_204_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_204_, 0, v___x_203_);
lean_ctor_set(v___x_204_, 1, v___x_202_);
lean_ctor_set(v___x_204_, 2, v___x_201_);
return v___x_204_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_monad__norm___closed__4(void){
_start:
{
lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; 
v___x_205_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__20));
v___x_206_ = lean_obj_once(&lp_mathlib_Parser_Attr_monad__norm___closed__3, &lp_mathlib_Parser_Attr_monad__norm___closed__3_once, _init_lp_mathlib_Parser_Attr_monad__norm___closed__3);
v___x_207_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_208_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_208_, 0, v___x_207_);
lean_ctor_set(v___x_208_, 1, v___x_206_);
lean_ctor_set(v___x_208_, 2, v___x_205_);
return v___x_208_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_monad__norm___closed__5(void){
_start:
{
lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; 
v___x_209_ = lean_obj_once(&lp_mathlib_Parser_Attr_monad__norm___closed__4, &lp_mathlib_Parser_Attr_monad__norm___closed__4_once, _init_lp_mathlib_Parser_Attr_monad__norm___closed__4);
v___x_210_ = lean_unsigned_to_nat(1022u);
v___x_211_ = ((lean_object*)(lp_mathlib_Parser_Attr_monad__norm___closed__0));
v___x_212_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_212_, 0, v___x_211_);
lean_ctor_set(v___x_212_, 1, v___x_210_);
lean_ctor_set(v___x_212_, 2, v___x_209_);
return v___x_212_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_monad__norm(void){
_start:
{
lean_object* v___x_213_; 
v___x_213_ = lean_obj_once(&lp_mathlib_Parser_Attr_monad__norm___closed__5, &lp_mathlib_Parser_Attr_monad__norm___closed__5_once, _init_lp_mathlib_Parser_Attr_monad__norm___closed__5);
return v___x_213_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; 
v___x_218_ = lean_unsigned_to_nat(3215534580u);
v___x_219_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_));
v___x_220_ = l_Lean_Name_num___override(v___x_219_, v___x_218_);
return v___x_220_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; 
v___x_221_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_222_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28_);
v___x_223_ = l_Lean_Name_str___override(v___x_222_, v___x_221_);
return v___x_223_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; 
v___x_224_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_225_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28_);
v___x_226_ = l_Lean_Name_str___override(v___x_225_, v___x_224_);
return v___x_226_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; 
v___x_227_ = lean_unsigned_to_nat(3u);
v___x_228_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28_);
v___x_229_ = l_Lean_Name_num___override(v___x_228_, v___x_227_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28_(){
_start:
{
lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; 
v___x_231_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28_));
v___x_232_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28_));
v___x_233_ = lean_box(0);
v___x_234_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28_);
v___x_235_ = l_Lean_Meta_Simp_registerSimprocAttr(v___x_231_, v___x_232_, v___x_233_, v___x_234_);
return v___x_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28____boxed(lean_object* v_a_236_){
_start:
{
lean_object* v_res_237_; 
v_res_237_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28_();
return v_res_237_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_monad__norm__proc___closed__2(void){
_start:
{
lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; 
v___x_245_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_246_ = ((lean_object*)(lp_mathlib_Parser_Attr_monad__norm__proc___closed__1));
v___x_247_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_248_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_248_, 0, v___x_247_);
lean_ctor_set(v___x_248_, 1, v___x_246_);
lean_ctor_set(v___x_248_, 2, v___x_245_);
return v___x_248_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_monad__norm__proc___closed__3(void){
_start:
{
lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; 
v___x_249_ = lean_obj_once(&lp_mathlib_Parser_Attr_monad__norm__proc___closed__2, &lp_mathlib_Parser_Attr_monad__norm__proc___closed__2_once, _init_lp_mathlib_Parser_Attr_monad__norm__proc___closed__2);
v___x_250_ = lean_unsigned_to_nat(1022u);
v___x_251_ = ((lean_object*)(lp_mathlib_Parser_Attr_monad__norm__proc___closed__0));
v___x_252_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_252_, 0, v___x_251_);
lean_ctor_set(v___x_252_, 1, v___x_250_);
lean_ctor_set(v___x_252_, 2, v___x_249_);
return v___x_252_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_monad__norm__proc(void){
_start:
{
lean_object* v___x_253_; 
v___x_253_ = lean_obj_once(&lp_mathlib_Parser_Attr_monad__norm__proc___closed__3, &lp_mathlib_Parser_Attr_monad__norm__proc___closed__3_once, _init_lp_mathlib_Parser_Attr_monad__norm__proc___closed__3);
return v___x_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; 
v___x_271_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5_));
v___x_272_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5_));
v___x_273_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5_));
v___x_274_ = l_Lean_Meta_registerSimpAttr(v___x_271_, v___x_272_, v___x_273_);
return v___x_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5____boxed(lean_object* v_a_275_){
_start:
{
lean_object* v_res_276_; 
v_res_276_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5_();
return v_res_276_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_parity__simps___closed__2(void){
_start:
{
lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; 
v___x_284_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_285_ = ((lean_object*)(lp_mathlib_Parser_Attr_parity__simps___closed__1));
v___x_286_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_287_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_287_, 0, v___x_286_);
lean_ctor_set(v___x_287_, 1, v___x_285_);
lean_ctor_set(v___x_287_, 2, v___x_284_);
return v___x_287_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_parity__simps___closed__3(void){
_start:
{
lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; 
v___x_288_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__15));
v___x_289_ = lean_obj_once(&lp_mathlib_Parser_Attr_parity__simps___closed__2, &lp_mathlib_Parser_Attr_parity__simps___closed__2_once, _init_lp_mathlib_Parser_Attr_parity__simps___closed__2);
v___x_290_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_291_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_291_, 0, v___x_290_);
lean_ctor_set(v___x_291_, 1, v___x_289_);
lean_ctor_set(v___x_291_, 2, v___x_288_);
return v___x_291_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_parity__simps___closed__4(void){
_start:
{
lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; 
v___x_292_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__20));
v___x_293_ = lean_obj_once(&lp_mathlib_Parser_Attr_parity__simps___closed__3, &lp_mathlib_Parser_Attr_parity__simps___closed__3_once, _init_lp_mathlib_Parser_Attr_parity__simps___closed__3);
v___x_294_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_295_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_295_, 0, v___x_294_);
lean_ctor_set(v___x_295_, 1, v___x_293_);
lean_ctor_set(v___x_295_, 2, v___x_292_);
return v___x_295_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_parity__simps___closed__5(void){
_start:
{
lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; 
v___x_296_ = lean_obj_once(&lp_mathlib_Parser_Attr_parity__simps___closed__4, &lp_mathlib_Parser_Attr_parity__simps___closed__4_once, _init_lp_mathlib_Parser_Attr_parity__simps___closed__4);
v___x_297_ = lean_unsigned_to_nat(1022u);
v___x_298_ = ((lean_object*)(lp_mathlib_Parser_Attr_parity__simps___closed__0));
v___x_299_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_299_, 0, v___x_298_);
lean_ctor_set(v___x_299_, 1, v___x_297_);
lean_ctor_set(v___x_299_, 2, v___x_296_);
return v___x_299_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_parity__simps(void){
_start:
{
lean_object* v___x_300_; 
v___x_300_ = lean_obj_once(&lp_mathlib_Parser_Attr_parity__simps___closed__5, &lp_mathlib_Parser_Attr_parity__simps___closed__5_once, _init_lp_mathlib_Parser_Attr_parity__simps___closed__5);
return v___x_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28_(){
_start:
{
lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; 
v___x_318_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28_));
v___x_319_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28_));
v___x_320_ = lean_box(0);
v___x_321_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28_));
v___x_322_ = l_Lean_Meta_Simp_registerSimprocAttr(v___x_318_, v___x_319_, v___x_320_, v___x_321_);
return v___x_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28____boxed(lean_object* v_a_323_){
_start:
{
lean_object* v_res_324_; 
v_res_324_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28_();
return v_res_324_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_parity__simps__proc___closed__2(void){
_start:
{
lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; 
v___x_332_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_333_ = ((lean_object*)(lp_mathlib_Parser_Attr_parity__simps__proc___closed__1));
v___x_334_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_335_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_335_, 0, v___x_334_);
lean_ctor_set(v___x_335_, 1, v___x_333_);
lean_ctor_set(v___x_335_, 2, v___x_332_);
return v___x_335_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_parity__simps__proc___closed__3(void){
_start:
{
lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; 
v___x_336_ = lean_obj_once(&lp_mathlib_Parser_Attr_parity__simps__proc___closed__2, &lp_mathlib_Parser_Attr_parity__simps__proc___closed__2_once, _init_lp_mathlib_Parser_Attr_parity__simps__proc___closed__2);
v___x_337_ = lean_unsigned_to_nat(1022u);
v___x_338_ = ((lean_object*)(lp_mathlib_Parser_Attr_parity__simps__proc___closed__0));
v___x_339_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_339_, 0, v___x_338_);
lean_ctor_set(v___x_339_, 1, v___x_337_);
lean_ctor_set(v___x_339_, 2, v___x_336_);
return v___x_339_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_parity__simps__proc(void){
_start:
{
lean_object* v___x_340_; 
v___x_340_ = lean_obj_once(&lp_mathlib_Parser_Attr_parity__simps__proc___closed__3, &lp_mathlib_Parser_Attr_parity__simps__proc___closed__3_once, _init_lp_mathlib_Parser_Attr_parity__simps__proc___closed__3);
return v___x_340_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; 
v___x_345_ = lean_unsigned_to_nat(2899626838u);
v___x_346_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__14_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_347_ = l_Lean_Name_num___override(v___x_346_, v___x_345_);
return v___x_347_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; 
v___x_348_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_349_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5_);
v___x_350_ = l_Lean_Name_str___override(v___x_349_, v___x_348_);
return v___x_350_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; 
v___x_351_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_352_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5_);
v___x_353_ = l_Lean_Name_str___override(v___x_352_, v___x_351_);
return v___x_353_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; 
v___x_354_ = lean_unsigned_to_nat(3u);
v___x_355_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5_);
v___x_356_ = l_Lean_Name_num___override(v___x_355_, v___x_354_);
return v___x_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; 
v___x_358_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5_));
v___x_359_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5_));
v___x_360_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5_);
v___x_361_ = l_Lean_Meta_registerSimpAttr(v___x_358_, v___x_359_, v___x_360_);
return v___x_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5____boxed(lean_object* v_a_362_){
_start:
{
lean_object* v_res_363_; 
v_res_363_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5_();
return v_res_363_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_rclike__simps___closed__2(void){
_start:
{
lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; 
v___x_371_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_372_ = ((lean_object*)(lp_mathlib_Parser_Attr_rclike__simps___closed__1));
v___x_373_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_374_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_374_, 0, v___x_373_);
lean_ctor_set(v___x_374_, 1, v___x_372_);
lean_ctor_set(v___x_374_, 2, v___x_371_);
return v___x_374_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_rclike__simps___closed__3(void){
_start:
{
lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; 
v___x_375_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__15));
v___x_376_ = lean_obj_once(&lp_mathlib_Parser_Attr_rclike__simps___closed__2, &lp_mathlib_Parser_Attr_rclike__simps___closed__2_once, _init_lp_mathlib_Parser_Attr_rclike__simps___closed__2);
v___x_377_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_378_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_378_, 0, v___x_377_);
lean_ctor_set(v___x_378_, 1, v___x_376_);
lean_ctor_set(v___x_378_, 2, v___x_375_);
return v___x_378_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_rclike__simps___closed__4(void){
_start:
{
lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; 
v___x_379_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__20));
v___x_380_ = lean_obj_once(&lp_mathlib_Parser_Attr_rclike__simps___closed__3, &lp_mathlib_Parser_Attr_rclike__simps___closed__3_once, _init_lp_mathlib_Parser_Attr_rclike__simps___closed__3);
v___x_381_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_382_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_382_, 0, v___x_381_);
lean_ctor_set(v___x_382_, 1, v___x_380_);
lean_ctor_set(v___x_382_, 2, v___x_379_);
return v___x_382_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_rclike__simps___closed__5(void){
_start:
{
lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; 
v___x_383_ = lean_obj_once(&lp_mathlib_Parser_Attr_rclike__simps___closed__4, &lp_mathlib_Parser_Attr_rclike__simps___closed__4_once, _init_lp_mathlib_Parser_Attr_rclike__simps___closed__4);
v___x_384_ = lean_unsigned_to_nat(1022u);
v___x_385_ = ((lean_object*)(lp_mathlib_Parser_Attr_rclike__simps___closed__0));
v___x_386_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_386_, 0, v___x_385_);
lean_ctor_set(v___x_386_, 1, v___x_384_);
lean_ctor_set(v___x_386_, 2, v___x_383_);
return v___x_386_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_rclike__simps(void){
_start:
{
lean_object* v___x_387_; 
v___x_387_ = lean_obj_once(&lp_mathlib_Parser_Attr_rclike__simps___closed__5, &lp_mathlib_Parser_Attr_rclike__simps___closed__5_once, _init_lp_mathlib_Parser_Attr_rclike__simps___closed__5);
return v___x_387_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; 
v___x_392_ = lean_unsigned_to_nat(2899626838u);
v___x_393_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_));
v___x_394_ = l_Lean_Name_num___override(v___x_393_, v___x_392_);
return v___x_394_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; 
v___x_395_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_396_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28_);
v___x_397_ = l_Lean_Name_str___override(v___x_396_, v___x_395_);
return v___x_397_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; 
v___x_398_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_399_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28_);
v___x_400_ = l_Lean_Name_str___override(v___x_399_, v___x_398_);
return v___x_400_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; 
v___x_401_ = lean_unsigned_to_nat(3u);
v___x_402_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28_);
v___x_403_ = l_Lean_Name_num___override(v___x_402_, v___x_401_);
return v___x_403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28_(){
_start:
{
lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; 
v___x_405_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28_));
v___x_406_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28_));
v___x_407_ = lean_box(0);
v___x_408_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28_);
v___x_409_ = l_Lean_Meta_Simp_registerSimprocAttr(v___x_405_, v___x_406_, v___x_407_, v___x_408_);
return v___x_409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28____boxed(lean_object* v_a_410_){
_start:
{
lean_object* v_res_411_; 
v_res_411_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28_();
return v_res_411_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_rclike__simps__proc___closed__2(void){
_start:
{
lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; 
v___x_419_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_420_ = ((lean_object*)(lp_mathlib_Parser_Attr_rclike__simps__proc___closed__1));
v___x_421_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_422_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_422_, 0, v___x_421_);
lean_ctor_set(v___x_422_, 1, v___x_420_);
lean_ctor_set(v___x_422_, 2, v___x_419_);
return v___x_422_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_rclike__simps__proc___closed__3(void){
_start:
{
lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; 
v___x_423_ = lean_obj_once(&lp_mathlib_Parser_Attr_rclike__simps__proc___closed__2, &lp_mathlib_Parser_Attr_rclike__simps__proc___closed__2_once, _init_lp_mathlib_Parser_Attr_rclike__simps__proc___closed__2);
v___x_424_ = lean_unsigned_to_nat(1022u);
v___x_425_ = ((lean_object*)(lp_mathlib_Parser_Attr_rclike__simps__proc___closed__0));
v___x_426_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_426_, 0, v___x_425_);
lean_ctor_set(v___x_426_, 1, v___x_424_);
lean_ctor_set(v___x_426_, 2, v___x_423_);
return v___x_426_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_rclike__simps__proc(void){
_start:
{
lean_object* v___x_427_; 
v___x_427_ = lean_obj_once(&lp_mathlib_Parser_Attr_rclike__simps__proc___closed__3, &lp_mathlib_Parser_Attr_rclike__simps__proc___closed__3_once, _init_lp_mathlib_Parser_Attr_rclike__simps__proc___closed__3);
return v___x_427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; 
v___x_445_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5_));
v___x_446_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5_));
v___x_447_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5_));
v___x_448_ = l_Lean_Meta_registerSimpAttr(v___x_445_, v___x_446_, v___x_447_);
return v___x_448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5____boxed(lean_object* v_a_449_){
_start:
{
lean_object* v_res_450_; 
v_res_450_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5_();
return v_res_450_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_rify__simps___closed__2(void){
_start:
{
lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; 
v___x_458_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_459_ = ((lean_object*)(lp_mathlib_Parser_Attr_rify__simps___closed__1));
v___x_460_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_461_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_461_, 0, v___x_460_);
lean_ctor_set(v___x_461_, 1, v___x_459_);
lean_ctor_set(v___x_461_, 2, v___x_458_);
return v___x_461_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_rify__simps___closed__3(void){
_start:
{
lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; 
v___x_462_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__15));
v___x_463_ = lean_obj_once(&lp_mathlib_Parser_Attr_rify__simps___closed__2, &lp_mathlib_Parser_Attr_rify__simps___closed__2_once, _init_lp_mathlib_Parser_Attr_rify__simps___closed__2);
v___x_464_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_465_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_465_, 0, v___x_464_);
lean_ctor_set(v___x_465_, 1, v___x_463_);
lean_ctor_set(v___x_465_, 2, v___x_462_);
return v___x_465_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_rify__simps___closed__4(void){
_start:
{
lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; 
v___x_466_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__20));
v___x_467_ = lean_obj_once(&lp_mathlib_Parser_Attr_rify__simps___closed__3, &lp_mathlib_Parser_Attr_rify__simps___closed__3_once, _init_lp_mathlib_Parser_Attr_rify__simps___closed__3);
v___x_468_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_469_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_469_, 0, v___x_468_);
lean_ctor_set(v___x_469_, 1, v___x_467_);
lean_ctor_set(v___x_469_, 2, v___x_466_);
return v___x_469_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_rify__simps___closed__5(void){
_start:
{
lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; 
v___x_470_ = lean_obj_once(&lp_mathlib_Parser_Attr_rify__simps___closed__4, &lp_mathlib_Parser_Attr_rify__simps___closed__4_once, _init_lp_mathlib_Parser_Attr_rify__simps___closed__4);
v___x_471_ = lean_unsigned_to_nat(1022u);
v___x_472_ = ((lean_object*)(lp_mathlib_Parser_Attr_rify__simps___closed__0));
v___x_473_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_473_, 0, v___x_472_);
lean_ctor_set(v___x_473_, 1, v___x_471_);
lean_ctor_set(v___x_473_, 2, v___x_470_);
return v___x_473_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_rify__simps(void){
_start:
{
lean_object* v___x_474_; 
v___x_474_ = lean_obj_once(&lp_mathlib_Parser_Attr_rify__simps___closed__5, &lp_mathlib_Parser_Attr_rify__simps___closed__5_once, _init_lp_mathlib_Parser_Attr_rify__simps___closed__5);
return v___x_474_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28_(){
_start:
{
lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; 
v___x_492_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28_));
v___x_493_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28_));
v___x_494_ = lean_box(0);
v___x_495_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28_));
v___x_496_ = l_Lean_Meta_Simp_registerSimprocAttr(v___x_492_, v___x_493_, v___x_494_, v___x_495_);
return v___x_496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28____boxed(lean_object* v_a_497_){
_start:
{
lean_object* v_res_498_; 
v_res_498_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28_();
return v_res_498_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_rify__simps__proc___closed__2(void){
_start:
{
lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_509_; 
v___x_506_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_507_ = ((lean_object*)(lp_mathlib_Parser_Attr_rify__simps__proc___closed__1));
v___x_508_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_509_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_509_, 0, v___x_508_);
lean_ctor_set(v___x_509_, 1, v___x_507_);
lean_ctor_set(v___x_509_, 2, v___x_506_);
return v___x_509_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_rify__simps__proc___closed__3(void){
_start:
{
lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; 
v___x_510_ = lean_obj_once(&lp_mathlib_Parser_Attr_rify__simps__proc___closed__2, &lp_mathlib_Parser_Attr_rify__simps__proc___closed__2_once, _init_lp_mathlib_Parser_Attr_rify__simps__proc___closed__2);
v___x_511_ = lean_unsigned_to_nat(1022u);
v___x_512_ = ((lean_object*)(lp_mathlib_Parser_Attr_rify__simps__proc___closed__0));
v___x_513_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_513_, 0, v___x_512_);
lean_ctor_set(v___x_513_, 1, v___x_511_);
lean_ctor_set(v___x_513_, 2, v___x_510_);
return v___x_513_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_rify__simps__proc(void){
_start:
{
lean_object* v___x_514_; 
v___x_514_ = lean_obj_once(&lp_mathlib_Parser_Attr_rify__simps__proc___closed__3, &lp_mathlib_Parser_Attr_rify__simps__proc___closed__3_once, _init_lp_mathlib_Parser_Attr_rify__simps__proc___closed__3);
return v___x_514_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; 
v___x_532_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5_));
v___x_533_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5_));
v___x_534_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5_));
v___x_535_ = l_Lean_Meta_registerSimpAttr(v___x_532_, v___x_533_, v___x_534_);
return v___x_535_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5____boxed(lean_object* v_a_536_){
_start:
{
lean_object* v_res_537_; 
v_res_537_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5_();
return v_res_537_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_qify__simps___closed__2(void){
_start:
{
lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; 
v___x_545_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_546_ = ((lean_object*)(lp_mathlib_Parser_Attr_qify__simps___closed__1));
v___x_547_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_548_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_548_, 0, v___x_547_);
lean_ctor_set(v___x_548_, 1, v___x_546_);
lean_ctor_set(v___x_548_, 2, v___x_545_);
return v___x_548_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_qify__simps___closed__3(void){
_start:
{
lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; 
v___x_549_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__15));
v___x_550_ = lean_obj_once(&lp_mathlib_Parser_Attr_qify__simps___closed__2, &lp_mathlib_Parser_Attr_qify__simps___closed__2_once, _init_lp_mathlib_Parser_Attr_qify__simps___closed__2);
v___x_551_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_552_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_552_, 0, v___x_551_);
lean_ctor_set(v___x_552_, 1, v___x_550_);
lean_ctor_set(v___x_552_, 2, v___x_549_);
return v___x_552_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_qify__simps___closed__4(void){
_start:
{
lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; 
v___x_553_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__20));
v___x_554_ = lean_obj_once(&lp_mathlib_Parser_Attr_qify__simps___closed__3, &lp_mathlib_Parser_Attr_qify__simps___closed__3_once, _init_lp_mathlib_Parser_Attr_qify__simps___closed__3);
v___x_555_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_556_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_556_, 0, v___x_555_);
lean_ctor_set(v___x_556_, 1, v___x_554_);
lean_ctor_set(v___x_556_, 2, v___x_553_);
return v___x_556_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_qify__simps___closed__5(void){
_start:
{
lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; 
v___x_557_ = lean_obj_once(&lp_mathlib_Parser_Attr_qify__simps___closed__4, &lp_mathlib_Parser_Attr_qify__simps___closed__4_once, _init_lp_mathlib_Parser_Attr_qify__simps___closed__4);
v___x_558_ = lean_unsigned_to_nat(1022u);
v___x_559_ = ((lean_object*)(lp_mathlib_Parser_Attr_qify__simps___closed__0));
v___x_560_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_560_, 0, v___x_559_);
lean_ctor_set(v___x_560_, 1, v___x_558_);
lean_ctor_set(v___x_560_, 2, v___x_557_);
return v___x_560_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_qify__simps(void){
_start:
{
lean_object* v___x_561_; 
v___x_561_ = lean_obj_once(&lp_mathlib_Parser_Attr_qify__simps___closed__5, &lp_mathlib_Parser_Attr_qify__simps___closed__5_once, _init_lp_mathlib_Parser_Attr_qify__simps___closed__5);
return v___x_561_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28_(){
_start:
{
lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; 
v___x_579_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28_));
v___x_580_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28_));
v___x_581_ = lean_box(0);
v___x_582_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28_));
v___x_583_ = l_Lean_Meta_Simp_registerSimprocAttr(v___x_579_, v___x_580_, v___x_581_, v___x_582_);
return v___x_583_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28____boxed(lean_object* v_a_584_){
_start:
{
lean_object* v_res_585_; 
v_res_585_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28_();
return v_res_585_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_qify__simps__proc___closed__2(void){
_start:
{
lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; 
v___x_593_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_594_ = ((lean_object*)(lp_mathlib_Parser_Attr_qify__simps__proc___closed__1));
v___x_595_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_596_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_596_, 0, v___x_595_);
lean_ctor_set(v___x_596_, 1, v___x_594_);
lean_ctor_set(v___x_596_, 2, v___x_593_);
return v___x_596_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_qify__simps__proc___closed__3(void){
_start:
{
lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; 
v___x_597_ = lean_obj_once(&lp_mathlib_Parser_Attr_qify__simps__proc___closed__2, &lp_mathlib_Parser_Attr_qify__simps__proc___closed__2_once, _init_lp_mathlib_Parser_Attr_qify__simps__proc___closed__2);
v___x_598_ = lean_unsigned_to_nat(1022u);
v___x_599_ = ((lean_object*)(lp_mathlib_Parser_Attr_qify__simps__proc___closed__0));
v___x_600_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_600_, 0, v___x_599_);
lean_ctor_set(v___x_600_, 1, v___x_598_);
lean_ctor_set(v___x_600_, 2, v___x_597_);
return v___x_600_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_qify__simps__proc(void){
_start:
{
lean_object* v___x_601_; 
v___x_601_ = lean_obj_once(&lp_mathlib_Parser_Attr_qify__simps__proc___closed__3, &lp_mathlib_Parser_Attr_qify__simps__proc___closed__3_once, _init_lp_mathlib_Parser_Attr_qify__simps__proc___closed__3);
return v___x_601_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; 
v___x_606_ = lean_unsigned_to_nat(2552594532u);
v___x_607_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__14_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_608_ = l_Lean_Name_num___override(v___x_607_, v___x_606_);
return v___x_608_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; 
v___x_609_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_610_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5_);
v___x_611_ = l_Lean_Name_str___override(v___x_610_, v___x_609_);
return v___x_611_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; 
v___x_612_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_613_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5_);
v___x_614_ = l_Lean_Name_str___override(v___x_613_, v___x_612_);
return v___x_614_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; 
v___x_615_ = lean_unsigned_to_nat(3u);
v___x_616_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5_);
v___x_617_ = l_Lean_Name_num___override(v___x_616_, v___x_615_);
return v___x_617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; 
v___x_619_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5_));
v___x_620_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5_));
v___x_621_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5_);
v___x_622_ = l_Lean_Meta_registerSimpAttr(v___x_619_, v___x_620_, v___x_621_);
return v___x_622_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5____boxed(lean_object* v_a_623_){
_start:
{
lean_object* v_res_624_; 
v_res_624_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5_();
return v_res_624_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_zify__simps___closed__2(void){
_start:
{
lean_object* v___x_632_; lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v___x_635_; 
v___x_632_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_633_ = ((lean_object*)(lp_mathlib_Parser_Attr_zify__simps___closed__1));
v___x_634_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_635_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_635_, 0, v___x_634_);
lean_ctor_set(v___x_635_, 1, v___x_633_);
lean_ctor_set(v___x_635_, 2, v___x_632_);
return v___x_635_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_zify__simps___closed__3(void){
_start:
{
lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v___x_639_; 
v___x_636_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__15));
v___x_637_ = lean_obj_once(&lp_mathlib_Parser_Attr_zify__simps___closed__2, &lp_mathlib_Parser_Attr_zify__simps___closed__2_once, _init_lp_mathlib_Parser_Attr_zify__simps___closed__2);
v___x_638_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_639_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_639_, 0, v___x_638_);
lean_ctor_set(v___x_639_, 1, v___x_637_);
lean_ctor_set(v___x_639_, 2, v___x_636_);
return v___x_639_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_zify__simps___closed__4(void){
_start:
{
lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; 
v___x_640_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__20));
v___x_641_ = lean_obj_once(&lp_mathlib_Parser_Attr_zify__simps___closed__3, &lp_mathlib_Parser_Attr_zify__simps___closed__3_once, _init_lp_mathlib_Parser_Attr_zify__simps___closed__3);
v___x_642_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_643_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_643_, 0, v___x_642_);
lean_ctor_set(v___x_643_, 1, v___x_641_);
lean_ctor_set(v___x_643_, 2, v___x_640_);
return v___x_643_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_zify__simps___closed__5(void){
_start:
{
lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; 
v___x_644_ = lean_obj_once(&lp_mathlib_Parser_Attr_zify__simps___closed__4, &lp_mathlib_Parser_Attr_zify__simps___closed__4_once, _init_lp_mathlib_Parser_Attr_zify__simps___closed__4);
v___x_645_ = lean_unsigned_to_nat(1022u);
v___x_646_ = ((lean_object*)(lp_mathlib_Parser_Attr_zify__simps___closed__0));
v___x_647_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_647_, 0, v___x_646_);
lean_ctor_set(v___x_647_, 1, v___x_645_);
lean_ctor_set(v___x_647_, 2, v___x_644_);
return v___x_647_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_zify__simps(void){
_start:
{
lean_object* v___x_648_; 
v___x_648_ = lean_obj_once(&lp_mathlib_Parser_Attr_zify__simps___closed__5, &lp_mathlib_Parser_Attr_zify__simps___closed__5_once, _init_lp_mathlib_Parser_Attr_zify__simps___closed__5);
return v___x_648_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v___x_655_; 
v___x_653_ = lean_unsigned_to_nat(2552594532u);
v___x_654_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_));
v___x_655_ = l_Lean_Name_num___override(v___x_654_, v___x_653_);
return v___x_655_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_656_; lean_object* v___x_657_; lean_object* v___x_658_; 
v___x_656_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_657_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28_);
v___x_658_ = l_Lean_Name_str___override(v___x_657_, v___x_656_);
return v___x_658_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; 
v___x_659_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_660_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28_);
v___x_661_ = l_Lean_Name_str___override(v___x_660_, v___x_659_);
return v___x_661_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___x_664_; 
v___x_662_ = lean_unsigned_to_nat(3u);
v___x_663_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28_);
v___x_664_ = l_Lean_Name_num___override(v___x_663_, v___x_662_);
return v___x_664_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28_(){
_start:
{
lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; 
v___x_666_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28_));
v___x_667_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28_));
v___x_668_ = lean_box(0);
v___x_669_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28_);
v___x_670_ = l_Lean_Meta_Simp_registerSimprocAttr(v___x_666_, v___x_667_, v___x_668_, v___x_669_);
return v___x_670_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28____boxed(lean_object* v_a_671_){
_start:
{
lean_object* v_res_672_; 
v_res_672_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28_();
return v_res_672_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_zify__simps__proc___closed__2(void){
_start:
{
lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; 
v___x_680_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_681_ = ((lean_object*)(lp_mathlib_Parser_Attr_zify__simps__proc___closed__1));
v___x_682_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_683_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_683_, 0, v___x_682_);
lean_ctor_set(v___x_683_, 1, v___x_681_);
lean_ctor_set(v___x_683_, 2, v___x_680_);
return v___x_683_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_zify__simps__proc___closed__3(void){
_start:
{
lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; 
v___x_684_ = lean_obj_once(&lp_mathlib_Parser_Attr_zify__simps__proc___closed__2, &lp_mathlib_Parser_Attr_zify__simps__proc___closed__2_once, _init_lp_mathlib_Parser_Attr_zify__simps__proc___closed__2);
v___x_685_ = lean_unsigned_to_nat(1022u);
v___x_686_ = ((lean_object*)(lp_mathlib_Parser_Attr_zify__simps__proc___closed__0));
v___x_687_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_687_, 0, v___x_686_);
lean_ctor_set(v___x_687_, 1, v___x_685_);
lean_ctor_set(v___x_687_, 2, v___x_684_);
return v___x_687_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_zify__simps__proc(void){
_start:
{
lean_object* v___x_688_; 
v___x_688_ = lean_obj_once(&lp_mathlib_Parser_Attr_zify__simps__proc___closed__3, &lp_mathlib_Parser_Attr_zify__simps__proc___closed__3_once, _init_lp_mathlib_Parser_Attr_zify__simps__proc___closed__3);
return v___x_688_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; 
v___x_706_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5_));
v___x_707_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5_));
v___x_708_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5_));
v___x_709_ = l_Lean_Meta_registerSimpAttr(v___x_706_, v___x_707_, v___x_708_);
return v___x_709_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5____boxed(lean_object* v_a_710_){
_start:
{
lean_object* v_res_711_; 
v_res_711_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5_();
return v_res_711_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_pull__end___closed__2(void){
_start:
{
lean_object* v___x_719_; lean_object* v___x_720_; lean_object* v___x_721_; lean_object* v___x_722_; 
v___x_719_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_720_ = ((lean_object*)(lp_mathlib_Parser_Attr_pull__end___closed__1));
v___x_721_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_722_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_722_, 0, v___x_721_);
lean_ctor_set(v___x_722_, 1, v___x_720_);
lean_ctor_set(v___x_722_, 2, v___x_719_);
return v___x_722_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_pull__end___closed__3(void){
_start:
{
lean_object* v___x_723_; lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; 
v___x_723_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__15));
v___x_724_ = lean_obj_once(&lp_mathlib_Parser_Attr_pull__end___closed__2, &lp_mathlib_Parser_Attr_pull__end___closed__2_once, _init_lp_mathlib_Parser_Attr_pull__end___closed__2);
v___x_725_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_726_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_726_, 0, v___x_725_);
lean_ctor_set(v___x_726_, 1, v___x_724_);
lean_ctor_set(v___x_726_, 2, v___x_723_);
return v___x_726_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_pull__end___closed__4(void){
_start:
{
lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; 
v___x_727_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__20));
v___x_728_ = lean_obj_once(&lp_mathlib_Parser_Attr_pull__end___closed__3, &lp_mathlib_Parser_Attr_pull__end___closed__3_once, _init_lp_mathlib_Parser_Attr_pull__end___closed__3);
v___x_729_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_730_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_730_, 0, v___x_729_);
lean_ctor_set(v___x_730_, 1, v___x_728_);
lean_ctor_set(v___x_730_, 2, v___x_727_);
return v___x_730_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_pull__end___closed__5(void){
_start:
{
lean_object* v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; 
v___x_731_ = lean_obj_once(&lp_mathlib_Parser_Attr_pull__end___closed__4, &lp_mathlib_Parser_Attr_pull__end___closed__4_once, _init_lp_mathlib_Parser_Attr_pull__end___closed__4);
v___x_732_ = lean_unsigned_to_nat(1022u);
v___x_733_ = ((lean_object*)(lp_mathlib_Parser_Attr_pull__end___closed__0));
v___x_734_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_734_, 0, v___x_733_);
lean_ctor_set(v___x_734_, 1, v___x_732_);
lean_ctor_set(v___x_734_, 2, v___x_731_);
return v___x_734_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_pull__end(void){
_start:
{
lean_object* v___x_735_; 
v___x_735_ = lean_obj_once(&lp_mathlib_Parser_Attr_pull__end___closed__5, &lp_mathlib_Parser_Attr_pull__end___closed__5_once, _init_lp_mathlib_Parser_Attr_pull__end___closed__5);
return v___x_735_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28_(){
_start:
{
lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; 
v___x_753_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28_));
v___x_754_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28_));
v___x_755_ = lean_box(0);
v___x_756_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28_));
v___x_757_ = l_Lean_Meta_Simp_registerSimprocAttr(v___x_753_, v___x_754_, v___x_755_, v___x_756_);
return v___x_757_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28____boxed(lean_object* v_a_758_){
_start:
{
lean_object* v_res_759_; 
v_res_759_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28_();
return v_res_759_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_pull__end__proc___closed__2(void){
_start:
{
lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_770_; 
v___x_767_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_768_ = ((lean_object*)(lp_mathlib_Parser_Attr_pull__end__proc___closed__1));
v___x_769_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_770_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_770_, 0, v___x_769_);
lean_ctor_set(v___x_770_, 1, v___x_768_);
lean_ctor_set(v___x_770_, 2, v___x_767_);
return v___x_770_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_pull__end__proc___closed__3(void){
_start:
{
lean_object* v___x_771_; lean_object* v___x_772_; lean_object* v___x_773_; lean_object* v___x_774_; 
v___x_771_ = lean_obj_once(&lp_mathlib_Parser_Attr_pull__end__proc___closed__2, &lp_mathlib_Parser_Attr_pull__end__proc___closed__2_once, _init_lp_mathlib_Parser_Attr_pull__end__proc___closed__2);
v___x_772_ = lean_unsigned_to_nat(1022u);
v___x_773_ = ((lean_object*)(lp_mathlib_Parser_Attr_pull__end__proc___closed__0));
v___x_774_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_774_, 0, v___x_773_);
lean_ctor_set(v___x_774_, 1, v___x_772_);
lean_ctor_set(v___x_774_, 2, v___x_771_);
return v___x_774_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_pull__end__proc(void){
_start:
{
lean_object* v___x_775_; 
v___x_775_ = lean_obj_once(&lp_mathlib_Parser_Attr_pull__end__proc___closed__3, &lp_mathlib_Parser_Attr_pull__end__proc___closed__3_once, _init_lp_mathlib_Parser_Attr_pull__end__proc___closed__3);
return v___x_775_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_793_; lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___x_796_; 
v___x_793_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5_));
v___x_794_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5_));
v___x_795_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5_));
v___x_796_ = l_Lean_Meta_registerSimpAttr(v___x_793_, v___x_794_, v___x_795_);
return v___x_796_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5____boxed(lean_object* v_a_797_){
_start:
{
lean_object* v_res_798_; 
v_res_798_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5_();
return v_res_798_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_push__end___closed__2(void){
_start:
{
lean_object* v___x_806_; lean_object* v___x_807_; lean_object* v___x_808_; lean_object* v___x_809_; 
v___x_806_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_807_ = ((lean_object*)(lp_mathlib_Parser_Attr_push__end___closed__1));
v___x_808_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_809_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_809_, 0, v___x_808_);
lean_ctor_set(v___x_809_, 1, v___x_807_);
lean_ctor_set(v___x_809_, 2, v___x_806_);
return v___x_809_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_push__end___closed__3(void){
_start:
{
lean_object* v___x_810_; lean_object* v___x_811_; lean_object* v___x_812_; lean_object* v___x_813_; 
v___x_810_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__15));
v___x_811_ = lean_obj_once(&lp_mathlib_Parser_Attr_push__end___closed__2, &lp_mathlib_Parser_Attr_push__end___closed__2_once, _init_lp_mathlib_Parser_Attr_push__end___closed__2);
v___x_812_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_813_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_813_, 0, v___x_812_);
lean_ctor_set(v___x_813_, 1, v___x_811_);
lean_ctor_set(v___x_813_, 2, v___x_810_);
return v___x_813_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_push__end___closed__4(void){
_start:
{
lean_object* v___x_814_; lean_object* v___x_815_; lean_object* v___x_816_; lean_object* v___x_817_; 
v___x_814_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__20));
v___x_815_ = lean_obj_once(&lp_mathlib_Parser_Attr_push__end___closed__3, &lp_mathlib_Parser_Attr_push__end___closed__3_once, _init_lp_mathlib_Parser_Attr_push__end___closed__3);
v___x_816_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_817_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_817_, 0, v___x_816_);
lean_ctor_set(v___x_817_, 1, v___x_815_);
lean_ctor_set(v___x_817_, 2, v___x_814_);
return v___x_817_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_push__end___closed__5(void){
_start:
{
lean_object* v___x_818_; lean_object* v___x_819_; lean_object* v___x_820_; lean_object* v___x_821_; 
v___x_818_ = lean_obj_once(&lp_mathlib_Parser_Attr_push__end___closed__4, &lp_mathlib_Parser_Attr_push__end___closed__4_once, _init_lp_mathlib_Parser_Attr_push__end___closed__4);
v___x_819_ = lean_unsigned_to_nat(1022u);
v___x_820_ = ((lean_object*)(lp_mathlib_Parser_Attr_push__end___closed__0));
v___x_821_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_821_, 0, v___x_820_);
lean_ctor_set(v___x_821_, 1, v___x_819_);
lean_ctor_set(v___x_821_, 2, v___x_818_);
return v___x_821_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_push__end(void){
_start:
{
lean_object* v___x_822_; 
v___x_822_ = lean_obj_once(&lp_mathlib_Parser_Attr_push__end___closed__5, &lp_mathlib_Parser_Attr_push__end___closed__5_once, _init_lp_mathlib_Parser_Attr_push__end___closed__5);
return v___x_822_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28_(){
_start:
{
lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v___x_842_; lean_object* v___x_843_; lean_object* v___x_844_; 
v___x_840_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28_));
v___x_841_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28_));
v___x_842_ = lean_box(0);
v___x_843_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28_));
v___x_844_ = l_Lean_Meta_Simp_registerSimprocAttr(v___x_840_, v___x_841_, v___x_842_, v___x_843_);
return v___x_844_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28____boxed(lean_object* v_a_845_){
_start:
{
lean_object* v_res_846_; 
v_res_846_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28_();
return v_res_846_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_push__end__proc___closed__2(void){
_start:
{
lean_object* v___x_854_; lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v___x_857_; 
v___x_854_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_855_ = ((lean_object*)(lp_mathlib_Parser_Attr_push__end__proc___closed__1));
v___x_856_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_857_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_857_, 0, v___x_856_);
lean_ctor_set(v___x_857_, 1, v___x_855_);
lean_ctor_set(v___x_857_, 2, v___x_854_);
return v___x_857_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_push__end__proc___closed__3(void){
_start:
{
lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_861_; 
v___x_858_ = lean_obj_once(&lp_mathlib_Parser_Attr_push__end__proc___closed__2, &lp_mathlib_Parser_Attr_push__end__proc___closed__2_once, _init_lp_mathlib_Parser_Attr_push__end__proc___closed__2);
v___x_859_ = lean_unsigned_to_nat(1022u);
v___x_860_ = ((lean_object*)(lp_mathlib_Parser_Attr_push__end__proc___closed__0));
v___x_861_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_861_, 0, v___x_860_);
lean_ctor_set(v___x_861_, 1, v___x_859_);
lean_ctor_set(v___x_861_, 2, v___x_858_);
return v___x_861_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_push__end__proc(void){
_start:
{
lean_object* v___x_862_; 
v___x_862_ = lean_obj_once(&lp_mathlib_Parser_Attr_push__end__proc___closed__3, &lp_mathlib_Parser_Attr_push__end__proc___closed__3_once, _init_lp_mathlib_Parser_Attr_push__end__proc___closed__3);
return v___x_862_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_880_; lean_object* v___x_881_; lean_object* v___x_882_; lean_object* v___x_883_; 
v___x_880_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5_));
v___x_881_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5_));
v___x_882_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5_));
v___x_883_ = l_Lean_Meta_registerSimpAttr(v___x_880_, v___x_881_, v___x_882_);
return v___x_883_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5____boxed(lean_object* v_a_884_){
_start:
{
lean_object* v_res_885_; 
v_res_885_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5_();
return v_res_885_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_mfld__simps___closed__2(void){
_start:
{
lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; lean_object* v___x_896_; 
v___x_893_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_894_ = ((lean_object*)(lp_mathlib_Parser_Attr_mfld__simps___closed__1));
v___x_895_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_896_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_896_, 0, v___x_895_);
lean_ctor_set(v___x_896_, 1, v___x_894_);
lean_ctor_set(v___x_896_, 2, v___x_893_);
return v___x_896_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_mfld__simps___closed__3(void){
_start:
{
lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; 
v___x_897_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__15));
v___x_898_ = lean_obj_once(&lp_mathlib_Parser_Attr_mfld__simps___closed__2, &lp_mathlib_Parser_Attr_mfld__simps___closed__2_once, _init_lp_mathlib_Parser_Attr_mfld__simps___closed__2);
v___x_899_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_900_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_900_, 0, v___x_899_);
lean_ctor_set(v___x_900_, 1, v___x_898_);
lean_ctor_set(v___x_900_, 2, v___x_897_);
return v___x_900_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_mfld__simps___closed__4(void){
_start:
{
lean_object* v___x_901_; lean_object* v___x_902_; lean_object* v___x_903_; lean_object* v___x_904_; 
v___x_901_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__20));
v___x_902_ = lean_obj_once(&lp_mathlib_Parser_Attr_mfld__simps___closed__3, &lp_mathlib_Parser_Attr_mfld__simps___closed__3_once, _init_lp_mathlib_Parser_Attr_mfld__simps___closed__3);
v___x_903_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_904_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_904_, 0, v___x_903_);
lean_ctor_set(v___x_904_, 1, v___x_902_);
lean_ctor_set(v___x_904_, 2, v___x_901_);
return v___x_904_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_mfld__simps___closed__5(void){
_start:
{
lean_object* v___x_905_; lean_object* v___x_906_; lean_object* v___x_907_; lean_object* v___x_908_; 
v___x_905_ = lean_obj_once(&lp_mathlib_Parser_Attr_mfld__simps___closed__4, &lp_mathlib_Parser_Attr_mfld__simps___closed__4_once, _init_lp_mathlib_Parser_Attr_mfld__simps___closed__4);
v___x_906_ = lean_unsigned_to_nat(1022u);
v___x_907_ = ((lean_object*)(lp_mathlib_Parser_Attr_mfld__simps___closed__0));
v___x_908_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_908_, 0, v___x_907_);
lean_ctor_set(v___x_908_, 1, v___x_906_);
lean_ctor_set(v___x_908_, 2, v___x_905_);
return v___x_908_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_mfld__simps(void){
_start:
{
lean_object* v___x_909_; 
v___x_909_ = lean_obj_once(&lp_mathlib_Parser_Attr_mfld__simps___closed__5, &lp_mathlib_Parser_Attr_mfld__simps___closed__5_once, _init_lp_mathlib_Parser_Attr_mfld__simps___closed__5);
return v___x_909_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28_(){
_start:
{
lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_930_; lean_object* v___x_931_; 
v___x_927_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28_));
v___x_928_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28_));
v___x_929_ = lean_box(0);
v___x_930_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28_));
v___x_931_ = l_Lean_Meta_Simp_registerSimprocAttr(v___x_927_, v___x_928_, v___x_929_, v___x_930_);
return v___x_931_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28____boxed(lean_object* v_a_932_){
_start:
{
lean_object* v_res_933_; 
v_res_933_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28_();
return v_res_933_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_mfld__simps__proc___closed__2(void){
_start:
{
lean_object* v___x_941_; lean_object* v___x_942_; lean_object* v___x_943_; lean_object* v___x_944_; 
v___x_941_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_942_ = ((lean_object*)(lp_mathlib_Parser_Attr_mfld__simps__proc___closed__1));
v___x_943_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_944_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_944_, 0, v___x_943_);
lean_ctor_set(v___x_944_, 1, v___x_942_);
lean_ctor_set(v___x_944_, 2, v___x_941_);
return v___x_944_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_mfld__simps__proc___closed__3(void){
_start:
{
lean_object* v___x_945_; lean_object* v___x_946_; lean_object* v___x_947_; lean_object* v___x_948_; 
v___x_945_ = lean_obj_once(&lp_mathlib_Parser_Attr_mfld__simps__proc___closed__2, &lp_mathlib_Parser_Attr_mfld__simps__proc___closed__2_once, _init_lp_mathlib_Parser_Attr_mfld__simps__proc___closed__2);
v___x_946_ = lean_unsigned_to_nat(1022u);
v___x_947_ = ((lean_object*)(lp_mathlib_Parser_Attr_mfld__simps__proc___closed__0));
v___x_948_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_948_, 0, v___x_947_);
lean_ctor_set(v___x_948_, 1, v___x_946_);
lean_ctor_set(v___x_948_, 2, v___x_945_);
return v___x_948_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_mfld__simps__proc(void){
_start:
{
lean_object* v___x_949_; 
v___x_949_ = lean_obj_once(&lp_mathlib_Parser_Attr_mfld__simps__proc___closed__3, &lp_mathlib_Parser_Attr_mfld__simps__proc___closed__3_once, _init_lp_mathlib_Parser_Attr_mfld__simps__proc___closed__3);
return v___x_949_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_954_; lean_object* v___x_955_; lean_object* v___x_956_; 
v___x_954_ = lean_unsigned_to_nat(3849875863u);
v___x_955_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__14_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_956_ = l_Lean_Name_num___override(v___x_955_, v___x_954_);
return v___x_956_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_957_; lean_object* v___x_958_; lean_object* v___x_959_; 
v___x_957_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_958_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5_);
v___x_959_ = l_Lean_Name_str___override(v___x_958_, v___x_957_);
return v___x_959_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_960_; lean_object* v___x_961_; lean_object* v___x_962_; 
v___x_960_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_961_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5_);
v___x_962_ = l_Lean_Name_str___override(v___x_961_, v___x_960_);
return v___x_962_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_963_; lean_object* v___x_964_; lean_object* v___x_965_; 
v___x_963_ = lean_unsigned_to_nat(3u);
v___x_964_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5_);
v___x_965_ = l_Lean_Name_num___override(v___x_964_, v___x_963_);
return v___x_965_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_967_; lean_object* v___x_968_; lean_object* v___x_969_; lean_object* v___x_970_; 
v___x_967_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5_));
v___x_968_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5_));
v___x_969_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5_);
v___x_970_ = l_Lean_Meta_registerSimpAttr(v___x_967_, v___x_968_, v___x_969_);
return v___x_970_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5____boxed(lean_object* v_a_971_){
_start:
{
lean_object* v_res_972_; 
v_res_972_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5_();
return v_res_972_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_integral__simps___closed__2(void){
_start:
{
lean_object* v___x_980_; lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; 
v___x_980_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_981_ = ((lean_object*)(lp_mathlib_Parser_Attr_integral__simps___closed__1));
v___x_982_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_983_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_983_, 0, v___x_982_);
lean_ctor_set(v___x_983_, 1, v___x_981_);
lean_ctor_set(v___x_983_, 2, v___x_980_);
return v___x_983_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_integral__simps___closed__3(void){
_start:
{
lean_object* v___x_984_; lean_object* v___x_985_; lean_object* v___x_986_; lean_object* v___x_987_; 
v___x_984_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__15));
v___x_985_ = lean_obj_once(&lp_mathlib_Parser_Attr_integral__simps___closed__2, &lp_mathlib_Parser_Attr_integral__simps___closed__2_once, _init_lp_mathlib_Parser_Attr_integral__simps___closed__2);
v___x_986_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_987_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_987_, 0, v___x_986_);
lean_ctor_set(v___x_987_, 1, v___x_985_);
lean_ctor_set(v___x_987_, 2, v___x_984_);
return v___x_987_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_integral__simps___closed__4(void){
_start:
{
lean_object* v___x_988_; lean_object* v___x_989_; lean_object* v___x_990_; lean_object* v___x_991_; 
v___x_988_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__20));
v___x_989_ = lean_obj_once(&lp_mathlib_Parser_Attr_integral__simps___closed__3, &lp_mathlib_Parser_Attr_integral__simps___closed__3_once, _init_lp_mathlib_Parser_Attr_integral__simps___closed__3);
v___x_990_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_991_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_991_, 0, v___x_990_);
lean_ctor_set(v___x_991_, 1, v___x_989_);
lean_ctor_set(v___x_991_, 2, v___x_988_);
return v___x_991_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_integral__simps___closed__5(void){
_start:
{
lean_object* v___x_992_; lean_object* v___x_993_; lean_object* v___x_994_; lean_object* v___x_995_; 
v___x_992_ = lean_obj_once(&lp_mathlib_Parser_Attr_integral__simps___closed__4, &lp_mathlib_Parser_Attr_integral__simps___closed__4_once, _init_lp_mathlib_Parser_Attr_integral__simps___closed__4);
v___x_993_ = lean_unsigned_to_nat(1022u);
v___x_994_ = ((lean_object*)(lp_mathlib_Parser_Attr_integral__simps___closed__0));
v___x_995_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_995_, 0, v___x_994_);
lean_ctor_set(v___x_995_, 1, v___x_993_);
lean_ctor_set(v___x_995_, 2, v___x_992_);
return v___x_995_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_integral__simps(void){
_start:
{
lean_object* v___x_996_; 
v___x_996_ = lean_obj_once(&lp_mathlib_Parser_Attr_integral__simps___closed__5, &lp_mathlib_Parser_Attr_integral__simps___closed__5_once, _init_lp_mathlib_Parser_Attr_integral__simps___closed__5);
return v___x_996_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_1001_; lean_object* v___x_1002_; lean_object* v___x_1003_; 
v___x_1001_ = lean_unsigned_to_nat(3849875863u);
v___x_1002_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_));
v___x_1003_ = l_Lean_Name_num___override(v___x_1002_, v___x_1001_);
return v___x_1003_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_1004_; lean_object* v___x_1005_; lean_object* v___x_1006_; 
v___x_1004_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_1005_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28_);
v___x_1006_ = l_Lean_Name_str___override(v___x_1005_, v___x_1004_);
return v___x_1006_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_1007_; lean_object* v___x_1008_; lean_object* v___x_1009_; 
v___x_1007_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_1008_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28_);
v___x_1009_ = l_Lean_Name_str___override(v___x_1008_, v___x_1007_);
return v___x_1009_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_1010_; lean_object* v___x_1011_; lean_object* v___x_1012_; 
v___x_1010_ = lean_unsigned_to_nat(3u);
v___x_1011_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28_);
v___x_1012_ = l_Lean_Name_num___override(v___x_1011_, v___x_1010_);
return v___x_1012_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28_(){
_start:
{
lean_object* v___x_1014_; lean_object* v___x_1015_; lean_object* v___x_1016_; lean_object* v___x_1017_; lean_object* v___x_1018_; 
v___x_1014_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28_));
v___x_1015_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28_));
v___x_1016_ = lean_box(0);
v___x_1017_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28_);
v___x_1018_ = l_Lean_Meta_Simp_registerSimprocAttr(v___x_1014_, v___x_1015_, v___x_1016_, v___x_1017_);
return v___x_1018_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28____boxed(lean_object* v_a_1019_){
_start:
{
lean_object* v_res_1020_; 
v_res_1020_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28_();
return v_res_1020_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_integral__simps__proc___closed__2(void){
_start:
{
lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; 
v___x_1028_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_1029_ = ((lean_object*)(lp_mathlib_Parser_Attr_integral__simps__proc___closed__1));
v___x_1030_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1031_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1031_, 0, v___x_1030_);
lean_ctor_set(v___x_1031_, 1, v___x_1029_);
lean_ctor_set(v___x_1031_, 2, v___x_1028_);
return v___x_1031_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_integral__simps__proc___closed__3(void){
_start:
{
lean_object* v___x_1032_; lean_object* v___x_1033_; lean_object* v___x_1034_; lean_object* v___x_1035_; 
v___x_1032_ = lean_obj_once(&lp_mathlib_Parser_Attr_integral__simps__proc___closed__2, &lp_mathlib_Parser_Attr_integral__simps__proc___closed__2_once, _init_lp_mathlib_Parser_Attr_integral__simps__proc___closed__2);
v___x_1033_ = lean_unsigned_to_nat(1022u);
v___x_1034_ = ((lean_object*)(lp_mathlib_Parser_Attr_integral__simps__proc___closed__0));
v___x_1035_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1035_, 0, v___x_1034_);
lean_ctor_set(v___x_1035_, 1, v___x_1033_);
lean_ctor_set(v___x_1035_, 2, v___x_1032_);
return v___x_1035_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_integral__simps__proc(void){
_start:
{
lean_object* v___x_1036_; 
v___x_1036_ = lean_obj_once(&lp_mathlib_Parser_Attr_integral__simps__proc___closed__3, &lp_mathlib_Parser_Attr_integral__simps__proc___closed__3_once, _init_lp_mathlib_Parser_Attr_integral__simps__proc___closed__3);
return v___x_1036_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_1054_; lean_object* v___x_1055_; lean_object* v___x_1056_; lean_object* v___x_1057_; 
v___x_1054_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5_));
v___x_1055_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5_));
v___x_1056_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5_));
v___x_1057_ = l_Lean_Meta_registerSimpAttr(v___x_1054_, v___x_1055_, v___x_1056_);
return v___x_1057_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5____boxed(lean_object* v_a_1058_){
_start:
{
lean_object* v_res_1059_; 
v_res_1059_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5_();
return v_res_1059_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_typevec___closed__2(void){
_start:
{
lean_object* v___x_1067_; lean_object* v___x_1068_; lean_object* v___x_1069_; lean_object* v___x_1070_; 
v___x_1067_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_1068_ = ((lean_object*)(lp_mathlib_Parser_Attr_typevec___closed__1));
v___x_1069_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1070_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1070_, 0, v___x_1069_);
lean_ctor_set(v___x_1070_, 1, v___x_1068_);
lean_ctor_set(v___x_1070_, 2, v___x_1067_);
return v___x_1070_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_typevec___closed__3(void){
_start:
{
lean_object* v___x_1071_; lean_object* v___x_1072_; lean_object* v___x_1073_; lean_object* v___x_1074_; 
v___x_1071_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__15));
v___x_1072_ = lean_obj_once(&lp_mathlib_Parser_Attr_typevec___closed__2, &lp_mathlib_Parser_Attr_typevec___closed__2_once, _init_lp_mathlib_Parser_Attr_typevec___closed__2);
v___x_1073_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1074_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1074_, 0, v___x_1073_);
lean_ctor_set(v___x_1074_, 1, v___x_1072_);
lean_ctor_set(v___x_1074_, 2, v___x_1071_);
return v___x_1074_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_typevec___closed__4(void){
_start:
{
lean_object* v___x_1075_; lean_object* v___x_1076_; lean_object* v___x_1077_; lean_object* v___x_1078_; 
v___x_1075_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__20));
v___x_1076_ = lean_obj_once(&lp_mathlib_Parser_Attr_typevec___closed__3, &lp_mathlib_Parser_Attr_typevec___closed__3_once, _init_lp_mathlib_Parser_Attr_typevec___closed__3);
v___x_1077_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1078_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1078_, 0, v___x_1077_);
lean_ctor_set(v___x_1078_, 1, v___x_1076_);
lean_ctor_set(v___x_1078_, 2, v___x_1075_);
return v___x_1078_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_typevec___closed__5(void){
_start:
{
lean_object* v___x_1079_; lean_object* v___x_1080_; lean_object* v___x_1081_; lean_object* v___x_1082_; 
v___x_1079_ = lean_obj_once(&lp_mathlib_Parser_Attr_typevec___closed__4, &lp_mathlib_Parser_Attr_typevec___closed__4_once, _init_lp_mathlib_Parser_Attr_typevec___closed__4);
v___x_1080_ = lean_unsigned_to_nat(1022u);
v___x_1081_ = ((lean_object*)(lp_mathlib_Parser_Attr_typevec___closed__0));
v___x_1082_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1082_, 0, v___x_1081_);
lean_ctor_set(v___x_1082_, 1, v___x_1080_);
lean_ctor_set(v___x_1082_, 2, v___x_1079_);
return v___x_1082_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_typevec(void){
_start:
{
lean_object* v___x_1083_; 
v___x_1083_ = lean_obj_once(&lp_mathlib_Parser_Attr_typevec___closed__5, &lp_mathlib_Parser_Attr_typevec___closed__5_once, _init_lp_mathlib_Parser_Attr_typevec___closed__5);
return v___x_1083_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28_(){
_start:
{
lean_object* v___x_1101_; lean_object* v___x_1102_; lean_object* v___x_1103_; lean_object* v___x_1104_; lean_object* v___x_1105_; 
v___x_1101_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28_));
v___x_1102_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28_));
v___x_1103_ = lean_box(0);
v___x_1104_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28_));
v___x_1105_ = l_Lean_Meta_Simp_registerSimprocAttr(v___x_1101_, v___x_1102_, v___x_1103_, v___x_1104_);
return v___x_1105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28____boxed(lean_object* v_a_1106_){
_start:
{
lean_object* v_res_1107_; 
v_res_1107_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28_();
return v_res_1107_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_typevec__proc___closed__2(void){
_start:
{
lean_object* v___x_1115_; lean_object* v___x_1116_; lean_object* v___x_1117_; lean_object* v___x_1118_; 
v___x_1115_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_1116_ = ((lean_object*)(lp_mathlib_Parser_Attr_typevec__proc___closed__1));
v___x_1117_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1118_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1118_, 0, v___x_1117_);
lean_ctor_set(v___x_1118_, 1, v___x_1116_);
lean_ctor_set(v___x_1118_, 2, v___x_1115_);
return v___x_1118_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_typevec__proc___closed__3(void){
_start:
{
lean_object* v___x_1119_; lean_object* v___x_1120_; lean_object* v___x_1121_; lean_object* v___x_1122_; 
v___x_1119_ = lean_obj_once(&lp_mathlib_Parser_Attr_typevec__proc___closed__2, &lp_mathlib_Parser_Attr_typevec__proc___closed__2_once, _init_lp_mathlib_Parser_Attr_typevec__proc___closed__2);
v___x_1120_ = lean_unsigned_to_nat(1022u);
v___x_1121_ = ((lean_object*)(lp_mathlib_Parser_Attr_typevec__proc___closed__0));
v___x_1122_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1122_, 0, v___x_1121_);
lean_ctor_set(v___x_1122_, 1, v___x_1120_);
lean_ctor_set(v___x_1122_, 2, v___x_1119_);
return v___x_1122_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_typevec__proc(void){
_start:
{
lean_object* v___x_1123_; 
v___x_1123_ = lean_obj_once(&lp_mathlib_Parser_Attr_typevec__proc___closed__3, &lp_mathlib_Parser_Attr_typevec__proc___closed__3_once, _init_lp_mathlib_Parser_Attr_typevec__proc___closed__3);
return v___x_1123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_1141_; lean_object* v___x_1142_; lean_object* v___x_1143_; lean_object* v___x_1144_; 
v___x_1141_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5_));
v___x_1142_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5_));
v___x_1143_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5_));
v___x_1144_ = l_Lean_Meta_registerSimpAttr(v___x_1141_, v___x_1142_, v___x_1143_);
return v___x_1144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5____boxed(lean_object* v_a_1145_){
_start:
{
lean_object* v_res_1146_; 
v_res_1146_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5_();
return v_res_1146_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_ghost__simps___closed__2(void){
_start:
{
lean_object* v___x_1154_; lean_object* v___x_1155_; lean_object* v___x_1156_; lean_object* v___x_1157_; 
v___x_1154_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_1155_ = ((lean_object*)(lp_mathlib_Parser_Attr_ghost__simps___closed__1));
v___x_1156_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1157_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1157_, 0, v___x_1156_);
lean_ctor_set(v___x_1157_, 1, v___x_1155_);
lean_ctor_set(v___x_1157_, 2, v___x_1154_);
return v___x_1157_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_ghost__simps___closed__3(void){
_start:
{
lean_object* v___x_1158_; lean_object* v___x_1159_; lean_object* v___x_1160_; lean_object* v___x_1161_; 
v___x_1158_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__15));
v___x_1159_ = lean_obj_once(&lp_mathlib_Parser_Attr_ghost__simps___closed__2, &lp_mathlib_Parser_Attr_ghost__simps___closed__2_once, _init_lp_mathlib_Parser_Attr_ghost__simps___closed__2);
v___x_1160_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1161_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1161_, 0, v___x_1160_);
lean_ctor_set(v___x_1161_, 1, v___x_1159_);
lean_ctor_set(v___x_1161_, 2, v___x_1158_);
return v___x_1161_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_ghost__simps___closed__4(void){
_start:
{
lean_object* v___x_1162_; lean_object* v___x_1163_; lean_object* v___x_1164_; lean_object* v___x_1165_; 
v___x_1162_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__20));
v___x_1163_ = lean_obj_once(&lp_mathlib_Parser_Attr_ghost__simps___closed__3, &lp_mathlib_Parser_Attr_ghost__simps___closed__3_once, _init_lp_mathlib_Parser_Attr_ghost__simps___closed__3);
v___x_1164_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1165_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1165_, 0, v___x_1164_);
lean_ctor_set(v___x_1165_, 1, v___x_1163_);
lean_ctor_set(v___x_1165_, 2, v___x_1162_);
return v___x_1165_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_ghost__simps___closed__5(void){
_start:
{
lean_object* v___x_1166_; lean_object* v___x_1167_; lean_object* v___x_1168_; lean_object* v___x_1169_; 
v___x_1166_ = lean_obj_once(&lp_mathlib_Parser_Attr_ghost__simps___closed__4, &lp_mathlib_Parser_Attr_ghost__simps___closed__4_once, _init_lp_mathlib_Parser_Attr_ghost__simps___closed__4);
v___x_1167_ = lean_unsigned_to_nat(1022u);
v___x_1168_ = ((lean_object*)(lp_mathlib_Parser_Attr_ghost__simps___closed__0));
v___x_1169_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1169_, 0, v___x_1168_);
lean_ctor_set(v___x_1169_, 1, v___x_1167_);
lean_ctor_set(v___x_1169_, 2, v___x_1166_);
return v___x_1169_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_ghost__simps(void){
_start:
{
lean_object* v___x_1170_; 
v___x_1170_ = lean_obj_once(&lp_mathlib_Parser_Attr_ghost__simps___closed__5, &lp_mathlib_Parser_Attr_ghost__simps___closed__5_once, _init_lp_mathlib_Parser_Attr_ghost__simps___closed__5);
return v___x_1170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28_(){
_start:
{
lean_object* v___x_1188_; lean_object* v___x_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; lean_object* v___x_1192_; 
v___x_1188_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28_));
v___x_1189_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28_));
v___x_1190_ = lean_box(0);
v___x_1191_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28_));
v___x_1192_ = l_Lean_Meta_Simp_registerSimprocAttr(v___x_1188_, v___x_1189_, v___x_1190_, v___x_1191_);
return v___x_1192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28____boxed(lean_object* v_a_1193_){
_start:
{
lean_object* v_res_1194_; 
v_res_1194_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28_();
return v_res_1194_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_ghost__simps__proc___closed__2(void){
_start:
{
lean_object* v___x_1202_; lean_object* v___x_1203_; lean_object* v___x_1204_; lean_object* v___x_1205_; 
v___x_1202_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_1203_ = ((lean_object*)(lp_mathlib_Parser_Attr_ghost__simps__proc___closed__1));
v___x_1204_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1205_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1205_, 0, v___x_1204_);
lean_ctor_set(v___x_1205_, 1, v___x_1203_);
lean_ctor_set(v___x_1205_, 2, v___x_1202_);
return v___x_1205_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_ghost__simps__proc___closed__3(void){
_start:
{
lean_object* v___x_1206_; lean_object* v___x_1207_; lean_object* v___x_1208_; lean_object* v___x_1209_; 
v___x_1206_ = lean_obj_once(&lp_mathlib_Parser_Attr_ghost__simps__proc___closed__2, &lp_mathlib_Parser_Attr_ghost__simps__proc___closed__2_once, _init_lp_mathlib_Parser_Attr_ghost__simps__proc___closed__2);
v___x_1207_ = lean_unsigned_to_nat(1022u);
v___x_1208_ = ((lean_object*)(lp_mathlib_Parser_Attr_ghost__simps__proc___closed__0));
v___x_1209_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1209_, 0, v___x_1208_);
lean_ctor_set(v___x_1209_, 1, v___x_1207_);
lean_ctor_set(v___x_1209_, 2, v___x_1206_);
return v___x_1209_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_ghost__simps__proc(void){
_start:
{
lean_object* v___x_1210_; 
v___x_1210_ = lean_obj_once(&lp_mathlib_Parser_Attr_ghost__simps__proc___closed__3, &lp_mathlib_Parser_Attr_ghost__simps__proc___closed__3_once, _init_lp_mathlib_Parser_Attr_ghost__simps__proc___closed__3);
return v___x_1210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_1228_; lean_object* v___x_1229_; lean_object* v___x_1230_; lean_object* v___x_1231_; 
v___x_1228_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5_));
v___x_1229_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5_));
v___x_1230_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5_));
v___x_1231_ = l_Lean_Meta_registerSimpAttr(v___x_1228_, v___x_1229_, v___x_1230_);
return v___x_1231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5____boxed(lean_object* v_a_1232_){
_start:
{
lean_object* v_res_1233_; 
v_res_1233_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5_();
return v_res_1233_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_nontriviality___closed__2(void){
_start:
{
lean_object* v___x_1241_; lean_object* v___x_1242_; lean_object* v___x_1243_; lean_object* v___x_1244_; 
v___x_1241_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_1242_ = ((lean_object*)(lp_mathlib_Parser_Attr_nontriviality___closed__1));
v___x_1243_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1244_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1244_, 0, v___x_1243_);
lean_ctor_set(v___x_1244_, 1, v___x_1242_);
lean_ctor_set(v___x_1244_, 2, v___x_1241_);
return v___x_1244_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_nontriviality___closed__3(void){
_start:
{
lean_object* v___x_1245_; lean_object* v___x_1246_; lean_object* v___x_1247_; lean_object* v___x_1248_; 
v___x_1245_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__15));
v___x_1246_ = lean_obj_once(&lp_mathlib_Parser_Attr_nontriviality___closed__2, &lp_mathlib_Parser_Attr_nontriviality___closed__2_once, _init_lp_mathlib_Parser_Attr_nontriviality___closed__2);
v___x_1247_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1248_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1248_, 0, v___x_1247_);
lean_ctor_set(v___x_1248_, 1, v___x_1246_);
lean_ctor_set(v___x_1248_, 2, v___x_1245_);
return v___x_1248_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_nontriviality___closed__4(void){
_start:
{
lean_object* v___x_1249_; lean_object* v___x_1250_; lean_object* v___x_1251_; lean_object* v___x_1252_; 
v___x_1249_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__20));
v___x_1250_ = lean_obj_once(&lp_mathlib_Parser_Attr_nontriviality___closed__3, &lp_mathlib_Parser_Attr_nontriviality___closed__3_once, _init_lp_mathlib_Parser_Attr_nontriviality___closed__3);
v___x_1251_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1252_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1252_, 0, v___x_1251_);
lean_ctor_set(v___x_1252_, 1, v___x_1250_);
lean_ctor_set(v___x_1252_, 2, v___x_1249_);
return v___x_1252_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_nontriviality___closed__5(void){
_start:
{
lean_object* v___x_1253_; lean_object* v___x_1254_; lean_object* v___x_1255_; lean_object* v___x_1256_; 
v___x_1253_ = lean_obj_once(&lp_mathlib_Parser_Attr_nontriviality___closed__4, &lp_mathlib_Parser_Attr_nontriviality___closed__4_once, _init_lp_mathlib_Parser_Attr_nontriviality___closed__4);
v___x_1254_ = lean_unsigned_to_nat(1022u);
v___x_1255_ = ((lean_object*)(lp_mathlib_Parser_Attr_nontriviality___closed__0));
v___x_1256_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1256_, 0, v___x_1255_);
lean_ctor_set(v___x_1256_, 1, v___x_1254_);
lean_ctor_set(v___x_1256_, 2, v___x_1253_);
return v___x_1256_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_nontriviality(void){
_start:
{
lean_object* v___x_1257_; 
v___x_1257_ = lean_obj_once(&lp_mathlib_Parser_Attr_nontriviality___closed__5, &lp_mathlib_Parser_Attr_nontriviality___closed__5_once, _init_lp_mathlib_Parser_Attr_nontriviality___closed__5);
return v___x_1257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28_(){
_start:
{
lean_object* v___x_1275_; lean_object* v___x_1276_; lean_object* v___x_1277_; lean_object* v___x_1278_; lean_object* v___x_1279_; 
v___x_1275_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28_));
v___x_1276_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28_));
v___x_1277_ = lean_box(0);
v___x_1278_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28_));
v___x_1279_ = l_Lean_Meta_Simp_registerSimprocAttr(v___x_1275_, v___x_1276_, v___x_1277_, v___x_1278_);
return v___x_1279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28____boxed(lean_object* v_a_1280_){
_start:
{
lean_object* v_res_1281_; 
v_res_1281_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28_();
return v_res_1281_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_nontriviality__proc___closed__2(void){
_start:
{
lean_object* v___x_1289_; lean_object* v___x_1290_; lean_object* v___x_1291_; lean_object* v___x_1292_; 
v___x_1289_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_1290_ = ((lean_object*)(lp_mathlib_Parser_Attr_nontriviality__proc___closed__1));
v___x_1291_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1292_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1292_, 0, v___x_1291_);
lean_ctor_set(v___x_1292_, 1, v___x_1290_);
lean_ctor_set(v___x_1292_, 2, v___x_1289_);
return v___x_1292_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_nontriviality__proc___closed__3(void){
_start:
{
lean_object* v___x_1293_; lean_object* v___x_1294_; lean_object* v___x_1295_; lean_object* v___x_1296_; 
v___x_1293_ = lean_obj_once(&lp_mathlib_Parser_Attr_nontriviality__proc___closed__2, &lp_mathlib_Parser_Attr_nontriviality__proc___closed__2_once, _init_lp_mathlib_Parser_Attr_nontriviality__proc___closed__2);
v___x_1294_ = lean_unsigned_to_nat(1022u);
v___x_1295_ = ((lean_object*)(lp_mathlib_Parser_Attr_nontriviality__proc___closed__0));
v___x_1296_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1296_, 0, v___x_1295_);
lean_ctor_set(v___x_1296_, 1, v___x_1294_);
lean_ctor_set(v___x_1296_, 2, v___x_1293_);
return v___x_1296_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_nontriviality__proc(void){
_start:
{
lean_object* v___x_1297_; 
v___x_1297_ = lean_obj_once(&lp_mathlib_Parser_Attr_nontriviality__proc___closed__3, &lp_mathlib_Parser_Attr_nontriviality__proc___closed__3_once, _init_lp_mathlib_Parser_Attr_nontriviality__proc___closed__3);
return v___x_1297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2811593030____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_1303_; lean_object* v___x_1304_; lean_object* v___x_1305_; 
v___x_1303_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2811593030____hygCtx___hyg_5_));
v___x_1304_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2811593030____hygCtx___hyg_5_));
v___x_1305_ = l_Lean_registerLabelAttr(v___x_1303_, v___x_1304_, v___x_1303_);
return v___x_1305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2811593030____hygCtx___hyg_5____boxed(lean_object* v_a_1306_){
_start:
{
lean_object* v_res_1307_; 
v_res_1307_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2811593030____hygCtx___hyg_5_();
return v_res_1307_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_1324_; lean_object* v___x_1325_; lean_object* v___x_1326_; 
v___x_1324_ = lean_unsigned_to_nat(3021440532u);
v___x_1325_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__14_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_1326_ = l_Lean_Name_num___override(v___x_1325_, v___x_1324_);
return v___x_1326_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_1327_; lean_object* v___x_1328_; lean_object* v___x_1329_; 
v___x_1327_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_1328_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5_);
v___x_1329_ = l_Lean_Name_str___override(v___x_1328_, v___x_1327_);
return v___x_1329_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_1330_; lean_object* v___x_1331_; lean_object* v___x_1332_; 
v___x_1330_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_1331_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5_);
v___x_1332_ = l_Lean_Name_str___override(v___x_1331_, v___x_1330_);
return v___x_1332_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_1333_; lean_object* v___x_1334_; lean_object* v___x_1335_; 
v___x_1333_ = lean_unsigned_to_nat(3u);
v___x_1334_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5_);
v___x_1335_ = l_Lean_Name_num___override(v___x_1334_, v___x_1333_);
return v___x_1335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_1337_; lean_object* v___x_1338_; lean_object* v___x_1339_; lean_object* v___x_1340_; 
v___x_1337_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5_));
v___x_1338_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5_));
v___x_1339_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5_);
v___x_1340_ = l_Lean_Meta_registerSimpAttr(v___x_1337_, v___x_1338_, v___x_1339_);
return v___x_1340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5____boxed(lean_object* v_a_1341_){
_start:
{
lean_object* v_res_1342_; 
v_res_1342_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5_();
return v_res_1342_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_fin__omega___closed__2(void){
_start:
{
lean_object* v___x_1350_; lean_object* v___x_1351_; lean_object* v___x_1352_; lean_object* v___x_1353_; 
v___x_1350_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_1351_ = ((lean_object*)(lp_mathlib_Parser_Attr_fin__omega___closed__1));
v___x_1352_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1353_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1353_, 0, v___x_1352_);
lean_ctor_set(v___x_1353_, 1, v___x_1351_);
lean_ctor_set(v___x_1353_, 2, v___x_1350_);
return v___x_1353_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_fin__omega___closed__3(void){
_start:
{
lean_object* v___x_1354_; lean_object* v___x_1355_; lean_object* v___x_1356_; lean_object* v___x_1357_; 
v___x_1354_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__15));
v___x_1355_ = lean_obj_once(&lp_mathlib_Parser_Attr_fin__omega___closed__2, &lp_mathlib_Parser_Attr_fin__omega___closed__2_once, _init_lp_mathlib_Parser_Attr_fin__omega___closed__2);
v___x_1356_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1357_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1357_, 0, v___x_1356_);
lean_ctor_set(v___x_1357_, 1, v___x_1355_);
lean_ctor_set(v___x_1357_, 2, v___x_1354_);
return v___x_1357_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_fin__omega___closed__4(void){
_start:
{
lean_object* v___x_1358_; lean_object* v___x_1359_; lean_object* v___x_1360_; lean_object* v___x_1361_; 
v___x_1358_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__20));
v___x_1359_ = lean_obj_once(&lp_mathlib_Parser_Attr_fin__omega___closed__3, &lp_mathlib_Parser_Attr_fin__omega___closed__3_once, _init_lp_mathlib_Parser_Attr_fin__omega___closed__3);
v___x_1360_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1361_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1361_, 0, v___x_1360_);
lean_ctor_set(v___x_1361_, 1, v___x_1359_);
lean_ctor_set(v___x_1361_, 2, v___x_1358_);
return v___x_1361_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_fin__omega___closed__5(void){
_start:
{
lean_object* v___x_1362_; lean_object* v___x_1363_; lean_object* v___x_1364_; lean_object* v___x_1365_; 
v___x_1362_ = lean_obj_once(&lp_mathlib_Parser_Attr_fin__omega___closed__4, &lp_mathlib_Parser_Attr_fin__omega___closed__4_once, _init_lp_mathlib_Parser_Attr_fin__omega___closed__4);
v___x_1363_ = lean_unsigned_to_nat(1022u);
v___x_1364_ = ((lean_object*)(lp_mathlib_Parser_Attr_fin__omega___closed__0));
v___x_1365_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1365_, 0, v___x_1364_);
lean_ctor_set(v___x_1365_, 1, v___x_1363_);
lean_ctor_set(v___x_1365_, 2, v___x_1362_);
return v___x_1365_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_fin__omega(void){
_start:
{
lean_object* v___x_1366_; 
v___x_1366_ = lean_obj_once(&lp_mathlib_Parser_Attr_fin__omega___closed__5, &lp_mathlib_Parser_Attr_fin__omega___closed__5_once, _init_lp_mathlib_Parser_Attr_fin__omega___closed__5);
return v___x_1366_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_1371_; lean_object* v___x_1372_; lean_object* v___x_1373_; 
v___x_1371_ = lean_unsigned_to_nat(3021440532u);
v___x_1372_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_));
v___x_1373_ = l_Lean_Name_num___override(v___x_1372_, v___x_1371_);
return v___x_1373_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_1374_; lean_object* v___x_1375_; lean_object* v___x_1376_; 
v___x_1374_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_1375_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28_);
v___x_1376_ = l_Lean_Name_str___override(v___x_1375_, v___x_1374_);
return v___x_1376_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_1377_; lean_object* v___x_1378_; lean_object* v___x_1379_; 
v___x_1377_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_1378_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28_);
v___x_1379_ = l_Lean_Name_str___override(v___x_1378_, v___x_1377_);
return v___x_1379_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_1380_; lean_object* v___x_1381_; lean_object* v___x_1382_; 
v___x_1380_ = lean_unsigned_to_nat(3u);
v___x_1381_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28_);
v___x_1382_ = l_Lean_Name_num___override(v___x_1381_, v___x_1380_);
return v___x_1382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28_(){
_start:
{
lean_object* v___x_1384_; lean_object* v___x_1385_; lean_object* v___x_1386_; lean_object* v___x_1387_; lean_object* v___x_1388_; 
v___x_1384_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28_));
v___x_1385_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28_));
v___x_1386_ = lean_box(0);
v___x_1387_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28_);
v___x_1388_ = l_Lean_Meta_Simp_registerSimprocAttr(v___x_1384_, v___x_1385_, v___x_1386_, v___x_1387_);
return v___x_1388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28____boxed(lean_object* v_a_1389_){
_start:
{
lean_object* v_res_1390_; 
v_res_1390_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28_();
return v_res_1390_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_fin__omega__proc___closed__2(void){
_start:
{
lean_object* v___x_1398_; lean_object* v___x_1399_; lean_object* v___x_1400_; lean_object* v___x_1401_; 
v___x_1398_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_1399_ = ((lean_object*)(lp_mathlib_Parser_Attr_fin__omega__proc___closed__1));
v___x_1400_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1401_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1401_, 0, v___x_1400_);
lean_ctor_set(v___x_1401_, 1, v___x_1399_);
lean_ctor_set(v___x_1401_, 2, v___x_1398_);
return v___x_1401_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_fin__omega__proc___closed__3(void){
_start:
{
lean_object* v___x_1402_; lean_object* v___x_1403_; lean_object* v___x_1404_; lean_object* v___x_1405_; 
v___x_1402_ = lean_obj_once(&lp_mathlib_Parser_Attr_fin__omega__proc___closed__2, &lp_mathlib_Parser_Attr_fin__omega__proc___closed__2_once, _init_lp_mathlib_Parser_Attr_fin__omega__proc___closed__2);
v___x_1403_ = lean_unsigned_to_nat(1022u);
v___x_1404_ = ((lean_object*)(lp_mathlib_Parser_Attr_fin__omega__proc___closed__0));
v___x_1405_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1405_, 0, v___x_1404_);
lean_ctor_set(v___x_1405_, 1, v___x_1403_);
lean_ctor_set(v___x_1405_, 2, v___x_1402_);
return v___x_1405_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_fin__omega__proc(void){
_start:
{
lean_object* v___x_1406_; 
v___x_1406_ = lean_obj_once(&lp_mathlib_Parser_Attr_fin__omega__proc___closed__3, &lp_mathlib_Parser_Attr_fin__omega__proc___closed__3_once, _init_lp_mathlib_Parser_Attr_fin__omega__proc___closed__3);
return v___x_1406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_1424_; lean_object* v___x_1425_; lean_object* v___x_1426_; lean_object* v___x_1427_; 
v___x_1424_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5_));
v___x_1425_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5_));
v___x_1426_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5_));
v___x_1427_ = l_Lean_Meta_registerSimpAttr(v___x_1424_, v___x_1425_, v___x_1426_);
return v___x_1427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5____boxed(lean_object* v_a_1428_){
_start:
{
lean_object* v_res_1429_; 
v_res_1429_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5_();
return v_res_1429_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_enat__to__nat__top___closed__2(void){
_start:
{
lean_object* v___x_1437_; lean_object* v___x_1438_; lean_object* v___x_1439_; lean_object* v___x_1440_; 
v___x_1437_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_1438_ = ((lean_object*)(lp_mathlib_Parser_Attr_enat__to__nat__top___closed__1));
v___x_1439_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1440_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1440_, 0, v___x_1439_);
lean_ctor_set(v___x_1440_, 1, v___x_1438_);
lean_ctor_set(v___x_1440_, 2, v___x_1437_);
return v___x_1440_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_enat__to__nat__top___closed__3(void){
_start:
{
lean_object* v___x_1441_; lean_object* v___x_1442_; lean_object* v___x_1443_; lean_object* v___x_1444_; 
v___x_1441_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__15));
v___x_1442_ = lean_obj_once(&lp_mathlib_Parser_Attr_enat__to__nat__top___closed__2, &lp_mathlib_Parser_Attr_enat__to__nat__top___closed__2_once, _init_lp_mathlib_Parser_Attr_enat__to__nat__top___closed__2);
v___x_1443_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1444_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1444_, 0, v___x_1443_);
lean_ctor_set(v___x_1444_, 1, v___x_1442_);
lean_ctor_set(v___x_1444_, 2, v___x_1441_);
return v___x_1444_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_enat__to__nat__top___closed__4(void){
_start:
{
lean_object* v___x_1445_; lean_object* v___x_1446_; lean_object* v___x_1447_; lean_object* v___x_1448_; 
v___x_1445_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__20));
v___x_1446_ = lean_obj_once(&lp_mathlib_Parser_Attr_enat__to__nat__top___closed__3, &lp_mathlib_Parser_Attr_enat__to__nat__top___closed__3_once, _init_lp_mathlib_Parser_Attr_enat__to__nat__top___closed__3);
v___x_1447_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1448_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1448_, 0, v___x_1447_);
lean_ctor_set(v___x_1448_, 1, v___x_1446_);
lean_ctor_set(v___x_1448_, 2, v___x_1445_);
return v___x_1448_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_enat__to__nat__top___closed__5(void){
_start:
{
lean_object* v___x_1449_; lean_object* v___x_1450_; lean_object* v___x_1451_; lean_object* v___x_1452_; 
v___x_1449_ = lean_obj_once(&lp_mathlib_Parser_Attr_enat__to__nat__top___closed__4, &lp_mathlib_Parser_Attr_enat__to__nat__top___closed__4_once, _init_lp_mathlib_Parser_Attr_enat__to__nat__top___closed__4);
v___x_1450_ = lean_unsigned_to_nat(1022u);
v___x_1451_ = ((lean_object*)(lp_mathlib_Parser_Attr_enat__to__nat__top___closed__0));
v___x_1452_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1452_, 0, v___x_1451_);
lean_ctor_set(v___x_1452_, 1, v___x_1450_);
lean_ctor_set(v___x_1452_, 2, v___x_1449_);
return v___x_1452_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_enat__to__nat__top(void){
_start:
{
lean_object* v___x_1453_; 
v___x_1453_ = lean_obj_once(&lp_mathlib_Parser_Attr_enat__to__nat__top___closed__5, &lp_mathlib_Parser_Attr_enat__to__nat__top___closed__5_once, _init_lp_mathlib_Parser_Attr_enat__to__nat__top___closed__5);
return v___x_1453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28_(){
_start:
{
lean_object* v___x_1471_; lean_object* v___x_1472_; lean_object* v___x_1473_; lean_object* v___x_1474_; lean_object* v___x_1475_; 
v___x_1471_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28_));
v___x_1472_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28_));
v___x_1473_ = lean_box(0);
v___x_1474_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28_));
v___x_1475_ = l_Lean_Meta_Simp_registerSimprocAttr(v___x_1471_, v___x_1472_, v___x_1473_, v___x_1474_);
return v___x_1475_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28____boxed(lean_object* v_a_1476_){
_start:
{
lean_object* v_res_1477_; 
v_res_1477_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28_();
return v_res_1477_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_enat__to__nat__top__proc___closed__2(void){
_start:
{
lean_object* v___x_1485_; lean_object* v___x_1486_; lean_object* v___x_1487_; lean_object* v___x_1488_; 
v___x_1485_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_1486_ = ((lean_object*)(lp_mathlib_Parser_Attr_enat__to__nat__top__proc___closed__1));
v___x_1487_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1488_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1488_, 0, v___x_1487_);
lean_ctor_set(v___x_1488_, 1, v___x_1486_);
lean_ctor_set(v___x_1488_, 2, v___x_1485_);
return v___x_1488_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_enat__to__nat__top__proc___closed__3(void){
_start:
{
lean_object* v___x_1489_; lean_object* v___x_1490_; lean_object* v___x_1491_; lean_object* v___x_1492_; 
v___x_1489_ = lean_obj_once(&lp_mathlib_Parser_Attr_enat__to__nat__top__proc___closed__2, &lp_mathlib_Parser_Attr_enat__to__nat__top__proc___closed__2_once, _init_lp_mathlib_Parser_Attr_enat__to__nat__top__proc___closed__2);
v___x_1490_ = lean_unsigned_to_nat(1022u);
v___x_1491_ = ((lean_object*)(lp_mathlib_Parser_Attr_enat__to__nat__top__proc___closed__0));
v___x_1492_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1492_, 0, v___x_1491_);
lean_ctor_set(v___x_1492_, 1, v___x_1490_);
lean_ctor_set(v___x_1492_, 2, v___x_1489_);
return v___x_1492_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_enat__to__nat__top__proc(void){
_start:
{
lean_object* v___x_1493_; 
v___x_1493_ = lean_obj_once(&lp_mathlib_Parser_Attr_enat__to__nat__top__proc___closed__3, &lp_mathlib_Parser_Attr_enat__to__nat__top__proc___closed__3_once, _init_lp_mathlib_Parser_Attr_enat__to__nat__top__proc___closed__3);
return v___x_1493_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_1511_; lean_object* v___x_1512_; lean_object* v___x_1513_; lean_object* v___x_1514_; 
v___x_1511_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5_));
v___x_1512_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5_));
v___x_1513_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5_));
v___x_1514_ = l_Lean_Meta_registerSimpAttr(v___x_1511_, v___x_1512_, v___x_1513_);
return v___x_1514_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5____boxed(lean_object* v_a_1515_){
_start:
{
lean_object* v_res_1516_; 
v_res_1516_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5_();
return v_res_1516_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__2(void){
_start:
{
lean_object* v___x_1524_; lean_object* v___x_1525_; lean_object* v___x_1526_; lean_object* v___x_1527_; 
v___x_1524_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_1525_ = ((lean_object*)(lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__1));
v___x_1526_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1527_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1527_, 0, v___x_1526_);
lean_ctor_set(v___x_1527_, 1, v___x_1525_);
lean_ctor_set(v___x_1527_, 2, v___x_1524_);
return v___x_1527_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__3(void){
_start:
{
lean_object* v___x_1528_; lean_object* v___x_1529_; lean_object* v___x_1530_; lean_object* v___x_1531_; 
v___x_1528_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__15));
v___x_1529_ = lean_obj_once(&lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__2, &lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__2_once, _init_lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__2);
v___x_1530_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1531_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1531_, 0, v___x_1530_);
lean_ctor_set(v___x_1531_, 1, v___x_1529_);
lean_ctor_set(v___x_1531_, 2, v___x_1528_);
return v___x_1531_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__4(void){
_start:
{
lean_object* v___x_1532_; lean_object* v___x_1533_; lean_object* v___x_1534_; lean_object* v___x_1535_; 
v___x_1532_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__20));
v___x_1533_ = lean_obj_once(&lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__3, &lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__3_once, _init_lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__3);
v___x_1534_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1535_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1535_, 0, v___x_1534_);
lean_ctor_set(v___x_1535_, 1, v___x_1533_);
lean_ctor_set(v___x_1535_, 2, v___x_1532_);
return v___x_1535_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__5(void){
_start:
{
lean_object* v___x_1536_; lean_object* v___x_1537_; lean_object* v___x_1538_; lean_object* v___x_1539_; 
v___x_1536_ = lean_obj_once(&lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__4, &lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__4_once, _init_lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__4);
v___x_1537_ = lean_unsigned_to_nat(1022u);
v___x_1538_ = ((lean_object*)(lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__0));
v___x_1539_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1539_, 0, v___x_1538_);
lean_ctor_set(v___x_1539_, 1, v___x_1537_);
lean_ctor_set(v___x_1539_, 2, v___x_1536_);
return v___x_1539_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_enat__to__nat__coe(void){
_start:
{
lean_object* v___x_1540_; 
v___x_1540_ = lean_obj_once(&lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__5, &lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__5_once, _init_lp_mathlib_Parser_Attr_enat__to__nat__coe___closed__5);
return v___x_1540_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28_(){
_start:
{
lean_object* v___x_1558_; lean_object* v___x_1559_; lean_object* v___x_1560_; lean_object* v___x_1561_; lean_object* v___x_1562_; 
v___x_1558_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28_));
v___x_1559_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28_));
v___x_1560_ = lean_box(0);
v___x_1561_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28_));
v___x_1562_ = l_Lean_Meta_Simp_registerSimprocAttr(v___x_1558_, v___x_1559_, v___x_1560_, v___x_1561_);
return v___x_1562_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28____boxed(lean_object* v_a_1563_){
_start:
{
lean_object* v_res_1564_; 
v_res_1564_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28_();
return v_res_1564_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_enat__to__nat__coe__proc___closed__2(void){
_start:
{
lean_object* v___x_1572_; lean_object* v___x_1573_; lean_object* v___x_1574_; lean_object* v___x_1575_; 
v___x_1572_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_1573_ = ((lean_object*)(lp_mathlib_Parser_Attr_enat__to__nat__coe__proc___closed__1));
v___x_1574_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1575_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1575_, 0, v___x_1574_);
lean_ctor_set(v___x_1575_, 1, v___x_1573_);
lean_ctor_set(v___x_1575_, 2, v___x_1572_);
return v___x_1575_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_enat__to__nat__coe__proc___closed__3(void){
_start:
{
lean_object* v___x_1576_; lean_object* v___x_1577_; lean_object* v___x_1578_; lean_object* v___x_1579_; 
v___x_1576_ = lean_obj_once(&lp_mathlib_Parser_Attr_enat__to__nat__coe__proc___closed__2, &lp_mathlib_Parser_Attr_enat__to__nat__coe__proc___closed__2_once, _init_lp_mathlib_Parser_Attr_enat__to__nat__coe__proc___closed__2);
v___x_1577_ = lean_unsigned_to_nat(1022u);
v___x_1578_ = ((lean_object*)(lp_mathlib_Parser_Attr_enat__to__nat__coe__proc___closed__0));
v___x_1579_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1579_, 0, v___x_1578_);
lean_ctor_set(v___x_1579_, 1, v___x_1577_);
lean_ctor_set(v___x_1579_, 2, v___x_1576_);
return v___x_1579_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_enat__to__nat__coe__proc(void){
_start:
{
lean_object* v___x_1580_; 
v___x_1580_ = lean_obj_once(&lp_mathlib_Parser_Attr_enat__to__nat__coe__proc___closed__3, &lp_mathlib_Parser_Attr_enat__to__nat__coe__proc___closed__3_once, _init_lp_mathlib_Parser_Attr_enat__to__nat__coe__proc___closed__3);
return v___x_1580_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_1585_; lean_object* v___x_1586_; lean_object* v___x_1587_; 
v___x_1585_ = lean_unsigned_to_nat(3369699945u);
v___x_1586_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__14_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_1587_ = l_Lean_Name_num___override(v___x_1586_, v___x_1585_);
return v___x_1587_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_1588_; lean_object* v___x_1589_; lean_object* v___x_1590_; 
v___x_1588_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_1589_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5_);
v___x_1590_ = l_Lean_Name_str___override(v___x_1589_, v___x_1588_);
return v___x_1590_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_1591_; lean_object* v___x_1592_; lean_object* v___x_1593_; 
v___x_1591_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_1592_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5_);
v___x_1593_ = l_Lean_Name_str___override(v___x_1592_, v___x_1591_);
return v___x_1593_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_1594_; lean_object* v___x_1595_; lean_object* v___x_1596_; 
v___x_1594_ = lean_unsigned_to_nat(3u);
v___x_1595_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5_);
v___x_1596_ = l_Lean_Name_num___override(v___x_1595_, v___x_1594_);
return v___x_1596_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_1598_; lean_object* v___x_1599_; lean_object* v___x_1600_; lean_object* v___x_1601_; 
v___x_1598_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5_));
v___x_1599_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5_));
v___x_1600_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5_);
v___x_1601_ = l_Lean_Meta_registerSimpAttr(v___x_1598_, v___x_1599_, v___x_1600_);
return v___x_1601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5____boxed(lean_object* v_a_1602_){
_start:
{
lean_object* v_res_1603_; 
v_res_1603_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5_();
return v_res_1603_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__2(void){
_start:
{
lean_object* v___x_1611_; lean_object* v___x_1612_; lean_object* v___x_1613_; lean_object* v___x_1614_; 
v___x_1611_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_1612_ = ((lean_object*)(lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__1));
v___x_1613_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1614_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1614_, 0, v___x_1613_);
lean_ctor_set(v___x_1614_, 1, v___x_1612_);
lean_ctor_set(v___x_1614_, 2, v___x_1611_);
return v___x_1614_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__3(void){
_start:
{
lean_object* v___x_1615_; lean_object* v___x_1616_; lean_object* v___x_1617_; lean_object* v___x_1618_; 
v___x_1615_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__15));
v___x_1616_ = lean_obj_once(&lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__2, &lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__2_once, _init_lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__2);
v___x_1617_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1618_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1618_, 0, v___x_1617_);
lean_ctor_set(v___x_1618_, 1, v___x_1616_);
lean_ctor_set(v___x_1618_, 2, v___x_1615_);
return v___x_1618_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__4(void){
_start:
{
lean_object* v___x_1619_; lean_object* v___x_1620_; lean_object* v___x_1621_; lean_object* v___x_1622_; 
v___x_1619_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__20));
v___x_1620_ = lean_obj_once(&lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__3, &lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__3_once, _init_lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__3);
v___x_1621_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1622_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1622_, 0, v___x_1621_);
lean_ctor_set(v___x_1622_, 1, v___x_1620_);
lean_ctor_set(v___x_1622_, 2, v___x_1619_);
return v___x_1622_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__5(void){
_start:
{
lean_object* v___x_1623_; lean_object* v___x_1624_; lean_object* v___x_1625_; lean_object* v___x_1626_; 
v___x_1623_ = lean_obj_once(&lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__4, &lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__4_once, _init_lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__4);
v___x_1624_ = lean_unsigned_to_nat(1022u);
v___x_1625_ = ((lean_object*)(lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__0));
v___x_1626_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1626_, 0, v___x_1625_);
lean_ctor_set(v___x_1626_, 1, v___x_1624_);
lean_ctor_set(v___x_1626_, 2, v___x_1623_);
return v___x_1626_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_pnat__to__nat__coe(void){
_start:
{
lean_object* v___x_1627_; 
v___x_1627_ = lean_obj_once(&lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__5, &lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__5_once, _init_lp_mathlib_Parser_Attr_pnat__to__nat__coe___closed__5);
return v___x_1627_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_1632_; lean_object* v___x_1633_; lean_object* v___x_1634_; 
v___x_1632_ = lean_unsigned_to_nat(3369699945u);
v___x_1633_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_));
v___x_1634_ = l_Lean_Name_num___override(v___x_1633_, v___x_1632_);
return v___x_1634_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_1635_; lean_object* v___x_1636_; lean_object* v___x_1637_; 
v___x_1635_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_1636_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28_);
v___x_1637_ = l_Lean_Name_str___override(v___x_1636_, v___x_1635_);
return v___x_1637_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_1638_; lean_object* v___x_1639_; lean_object* v___x_1640_; 
v___x_1638_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_1639_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28_);
v___x_1640_ = l_Lean_Name_str___override(v___x_1639_, v___x_1638_);
return v___x_1640_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_1641_; lean_object* v___x_1642_; lean_object* v___x_1643_; 
v___x_1641_ = lean_unsigned_to_nat(3u);
v___x_1642_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28_);
v___x_1643_ = l_Lean_Name_num___override(v___x_1642_, v___x_1641_);
return v___x_1643_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28_(){
_start:
{
lean_object* v___x_1645_; lean_object* v___x_1646_; lean_object* v___x_1647_; lean_object* v___x_1648_; lean_object* v___x_1649_; 
v___x_1645_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28_));
v___x_1646_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28_));
v___x_1647_ = lean_box(0);
v___x_1648_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28_);
v___x_1649_ = l_Lean_Meta_Simp_registerSimprocAttr(v___x_1645_, v___x_1646_, v___x_1647_, v___x_1648_);
return v___x_1649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28____boxed(lean_object* v_a_1650_){
_start:
{
lean_object* v_res_1651_; 
v_res_1651_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28_();
return v_res_1651_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc___closed__2(void){
_start:
{
lean_object* v___x_1659_; lean_object* v___x_1660_; lean_object* v___x_1661_; lean_object* v___x_1662_; 
v___x_1659_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_1660_ = ((lean_object*)(lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc___closed__1));
v___x_1661_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1662_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1662_, 0, v___x_1661_);
lean_ctor_set(v___x_1662_, 1, v___x_1660_);
lean_ctor_set(v___x_1662_, 2, v___x_1659_);
return v___x_1662_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc___closed__3(void){
_start:
{
lean_object* v___x_1663_; lean_object* v___x_1664_; lean_object* v___x_1665_; lean_object* v___x_1666_; 
v___x_1663_ = lean_obj_once(&lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc___closed__2, &lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc___closed__2_once, _init_lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc___closed__2);
v___x_1664_ = lean_unsigned_to_nat(1022u);
v___x_1665_ = ((lean_object*)(lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc___closed__0));
v___x_1666_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1666_, 0, v___x_1665_);
lean_ctor_set(v___x_1666_, 1, v___x_1664_);
lean_ctor_set(v___x_1666_, 2, v___x_1663_);
return v___x_1666_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc(void){
_start:
{
lean_object* v___x_1667_; 
v___x_1667_ = lean_obj_once(&lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc___closed__3, &lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc___closed__3_once, _init_lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc___closed__3);
return v___x_1667_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_1672_; lean_object* v___x_1673_; lean_object* v___x_1674_; 
v___x_1672_ = lean_unsigned_to_nat(3059584346u);
v___x_1673_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__14_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_1674_ = l_Lean_Name_num___override(v___x_1673_, v___x_1672_);
return v___x_1674_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_1675_; lean_object* v___x_1676_; lean_object* v___x_1677_; 
v___x_1675_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_1676_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5_);
v___x_1677_ = l_Lean_Name_str___override(v___x_1676_, v___x_1675_);
return v___x_1677_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_1678_; lean_object* v___x_1679_; lean_object* v___x_1680_; 
v___x_1678_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_1679_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5_);
v___x_1680_ = l_Lean_Name_str___override(v___x_1679_, v___x_1678_);
return v___x_1680_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5_(void){
_start:
{
lean_object* v___x_1681_; lean_object* v___x_1682_; lean_object* v___x_1683_; 
v___x_1681_ = lean_unsigned_to_nat(3u);
v___x_1682_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5_);
v___x_1683_ = l_Lean_Name_num___override(v___x_1682_, v___x_1681_);
return v___x_1683_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_1685_; lean_object* v___x_1686_; lean_object* v___x_1687_; lean_object* v___x_1688_; 
v___x_1685_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5_));
v___x_1686_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5_));
v___x_1687_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5_);
v___x_1688_ = l_Lean_Meta_registerSimpAttr(v___x_1685_, v___x_1686_, v___x_1687_);
return v___x_1688_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5____boxed(lean_object* v_a_1689_){
_start:
{
lean_object* v_res_1690_; 
v_res_1690_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5_();
return v_res_1690_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_mon__tauto___closed__2(void){
_start:
{
lean_object* v___x_1698_; lean_object* v___x_1699_; lean_object* v___x_1700_; lean_object* v___x_1701_; 
v___x_1698_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_1699_ = ((lean_object*)(lp_mathlib_Parser_Attr_mon__tauto___closed__1));
v___x_1700_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1701_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1701_, 0, v___x_1700_);
lean_ctor_set(v___x_1701_, 1, v___x_1699_);
lean_ctor_set(v___x_1701_, 2, v___x_1698_);
return v___x_1701_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_mon__tauto___closed__3(void){
_start:
{
lean_object* v___x_1702_; lean_object* v___x_1703_; lean_object* v___x_1704_; lean_object* v___x_1705_; 
v___x_1702_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__15));
v___x_1703_ = lean_obj_once(&lp_mathlib_Parser_Attr_mon__tauto___closed__2, &lp_mathlib_Parser_Attr_mon__tauto___closed__2_once, _init_lp_mathlib_Parser_Attr_mon__tauto___closed__2);
v___x_1704_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1705_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1705_, 0, v___x_1704_);
lean_ctor_set(v___x_1705_, 1, v___x_1703_);
lean_ctor_set(v___x_1705_, 2, v___x_1702_);
return v___x_1705_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_mon__tauto___closed__4(void){
_start:
{
lean_object* v___x_1706_; lean_object* v___x_1707_; lean_object* v___x_1708_; lean_object* v___x_1709_; 
v___x_1706_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__20));
v___x_1707_ = lean_obj_once(&lp_mathlib_Parser_Attr_mon__tauto___closed__3, &lp_mathlib_Parser_Attr_mon__tauto___closed__3_once, _init_lp_mathlib_Parser_Attr_mon__tauto___closed__3);
v___x_1708_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1709_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1709_, 0, v___x_1708_);
lean_ctor_set(v___x_1709_, 1, v___x_1707_);
lean_ctor_set(v___x_1709_, 2, v___x_1706_);
return v___x_1709_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_mon__tauto___closed__5(void){
_start:
{
lean_object* v___x_1710_; lean_object* v___x_1711_; lean_object* v___x_1712_; lean_object* v___x_1713_; 
v___x_1710_ = lean_obj_once(&lp_mathlib_Parser_Attr_mon__tauto___closed__4, &lp_mathlib_Parser_Attr_mon__tauto___closed__4_once, _init_lp_mathlib_Parser_Attr_mon__tauto___closed__4);
v___x_1711_ = lean_unsigned_to_nat(1022u);
v___x_1712_ = ((lean_object*)(lp_mathlib_Parser_Attr_mon__tauto___closed__0));
v___x_1713_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1713_, 0, v___x_1712_);
lean_ctor_set(v___x_1713_, 1, v___x_1711_);
lean_ctor_set(v___x_1713_, 2, v___x_1710_);
return v___x_1713_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_mon__tauto(void){
_start:
{
lean_object* v___x_1714_; 
v___x_1714_ = lean_obj_once(&lp_mathlib_Parser_Attr_mon__tauto___closed__5, &lp_mathlib_Parser_Attr_mon__tauto___closed__5_once, _init_lp_mathlib_Parser_Attr_mon__tauto___closed__5);
return v___x_1714_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_1719_; lean_object* v___x_1720_; lean_object* v___x_1721_; 
v___x_1719_ = lean_unsigned_to_nat(3059584346u);
v___x_1720_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__9_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_));
v___x_1721_ = l_Lean_Name_num___override(v___x_1720_, v___x_1719_);
return v___x_1721_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_1722_; lean_object* v___x_1723_; lean_object* v___x_1724_; 
v___x_1722_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__16_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_1723_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__3_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28_);
v___x_1724_ = l_Lean_Name_str___override(v___x_1723_, v___x_1722_);
return v___x_1724_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_1725_; lean_object* v___x_1726_; lean_object* v___x_1727_; 
v___x_1725_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__18_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_));
v___x_1726_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__4_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28_);
v___x_1727_ = l_Lean_Name_str___override(v___x_1726_, v___x_1725_);
return v___x_1727_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28_(void){
_start:
{
lean_object* v___x_1728_; lean_object* v___x_1729_; lean_object* v___x_1730_; 
v___x_1728_ = lean_unsigned_to_nat(3u);
v___x_1729_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__5_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28_);
v___x_1730_ = l_Lean_Name_num___override(v___x_1729_, v___x_1728_);
return v___x_1730_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28_(){
_start:
{
lean_object* v___x_1732_; lean_object* v___x_1733_; lean_object* v___x_1734_; lean_object* v___x_1735_; lean_object* v___x_1736_; 
v___x_1732_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28_));
v___x_1733_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28_));
v___x_1734_ = lean_box(0);
v___x_1735_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28_, &lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28__once, _init_lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28_);
v___x_1736_ = l_Lean_Meta_Simp_registerSimprocAttr(v___x_1732_, v___x_1733_, v___x_1734_, v___x_1735_);
return v___x_1736_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28____boxed(lean_object* v_a_1737_){
_start:
{
lean_object* v_res_1738_; 
v_res_1738_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28_();
return v_res_1738_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_mon__tauto__proc___closed__2(void){
_start:
{
lean_object* v___x_1746_; lean_object* v___x_1747_; lean_object* v___x_1748_; lean_object* v___x_1749_; 
v___x_1746_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_1747_ = ((lean_object*)(lp_mathlib_Parser_Attr_mon__tauto__proc___closed__1));
v___x_1748_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1749_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1749_, 0, v___x_1748_);
lean_ctor_set(v___x_1749_, 1, v___x_1747_);
lean_ctor_set(v___x_1749_, 2, v___x_1746_);
return v___x_1749_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_mon__tauto__proc___closed__3(void){
_start:
{
lean_object* v___x_1750_; lean_object* v___x_1751_; lean_object* v___x_1752_; lean_object* v___x_1753_; 
v___x_1750_ = lean_obj_once(&lp_mathlib_Parser_Attr_mon__tauto__proc___closed__2, &lp_mathlib_Parser_Attr_mon__tauto__proc___closed__2_once, _init_lp_mathlib_Parser_Attr_mon__tauto__proc___closed__2);
v___x_1751_ = lean_unsigned_to_nat(1022u);
v___x_1752_ = ((lean_object*)(lp_mathlib_Parser_Attr_mon__tauto__proc___closed__0));
v___x_1753_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1753_, 0, v___x_1752_);
lean_ctor_set(v___x_1753_, 1, v___x_1751_);
lean_ctor_set(v___x_1753_, 2, v___x_1750_);
return v___x_1753_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_mon__tauto__proc(void){
_start:
{
lean_object* v___x_1754_; 
v___x_1754_ = lean_obj_once(&lp_mathlib_Parser_Attr_mon__tauto__proc___closed__3, &lp_mathlib_Parser_Attr_mon__tauto__proc___closed__3_once, _init_lp_mathlib_Parser_Attr_mon__tauto__proc___closed__3);
return v___x_1754_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_1772_; lean_object* v___x_1773_; lean_object* v___x_1774_; lean_object* v___x_1775_; 
v___x_1772_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5_));
v___x_1773_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5_));
v___x_1774_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5_));
v___x_1775_ = l_Lean_Meta_registerSimpAttr(v___x_1772_, v___x_1773_, v___x_1774_);
return v___x_1775_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5____boxed(lean_object* v_a_1776_){
_start:
{
lean_object* v_res_1777_; 
v_res_1777_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5_();
return v_res_1777_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_coassoc__simps___closed__2(void){
_start:
{
lean_object* v___x_1785_; lean_object* v___x_1786_; lean_object* v___x_1787_; lean_object* v___x_1788_; 
v___x_1785_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_1786_ = ((lean_object*)(lp_mathlib_Parser_Attr_coassoc__simps___closed__1));
v___x_1787_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1788_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1788_, 0, v___x_1787_);
lean_ctor_set(v___x_1788_, 1, v___x_1786_);
lean_ctor_set(v___x_1788_, 2, v___x_1785_);
return v___x_1788_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_coassoc__simps___closed__3(void){
_start:
{
lean_object* v___x_1789_; lean_object* v___x_1790_; lean_object* v___x_1791_; lean_object* v___x_1792_; 
v___x_1789_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__15));
v___x_1790_ = lean_obj_once(&lp_mathlib_Parser_Attr_coassoc__simps___closed__2, &lp_mathlib_Parser_Attr_coassoc__simps___closed__2_once, _init_lp_mathlib_Parser_Attr_coassoc__simps___closed__2);
v___x_1791_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1792_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1792_, 0, v___x_1791_);
lean_ctor_set(v___x_1792_, 1, v___x_1790_);
lean_ctor_set(v___x_1792_, 2, v___x_1789_);
return v___x_1792_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_coassoc__simps___closed__4(void){
_start:
{
lean_object* v___x_1793_; lean_object* v___x_1794_; lean_object* v___x_1795_; lean_object* v___x_1796_; 
v___x_1793_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__20));
v___x_1794_ = lean_obj_once(&lp_mathlib_Parser_Attr_coassoc__simps___closed__3, &lp_mathlib_Parser_Attr_coassoc__simps___closed__3_once, _init_lp_mathlib_Parser_Attr_coassoc__simps___closed__3);
v___x_1795_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1796_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1796_, 0, v___x_1795_);
lean_ctor_set(v___x_1796_, 1, v___x_1794_);
lean_ctor_set(v___x_1796_, 2, v___x_1793_);
return v___x_1796_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_coassoc__simps___closed__5(void){
_start:
{
lean_object* v___x_1797_; lean_object* v___x_1798_; lean_object* v___x_1799_; lean_object* v___x_1800_; 
v___x_1797_ = lean_obj_once(&lp_mathlib_Parser_Attr_coassoc__simps___closed__4, &lp_mathlib_Parser_Attr_coassoc__simps___closed__4_once, _init_lp_mathlib_Parser_Attr_coassoc__simps___closed__4);
v___x_1798_ = lean_unsigned_to_nat(1022u);
v___x_1799_ = ((lean_object*)(lp_mathlib_Parser_Attr_coassoc__simps___closed__0));
v___x_1800_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1800_, 0, v___x_1799_);
lean_ctor_set(v___x_1800_, 1, v___x_1798_);
lean_ctor_set(v___x_1800_, 2, v___x_1797_);
return v___x_1800_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_coassoc__simps(void){
_start:
{
lean_object* v___x_1801_; 
v___x_1801_ = lean_obj_once(&lp_mathlib_Parser_Attr_coassoc__simps___closed__5, &lp_mathlib_Parser_Attr_coassoc__simps___closed__5_once, _init_lp_mathlib_Parser_Attr_coassoc__simps___closed__5);
return v___x_1801_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28_(){
_start:
{
lean_object* v___x_1819_; lean_object* v___x_1820_; lean_object* v___x_1821_; lean_object* v___x_1822_; lean_object* v___x_1823_; 
v___x_1819_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__1_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28_));
v___x_1820_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__2_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28_));
v___x_1821_ = lean_box(0);
v___x_1822_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn___closed__6_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28_));
v___x_1823_ = l_Lean_Meta_Simp_registerSimprocAttr(v___x_1819_, v___x_1820_, v___x_1821_, v___x_1822_);
return v___x_1823_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28____boxed(lean_object* v_a_1824_){
_start:
{
lean_object* v_res_1825_; 
v_res_1825_ = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28_();
return v_res_1825_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_coassoc__simps__proc___closed__2(void){
_start:
{
lean_object* v___x_1833_; lean_object* v___x_1834_; lean_object* v___x_1835_; lean_object* v___x_1836_; 
v___x_1833_ = lean_obj_once(&lp_mathlib_Parser_Attr_functor__norm___closed__10, &lp_mathlib_Parser_Attr_functor__norm___closed__10_once, _init_lp_mathlib_Parser_Attr_functor__norm___closed__10);
v___x_1834_ = ((lean_object*)(lp_mathlib_Parser_Attr_coassoc__simps__proc___closed__1));
v___x_1835_ = ((lean_object*)(lp_mathlib_Parser_Attr_functor__norm___closed__3));
v___x_1836_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1836_, 0, v___x_1835_);
lean_ctor_set(v___x_1836_, 1, v___x_1834_);
lean_ctor_set(v___x_1836_, 2, v___x_1833_);
return v___x_1836_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_coassoc__simps__proc___closed__3(void){
_start:
{
lean_object* v___x_1837_; lean_object* v___x_1838_; lean_object* v___x_1839_; lean_object* v___x_1840_; 
v___x_1837_ = lean_obj_once(&lp_mathlib_Parser_Attr_coassoc__simps__proc___closed__2, &lp_mathlib_Parser_Attr_coassoc__simps__proc___closed__2_once, _init_lp_mathlib_Parser_Attr_coassoc__simps__proc___closed__2);
v___x_1838_ = lean_unsigned_to_nat(1022u);
v___x_1839_ = ((lean_object*)(lp_mathlib_Parser_Attr_coassoc__simps__proc___closed__0));
v___x_1840_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1840_, 0, v___x_1839_);
lean_ctor_set(v___x_1840_, 1, v___x_1838_);
lean_ctor_set(v___x_1840_, 2, v___x_1837_);
return v___x_1840_;
}
}
static lean_object* _init_lp_mathlib_Parser_Attr_coassoc__simps__proc(void){
_start:
{
lean_object* v___x_1841_; 
v___x_1841_ = lean_obj_once(&lp_mathlib_Parser_Attr_coassoc__simps__proc___closed__3, &lp_mathlib_Parser_Attr_coassoc__simps__proc___closed__3_once, _init_lp_mathlib_Parser_Attr_coassoc__simps__proc___closed__3);
return v___x_1841_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_LabelAttribute(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Simp(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Register(uint8_t builtin) {
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
res = runtime_initialize_Lean_LabelAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_LabelAttribute(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Attr_Register(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_LabelAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_functor__norm = _init_lp_mathlib_Parser_Attr_functor__norm();
lean_mark_persistent(lp_mathlib_Parser_Attr_functor__norm);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_28_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_2336186397____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_functor__norm__proc = _init_lp_mathlib_Parser_Attr_functor__norm__proc();
lean_mark_persistent(lp_mathlib_Parser_Attr_functor__norm__proc);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_monad__norm = _init_lp_mathlib_Parser_Attr_monad__norm();
lean_mark_persistent(lp_mathlib_Parser_Attr_monad__norm);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_28_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_3215534580____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_monad__norm__proc = _init_lp_mathlib_Parser_Attr_monad__norm__proc();
lean_mark_persistent(lp_mathlib_Parser_Attr_monad__norm__proc);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_parity__simps = _init_lp_mathlib_Parser_Attr_parity__simps();
lean_mark_persistent(lp_mathlib_Parser_Attr_parity__simps);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_28_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_1568842782____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_parity__simps__proc = _init_lp_mathlib_Parser_Attr_parity__simps__proc();
lean_mark_persistent(lp_mathlib_Parser_Attr_parity__simps__proc);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_rclike__simps = _init_lp_mathlib_Parser_Attr_rclike__simps();
lean_mark_persistent(lp_mathlib_Parser_Attr_rclike__simps);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_28_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_2899626838____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_rclike__simps__proc = _init_lp_mathlib_Parser_Attr_rclike__simps__proc();
lean_mark_persistent(lp_mathlib_Parser_Attr_rclike__simps__proc);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_rify__simps = _init_lp_mathlib_Parser_Attr_rify__simps();
lean_mark_persistent(lp_mathlib_Parser_Attr_rify__simps);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_28_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_612238087____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_rify__simps__proc = _init_lp_mathlib_Parser_Attr_rify__simps__proc();
lean_mark_persistent(lp_mathlib_Parser_Attr_rify__simps__proc);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_qify__simps = _init_lp_mathlib_Parser_Attr_qify__simps();
lean_mark_persistent(lp_mathlib_Parser_Attr_qify__simps);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_28_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_1503578496____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_qify__simps__proc = _init_lp_mathlib_Parser_Attr_qify__simps__proc();
lean_mark_persistent(lp_mathlib_Parser_Attr_qify__simps__proc);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_zify__simps = _init_lp_mathlib_Parser_Attr_zify__simps();
lean_mark_persistent(lp_mathlib_Parser_Attr_zify__simps);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_28_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_2552594532____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_zify__simps__proc = _init_lp_mathlib_Parser_Attr_zify__simps__proc();
lean_mark_persistent(lp_mathlib_Parser_Attr_zify__simps__proc);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_pull__end = _init_lp_mathlib_Parser_Attr_pull__end();
lean_mark_persistent(lp_mathlib_Parser_Attr_pull__end);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_28_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_1133190711____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_pull__end__proc = _init_lp_mathlib_Parser_Attr_pull__end__proc();
lean_mark_persistent(lp_mathlib_Parser_Attr_pull__end__proc);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_push__end = _init_lp_mathlib_Parser_Attr_push__end();
lean_mark_persistent(lp_mathlib_Parser_Attr_push__end);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_28_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_1988518680____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_push__end__proc = _init_lp_mathlib_Parser_Attr_push__end__proc();
lean_mark_persistent(lp_mathlib_Parser_Attr_push__end__proc);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_mfld__simps = _init_lp_mathlib_Parser_Attr_mfld__simps();
lean_mark_persistent(lp_mathlib_Parser_Attr_mfld__simps);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_28_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_479538640____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_mfld__simps__proc = _init_lp_mathlib_Parser_Attr_mfld__simps__proc();
lean_mark_persistent(lp_mathlib_Parser_Attr_mfld__simps__proc);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_integral__simps = _init_lp_mathlib_Parser_Attr_integral__simps();
lean_mark_persistent(lp_mathlib_Parser_Attr_integral__simps);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_28_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_3849875863____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_integral__simps__proc = _init_lp_mathlib_Parser_Attr_integral__simps__proc();
lean_mark_persistent(lp_mathlib_Parser_Attr_integral__simps__proc);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_typevec = _init_lp_mathlib_Parser_Attr_typevec();
lean_mark_persistent(lp_mathlib_Parser_Attr_typevec);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_28_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_1131139975____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_typevec__proc = _init_lp_mathlib_Parser_Attr_typevec__proc();
lean_mark_persistent(lp_mathlib_Parser_Attr_typevec__proc);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_ghost__simps = _init_lp_mathlib_Parser_Attr_ghost__simps();
lean_mark_persistent(lp_mathlib_Parser_Attr_ghost__simps);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_28_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_1086225606____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_ghost__simps__proc = _init_lp_mathlib_Parser_Attr_ghost__simps__proc();
lean_mark_persistent(lp_mathlib_Parser_Attr_ghost__simps__proc);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_nontriviality = _init_lp_mathlib_Parser_Attr_nontriviality();
lean_mark_persistent(lp_mathlib_Parser_Attr_nontriviality);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_28_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_2133673922____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_nontriviality__proc = _init_lp_mathlib_Parser_Attr_nontriviality__proc();
lean_mark_persistent(lp_mathlib_Parser_Attr_nontriviality__proc);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2811593030____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_2811593030____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_2811593030____hygCtx___hyg_3_);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_fin__omega = _init_lp_mathlib_Parser_Attr_fin__omega();
lean_mark_persistent(lp_mathlib_Parser_Attr_fin__omega);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_28_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_3021440532____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_fin__omega__proc = _init_lp_mathlib_Parser_Attr_fin__omega__proc();
lean_mark_persistent(lp_mathlib_Parser_Attr_fin__omega__proc);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_enat__to__nat__top = _init_lp_mathlib_Parser_Attr_enat__to__nat__top();
lean_mark_persistent(lp_mathlib_Parser_Attr_enat__to__nat__top);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_28_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_2016062075____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_enat__to__nat__top__proc = _init_lp_mathlib_Parser_Attr_enat__to__nat__top__proc();
lean_mark_persistent(lp_mathlib_Parser_Attr_enat__to__nat__top__proc);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_enat__to__nat__coe = _init_lp_mathlib_Parser_Attr_enat__to__nat__coe();
lean_mark_persistent(lp_mathlib_Parser_Attr_enat__to__nat__coe);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_28_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_668879677____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_enat__to__nat__coe__proc = _init_lp_mathlib_Parser_Attr_enat__to__nat__coe__proc();
lean_mark_persistent(lp_mathlib_Parser_Attr_enat__to__nat__coe__proc);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_pnat__to__nat__coe = _init_lp_mathlib_Parser_Attr_pnat__to__nat__coe();
lean_mark_persistent(lp_mathlib_Parser_Attr_pnat__to__nat__coe);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_28_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_3369699945____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc = _init_lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc();
lean_mark_persistent(lp_mathlib_Parser_Attr_pnat__to__nat__coe__proc);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_mon__tauto = _init_lp_mathlib_Parser_Attr_mon__tauto();
lean_mark_persistent(lp_mathlib_Parser_Attr_mon__tauto);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_28_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_3059584346____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_mon__tauto__proc = _init_lp_mathlib_Parser_Attr_mon__tauto__proc();
lean_mark_persistent(lp_mathlib_Parser_Attr_mon__tauto__proc);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_ext_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_coassoc__simps = _init_lp_mathlib_Parser_Attr_coassoc__simps();
lean_mark_persistent(lp_mathlib_Parser_Attr_coassoc__simps);
res = lp_mathlib___private_Mathlib_Tactic_Attr_Register_0__initFn_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_28_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_3_ = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_extProc_00___x40_Mathlib_Tactic_Attr_Register_85691930____hygCtx___hyg_3_);
lean_dec_ref(res);
lp_mathlib_Parser_Attr_coassoc__simps__proc = _init_lp_mathlib_Parser_Attr_coassoc__simps__proc();
lean_mark_persistent(lp_mathlib_Parser_Attr_coassoc__simps__proc);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_LabelAttribute(uint8_t builtin);
lean_object* initialize_Lean_LabelAttribute(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Simp(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Register(uint8_t builtin) {
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
res = initialize_Lean_LabelAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_LabelAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Register(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Attr_Register(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Attr_Register(builtin);
}
#ifdef __cplusplus
}
#endif
