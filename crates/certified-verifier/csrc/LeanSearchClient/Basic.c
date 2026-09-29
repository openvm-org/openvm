// Lean compiler output
// Module: LeanSearchClient.Basic
// Imports: public import Init public meta import Init public meta import Lean.Data.Options
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
lean_object* lean_register_option(lean_object*, lean_object*);
extern lean_object* l_Lean_versionString;
lean_object* lean_string_append(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Option_register___at___00__private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Option_register___at___00__private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "leansearch"};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__value;
static const lean_string_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "queries"};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__value;
static const lean_ctor_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(198, 111, 195, 98, 28, 54, 228, 64)}};
static const lean_ctor_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(210, 51, 106, 231, 247, 208, 171, 234)}};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__value;
static const lean_string_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 56, .m_capacity = 56, .m_length = 55, .m_data = "Number of results requested from leansearch (default 6)"};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__value;
static const lean_ctor_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__4_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(6) << 1) | 1)),((lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__4_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__4_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_leansearch_queries;
static const lean_string_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_3678057566____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "loogle"};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_3678057566____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_3678057566____hygCtx___hyg_4__value;
static const lean_ctor_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_3678057566____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_3678057566____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(1, 237, 9, 99, 66, 242, 124, 118)}};
static const lean_ctor_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_3678057566____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_3678057566____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(81, 202, 153, 61, 23, 63, 143, 145)}};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_3678057566____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_3678057566____hygCtx___hyg_4__value;
static const lean_string_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_3678057566____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 52, .m_capacity = 52, .m_length = 51, .m_data = "Number of results requested from loogle (default 6)"};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_3678057566____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_3678057566____hygCtx___hyg_4__value;
static const lean_ctor_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_3678057566____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(6) << 1) | 1)),((lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_3678057566____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_3678057566____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_3678057566____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_3678057566____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_3678057566____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_loogle_queries;
static const lean_string_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_967923280____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "statesearch"};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_967923280____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_967923280____hygCtx___hyg_4__value;
static const lean_ctor_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_967923280____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_967923280____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(103, 233, 168, 19, 111, 44, 235, 225)}};
static const lean_ctor_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_967923280____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_967923280____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(135, 112, 157, 167, 251, 185, 144, 202)}};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_967923280____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_967923280____hygCtx___hyg_4__value;
static const lean_string_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_967923280____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 57, .m_capacity = 57, .m_length = 56, .m_data = "Number of results requested from statesearch (default 6)"};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_967923280____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_967923280____hygCtx___hyg_4__value;
static const lean_ctor_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_967923280____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(6) << 1) | 1)),((lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_967923280____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_967923280____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_967923280____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_967923280____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_967923280____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_statesearch_queries;
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Option_register___at___00__private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Option_register___at___00__private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "revision"};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4__value;
static const lean_ctor_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_967923280____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(103, 233, 168, 19, 111, 44, 235, 225)}};
static const lean_ctor_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(6, 44, 47, 159, 227, 54, 240, 197)}};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4__value;
static const lean_string_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "v"};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4__value;
static lean_once_cell_t lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4_;
static const lean_string_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__4_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "Revision of LeanStateSearch to use"};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__4_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__4_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4__value;
static lean_once_cell_t lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__5_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__5_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4_;
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_statesearch_revision;
static const lean_string_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "leansearchclient"};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4__value;
static const lean_string_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "useragent"};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4__value;
static const lean_ctor_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(220, 134, 209, 202, 111, 70, 155, 103)}};
static const lean_ctor_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(198, 207, 175, 231, 191, 32, 173, 3)}};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4__value;
static const lean_string_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "LeanSearchClient"};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4__value;
static const lean_string_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__4_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "Username for leansearchclient"};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__4_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__4_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4__value;
static const lean_ctor_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__5_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4__value),((lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__4_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__5_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__5_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_leansearchclient_useragent;
static const lean_string_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_3856181077____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "backend"};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_3856181077____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_3856181077____hygCtx___hyg_4__value;
static const lean_ctor_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_3856181077____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(220, 134, 209, 202, 111, 70, 155, 103)}};
static const lean_ctor_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_3856181077____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_3856181077____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_3856181077____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(27, 198, 96, 209, 246, 71, 232, 207)}};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_3856181077____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_3856181077____hygCtx___hyg_4__value;
static const lean_string_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_3856181077____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 57, .m_capacity = 57, .m_length = 56, .m_data = "The backend to use by default, currently only leansearch"};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_3856181077____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_3856181077____hygCtx___hyg_4__value;
static const lean_ctor_object lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_3856181077____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__0_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__value),((lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_3856181077____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_3856181077____hygCtx___hyg_4_ = (const lean_object*)&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_3856181077____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_3856181077____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_3856181077____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_LeanSearchClient_leansearchclient_backend;
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Option_register___at___00__private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__spec__0(lean_object* v_name_1_, lean_object* v_decl_2_, lean_object* v_ref_3_){
_start:
{
lean_object* v_defValue_5_; lean_object* v_descr_6_; lean_object* v_deprecation_x3f_7_; lean_object* v___x_8_; lean_object* v___x_9_; lean_object* v___x_10_; 
v_defValue_5_ = lean_ctor_get(v_decl_2_, 0);
v_descr_6_ = lean_ctor_get(v_decl_2_, 1);
v_deprecation_x3f_7_ = lean_ctor_get(v_decl_2_, 2);
lean_inc(v_defValue_5_);
v___x_8_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_8_, 0, v_defValue_5_);
lean_inc(v_deprecation_x3f_7_);
lean_inc_ref(v_descr_6_);
lean_inc_n(v_name_1_, 2);
v___x_9_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_9_, 0, v_name_1_);
lean_ctor_set(v___x_9_, 1, v_ref_3_);
lean_ctor_set(v___x_9_, 2, v___x_8_);
lean_ctor_set(v___x_9_, 3, v_descr_6_);
lean_ctor_set(v___x_9_, 4, v_deprecation_x3f_7_);
v___x_10_ = lean_register_option(v_name_1_, v___x_9_);
if (lean_obj_tag(v___x_10_) == 0)
{
lean_object* v___x_12_; uint8_t v_isShared_13_; uint8_t v_isSharedCheck_18_; 
v_isSharedCheck_18_ = !lean_is_exclusive(v___x_10_);
if (v_isSharedCheck_18_ == 0)
{
lean_object* v_unused_19_; 
v_unused_19_ = lean_ctor_get(v___x_10_, 0);
lean_dec(v_unused_19_);
v___x_12_ = v___x_10_;
v_isShared_13_ = v_isSharedCheck_18_;
goto v_resetjp_11_;
}
else
{
lean_dec(v___x_10_);
v___x_12_ = lean_box(0);
v_isShared_13_ = v_isSharedCheck_18_;
goto v_resetjp_11_;
}
v_resetjp_11_:
{
lean_object* v___x_14_; lean_object* v___x_16_; 
lean_inc(v_defValue_5_);
v___x_14_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_14_, 0, v_name_1_);
lean_ctor_set(v___x_14_, 1, v_defValue_5_);
if (v_isShared_13_ == 0)
{
lean_ctor_set(v___x_12_, 0, v___x_14_);
v___x_16_ = v___x_12_;
goto v_reusejp_15_;
}
else
{
lean_object* v_reuseFailAlloc_17_; 
v_reuseFailAlloc_17_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_17_, 0, v___x_14_);
v___x_16_ = v_reuseFailAlloc_17_;
goto v_reusejp_15_;
}
v_reusejp_15_:
{
return v___x_16_;
}
}
}
else
{
lean_object* v_a_20_; lean_object* v___x_22_; uint8_t v_isShared_23_; uint8_t v_isSharedCheck_27_; 
lean_dec(v_name_1_);
v_a_20_ = lean_ctor_get(v___x_10_, 0);
v_isSharedCheck_27_ = !lean_is_exclusive(v___x_10_);
if (v_isSharedCheck_27_ == 0)
{
v___x_22_ = v___x_10_;
v_isShared_23_ = v_isSharedCheck_27_;
goto v_resetjp_21_;
}
else
{
lean_inc(v_a_20_);
lean_dec(v___x_10_);
v___x_22_ = lean_box(0);
v_isShared_23_ = v_isSharedCheck_27_;
goto v_resetjp_21_;
}
v_resetjp_21_:
{
lean_object* v___x_25_; 
if (v_isShared_23_ == 0)
{
v___x_25_ = v___x_22_;
goto v_reusejp_24_;
}
else
{
lean_object* v_reuseFailAlloc_26_; 
v_reuseFailAlloc_26_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_26_, 0, v_a_20_);
v___x_25_ = v_reuseFailAlloc_26_;
goto v_reusejp_24_;
}
v_reusejp_24_:
{
return v___x_25_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Option_register___at___00__private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_28_, lean_object* v_decl_29_, lean_object* v_ref_30_, lean_object* v_a_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_LeanSearchClient_Lean_Option_register___at___00__private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__spec__0(v_name_28_, v_decl_29_, v_ref_30_);
lean_dec_ref(v_decl_29_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; 
v___x_44_ = ((lean_object*)(lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4_));
v___x_45_ = ((lean_object*)(lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__4_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4_));
v___x_46_ = lp_LeanSearchClient_Lean_Option_register___at___00__private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__spec__0(v___x_44_, v___x_45_, v___x_44_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4____boxed(lean_object* v_a_47_){
_start:
{
lean_object* v_res_48_; 
v_res_48_ = lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4_();
return v_res_48_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_3678057566____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; 
v___x_59_ = ((lean_object*)(lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_3678057566____hygCtx___hyg_4_));
v___x_60_ = ((lean_object*)(lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_3678057566____hygCtx___hyg_4_));
v___x_61_ = lp_LeanSearchClient_Lean_Option_register___at___00__private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__spec__0(v___x_59_, v___x_60_, v___x_59_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_3678057566____hygCtx___hyg_4____boxed(lean_object* v_a_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_3678057566____hygCtx___hyg_4_();
return v_res_63_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_967923280____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; 
v___x_74_ = ((lean_object*)(lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_967923280____hygCtx___hyg_4_));
v___x_75_ = ((lean_object*)(lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_967923280____hygCtx___hyg_4_));
v___x_76_ = lp_LeanSearchClient_Lean_Option_register___at___00__private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4__spec__0(v___x_74_, v___x_75_, v___x_74_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_967923280____hygCtx___hyg_4____boxed(lean_object* v_a_77_){
_start:
{
lean_object* v_res_78_; 
v_res_78_ = lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_967923280____hygCtx___hyg_4_();
return v_res_78_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Option_register___at___00__private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4__spec__0(lean_object* v_name_79_, lean_object* v_decl_80_, lean_object* v_ref_81_){
_start:
{
lean_object* v_defValue_83_; lean_object* v_descr_84_; lean_object* v_deprecation_x3f_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; 
v_defValue_83_ = lean_ctor_get(v_decl_80_, 0);
v_descr_84_ = lean_ctor_get(v_decl_80_, 1);
v_deprecation_x3f_85_ = lean_ctor_get(v_decl_80_, 2);
lean_inc(v_defValue_83_);
v___x_86_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_86_, 0, v_defValue_83_);
lean_inc(v_deprecation_x3f_85_);
lean_inc_ref(v_descr_84_);
lean_inc_n(v_name_79_, 2);
v___x_87_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_87_, 0, v_name_79_);
lean_ctor_set(v___x_87_, 1, v_ref_81_);
lean_ctor_set(v___x_87_, 2, v___x_86_);
lean_ctor_set(v___x_87_, 3, v_descr_84_);
lean_ctor_set(v___x_87_, 4, v_deprecation_x3f_85_);
v___x_88_ = lean_register_option(v_name_79_, v___x_87_);
if (lean_obj_tag(v___x_88_) == 0)
{
lean_object* v___x_90_; uint8_t v_isShared_91_; uint8_t v_isSharedCheck_96_; 
v_isSharedCheck_96_ = !lean_is_exclusive(v___x_88_);
if (v_isSharedCheck_96_ == 0)
{
lean_object* v_unused_97_; 
v_unused_97_ = lean_ctor_get(v___x_88_, 0);
lean_dec(v_unused_97_);
v___x_90_ = v___x_88_;
v_isShared_91_ = v_isSharedCheck_96_;
goto v_resetjp_89_;
}
else
{
lean_dec(v___x_88_);
v___x_90_ = lean_box(0);
v_isShared_91_ = v_isSharedCheck_96_;
goto v_resetjp_89_;
}
v_resetjp_89_:
{
lean_object* v___x_92_; lean_object* v___x_94_; 
lean_inc(v_defValue_83_);
v___x_92_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_92_, 0, v_name_79_);
lean_ctor_set(v___x_92_, 1, v_defValue_83_);
if (v_isShared_91_ == 0)
{
lean_ctor_set(v___x_90_, 0, v___x_92_);
v___x_94_ = v___x_90_;
goto v_reusejp_93_;
}
else
{
lean_object* v_reuseFailAlloc_95_; 
v_reuseFailAlloc_95_ = lean_alloc_ctor(0, 1, 0);
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
else
{
lean_object* v_a_98_; lean_object* v___x_100_; uint8_t v_isShared_101_; uint8_t v_isSharedCheck_105_; 
lean_dec(v_name_79_);
v_a_98_ = lean_ctor_get(v___x_88_, 0);
v_isSharedCheck_105_ = !lean_is_exclusive(v___x_88_);
if (v_isSharedCheck_105_ == 0)
{
v___x_100_ = v___x_88_;
v_isShared_101_ = v_isSharedCheck_105_;
goto v_resetjp_99_;
}
else
{
lean_inc(v_a_98_);
lean_dec(v___x_88_);
v___x_100_ = lean_box(0);
v_isShared_101_ = v_isSharedCheck_105_;
goto v_resetjp_99_;
}
v_resetjp_99_:
{
lean_object* v___x_103_; 
if (v_isShared_101_ == 0)
{
v___x_103_ = v___x_100_;
goto v_reusejp_102_;
}
else
{
lean_object* v_reuseFailAlloc_104_; 
v_reuseFailAlloc_104_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_104_, 0, v_a_98_);
v___x_103_ = v_reuseFailAlloc_104_;
goto v_reusejp_102_;
}
v_reusejp_102_:
{
return v___x_103_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient_Lean_Option_register___at___00__private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_106_, lean_object* v_decl_107_, lean_object* v_ref_108_, lean_object* v_a_109_){
_start:
{
lean_object* v_res_110_; 
v_res_110_ = lp_LeanSearchClient_Lean_Option_register___at___00__private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4__spec__0(v_name_106_, v_decl_107_, v_ref_108_);
lean_dec_ref(v_decl_107_);
return v_res_110_;
}
}
static lean_object* _init_lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4_(void){
_start:
{
lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; 
v___x_116_ = l_Lean_versionString;
v___x_117_ = ((lean_object*)(lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4_));
v___x_118_ = lean_string_append(v___x_117_, v___x_116_);
return v___x_118_;
}
}
static lean_object* _init_lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__5_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4_(void){
_start:
{
lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; 
v___x_120_ = lean_box(0);
v___x_121_ = ((lean_object*)(lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__4_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4_));
v___x_122_ = lean_obj_once(&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4_, &lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4__once, _init_lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4_);
v___x_123_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_123_, 0, v___x_122_);
lean_ctor_set(v___x_123_, 1, v___x_121_);
lean_ctor_set(v___x_123_, 2, v___x_120_);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; 
v___x_125_ = ((lean_object*)(lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4_));
v___x_126_ = lean_obj_once(&lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__5_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4_, &lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__5_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4__once, _init_lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__5_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4_);
v___x_127_ = lp_LeanSearchClient_Lean_Option_register___at___00__private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4__spec__0(v___x_125_, v___x_126_, v___x_125_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4____boxed(lean_object* v_a_128_){
_start:
{
lean_object* v_res_129_; 
v_res_129_ = lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4_();
return v_res_129_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; 
v___x_142_ = ((lean_object*)(lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__2_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4_));
v___x_143_ = ((lean_object*)(lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__5_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4_));
v___x_144_ = lp_LeanSearchClient_Lean_Option_register___at___00__private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4__spec__0(v___x_142_, v___x_143_, v___x_142_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4____boxed(lean_object* v_a_145_){
_start:
{
lean_object* v_res_146_; 
v_res_146_ = lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4_();
return v_res_146_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_3856181077____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; 
v___x_157_ = ((lean_object*)(lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__1_00___x40_LeanSearchClient_Basic_3856181077____hygCtx___hyg_4_));
v___x_158_ = ((lean_object*)(lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn___closed__3_00___x40_LeanSearchClient_Basic_3856181077____hygCtx___hyg_4_));
v___x_159_ = lp_LeanSearchClient_Lean_Option_register___at___00__private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4__spec__0(v___x_157_, v___x_158_, v___x_157_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_3856181077____hygCtx___hyg_4____boxed(lean_object* v_a_160_){
_start:
{
lean_object* v_res_161_; 
v_res_161_ = lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_3856181077____hygCtx___hyg_4_();
return v_res_161_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_LeanSearchClient_LeanSearchClient_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Data_Options(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_LeanSearchClient_LeanSearchClient_Basic(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Data_Options(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_2173077481____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_LeanSearchClient_leansearch_queries = lean_io_result_get_value(res);
lean_mark_persistent(lp_LeanSearchClient_leansearch_queries);
lean_dec_ref(res);
res = lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_3678057566____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_LeanSearchClient_loogle_queries = lean_io_result_get_value(res);
lean_mark_persistent(lp_LeanSearchClient_loogle_queries);
lean_dec_ref(res);
res = lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_967923280____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_LeanSearchClient_statesearch_queries = lean_io_result_get_value(res);
lean_mark_persistent(lp_LeanSearchClient_statesearch_queries);
lean_dec_ref(res);
res = lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_1381885777____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_LeanSearchClient_statesearch_revision = lean_io_result_get_value(res);
lean_mark_persistent(lp_LeanSearchClient_statesearch_revision);
lean_dec_ref(res);
res = lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_611040542____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_LeanSearchClient_leansearchclient_useragent = lean_io_result_get_value(res);
lean_mark_persistent(lp_LeanSearchClient_leansearchclient_useragent);
lean_dec_ref(res);
res = lp_LeanSearchClient___private_LeanSearchClient_Basic_0__initFn_00___x40_LeanSearchClient_Basic_3856181077____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_LeanSearchClient_leansearchclient_backend = lean_io_result_get_value(res);
lean_mark_persistent(lp_LeanSearchClient_leansearchclient_backend);
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Data_Options(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_LeanSearchClient_LeanSearchClient_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Data_Options(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_LeanSearchClient_LeanSearchClient_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_LeanSearchClient_LeanSearchClient_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_LeanSearchClient_LeanSearchClient_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
