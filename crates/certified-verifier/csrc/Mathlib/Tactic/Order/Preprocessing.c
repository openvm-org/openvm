// Lean compiler output
// Module: Mathlib.Tactic.Order.Preprocessing
// Imports: public import Init public meta import Init public import Mathlib.Tactic.Order.CollectFacts public meta import Mathlib.Util.AtomM
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
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
size_t lean_array_size(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_lt(size_t, size_t);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_zipIdx___redArg(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppOptM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_synthInstance_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_lin_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_lin_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_lin_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_lin_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_part_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_part_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_part_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_part_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_pre_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_pre_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_pre_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_pre_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Order_instBEqOrderType_beq(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_instBEqOrderType_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Order_instBEqOrderType___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Order_instBEqOrderType_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_instBEqOrderType___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instBEqOrderType___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Order_instBEqOrderType = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instBEqOrderType___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_instToStringOrderType___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "linear order"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringOrderType___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instToStringOrderType___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_instToStringOrderType___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "partial order"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringOrderType___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instToStringOrderType___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_instToStringOrderType___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "preorder"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringOrderType___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instToStringOrderType___lam__0___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringOrderType___lam__0(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringOrderType___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Order_instToStringOrderType___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Order_instToStringOrderType___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringOrderType___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instToStringOrderType___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringOrderType = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instToStringOrderType___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "LinearOrder"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__0_value),LEAN_SCALAR_PTR_LITERAL(99, 217, 57, 95, 10, 230, 120, 46)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "PartialOrder"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__2_value),LEAN_SCALAR_PTR_LITERAL(47, 196, 146, 225, 179, 207, 152, 76)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Preorder"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__4_value),LEAN_SCALAR_PTR_LITERAL(171, 85, 2, 192, 23, 244, 204, 242)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "bot_le"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(173, 96, 93, 187, 54, 191, 79, 229)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__2;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__3;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__4;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "le_top"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__1___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__1___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(200, 118, 12, 168, 18, 79, 53, 157)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__1___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__1___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__1___redArg(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__2(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Tactic_Order_replaceBotTop___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_replaceBotTop___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_replaceBotTop___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_replaceBotTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_replaceBotTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__1(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "le_of_lt"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(26, 46, 54, 245, 81, 108, 136, 63)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "not_le_of_gt"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(148, 76, 46, 49, 146, 193, 88, 207)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__3_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "le_of_eq"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(205, 115, 143, 134, 128, 231, 21, 210)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__5 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__5_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "ge_of_eq"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__6 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__6_value),LEAN_SCALAR_PTR_LITERAL(9, 62, 101, 249, 53, 161, 236, 54)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__7 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_preprocessFactsPreorder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_preprocessFactsPreorder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "ne_of_lt"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(200, 220, 89, 20, 68, 185, 33, 223)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "ne_of_not_le"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(164, 235, 205, 139, 38, 252, 174, 220)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__3_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__4_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__5 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__5_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Order"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__6 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__6_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "not_lt_of_not_le"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__7 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__8_value_aux_1),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(86, 184, 242, 98, 125, 170, 160, 198)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__8_value_aux_2),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__7_value),LEAN_SCALAR_PTR_LITERAL(103, 60, 94, 126, 142, 167, 239, 232)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__8 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__8_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "le_sup_left"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__9 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__9_value),LEAN_SCALAR_PTR_LITERAL(238, 42, 110, 130, 88, 24, 244, 32)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__10 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__10_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "le_sup_right"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__11 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__11_value),LEAN_SCALAR_PTR_LITERAL(89, 33, 25, 71, 158, 228, 205, 91)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__12 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__12_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "inf_le_left"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__13 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(65, 207, 220, 239, 175, 189, 138, 196)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__14 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__14_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "inf_le_right"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__15 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(193, 214, 23, 245, 229, 236, 115, 72)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__16 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__16_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_preprocessFactsPartial(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_preprocessFactsPartial___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "le_of_not_ge"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(130, 157, 16, 146, 211, 70, 177, 90)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0___redArg___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "le_of_not_gt"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(20, 53, 39, 207, 43, 74, 14, 72)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0___redArg___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_preprocessFactsLinear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_preprocessFactsLinear___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_preprocessFacts(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_preprocessFacts___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_ctorIdx(uint8_t v_x_1_){
_start:
{
switch(v_x_1_)
{
case 0:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
case 1:
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
default: 
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(2u);
return v___x_4_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_ctorIdx___boxed(lean_object* v_x_5_){
_start:
{
uint8_t v_x_boxed_6_; lean_object* v_res_7_; 
v_x_boxed_6_ = lean_unbox(v_x_5_);
v_res_7_ = lp_mathlib_Mathlib_Tactic_Order_OrderType_ctorIdx(v_x_boxed_6_);
return v_res_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_ctorElim___redArg(lean_object* v_k_8_){
_start:
{
lean_inc(v_k_8_);
return v_k_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_ctorElim___redArg___boxed(lean_object* v_k_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_Mathlib_Tactic_Order_OrderType_ctorElim___redArg(v_k_9_);
lean_dec(v_k_9_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_ctorElim(lean_object* v_motive_11_, lean_object* v_ctorIdx_12_, uint8_t v_t_13_, lean_object* v_h_14_, lean_object* v_k_15_){
_start:
{
lean_inc(v_k_15_);
return v_k_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_ctorElim___boxed(lean_object* v_motive_16_, lean_object* v_ctorIdx_17_, lean_object* v_t_18_, lean_object* v_h_19_, lean_object* v_k_20_){
_start:
{
uint8_t v_t_boxed_21_; lean_object* v_res_22_; 
v_t_boxed_21_ = lean_unbox(v_t_18_);
v_res_22_ = lp_mathlib_Mathlib_Tactic_Order_OrderType_ctorElim(v_motive_16_, v_ctorIdx_17_, v_t_boxed_21_, v_h_19_, v_k_20_);
lean_dec(v_k_20_);
lean_dec(v_ctorIdx_17_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_lin_elim___redArg(lean_object* v_lin_23_){
_start:
{
lean_inc(v_lin_23_);
return v_lin_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_lin_elim___redArg___boxed(lean_object* v_lin_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_mathlib_Mathlib_Tactic_Order_OrderType_lin_elim___redArg(v_lin_24_);
lean_dec(v_lin_24_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_lin_elim(lean_object* v_motive_26_, uint8_t v_t_27_, lean_object* v_h_28_, lean_object* v_lin_29_){
_start:
{
lean_inc(v_lin_29_);
return v_lin_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_lin_elim___boxed(lean_object* v_motive_30_, lean_object* v_t_31_, lean_object* v_h_32_, lean_object* v_lin_33_){
_start:
{
uint8_t v_t_boxed_34_; lean_object* v_res_35_; 
v_t_boxed_34_ = lean_unbox(v_t_31_);
v_res_35_ = lp_mathlib_Mathlib_Tactic_Order_OrderType_lin_elim(v_motive_30_, v_t_boxed_34_, v_h_32_, v_lin_33_);
lean_dec(v_lin_33_);
return v_res_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_part_elim___redArg(lean_object* v_part_36_){
_start:
{
lean_inc(v_part_36_);
return v_part_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_part_elim___redArg___boxed(lean_object* v_part_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_Mathlib_Tactic_Order_OrderType_part_elim___redArg(v_part_37_);
lean_dec(v_part_37_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_part_elim(lean_object* v_motive_39_, uint8_t v_t_40_, lean_object* v_h_41_, lean_object* v_part_42_){
_start:
{
lean_inc(v_part_42_);
return v_part_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_part_elim___boxed(lean_object* v_motive_43_, lean_object* v_t_44_, lean_object* v_h_45_, lean_object* v_part_46_){
_start:
{
uint8_t v_t_boxed_47_; lean_object* v_res_48_; 
v_t_boxed_47_ = lean_unbox(v_t_44_);
v_res_48_ = lp_mathlib_Mathlib_Tactic_Order_OrderType_part_elim(v_motive_43_, v_t_boxed_47_, v_h_45_, v_part_46_);
lean_dec(v_part_46_);
return v_res_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_pre_elim___redArg(lean_object* v_pre_49_){
_start:
{
lean_inc(v_pre_49_);
return v_pre_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_pre_elim___redArg___boxed(lean_object* v_pre_50_){
_start:
{
lean_object* v_res_51_; 
v_res_51_ = lp_mathlib_Mathlib_Tactic_Order_OrderType_pre_elim___redArg(v_pre_50_);
lean_dec(v_pre_50_);
return v_res_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_pre_elim(lean_object* v_motive_52_, uint8_t v_t_53_, lean_object* v_h_54_, lean_object* v_pre_55_){
_start:
{
lean_inc(v_pre_55_);
return v_pre_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_OrderType_pre_elim___boxed(lean_object* v_motive_56_, lean_object* v_t_57_, lean_object* v_h_58_, lean_object* v_pre_59_){
_start:
{
uint8_t v_t_boxed_60_; lean_object* v_res_61_; 
v_t_boxed_60_ = lean_unbox(v_t_57_);
v_res_61_ = lp_mathlib_Mathlib_Tactic_Order_OrderType_pre_elim(v_motive_56_, v_t_boxed_60_, v_h_58_, v_pre_59_);
lean_dec(v_pre_59_);
return v_res_61_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_Order_instBEqOrderType_beq(uint8_t v_x_62_, uint8_t v_y_63_){
_start:
{
lean_object* v___x_64_; lean_object* v___x_65_; uint8_t v___x_66_; 
v___x_64_ = lp_mathlib_Mathlib_Tactic_Order_OrderType_ctorIdx(v_x_62_);
v___x_65_ = lp_mathlib_Mathlib_Tactic_Order_OrderType_ctorIdx(v_y_63_);
v___x_66_ = lean_nat_dec_eq(v___x_64_, v___x_65_);
lean_dec(v___x_65_);
lean_dec(v___x_64_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_instBEqOrderType_beq___boxed(lean_object* v_x_67_, lean_object* v_y_68_){
_start:
{
uint8_t v_x_17__boxed_69_; uint8_t v_y_18__boxed_70_; uint8_t v_res_71_; lean_object* v_r_72_; 
v_x_17__boxed_69_ = lean_unbox(v_x_67_);
v_y_18__boxed_70_ = lean_unbox(v_y_68_);
v_res_71_ = lp_mathlib_Mathlib_Tactic_Order_instBEqOrderType_beq(v_x_17__boxed_69_, v_y_18__boxed_70_);
v_r_72_ = lean_box(v_res_71_);
return v_r_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringOrderType___lam__0(uint8_t v_x_78_){
_start:
{
switch(v_x_78_)
{
case 0:
{
lean_object* v___x_79_; 
v___x_79_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringOrderType___lam__0___closed__0));
return v___x_79_;
}
case 1:
{
lean_object* v___x_80_; 
v___x_80_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringOrderType___lam__0___closed__1));
return v___x_80_;
}
default: 
{
lean_object* v___x_81_; 
v___x_81_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringOrderType___lam__0___closed__2));
return v___x_81_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringOrderType___lam__0___boxed(lean_object* v_x_82_){
_start:
{
uint8_t v_x_36__boxed_83_; lean_object* v_res_84_; 
v_x_36__boxed_83_ = lean_unbox(v_x_82_);
v_res_84_ = lp_mathlib_Mathlib_Tactic_Order_instToStringOrderType___lam__0(v_x_36__boxed_83_);
return v_res_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance(lean_object* v_type_105_, lean_object* v_a_106_, lean_object* v_a_107_, lean_object* v_a_108_, lean_object* v_a_109_){
_start:
{
lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_111_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__1));
v___x_112_ = lean_unsigned_to_nat(1u);
v___x_113_ = lean_mk_empty_array_with_capacity(v___x_112_);
v___x_114_ = lean_array_push(v___x_113_, v_type_105_);
lean_inc_ref(v___x_114_);
v___x_115_ = l_Lean_Meta_mkAppM(v___x_111_, v___x_114_, v_a_106_, v_a_107_, v_a_108_, v_a_109_);
if (lean_obj_tag(v___x_115_) == 0)
{
lean_object* v_a_116_; lean_object* v___x_117_; lean_object* v___x_118_; 
v_a_116_ = lean_ctor_get(v___x_115_, 0);
lean_inc(v_a_116_);
lean_dec_ref_known(v___x_115_, 1);
v___x_117_ = lean_box(0);
v___x_118_ = l_Lean_Meta_synthInstance_x3f(v_a_116_, v___x_117_, v_a_106_, v_a_107_, v_a_108_, v_a_109_);
if (lean_obj_tag(v___x_118_) == 0)
{
lean_object* v_a_119_; lean_object* v___x_121_; uint8_t v_isShared_122_; uint8_t v_isSharedCheck_188_; 
v_a_119_ = lean_ctor_get(v___x_118_, 0);
v_isSharedCheck_188_ = !lean_is_exclusive(v___x_118_);
if (v_isSharedCheck_188_ == 0)
{
v___x_121_ = v___x_118_;
v_isShared_122_ = v_isSharedCheck_188_;
goto v_resetjp_120_;
}
else
{
lean_inc(v_a_119_);
lean_dec(v___x_118_);
v___x_121_ = lean_box(0);
v_isShared_122_ = v_isSharedCheck_188_;
goto v_resetjp_120_;
}
v_resetjp_120_:
{
if (lean_obj_tag(v_a_119_) == 0)
{
lean_object* v___x_123_; lean_object* v___x_124_; 
lean_del_object(v___x_121_);
v___x_123_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__3));
lean_inc_ref(v___x_114_);
v___x_124_ = l_Lean_Meta_mkAppM(v___x_123_, v___x_114_, v_a_106_, v_a_107_, v_a_108_, v_a_109_);
if (lean_obj_tag(v___x_124_) == 0)
{
lean_object* v_a_125_; lean_object* v___x_126_; 
v_a_125_ = lean_ctor_get(v___x_124_, 0);
lean_inc(v_a_125_);
lean_dec_ref_known(v___x_124_, 1);
v___x_126_ = l_Lean_Meta_synthInstance_x3f(v_a_125_, v___x_117_, v_a_106_, v_a_107_, v_a_108_, v_a_109_);
if (lean_obj_tag(v___x_126_) == 0)
{
lean_object* v_a_127_; lean_object* v___x_129_; uint8_t v_isShared_130_; uint8_t v_isSharedCheck_167_; 
v_a_127_ = lean_ctor_get(v___x_126_, 0);
v_isSharedCheck_167_ = !lean_is_exclusive(v___x_126_);
if (v_isSharedCheck_167_ == 0)
{
v___x_129_ = v___x_126_;
v_isShared_130_ = v_isSharedCheck_167_;
goto v_resetjp_128_;
}
else
{
lean_inc(v_a_127_);
lean_dec(v___x_126_);
v___x_129_ = lean_box(0);
v_isShared_130_ = v_isSharedCheck_167_;
goto v_resetjp_128_;
}
v_resetjp_128_:
{
if (lean_obj_tag(v_a_127_) == 0)
{
lean_object* v___x_131_; lean_object* v___x_132_; 
lean_del_object(v___x_129_);
v___x_131_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__5));
v___x_132_ = l_Lean_Meta_mkAppM(v___x_131_, v___x_114_, v_a_106_, v_a_107_, v_a_108_, v_a_109_);
if (lean_obj_tag(v___x_132_) == 0)
{
lean_object* v_a_133_; lean_object* v___x_134_; 
v_a_133_ = lean_ctor_get(v___x_132_, 0);
lean_inc(v_a_133_);
lean_dec_ref_known(v___x_132_, 1);
v___x_134_ = l_Lean_Meta_synthInstance_x3f(v_a_133_, v___x_117_, v_a_106_, v_a_107_, v_a_108_, v_a_109_);
if (lean_obj_tag(v___x_134_) == 0)
{
lean_object* v_a_135_; lean_object* v___x_137_; uint8_t v_isShared_138_; uint8_t v_isSharedCheck_146_; 
v_a_135_ = lean_ctor_get(v___x_134_, 0);
v_isSharedCheck_146_ = !lean_is_exclusive(v___x_134_);
if (v_isSharedCheck_146_ == 0)
{
v___x_137_ = v___x_134_;
v_isShared_138_ = v_isSharedCheck_146_;
goto v_resetjp_136_;
}
else
{
lean_inc(v_a_135_);
lean_dec(v___x_134_);
v___x_137_ = lean_box(0);
v_isShared_138_ = v_isSharedCheck_146_;
goto v_resetjp_136_;
}
v_resetjp_136_:
{
if (lean_obj_tag(v_a_135_) == 0)
{
lean_object* v___x_140_; 
if (v_isShared_138_ == 0)
{
lean_ctor_set(v___x_137_, 0, v___x_117_);
v___x_140_ = v___x_137_;
goto v_reusejp_139_;
}
else
{
lean_object* v_reuseFailAlloc_141_; 
v_reuseFailAlloc_141_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_141_, 0, v___x_117_);
v___x_140_ = v_reuseFailAlloc_141_;
goto v_reusejp_139_;
}
v_reusejp_139_:
{
return v___x_140_;
}
}
else
{
lean_object* v___x_142_; lean_object* v___x_144_; 
lean_dec_ref_known(v_a_135_, 1);
v___x_142_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__6));
if (v_isShared_138_ == 0)
{
lean_ctor_set(v___x_137_, 0, v___x_142_);
v___x_144_ = v___x_137_;
goto v_reusejp_143_;
}
else
{
lean_object* v_reuseFailAlloc_145_; 
v_reuseFailAlloc_145_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_145_, 0, v___x_142_);
v___x_144_ = v_reuseFailAlloc_145_;
goto v_reusejp_143_;
}
v_reusejp_143_:
{
return v___x_144_;
}
}
}
}
else
{
lean_object* v_a_147_; lean_object* v___x_149_; uint8_t v_isShared_150_; uint8_t v_isSharedCheck_154_; 
v_a_147_ = lean_ctor_get(v___x_134_, 0);
v_isSharedCheck_154_ = !lean_is_exclusive(v___x_134_);
if (v_isSharedCheck_154_ == 0)
{
v___x_149_ = v___x_134_;
v_isShared_150_ = v_isSharedCheck_154_;
goto v_resetjp_148_;
}
else
{
lean_inc(v_a_147_);
lean_dec(v___x_134_);
v___x_149_ = lean_box(0);
v_isShared_150_ = v_isSharedCheck_154_;
goto v_resetjp_148_;
}
v_resetjp_148_:
{
lean_object* v___x_152_; 
if (v_isShared_150_ == 0)
{
v___x_152_ = v___x_149_;
goto v_reusejp_151_;
}
else
{
lean_object* v_reuseFailAlloc_153_; 
v_reuseFailAlloc_153_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_153_, 0, v_a_147_);
v___x_152_ = v_reuseFailAlloc_153_;
goto v_reusejp_151_;
}
v_reusejp_151_:
{
return v___x_152_;
}
}
}
}
else
{
lean_object* v_a_155_; lean_object* v___x_157_; uint8_t v_isShared_158_; uint8_t v_isSharedCheck_162_; 
v_a_155_ = lean_ctor_get(v___x_132_, 0);
v_isSharedCheck_162_ = !lean_is_exclusive(v___x_132_);
if (v_isSharedCheck_162_ == 0)
{
v___x_157_ = v___x_132_;
v_isShared_158_ = v_isSharedCheck_162_;
goto v_resetjp_156_;
}
else
{
lean_inc(v_a_155_);
lean_dec(v___x_132_);
v___x_157_ = lean_box(0);
v_isShared_158_ = v_isSharedCheck_162_;
goto v_resetjp_156_;
}
v_resetjp_156_:
{
lean_object* v___x_160_; 
if (v_isShared_158_ == 0)
{
v___x_160_ = v___x_157_;
goto v_reusejp_159_;
}
else
{
lean_object* v_reuseFailAlloc_161_; 
v_reuseFailAlloc_161_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_161_, 0, v_a_155_);
v___x_160_ = v_reuseFailAlloc_161_;
goto v_reusejp_159_;
}
v_reusejp_159_:
{
return v___x_160_;
}
}
}
}
else
{
lean_object* v___x_163_; lean_object* v___x_165_; 
lean_dec_ref_known(v_a_127_, 1);
lean_dec_ref(v___x_114_);
v___x_163_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__7));
if (v_isShared_130_ == 0)
{
lean_ctor_set(v___x_129_, 0, v___x_163_);
v___x_165_ = v___x_129_;
goto v_reusejp_164_;
}
else
{
lean_object* v_reuseFailAlloc_166_; 
v_reuseFailAlloc_166_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_166_, 0, v___x_163_);
v___x_165_ = v_reuseFailAlloc_166_;
goto v_reusejp_164_;
}
v_reusejp_164_:
{
return v___x_165_;
}
}
}
}
else
{
lean_object* v_a_168_; lean_object* v___x_170_; uint8_t v_isShared_171_; uint8_t v_isSharedCheck_175_; 
lean_dec_ref(v___x_114_);
v_a_168_ = lean_ctor_get(v___x_126_, 0);
v_isSharedCheck_175_ = !lean_is_exclusive(v___x_126_);
if (v_isSharedCheck_175_ == 0)
{
v___x_170_ = v___x_126_;
v_isShared_171_ = v_isSharedCheck_175_;
goto v_resetjp_169_;
}
else
{
lean_inc(v_a_168_);
lean_dec(v___x_126_);
v___x_170_ = lean_box(0);
v_isShared_171_ = v_isSharedCheck_175_;
goto v_resetjp_169_;
}
v_resetjp_169_:
{
lean_object* v___x_173_; 
if (v_isShared_171_ == 0)
{
v___x_173_ = v___x_170_;
goto v_reusejp_172_;
}
else
{
lean_object* v_reuseFailAlloc_174_; 
v_reuseFailAlloc_174_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_174_, 0, v_a_168_);
v___x_173_ = v_reuseFailAlloc_174_;
goto v_reusejp_172_;
}
v_reusejp_172_:
{
return v___x_173_;
}
}
}
}
else
{
lean_object* v_a_176_; lean_object* v___x_178_; uint8_t v_isShared_179_; uint8_t v_isSharedCheck_183_; 
lean_dec_ref(v___x_114_);
v_a_176_ = lean_ctor_get(v___x_124_, 0);
v_isSharedCheck_183_ = !lean_is_exclusive(v___x_124_);
if (v_isSharedCheck_183_ == 0)
{
v___x_178_ = v___x_124_;
v_isShared_179_ = v_isSharedCheck_183_;
goto v_resetjp_177_;
}
else
{
lean_inc(v_a_176_);
lean_dec(v___x_124_);
v___x_178_ = lean_box(0);
v_isShared_179_ = v_isSharedCheck_183_;
goto v_resetjp_177_;
}
v_resetjp_177_:
{
lean_object* v___x_181_; 
if (v_isShared_179_ == 0)
{
v___x_181_ = v___x_178_;
goto v_reusejp_180_;
}
else
{
lean_object* v_reuseFailAlloc_182_; 
v_reuseFailAlloc_182_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_182_, 0, v_a_176_);
v___x_181_ = v_reuseFailAlloc_182_;
goto v_reusejp_180_;
}
v_reusejp_180_:
{
return v___x_181_;
}
}
}
}
else
{
lean_object* v___x_184_; lean_object* v___x_186_; 
lean_dec_ref_known(v_a_119_, 1);
lean_dec_ref(v___x_114_);
v___x_184_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___closed__8));
if (v_isShared_122_ == 0)
{
lean_ctor_set(v___x_121_, 0, v___x_184_);
v___x_186_ = v___x_121_;
goto v_reusejp_185_;
}
else
{
lean_object* v_reuseFailAlloc_187_; 
v_reuseFailAlloc_187_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_187_, 0, v___x_184_);
v___x_186_ = v_reuseFailAlloc_187_;
goto v_reusejp_185_;
}
v_reusejp_185_:
{
return v___x_186_;
}
}
}
}
else
{
lean_object* v_a_189_; lean_object* v___x_191_; uint8_t v_isShared_192_; uint8_t v_isSharedCheck_196_; 
lean_dec_ref(v___x_114_);
v_a_189_ = lean_ctor_get(v___x_118_, 0);
v_isSharedCheck_196_ = !lean_is_exclusive(v___x_118_);
if (v_isSharedCheck_196_ == 0)
{
v___x_191_ = v___x_118_;
v_isShared_192_ = v_isSharedCheck_196_;
goto v_resetjp_190_;
}
else
{
lean_inc(v_a_189_);
lean_dec(v___x_118_);
v___x_191_ = lean_box(0);
v_isShared_192_ = v_isSharedCheck_196_;
goto v_resetjp_190_;
}
v_resetjp_190_:
{
lean_object* v___x_194_; 
if (v_isShared_192_ == 0)
{
v___x_194_ = v___x_191_;
goto v_reusejp_193_;
}
else
{
lean_object* v_reuseFailAlloc_195_; 
v_reuseFailAlloc_195_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_195_, 0, v_a_189_);
v___x_194_ = v_reuseFailAlloc_195_;
goto v_reusejp_193_;
}
v_reusejp_193_:
{
return v___x_194_;
}
}
}
}
else
{
lean_object* v_a_197_; lean_object* v___x_199_; uint8_t v_isShared_200_; uint8_t v_isSharedCheck_204_; 
lean_dec_ref(v___x_114_);
v_a_197_ = lean_ctor_get(v___x_115_, 0);
v_isSharedCheck_204_ = !lean_is_exclusive(v___x_115_);
if (v_isSharedCheck_204_ == 0)
{
v___x_199_ = v___x_115_;
v_isShared_200_ = v_isSharedCheck_204_;
goto v_resetjp_198_;
}
else
{
lean_inc(v_a_197_);
lean_dec(v___x_115_);
v___x_199_ = lean_box(0);
v_isShared_200_ = v_isSharedCheck_204_;
goto v_resetjp_198_;
}
v_resetjp_198_:
{
lean_object* v___x_202_; 
if (v_isShared_200_ == 0)
{
v___x_202_ = v___x_199_;
goto v_reusejp_201_;
}
else
{
lean_object* v_reuseFailAlloc_203_; 
v_reuseFailAlloc_203_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_203_, 0, v_a_197_);
v___x_202_ = v_reuseFailAlloc_203_;
goto v_reusejp_201_;
}
v_reusejp_201_:
{
return v___x_202_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance___boxed(lean_object* v_type_205_, lean_object* v_a_206_, lean_object* v_a_207_, lean_object* v_a_208_, lean_object* v_a_209_, lean_object* v_a_210_){
_start:
{
lean_object* v_res_211_; 
v_res_211_ = lp_mathlib_Mathlib_Tactic_Order_findBestOrderInstance(v_type_205_, v_a_206_, v_a_207_, v_a_208_, v_a_209_);
lean_dec(v_a_209_);
lean_dec_ref(v_a_208_);
lean_dec(v_a_207_);
lean_dec_ref(v_a_206_);
return v_res_211_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__2(void){
_start:
{
lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; 
v___x_215_ = lean_box(0);
v___x_216_ = lean_unsigned_to_nat(4u);
v___x_217_ = lean_mk_empty_array_with_capacity(v___x_216_);
v___x_218_ = lean_array_push(v___x_217_, v___x_215_);
return v___x_218_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__3(void){
_start:
{
lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; 
v___x_219_ = lean_box(0);
v___x_220_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__2, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__2_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__2);
v___x_221_ = lean_array_push(v___x_220_, v___x_219_);
return v___x_221_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__4(void){
_start:
{
lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; 
v___x_222_ = lean_box(0);
v___x_223_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__3);
v___x_224_ = lean_array_push(v___x_223_, v___x_222_);
return v___x_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg(lean_object* v_idx_225_, lean_object* v_a_226_, lean_object* v_as_227_, size_t v_sz_228_, size_t v_i_229_, lean_object* v_b_230_, lean_object* v___y_231_, lean_object* v___y_232_, lean_object* v___y_233_, lean_object* v___y_234_){
_start:
{
lean_object* v_a_237_; uint8_t v___x_241_; 
v___x_241_ = lean_usize_dec_lt(v_i_229_, v_sz_228_);
if (v___x_241_ == 0)
{
lean_object* v___x_242_; 
lean_dec_ref(v_a_226_);
lean_dec(v_idx_225_);
v___x_242_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_242_, 0, v_b_230_);
return v___x_242_;
}
else
{
lean_object* v_a_243_; lean_object* v_fst_244_; lean_object* v_snd_245_; uint8_t v_a_247_; lean_object* v___x_265_; 
v_a_243_ = lean_array_uget_borrowed(v_as_227_, v_i_229_);
v_fst_244_ = lean_ctor_get(v_a_243_, 0);
v_snd_245_ = lean_ctor_get(v_a_243_, 1);
lean_inc(v___y_234_);
lean_inc_ref(v___y_233_);
lean_inc(v___y_232_);
lean_inc_ref(v___y_231_);
lean_inc(v_fst_244_);
v___x_265_ = lean_infer_type(v_fst_244_, v___y_231_, v___y_232_, v___y_233_, v___y_234_);
if (lean_obj_tag(v___x_265_) == 0)
{
lean_object* v_a_266_; lean_object* v_keyedConfig_267_; uint8_t v_trackZetaDelta_268_; lean_object* v_zetaDeltaSet_269_; lean_object* v_lctx_270_; lean_object* v_localInstances_271_; lean_object* v_defEqCtx_x3f_272_; lean_object* v_synthPendingDepth_273_; lean_object* v_customCanUnfoldPredicate_x3f_274_; uint8_t v_univApprox_275_; uint8_t v_inTypeClassResolution_276_; uint8_t v_cacheInferType_277_; uint8_t v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; 
v_a_266_ = lean_ctor_get(v___x_265_, 0);
lean_inc(v_a_266_);
lean_dec_ref_known(v___x_265_, 1);
v_keyedConfig_267_ = lean_ctor_get(v___y_231_, 0);
v_trackZetaDelta_268_ = lean_ctor_get_uint8(v___y_231_, sizeof(void*)*7);
v_zetaDeltaSet_269_ = lean_ctor_get(v___y_231_, 1);
v_lctx_270_ = lean_ctor_get(v___y_231_, 2);
v_localInstances_271_ = lean_ctor_get(v___y_231_, 3);
v_defEqCtx_x3f_272_ = lean_ctor_get(v___y_231_, 4);
v_synthPendingDepth_273_ = lean_ctor_get(v___y_231_, 5);
v_customCanUnfoldPredicate_x3f_274_ = lean_ctor_get(v___y_231_, 6);
v_univApprox_275_ = lean_ctor_get_uint8(v___y_231_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_276_ = lean_ctor_get_uint8(v___y_231_, sizeof(void*)*7 + 2);
v_cacheInferType_277_ = lean_ctor_get_uint8(v___y_231_, sizeof(void*)*7 + 3);
v___x_278_ = 2;
lean_inc_ref(v_keyedConfig_267_);
v___x_279_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_278_, v_keyedConfig_267_);
lean_inc(v_customCanUnfoldPredicate_x3f_274_);
lean_inc(v_synthPendingDepth_273_);
lean_inc(v_defEqCtx_x3f_272_);
lean_inc_ref(v_localInstances_271_);
lean_inc_ref(v_lctx_270_);
lean_inc(v_zetaDeltaSet_269_);
v___x_280_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_280_, 0, v___x_279_);
lean_ctor_set(v___x_280_, 1, v_zetaDeltaSet_269_);
lean_ctor_set(v___x_280_, 2, v_lctx_270_);
lean_ctor_set(v___x_280_, 3, v_localInstances_271_);
lean_ctor_set(v___x_280_, 4, v_defEqCtx_x3f_272_);
lean_ctor_set(v___x_280_, 5, v_synthPendingDepth_273_);
lean_ctor_set(v___x_280_, 6, v_customCanUnfoldPredicate_x3f_274_);
lean_ctor_set_uint8(v___x_280_, sizeof(void*)*7, v_trackZetaDelta_268_);
lean_ctor_set_uint8(v___x_280_, sizeof(void*)*7 + 1, v_univApprox_275_);
lean_ctor_set_uint8(v___x_280_, sizeof(void*)*7 + 2, v_inTypeClassResolution_276_);
lean_ctor_set_uint8(v___x_280_, sizeof(void*)*7 + 3, v_cacheInferType_277_);
lean_inc_ref(v_a_226_);
v___x_281_ = l_Lean_Meta_isExprDefEq(v_a_226_, v_a_266_, v___x_280_, v___y_232_, v___y_233_, v___y_234_);
lean_dec_ref_known(v___x_280_, 7);
if (lean_obj_tag(v___x_281_) == 0)
{
lean_object* v_a_282_; uint8_t v___x_283_; 
v_a_282_ = lean_ctor_get(v___x_281_, 0);
lean_inc(v_a_282_);
lean_dec_ref_known(v___x_281_, 1);
v___x_283_ = lean_unbox(v_a_282_);
lean_dec(v_a_282_);
v_a_247_ = v___x_283_;
goto v___jp_246_;
}
else
{
if (lean_obj_tag(v___x_281_) == 0)
{
lean_object* v_a_284_; uint8_t v___x_285_; 
v_a_284_ = lean_ctor_get(v___x_281_, 0);
lean_inc(v_a_284_);
lean_dec_ref_known(v___x_281_, 1);
v___x_285_ = lean_unbox(v_a_284_);
lean_dec(v_a_284_);
v_a_247_ = v___x_285_;
goto v___jp_246_;
}
else
{
lean_object* v_a_286_; lean_object* v___x_288_; uint8_t v_isShared_289_; uint8_t v_isSharedCheck_293_; 
lean_dec_ref(v_b_230_);
lean_dec_ref(v_a_226_);
lean_dec(v_idx_225_);
v_a_286_ = lean_ctor_get(v___x_281_, 0);
v_isSharedCheck_293_ = !lean_is_exclusive(v___x_281_);
if (v_isSharedCheck_293_ == 0)
{
v___x_288_ = v___x_281_;
v_isShared_289_ = v_isSharedCheck_293_;
goto v_resetjp_287_;
}
else
{
lean_inc(v_a_286_);
lean_dec(v___x_281_);
v___x_288_ = lean_box(0);
v_isShared_289_ = v_isSharedCheck_293_;
goto v_resetjp_287_;
}
v_resetjp_287_:
{
lean_object* v___x_291_; 
if (v_isShared_289_ == 0)
{
v___x_291_ = v___x_288_;
goto v_reusejp_290_;
}
else
{
lean_object* v_reuseFailAlloc_292_; 
v_reuseFailAlloc_292_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_292_, 0, v_a_286_);
v___x_291_ = v_reuseFailAlloc_292_;
goto v_reusejp_290_;
}
v_reusejp_290_:
{
return v___x_291_;
}
}
}
}
}
else
{
lean_object* v_a_294_; lean_object* v___x_296_; uint8_t v_isShared_297_; uint8_t v_isSharedCheck_301_; 
lean_dec_ref(v_b_230_);
lean_dec_ref(v_a_226_);
lean_dec(v_idx_225_);
v_a_294_ = lean_ctor_get(v___x_265_, 0);
v_isSharedCheck_301_ = !lean_is_exclusive(v___x_265_);
if (v_isSharedCheck_301_ == 0)
{
v___x_296_ = v___x_265_;
v_isShared_297_ = v_isSharedCheck_301_;
goto v_resetjp_295_;
}
else
{
lean_inc(v_a_294_);
lean_dec(v___x_265_);
v___x_296_ = lean_box(0);
v_isShared_297_ = v_isSharedCheck_301_;
goto v_resetjp_295_;
}
v_resetjp_295_:
{
lean_object* v___x_299_; 
if (v_isShared_297_ == 0)
{
v___x_299_ = v___x_296_;
goto v_reusejp_298_;
}
else
{
lean_object* v_reuseFailAlloc_300_; 
v_reuseFailAlloc_300_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_300_, 0, v_a_294_);
v___x_299_ = v_reuseFailAlloc_300_;
goto v_reusejp_298_;
}
v_reusejp_298_:
{
return v___x_299_;
}
}
}
v___jp_246_:
{
if (v_a_247_ == 0)
{
v_a_237_ = v_b_230_;
goto v___jp_236_;
}
else
{
uint8_t v___x_248_; 
v___x_248_ = lean_nat_dec_eq(v_snd_245_, v_idx_225_);
if (v___x_248_ == 0)
{
lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; 
v___x_249_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__1));
lean_inc(v_fst_244_);
v___x_250_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_250_, 0, v_fst_244_);
v___x_251_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__4, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__4_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__4);
v___x_252_ = lean_array_push(v___x_251_, v___x_250_);
v___x_253_ = l_Lean_Meta_mkAppOptM(v___x_249_, v___x_252_, v___y_231_, v___y_232_, v___y_233_, v___y_234_);
if (lean_obj_tag(v___x_253_) == 0)
{
lean_object* v_a_254_; lean_object* v___x_255_; lean_object* v___x_256_; 
v_a_254_ = lean_ctor_get(v___x_253_, 0);
lean_inc(v_a_254_);
lean_dec_ref_known(v___x_253_, 1);
lean_inc(v_snd_245_);
lean_inc(v_idx_225_);
v___x_255_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_255_, 0, v_idx_225_);
lean_ctor_set(v___x_255_, 1, v_snd_245_);
lean_ctor_set(v___x_255_, 2, v_a_254_);
v___x_256_ = lean_array_push(v_b_230_, v___x_255_);
v_a_237_ = v___x_256_;
goto v___jp_236_;
}
else
{
lean_object* v_a_257_; lean_object* v___x_259_; uint8_t v_isShared_260_; uint8_t v_isSharedCheck_264_; 
lean_dec_ref(v_b_230_);
lean_dec_ref(v_a_226_);
lean_dec(v_idx_225_);
v_a_257_ = lean_ctor_get(v___x_253_, 0);
v_isSharedCheck_264_ = !lean_is_exclusive(v___x_253_);
if (v_isSharedCheck_264_ == 0)
{
v___x_259_ = v___x_253_;
v_isShared_260_ = v_isSharedCheck_264_;
goto v_resetjp_258_;
}
else
{
lean_inc(v_a_257_);
lean_dec(v___x_253_);
v___x_259_ = lean_box(0);
v_isShared_260_ = v_isSharedCheck_264_;
goto v_resetjp_258_;
}
v_resetjp_258_:
{
lean_object* v___x_262_; 
if (v_isShared_260_ == 0)
{
v___x_262_ = v___x_259_;
goto v_reusejp_261_;
}
else
{
lean_object* v_reuseFailAlloc_263_; 
v_reuseFailAlloc_263_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_263_, 0, v_a_257_);
v___x_262_ = v_reuseFailAlloc_263_;
goto v_reusejp_261_;
}
v_reusejp_261_:
{
return v___x_262_;
}
}
}
}
else
{
v_a_237_ = v_b_230_;
goto v___jp_236_;
}
}
}
}
v___jp_236_:
{
size_t v___x_238_; size_t v___x_239_; 
v___x_238_ = ((size_t)1ULL);
v___x_239_ = lean_usize_add(v_i_229_, v___x_238_);
v_i_229_ = v___x_239_;
v_b_230_ = v_a_237_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___boxed(lean_object* v_idx_302_, lean_object* v_a_303_, lean_object* v_as_304_, lean_object* v_sz_305_, lean_object* v_i_306_, lean_object* v_b_307_, lean_object* v___y_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_){
_start:
{
size_t v_sz_boxed_313_; size_t v_i_boxed_314_; lean_object* v_res_315_; 
v_sz_boxed_313_ = lean_unbox_usize(v_sz_305_);
lean_dec(v_sz_305_);
v_i_boxed_314_ = lean_unbox_usize(v_i_306_);
lean_dec(v_i_306_);
v_res_315_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg(v_idx_302_, v_a_303_, v_as_304_, v_sz_boxed_313_, v_i_boxed_314_, v_b_307_, v___y_308_, v___y_309_, v___y_310_, v___y_311_);
lean_dec(v___y_311_);
lean_dec_ref(v___y_310_);
lean_dec(v___y_309_);
lean_dec_ref(v___y_308_);
lean_dec_ref(v_as_304_);
return v_res_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__1___redArg(lean_object* v_idx_319_, lean_object* v_a_320_, lean_object* v_as_321_, size_t v_sz_322_, size_t v_i_323_, lean_object* v_b_324_, lean_object* v___y_325_, lean_object* v___y_326_, lean_object* v___y_327_, lean_object* v___y_328_){
_start:
{
lean_object* v_a_331_; uint8_t v___x_335_; 
v___x_335_ = lean_usize_dec_lt(v_i_323_, v_sz_322_);
if (v___x_335_ == 0)
{
lean_object* v___x_336_; 
lean_dec_ref(v_a_320_);
lean_dec(v_idx_319_);
v___x_336_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_336_, 0, v_b_324_);
return v___x_336_;
}
else
{
lean_object* v_a_337_; lean_object* v_fst_338_; lean_object* v_snd_339_; uint8_t v_a_341_; lean_object* v___x_359_; 
v_a_337_ = lean_array_uget_borrowed(v_as_321_, v_i_323_);
v_fst_338_ = lean_ctor_get(v_a_337_, 0);
v_snd_339_ = lean_ctor_get(v_a_337_, 1);
lean_inc(v___y_328_);
lean_inc_ref(v___y_327_);
lean_inc(v___y_326_);
lean_inc_ref(v___y_325_);
lean_inc(v_fst_338_);
v___x_359_ = lean_infer_type(v_fst_338_, v___y_325_, v___y_326_, v___y_327_, v___y_328_);
if (lean_obj_tag(v___x_359_) == 0)
{
lean_object* v_a_360_; lean_object* v_keyedConfig_361_; uint8_t v_trackZetaDelta_362_; lean_object* v_zetaDeltaSet_363_; lean_object* v_lctx_364_; lean_object* v_localInstances_365_; lean_object* v_defEqCtx_x3f_366_; lean_object* v_synthPendingDepth_367_; lean_object* v_customCanUnfoldPredicate_x3f_368_; uint8_t v_univApprox_369_; uint8_t v_inTypeClassResolution_370_; uint8_t v_cacheInferType_371_; uint8_t v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; 
v_a_360_ = lean_ctor_get(v___x_359_, 0);
lean_inc(v_a_360_);
lean_dec_ref_known(v___x_359_, 1);
v_keyedConfig_361_ = lean_ctor_get(v___y_325_, 0);
v_trackZetaDelta_362_ = lean_ctor_get_uint8(v___y_325_, sizeof(void*)*7);
v_zetaDeltaSet_363_ = lean_ctor_get(v___y_325_, 1);
v_lctx_364_ = lean_ctor_get(v___y_325_, 2);
v_localInstances_365_ = lean_ctor_get(v___y_325_, 3);
v_defEqCtx_x3f_366_ = lean_ctor_get(v___y_325_, 4);
v_synthPendingDepth_367_ = lean_ctor_get(v___y_325_, 5);
v_customCanUnfoldPredicate_x3f_368_ = lean_ctor_get(v___y_325_, 6);
v_univApprox_369_ = lean_ctor_get_uint8(v___y_325_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_370_ = lean_ctor_get_uint8(v___y_325_, sizeof(void*)*7 + 2);
v_cacheInferType_371_ = lean_ctor_get_uint8(v___y_325_, sizeof(void*)*7 + 3);
v___x_372_ = 2;
lean_inc_ref(v_keyedConfig_361_);
v___x_373_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_372_, v_keyedConfig_361_);
lean_inc(v_customCanUnfoldPredicate_x3f_368_);
lean_inc(v_synthPendingDepth_367_);
lean_inc(v_defEqCtx_x3f_366_);
lean_inc_ref(v_localInstances_365_);
lean_inc_ref(v_lctx_364_);
lean_inc(v_zetaDeltaSet_363_);
v___x_374_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_374_, 0, v___x_373_);
lean_ctor_set(v___x_374_, 1, v_zetaDeltaSet_363_);
lean_ctor_set(v___x_374_, 2, v_lctx_364_);
lean_ctor_set(v___x_374_, 3, v_localInstances_365_);
lean_ctor_set(v___x_374_, 4, v_defEqCtx_x3f_366_);
lean_ctor_set(v___x_374_, 5, v_synthPendingDepth_367_);
lean_ctor_set(v___x_374_, 6, v_customCanUnfoldPredicate_x3f_368_);
lean_ctor_set_uint8(v___x_374_, sizeof(void*)*7, v_trackZetaDelta_362_);
lean_ctor_set_uint8(v___x_374_, sizeof(void*)*7 + 1, v_univApprox_369_);
lean_ctor_set_uint8(v___x_374_, sizeof(void*)*7 + 2, v_inTypeClassResolution_370_);
lean_ctor_set_uint8(v___x_374_, sizeof(void*)*7 + 3, v_cacheInferType_371_);
lean_inc_ref(v_a_320_);
v___x_375_ = l_Lean_Meta_isExprDefEq(v_a_320_, v_a_360_, v___x_374_, v___y_326_, v___y_327_, v___y_328_);
lean_dec_ref_known(v___x_374_, 7);
if (lean_obj_tag(v___x_375_) == 0)
{
lean_object* v_a_376_; uint8_t v___x_377_; 
v_a_376_ = lean_ctor_get(v___x_375_, 0);
lean_inc(v_a_376_);
lean_dec_ref_known(v___x_375_, 1);
v___x_377_ = lean_unbox(v_a_376_);
lean_dec(v_a_376_);
v_a_341_ = v___x_377_;
goto v___jp_340_;
}
else
{
if (lean_obj_tag(v___x_375_) == 0)
{
lean_object* v_a_378_; uint8_t v___x_379_; 
v_a_378_ = lean_ctor_get(v___x_375_, 0);
lean_inc(v_a_378_);
lean_dec_ref_known(v___x_375_, 1);
v___x_379_ = lean_unbox(v_a_378_);
lean_dec(v_a_378_);
v_a_341_ = v___x_379_;
goto v___jp_340_;
}
else
{
lean_object* v_a_380_; lean_object* v___x_382_; uint8_t v_isShared_383_; uint8_t v_isSharedCheck_387_; 
lean_dec_ref(v_b_324_);
lean_dec_ref(v_a_320_);
lean_dec(v_idx_319_);
v_a_380_ = lean_ctor_get(v___x_375_, 0);
v_isSharedCheck_387_ = !lean_is_exclusive(v___x_375_);
if (v_isSharedCheck_387_ == 0)
{
v___x_382_ = v___x_375_;
v_isShared_383_ = v_isSharedCheck_387_;
goto v_resetjp_381_;
}
else
{
lean_inc(v_a_380_);
lean_dec(v___x_375_);
v___x_382_ = lean_box(0);
v_isShared_383_ = v_isSharedCheck_387_;
goto v_resetjp_381_;
}
v_resetjp_381_:
{
lean_object* v___x_385_; 
if (v_isShared_383_ == 0)
{
v___x_385_ = v___x_382_;
goto v_reusejp_384_;
}
else
{
lean_object* v_reuseFailAlloc_386_; 
v_reuseFailAlloc_386_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_386_, 0, v_a_380_);
v___x_385_ = v_reuseFailAlloc_386_;
goto v_reusejp_384_;
}
v_reusejp_384_:
{
return v___x_385_;
}
}
}
}
}
else
{
lean_object* v_a_388_; lean_object* v___x_390_; uint8_t v_isShared_391_; uint8_t v_isSharedCheck_395_; 
lean_dec_ref(v_b_324_);
lean_dec_ref(v_a_320_);
lean_dec(v_idx_319_);
v_a_388_ = lean_ctor_get(v___x_359_, 0);
v_isSharedCheck_395_ = !lean_is_exclusive(v___x_359_);
if (v_isSharedCheck_395_ == 0)
{
v___x_390_ = v___x_359_;
v_isShared_391_ = v_isSharedCheck_395_;
goto v_resetjp_389_;
}
else
{
lean_inc(v_a_388_);
lean_dec(v___x_359_);
v___x_390_ = lean_box(0);
v_isShared_391_ = v_isSharedCheck_395_;
goto v_resetjp_389_;
}
v_resetjp_389_:
{
lean_object* v___x_393_; 
if (v_isShared_391_ == 0)
{
v___x_393_ = v___x_390_;
goto v_reusejp_392_;
}
else
{
lean_object* v_reuseFailAlloc_394_; 
v_reuseFailAlloc_394_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_394_, 0, v_a_388_);
v___x_393_ = v_reuseFailAlloc_394_;
goto v_reusejp_392_;
}
v_reusejp_392_:
{
return v___x_393_;
}
}
}
v___jp_340_:
{
if (v_a_341_ == 0)
{
v_a_331_ = v_b_324_;
goto v___jp_330_;
}
else
{
uint8_t v___x_342_; 
v___x_342_ = lean_nat_dec_eq(v_snd_339_, v_idx_319_);
if (v___x_342_ == 0)
{
lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; 
v___x_343_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__1___redArg___closed__1));
lean_inc(v_fst_338_);
v___x_344_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_344_, 0, v_fst_338_);
v___x_345_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__4, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__4_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__4);
v___x_346_ = lean_array_push(v___x_345_, v___x_344_);
v___x_347_ = l_Lean_Meta_mkAppOptM(v___x_343_, v___x_346_, v___y_325_, v___y_326_, v___y_327_, v___y_328_);
if (lean_obj_tag(v___x_347_) == 0)
{
lean_object* v_a_348_; lean_object* v___x_349_; lean_object* v___x_350_; 
v_a_348_ = lean_ctor_get(v___x_347_, 0);
lean_inc(v_a_348_);
lean_dec_ref_known(v___x_347_, 1);
lean_inc(v_idx_319_);
lean_inc(v_snd_339_);
v___x_349_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_349_, 0, v_snd_339_);
lean_ctor_set(v___x_349_, 1, v_idx_319_);
lean_ctor_set(v___x_349_, 2, v_a_348_);
v___x_350_ = lean_array_push(v_b_324_, v___x_349_);
v_a_331_ = v___x_350_;
goto v___jp_330_;
}
else
{
lean_object* v_a_351_; lean_object* v___x_353_; uint8_t v_isShared_354_; uint8_t v_isSharedCheck_358_; 
lean_dec_ref(v_b_324_);
lean_dec_ref(v_a_320_);
lean_dec(v_idx_319_);
v_a_351_ = lean_ctor_get(v___x_347_, 0);
v_isSharedCheck_358_ = !lean_is_exclusive(v___x_347_);
if (v_isSharedCheck_358_ == 0)
{
v___x_353_ = v___x_347_;
v_isShared_354_ = v_isSharedCheck_358_;
goto v_resetjp_352_;
}
else
{
lean_inc(v_a_351_);
lean_dec(v___x_347_);
v___x_353_ = lean_box(0);
v_isShared_354_ = v_isSharedCheck_358_;
goto v_resetjp_352_;
}
v_resetjp_352_:
{
lean_object* v___x_356_; 
if (v_isShared_354_ == 0)
{
v___x_356_ = v___x_353_;
goto v_reusejp_355_;
}
else
{
lean_object* v_reuseFailAlloc_357_; 
v_reuseFailAlloc_357_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_357_, 0, v_a_351_);
v___x_356_ = v_reuseFailAlloc_357_;
goto v_reusejp_355_;
}
v_reusejp_355_:
{
return v___x_356_;
}
}
}
}
else
{
v_a_331_ = v_b_324_;
goto v___jp_330_;
}
}
}
}
v___jp_330_:
{
size_t v___x_332_; size_t v___x_333_; 
v___x_332_ = ((size_t)1ULL);
v___x_333_ = lean_usize_add(v_i_323_, v___x_332_);
v_i_323_ = v___x_333_;
v_b_324_ = v_a_331_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__1___redArg___boxed(lean_object* v_idx_396_, lean_object* v_a_397_, lean_object* v_as_398_, lean_object* v_sz_399_, lean_object* v_i_400_, lean_object* v_b_401_, lean_object* v___y_402_, lean_object* v___y_403_, lean_object* v___y_404_, lean_object* v___y_405_, lean_object* v___y_406_){
_start:
{
size_t v_sz_boxed_407_; size_t v_i_boxed_408_; lean_object* v_res_409_; 
v_sz_boxed_407_ = lean_unbox_usize(v_sz_399_);
lean_dec(v_sz_399_);
v_i_boxed_408_ = lean_unbox_usize(v_i_400_);
lean_dec(v_i_400_);
v_res_409_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__1___redArg(v_idx_396_, v_a_397_, v_as_398_, v_sz_boxed_407_, v_i_boxed_408_, v_b_401_, v___y_402_, v___y_403_, v___y_404_, v___y_405_);
lean_dec(v___y_405_);
lean_dec_ref(v___y_404_);
lean_dec(v___y_403_);
lean_dec_ref(v___y_402_);
lean_dec_ref(v_as_398_);
return v_res_409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__2(lean_object* v_as_410_, size_t v_sz_411_, size_t v_i_412_, lean_object* v_b_413_, lean_object* v___y_414_, lean_object* v___y_415_, lean_object* v___y_416_, lean_object* v___y_417_, lean_object* v___y_418_, lean_object* v___y_419_){
_start:
{
lean_object* v_a_422_; uint8_t v___x_426_; 
v___x_426_ = lean_usize_dec_lt(v_i_412_, v_sz_411_);
if (v___x_426_ == 0)
{
lean_object* v___x_427_; 
v___x_427_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_427_, 0, v_b_413_);
return v___x_427_;
}
else
{
lean_object* v___x_428_; lean_object* v_a_429_; 
v___x_428_ = l_Lean_instInhabitedExpr;
v_a_429_ = lean_array_uget_borrowed(v_as_410_, v_i_412_);
switch(lean_obj_tag(v_a_429_))
{
case 7:
{
lean_object* v_idx_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; 
v_idx_430_ = lean_ctor_get(v_a_429_, 0);
v___x_431_ = lean_st_ref_get(v___y_415_);
v___x_432_ = lean_array_get(v___x_428_, v___x_431_, v_idx_430_);
lean_dec(v___x_431_);
lean_inc(v___y_419_);
lean_inc_ref(v___y_418_);
lean_inc(v___y_417_);
lean_inc_ref(v___y_416_);
v___x_433_ = lean_infer_type(v___x_432_, v___y_416_, v___y_417_, v___y_418_, v___y_419_);
if (lean_obj_tag(v___x_433_) == 0)
{
lean_object* v_a_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; size_t v_sz_438_; size_t v___x_439_; lean_object* v___x_440_; 
v_a_434_ = lean_ctor_get(v___x_433_, 0);
lean_inc(v_a_434_);
lean_dec_ref_known(v___x_433_, 1);
v___x_435_ = lean_st_ref_get(v___y_415_);
v___x_436_ = lean_unsigned_to_nat(0u);
v___x_437_ = l_Array_zipIdx___redArg(v___x_435_, v___x_436_);
v_sz_438_ = lean_array_size(v___x_437_);
v___x_439_ = ((size_t)0ULL);
lean_inc(v_idx_430_);
v___x_440_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg(v_idx_430_, v_a_434_, v___x_437_, v_sz_438_, v___x_439_, v_b_413_, v___y_416_, v___y_417_, v___y_418_, v___y_419_);
lean_dec_ref(v___x_437_);
if (lean_obj_tag(v___x_440_) == 0)
{
lean_object* v_a_441_; 
v_a_441_ = lean_ctor_get(v___x_440_, 0);
lean_inc(v_a_441_);
lean_dec_ref_known(v___x_440_, 1);
v_a_422_ = v_a_441_;
goto v___jp_421_;
}
else
{
return v___x_440_;
}
}
else
{
lean_object* v_a_442_; lean_object* v___x_444_; uint8_t v_isShared_445_; uint8_t v_isSharedCheck_449_; 
lean_dec_ref(v_b_413_);
v_a_442_ = lean_ctor_get(v___x_433_, 0);
v_isSharedCheck_449_ = !lean_is_exclusive(v___x_433_);
if (v_isSharedCheck_449_ == 0)
{
v___x_444_ = v___x_433_;
v_isShared_445_ = v_isSharedCheck_449_;
goto v_resetjp_443_;
}
else
{
lean_inc(v_a_442_);
lean_dec(v___x_433_);
v___x_444_ = lean_box(0);
v_isShared_445_ = v_isSharedCheck_449_;
goto v_resetjp_443_;
}
v_resetjp_443_:
{
lean_object* v___x_447_; 
if (v_isShared_445_ == 0)
{
v___x_447_ = v___x_444_;
goto v_reusejp_446_;
}
else
{
lean_object* v_reuseFailAlloc_448_; 
v_reuseFailAlloc_448_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_448_, 0, v_a_442_);
v___x_447_ = v_reuseFailAlloc_448_;
goto v_reusejp_446_;
}
v_reusejp_446_:
{
return v___x_447_;
}
}
}
}
case 6:
{
lean_object* v_idx_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; 
v_idx_450_ = lean_ctor_get(v_a_429_, 0);
v___x_451_ = lean_st_ref_get(v___y_415_);
v___x_452_ = lean_array_get(v___x_428_, v___x_451_, v_idx_450_);
lean_dec(v___x_451_);
lean_inc(v___y_419_);
lean_inc_ref(v___y_418_);
lean_inc(v___y_417_);
lean_inc_ref(v___y_416_);
v___x_453_ = lean_infer_type(v___x_452_, v___y_416_, v___y_417_, v___y_418_, v___y_419_);
if (lean_obj_tag(v___x_453_) == 0)
{
lean_object* v_a_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; size_t v_sz_458_; size_t v___x_459_; lean_object* v___x_460_; 
v_a_454_ = lean_ctor_get(v___x_453_, 0);
lean_inc(v_a_454_);
lean_dec_ref_known(v___x_453_, 1);
v___x_455_ = lean_st_ref_get(v___y_415_);
v___x_456_ = lean_unsigned_to_nat(0u);
v___x_457_ = l_Array_zipIdx___redArg(v___x_455_, v___x_456_);
v_sz_458_ = lean_array_size(v___x_457_);
v___x_459_ = ((size_t)0ULL);
lean_inc(v_idx_450_);
v___x_460_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__1___redArg(v_idx_450_, v_a_454_, v___x_457_, v_sz_458_, v___x_459_, v_b_413_, v___y_416_, v___y_417_, v___y_418_, v___y_419_);
lean_dec_ref(v___x_457_);
if (lean_obj_tag(v___x_460_) == 0)
{
lean_object* v_a_461_; 
v_a_461_ = lean_ctor_get(v___x_460_, 0);
lean_inc(v_a_461_);
lean_dec_ref_known(v___x_460_, 1);
v_a_422_ = v_a_461_;
goto v___jp_421_;
}
else
{
return v___x_460_;
}
}
else
{
lean_object* v_a_462_; lean_object* v___x_464_; uint8_t v_isShared_465_; uint8_t v_isSharedCheck_469_; 
lean_dec_ref(v_b_413_);
v_a_462_ = lean_ctor_get(v___x_453_, 0);
v_isSharedCheck_469_ = !lean_is_exclusive(v___x_453_);
if (v_isSharedCheck_469_ == 0)
{
v___x_464_ = v___x_453_;
v_isShared_465_ = v_isSharedCheck_469_;
goto v_resetjp_463_;
}
else
{
lean_inc(v_a_462_);
lean_dec(v___x_453_);
v___x_464_ = lean_box(0);
v_isShared_465_ = v_isSharedCheck_469_;
goto v_resetjp_463_;
}
v_resetjp_463_:
{
lean_object* v___x_467_; 
if (v_isShared_465_ == 0)
{
v___x_467_ = v___x_464_;
goto v_reusejp_466_;
}
else
{
lean_object* v_reuseFailAlloc_468_; 
v_reuseFailAlloc_468_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_468_, 0, v_a_462_);
v___x_467_ = v_reuseFailAlloc_468_;
goto v_reusejp_466_;
}
v_reusejp_466_:
{
return v___x_467_;
}
}
}
}
default: 
{
lean_object* v___x_470_; 
lean_inc(v_a_429_);
v___x_470_ = lean_array_push(v_b_413_, v_a_429_);
v_a_422_ = v___x_470_;
goto v___jp_421_;
}
}
}
v___jp_421_:
{
size_t v___x_423_; size_t v___x_424_; 
v___x_423_ = ((size_t)1ULL);
v___x_424_ = lean_usize_add(v_i_412_, v___x_423_);
v_i_412_ = v___x_424_;
v_b_413_ = v_a_422_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__2___boxed(lean_object* v_as_471_, lean_object* v_sz_472_, lean_object* v_i_473_, lean_object* v_b_474_, lean_object* v___y_475_, lean_object* v___y_476_, lean_object* v___y_477_, lean_object* v___y_478_, lean_object* v___y_479_, lean_object* v___y_480_, lean_object* v___y_481_){
_start:
{
size_t v_sz_boxed_482_; size_t v_i_boxed_483_; lean_object* v_res_484_; 
v_sz_boxed_482_ = lean_unbox_usize(v_sz_472_);
lean_dec(v_sz_472_);
v_i_boxed_483_ = lean_unbox_usize(v_i_473_);
lean_dec(v_i_473_);
v_res_484_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__2(v_as_471_, v_sz_boxed_482_, v_i_boxed_483_, v_b_474_, v___y_475_, v___y_476_, v___y_477_, v___y_478_, v___y_479_, v___y_480_);
lean_dec(v___y_480_);
lean_dec_ref(v___y_479_);
lean_dec(v___y_478_);
lean_dec_ref(v___y_477_);
lean_dec(v___y_476_);
lean_dec_ref(v___y_475_);
lean_dec_ref(v_as_471_);
return v_res_484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_replaceBotTop(lean_object* v_facts_487_, lean_object* v_a_488_, lean_object* v_a_489_, lean_object* v_a_490_, lean_object* v_a_491_, lean_object* v_a_492_, lean_object* v_a_493_){
_start:
{
lean_object* v_res_495_; size_t v_sz_496_; size_t v___x_497_; lean_object* v___x_498_; 
v_res_495_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_replaceBotTop___closed__0));
v_sz_496_ = lean_array_size(v_facts_487_);
v___x_497_ = ((size_t)0ULL);
v___x_498_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__2(v_facts_487_, v_sz_496_, v___x_497_, v_res_495_, v_a_488_, v_a_489_, v_a_490_, v_a_491_, v_a_492_, v_a_493_);
return v___x_498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_replaceBotTop___boxed(lean_object* v_facts_499_, lean_object* v_a_500_, lean_object* v_a_501_, lean_object* v_a_502_, lean_object* v_a_503_, lean_object* v_a_504_, lean_object* v_a_505_, lean_object* v_a_506_){
_start:
{
lean_object* v_res_507_; 
v_res_507_ = lp_mathlib_Mathlib_Tactic_Order_replaceBotTop(v_facts_499_, v_a_500_, v_a_501_, v_a_502_, v_a_503_, v_a_504_, v_a_505_);
lean_dec(v_a_505_);
lean_dec_ref(v_a_504_);
lean_dec(v_a_503_);
lean_dec_ref(v_a_502_);
lean_dec(v_a_501_);
lean_dec_ref(v_a_500_);
lean_dec_ref(v_facts_499_);
return v_res_507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0(lean_object* v_idx_508_, lean_object* v_a_509_, lean_object* v_as_510_, size_t v_sz_511_, size_t v_i_512_, lean_object* v_b_513_, lean_object* v___y_514_, lean_object* v___y_515_, lean_object* v___y_516_, lean_object* v___y_517_, lean_object* v___y_518_, lean_object* v___y_519_){
_start:
{
lean_object* v___x_521_; 
v___x_521_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg(v_idx_508_, v_a_509_, v_as_510_, v_sz_511_, v_i_512_, v_b_513_, v___y_516_, v___y_517_, v___y_518_, v___y_519_);
return v___x_521_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___boxed(lean_object* v_idx_522_, lean_object* v_a_523_, lean_object* v_as_524_, lean_object* v_sz_525_, lean_object* v_i_526_, lean_object* v_b_527_, lean_object* v___y_528_, lean_object* v___y_529_, lean_object* v___y_530_, lean_object* v___y_531_, lean_object* v___y_532_, lean_object* v___y_533_, lean_object* v___y_534_){
_start:
{
size_t v_sz_boxed_535_; size_t v_i_boxed_536_; lean_object* v_res_537_; 
v_sz_boxed_535_ = lean_unbox_usize(v_sz_525_);
lean_dec(v_sz_525_);
v_i_boxed_536_ = lean_unbox_usize(v_i_526_);
lean_dec(v_i_526_);
v_res_537_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0(v_idx_522_, v_a_523_, v_as_524_, v_sz_boxed_535_, v_i_boxed_536_, v_b_527_, v___y_528_, v___y_529_, v___y_530_, v___y_531_, v___y_532_, v___y_533_);
lean_dec(v___y_533_);
lean_dec_ref(v___y_532_);
lean_dec(v___y_531_);
lean_dec_ref(v___y_530_);
lean_dec(v___y_529_);
lean_dec_ref(v___y_528_);
lean_dec_ref(v_as_524_);
return v_res_537_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__1(lean_object* v_idx_538_, lean_object* v_a_539_, lean_object* v_as_540_, size_t v_sz_541_, size_t v_i_542_, lean_object* v_b_543_, lean_object* v___y_544_, lean_object* v___y_545_, lean_object* v___y_546_, lean_object* v___y_547_, lean_object* v___y_548_, lean_object* v___y_549_){
_start:
{
lean_object* v___x_551_; 
v___x_551_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__1___redArg(v_idx_538_, v_a_539_, v_as_540_, v_sz_541_, v_i_542_, v_b_543_, v___y_546_, v___y_547_, v___y_548_, v___y_549_);
return v___x_551_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__1___boxed(lean_object* v_idx_552_, lean_object* v_a_553_, lean_object* v_as_554_, lean_object* v_sz_555_, lean_object* v_i_556_, lean_object* v_b_557_, lean_object* v___y_558_, lean_object* v___y_559_, lean_object* v___y_560_, lean_object* v___y_561_, lean_object* v___y_562_, lean_object* v___y_563_, lean_object* v___y_564_){
_start:
{
size_t v_sz_boxed_565_; size_t v_i_boxed_566_; lean_object* v_res_567_; 
v_sz_boxed_565_ = lean_unbox_usize(v_sz_555_);
lean_dec(v_sz_555_);
v_i_boxed_566_ = lean_unbox_usize(v_i_556_);
lean_dec(v_i_556_);
v_res_567_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__1(v_idx_552_, v_a_553_, v_as_554_, v_sz_boxed_565_, v_i_boxed_566_, v_b_557_, v___y_558_, v___y_559_, v___y_560_, v___y_561_, v___y_562_, v___y_563_);
lean_dec(v___y_563_);
lean_dec_ref(v___y_562_);
lean_dec(v___y_561_);
lean_dec_ref(v___y_560_);
lean_dec(v___y_559_);
lean_dec_ref(v___y_558_);
lean_dec_ref(v_as_554_);
return v_res_567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0(lean_object* v_as_580_, size_t v_sz_581_, size_t v_i_582_, lean_object* v_b_583_, lean_object* v___y_584_, lean_object* v___y_585_, lean_object* v___y_586_, lean_object* v___y_587_){
_start:
{
lean_object* v_a_590_; uint8_t v___x_594_; 
v___x_594_ = lean_usize_dec_lt(v_i_582_, v_sz_581_);
if (v___x_594_ == 0)
{
lean_object* v___x_595_; 
v___x_595_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_595_, 0, v_b_583_);
return v___x_595_;
}
else
{
lean_object* v_a_596_; 
v_a_596_ = lean_array_uget(v_as_580_, v_i_582_);
switch(lean_obj_tag(v_a_596_))
{
case 4:
{
lean_object* v_lhs_597_; lean_object* v_rhs_598_; lean_object* v_proof_599_; lean_object* v___x_601_; uint8_t v_isShared_602_; uint8_t v_isSharedCheck_634_; 
v_lhs_597_ = lean_ctor_get(v_a_596_, 0);
v_rhs_598_ = lean_ctor_get(v_a_596_, 1);
v_proof_599_ = lean_ctor_get(v_a_596_, 2);
v_isSharedCheck_634_ = !lean_is_exclusive(v_a_596_);
if (v_isSharedCheck_634_ == 0)
{
v___x_601_ = v_a_596_;
v_isShared_602_ = v_isSharedCheck_634_;
goto v_resetjp_600_;
}
else
{
lean_inc(v_proof_599_);
lean_inc(v_rhs_598_);
lean_inc(v_lhs_597_);
lean_dec(v_a_596_);
v___x_601_ = lean_box(0);
v_isShared_602_ = v_isSharedCheck_634_;
goto v_resetjp_600_;
}
v_resetjp_600_:
{
lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_607_; 
v___x_603_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__1));
v___x_604_ = lean_unsigned_to_nat(1u);
v___x_605_ = lean_mk_empty_array_with_capacity(v___x_604_);
v___x_606_ = lean_array_push(v___x_605_, v_proof_599_);
lean_inc_ref(v___x_606_);
v___x_607_ = l_Lean_Meta_mkAppM(v___x_603_, v___x_606_, v___y_584_, v___y_585_, v___y_586_, v___y_587_);
if (lean_obj_tag(v___x_607_) == 0)
{
lean_object* v_a_608_; lean_object* v___x_609_; lean_object* v___x_610_; 
v_a_608_ = lean_ctor_get(v___x_607_, 0);
lean_inc(v_a_608_);
lean_dec_ref_known(v___x_607_, 1);
v___x_609_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__3));
v___x_610_ = l_Lean_Meta_mkAppM(v___x_609_, v___x_606_, v___y_584_, v___y_585_, v___y_586_, v___y_587_);
if (lean_obj_tag(v___x_610_) == 0)
{
lean_object* v_a_611_; lean_object* v___x_613_; 
v_a_611_ = lean_ctor_get(v___x_610_, 0);
lean_inc(v_a_611_);
lean_dec_ref_known(v___x_610_, 1);
lean_inc(v_rhs_598_);
lean_inc(v_lhs_597_);
if (v_isShared_602_ == 0)
{
lean_ctor_set_tag(v___x_601_, 2);
lean_ctor_set(v___x_601_, 2, v_a_608_);
v___x_613_ = v___x_601_;
goto v_reusejp_612_;
}
else
{
lean_object* v_reuseFailAlloc_617_; 
v_reuseFailAlloc_617_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v_reuseFailAlloc_617_, 0, v_lhs_597_);
lean_ctor_set(v_reuseFailAlloc_617_, 1, v_rhs_598_);
lean_ctor_set(v_reuseFailAlloc_617_, 2, v_a_608_);
v___x_613_ = v_reuseFailAlloc_617_;
goto v_reusejp_612_;
}
v_reusejp_612_:
{
lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; 
v___x_614_ = lean_array_push(v_b_583_, v___x_613_);
v___x_615_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_615_, 0, v_rhs_598_);
lean_ctor_set(v___x_615_, 1, v_lhs_597_);
lean_ctor_set(v___x_615_, 2, v_a_611_);
v___x_616_ = lean_array_push(v___x_614_, v___x_615_);
v_a_590_ = v___x_616_;
goto v___jp_589_;
}
}
else
{
lean_object* v_a_618_; lean_object* v___x_620_; uint8_t v_isShared_621_; uint8_t v_isSharedCheck_625_; 
lean_dec(v_a_608_);
lean_del_object(v___x_601_);
lean_dec(v_rhs_598_);
lean_dec(v_lhs_597_);
lean_dec_ref(v_b_583_);
v_a_618_ = lean_ctor_get(v___x_610_, 0);
v_isSharedCheck_625_ = !lean_is_exclusive(v___x_610_);
if (v_isSharedCheck_625_ == 0)
{
v___x_620_ = v___x_610_;
v_isShared_621_ = v_isSharedCheck_625_;
goto v_resetjp_619_;
}
else
{
lean_inc(v_a_618_);
lean_dec(v___x_610_);
v___x_620_ = lean_box(0);
v_isShared_621_ = v_isSharedCheck_625_;
goto v_resetjp_619_;
}
v_resetjp_619_:
{
lean_object* v___x_623_; 
if (v_isShared_621_ == 0)
{
v___x_623_ = v___x_620_;
goto v_reusejp_622_;
}
else
{
lean_object* v_reuseFailAlloc_624_; 
v_reuseFailAlloc_624_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_624_, 0, v_a_618_);
v___x_623_ = v_reuseFailAlloc_624_;
goto v_reusejp_622_;
}
v_reusejp_622_:
{
return v___x_623_;
}
}
}
}
else
{
lean_object* v_a_626_; lean_object* v___x_628_; uint8_t v_isShared_629_; uint8_t v_isSharedCheck_633_; 
lean_dec_ref(v___x_606_);
lean_del_object(v___x_601_);
lean_dec(v_rhs_598_);
lean_dec(v_lhs_597_);
lean_dec_ref(v_b_583_);
v_a_626_ = lean_ctor_get(v___x_607_, 0);
v_isSharedCheck_633_ = !lean_is_exclusive(v___x_607_);
if (v_isSharedCheck_633_ == 0)
{
v___x_628_ = v___x_607_;
v_isShared_629_ = v_isSharedCheck_633_;
goto v_resetjp_627_;
}
else
{
lean_inc(v_a_626_);
lean_dec(v___x_607_);
v___x_628_ = lean_box(0);
v_isShared_629_ = v_isSharedCheck_633_;
goto v_resetjp_627_;
}
v_resetjp_627_:
{
lean_object* v___x_631_; 
if (v_isShared_629_ == 0)
{
v___x_631_ = v___x_628_;
goto v_reusejp_630_;
}
else
{
lean_object* v_reuseFailAlloc_632_; 
v_reuseFailAlloc_632_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_632_, 0, v_a_626_);
v___x_631_ = v_reuseFailAlloc_632_;
goto v_reusejp_630_;
}
v_reusejp_630_:
{
return v___x_631_;
}
}
}
}
}
case 0:
{
lean_object* v_lhs_635_; lean_object* v_rhs_636_; lean_object* v_proof_637_; lean_object* v___x_639_; uint8_t v_isShared_640_; uint8_t v_isSharedCheck_672_; 
v_lhs_635_ = lean_ctor_get(v_a_596_, 0);
v_rhs_636_ = lean_ctor_get(v_a_596_, 1);
v_proof_637_ = lean_ctor_get(v_a_596_, 2);
v_isSharedCheck_672_ = !lean_is_exclusive(v_a_596_);
if (v_isSharedCheck_672_ == 0)
{
v___x_639_ = v_a_596_;
v_isShared_640_ = v_isSharedCheck_672_;
goto v_resetjp_638_;
}
else
{
lean_inc(v_proof_637_);
lean_inc(v_rhs_636_);
lean_inc(v_lhs_635_);
lean_dec(v_a_596_);
v___x_639_ = lean_box(0);
v_isShared_640_ = v_isSharedCheck_672_;
goto v_resetjp_638_;
}
v_resetjp_638_:
{
lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; 
v___x_641_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__5));
v___x_642_ = lean_unsigned_to_nat(1u);
v___x_643_ = lean_mk_empty_array_with_capacity(v___x_642_);
v___x_644_ = lean_array_push(v___x_643_, v_proof_637_);
lean_inc_ref(v___x_644_);
v___x_645_ = l_Lean_Meta_mkAppM(v___x_641_, v___x_644_, v___y_584_, v___y_585_, v___y_586_, v___y_587_);
if (lean_obj_tag(v___x_645_) == 0)
{
lean_object* v_a_646_; lean_object* v___x_647_; lean_object* v___x_648_; 
v_a_646_ = lean_ctor_get(v___x_645_, 0);
lean_inc(v_a_646_);
lean_dec_ref_known(v___x_645_, 1);
v___x_647_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__7));
v___x_648_ = l_Lean_Meta_mkAppM(v___x_647_, v___x_644_, v___y_584_, v___y_585_, v___y_586_, v___y_587_);
if (lean_obj_tag(v___x_648_) == 0)
{
lean_object* v_a_649_; lean_object* v___x_651_; 
v_a_649_ = lean_ctor_get(v___x_648_, 0);
lean_inc(v_a_649_);
lean_dec_ref_known(v___x_648_, 1);
lean_inc(v_rhs_636_);
lean_inc(v_lhs_635_);
if (v_isShared_640_ == 0)
{
lean_ctor_set_tag(v___x_639_, 2);
lean_ctor_set(v___x_639_, 2, v_a_646_);
v___x_651_ = v___x_639_;
goto v_reusejp_650_;
}
else
{
lean_object* v_reuseFailAlloc_655_; 
v_reuseFailAlloc_655_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v_reuseFailAlloc_655_, 0, v_lhs_635_);
lean_ctor_set(v_reuseFailAlloc_655_, 1, v_rhs_636_);
lean_ctor_set(v_reuseFailAlloc_655_, 2, v_a_646_);
v___x_651_ = v_reuseFailAlloc_655_;
goto v_reusejp_650_;
}
v_reusejp_650_:
{
lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; 
v___x_652_ = lean_array_push(v_b_583_, v___x_651_);
v___x_653_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_653_, 0, v_rhs_636_);
lean_ctor_set(v___x_653_, 1, v_lhs_635_);
lean_ctor_set(v___x_653_, 2, v_a_649_);
v___x_654_ = lean_array_push(v___x_652_, v___x_653_);
v_a_590_ = v___x_654_;
goto v___jp_589_;
}
}
else
{
lean_object* v_a_656_; lean_object* v___x_658_; uint8_t v_isShared_659_; uint8_t v_isSharedCheck_663_; 
lean_dec(v_a_646_);
lean_del_object(v___x_639_);
lean_dec(v_rhs_636_);
lean_dec(v_lhs_635_);
lean_dec_ref(v_b_583_);
v_a_656_ = lean_ctor_get(v___x_648_, 0);
v_isSharedCheck_663_ = !lean_is_exclusive(v___x_648_);
if (v_isSharedCheck_663_ == 0)
{
v___x_658_ = v___x_648_;
v_isShared_659_ = v_isSharedCheck_663_;
goto v_resetjp_657_;
}
else
{
lean_inc(v_a_656_);
lean_dec(v___x_648_);
v___x_658_ = lean_box(0);
v_isShared_659_ = v_isSharedCheck_663_;
goto v_resetjp_657_;
}
v_resetjp_657_:
{
lean_object* v___x_661_; 
if (v_isShared_659_ == 0)
{
v___x_661_ = v___x_658_;
goto v_reusejp_660_;
}
else
{
lean_object* v_reuseFailAlloc_662_; 
v_reuseFailAlloc_662_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_662_, 0, v_a_656_);
v___x_661_ = v_reuseFailAlloc_662_;
goto v_reusejp_660_;
}
v_reusejp_660_:
{
return v___x_661_;
}
}
}
}
else
{
lean_object* v_a_664_; lean_object* v___x_666_; uint8_t v_isShared_667_; uint8_t v_isSharedCheck_671_; 
lean_dec_ref(v___x_644_);
lean_del_object(v___x_639_);
lean_dec(v_rhs_636_);
lean_dec(v_lhs_635_);
lean_dec_ref(v_b_583_);
v_a_664_ = lean_ctor_get(v___x_645_, 0);
v_isSharedCheck_671_ = !lean_is_exclusive(v___x_645_);
if (v_isSharedCheck_671_ == 0)
{
v___x_666_ = v___x_645_;
v_isShared_667_ = v_isSharedCheck_671_;
goto v_resetjp_665_;
}
else
{
lean_inc(v_a_664_);
lean_dec(v___x_645_);
v___x_666_ = lean_box(0);
v_isShared_667_ = v_isSharedCheck_671_;
goto v_resetjp_665_;
}
v_resetjp_665_:
{
lean_object* v___x_669_; 
if (v_isShared_667_ == 0)
{
v___x_669_ = v___x_666_;
goto v_reusejp_668_;
}
else
{
lean_object* v_reuseFailAlloc_670_; 
v_reuseFailAlloc_670_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_670_, 0, v_a_664_);
v___x_669_ = v_reuseFailAlloc_670_;
goto v_reusejp_668_;
}
v_reusejp_668_:
{
return v___x_669_;
}
}
}
}
}
case 1:
{
lean_dec_ref_known(v_a_596_, 3);
v_a_590_ = v_b_583_;
goto v___jp_589_;
}
default: 
{
lean_object* v___x_673_; 
v___x_673_ = lean_array_push(v_b_583_, v_a_596_);
v_a_590_ = v___x_673_;
goto v___jp_589_;
}
}
}
v___jp_589_:
{
size_t v___x_591_; size_t v___x_592_; 
v___x_591_ = ((size_t)1ULL);
v___x_592_ = lean_usize_add(v_i_582_, v___x_591_);
v_i_582_ = v___x_592_;
v_b_583_ = v_a_590_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___boxed(lean_object* v_as_674_, lean_object* v_sz_675_, lean_object* v_i_676_, lean_object* v_b_677_, lean_object* v___y_678_, lean_object* v___y_679_, lean_object* v___y_680_, lean_object* v___y_681_, lean_object* v___y_682_){
_start:
{
size_t v_sz_boxed_683_; size_t v_i_boxed_684_; lean_object* v_res_685_; 
v_sz_boxed_683_ = lean_unbox_usize(v_sz_675_);
lean_dec(v_sz_675_);
v_i_boxed_684_ = lean_unbox_usize(v_i_676_);
lean_dec(v_i_676_);
v_res_685_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0(v_as_674_, v_sz_boxed_683_, v_i_boxed_684_, v_b_677_, v___y_678_, v___y_679_, v___y_680_, v___y_681_);
lean_dec(v___y_681_);
lean_dec_ref(v___y_680_);
lean_dec(v___y_679_);
lean_dec_ref(v___y_678_);
lean_dec_ref(v_as_674_);
return v_res_685_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_preprocessFactsPreorder(lean_object* v_facts_686_, lean_object* v_a_687_, lean_object* v_a_688_, lean_object* v_a_689_, lean_object* v_a_690_){
_start:
{
lean_object* v_res_692_; size_t v_sz_693_; size_t v___x_694_; lean_object* v___x_695_; 
v_res_692_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_replaceBotTop___closed__0));
v_sz_693_ = lean_array_size(v_facts_686_);
v___x_694_ = ((size_t)0ULL);
v___x_695_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0(v_facts_686_, v_sz_693_, v___x_694_, v_res_692_, v_a_687_, v_a_688_, v_a_689_, v_a_690_);
return v___x_695_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_preprocessFactsPreorder___boxed(lean_object* v_facts_696_, lean_object* v_a_697_, lean_object* v_a_698_, lean_object* v_a_699_, lean_object* v_a_700_, lean_object* v_a_701_){
_start:
{
lean_object* v_res_702_; 
v_res_702_ = lp_mathlib_Mathlib_Tactic_Order_preprocessFactsPreorder(v_facts_696_, v_a_697_, v_a_698_, v_a_699_, v_a_700_);
lean_dec(v_a_700_);
lean_dec_ref(v_a_699_);
lean_dec(v_a_698_);
lean_dec_ref(v_a_697_);
lean_dec_ref(v_facts_696_);
return v_res_702_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg(lean_object* v_as_730_, size_t v_sz_731_, size_t v_i_732_, lean_object* v_b_733_, lean_object* v___y_734_, lean_object* v___y_735_, lean_object* v___y_736_, lean_object* v___y_737_, lean_object* v___y_738_){
_start:
{
lean_object* v_a_741_; uint8_t v___x_745_; 
v___x_745_ = lean_usize_dec_lt(v_i_732_, v_sz_731_);
if (v___x_745_ == 0)
{
lean_object* v___x_746_; 
v___x_746_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_746_, 0, v_b_733_);
return v___x_746_;
}
else
{
lean_object* v___x_747_; lean_object* v_a_748_; 
v___x_747_ = l_Lean_instInhabitedExpr;
v_a_748_ = lean_array_uget(v_as_730_, v_i_732_);
switch(lean_obj_tag(v_a_748_))
{
case 4:
{
lean_object* v_lhs_749_; lean_object* v_rhs_750_; lean_object* v_proof_751_; lean_object* v___x_753_; uint8_t v_isShared_754_; uint8_t v_isSharedCheck_786_; 
v_lhs_749_ = lean_ctor_get(v_a_748_, 0);
v_rhs_750_ = lean_ctor_get(v_a_748_, 1);
v_proof_751_ = lean_ctor_get(v_a_748_, 2);
v_isSharedCheck_786_ = !lean_is_exclusive(v_a_748_);
if (v_isSharedCheck_786_ == 0)
{
v___x_753_ = v_a_748_;
v_isShared_754_ = v_isSharedCheck_786_;
goto v_resetjp_752_;
}
else
{
lean_inc(v_proof_751_);
lean_inc(v_rhs_750_);
lean_inc(v_lhs_749_);
lean_dec(v_a_748_);
v___x_753_ = lean_box(0);
v_isShared_754_ = v_isSharedCheck_786_;
goto v_resetjp_752_;
}
v_resetjp_752_:
{
lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; 
v___x_755_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__1));
v___x_756_ = lean_unsigned_to_nat(1u);
v___x_757_ = lean_mk_empty_array_with_capacity(v___x_756_);
v___x_758_ = lean_array_push(v___x_757_, v_proof_751_);
lean_inc_ref(v___x_758_);
v___x_759_ = l_Lean_Meta_mkAppM(v___x_755_, v___x_758_, v___y_735_, v___y_736_, v___y_737_, v___y_738_);
if (lean_obj_tag(v___x_759_) == 0)
{
lean_object* v_a_760_; lean_object* v___x_761_; lean_object* v___x_762_; 
v_a_760_ = lean_ctor_get(v___x_759_, 0);
lean_inc(v_a_760_);
lean_dec_ref_known(v___x_759_, 1);
v___x_761_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__1));
v___x_762_ = l_Lean_Meta_mkAppM(v___x_761_, v___x_758_, v___y_735_, v___y_736_, v___y_737_, v___y_738_);
if (lean_obj_tag(v___x_762_) == 0)
{
lean_object* v_a_763_; lean_object* v___x_765_; 
v_a_763_ = lean_ctor_get(v___x_762_, 0);
lean_inc(v_a_763_);
lean_dec_ref_known(v___x_762_, 1);
lean_inc(v_rhs_750_);
lean_inc(v_lhs_749_);
if (v_isShared_754_ == 0)
{
lean_ctor_set_tag(v___x_753_, 1);
lean_ctor_set(v___x_753_, 2, v_a_760_);
v___x_765_ = v___x_753_;
goto v_reusejp_764_;
}
else
{
lean_object* v_reuseFailAlloc_769_; 
v_reuseFailAlloc_769_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_769_, 0, v_lhs_749_);
lean_ctor_set(v_reuseFailAlloc_769_, 1, v_rhs_750_);
lean_ctor_set(v_reuseFailAlloc_769_, 2, v_a_760_);
v___x_765_ = v_reuseFailAlloc_769_;
goto v_reusejp_764_;
}
v_reusejp_764_:
{
lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; 
v___x_766_ = lean_array_push(v_b_733_, v___x_765_);
v___x_767_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_767_, 0, v_lhs_749_);
lean_ctor_set(v___x_767_, 1, v_rhs_750_);
lean_ctor_set(v___x_767_, 2, v_a_763_);
v___x_768_ = lean_array_push(v___x_766_, v___x_767_);
v_a_741_ = v___x_768_;
goto v___jp_740_;
}
}
else
{
lean_object* v_a_770_; lean_object* v___x_772_; uint8_t v_isShared_773_; uint8_t v_isSharedCheck_777_; 
lean_dec(v_a_760_);
lean_del_object(v___x_753_);
lean_dec(v_rhs_750_);
lean_dec(v_lhs_749_);
lean_dec_ref(v_b_733_);
v_a_770_ = lean_ctor_get(v___x_762_, 0);
v_isSharedCheck_777_ = !lean_is_exclusive(v___x_762_);
if (v_isSharedCheck_777_ == 0)
{
v___x_772_ = v___x_762_;
v_isShared_773_ = v_isSharedCheck_777_;
goto v_resetjp_771_;
}
else
{
lean_inc(v_a_770_);
lean_dec(v___x_762_);
v___x_772_ = lean_box(0);
v_isShared_773_ = v_isSharedCheck_777_;
goto v_resetjp_771_;
}
v_resetjp_771_:
{
lean_object* v___x_775_; 
if (v_isShared_773_ == 0)
{
v___x_775_ = v___x_772_;
goto v_reusejp_774_;
}
else
{
lean_object* v_reuseFailAlloc_776_; 
v_reuseFailAlloc_776_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_776_, 0, v_a_770_);
v___x_775_ = v_reuseFailAlloc_776_;
goto v_reusejp_774_;
}
v_reusejp_774_:
{
return v___x_775_;
}
}
}
}
else
{
lean_object* v_a_778_; lean_object* v___x_780_; uint8_t v_isShared_781_; uint8_t v_isSharedCheck_785_; 
lean_dec_ref(v___x_758_);
lean_del_object(v___x_753_);
lean_dec(v_rhs_750_);
lean_dec(v_lhs_749_);
lean_dec_ref(v_b_733_);
v_a_778_ = lean_ctor_get(v___x_759_, 0);
v_isSharedCheck_785_ = !lean_is_exclusive(v___x_759_);
if (v_isSharedCheck_785_ == 0)
{
v___x_780_ = v___x_759_;
v_isShared_781_ = v_isSharedCheck_785_;
goto v_resetjp_779_;
}
else
{
lean_inc(v_a_778_);
lean_dec(v___x_759_);
v___x_780_ = lean_box(0);
v_isShared_781_ = v_isSharedCheck_785_;
goto v_resetjp_779_;
}
v_resetjp_779_:
{
lean_object* v___x_783_; 
if (v_isShared_781_ == 0)
{
v___x_783_ = v___x_780_;
goto v_reusejp_782_;
}
else
{
lean_object* v_reuseFailAlloc_784_; 
v_reuseFailAlloc_784_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_784_, 0, v_a_778_);
v___x_783_ = v_reuseFailAlloc_784_;
goto v_reusejp_782_;
}
v_reusejp_782_:
{
return v___x_783_;
}
}
}
}
}
case 3:
{
lean_object* v_lhs_787_; lean_object* v_rhs_788_; lean_object* v_proof_789_; lean_object* v___x_791_; uint8_t v_isShared_792_; uint8_t v_isSharedCheck_824_; 
v_lhs_787_ = lean_ctor_get(v_a_748_, 0);
v_rhs_788_ = lean_ctor_get(v_a_748_, 1);
v_proof_789_ = lean_ctor_get(v_a_748_, 2);
v_isSharedCheck_824_ = !lean_is_exclusive(v_a_748_);
if (v_isSharedCheck_824_ == 0)
{
v___x_791_ = v_a_748_;
v_isShared_792_ = v_isSharedCheck_824_;
goto v_resetjp_790_;
}
else
{
lean_inc(v_proof_789_);
lean_inc(v_rhs_788_);
lean_inc(v_lhs_787_);
lean_dec(v_a_748_);
v___x_791_ = lean_box(0);
v_isShared_792_ = v_isSharedCheck_824_;
goto v_resetjp_790_;
}
v_resetjp_790_:
{
lean_object* v___x_793_; lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___x_796_; lean_object* v___x_797_; 
v___x_793_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__3));
v___x_794_ = lean_unsigned_to_nat(1u);
v___x_795_ = lean_mk_empty_array_with_capacity(v___x_794_);
v___x_796_ = lean_array_push(v___x_795_, v_proof_789_);
lean_inc_ref(v___x_796_);
v___x_797_ = l_Lean_Meta_mkAppM(v___x_793_, v___x_796_, v___y_735_, v___y_736_, v___y_737_, v___y_738_);
if (lean_obj_tag(v___x_797_) == 0)
{
lean_object* v_a_798_; lean_object* v___x_799_; lean_object* v___x_800_; 
v_a_798_ = lean_ctor_get(v___x_797_, 0);
lean_inc(v_a_798_);
lean_dec_ref_known(v___x_797_, 1);
v___x_799_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__8));
v___x_800_ = l_Lean_Meta_mkAppM(v___x_799_, v___x_796_, v___y_735_, v___y_736_, v___y_737_, v___y_738_);
if (lean_obj_tag(v___x_800_) == 0)
{
lean_object* v_a_801_; lean_object* v___x_803_; 
v_a_801_ = lean_ctor_get(v___x_800_, 0);
lean_inc(v_a_801_);
lean_dec_ref_known(v___x_800_, 1);
lean_inc(v_rhs_788_);
lean_inc(v_lhs_787_);
if (v_isShared_792_ == 0)
{
lean_ctor_set_tag(v___x_791_, 1);
lean_ctor_set(v___x_791_, 2, v_a_798_);
v___x_803_ = v___x_791_;
goto v_reusejp_802_;
}
else
{
lean_object* v_reuseFailAlloc_807_; 
v_reuseFailAlloc_807_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_807_, 0, v_lhs_787_);
lean_ctor_set(v_reuseFailAlloc_807_, 1, v_rhs_788_);
lean_ctor_set(v_reuseFailAlloc_807_, 2, v_a_798_);
v___x_803_ = v_reuseFailAlloc_807_;
goto v_reusejp_802_;
}
v_reusejp_802_:
{
lean_object* v___x_804_; lean_object* v___x_805_; lean_object* v___x_806_; 
v___x_804_ = lean_array_push(v_b_733_, v___x_803_);
v___x_805_ = lean_alloc_ctor(5, 3, 0);
lean_ctor_set(v___x_805_, 0, v_lhs_787_);
lean_ctor_set(v___x_805_, 1, v_rhs_788_);
lean_ctor_set(v___x_805_, 2, v_a_801_);
v___x_806_ = lean_array_push(v___x_804_, v___x_805_);
v_a_741_ = v___x_806_;
goto v___jp_740_;
}
}
else
{
lean_object* v_a_808_; lean_object* v___x_810_; uint8_t v_isShared_811_; uint8_t v_isSharedCheck_815_; 
lean_dec(v_a_798_);
lean_del_object(v___x_791_);
lean_dec(v_rhs_788_);
lean_dec(v_lhs_787_);
lean_dec_ref(v_b_733_);
v_a_808_ = lean_ctor_get(v___x_800_, 0);
v_isSharedCheck_815_ = !lean_is_exclusive(v___x_800_);
if (v_isSharedCheck_815_ == 0)
{
v___x_810_ = v___x_800_;
v_isShared_811_ = v_isSharedCheck_815_;
goto v_resetjp_809_;
}
else
{
lean_inc(v_a_808_);
lean_dec(v___x_800_);
v___x_810_ = lean_box(0);
v_isShared_811_ = v_isSharedCheck_815_;
goto v_resetjp_809_;
}
v_resetjp_809_:
{
lean_object* v___x_813_; 
if (v_isShared_811_ == 0)
{
v___x_813_ = v___x_810_;
goto v_reusejp_812_;
}
else
{
lean_object* v_reuseFailAlloc_814_; 
v_reuseFailAlloc_814_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_814_, 0, v_a_808_);
v___x_813_ = v_reuseFailAlloc_814_;
goto v_reusejp_812_;
}
v_reusejp_812_:
{
return v___x_813_;
}
}
}
}
else
{
lean_object* v_a_816_; lean_object* v___x_818_; uint8_t v_isShared_819_; uint8_t v_isSharedCheck_823_; 
lean_dec_ref(v___x_796_);
lean_del_object(v___x_791_);
lean_dec(v_rhs_788_);
lean_dec(v_lhs_787_);
lean_dec_ref(v_b_733_);
v_a_816_ = lean_ctor_get(v___x_797_, 0);
v_isSharedCheck_823_ = !lean_is_exclusive(v___x_797_);
if (v_isSharedCheck_823_ == 0)
{
v___x_818_ = v___x_797_;
v_isShared_819_ = v_isSharedCheck_823_;
goto v_resetjp_817_;
}
else
{
lean_inc(v_a_816_);
lean_dec(v___x_797_);
v___x_818_ = lean_box(0);
v_isShared_819_ = v_isSharedCheck_823_;
goto v_resetjp_817_;
}
v_resetjp_817_:
{
lean_object* v___x_821_; 
if (v_isShared_819_ == 0)
{
v___x_821_ = v___x_818_;
goto v_reusejp_820_;
}
else
{
lean_object* v_reuseFailAlloc_822_; 
v_reuseFailAlloc_822_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_822_, 0, v_a_816_);
v___x_821_ = v_reuseFailAlloc_822_;
goto v_reusejp_820_;
}
v_reusejp_820_:
{
return v___x_821_;
}
}
}
}
}
case 0:
{
lean_object* v_lhs_825_; lean_object* v_rhs_826_; lean_object* v_proof_827_; lean_object* v___x_829_; uint8_t v_isShared_830_; uint8_t v_isSharedCheck_862_; 
v_lhs_825_ = lean_ctor_get(v_a_748_, 0);
v_rhs_826_ = lean_ctor_get(v_a_748_, 1);
v_proof_827_ = lean_ctor_get(v_a_748_, 2);
v_isSharedCheck_862_ = !lean_is_exclusive(v_a_748_);
if (v_isSharedCheck_862_ == 0)
{
v___x_829_ = v_a_748_;
v_isShared_830_ = v_isSharedCheck_862_;
goto v_resetjp_828_;
}
else
{
lean_inc(v_proof_827_);
lean_inc(v_rhs_826_);
lean_inc(v_lhs_825_);
lean_dec(v_a_748_);
v___x_829_ = lean_box(0);
v_isShared_830_ = v_isSharedCheck_862_;
goto v_resetjp_828_;
}
v_resetjp_828_:
{
lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v___x_835_; 
v___x_831_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__5));
v___x_832_ = lean_unsigned_to_nat(1u);
v___x_833_ = lean_mk_empty_array_with_capacity(v___x_832_);
v___x_834_ = lean_array_push(v___x_833_, v_proof_827_);
lean_inc_ref(v___x_834_);
v___x_835_ = l_Lean_Meta_mkAppM(v___x_831_, v___x_834_, v___y_735_, v___y_736_, v___y_737_, v___y_738_);
if (lean_obj_tag(v___x_835_) == 0)
{
lean_object* v_a_836_; lean_object* v___x_837_; lean_object* v___x_838_; 
v_a_836_ = lean_ctor_get(v___x_835_, 0);
lean_inc(v_a_836_);
lean_dec_ref_known(v___x_835_, 1);
v___x_837_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__7));
v___x_838_ = l_Lean_Meta_mkAppM(v___x_837_, v___x_834_, v___y_735_, v___y_736_, v___y_737_, v___y_738_);
if (lean_obj_tag(v___x_838_) == 0)
{
lean_object* v_a_839_; lean_object* v___x_841_; 
v_a_839_ = lean_ctor_get(v___x_838_, 0);
lean_inc(v_a_839_);
lean_dec_ref_known(v___x_838_, 1);
lean_inc(v_rhs_826_);
lean_inc(v_lhs_825_);
if (v_isShared_830_ == 0)
{
lean_ctor_set_tag(v___x_829_, 2);
lean_ctor_set(v___x_829_, 2, v_a_836_);
v___x_841_ = v___x_829_;
goto v_reusejp_840_;
}
else
{
lean_object* v_reuseFailAlloc_845_; 
v_reuseFailAlloc_845_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v_reuseFailAlloc_845_, 0, v_lhs_825_);
lean_ctor_set(v_reuseFailAlloc_845_, 1, v_rhs_826_);
lean_ctor_set(v_reuseFailAlloc_845_, 2, v_a_836_);
v___x_841_ = v_reuseFailAlloc_845_;
goto v_reusejp_840_;
}
v_reusejp_840_:
{
lean_object* v___x_842_; lean_object* v___x_843_; lean_object* v___x_844_; 
v___x_842_ = lean_array_push(v_b_733_, v___x_841_);
v___x_843_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_843_, 0, v_rhs_826_);
lean_ctor_set(v___x_843_, 1, v_lhs_825_);
lean_ctor_set(v___x_843_, 2, v_a_839_);
v___x_844_ = lean_array_push(v___x_842_, v___x_843_);
v_a_741_ = v___x_844_;
goto v___jp_740_;
}
}
else
{
lean_object* v_a_846_; lean_object* v___x_848_; uint8_t v_isShared_849_; uint8_t v_isSharedCheck_853_; 
lean_dec(v_a_836_);
lean_del_object(v___x_829_);
lean_dec(v_rhs_826_);
lean_dec(v_lhs_825_);
lean_dec_ref(v_b_733_);
v_a_846_ = lean_ctor_get(v___x_838_, 0);
v_isSharedCheck_853_ = !lean_is_exclusive(v___x_838_);
if (v_isSharedCheck_853_ == 0)
{
v___x_848_ = v___x_838_;
v_isShared_849_ = v_isSharedCheck_853_;
goto v_resetjp_847_;
}
else
{
lean_inc(v_a_846_);
lean_dec(v___x_838_);
v___x_848_ = lean_box(0);
v_isShared_849_ = v_isSharedCheck_853_;
goto v_resetjp_847_;
}
v_resetjp_847_:
{
lean_object* v___x_851_; 
if (v_isShared_849_ == 0)
{
v___x_851_ = v___x_848_;
goto v_reusejp_850_;
}
else
{
lean_object* v_reuseFailAlloc_852_; 
v_reuseFailAlloc_852_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_852_, 0, v_a_846_);
v___x_851_ = v_reuseFailAlloc_852_;
goto v_reusejp_850_;
}
v_reusejp_850_:
{
return v___x_851_;
}
}
}
}
else
{
lean_object* v_a_854_; lean_object* v___x_856_; uint8_t v_isShared_857_; uint8_t v_isSharedCheck_861_; 
lean_dec_ref(v___x_834_);
lean_del_object(v___x_829_);
lean_dec(v_rhs_826_);
lean_dec(v_lhs_825_);
lean_dec_ref(v_b_733_);
v_a_854_ = lean_ctor_get(v___x_835_, 0);
v_isSharedCheck_861_ = !lean_is_exclusive(v___x_835_);
if (v_isSharedCheck_861_ == 0)
{
v___x_856_ = v___x_835_;
v_isShared_857_ = v_isSharedCheck_861_;
goto v_resetjp_855_;
}
else
{
lean_inc(v_a_854_);
lean_dec(v___x_835_);
v___x_856_ = lean_box(0);
v_isShared_857_ = v_isSharedCheck_861_;
goto v_resetjp_855_;
}
v_resetjp_855_:
{
lean_object* v___x_859_; 
if (v_isShared_857_ == 0)
{
v___x_859_ = v___x_856_;
goto v_reusejp_858_;
}
else
{
lean_object* v_reuseFailAlloc_860_; 
v_reuseFailAlloc_860_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_860_, 0, v_a_854_);
v___x_859_ = v_reuseFailAlloc_860_;
goto v_reusejp_858_;
}
v_reusejp_858_:
{
return v___x_859_;
}
}
}
}
}
case 9:
{
lean_object* v_lhs_863_; lean_object* v_rhs_864_; lean_object* v_res_865_; lean_object* v___x_866_; lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; lean_object* v___x_870_; lean_object* v___x_871_; lean_object* v___x_872_; lean_object* v___x_873_; lean_object* v___x_874_; lean_object* v___x_875_; lean_object* v___x_876_; 
v_lhs_863_ = lean_ctor_get(v_a_748_, 0);
v_rhs_864_ = lean_ctor_get(v_a_748_, 1);
v_res_865_ = lean_ctor_get(v_a_748_, 2);
v___x_866_ = lean_st_ref_get(v___y_734_);
v___x_867_ = lean_st_ref_get(v___y_734_);
v___x_868_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__10));
v___x_869_ = lean_array_get(v___x_747_, v___x_866_, v_lhs_863_);
lean_dec(v___x_866_);
v___x_870_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_870_, 0, v___x_869_);
v___x_871_ = lean_array_get(v___x_747_, v___x_867_, v_rhs_864_);
lean_dec(v___x_867_);
v___x_872_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_872_, 0, v___x_871_);
v___x_873_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__3);
v___x_874_ = lean_array_push(v___x_873_, v___x_870_);
v___x_875_ = lean_array_push(v___x_874_, v___x_872_);
v___x_876_ = l_Lean_Meta_mkAppOptM(v___x_868_, v___x_875_, v___y_735_, v___y_736_, v___y_737_, v___y_738_);
if (lean_obj_tag(v___x_876_) == 0)
{
lean_object* v_a_877_; lean_object* v___x_878_; lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v___x_881_; lean_object* v___x_882_; lean_object* v___x_883_; lean_object* v___x_884_; lean_object* v___x_885_; lean_object* v___x_886_; lean_object* v___x_887_; 
v_a_877_ = lean_ctor_get(v___x_876_, 0);
lean_inc(v_a_877_);
lean_dec_ref_known(v___x_876_, 1);
v___x_878_ = lean_st_ref_get(v___y_734_);
v___x_879_ = lean_st_ref_get(v___y_734_);
v___x_880_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__12));
v___x_881_ = lean_array_get(v___x_747_, v___x_878_, v_lhs_863_);
lean_dec(v___x_878_);
v___x_882_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_882_, 0, v___x_881_);
v___x_883_ = lean_array_get(v___x_747_, v___x_879_, v_rhs_864_);
lean_dec(v___x_879_);
v___x_884_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_884_, 0, v___x_883_);
v___x_885_ = lean_array_push(v___x_873_, v___x_882_);
v___x_886_ = lean_array_push(v___x_885_, v___x_884_);
v___x_887_ = l_Lean_Meta_mkAppOptM(v___x_880_, v___x_886_, v___y_735_, v___y_736_, v___y_737_, v___y_738_);
if (lean_obj_tag(v___x_887_) == 0)
{
lean_object* v_a_888_; lean_object* v___x_889_; lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; lean_object* v___x_893_; 
v_a_888_ = lean_ctor_get(v___x_887_, 0);
lean_inc(v_a_888_);
lean_dec_ref_known(v___x_887_, 1);
lean_inc_n(v_res_865_, 2);
lean_inc(v_lhs_863_);
v___x_889_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_889_, 0, v_lhs_863_);
lean_ctor_set(v___x_889_, 1, v_res_865_);
lean_ctor_set(v___x_889_, 2, v_a_877_);
v___x_890_ = lean_array_push(v_b_733_, v___x_889_);
lean_inc(v_rhs_864_);
v___x_891_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_891_, 0, v_rhs_864_);
lean_ctor_set(v___x_891_, 1, v_res_865_);
lean_ctor_set(v___x_891_, 2, v_a_888_);
v___x_892_ = lean_array_push(v___x_890_, v___x_891_);
v___x_893_ = lean_array_push(v___x_892_, v_a_748_);
v_a_741_ = v___x_893_;
goto v___jp_740_;
}
else
{
lean_object* v_a_894_; lean_object* v___x_896_; uint8_t v_isShared_897_; uint8_t v_isSharedCheck_901_; 
lean_dec(v_a_877_);
lean_dec_ref_known(v_a_748_, 3);
lean_dec_ref(v_b_733_);
v_a_894_ = lean_ctor_get(v___x_887_, 0);
v_isSharedCheck_901_ = !lean_is_exclusive(v___x_887_);
if (v_isSharedCheck_901_ == 0)
{
v___x_896_ = v___x_887_;
v_isShared_897_ = v_isSharedCheck_901_;
goto v_resetjp_895_;
}
else
{
lean_inc(v_a_894_);
lean_dec(v___x_887_);
v___x_896_ = lean_box(0);
v_isShared_897_ = v_isSharedCheck_901_;
goto v_resetjp_895_;
}
v_resetjp_895_:
{
lean_object* v___x_899_; 
if (v_isShared_897_ == 0)
{
v___x_899_ = v___x_896_;
goto v_reusejp_898_;
}
else
{
lean_object* v_reuseFailAlloc_900_; 
v_reuseFailAlloc_900_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_900_, 0, v_a_894_);
v___x_899_ = v_reuseFailAlloc_900_;
goto v_reusejp_898_;
}
v_reusejp_898_:
{
return v___x_899_;
}
}
}
}
else
{
lean_object* v_a_902_; lean_object* v___x_904_; uint8_t v_isShared_905_; uint8_t v_isSharedCheck_909_; 
lean_dec_ref_known(v_a_748_, 3);
lean_dec_ref(v_b_733_);
v_a_902_ = lean_ctor_get(v___x_876_, 0);
v_isSharedCheck_909_ = !lean_is_exclusive(v___x_876_);
if (v_isSharedCheck_909_ == 0)
{
v___x_904_ = v___x_876_;
v_isShared_905_ = v_isSharedCheck_909_;
goto v_resetjp_903_;
}
else
{
lean_inc(v_a_902_);
lean_dec(v___x_876_);
v___x_904_ = lean_box(0);
v_isShared_905_ = v_isSharedCheck_909_;
goto v_resetjp_903_;
}
v_resetjp_903_:
{
lean_object* v___x_907_; 
if (v_isShared_905_ == 0)
{
v___x_907_ = v___x_904_;
goto v_reusejp_906_;
}
else
{
lean_object* v_reuseFailAlloc_908_; 
v_reuseFailAlloc_908_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_908_, 0, v_a_902_);
v___x_907_ = v_reuseFailAlloc_908_;
goto v_reusejp_906_;
}
v_reusejp_906_:
{
return v___x_907_;
}
}
}
}
case 8:
{
lean_object* v_lhs_910_; lean_object* v_rhs_911_; lean_object* v_res_912_; lean_object* v___x_913_; lean_object* v___x_914_; lean_object* v___x_915_; lean_object* v___x_916_; lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v___x_921_; lean_object* v___x_922_; lean_object* v___x_923_; 
v_lhs_910_ = lean_ctor_get(v_a_748_, 0);
v_rhs_911_ = lean_ctor_get(v_a_748_, 1);
v_res_912_ = lean_ctor_get(v_a_748_, 2);
v___x_913_ = lean_st_ref_get(v___y_734_);
v___x_914_ = lean_st_ref_get(v___y_734_);
v___x_915_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__14));
v___x_916_ = lean_array_get(v___x_747_, v___x_913_, v_lhs_910_);
lean_dec(v___x_913_);
v___x_917_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_917_, 0, v___x_916_);
v___x_918_ = lean_array_get(v___x_747_, v___x_914_, v_rhs_911_);
lean_dec(v___x_914_);
v___x_919_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_919_, 0, v___x_918_);
v___x_920_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__3);
v___x_921_ = lean_array_push(v___x_920_, v___x_917_);
v___x_922_ = lean_array_push(v___x_921_, v___x_919_);
v___x_923_ = l_Lean_Meta_mkAppOptM(v___x_915_, v___x_922_, v___y_735_, v___y_736_, v___y_737_, v___y_738_);
if (lean_obj_tag(v___x_923_) == 0)
{
lean_object* v_a_924_; lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_930_; lean_object* v___x_931_; lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; 
v_a_924_ = lean_ctor_get(v___x_923_, 0);
lean_inc(v_a_924_);
lean_dec_ref_known(v___x_923_, 1);
v___x_925_ = lean_st_ref_get(v___y_734_);
v___x_926_ = lean_st_ref_get(v___y_734_);
v___x_927_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__16));
v___x_928_ = lean_array_get(v___x_747_, v___x_925_, v_lhs_910_);
lean_dec(v___x_925_);
v___x_929_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_929_, 0, v___x_928_);
v___x_930_ = lean_array_get(v___x_747_, v___x_926_, v_rhs_911_);
lean_dec(v___x_926_);
v___x_931_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_931_, 0, v___x_930_);
v___x_932_ = lean_array_push(v___x_920_, v___x_929_);
v___x_933_ = lean_array_push(v___x_932_, v___x_931_);
v___x_934_ = l_Lean_Meta_mkAppOptM(v___x_927_, v___x_933_, v___y_735_, v___y_736_, v___y_737_, v___y_738_);
if (lean_obj_tag(v___x_934_) == 0)
{
lean_object* v_a_935_; lean_object* v___x_936_; lean_object* v___x_937_; lean_object* v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; 
v_a_935_ = lean_ctor_get(v___x_934_, 0);
lean_inc(v_a_935_);
lean_dec_ref_known(v___x_934_, 1);
lean_inc(v_lhs_910_);
lean_inc_n(v_res_912_, 2);
v___x_936_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_936_, 0, v_res_912_);
lean_ctor_set(v___x_936_, 1, v_lhs_910_);
lean_ctor_set(v___x_936_, 2, v_a_924_);
v___x_937_ = lean_array_push(v_b_733_, v___x_936_);
lean_inc(v_rhs_911_);
v___x_938_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_938_, 0, v_res_912_);
lean_ctor_set(v___x_938_, 1, v_rhs_911_);
lean_ctor_set(v___x_938_, 2, v_a_935_);
v___x_939_ = lean_array_push(v___x_937_, v___x_938_);
v___x_940_ = lean_array_push(v___x_939_, v_a_748_);
v_a_741_ = v___x_940_;
goto v___jp_740_;
}
else
{
lean_object* v_a_941_; lean_object* v___x_943_; uint8_t v_isShared_944_; uint8_t v_isSharedCheck_948_; 
lean_dec(v_a_924_);
lean_dec_ref_known(v_a_748_, 3);
lean_dec_ref(v_b_733_);
v_a_941_ = lean_ctor_get(v___x_934_, 0);
v_isSharedCheck_948_ = !lean_is_exclusive(v___x_934_);
if (v_isSharedCheck_948_ == 0)
{
v___x_943_ = v___x_934_;
v_isShared_944_ = v_isSharedCheck_948_;
goto v_resetjp_942_;
}
else
{
lean_inc(v_a_941_);
lean_dec(v___x_934_);
v___x_943_ = lean_box(0);
v_isShared_944_ = v_isSharedCheck_948_;
goto v_resetjp_942_;
}
v_resetjp_942_:
{
lean_object* v___x_946_; 
if (v_isShared_944_ == 0)
{
v___x_946_ = v___x_943_;
goto v_reusejp_945_;
}
else
{
lean_object* v_reuseFailAlloc_947_; 
v_reuseFailAlloc_947_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_947_, 0, v_a_941_);
v___x_946_ = v_reuseFailAlloc_947_;
goto v_reusejp_945_;
}
v_reusejp_945_:
{
return v___x_946_;
}
}
}
}
else
{
lean_object* v_a_949_; lean_object* v___x_951_; uint8_t v_isShared_952_; uint8_t v_isSharedCheck_956_; 
lean_dec_ref_known(v_a_748_, 3);
lean_dec_ref(v_b_733_);
v_a_949_ = lean_ctor_get(v___x_923_, 0);
v_isSharedCheck_956_ = !lean_is_exclusive(v___x_923_);
if (v_isSharedCheck_956_ == 0)
{
v___x_951_ = v___x_923_;
v_isShared_952_ = v_isSharedCheck_956_;
goto v_resetjp_950_;
}
else
{
lean_inc(v_a_949_);
lean_dec(v___x_923_);
v___x_951_ = lean_box(0);
v_isShared_952_ = v_isSharedCheck_956_;
goto v_resetjp_950_;
}
v_resetjp_950_:
{
lean_object* v___x_954_; 
if (v_isShared_952_ == 0)
{
v___x_954_ = v___x_951_;
goto v_reusejp_953_;
}
else
{
lean_object* v_reuseFailAlloc_955_; 
v_reuseFailAlloc_955_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_955_, 0, v_a_949_);
v___x_954_ = v_reuseFailAlloc_955_;
goto v_reusejp_953_;
}
v_reusejp_953_:
{
return v___x_954_;
}
}
}
}
default: 
{
lean_object* v___x_957_; 
v___x_957_ = lean_array_push(v_b_733_, v_a_748_);
v_a_741_ = v___x_957_;
goto v___jp_740_;
}
}
}
v___jp_740_:
{
size_t v___x_742_; size_t v___x_743_; 
v___x_742_ = ((size_t)1ULL);
v___x_743_ = lean_usize_add(v_i_732_, v___x_742_);
v_i_732_ = v___x_743_;
v_b_733_ = v_a_741_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___boxed(lean_object* v_as_958_, lean_object* v_sz_959_, lean_object* v_i_960_, lean_object* v_b_961_, lean_object* v___y_962_, lean_object* v___y_963_, lean_object* v___y_964_, lean_object* v___y_965_, lean_object* v___y_966_, lean_object* v___y_967_){
_start:
{
size_t v_sz_boxed_968_; size_t v_i_boxed_969_; lean_object* v_res_970_; 
v_sz_boxed_968_ = lean_unbox_usize(v_sz_959_);
lean_dec(v_sz_959_);
v_i_boxed_969_ = lean_unbox_usize(v_i_960_);
lean_dec(v_i_960_);
v_res_970_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg(v_as_958_, v_sz_boxed_968_, v_i_boxed_969_, v_b_961_, v___y_962_, v___y_963_, v___y_964_, v___y_965_, v___y_966_);
lean_dec(v___y_966_);
lean_dec_ref(v___y_965_);
lean_dec(v___y_964_);
lean_dec_ref(v___y_963_);
lean_dec(v___y_962_);
lean_dec_ref(v_as_958_);
return v_res_970_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_preprocessFactsPartial(lean_object* v_facts_971_, lean_object* v_a_972_, lean_object* v_a_973_, lean_object* v_a_974_, lean_object* v_a_975_, lean_object* v_a_976_, lean_object* v_a_977_){
_start:
{
lean_object* v_res_979_; size_t v_sz_980_; size_t v___x_981_; lean_object* v___x_982_; 
v_res_979_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_replaceBotTop___closed__0));
v_sz_980_ = lean_array_size(v_facts_971_);
v___x_981_ = ((size_t)0ULL);
v___x_982_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg(v_facts_971_, v_sz_980_, v___x_981_, v_res_979_, v_a_973_, v_a_974_, v_a_975_, v_a_976_, v_a_977_);
return v___x_982_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_preprocessFactsPartial___boxed(lean_object* v_facts_983_, lean_object* v_a_984_, lean_object* v_a_985_, lean_object* v_a_986_, lean_object* v_a_987_, lean_object* v_a_988_, lean_object* v_a_989_, lean_object* v_a_990_){
_start:
{
lean_object* v_res_991_; 
v_res_991_ = lp_mathlib_Mathlib_Tactic_Order_preprocessFactsPartial(v_facts_983_, v_a_984_, v_a_985_, v_a_986_, v_a_987_, v_a_988_, v_a_989_);
lean_dec(v_a_989_);
lean_dec_ref(v_a_988_);
lean_dec(v_a_987_);
lean_dec_ref(v_a_986_);
lean_dec(v_a_985_);
lean_dec_ref(v_a_984_);
lean_dec_ref(v_facts_983_);
return v_res_991_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0(lean_object* v_as_992_, size_t v_sz_993_, size_t v_i_994_, lean_object* v_b_995_, lean_object* v___y_996_, lean_object* v___y_997_, lean_object* v___y_998_, lean_object* v___y_999_, lean_object* v___y_1000_, lean_object* v___y_1001_){
_start:
{
lean_object* v___x_1003_; 
v___x_1003_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg(v_as_992_, v_sz_993_, v_i_994_, v_b_995_, v___y_997_, v___y_998_, v___y_999_, v___y_1000_, v___y_1001_);
return v___x_1003_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___boxed(lean_object* v_as_1004_, lean_object* v_sz_1005_, lean_object* v_i_1006_, lean_object* v_b_1007_, lean_object* v___y_1008_, lean_object* v___y_1009_, lean_object* v___y_1010_, lean_object* v___y_1011_, lean_object* v___y_1012_, lean_object* v___y_1013_, lean_object* v___y_1014_){
_start:
{
size_t v_sz_boxed_1015_; size_t v_i_boxed_1016_; lean_object* v_res_1017_; 
v_sz_boxed_1015_ = lean_unbox_usize(v_sz_1005_);
lean_dec(v_sz_1005_);
v_i_boxed_1016_ = lean_unbox_usize(v_i_1006_);
lean_dec(v_i_1006_);
v_res_1017_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0(v_as_1004_, v_sz_boxed_1015_, v_i_boxed_1016_, v_b_1007_, v___y_1008_, v___y_1009_, v___y_1010_, v___y_1011_, v___y_1012_, v___y_1013_);
lean_dec(v___y_1013_);
lean_dec_ref(v___y_1012_);
lean_dec(v___y_1011_);
lean_dec_ref(v___y_1010_);
lean_dec(v___y_1009_);
lean_dec_ref(v___y_1008_);
lean_dec_ref(v_as_1004_);
return v_res_1017_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0___redArg(lean_object* v_as_1024_, size_t v_sz_1025_, size_t v_i_1026_, lean_object* v_b_1027_, lean_object* v___y_1028_, lean_object* v___y_1029_, lean_object* v___y_1030_, lean_object* v___y_1031_, lean_object* v___y_1032_){
_start:
{
lean_object* v_a_1035_; uint8_t v___x_1039_; 
v___x_1039_ = lean_usize_dec_lt(v_i_1026_, v_sz_1025_);
if (v___x_1039_ == 0)
{
lean_object* v___x_1040_; 
v___x_1040_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1040_, 0, v_b_1027_);
return v___x_1040_;
}
else
{
lean_object* v___x_1041_; lean_object* v_a_1042_; 
v___x_1041_ = l_Lean_instInhabitedExpr;
v_a_1042_ = lean_array_uget(v_as_1024_, v_i_1026_);
switch(lean_obj_tag(v_a_1042_))
{
case 4:
{
lean_object* v_lhs_1043_; lean_object* v_rhs_1044_; lean_object* v_proof_1045_; lean_object* v___x_1047_; uint8_t v_isShared_1048_; uint8_t v_isSharedCheck_1080_; 
v_lhs_1043_ = lean_ctor_get(v_a_1042_, 0);
v_rhs_1044_ = lean_ctor_get(v_a_1042_, 1);
v_proof_1045_ = lean_ctor_get(v_a_1042_, 2);
v_isSharedCheck_1080_ = !lean_is_exclusive(v_a_1042_);
if (v_isSharedCheck_1080_ == 0)
{
v___x_1047_ = v_a_1042_;
v_isShared_1048_ = v_isSharedCheck_1080_;
goto v_resetjp_1046_;
}
else
{
lean_inc(v_proof_1045_);
lean_inc(v_rhs_1044_);
lean_inc(v_lhs_1043_);
lean_dec(v_a_1042_);
v___x_1047_ = lean_box(0);
v_isShared_1048_ = v_isSharedCheck_1080_;
goto v_resetjp_1046_;
}
v_resetjp_1046_:
{
lean_object* v___x_1049_; lean_object* v___x_1050_; lean_object* v___x_1051_; lean_object* v___x_1052_; lean_object* v___x_1053_; 
v___x_1049_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__1));
v___x_1050_ = lean_unsigned_to_nat(1u);
v___x_1051_ = lean_mk_empty_array_with_capacity(v___x_1050_);
v___x_1052_ = lean_array_push(v___x_1051_, v_proof_1045_);
lean_inc_ref(v___x_1052_);
v___x_1053_ = l_Lean_Meta_mkAppM(v___x_1049_, v___x_1052_, v___y_1029_, v___y_1030_, v___y_1031_, v___y_1032_);
if (lean_obj_tag(v___x_1053_) == 0)
{
lean_object* v_a_1054_; lean_object* v___x_1055_; lean_object* v___x_1056_; 
v_a_1054_ = lean_ctor_get(v___x_1053_, 0);
lean_inc(v_a_1054_);
lean_dec_ref_known(v___x_1053_, 1);
v___x_1055_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__1));
v___x_1056_ = l_Lean_Meta_mkAppM(v___x_1055_, v___x_1052_, v___y_1029_, v___y_1030_, v___y_1031_, v___y_1032_);
if (lean_obj_tag(v___x_1056_) == 0)
{
lean_object* v_a_1057_; lean_object* v___x_1059_; 
v_a_1057_ = lean_ctor_get(v___x_1056_, 0);
lean_inc(v_a_1057_);
lean_dec_ref_known(v___x_1056_, 1);
lean_inc(v_rhs_1044_);
lean_inc(v_lhs_1043_);
if (v_isShared_1048_ == 0)
{
lean_ctor_set_tag(v___x_1047_, 1);
lean_ctor_set(v___x_1047_, 2, v_a_1054_);
v___x_1059_ = v___x_1047_;
goto v_reusejp_1058_;
}
else
{
lean_object* v_reuseFailAlloc_1063_; 
v_reuseFailAlloc_1063_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1063_, 0, v_lhs_1043_);
lean_ctor_set(v_reuseFailAlloc_1063_, 1, v_rhs_1044_);
lean_ctor_set(v_reuseFailAlloc_1063_, 2, v_a_1054_);
v___x_1059_ = v_reuseFailAlloc_1063_;
goto v_reusejp_1058_;
}
v_reusejp_1058_:
{
lean_object* v___x_1060_; lean_object* v___x_1061_; lean_object* v___x_1062_; 
v___x_1060_ = lean_array_push(v_b_1027_, v___x_1059_);
v___x_1061_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1061_, 0, v_lhs_1043_);
lean_ctor_set(v___x_1061_, 1, v_rhs_1044_);
lean_ctor_set(v___x_1061_, 2, v_a_1057_);
v___x_1062_ = lean_array_push(v___x_1060_, v___x_1061_);
v_a_1035_ = v___x_1062_;
goto v___jp_1034_;
}
}
else
{
lean_object* v_a_1064_; lean_object* v___x_1066_; uint8_t v_isShared_1067_; uint8_t v_isSharedCheck_1071_; 
lean_dec(v_a_1054_);
lean_del_object(v___x_1047_);
lean_dec(v_rhs_1044_);
lean_dec(v_lhs_1043_);
lean_dec_ref(v_b_1027_);
v_a_1064_ = lean_ctor_get(v___x_1056_, 0);
v_isSharedCheck_1071_ = !lean_is_exclusive(v___x_1056_);
if (v_isSharedCheck_1071_ == 0)
{
v___x_1066_ = v___x_1056_;
v_isShared_1067_ = v_isSharedCheck_1071_;
goto v_resetjp_1065_;
}
else
{
lean_inc(v_a_1064_);
lean_dec(v___x_1056_);
v___x_1066_ = lean_box(0);
v_isShared_1067_ = v_isSharedCheck_1071_;
goto v_resetjp_1065_;
}
v_resetjp_1065_:
{
lean_object* v___x_1069_; 
if (v_isShared_1067_ == 0)
{
v___x_1069_ = v___x_1066_;
goto v_reusejp_1068_;
}
else
{
lean_object* v_reuseFailAlloc_1070_; 
v_reuseFailAlloc_1070_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1070_, 0, v_a_1064_);
v___x_1069_ = v_reuseFailAlloc_1070_;
goto v_reusejp_1068_;
}
v_reusejp_1068_:
{
return v___x_1069_;
}
}
}
}
else
{
lean_object* v_a_1072_; lean_object* v___x_1074_; uint8_t v_isShared_1075_; uint8_t v_isSharedCheck_1079_; 
lean_dec_ref(v___x_1052_);
lean_del_object(v___x_1047_);
lean_dec(v_rhs_1044_);
lean_dec(v_lhs_1043_);
lean_dec_ref(v_b_1027_);
v_a_1072_ = lean_ctor_get(v___x_1053_, 0);
v_isSharedCheck_1079_ = !lean_is_exclusive(v___x_1053_);
if (v_isSharedCheck_1079_ == 0)
{
v___x_1074_ = v___x_1053_;
v_isShared_1075_ = v_isSharedCheck_1079_;
goto v_resetjp_1073_;
}
else
{
lean_inc(v_a_1072_);
lean_dec(v___x_1053_);
v___x_1074_ = lean_box(0);
v_isShared_1075_ = v_isSharedCheck_1079_;
goto v_resetjp_1073_;
}
v_resetjp_1073_:
{
lean_object* v___x_1077_; 
if (v_isShared_1075_ == 0)
{
v___x_1077_ = v___x_1074_;
goto v_reusejp_1076_;
}
else
{
lean_object* v_reuseFailAlloc_1078_; 
v_reuseFailAlloc_1078_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1078_, 0, v_a_1072_);
v___x_1077_ = v_reuseFailAlloc_1078_;
goto v_reusejp_1076_;
}
v_reusejp_1076_:
{
return v___x_1077_;
}
}
}
}
}
case 3:
{
lean_object* v_lhs_1081_; lean_object* v_rhs_1082_; lean_object* v_proof_1083_; lean_object* v___x_1085_; uint8_t v_isShared_1086_; uint8_t v_isSharedCheck_1118_; 
v_lhs_1081_ = lean_ctor_get(v_a_1042_, 0);
v_rhs_1082_ = lean_ctor_get(v_a_1042_, 1);
v_proof_1083_ = lean_ctor_get(v_a_1042_, 2);
v_isSharedCheck_1118_ = !lean_is_exclusive(v_a_1042_);
if (v_isSharedCheck_1118_ == 0)
{
v___x_1085_ = v_a_1042_;
v_isShared_1086_ = v_isSharedCheck_1118_;
goto v_resetjp_1084_;
}
else
{
lean_inc(v_proof_1083_);
lean_inc(v_rhs_1082_);
lean_inc(v_lhs_1081_);
lean_dec(v_a_1042_);
v___x_1085_ = lean_box(0);
v_isShared_1086_ = v_isSharedCheck_1118_;
goto v_resetjp_1084_;
}
v_resetjp_1084_:
{
lean_object* v___x_1087_; lean_object* v___x_1088_; lean_object* v___x_1089_; lean_object* v___x_1090_; lean_object* v___x_1091_; 
v___x_1087_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__3));
v___x_1088_ = lean_unsigned_to_nat(1u);
v___x_1089_ = lean_mk_empty_array_with_capacity(v___x_1088_);
v___x_1090_ = lean_array_push(v___x_1089_, v_proof_1083_);
lean_inc_ref(v___x_1090_);
v___x_1091_ = l_Lean_Meta_mkAppM(v___x_1087_, v___x_1090_, v___y_1029_, v___y_1030_, v___y_1031_, v___y_1032_);
if (lean_obj_tag(v___x_1091_) == 0)
{
lean_object* v_a_1092_; lean_object* v___x_1093_; lean_object* v___x_1094_; 
v_a_1092_ = lean_ctor_get(v___x_1091_, 0);
lean_inc(v_a_1092_);
lean_dec_ref_known(v___x_1091_, 1);
v___x_1093_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0___redArg___closed__1));
v___x_1094_ = l_Lean_Meta_mkAppM(v___x_1093_, v___x_1090_, v___y_1029_, v___y_1030_, v___y_1031_, v___y_1032_);
if (lean_obj_tag(v___x_1094_) == 0)
{
lean_object* v_a_1095_; lean_object* v___x_1097_; 
v_a_1095_ = lean_ctor_get(v___x_1094_, 0);
lean_inc(v_a_1095_);
lean_dec_ref_known(v___x_1094_, 1);
lean_inc(v_rhs_1082_);
lean_inc(v_lhs_1081_);
if (v_isShared_1086_ == 0)
{
lean_ctor_set_tag(v___x_1085_, 1);
lean_ctor_set(v___x_1085_, 2, v_a_1092_);
v___x_1097_ = v___x_1085_;
goto v_reusejp_1096_;
}
else
{
lean_object* v_reuseFailAlloc_1101_; 
v_reuseFailAlloc_1101_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1101_, 0, v_lhs_1081_);
lean_ctor_set(v_reuseFailAlloc_1101_, 1, v_rhs_1082_);
lean_ctor_set(v_reuseFailAlloc_1101_, 2, v_a_1092_);
v___x_1097_ = v_reuseFailAlloc_1101_;
goto v_reusejp_1096_;
}
v_reusejp_1096_:
{
lean_object* v___x_1098_; lean_object* v___x_1099_; lean_object* v___x_1100_; 
v___x_1098_ = lean_array_push(v_b_1027_, v___x_1097_);
v___x_1099_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1099_, 0, v_rhs_1082_);
lean_ctor_set(v___x_1099_, 1, v_lhs_1081_);
lean_ctor_set(v___x_1099_, 2, v_a_1095_);
v___x_1100_ = lean_array_push(v___x_1098_, v___x_1099_);
v_a_1035_ = v___x_1100_;
goto v___jp_1034_;
}
}
else
{
lean_object* v_a_1102_; lean_object* v___x_1104_; uint8_t v_isShared_1105_; uint8_t v_isSharedCheck_1109_; 
lean_dec(v_a_1092_);
lean_del_object(v___x_1085_);
lean_dec(v_rhs_1082_);
lean_dec(v_lhs_1081_);
lean_dec_ref(v_b_1027_);
v_a_1102_ = lean_ctor_get(v___x_1094_, 0);
v_isSharedCheck_1109_ = !lean_is_exclusive(v___x_1094_);
if (v_isSharedCheck_1109_ == 0)
{
v___x_1104_ = v___x_1094_;
v_isShared_1105_ = v_isSharedCheck_1109_;
goto v_resetjp_1103_;
}
else
{
lean_inc(v_a_1102_);
lean_dec(v___x_1094_);
v___x_1104_ = lean_box(0);
v_isShared_1105_ = v_isSharedCheck_1109_;
goto v_resetjp_1103_;
}
v_resetjp_1103_:
{
lean_object* v___x_1107_; 
if (v_isShared_1105_ == 0)
{
v___x_1107_ = v___x_1104_;
goto v_reusejp_1106_;
}
else
{
lean_object* v_reuseFailAlloc_1108_; 
v_reuseFailAlloc_1108_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1108_, 0, v_a_1102_);
v___x_1107_ = v_reuseFailAlloc_1108_;
goto v_reusejp_1106_;
}
v_reusejp_1106_:
{
return v___x_1107_;
}
}
}
}
else
{
lean_object* v_a_1110_; lean_object* v___x_1112_; uint8_t v_isShared_1113_; uint8_t v_isSharedCheck_1117_; 
lean_dec_ref(v___x_1090_);
lean_del_object(v___x_1085_);
lean_dec(v_rhs_1082_);
lean_dec(v_lhs_1081_);
lean_dec_ref(v_b_1027_);
v_a_1110_ = lean_ctor_get(v___x_1091_, 0);
v_isSharedCheck_1117_ = !lean_is_exclusive(v___x_1091_);
if (v_isSharedCheck_1117_ == 0)
{
v___x_1112_ = v___x_1091_;
v_isShared_1113_ = v_isSharedCheck_1117_;
goto v_resetjp_1111_;
}
else
{
lean_inc(v_a_1110_);
lean_dec(v___x_1091_);
v___x_1112_ = lean_box(0);
v_isShared_1113_ = v_isSharedCheck_1117_;
goto v_resetjp_1111_;
}
v_resetjp_1111_:
{
lean_object* v___x_1115_; 
if (v_isShared_1113_ == 0)
{
v___x_1115_ = v___x_1112_;
goto v_reusejp_1114_;
}
else
{
lean_object* v_reuseFailAlloc_1116_; 
v_reuseFailAlloc_1116_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1116_, 0, v_a_1110_);
v___x_1115_ = v_reuseFailAlloc_1116_;
goto v_reusejp_1114_;
}
v_reusejp_1114_:
{
return v___x_1115_;
}
}
}
}
}
case 5:
{
lean_object* v_lhs_1119_; lean_object* v_rhs_1120_; lean_object* v_proof_1121_; lean_object* v___x_1123_; uint8_t v_isShared_1124_; uint8_t v_isSharedCheck_1143_; 
v_lhs_1119_ = lean_ctor_get(v_a_1042_, 0);
v_rhs_1120_ = lean_ctor_get(v_a_1042_, 1);
v_proof_1121_ = lean_ctor_get(v_a_1042_, 2);
v_isSharedCheck_1143_ = !lean_is_exclusive(v_a_1042_);
if (v_isSharedCheck_1143_ == 0)
{
v___x_1123_ = v_a_1042_;
v_isShared_1124_ = v_isSharedCheck_1143_;
goto v_resetjp_1122_;
}
else
{
lean_inc(v_proof_1121_);
lean_inc(v_rhs_1120_);
lean_inc(v_lhs_1119_);
lean_dec(v_a_1042_);
v___x_1123_ = lean_box(0);
v_isShared_1124_ = v_isSharedCheck_1143_;
goto v_resetjp_1122_;
}
v_resetjp_1122_:
{
lean_object* v___x_1125_; lean_object* v___x_1126_; lean_object* v___x_1127_; lean_object* v___x_1128_; lean_object* v___x_1129_; 
v___x_1125_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0___redArg___closed__3));
v___x_1126_ = lean_unsigned_to_nat(1u);
v___x_1127_ = lean_mk_empty_array_with_capacity(v___x_1126_);
v___x_1128_ = lean_array_push(v___x_1127_, v_proof_1121_);
v___x_1129_ = l_Lean_Meta_mkAppM(v___x_1125_, v___x_1128_, v___y_1029_, v___y_1030_, v___y_1031_, v___y_1032_);
if (lean_obj_tag(v___x_1129_) == 0)
{
lean_object* v_a_1130_; lean_object* v___x_1132_; 
v_a_1130_ = lean_ctor_get(v___x_1129_, 0);
lean_inc(v_a_1130_);
lean_dec_ref_known(v___x_1129_, 1);
if (v_isShared_1124_ == 0)
{
lean_ctor_set_tag(v___x_1123_, 2);
lean_ctor_set(v___x_1123_, 2, v_a_1130_);
lean_ctor_set(v___x_1123_, 1, v_lhs_1119_);
lean_ctor_set(v___x_1123_, 0, v_rhs_1120_);
v___x_1132_ = v___x_1123_;
goto v_reusejp_1131_;
}
else
{
lean_object* v_reuseFailAlloc_1134_; 
v_reuseFailAlloc_1134_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1134_, 0, v_rhs_1120_);
lean_ctor_set(v_reuseFailAlloc_1134_, 1, v_lhs_1119_);
lean_ctor_set(v_reuseFailAlloc_1134_, 2, v_a_1130_);
v___x_1132_ = v_reuseFailAlloc_1134_;
goto v_reusejp_1131_;
}
v_reusejp_1131_:
{
lean_object* v___x_1133_; 
v___x_1133_ = lean_array_push(v_b_1027_, v___x_1132_);
v_a_1035_ = v___x_1133_;
goto v___jp_1034_;
}
}
else
{
lean_object* v_a_1135_; lean_object* v___x_1137_; uint8_t v_isShared_1138_; uint8_t v_isSharedCheck_1142_; 
lean_del_object(v___x_1123_);
lean_dec(v_rhs_1120_);
lean_dec(v_lhs_1119_);
lean_dec_ref(v_b_1027_);
v_a_1135_ = lean_ctor_get(v___x_1129_, 0);
v_isSharedCheck_1142_ = !lean_is_exclusive(v___x_1129_);
if (v_isSharedCheck_1142_ == 0)
{
v___x_1137_ = v___x_1129_;
v_isShared_1138_ = v_isSharedCheck_1142_;
goto v_resetjp_1136_;
}
else
{
lean_inc(v_a_1135_);
lean_dec(v___x_1129_);
v___x_1137_ = lean_box(0);
v_isShared_1138_ = v_isSharedCheck_1142_;
goto v_resetjp_1136_;
}
v_resetjp_1136_:
{
lean_object* v___x_1140_; 
if (v_isShared_1138_ == 0)
{
v___x_1140_ = v___x_1137_;
goto v_reusejp_1139_;
}
else
{
lean_object* v_reuseFailAlloc_1141_; 
v_reuseFailAlloc_1141_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1141_, 0, v_a_1135_);
v___x_1140_ = v_reuseFailAlloc_1141_;
goto v_reusejp_1139_;
}
v_reusejp_1139_:
{
return v___x_1140_;
}
}
}
}
}
case 0:
{
lean_object* v_lhs_1144_; lean_object* v_rhs_1145_; lean_object* v_proof_1146_; lean_object* v___x_1148_; uint8_t v_isShared_1149_; uint8_t v_isSharedCheck_1181_; 
v_lhs_1144_ = lean_ctor_get(v_a_1042_, 0);
v_rhs_1145_ = lean_ctor_get(v_a_1042_, 1);
v_proof_1146_ = lean_ctor_get(v_a_1042_, 2);
v_isSharedCheck_1181_ = !lean_is_exclusive(v_a_1042_);
if (v_isSharedCheck_1181_ == 0)
{
v___x_1148_ = v_a_1042_;
v_isShared_1149_ = v_isSharedCheck_1181_;
goto v_resetjp_1147_;
}
else
{
lean_inc(v_proof_1146_);
lean_inc(v_rhs_1145_);
lean_inc(v_lhs_1144_);
lean_dec(v_a_1042_);
v___x_1148_ = lean_box(0);
v_isShared_1149_ = v_isSharedCheck_1181_;
goto v_resetjp_1147_;
}
v_resetjp_1147_:
{
lean_object* v___x_1150_; lean_object* v___x_1151_; lean_object* v___x_1152_; lean_object* v___x_1153_; lean_object* v___x_1154_; 
v___x_1150_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__5));
v___x_1151_ = lean_unsigned_to_nat(1u);
v___x_1152_ = lean_mk_empty_array_with_capacity(v___x_1151_);
v___x_1153_ = lean_array_push(v___x_1152_, v_proof_1146_);
lean_inc_ref(v___x_1153_);
v___x_1154_ = l_Lean_Meta_mkAppM(v___x_1150_, v___x_1153_, v___y_1029_, v___y_1030_, v___y_1031_, v___y_1032_);
if (lean_obj_tag(v___x_1154_) == 0)
{
lean_object* v_a_1155_; lean_object* v___x_1156_; lean_object* v___x_1157_; 
v_a_1155_ = lean_ctor_get(v___x_1154_, 0);
lean_inc(v_a_1155_);
lean_dec_ref_known(v___x_1154_, 1);
v___x_1156_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPreorder_spec__0___closed__7));
v___x_1157_ = l_Lean_Meta_mkAppM(v___x_1156_, v___x_1153_, v___y_1029_, v___y_1030_, v___y_1031_, v___y_1032_);
if (lean_obj_tag(v___x_1157_) == 0)
{
lean_object* v_a_1158_; lean_object* v___x_1160_; 
v_a_1158_ = lean_ctor_get(v___x_1157_, 0);
lean_inc(v_a_1158_);
lean_dec_ref_known(v___x_1157_, 1);
lean_inc(v_rhs_1145_);
lean_inc(v_lhs_1144_);
if (v_isShared_1149_ == 0)
{
lean_ctor_set_tag(v___x_1148_, 2);
lean_ctor_set(v___x_1148_, 2, v_a_1155_);
v___x_1160_ = v___x_1148_;
goto v_reusejp_1159_;
}
else
{
lean_object* v_reuseFailAlloc_1164_; 
v_reuseFailAlloc_1164_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1164_, 0, v_lhs_1144_);
lean_ctor_set(v_reuseFailAlloc_1164_, 1, v_rhs_1145_);
lean_ctor_set(v_reuseFailAlloc_1164_, 2, v_a_1155_);
v___x_1160_ = v_reuseFailAlloc_1164_;
goto v_reusejp_1159_;
}
v_reusejp_1159_:
{
lean_object* v___x_1161_; lean_object* v___x_1162_; lean_object* v___x_1163_; 
v___x_1161_ = lean_array_push(v_b_1027_, v___x_1160_);
v___x_1162_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1162_, 0, v_rhs_1145_);
lean_ctor_set(v___x_1162_, 1, v_lhs_1144_);
lean_ctor_set(v___x_1162_, 2, v_a_1158_);
v___x_1163_ = lean_array_push(v___x_1161_, v___x_1162_);
v_a_1035_ = v___x_1163_;
goto v___jp_1034_;
}
}
else
{
lean_object* v_a_1165_; lean_object* v___x_1167_; uint8_t v_isShared_1168_; uint8_t v_isSharedCheck_1172_; 
lean_dec(v_a_1155_);
lean_del_object(v___x_1148_);
lean_dec(v_rhs_1145_);
lean_dec(v_lhs_1144_);
lean_dec_ref(v_b_1027_);
v_a_1165_ = lean_ctor_get(v___x_1157_, 0);
v_isSharedCheck_1172_ = !lean_is_exclusive(v___x_1157_);
if (v_isSharedCheck_1172_ == 0)
{
v___x_1167_ = v___x_1157_;
v_isShared_1168_ = v_isSharedCheck_1172_;
goto v_resetjp_1166_;
}
else
{
lean_inc(v_a_1165_);
lean_dec(v___x_1157_);
v___x_1167_ = lean_box(0);
v_isShared_1168_ = v_isSharedCheck_1172_;
goto v_resetjp_1166_;
}
v_resetjp_1166_:
{
lean_object* v___x_1170_; 
if (v_isShared_1168_ == 0)
{
v___x_1170_ = v___x_1167_;
goto v_reusejp_1169_;
}
else
{
lean_object* v_reuseFailAlloc_1171_; 
v_reuseFailAlloc_1171_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1171_, 0, v_a_1165_);
v___x_1170_ = v_reuseFailAlloc_1171_;
goto v_reusejp_1169_;
}
v_reusejp_1169_:
{
return v___x_1170_;
}
}
}
}
else
{
lean_object* v_a_1173_; lean_object* v___x_1175_; uint8_t v_isShared_1176_; uint8_t v_isSharedCheck_1180_; 
lean_dec_ref(v___x_1153_);
lean_del_object(v___x_1148_);
lean_dec(v_rhs_1145_);
lean_dec(v_lhs_1144_);
lean_dec_ref(v_b_1027_);
v_a_1173_ = lean_ctor_get(v___x_1154_, 0);
v_isSharedCheck_1180_ = !lean_is_exclusive(v___x_1154_);
if (v_isSharedCheck_1180_ == 0)
{
v___x_1175_ = v___x_1154_;
v_isShared_1176_ = v_isSharedCheck_1180_;
goto v_resetjp_1174_;
}
else
{
lean_inc(v_a_1173_);
lean_dec(v___x_1154_);
v___x_1175_ = lean_box(0);
v_isShared_1176_ = v_isSharedCheck_1180_;
goto v_resetjp_1174_;
}
v_resetjp_1174_:
{
lean_object* v___x_1178_; 
if (v_isShared_1176_ == 0)
{
v___x_1178_ = v___x_1175_;
goto v_reusejp_1177_;
}
else
{
lean_object* v_reuseFailAlloc_1179_; 
v_reuseFailAlloc_1179_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1179_, 0, v_a_1173_);
v___x_1178_ = v_reuseFailAlloc_1179_;
goto v_reusejp_1177_;
}
v_reusejp_1177_:
{
return v___x_1178_;
}
}
}
}
}
case 9:
{
lean_object* v_lhs_1182_; lean_object* v_rhs_1183_; lean_object* v_res_1184_; lean_object* v___x_1185_; lean_object* v___x_1186_; lean_object* v___x_1187_; lean_object* v___x_1188_; lean_object* v___x_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; lean_object* v___x_1192_; lean_object* v___x_1193_; lean_object* v___x_1194_; lean_object* v___x_1195_; 
v_lhs_1182_ = lean_ctor_get(v_a_1042_, 0);
v_rhs_1183_ = lean_ctor_get(v_a_1042_, 1);
v_res_1184_ = lean_ctor_get(v_a_1042_, 2);
v___x_1185_ = lean_st_ref_get(v___y_1028_);
v___x_1186_ = lean_st_ref_get(v___y_1028_);
v___x_1187_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__10));
v___x_1188_ = lean_array_get(v___x_1041_, v___x_1185_, v_lhs_1182_);
lean_dec(v___x_1185_);
v___x_1189_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1189_, 0, v___x_1188_);
v___x_1190_ = lean_array_get(v___x_1041_, v___x_1186_, v_rhs_1183_);
lean_dec(v___x_1186_);
v___x_1191_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1191_, 0, v___x_1190_);
v___x_1192_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__3);
v___x_1193_ = lean_array_push(v___x_1192_, v___x_1189_);
v___x_1194_ = lean_array_push(v___x_1193_, v___x_1191_);
v___x_1195_ = l_Lean_Meta_mkAppOptM(v___x_1187_, v___x_1194_, v___y_1029_, v___y_1030_, v___y_1031_, v___y_1032_);
if (lean_obj_tag(v___x_1195_) == 0)
{
lean_object* v_a_1196_; lean_object* v___x_1197_; lean_object* v___x_1198_; lean_object* v___x_1199_; lean_object* v___x_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; lean_object* v___x_1203_; lean_object* v___x_1204_; lean_object* v___x_1205_; lean_object* v___x_1206_; 
v_a_1196_ = lean_ctor_get(v___x_1195_, 0);
lean_inc(v_a_1196_);
lean_dec_ref_known(v___x_1195_, 1);
v___x_1197_ = lean_st_ref_get(v___y_1028_);
v___x_1198_ = lean_st_ref_get(v___y_1028_);
v___x_1199_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__12));
v___x_1200_ = lean_array_get(v___x_1041_, v___x_1197_, v_lhs_1182_);
lean_dec(v___x_1197_);
v___x_1201_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1201_, 0, v___x_1200_);
v___x_1202_ = lean_array_get(v___x_1041_, v___x_1198_, v_rhs_1183_);
lean_dec(v___x_1198_);
v___x_1203_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1203_, 0, v___x_1202_);
v___x_1204_ = lean_array_push(v___x_1192_, v___x_1201_);
v___x_1205_ = lean_array_push(v___x_1204_, v___x_1203_);
v___x_1206_ = l_Lean_Meta_mkAppOptM(v___x_1199_, v___x_1205_, v___y_1029_, v___y_1030_, v___y_1031_, v___y_1032_);
if (lean_obj_tag(v___x_1206_) == 0)
{
lean_object* v_a_1207_; lean_object* v___x_1208_; lean_object* v___x_1209_; lean_object* v___x_1210_; lean_object* v___x_1211_; lean_object* v___x_1212_; 
v_a_1207_ = lean_ctor_get(v___x_1206_, 0);
lean_inc(v_a_1207_);
lean_dec_ref_known(v___x_1206_, 1);
lean_inc_n(v_res_1184_, 2);
lean_inc(v_lhs_1182_);
v___x_1208_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1208_, 0, v_lhs_1182_);
lean_ctor_set(v___x_1208_, 1, v_res_1184_);
lean_ctor_set(v___x_1208_, 2, v_a_1196_);
v___x_1209_ = lean_array_push(v_b_1027_, v___x_1208_);
lean_inc(v_rhs_1183_);
v___x_1210_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1210_, 0, v_rhs_1183_);
lean_ctor_set(v___x_1210_, 1, v_res_1184_);
lean_ctor_set(v___x_1210_, 2, v_a_1207_);
v___x_1211_ = lean_array_push(v___x_1209_, v___x_1210_);
v___x_1212_ = lean_array_push(v___x_1211_, v_a_1042_);
v_a_1035_ = v___x_1212_;
goto v___jp_1034_;
}
else
{
lean_object* v_a_1213_; lean_object* v___x_1215_; uint8_t v_isShared_1216_; uint8_t v_isSharedCheck_1220_; 
lean_dec(v_a_1196_);
lean_dec_ref_known(v_a_1042_, 3);
lean_dec_ref(v_b_1027_);
v_a_1213_ = lean_ctor_get(v___x_1206_, 0);
v_isSharedCheck_1220_ = !lean_is_exclusive(v___x_1206_);
if (v_isSharedCheck_1220_ == 0)
{
v___x_1215_ = v___x_1206_;
v_isShared_1216_ = v_isSharedCheck_1220_;
goto v_resetjp_1214_;
}
else
{
lean_inc(v_a_1213_);
lean_dec(v___x_1206_);
v___x_1215_ = lean_box(0);
v_isShared_1216_ = v_isSharedCheck_1220_;
goto v_resetjp_1214_;
}
v_resetjp_1214_:
{
lean_object* v___x_1218_; 
if (v_isShared_1216_ == 0)
{
v___x_1218_ = v___x_1215_;
goto v_reusejp_1217_;
}
else
{
lean_object* v_reuseFailAlloc_1219_; 
v_reuseFailAlloc_1219_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1219_, 0, v_a_1213_);
v___x_1218_ = v_reuseFailAlloc_1219_;
goto v_reusejp_1217_;
}
v_reusejp_1217_:
{
return v___x_1218_;
}
}
}
}
else
{
lean_object* v_a_1221_; lean_object* v___x_1223_; uint8_t v_isShared_1224_; uint8_t v_isSharedCheck_1228_; 
lean_dec_ref_known(v_a_1042_, 3);
lean_dec_ref(v_b_1027_);
v_a_1221_ = lean_ctor_get(v___x_1195_, 0);
v_isSharedCheck_1228_ = !lean_is_exclusive(v___x_1195_);
if (v_isSharedCheck_1228_ == 0)
{
v___x_1223_ = v___x_1195_;
v_isShared_1224_ = v_isSharedCheck_1228_;
goto v_resetjp_1222_;
}
else
{
lean_inc(v_a_1221_);
lean_dec(v___x_1195_);
v___x_1223_ = lean_box(0);
v_isShared_1224_ = v_isSharedCheck_1228_;
goto v_resetjp_1222_;
}
v_resetjp_1222_:
{
lean_object* v___x_1226_; 
if (v_isShared_1224_ == 0)
{
v___x_1226_ = v___x_1223_;
goto v_reusejp_1225_;
}
else
{
lean_object* v_reuseFailAlloc_1227_; 
v_reuseFailAlloc_1227_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1227_, 0, v_a_1221_);
v___x_1226_ = v_reuseFailAlloc_1227_;
goto v_reusejp_1225_;
}
v_reusejp_1225_:
{
return v___x_1226_;
}
}
}
}
case 8:
{
lean_object* v_lhs_1229_; lean_object* v_rhs_1230_; lean_object* v_res_1231_; lean_object* v___x_1232_; lean_object* v___x_1233_; lean_object* v___x_1234_; lean_object* v___x_1235_; lean_object* v___x_1236_; lean_object* v___x_1237_; lean_object* v___x_1238_; lean_object* v___x_1239_; lean_object* v___x_1240_; lean_object* v___x_1241_; lean_object* v___x_1242_; 
v_lhs_1229_ = lean_ctor_get(v_a_1042_, 0);
v_rhs_1230_ = lean_ctor_get(v_a_1042_, 1);
v_res_1231_ = lean_ctor_get(v_a_1042_, 2);
v___x_1232_ = lean_st_ref_get(v___y_1028_);
v___x_1233_ = lean_st_ref_get(v___y_1028_);
v___x_1234_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__14));
v___x_1235_ = lean_array_get(v___x_1041_, v___x_1232_, v_lhs_1229_);
lean_dec(v___x_1232_);
v___x_1236_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1236_, 0, v___x_1235_);
v___x_1237_ = lean_array_get(v___x_1041_, v___x_1233_, v_rhs_1230_);
lean_dec(v___x_1233_);
v___x_1238_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1238_, 0, v___x_1237_);
v___x_1239_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__3, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__3_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_replaceBotTop_spec__0___redArg___closed__3);
v___x_1240_ = lean_array_push(v___x_1239_, v___x_1236_);
v___x_1241_ = lean_array_push(v___x_1240_, v___x_1238_);
v___x_1242_ = l_Lean_Meta_mkAppOptM(v___x_1234_, v___x_1241_, v___y_1029_, v___y_1030_, v___y_1031_, v___y_1032_);
if (lean_obj_tag(v___x_1242_) == 0)
{
lean_object* v_a_1243_; lean_object* v___x_1244_; lean_object* v___x_1245_; lean_object* v___x_1246_; lean_object* v___x_1247_; lean_object* v___x_1248_; lean_object* v___x_1249_; lean_object* v___x_1250_; lean_object* v___x_1251_; lean_object* v___x_1252_; lean_object* v___x_1253_; 
v_a_1243_ = lean_ctor_get(v___x_1242_, 0);
lean_inc(v_a_1243_);
lean_dec_ref_known(v___x_1242_, 1);
v___x_1244_ = lean_st_ref_get(v___y_1028_);
v___x_1245_ = lean_st_ref_get(v___y_1028_);
v___x_1246_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsPartial_spec__0___redArg___closed__16));
v___x_1247_ = lean_array_get(v___x_1041_, v___x_1244_, v_lhs_1229_);
lean_dec(v___x_1244_);
v___x_1248_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1248_, 0, v___x_1247_);
v___x_1249_ = lean_array_get(v___x_1041_, v___x_1245_, v_rhs_1230_);
lean_dec(v___x_1245_);
v___x_1250_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1250_, 0, v___x_1249_);
v___x_1251_ = lean_array_push(v___x_1239_, v___x_1248_);
v___x_1252_ = lean_array_push(v___x_1251_, v___x_1250_);
v___x_1253_ = l_Lean_Meta_mkAppOptM(v___x_1246_, v___x_1252_, v___y_1029_, v___y_1030_, v___y_1031_, v___y_1032_);
if (lean_obj_tag(v___x_1253_) == 0)
{
lean_object* v_a_1254_; lean_object* v___x_1255_; lean_object* v___x_1256_; lean_object* v___x_1257_; lean_object* v___x_1258_; lean_object* v___x_1259_; 
v_a_1254_ = lean_ctor_get(v___x_1253_, 0);
lean_inc(v_a_1254_);
lean_dec_ref_known(v___x_1253_, 1);
lean_inc(v_lhs_1229_);
lean_inc_n(v_res_1231_, 2);
v___x_1255_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1255_, 0, v_res_1231_);
lean_ctor_set(v___x_1255_, 1, v_lhs_1229_);
lean_ctor_set(v___x_1255_, 2, v_a_1243_);
v___x_1256_ = lean_array_push(v_b_1027_, v___x_1255_);
lean_inc(v_rhs_1230_);
v___x_1257_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1257_, 0, v_res_1231_);
lean_ctor_set(v___x_1257_, 1, v_rhs_1230_);
lean_ctor_set(v___x_1257_, 2, v_a_1254_);
v___x_1258_ = lean_array_push(v___x_1256_, v___x_1257_);
v___x_1259_ = lean_array_push(v___x_1258_, v_a_1042_);
v_a_1035_ = v___x_1259_;
goto v___jp_1034_;
}
else
{
lean_object* v_a_1260_; lean_object* v___x_1262_; uint8_t v_isShared_1263_; uint8_t v_isSharedCheck_1267_; 
lean_dec(v_a_1243_);
lean_dec_ref_known(v_a_1042_, 3);
lean_dec_ref(v_b_1027_);
v_a_1260_ = lean_ctor_get(v___x_1253_, 0);
v_isSharedCheck_1267_ = !lean_is_exclusive(v___x_1253_);
if (v_isSharedCheck_1267_ == 0)
{
v___x_1262_ = v___x_1253_;
v_isShared_1263_ = v_isSharedCheck_1267_;
goto v_resetjp_1261_;
}
else
{
lean_inc(v_a_1260_);
lean_dec(v___x_1253_);
v___x_1262_ = lean_box(0);
v_isShared_1263_ = v_isSharedCheck_1267_;
goto v_resetjp_1261_;
}
v_resetjp_1261_:
{
lean_object* v___x_1265_; 
if (v_isShared_1263_ == 0)
{
v___x_1265_ = v___x_1262_;
goto v_reusejp_1264_;
}
else
{
lean_object* v_reuseFailAlloc_1266_; 
v_reuseFailAlloc_1266_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1266_, 0, v_a_1260_);
v___x_1265_ = v_reuseFailAlloc_1266_;
goto v_reusejp_1264_;
}
v_reusejp_1264_:
{
return v___x_1265_;
}
}
}
}
else
{
lean_object* v_a_1268_; lean_object* v___x_1270_; uint8_t v_isShared_1271_; uint8_t v_isSharedCheck_1275_; 
lean_dec_ref_known(v_a_1042_, 3);
lean_dec_ref(v_b_1027_);
v_a_1268_ = lean_ctor_get(v___x_1242_, 0);
v_isSharedCheck_1275_ = !lean_is_exclusive(v___x_1242_);
if (v_isSharedCheck_1275_ == 0)
{
v___x_1270_ = v___x_1242_;
v_isShared_1271_ = v_isSharedCheck_1275_;
goto v_resetjp_1269_;
}
else
{
lean_inc(v_a_1268_);
lean_dec(v___x_1242_);
v___x_1270_ = lean_box(0);
v_isShared_1271_ = v_isSharedCheck_1275_;
goto v_resetjp_1269_;
}
v_resetjp_1269_:
{
lean_object* v___x_1273_; 
if (v_isShared_1271_ == 0)
{
v___x_1273_ = v___x_1270_;
goto v_reusejp_1272_;
}
else
{
lean_object* v_reuseFailAlloc_1274_; 
v_reuseFailAlloc_1274_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1274_, 0, v_a_1268_);
v___x_1273_ = v_reuseFailAlloc_1274_;
goto v_reusejp_1272_;
}
v_reusejp_1272_:
{
return v___x_1273_;
}
}
}
}
default: 
{
lean_object* v___x_1276_; 
v___x_1276_ = lean_array_push(v_b_1027_, v_a_1042_);
v_a_1035_ = v___x_1276_;
goto v___jp_1034_;
}
}
}
v___jp_1034_:
{
size_t v___x_1036_; size_t v___x_1037_; 
v___x_1036_ = ((size_t)1ULL);
v___x_1037_ = lean_usize_add(v_i_1026_, v___x_1036_);
v_i_1026_ = v___x_1037_;
v_b_1027_ = v_a_1035_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0___redArg___boxed(lean_object* v_as_1277_, lean_object* v_sz_1278_, lean_object* v_i_1279_, lean_object* v_b_1280_, lean_object* v___y_1281_, lean_object* v___y_1282_, lean_object* v___y_1283_, lean_object* v___y_1284_, lean_object* v___y_1285_, lean_object* v___y_1286_){
_start:
{
size_t v_sz_boxed_1287_; size_t v_i_boxed_1288_; lean_object* v_res_1289_; 
v_sz_boxed_1287_ = lean_unbox_usize(v_sz_1278_);
lean_dec(v_sz_1278_);
v_i_boxed_1288_ = lean_unbox_usize(v_i_1279_);
lean_dec(v_i_1279_);
v_res_1289_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0___redArg(v_as_1277_, v_sz_boxed_1287_, v_i_boxed_1288_, v_b_1280_, v___y_1281_, v___y_1282_, v___y_1283_, v___y_1284_, v___y_1285_);
lean_dec(v___y_1285_);
lean_dec_ref(v___y_1284_);
lean_dec(v___y_1283_);
lean_dec_ref(v___y_1282_);
lean_dec(v___y_1281_);
lean_dec_ref(v_as_1277_);
return v_res_1289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_preprocessFactsLinear(lean_object* v_facts_1290_, lean_object* v_a_1291_, lean_object* v_a_1292_, lean_object* v_a_1293_, lean_object* v_a_1294_, lean_object* v_a_1295_, lean_object* v_a_1296_){
_start:
{
lean_object* v_res_1298_; size_t v_sz_1299_; size_t v___x_1300_; lean_object* v___x_1301_; 
v_res_1298_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_replaceBotTop___closed__0));
v_sz_1299_ = lean_array_size(v_facts_1290_);
v___x_1300_ = ((size_t)0ULL);
v___x_1301_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0___redArg(v_facts_1290_, v_sz_1299_, v___x_1300_, v_res_1298_, v_a_1292_, v_a_1293_, v_a_1294_, v_a_1295_, v_a_1296_);
return v___x_1301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_preprocessFactsLinear___boxed(lean_object* v_facts_1302_, lean_object* v_a_1303_, lean_object* v_a_1304_, lean_object* v_a_1305_, lean_object* v_a_1306_, lean_object* v_a_1307_, lean_object* v_a_1308_, lean_object* v_a_1309_){
_start:
{
lean_object* v_res_1310_; 
v_res_1310_ = lp_mathlib_Mathlib_Tactic_Order_preprocessFactsLinear(v_facts_1302_, v_a_1303_, v_a_1304_, v_a_1305_, v_a_1306_, v_a_1307_, v_a_1308_);
lean_dec(v_a_1308_);
lean_dec_ref(v_a_1307_);
lean_dec(v_a_1306_);
lean_dec_ref(v_a_1305_);
lean_dec(v_a_1304_);
lean_dec_ref(v_a_1303_);
lean_dec_ref(v_facts_1302_);
return v_res_1310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0(lean_object* v_as_1311_, size_t v_sz_1312_, size_t v_i_1313_, lean_object* v_b_1314_, lean_object* v___y_1315_, lean_object* v___y_1316_, lean_object* v___y_1317_, lean_object* v___y_1318_, lean_object* v___y_1319_, lean_object* v___y_1320_){
_start:
{
lean_object* v___x_1322_; 
v___x_1322_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0___redArg(v_as_1311_, v_sz_1312_, v_i_1313_, v_b_1314_, v___y_1316_, v___y_1317_, v___y_1318_, v___y_1319_, v___y_1320_);
return v___x_1322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0___boxed(lean_object* v_as_1323_, lean_object* v_sz_1324_, lean_object* v_i_1325_, lean_object* v_b_1326_, lean_object* v___y_1327_, lean_object* v___y_1328_, lean_object* v___y_1329_, lean_object* v___y_1330_, lean_object* v___y_1331_, lean_object* v___y_1332_, lean_object* v___y_1333_){
_start:
{
size_t v_sz_boxed_1334_; size_t v_i_boxed_1335_; lean_object* v_res_1336_; 
v_sz_boxed_1334_ = lean_unbox_usize(v_sz_1324_);
lean_dec(v_sz_1324_);
v_i_boxed_1335_ = lean_unbox_usize(v_i_1325_);
lean_dec(v_i_1325_);
v_res_1336_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_preprocessFactsLinear_spec__0(v_as_1323_, v_sz_boxed_1334_, v_i_boxed_1335_, v_b_1326_, v___y_1327_, v___y_1328_, v___y_1329_, v___y_1330_, v___y_1331_, v___y_1332_);
lean_dec(v___y_1332_);
lean_dec_ref(v___y_1331_);
lean_dec(v___y_1330_);
lean_dec_ref(v___y_1329_);
lean_dec(v___y_1328_);
lean_dec_ref(v___y_1327_);
lean_dec_ref(v_as_1323_);
return v_res_1336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_preprocessFacts(lean_object* v_facts_1337_, uint8_t v_orderType_1338_, lean_object* v_a_1339_, lean_object* v_a_1340_, lean_object* v_a_1341_, lean_object* v_a_1342_, lean_object* v_a_1343_, lean_object* v_a_1344_){
_start:
{
switch(v_orderType_1338_)
{
case 0:
{
lean_object* v___x_1346_; 
v___x_1346_ = lp_mathlib_Mathlib_Tactic_Order_preprocessFactsLinear(v_facts_1337_, v_a_1339_, v_a_1340_, v_a_1341_, v_a_1342_, v_a_1343_, v_a_1344_);
return v___x_1346_;
}
case 1:
{
lean_object* v___x_1347_; 
v___x_1347_ = lp_mathlib_Mathlib_Tactic_Order_preprocessFactsPartial(v_facts_1337_, v_a_1339_, v_a_1340_, v_a_1341_, v_a_1342_, v_a_1343_, v_a_1344_);
return v___x_1347_;
}
default: 
{
lean_object* v___x_1348_; 
v___x_1348_ = lp_mathlib_Mathlib_Tactic_Order_preprocessFactsPreorder(v_facts_1337_, v_a_1341_, v_a_1342_, v_a_1343_, v_a_1344_);
return v___x_1348_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_preprocessFacts___boxed(lean_object* v_facts_1349_, lean_object* v_orderType_1350_, lean_object* v_a_1351_, lean_object* v_a_1352_, lean_object* v_a_1353_, lean_object* v_a_1354_, lean_object* v_a_1355_, lean_object* v_a_1356_, lean_object* v_a_1357_){
_start:
{
uint8_t v_orderType_boxed_1358_; lean_object* v_res_1359_; 
v_orderType_boxed_1358_ = lean_unbox(v_orderType_1350_);
v_res_1359_ = lp_mathlib_Mathlib_Tactic_Order_preprocessFacts(v_facts_1349_, v_orderType_boxed_1358_, v_a_1351_, v_a_1352_, v_a_1353_, v_a_1354_, v_a_1355_, v_a_1356_);
lean_dec(v_a_1356_);
lean_dec_ref(v_a_1355_);
lean_dec(v_a_1354_);
lean_dec_ref(v_a_1353_);
lean_dec(v_a_1352_);
lean_dec_ref(v_a_1351_);
lean_dec_ref(v_facts_1349_);
return v_res_1359_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Order_Preprocessing(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_AtomM(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Order_Preprocessing(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_AtomM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_AtomM(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Order_Preprocessing(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_AtomM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Order_Preprocessing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Order_Preprocessing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Order_Preprocessing(builtin);
}
#ifdef __cplusplus
}
#endif
