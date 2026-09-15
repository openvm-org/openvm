// Lean compiler output
// Module: Swirl.Protocol.Noninteractive.Wire.Raw
// Imports: public import Init public meta import Init
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
lean_object* lean_nat_to_int(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
lean_object* lean_uint32_to_nat(uint32_t);
lean_object* l_Nat_reprFast(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* lean_uint64_to_nat(uint64_t);
lean_object* l_Std_Format_fill(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_byte_array_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_byte_array_fget(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint32_t lean_uint8_to_uint32(uint8_t);
uint32_t lean_uint32_shift_left(uint32_t, uint32_t);
uint32_t lean_uint32_lor(uint32_t, uint32_t);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
extern lean_object* l_ByteArray_empty;
lean_object* l_Bool_repr___redArg(uint8_t);
uint8_t lean_uint8_dec_eq(uint8_t, uint8_t);
lean_object* lean_uint8_to_nat(uint8_t);
uint64_t lean_uint32_to_uint64(uint32_t);
uint64_t lean_uint64_shift_left(uint64_t, uint64_t);
uint64_t lean_uint64_lor(uint64_t, uint64_t);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_byte_array_mk(lean_object*);
uint8_t lean_sarray_dec_eq(lean_object*, lean_object*);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_int_sub(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_BB__prime;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_magicMismatch_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_magicMismatch_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_versionMismatch_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_versionMismatch_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_unexpectedEnd_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_unexpectedEnd_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_decodeFailure_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_decodeFailure_elim(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedParseError_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedParseError_default___closed__0;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedParseError_default;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedParseError;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "magicMismatch"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__0_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "versionMismatch: expected "};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__1_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = ", got "};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__2_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "unexpectedEnd at offset "};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__3_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = ", need "};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__4 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__4_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = " more bytes"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__5_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "decodeFailure at offset "};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__6 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__6_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ": "};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__7 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__7_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString(lean_object*);
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_instToString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_instToString___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_instToString___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_instToString = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_instToString___closed__0_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedCursor_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedCursor_default___closed__0;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedCursor_default;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedCursor;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_runParser___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_runParser(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readByte(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt64(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readNat(lean_object*);
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readInt32___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readInt32___closed__0;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readInt32(lean_object*);
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBool___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "bool tag "};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBool___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBool___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBool(lean_object*);
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readCanonicalFBB___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "non-canonical babybear value "};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readCanonicalFBB___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readCanonicalFBB___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readCanonicalFBB(lean_object*);
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readOption___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "option tag "};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readOption___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readOption___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readOption___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readOption(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArrayExact___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArrayExact(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readVectorN___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readVectorN___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readVectorN___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readVectorN___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readVectorN___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readVectorN(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readVectorN___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readDigest(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readExt4(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_asciiMagic(uint8_t, uint8_t, uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_asciiMagic___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicProof___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicProof___closed__0;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicProof;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicVk___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicVk___closed__0;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicVk;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicPv___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicPv___closed__0;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicPv;
LEAN_EXPORT uint32_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_wireVersion;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readHeader(lean_object*, lean_object*);
LEAN_EXPORT uint32_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirRoundConfig_default;
LEAN_EXPORT uint32_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirRoundConfig;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "{ "};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__0_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "numQueries"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__2_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__3_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__4 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__4_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__5_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__3_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__6_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__7;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " }"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__8 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__8_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__9;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__11 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__11_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__8_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__12 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__12_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg(uint32_t);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr(uint32_t, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_ctorElim___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_uniqueDecoding_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_uniqueDecoding_elim___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_uniqueDecoding_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_uniqueDecoding_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_splitUniqueList_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_splitUniqueList_elim___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_splitUniqueList_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_splitUniqueList_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_listDecoding_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_listDecoding_elim___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_listDecoding_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_listDecoding_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProximityStrategy_default;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProximityStrategy;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 79, .m_capacity = 79, .m_length = 78, .m_data = "Swirl.Protocol.Noninteractive.Wire.Raw.RawWhirProximityStrategy.uniqueDecoding"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__1_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 80, .m_capacity = 80, .m_length = 79, .m_data = "Swirl.Protocol.Noninteractive.Wire.Raw.RawWhirProximityStrategy.splitUniqueList"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__4 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__4_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__4_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__5_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__5_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__6 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__6_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 77, .m_capacity = 77, .m_length = 76, .m_data = "Swirl.Protocol.Noninteractive.Wire.Raw.RawWhirProximityStrategy.listDecoding"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__7 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__7_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__7_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__8 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__8_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__8_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__9 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__9_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy___closed__0_value;
static const lean_array_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirConfig_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirConfig_default___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirConfig_default___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirConfig_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 16, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirConfig_default___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirConfig_default___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirConfig_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirConfig_default = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirConfig_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirConfig = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirConfig_default___closed__1_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "#["};
static const lean_object* lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__0_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__1_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__1_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__2_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__2_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__3_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__4 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__4_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__5;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__6;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__7 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__7_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__4_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__8 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__8_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "#[]"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__9 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__9_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__9_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__10 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__10_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0(lean_object*);
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "k"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__2_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__3_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__4;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "rounds"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__5_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__6_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__7;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "muPowBits"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__8 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__8_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__8_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__9 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__9_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__10;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "queryPhasePowBits"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__11 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__11_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__11_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__12 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__12_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__13;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "foldingPowBits"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__14 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__14_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__14_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__15 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__15_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__16;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "proximity"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__17 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__17_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__17_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__18 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__18_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawLogUpSecurityParameters_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 16, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawLogUpSecurityParameters_default___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawLogUpSecurityParameters_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawLogUpSecurityParameters_default = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawLogUpSecurityParameters_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawLogUpSecurityParameters = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawLogUpSecurityParameters_default___closed__0_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "maxInteractionCount"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__2_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__3_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__4;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "logMaxMessageLength"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__5_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__6_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "powBits"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__7 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__7_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__7_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__8 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__8_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__9;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSystemParams_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 24, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirConfig_default___closed__1_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawLogUpSecurityParameters_default___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSystemParams_default___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSystemParams_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSystemParams_default = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSystemParams_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSystemParams = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSystemParams_default___closed__0_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "lSkip"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__2_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__3_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__4;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "nStack"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__5_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__6_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "wStack"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__7 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__7_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__7_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__8 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__8_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "logBlowup"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__9 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__9_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__9_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__10 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__10_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "whir"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__11 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__11_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__11_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__12 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__12_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__13;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "logup"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__14 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__14_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__14_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__15 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__15_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "maxConstraintDegree"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__16 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__16_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__16_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__17 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__17_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceWidth_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirConfig_default___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceWidth_default___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceWidth_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceWidth_default = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceWidth_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceWidth = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceWidth_default___closed__0_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Option_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "none"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Option_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__0___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Option_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__0___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Option_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Option_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__0___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Option_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__0___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Option_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__0___closed__1_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Option_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "some "};
static const lean_object* lp_swirl_x2drbr_x2dformal_Option_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__0___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Option_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__0___closed__2_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Option_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Option_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__0___closed__2_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Option_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__0___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Option_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__0___closed__3_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Option_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Option_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__1_spec__1___lam__0(uint32_t);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__1_spec__1___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__1_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__1_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__1_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__1(lean_object*);
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "preprocessed"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__2_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__3_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__4;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "cachedMains"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__5_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__6_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__7;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "commonMain"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__8 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__8_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__8_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__9 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__9_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Swirl_Protocol_Noninteractive_Wire_Raw_RawTraceWidth_totalWidth_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Swirl_Protocol_Noninteractive_Wire_Raw_RawTraceWidth_totalWidth_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawTraceWidth_totalWidth(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawTraceWidth_totalWidth___boxed(lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStarkVerifyingParams_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceWidth_default___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStarkVerifyingParams_default___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStarkVerifyingParams_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStarkVerifyingParams_default = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStarkVerifyingParams_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStarkVerifyingParams = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStarkVerifyingParams_default___closed__0_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "width"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__2_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__3_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "numPublicValues"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__4 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__4_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__5_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__6;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "needRot"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__7 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__7_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__7_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__8 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__8_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__0___boxed__const__1;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__0;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__1;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__2;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawLinearConstraint_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirConfig_default___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawLinearConstraint_default___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawLinearConstraint_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawLinearConstraint_default = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawLinearConstraint_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawLinearConstraint = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawLinearConstraint_default___closed__0_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "coefficients"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg___closed__2_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg___closed__3_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "threshold"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg___closed__4 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg___closed__4_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg___closed__5_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_ctorElim___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_preprocessed_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_preprocessed_elim___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_preprocessed_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_preprocessed_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_main_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_main_elim___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_main_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_main_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_publicInput_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_publicInput_elim___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_publicInput_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_publicInput_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_challenge_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_challenge_elim___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_challenge_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_challenge_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawEntry_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 8, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawEntry_default___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawEntry_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawEntry_default = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawEntry_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawEntry = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawEntry_default___closed__0_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 60, .m_capacity = 60, .m_length = 59, .m_data = "Swirl.Protocol.Noninteractive.Wire.Raw.RawEntry.publicInput"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__1_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 58, .m_capacity = 58, .m_length = 57, .m_data = "Swirl.Protocol.Noninteractive.Wire.Raw.RawEntry.challenge"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__2_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__2_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__3_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 61, .m_capacity = 61, .m_length = 60, .m_data = "Swirl.Protocol.Noninteractive.Wire.Raw.RawEntry.preprocessed"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__4 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__4_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__4_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__5_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__5_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__6 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__6_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 53, .m_capacity = 53, .m_length = 52, .m_data = "Swirl.Protocol.Noninteractive.Wire.Raw.RawEntry.main"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__7 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__7_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__7_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__8 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__8_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__8_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__9 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__9_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicVariable_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawEntry_default___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicVariable_default___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicVariable_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicVariable_default = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicVariable_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicVariable = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicVariable_default___closed__0_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "entry"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___closed__2_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___closed__3_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "index"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___closed__4 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___closed__4_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___closed__5_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_variable_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_variable_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_isFirstRow_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_isFirstRow_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_isLastRow_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_isLastRow_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_isTransition_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_isTransition_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_constant_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_constant_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_add_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_add_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_sub_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_sub_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_neg_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_neg_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_mul_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_mul_elim(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicExpressionNode_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicVariable_default___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicExpressionNode_default___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicExpressionNode_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicExpressionNode_default = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicExpressionNode_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicExpressionNode = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicExpressionNode_default___closed__0_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 78, .m_capacity = 78, .m_length = 77, .m_data = "Swirl.Protocol.Noninteractive.Wire.Raw.RawSymbolicExpressionNode.isTransition"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__1_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 75, .m_capacity = 75, .m_length = 74, .m_data = "Swirl.Protocol.Noninteractive.Wire.Raw.RawSymbolicExpressionNode.isLastRow"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__2_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__2_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__3_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 76, .m_capacity = 76, .m_length = 75, .m_data = "Swirl.Protocol.Noninteractive.Wire.Raw.RawSymbolicExpressionNode.isFirstRow"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__4 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__4_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__4_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__5_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 74, .m_capacity = 74, .m_length = 73, .m_data = "Swirl.Protocol.Noninteractive.Wire.Raw.RawSymbolicExpressionNode.variable"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__6 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__6_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__6_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__7 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__7_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__7_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__8 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__8_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 74, .m_capacity = 74, .m_length = 73, .m_data = "Swirl.Protocol.Noninteractive.Wire.Raw.RawSymbolicExpressionNode.constant"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__9 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__9_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__9_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__10 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__10_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__10_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__11 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__11_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 69, .m_capacity = 69, .m_length = 68, .m_data = "Swirl.Protocol.Noninteractive.Wire.Raw.RawSymbolicExpressionNode.add"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__12 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__12_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__12_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__13 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__13_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__13_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__14 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__14_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 69, .m_capacity = 69, .m_length = 68, .m_data = "Swirl.Protocol.Noninteractive.Wire.Raw.RawSymbolicExpressionNode.sub"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__15 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__15_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__15_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__16 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__16_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__16_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__17 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__17_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 69, .m_capacity = 69, .m_length = 68, .m_data = "Swirl.Protocol.Noninteractive.Wire.Raw.RawSymbolicExpressionNode.neg"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__18 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__18_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__18_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__19 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__19_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__19_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__20 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__20_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 69, .m_capacity = 69, .m_length = 68, .m_data = "Swirl.Protocol.Noninteractive.Wire.Raw.RawSymbolicExpressionNode.mul"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__21 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__21_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__21_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__22 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__22_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__22_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__23 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__23_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode___closed__0_value;
static const lean_array_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicExpressionDag_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicExpressionDag_default___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicExpressionDag_default___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicExpressionDag_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicExpressionDag_default___closed__0_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicExpressionDag_default___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicExpressionDag_default___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicExpressionDag_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicExpressionDag_default = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicExpressionDag_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicExpressionDag = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicExpressionDag_default___closed__1_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr_spec__0_spec__0___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr_spec__0_spec__0___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr_spec__0(lean_object*);
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "nodes"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__2_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__3_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "constraintIdx"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__4 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__4_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__5_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__6;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicInteraction_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 16, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirConfig_default___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicInteraction_default___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicInteraction_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicInteraction_default = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicInteraction_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicInteraction = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicInteraction_default___closed__0_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "message"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__2_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__3_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "count"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__4 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__4_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__5_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "busIndex"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__6 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__6_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__6_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__7 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__7_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__8;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "countWeight"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__9 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__9_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__9_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__10 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__10_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction___closed__0_value;
static const lean_array_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicConstraintsDag_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicConstraintsDag_default___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicConstraintsDag_default___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicConstraintsDag_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicExpressionDag_default___closed__1_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicConstraintsDag_default___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicConstraintsDag_default___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicConstraintsDag_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicConstraintsDag_default = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicConstraintsDag_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicConstraintsDag = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicConstraintsDag_default___closed__1_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr_spec__0(lean_object*);
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "constraints"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg___closed__1_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg___closed__1_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg___closed__2 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg___closed__2_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg___closed__2_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__5_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg___closed__3 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg___closed__3_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "interactions"};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg___closed__4 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg___closed__4_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg___closed__4_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg___closed__5 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg___closed__5_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag___closed__0_value;
static const lean_array_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStarkVerifyingKey_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStarkVerifyingKey_default___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStarkVerifyingKey_default___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStarkVerifyingKey_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStarkVerifyingParams_default___closed__0_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSymbolicConstraintsDag_default___closed__1_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStarkVerifyingKey_default___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStarkVerifyingKey_default___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStarkVerifyingKey_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStarkVerifyingKey_default = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStarkVerifyingKey_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStarkVerifyingKey = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStarkVerifyingKey_default___closed__1_value;
static const lean_array_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawMultiStarkVerifyingKey0_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawMultiStarkVerifyingKey0_default___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawMultiStarkVerifyingKey0_default___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawMultiStarkVerifyingKey0_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawSystemParams_default___closed__0_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawMultiStarkVerifyingKey0_default___closed__0_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawMultiStarkVerifyingKey0_default___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawMultiStarkVerifyingKey0_default___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawMultiStarkVerifyingKey0_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawMultiStarkVerifyingKey0_default = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawMultiStarkVerifyingKey0_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawMultiStarkVerifyingKey0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawMultiStarkVerifyingKey0_default___closed__1_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVk_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVk_default___closed__0;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVk_default;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVk;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawVk_airCount(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawVk_airCount___boxed(lean_object*);
static const lean_array_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceVData_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceVData_default___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceVData_default___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceVData_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceVData_default___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceVData_default___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceVData_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceVData_default = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceVData_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceVData = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceVData_default___closed__1_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims_default___closed__0;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims_default___closed__1;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims_default;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims;
static const lean_array_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrProof_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrProof_default___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrProof_default___closed__0_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrProof_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrProof_default___closed__1;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrProof_default;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrProof;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawBatchConstraintProof_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceVData_default___closed__0_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceVData_default___closed__0_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceVData_default___closed__0_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceVData_default___closed__0_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceVData_default___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawBatchConstraintProof_default___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawBatchConstraintProof_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawBatchConstraintProof_default = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawBatchConstraintProof_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawBatchConstraintProof = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawBatchConstraintProof_default___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStackingProof_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceVData_default___closed__0_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceVData_default___closed__0_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceVData_default___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStackingProof_default___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStackingProof_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStackingProof_default = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStackingProof_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStackingProof = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStackingProof_default___closed__0_value;
static const lean_array_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProof_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProof_default___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProof_default___closed__0_value;
static const lean_ctor_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProof_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*10 + 0, .m_other = 10, .m_tag = 0}, .m_objs = {((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProof_default___closed__0_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProof_default___closed__0_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProof_default___closed__0_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProof_default___closed__0_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProof_default___closed__0_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProof_default___closed__0_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProof_default___closed__0_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProof_default___closed__0_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProof_default___closed__0_value),((lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProof_default___closed__0_value)}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProof_default___closed__1 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProof_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProof_default = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProof_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProof = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProof_default___closed__1_value;
static const lean_array_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawProof_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawProof_default___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawProof_default___closed__0_value;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawProof_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawProof_default___closed__1;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawProof_default;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawProof;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawPublicValues_default = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceVData_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawPublicValues = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawTraceVData_default___closed__0_value;
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readEntry___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "entry-tag "};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readEntry___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readEntry___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readEntry(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readSymbolicVariable(lean_object*);
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readSymbolicExpressionNode___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "symbolic-node-tag "};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readSymbolicExpressionNode___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readSymbolicExpressionNode___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readSymbolicExpressionNode(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readSymbolicExpressionDag(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readSymbolicInteraction(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readSymbolicConstraintsDag(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirRoundConfig(lean_object*);
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProximityStrategy___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "proximity-tag "};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProximityStrategy___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProximityStrategy___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProximityStrategy(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirConfig(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readLogUpSecurityParameters(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readSystemParams(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readVerifierSinglePreprocessedData(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readTraceWidth(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readStarkVerifyingParams(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readLinearConstraint(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readStarkVerifyingKey(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readMultiStarkVerifyingKey0(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readRawVkM(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readRawVk(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readTraceVData(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readGkrLayerClaims(lean_object*);
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readGkrProof___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readGkrProof___closed__0;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readGkrProof___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readGkrProof___closed__1;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readGkrProof(lean_object*);
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBatchConstraintProof___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBatchConstraintProof___closed__0;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBatchConstraintProof___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBatchConstraintProof___closed__1;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBatchConstraintProof(lean_object*);
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readStackingProof___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readStackingProof___closed__0;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readStackingProof(lean_object*);
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__0;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__1;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__2;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__3;
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__4;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof(lean_object*);
static lean_once_cell_t lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readRawProofM___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readRawProofM___closed__0;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readRawProofM(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readRawProof(lean_object*);
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readPvForArities___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "pv-air-len: expected 0 or "};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readPvForArities___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readPvForArities___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readPvForArities(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readPublicValueRowsM___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "pv-air-count: expected "};
static const lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readPublicValueRowsM___closed__0 = (const lean_object*)&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readPublicValueRowsM___closed__0_value;
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readPublicValueRowsM(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readPublicValueRows(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Swirl_Protocol_Noninteractive_Wire_Raw_RawVk_publicValueArities_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Swirl_Protocol_Noninteractive_Wire_Raw_RawVk_publicValueArities_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawVk_publicValueArities(lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readRawPublicValues(lean_object*, lean_object*);
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_BB__prime(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lean_unsigned_to_nat(2013265921u);
return v___x_1_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_ctorIdx(lean_object* v_x_2_){
_start:
{
switch(lean_obj_tag(v_x_2_))
{
case 0:
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(0u);
return v___x_3_;
}
case 1:
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(1u);
return v___x_4_;
}
case 2:
{
lean_object* v___x_5_; 
v___x_5_ = lean_unsigned_to_nat(2u);
return v___x_5_;
}
default: 
{
lean_object* v___x_6_; 
v___x_6_ = lean_unsigned_to_nat(3u);
return v___x_6_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_ctorIdx___boxed(lean_object* v_x_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_ctorIdx(v_x_7_);
lean_dec_ref(v_x_7_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_ctorElim___redArg(lean_object* v_t_9_, lean_object* v_k_10_){
_start:
{
switch(lean_obj_tag(v_t_9_))
{
case 0:
{
lean_object* v_expected_11_; lean_object* v_actual_12_; lean_object* v___x_13_; 
v_expected_11_ = lean_ctor_get(v_t_9_, 0);
lean_inc_ref(v_expected_11_);
v_actual_12_ = lean_ctor_get(v_t_9_, 1);
lean_inc_ref(v_actual_12_);
lean_dec_ref_known(v_t_9_, 2);
v___x_13_ = lean_apply_2(v_k_10_, v_expected_11_, v_actual_12_);
return v___x_13_;
}
case 1:
{
uint32_t v_expected_14_; uint32_t v_actual_15_; lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_18_; 
v_expected_14_ = lean_ctor_get_uint32(v_t_9_, 0);
v_actual_15_ = lean_ctor_get_uint32(v_t_9_, 4);
lean_dec_ref_known(v_t_9_, 0);
v___x_16_ = lean_box_uint32(v_expected_14_);
v___x_17_ = lean_box_uint32(v_actual_15_);
v___x_18_ = lean_apply_2(v_k_10_, v___x_16_, v___x_17_);
return v___x_18_;
}
case 2:
{
lean_object* v_offset_19_; lean_object* v_need_20_; lean_object* v___x_21_; 
v_offset_19_ = lean_ctor_get(v_t_9_, 0);
lean_inc(v_offset_19_);
v_need_20_ = lean_ctor_get(v_t_9_, 1);
lean_inc(v_need_20_);
lean_dec_ref_known(v_t_9_, 2);
v___x_21_ = lean_apply_2(v_k_10_, v_offset_19_, v_need_20_);
return v___x_21_;
}
default: 
{
lean_object* v_offset_22_; lean_object* v_msg_23_; lean_object* v___x_24_; 
v_offset_22_ = lean_ctor_get(v_t_9_, 0);
lean_inc(v_offset_22_);
v_msg_23_ = lean_ctor_get(v_t_9_, 1);
lean_inc_ref(v_msg_23_);
lean_dec_ref_known(v_t_9_, 2);
v___x_24_ = lean_apply_2(v_k_10_, v_offset_22_, v_msg_23_);
return v___x_24_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_ctorElim(lean_object* v_motive_25_, lean_object* v_ctorIdx_26_, lean_object* v_t_27_, lean_object* v_h_28_, lean_object* v_k_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_ctorElim___redArg(v_t_27_, v_k_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_ctorElim___boxed(lean_object* v_motive_31_, lean_object* v_ctorIdx_32_, lean_object* v_t_33_, lean_object* v_h_34_, lean_object* v_k_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_ctorElim(v_motive_31_, v_ctorIdx_32_, v_t_33_, v_h_34_, v_k_35_);
lean_dec(v_ctorIdx_32_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_magicMismatch_elim___redArg(lean_object* v_t_37_, lean_object* v_magicMismatch_38_){
_start:
{
lean_object* v___x_39_; 
v___x_39_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_ctorElim___redArg(v_t_37_, v_magicMismatch_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_magicMismatch_elim(lean_object* v_motive_40_, lean_object* v_t_41_, lean_object* v_h_42_, lean_object* v_magicMismatch_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_ctorElim___redArg(v_t_41_, v_magicMismatch_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_versionMismatch_elim___redArg(lean_object* v_t_45_, lean_object* v_versionMismatch_46_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_ctorElim___redArg(v_t_45_, v_versionMismatch_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_versionMismatch_elim(lean_object* v_motive_48_, lean_object* v_t_49_, lean_object* v_h_50_, lean_object* v_versionMismatch_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_ctorElim___redArg(v_t_49_, v_versionMismatch_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_unexpectedEnd_elim___redArg(lean_object* v_t_53_, lean_object* v_unexpectedEnd_54_){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_ctorElim___redArg(v_t_53_, v_unexpectedEnd_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_unexpectedEnd_elim(lean_object* v_motive_56_, lean_object* v_t_57_, lean_object* v_h_58_, lean_object* v_unexpectedEnd_59_){
_start:
{
lean_object* v___x_60_; 
v___x_60_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_ctorElim___redArg(v_t_57_, v_unexpectedEnd_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_decodeFailure_elim___redArg(lean_object* v_t_61_, lean_object* v_decodeFailure_62_){
_start:
{
lean_object* v___x_63_; 
v___x_63_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_ctorElim___redArg(v_t_61_, v_decodeFailure_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_decodeFailure_elim(lean_object* v_motive_64_, lean_object* v_t_65_, lean_object* v_h_66_, lean_object* v_decodeFailure_67_){
_start:
{
lean_object* v___x_68_; 
v___x_68_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_ctorElim___redArg(v_t_65_, v_decodeFailure_67_);
return v___x_68_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedParseError_default___closed__0(void){
_start:
{
lean_object* v___x_69_; lean_object* v___x_70_; 
v___x_69_ = l_ByteArray_empty;
v___x_70_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_70_, 0, v___x_69_);
lean_ctor_set(v___x_70_, 1, v___x_69_);
return v___x_70_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedParseError_default(void){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedParseError_default___closed__0, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedParseError_default___closed__0_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedParseError_default___closed__0);
return v___x_71_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedParseError(void){
_start:
{
lean_object* v___x_72_; 
v___x_72_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedParseError_default;
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString(lean_object* v_x_81_){
_start:
{
switch(lean_obj_tag(v_x_81_))
{
case 0:
{
lean_object* v___x_82_; 
lean_dec_ref_known(v_x_81_, 2);
v___x_82_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__0));
return v___x_82_;
}
case 1:
{
uint32_t v_expected_83_; uint32_t v_actual_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; 
v_expected_83_ = lean_ctor_get_uint32(v_x_81_, 0);
v_actual_84_ = lean_ctor_get_uint32(v_x_81_, 4);
lean_dec_ref_known(v_x_81_, 0);
v___x_85_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__1));
v___x_86_ = lean_uint32_to_nat(v_expected_83_);
v___x_87_ = l_Nat_reprFast(v___x_86_);
v___x_88_ = lean_string_append(v___x_85_, v___x_87_);
lean_dec_ref(v___x_87_);
v___x_89_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__2));
v___x_90_ = lean_string_append(v___x_88_, v___x_89_);
v___x_91_ = lean_uint32_to_nat(v_actual_84_);
v___x_92_ = l_Nat_reprFast(v___x_91_);
v___x_93_ = lean_string_append(v___x_90_, v___x_92_);
lean_dec_ref(v___x_92_);
return v___x_93_;
}
case 2:
{
lean_object* v_offset_94_; lean_object* v_need_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; 
v_offset_94_ = lean_ctor_get(v_x_81_, 0);
lean_inc(v_offset_94_);
v_need_95_ = lean_ctor_get(v_x_81_, 1);
lean_inc(v_need_95_);
lean_dec_ref_known(v_x_81_, 2);
v___x_96_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__3));
v___x_97_ = l_Nat_reprFast(v_offset_94_);
v___x_98_ = lean_string_append(v___x_96_, v___x_97_);
lean_dec_ref(v___x_97_);
v___x_99_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__4));
v___x_100_ = lean_string_append(v___x_98_, v___x_99_);
v___x_101_ = l_Nat_reprFast(v_need_95_);
v___x_102_ = lean_string_append(v___x_100_, v___x_101_);
lean_dec_ref(v___x_101_);
v___x_103_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__5));
v___x_104_ = lean_string_append(v___x_102_, v___x_103_);
return v___x_104_;
}
default: 
{
lean_object* v_offset_105_; lean_object* v_msg_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; 
v_offset_105_ = lean_ctor_get(v_x_81_, 0);
lean_inc(v_offset_105_);
v_msg_106_ = lean_ctor_get(v_x_81_, 1);
lean_inc_ref(v_msg_106_);
lean_dec_ref_known(v_x_81_, 2);
v___x_107_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__6));
v___x_108_ = l_Nat_reprFast(v_offset_105_);
v___x_109_ = lean_string_append(v___x_107_, v___x_108_);
lean_dec_ref(v___x_108_);
v___x_110_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__7));
v___x_111_ = lean_string_append(v___x_109_, v___x_110_);
v___x_112_ = lean_string_append(v___x_111_, v_msg_106_);
lean_dec_ref(v_msg_106_);
return v___x_112_;
}
}
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedCursor_default___closed__0(void){
_start:
{
lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; 
v___x_115_ = lean_unsigned_to_nat(0u);
v___x_116_ = l_ByteArray_empty;
v___x_117_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_117_, 0, v___x_116_);
lean_ctor_set(v___x_117_, 1, v___x_115_);
return v___x_117_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedCursor_default(void){
_start:
{
lean_object* v___x_118_; 
v___x_118_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedCursor_default___closed__0, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedCursor_default___closed__0_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedCursor_default___closed__0);
return v___x_118_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedCursor(void){
_start:
{
lean_object* v___x_119_; 
v___x_119_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedCursor_default;
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_runParser___redArg(lean_object* v_p_120_, lean_object* v_data_121_){
_start:
{
lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; 
v___x_122_ = lean_unsigned_to_nat(0u);
v___x_123_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_123_, 0, v_data_121_);
lean_ctor_set(v___x_123_, 1, v___x_122_);
v___x_124_ = lean_apply_1(v_p_120_, v___x_123_);
if (lean_obj_tag(v___x_124_) == 0)
{
lean_object* v_a_125_; lean_object* v___x_126_; 
v_a_125_ = lean_ctor_get(v___x_124_, 0);
lean_inc(v_a_125_);
lean_dec_ref_known(v___x_124_, 2);
v___x_126_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_126_, 0, v_a_125_);
return v___x_126_;
}
else
{
lean_object* v_a_127_; lean_object* v___x_128_; 
v_a_127_ = lean_ctor_get(v___x_124_, 0);
lean_inc(v_a_127_);
lean_dec_ref_known(v___x_124_, 2);
v___x_128_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_128_, 0, v_a_127_);
return v___x_128_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_runParser(lean_object* v_00_u03b1_129_, lean_object* v_p_130_, lean_object* v_data_131_){
_start:
{
lean_object* v___x_132_; 
v___x_132_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_runParser___redArg(v_p_130_, v_data_131_);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readByte(lean_object* v_a_133_){
_start:
{
lean_object* v_data_134_; lean_object* v_offset_135_; lean_object* v___x_136_; uint8_t v___x_137_; 
v_data_134_ = lean_ctor_get(v_a_133_, 0);
v_offset_135_ = lean_ctor_get(v_a_133_, 1);
v___x_136_ = lean_byte_array_size(v_data_134_);
v___x_137_ = lean_nat_dec_lt(v_offset_135_, v___x_136_);
if (v___x_137_ == 0)
{
lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; 
v___x_138_ = lean_unsigned_to_nat(1u);
lean_inc(v_offset_135_);
v___x_139_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_139_, 0, v_offset_135_);
lean_ctor_set(v___x_139_, 1, v___x_138_);
v___x_140_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_140_, 0, v___x_139_);
lean_ctor_set(v___x_140_, 1, v_a_133_);
return v___x_140_;
}
else
{
lean_object* v___x_142_; uint8_t v_isShared_143_; uint8_t v_isSharedCheck_152_; 
lean_inc(v_offset_135_);
lean_inc_ref(v_data_134_);
v_isSharedCheck_152_ = !lean_is_exclusive(v_a_133_);
if (v_isSharedCheck_152_ == 0)
{
lean_object* v_unused_153_; lean_object* v_unused_154_; 
v_unused_153_ = lean_ctor_get(v_a_133_, 1);
lean_dec(v_unused_153_);
v_unused_154_ = lean_ctor_get(v_a_133_, 0);
lean_dec(v_unused_154_);
v___x_142_ = v_a_133_;
v_isShared_143_ = v_isSharedCheck_152_;
goto v_resetjp_141_;
}
else
{
lean_dec(v_a_133_);
v___x_142_ = lean_box(0);
v_isShared_143_ = v_isSharedCheck_152_;
goto v_resetjp_141_;
}
v_resetjp_141_:
{
uint8_t v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_148_; 
v___x_144_ = lean_byte_array_fget(v_data_134_, v_offset_135_);
v___x_145_ = lean_unsigned_to_nat(1u);
v___x_146_ = lean_nat_add(v_offset_135_, v___x_145_);
lean_dec(v_offset_135_);
if (v_isShared_143_ == 0)
{
lean_ctor_set(v___x_142_, 1, v___x_146_);
v___x_148_ = v___x_142_;
goto v_reusejp_147_;
}
else
{
lean_object* v_reuseFailAlloc_151_; 
v_reuseFailAlloc_151_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_151_, 0, v_data_134_);
lean_ctor_set(v_reuseFailAlloc_151_, 1, v___x_146_);
v___x_148_ = v_reuseFailAlloc_151_;
goto v_reusejp_147_;
}
v_reusejp_147_:
{
lean_object* v___x_149_; lean_object* v___x_150_; 
v___x_149_ = lean_box(v___x_144_);
v___x_150_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_150_, 0, v___x_149_);
lean_ctor_set(v___x_150_, 1, v___x_148_);
return v___x_150_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(lean_object* v_a_155_){
_start:
{
lean_object* v___x_156_; 
v___x_156_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readByte(v_a_155_);
if (lean_obj_tag(v___x_156_) == 0)
{
lean_object* v_a_157_; lean_object* v_a_158_; lean_object* v___x_159_; 
v_a_157_ = lean_ctor_get(v___x_156_, 0);
lean_inc(v_a_157_);
v_a_158_ = lean_ctor_get(v___x_156_, 1);
lean_inc(v_a_158_);
lean_dec_ref_known(v___x_156_, 2);
v___x_159_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readByte(v_a_158_);
if (lean_obj_tag(v___x_159_) == 0)
{
lean_object* v_a_160_; lean_object* v_a_161_; lean_object* v___x_162_; 
v_a_160_ = lean_ctor_get(v___x_159_, 0);
lean_inc(v_a_160_);
v_a_161_ = lean_ctor_get(v___x_159_, 1);
lean_inc(v_a_161_);
lean_dec_ref_known(v___x_159_, 2);
v___x_162_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readByte(v_a_161_);
if (lean_obj_tag(v___x_162_) == 0)
{
lean_object* v_a_163_; lean_object* v_a_164_; lean_object* v___x_165_; 
v_a_163_ = lean_ctor_get(v___x_162_, 0);
lean_inc(v_a_163_);
v_a_164_ = lean_ctor_get(v___x_162_, 1);
lean_inc(v_a_164_);
lean_dec_ref_known(v___x_162_, 2);
v___x_165_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readByte(v_a_164_);
if (lean_obj_tag(v___x_165_) == 0)
{
lean_object* v_a_166_; lean_object* v_a_167_; lean_object* v___x_169_; uint8_t v_isShared_170_; uint8_t v_isSharedCheck_192_; 
v_a_166_ = lean_ctor_get(v___x_165_, 0);
v_a_167_ = lean_ctor_get(v___x_165_, 1);
v_isSharedCheck_192_ = !lean_is_exclusive(v___x_165_);
if (v_isSharedCheck_192_ == 0)
{
v___x_169_ = v___x_165_;
v_isShared_170_ = v_isSharedCheck_192_;
goto v_resetjp_168_;
}
else
{
lean_inc(v_a_167_);
lean_inc(v_a_166_);
lean_dec(v___x_165_);
v___x_169_ = lean_box(0);
v_isShared_170_ = v_isSharedCheck_192_;
goto v_resetjp_168_;
}
v_resetjp_168_:
{
uint8_t v___x_171_; uint32_t v___x_172_; uint8_t v___x_173_; uint32_t v___x_174_; uint32_t v___x_175_; uint32_t v___x_176_; uint32_t v___x_177_; uint8_t v___x_178_; uint32_t v___x_179_; uint32_t v___x_180_; uint32_t v___x_181_; uint32_t v___x_182_; uint8_t v___x_183_; uint32_t v___x_184_; uint32_t v___x_185_; uint32_t v___x_186_; uint32_t v___x_187_; lean_object* v___x_188_; lean_object* v___x_190_; 
v___x_171_ = lean_unbox(v_a_157_);
lean_dec(v_a_157_);
v___x_172_ = lean_uint8_to_uint32(v___x_171_);
v___x_173_ = lean_unbox(v_a_160_);
lean_dec(v_a_160_);
v___x_174_ = lean_uint8_to_uint32(v___x_173_);
v___x_175_ = 8;
v___x_176_ = lean_uint32_shift_left(v___x_174_, v___x_175_);
v___x_177_ = lean_uint32_lor(v___x_172_, v___x_176_);
v___x_178_ = lean_unbox(v_a_163_);
lean_dec(v_a_163_);
v___x_179_ = lean_uint8_to_uint32(v___x_178_);
v___x_180_ = 16;
v___x_181_ = lean_uint32_shift_left(v___x_179_, v___x_180_);
v___x_182_ = lean_uint32_lor(v___x_177_, v___x_181_);
v___x_183_ = lean_unbox(v_a_166_);
lean_dec(v_a_166_);
v___x_184_ = lean_uint8_to_uint32(v___x_183_);
v___x_185_ = 24;
v___x_186_ = lean_uint32_shift_left(v___x_184_, v___x_185_);
v___x_187_ = lean_uint32_lor(v___x_182_, v___x_186_);
v___x_188_ = lean_box_uint32(v___x_187_);
if (v_isShared_170_ == 0)
{
lean_ctor_set(v___x_169_, 0, v___x_188_);
v___x_190_ = v___x_169_;
goto v_reusejp_189_;
}
else
{
lean_object* v_reuseFailAlloc_191_; 
v_reuseFailAlloc_191_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_191_, 0, v___x_188_);
lean_ctor_set(v_reuseFailAlloc_191_, 1, v_a_167_);
v___x_190_ = v_reuseFailAlloc_191_;
goto v_reusejp_189_;
}
v_reusejp_189_:
{
return v___x_190_;
}
}
}
else
{
lean_object* v_a_193_; lean_object* v_a_194_; lean_object* v___x_196_; uint8_t v_isShared_197_; uint8_t v_isSharedCheck_201_; 
lean_dec(v_a_163_);
lean_dec(v_a_160_);
lean_dec(v_a_157_);
v_a_193_ = lean_ctor_get(v___x_165_, 0);
v_a_194_ = lean_ctor_get(v___x_165_, 1);
v_isSharedCheck_201_ = !lean_is_exclusive(v___x_165_);
if (v_isSharedCheck_201_ == 0)
{
v___x_196_ = v___x_165_;
v_isShared_197_ = v_isSharedCheck_201_;
goto v_resetjp_195_;
}
else
{
lean_inc(v_a_194_);
lean_inc(v_a_193_);
lean_dec(v___x_165_);
v___x_196_ = lean_box(0);
v_isShared_197_ = v_isSharedCheck_201_;
goto v_resetjp_195_;
}
v_resetjp_195_:
{
lean_object* v___x_199_; 
if (v_isShared_197_ == 0)
{
v___x_199_ = v___x_196_;
goto v_reusejp_198_;
}
else
{
lean_object* v_reuseFailAlloc_200_; 
v_reuseFailAlloc_200_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_200_, 0, v_a_193_);
lean_ctor_set(v_reuseFailAlloc_200_, 1, v_a_194_);
v___x_199_ = v_reuseFailAlloc_200_;
goto v_reusejp_198_;
}
v_reusejp_198_:
{
return v___x_199_;
}
}
}
}
else
{
lean_object* v_a_202_; lean_object* v_a_203_; lean_object* v___x_205_; uint8_t v_isShared_206_; uint8_t v_isSharedCheck_210_; 
lean_dec(v_a_160_);
lean_dec(v_a_157_);
v_a_202_ = lean_ctor_get(v___x_162_, 0);
v_a_203_ = lean_ctor_get(v___x_162_, 1);
v_isSharedCheck_210_ = !lean_is_exclusive(v___x_162_);
if (v_isSharedCheck_210_ == 0)
{
v___x_205_ = v___x_162_;
v_isShared_206_ = v_isSharedCheck_210_;
goto v_resetjp_204_;
}
else
{
lean_inc(v_a_203_);
lean_inc(v_a_202_);
lean_dec(v___x_162_);
v___x_205_ = lean_box(0);
v_isShared_206_ = v_isSharedCheck_210_;
goto v_resetjp_204_;
}
v_resetjp_204_:
{
lean_object* v___x_208_; 
if (v_isShared_206_ == 0)
{
v___x_208_ = v___x_205_;
goto v_reusejp_207_;
}
else
{
lean_object* v_reuseFailAlloc_209_; 
v_reuseFailAlloc_209_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_209_, 0, v_a_202_);
lean_ctor_set(v_reuseFailAlloc_209_, 1, v_a_203_);
v___x_208_ = v_reuseFailAlloc_209_;
goto v_reusejp_207_;
}
v_reusejp_207_:
{
return v___x_208_;
}
}
}
}
else
{
lean_object* v_a_211_; lean_object* v_a_212_; lean_object* v___x_214_; uint8_t v_isShared_215_; uint8_t v_isSharedCheck_219_; 
lean_dec(v_a_157_);
v_a_211_ = lean_ctor_get(v___x_159_, 0);
v_a_212_ = lean_ctor_get(v___x_159_, 1);
v_isSharedCheck_219_ = !lean_is_exclusive(v___x_159_);
if (v_isSharedCheck_219_ == 0)
{
v___x_214_ = v___x_159_;
v_isShared_215_ = v_isSharedCheck_219_;
goto v_resetjp_213_;
}
else
{
lean_inc(v_a_212_);
lean_inc(v_a_211_);
lean_dec(v___x_159_);
v___x_214_ = lean_box(0);
v_isShared_215_ = v_isSharedCheck_219_;
goto v_resetjp_213_;
}
v_resetjp_213_:
{
lean_object* v___x_217_; 
if (v_isShared_215_ == 0)
{
v___x_217_ = v___x_214_;
goto v_reusejp_216_;
}
else
{
lean_object* v_reuseFailAlloc_218_; 
v_reuseFailAlloc_218_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_218_, 0, v_a_211_);
lean_ctor_set(v_reuseFailAlloc_218_, 1, v_a_212_);
v___x_217_ = v_reuseFailAlloc_218_;
goto v_reusejp_216_;
}
v_reusejp_216_:
{
return v___x_217_;
}
}
}
}
else
{
lean_object* v_a_220_; lean_object* v_a_221_; lean_object* v___x_223_; uint8_t v_isShared_224_; uint8_t v_isSharedCheck_228_; 
v_a_220_ = lean_ctor_get(v___x_156_, 0);
v_a_221_ = lean_ctor_get(v___x_156_, 1);
v_isSharedCheck_228_ = !lean_is_exclusive(v___x_156_);
if (v_isSharedCheck_228_ == 0)
{
v___x_223_ = v___x_156_;
v_isShared_224_ = v_isSharedCheck_228_;
goto v_resetjp_222_;
}
else
{
lean_inc(v_a_221_);
lean_inc(v_a_220_);
lean_dec(v___x_156_);
v___x_223_ = lean_box(0);
v_isShared_224_ = v_isSharedCheck_228_;
goto v_resetjp_222_;
}
v_resetjp_222_:
{
lean_object* v___x_226_; 
if (v_isShared_224_ == 0)
{
v___x_226_ = v___x_223_;
goto v_reusejp_225_;
}
else
{
lean_object* v_reuseFailAlloc_227_; 
v_reuseFailAlloc_227_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_227_, 0, v_a_220_);
lean_ctor_set(v_reuseFailAlloc_227_, 1, v_a_221_);
v___x_226_ = v_reuseFailAlloc_227_;
goto v_reusejp_225_;
}
v_reusejp_225_:
{
return v___x_226_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt64(lean_object* v_a_229_){
_start:
{
lean_object* v___x_230_; 
v___x_230_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_229_);
if (lean_obj_tag(v___x_230_) == 0)
{
lean_object* v_a_231_; lean_object* v_a_232_; lean_object* v___x_233_; 
v_a_231_ = lean_ctor_get(v___x_230_, 0);
lean_inc(v_a_231_);
v_a_232_ = lean_ctor_get(v___x_230_, 1);
lean_inc(v_a_232_);
lean_dec_ref_known(v___x_230_, 2);
v___x_233_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_232_);
if (lean_obj_tag(v___x_233_) == 0)
{
lean_object* v_a_234_; lean_object* v_a_235_; lean_object* v___x_237_; uint8_t v_isShared_238_; uint8_t v_isSharedCheck_250_; 
v_a_234_ = lean_ctor_get(v___x_233_, 0);
v_a_235_ = lean_ctor_get(v___x_233_, 1);
v_isSharedCheck_250_ = !lean_is_exclusive(v___x_233_);
if (v_isSharedCheck_250_ == 0)
{
v___x_237_ = v___x_233_;
v_isShared_238_ = v_isSharedCheck_250_;
goto v_resetjp_236_;
}
else
{
lean_inc(v_a_235_);
lean_inc(v_a_234_);
lean_dec(v___x_233_);
v___x_237_ = lean_box(0);
v_isShared_238_ = v_isSharedCheck_250_;
goto v_resetjp_236_;
}
v_resetjp_236_:
{
uint32_t v___x_239_; uint64_t v___x_240_; uint32_t v___x_241_; uint64_t v___x_242_; uint64_t v___x_243_; uint64_t v___x_244_; uint64_t v___x_245_; lean_object* v___x_246_; lean_object* v___x_248_; 
v___x_239_ = lean_unbox_uint32(v_a_231_);
lean_dec(v_a_231_);
v___x_240_ = lean_uint32_to_uint64(v___x_239_);
v___x_241_ = lean_unbox_uint32(v_a_234_);
lean_dec(v_a_234_);
v___x_242_ = lean_uint32_to_uint64(v___x_241_);
v___x_243_ = 32ULL;
v___x_244_ = lean_uint64_shift_left(v___x_242_, v___x_243_);
v___x_245_ = lean_uint64_lor(v___x_240_, v___x_244_);
v___x_246_ = lean_box_uint64(v___x_245_);
if (v_isShared_238_ == 0)
{
lean_ctor_set(v___x_237_, 0, v___x_246_);
v___x_248_ = v___x_237_;
goto v_reusejp_247_;
}
else
{
lean_object* v_reuseFailAlloc_249_; 
v_reuseFailAlloc_249_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_249_, 0, v___x_246_);
lean_ctor_set(v_reuseFailAlloc_249_, 1, v_a_235_);
v___x_248_ = v_reuseFailAlloc_249_;
goto v_reusejp_247_;
}
v_reusejp_247_:
{
return v___x_248_;
}
}
}
else
{
lean_object* v_a_251_; lean_object* v_a_252_; lean_object* v___x_254_; uint8_t v_isShared_255_; uint8_t v_isSharedCheck_259_; 
lean_dec(v_a_231_);
v_a_251_ = lean_ctor_get(v___x_233_, 0);
v_a_252_ = lean_ctor_get(v___x_233_, 1);
v_isSharedCheck_259_ = !lean_is_exclusive(v___x_233_);
if (v_isSharedCheck_259_ == 0)
{
v___x_254_ = v___x_233_;
v_isShared_255_ = v_isSharedCheck_259_;
goto v_resetjp_253_;
}
else
{
lean_inc(v_a_252_);
lean_inc(v_a_251_);
lean_dec(v___x_233_);
v___x_254_ = lean_box(0);
v_isShared_255_ = v_isSharedCheck_259_;
goto v_resetjp_253_;
}
v_resetjp_253_:
{
lean_object* v___x_257_; 
if (v_isShared_255_ == 0)
{
v___x_257_ = v___x_254_;
goto v_reusejp_256_;
}
else
{
lean_object* v_reuseFailAlloc_258_; 
v_reuseFailAlloc_258_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_258_, 0, v_a_251_);
lean_ctor_set(v_reuseFailAlloc_258_, 1, v_a_252_);
v___x_257_ = v_reuseFailAlloc_258_;
goto v_reusejp_256_;
}
v_reusejp_256_:
{
return v___x_257_;
}
}
}
}
else
{
lean_object* v_a_260_; lean_object* v_a_261_; lean_object* v___x_263_; uint8_t v_isShared_264_; uint8_t v_isSharedCheck_268_; 
v_a_260_ = lean_ctor_get(v___x_230_, 0);
v_a_261_ = lean_ctor_get(v___x_230_, 1);
v_isSharedCheck_268_ = !lean_is_exclusive(v___x_230_);
if (v_isSharedCheck_268_ == 0)
{
v___x_263_ = v___x_230_;
v_isShared_264_ = v_isSharedCheck_268_;
goto v_resetjp_262_;
}
else
{
lean_inc(v_a_261_);
lean_inc(v_a_260_);
lean_dec(v___x_230_);
v___x_263_ = lean_box(0);
v_isShared_264_ = v_isSharedCheck_268_;
goto v_resetjp_262_;
}
v_resetjp_262_:
{
lean_object* v___x_266_; 
if (v_isShared_264_ == 0)
{
v___x_266_ = v___x_263_;
goto v_reusejp_265_;
}
else
{
lean_object* v_reuseFailAlloc_267_; 
v_reuseFailAlloc_267_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_267_, 0, v_a_260_);
lean_ctor_set(v_reuseFailAlloc_267_, 1, v_a_261_);
v___x_266_ = v_reuseFailAlloc_267_;
goto v_reusejp_265_;
}
v_reusejp_265_:
{
return v___x_266_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readNat(lean_object* v_a_269_){
_start:
{
lean_object* v___x_270_; 
v___x_270_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_269_);
if (lean_obj_tag(v___x_270_) == 0)
{
lean_object* v_a_271_; lean_object* v_a_272_; lean_object* v___x_274_; uint8_t v_isShared_275_; uint8_t v_isSharedCheck_281_; 
v_a_271_ = lean_ctor_get(v___x_270_, 0);
v_a_272_ = lean_ctor_get(v___x_270_, 1);
v_isSharedCheck_281_ = !lean_is_exclusive(v___x_270_);
if (v_isSharedCheck_281_ == 0)
{
v___x_274_ = v___x_270_;
v_isShared_275_ = v_isSharedCheck_281_;
goto v_resetjp_273_;
}
else
{
lean_inc(v_a_272_);
lean_inc(v_a_271_);
lean_dec(v___x_270_);
v___x_274_ = lean_box(0);
v_isShared_275_ = v_isSharedCheck_281_;
goto v_resetjp_273_;
}
v_resetjp_273_:
{
uint32_t v___x_276_; lean_object* v___x_277_; lean_object* v___x_279_; 
v___x_276_ = lean_unbox_uint32(v_a_271_);
lean_dec(v_a_271_);
v___x_277_ = lean_uint32_to_nat(v___x_276_);
if (v_isShared_275_ == 0)
{
lean_ctor_set(v___x_274_, 0, v___x_277_);
v___x_279_ = v___x_274_;
goto v_reusejp_278_;
}
else
{
lean_object* v_reuseFailAlloc_280_; 
v_reuseFailAlloc_280_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_280_, 0, v___x_277_);
lean_ctor_set(v_reuseFailAlloc_280_, 1, v_a_272_);
v___x_279_ = v_reuseFailAlloc_280_;
goto v_reusejp_278_;
}
v_reusejp_278_:
{
return v___x_279_;
}
}
}
else
{
lean_object* v_a_282_; lean_object* v_a_283_; lean_object* v___x_285_; uint8_t v_isShared_286_; uint8_t v_isSharedCheck_290_; 
v_a_282_ = lean_ctor_get(v___x_270_, 0);
v_a_283_ = lean_ctor_get(v___x_270_, 1);
v_isSharedCheck_290_ = !lean_is_exclusive(v___x_270_);
if (v_isSharedCheck_290_ == 0)
{
v___x_285_ = v___x_270_;
v_isShared_286_ = v_isSharedCheck_290_;
goto v_resetjp_284_;
}
else
{
lean_inc(v_a_283_);
lean_inc(v_a_282_);
lean_dec(v___x_270_);
v___x_285_ = lean_box(0);
v_isShared_286_ = v_isSharedCheck_290_;
goto v_resetjp_284_;
}
v_resetjp_284_:
{
lean_object* v___x_288_; 
if (v_isShared_286_ == 0)
{
v___x_288_ = v___x_285_;
goto v_reusejp_287_;
}
else
{
lean_object* v_reuseFailAlloc_289_; 
v_reuseFailAlloc_289_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_289_, 0, v_a_282_);
lean_ctor_set(v_reuseFailAlloc_289_, 1, v_a_283_);
v___x_288_ = v_reuseFailAlloc_289_;
goto v_reusejp_287_;
}
v_reusejp_287_:
{
return v___x_288_;
}
}
}
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readInt32___closed__0(void){
_start:
{
lean_object* v___x_291_; lean_object* v___x_292_; 
v___x_291_ = lean_cstr_to_nat("4294967296");
v___x_292_ = lean_nat_to_int(v___x_291_);
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readInt32(lean_object* v_a_293_){
_start:
{
lean_object* v___x_294_; 
v___x_294_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_293_);
if (lean_obj_tag(v___x_294_) == 0)
{
lean_object* v_a_295_; lean_object* v_a_296_; lean_object* v___x_298_; uint8_t v_isShared_299_; uint8_t v_isSharedCheck_314_; 
v_a_295_ = lean_ctor_get(v___x_294_, 0);
v_a_296_ = lean_ctor_get(v___x_294_, 1);
v_isSharedCheck_314_ = !lean_is_exclusive(v___x_294_);
if (v_isSharedCheck_314_ == 0)
{
v___x_298_ = v___x_294_;
v_isShared_299_ = v_isSharedCheck_314_;
goto v_resetjp_297_;
}
else
{
lean_inc(v_a_296_);
lean_inc(v_a_295_);
lean_dec(v___x_294_);
v___x_298_ = lean_box(0);
v_isShared_299_ = v_isSharedCheck_314_;
goto v_resetjp_297_;
}
v_resetjp_297_:
{
uint32_t v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; uint8_t v___x_303_; 
v___x_300_ = lean_unbox_uint32(v_a_295_);
lean_dec(v_a_295_);
v___x_301_ = lean_uint32_to_nat(v___x_300_);
v___x_302_ = lean_unsigned_to_nat(2147483648u);
v___x_303_ = lean_nat_dec_lt(v___x_301_, v___x_302_);
if (v___x_303_ == 0)
{
lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_308_; 
v___x_304_ = lean_nat_to_int(v___x_301_);
v___x_305_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readInt32___closed__0, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readInt32___closed__0_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readInt32___closed__0);
v___x_306_ = lean_int_sub(v___x_304_, v___x_305_);
lean_dec(v___x_304_);
if (v_isShared_299_ == 0)
{
lean_ctor_set(v___x_298_, 0, v___x_306_);
v___x_308_ = v___x_298_;
goto v_reusejp_307_;
}
else
{
lean_object* v_reuseFailAlloc_309_; 
v_reuseFailAlloc_309_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_309_, 0, v___x_306_);
lean_ctor_set(v_reuseFailAlloc_309_, 1, v_a_296_);
v___x_308_ = v_reuseFailAlloc_309_;
goto v_reusejp_307_;
}
v_reusejp_307_:
{
return v___x_308_;
}
}
else
{
lean_object* v___x_310_; lean_object* v___x_312_; 
v___x_310_ = lean_nat_to_int(v___x_301_);
if (v_isShared_299_ == 0)
{
lean_ctor_set(v___x_298_, 0, v___x_310_);
v___x_312_ = v___x_298_;
goto v_reusejp_311_;
}
else
{
lean_object* v_reuseFailAlloc_313_; 
v_reuseFailAlloc_313_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_313_, 0, v___x_310_);
lean_ctor_set(v_reuseFailAlloc_313_, 1, v_a_296_);
v___x_312_ = v_reuseFailAlloc_313_;
goto v_reusejp_311_;
}
v_reusejp_311_:
{
return v___x_312_;
}
}
}
}
else
{
lean_object* v_a_315_; lean_object* v_a_316_; lean_object* v___x_318_; uint8_t v_isShared_319_; uint8_t v_isSharedCheck_323_; 
v_a_315_ = lean_ctor_get(v___x_294_, 0);
v_a_316_ = lean_ctor_get(v___x_294_, 1);
v_isSharedCheck_323_ = !lean_is_exclusive(v___x_294_);
if (v_isSharedCheck_323_ == 0)
{
v___x_318_ = v___x_294_;
v_isShared_319_ = v_isSharedCheck_323_;
goto v_resetjp_317_;
}
else
{
lean_inc(v_a_316_);
lean_inc(v_a_315_);
lean_dec(v___x_294_);
v___x_318_ = lean_box(0);
v_isShared_319_ = v_isSharedCheck_323_;
goto v_resetjp_317_;
}
v_resetjp_317_:
{
lean_object* v___x_321_; 
if (v_isShared_319_ == 0)
{
v___x_321_ = v___x_318_;
goto v_reusejp_320_;
}
else
{
lean_object* v_reuseFailAlloc_322_; 
v_reuseFailAlloc_322_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_322_, 0, v_a_315_);
lean_ctor_set(v_reuseFailAlloc_322_, 1, v_a_316_);
v___x_321_ = v_reuseFailAlloc_322_;
goto v_reusejp_320_;
}
v_reusejp_320_:
{
return v___x_321_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBool(lean_object* v_a_325_){
_start:
{
lean_object* v___x_326_; 
lean_inc_ref(v_a_325_);
v___x_326_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readByte(v_a_325_);
if (lean_obj_tag(v___x_326_) == 0)
{
lean_object* v_a_327_; lean_object* v_a_328_; lean_object* v___x_330_; uint8_t v_isShared_331_; uint8_t v_isSharedCheck_364_; 
v_a_327_ = lean_ctor_get(v___x_326_, 0);
v_a_328_ = lean_ctor_get(v___x_326_, 1);
v_isSharedCheck_364_ = !lean_is_exclusive(v___x_326_);
if (v_isSharedCheck_364_ == 0)
{
v___x_330_ = v___x_326_;
v_isShared_331_ = v_isSharedCheck_364_;
goto v_resetjp_329_;
}
else
{
lean_inc(v_a_328_);
lean_inc(v_a_327_);
lean_dec(v___x_326_);
v___x_330_ = lean_box(0);
v_isShared_331_ = v_isSharedCheck_364_;
goto v_resetjp_329_;
}
v_resetjp_329_:
{
uint8_t v___x_332_; uint8_t v___x_333_; uint8_t v___x_334_; 
v___x_332_ = 0;
v___x_333_ = lean_unbox(v_a_327_);
v___x_334_ = lean_uint8_dec_eq(v___x_333_, v___x_332_);
if (v___x_334_ == 0)
{
uint8_t v___x_335_; uint8_t v___x_336_; uint8_t v___x_337_; 
v___x_335_ = 1;
v___x_336_ = lean_unbox(v_a_327_);
v___x_337_ = lean_uint8_dec_eq(v___x_336_, v___x_335_);
if (v___x_337_ == 0)
{
lean_object* v_offset_338_; lean_object* v___x_340_; uint8_t v_isShared_341_; uint8_t v_isSharedCheck_353_; 
v_offset_338_ = lean_ctor_get(v_a_325_, 1);
v_isSharedCheck_353_ = !lean_is_exclusive(v_a_325_);
if (v_isSharedCheck_353_ == 0)
{
lean_object* v_unused_354_; 
v_unused_354_ = lean_ctor_get(v_a_325_, 0);
lean_dec(v_unused_354_);
v___x_340_ = v_a_325_;
v_isShared_341_ = v_isSharedCheck_353_;
goto v_resetjp_339_;
}
else
{
lean_inc(v_offset_338_);
lean_dec(v_a_325_);
v___x_340_ = lean_box(0);
v_isShared_341_ = v_isSharedCheck_353_;
goto v_resetjp_339_;
}
v_resetjp_339_:
{
lean_object* v___x_342_; uint8_t v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_348_; 
v___x_342_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBool___closed__0));
v___x_343_ = lean_unbox(v_a_327_);
lean_dec(v_a_327_);
v___x_344_ = lean_uint8_to_nat(v___x_343_);
v___x_345_ = l_Nat_reprFast(v___x_344_);
v___x_346_ = lean_string_append(v___x_342_, v___x_345_);
lean_dec_ref(v___x_345_);
if (v_isShared_341_ == 0)
{
lean_ctor_set_tag(v___x_340_, 3);
lean_ctor_set(v___x_340_, 1, v___x_346_);
lean_ctor_set(v___x_340_, 0, v_offset_338_);
v___x_348_ = v___x_340_;
goto v_reusejp_347_;
}
else
{
lean_object* v_reuseFailAlloc_352_; 
v_reuseFailAlloc_352_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_reuseFailAlloc_352_, 0, v_offset_338_);
lean_ctor_set(v_reuseFailAlloc_352_, 1, v___x_346_);
v___x_348_ = v_reuseFailAlloc_352_;
goto v_reusejp_347_;
}
v_reusejp_347_:
{
lean_object* v___x_350_; 
if (v_isShared_331_ == 0)
{
lean_ctor_set_tag(v___x_330_, 1);
lean_ctor_set(v___x_330_, 0, v___x_348_);
v___x_350_ = v___x_330_;
goto v_reusejp_349_;
}
else
{
lean_object* v_reuseFailAlloc_351_; 
v_reuseFailAlloc_351_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_351_, 0, v___x_348_);
lean_ctor_set(v_reuseFailAlloc_351_, 1, v_a_328_);
v___x_350_ = v_reuseFailAlloc_351_;
goto v_reusejp_349_;
}
v_reusejp_349_:
{
return v___x_350_;
}
}
}
}
else
{
lean_object* v___x_355_; lean_object* v___x_357_; 
lean_dec(v_a_327_);
lean_dec_ref(v_a_325_);
v___x_355_ = lean_box(v___x_337_);
if (v_isShared_331_ == 0)
{
lean_ctor_set(v___x_330_, 0, v___x_355_);
v___x_357_ = v___x_330_;
goto v_reusejp_356_;
}
else
{
lean_object* v_reuseFailAlloc_358_; 
v_reuseFailAlloc_358_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_358_, 0, v___x_355_);
lean_ctor_set(v_reuseFailAlloc_358_, 1, v_a_328_);
v___x_357_ = v_reuseFailAlloc_358_;
goto v_reusejp_356_;
}
v_reusejp_356_:
{
return v___x_357_;
}
}
}
else
{
uint8_t v___x_359_; lean_object* v___x_360_; lean_object* v___x_362_; 
lean_dec(v_a_327_);
lean_dec_ref(v_a_325_);
v___x_359_ = 0;
v___x_360_ = lean_box(v___x_359_);
if (v_isShared_331_ == 0)
{
lean_ctor_set(v___x_330_, 0, v___x_360_);
v___x_362_ = v___x_330_;
goto v_reusejp_361_;
}
else
{
lean_object* v_reuseFailAlloc_363_; 
v_reuseFailAlloc_363_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_363_, 0, v___x_360_);
lean_ctor_set(v_reuseFailAlloc_363_, 1, v_a_328_);
v___x_362_ = v_reuseFailAlloc_363_;
goto v_reusejp_361_;
}
v_reusejp_361_:
{
return v___x_362_;
}
}
}
}
else
{
lean_object* v_a_365_; lean_object* v_a_366_; lean_object* v___x_368_; uint8_t v_isShared_369_; uint8_t v_isSharedCheck_373_; 
lean_dec_ref(v_a_325_);
v_a_365_ = lean_ctor_get(v___x_326_, 0);
v_a_366_ = lean_ctor_get(v___x_326_, 1);
v_isSharedCheck_373_ = !lean_is_exclusive(v___x_326_);
if (v_isSharedCheck_373_ == 0)
{
v___x_368_ = v___x_326_;
v_isShared_369_ = v_isSharedCheck_373_;
goto v_resetjp_367_;
}
else
{
lean_inc(v_a_366_);
lean_inc(v_a_365_);
lean_dec(v___x_326_);
v___x_368_ = lean_box(0);
v_isShared_369_ = v_isSharedCheck_373_;
goto v_resetjp_367_;
}
v_resetjp_367_:
{
lean_object* v___x_371_; 
if (v_isShared_369_ == 0)
{
v___x_371_ = v___x_368_;
goto v_reusejp_370_;
}
else
{
lean_object* v_reuseFailAlloc_372_; 
v_reuseFailAlloc_372_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_372_, 0, v_a_365_);
lean_ctor_set(v_reuseFailAlloc_372_, 1, v_a_366_);
v___x_371_ = v_reuseFailAlloc_372_;
goto v_reusejp_370_;
}
v_reusejp_370_:
{
return v___x_371_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readCanonicalFBB(lean_object* v_a_375_){
_start:
{
lean_object* v___x_376_; 
lean_inc_ref(v_a_375_);
v___x_376_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_375_);
if (lean_obj_tag(v___x_376_) == 0)
{
lean_object* v_a_377_; lean_object* v_a_378_; uint32_t v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; uint8_t v___x_382_; 
v_a_377_ = lean_ctor_get(v___x_376_, 0);
lean_inc(v_a_377_);
v_a_378_ = lean_ctor_get(v___x_376_, 1);
lean_inc(v_a_378_);
v___x_379_ = lean_unbox_uint32(v_a_377_);
lean_dec(v_a_377_);
v___x_380_ = lean_uint32_to_nat(v___x_379_);
v___x_381_ = lean_unsigned_to_nat(2013265921u);
v___x_382_ = lean_nat_dec_lt(v___x_380_, v___x_381_);
if (v___x_382_ == 0)
{
lean_object* v___x_384_; uint8_t v_isShared_385_; uint8_t v_isSharedCheck_401_; 
v_isSharedCheck_401_ = !lean_is_exclusive(v___x_376_);
if (v_isSharedCheck_401_ == 0)
{
lean_object* v_unused_402_; lean_object* v_unused_403_; 
v_unused_402_ = lean_ctor_get(v___x_376_, 1);
lean_dec(v_unused_402_);
v_unused_403_ = lean_ctor_get(v___x_376_, 0);
lean_dec(v_unused_403_);
v___x_384_ = v___x_376_;
v_isShared_385_ = v_isSharedCheck_401_;
goto v_resetjp_383_;
}
else
{
lean_dec(v___x_376_);
v___x_384_ = lean_box(0);
v_isShared_385_ = v_isSharedCheck_401_;
goto v_resetjp_383_;
}
v_resetjp_383_:
{
lean_object* v_offset_386_; lean_object* v___x_388_; uint8_t v_isShared_389_; uint8_t v_isSharedCheck_399_; 
v_offset_386_ = lean_ctor_get(v_a_375_, 1);
v_isSharedCheck_399_ = !lean_is_exclusive(v_a_375_);
if (v_isSharedCheck_399_ == 0)
{
lean_object* v_unused_400_; 
v_unused_400_ = lean_ctor_get(v_a_375_, 0);
lean_dec(v_unused_400_);
v___x_388_ = v_a_375_;
v_isShared_389_ = v_isSharedCheck_399_;
goto v_resetjp_387_;
}
else
{
lean_inc(v_offset_386_);
lean_dec(v_a_375_);
v___x_388_ = lean_box(0);
v_isShared_389_ = v_isSharedCheck_399_;
goto v_resetjp_387_;
}
v_resetjp_387_:
{
lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_394_; 
v___x_390_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readCanonicalFBB___closed__0));
v___x_391_ = l_Nat_reprFast(v___x_380_);
v___x_392_ = lean_string_append(v___x_390_, v___x_391_);
lean_dec_ref(v___x_391_);
if (v_isShared_389_ == 0)
{
lean_ctor_set_tag(v___x_388_, 3);
lean_ctor_set(v___x_388_, 1, v___x_392_);
lean_ctor_set(v___x_388_, 0, v_offset_386_);
v___x_394_ = v___x_388_;
goto v_reusejp_393_;
}
else
{
lean_object* v_reuseFailAlloc_398_; 
v_reuseFailAlloc_398_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_reuseFailAlloc_398_, 0, v_offset_386_);
lean_ctor_set(v_reuseFailAlloc_398_, 1, v___x_392_);
v___x_394_ = v_reuseFailAlloc_398_;
goto v_reusejp_393_;
}
v_reusejp_393_:
{
lean_object* v___x_396_; 
if (v_isShared_385_ == 0)
{
lean_ctor_set_tag(v___x_384_, 1);
lean_ctor_set(v___x_384_, 0, v___x_394_);
v___x_396_ = v___x_384_;
goto v_reusejp_395_;
}
else
{
lean_object* v_reuseFailAlloc_397_; 
v_reuseFailAlloc_397_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_397_, 0, v___x_394_);
lean_ctor_set(v_reuseFailAlloc_397_, 1, v_a_378_);
v___x_396_ = v_reuseFailAlloc_397_;
goto v_reusejp_395_;
}
v_reusejp_395_:
{
return v___x_396_;
}
}
}
}
}
else
{
lean_dec(v___x_380_);
lean_dec(v_a_378_);
lean_dec_ref(v_a_375_);
return v___x_376_;
}
}
else
{
lean_dec_ref(v_a_375_);
return v___x_376_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readOption___redArg(lean_object* v_read_405_, lean_object* v_a_406_){
_start:
{
lean_object* v___x_407_; 
lean_inc_ref(v_a_406_);
v___x_407_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readByte(v_a_406_);
if (lean_obj_tag(v___x_407_) == 0)
{
lean_object* v_a_408_; lean_object* v_a_409_; lean_object* v___x_411_; uint8_t v_isShared_412_; uint8_t v_isSharedCheck_460_; 
v_a_408_ = lean_ctor_get(v___x_407_, 0);
v_a_409_ = lean_ctor_get(v___x_407_, 1);
v_isSharedCheck_460_ = !lean_is_exclusive(v___x_407_);
if (v_isSharedCheck_460_ == 0)
{
v___x_411_ = v___x_407_;
v_isShared_412_ = v_isSharedCheck_460_;
goto v_resetjp_410_;
}
else
{
lean_inc(v_a_409_);
lean_inc(v_a_408_);
lean_dec(v___x_407_);
v___x_411_ = lean_box(0);
v_isShared_412_ = v_isSharedCheck_460_;
goto v_resetjp_410_;
}
v_resetjp_410_:
{
uint8_t v___x_413_; uint8_t v___x_414_; uint8_t v___x_415_; 
v___x_413_ = 0;
v___x_414_ = lean_unbox(v_a_408_);
v___x_415_ = lean_uint8_dec_eq(v___x_414_, v___x_413_);
if (v___x_415_ == 0)
{
uint8_t v___x_416_; uint8_t v___x_417_; uint8_t v___x_418_; 
v___x_416_ = 1;
v___x_417_ = lean_unbox(v_a_408_);
v___x_418_ = lean_uint8_dec_eq(v___x_417_, v___x_416_);
if (v___x_418_ == 0)
{
lean_object* v_offset_419_; lean_object* v___x_421_; uint8_t v_isShared_422_; uint8_t v_isSharedCheck_434_; 
lean_dec_ref(v_read_405_);
v_offset_419_ = lean_ctor_get(v_a_406_, 1);
v_isSharedCheck_434_ = !lean_is_exclusive(v_a_406_);
if (v_isSharedCheck_434_ == 0)
{
lean_object* v_unused_435_; 
v_unused_435_ = lean_ctor_get(v_a_406_, 0);
lean_dec(v_unused_435_);
v___x_421_ = v_a_406_;
v_isShared_422_ = v_isSharedCheck_434_;
goto v_resetjp_420_;
}
else
{
lean_inc(v_offset_419_);
lean_dec(v_a_406_);
v___x_421_ = lean_box(0);
v_isShared_422_ = v_isSharedCheck_434_;
goto v_resetjp_420_;
}
v_resetjp_420_:
{
lean_object* v___x_423_; uint8_t v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_429_; 
v___x_423_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readOption___redArg___closed__0));
v___x_424_ = lean_unbox(v_a_408_);
lean_dec(v_a_408_);
v___x_425_ = lean_uint8_to_nat(v___x_424_);
v___x_426_ = l_Nat_reprFast(v___x_425_);
v___x_427_ = lean_string_append(v___x_423_, v___x_426_);
lean_dec_ref(v___x_426_);
if (v_isShared_422_ == 0)
{
lean_ctor_set_tag(v___x_421_, 3);
lean_ctor_set(v___x_421_, 1, v___x_427_);
lean_ctor_set(v___x_421_, 0, v_offset_419_);
v___x_429_ = v___x_421_;
goto v_reusejp_428_;
}
else
{
lean_object* v_reuseFailAlloc_433_; 
v_reuseFailAlloc_433_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_reuseFailAlloc_433_, 0, v_offset_419_);
lean_ctor_set(v_reuseFailAlloc_433_, 1, v___x_427_);
v___x_429_ = v_reuseFailAlloc_433_;
goto v_reusejp_428_;
}
v_reusejp_428_:
{
lean_object* v___x_431_; 
if (v_isShared_412_ == 0)
{
lean_ctor_set_tag(v___x_411_, 1);
lean_ctor_set(v___x_411_, 0, v___x_429_);
v___x_431_ = v___x_411_;
goto v_reusejp_430_;
}
else
{
lean_object* v_reuseFailAlloc_432_; 
v_reuseFailAlloc_432_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_432_, 0, v___x_429_);
lean_ctor_set(v_reuseFailAlloc_432_, 1, v_a_409_);
v___x_431_ = v_reuseFailAlloc_432_;
goto v_reusejp_430_;
}
v_reusejp_430_:
{
return v___x_431_;
}
}
}
}
else
{
lean_object* v___x_436_; 
lean_del_object(v___x_411_);
lean_dec(v_a_408_);
lean_dec_ref(v_a_406_);
v___x_436_ = lean_apply_1(v_read_405_, v_a_409_);
if (lean_obj_tag(v___x_436_) == 0)
{
lean_object* v_a_437_; lean_object* v_a_438_; lean_object* v___x_440_; uint8_t v_isShared_441_; uint8_t v_isSharedCheck_446_; 
v_a_437_ = lean_ctor_get(v___x_436_, 0);
v_a_438_ = lean_ctor_get(v___x_436_, 1);
v_isSharedCheck_446_ = !lean_is_exclusive(v___x_436_);
if (v_isSharedCheck_446_ == 0)
{
v___x_440_ = v___x_436_;
v_isShared_441_ = v_isSharedCheck_446_;
goto v_resetjp_439_;
}
else
{
lean_inc(v_a_438_);
lean_inc(v_a_437_);
lean_dec(v___x_436_);
v___x_440_ = lean_box(0);
v_isShared_441_ = v_isSharedCheck_446_;
goto v_resetjp_439_;
}
v_resetjp_439_:
{
lean_object* v___x_442_; lean_object* v___x_444_; 
v___x_442_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_442_, 0, v_a_437_);
if (v_isShared_441_ == 0)
{
lean_ctor_set(v___x_440_, 0, v___x_442_);
v___x_444_ = v___x_440_;
goto v_reusejp_443_;
}
else
{
lean_object* v_reuseFailAlloc_445_; 
v_reuseFailAlloc_445_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_445_, 0, v___x_442_);
lean_ctor_set(v_reuseFailAlloc_445_, 1, v_a_438_);
v___x_444_ = v_reuseFailAlloc_445_;
goto v_reusejp_443_;
}
v_reusejp_443_:
{
return v___x_444_;
}
}
}
else
{
lean_object* v_a_447_; lean_object* v_a_448_; lean_object* v___x_450_; uint8_t v_isShared_451_; uint8_t v_isSharedCheck_455_; 
v_a_447_ = lean_ctor_get(v___x_436_, 0);
v_a_448_ = lean_ctor_get(v___x_436_, 1);
v_isSharedCheck_455_ = !lean_is_exclusive(v___x_436_);
if (v_isSharedCheck_455_ == 0)
{
v___x_450_ = v___x_436_;
v_isShared_451_ = v_isSharedCheck_455_;
goto v_resetjp_449_;
}
else
{
lean_inc(v_a_448_);
lean_inc(v_a_447_);
lean_dec(v___x_436_);
v___x_450_ = lean_box(0);
v_isShared_451_ = v_isSharedCheck_455_;
goto v_resetjp_449_;
}
v_resetjp_449_:
{
lean_object* v___x_453_; 
if (v_isShared_451_ == 0)
{
v___x_453_ = v___x_450_;
goto v_reusejp_452_;
}
else
{
lean_object* v_reuseFailAlloc_454_; 
v_reuseFailAlloc_454_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_454_, 0, v_a_447_);
lean_ctor_set(v_reuseFailAlloc_454_, 1, v_a_448_);
v___x_453_ = v_reuseFailAlloc_454_;
goto v_reusejp_452_;
}
v_reusejp_452_:
{
return v___x_453_;
}
}
}
}
}
else
{
lean_object* v___x_456_; lean_object* v___x_458_; 
lean_dec(v_a_408_);
lean_dec_ref(v_a_406_);
lean_dec_ref(v_read_405_);
v___x_456_ = lean_box(0);
if (v_isShared_412_ == 0)
{
lean_ctor_set(v___x_411_, 0, v___x_456_);
v___x_458_ = v___x_411_;
goto v_reusejp_457_;
}
else
{
lean_object* v_reuseFailAlloc_459_; 
v_reuseFailAlloc_459_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_459_, 0, v___x_456_);
lean_ctor_set(v_reuseFailAlloc_459_, 1, v_a_409_);
v___x_458_ = v_reuseFailAlloc_459_;
goto v_reusejp_457_;
}
v_reusejp_457_:
{
return v___x_458_;
}
}
}
}
else
{
lean_object* v_a_461_; lean_object* v_a_462_; lean_object* v___x_464_; uint8_t v_isShared_465_; uint8_t v_isSharedCheck_469_; 
lean_dec_ref(v_a_406_);
lean_dec_ref(v_read_405_);
v_a_461_ = lean_ctor_get(v___x_407_, 0);
v_a_462_ = lean_ctor_get(v___x_407_, 1);
v_isSharedCheck_469_ = !lean_is_exclusive(v___x_407_);
if (v_isSharedCheck_469_ == 0)
{
v___x_464_ = v___x_407_;
v_isShared_465_ = v_isSharedCheck_469_;
goto v_resetjp_463_;
}
else
{
lean_inc(v_a_462_);
lean_inc(v_a_461_);
lean_dec(v___x_407_);
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
v_reuseFailAlloc_468_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_468_, 0, v_a_461_);
lean_ctor_set(v_reuseFailAlloc_468_, 1, v_a_462_);
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
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readOption(lean_object* v_00_u03b1_470_, lean_object* v_read_471_, lean_object* v_a_472_){
_start:
{
lean_object* v___x_473_; 
v___x_473_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readOption___redArg(v_read_471_, v_a_472_);
return v___x_473_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArrayExact___redArg(lean_object* v_read_474_, lean_object* v_x_475_, lean_object* v_x_476_, lean_object* v_a_477_){
_start:
{
lean_object* v_zero_478_; uint8_t v_isZero_479_; 
v_zero_478_ = lean_unsigned_to_nat(0u);
v_isZero_479_ = lean_nat_dec_eq(v_x_475_, v_zero_478_);
if (v_isZero_479_ == 1)
{
lean_object* v___x_480_; 
lean_dec(v_x_475_);
lean_dec_ref(v_read_474_);
v___x_480_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_480_, 0, v_x_476_);
lean_ctor_set(v___x_480_, 1, v_a_477_);
return v___x_480_;
}
else
{
lean_object* v___x_481_; 
lean_inc_ref(v_read_474_);
v___x_481_ = lean_apply_1(v_read_474_, v_a_477_);
if (lean_obj_tag(v___x_481_) == 0)
{
lean_object* v_a_482_; lean_object* v_a_483_; lean_object* v_one_484_; lean_object* v_n_485_; lean_object* v___x_486_; 
v_a_482_ = lean_ctor_get(v___x_481_, 0);
lean_inc(v_a_482_);
v_a_483_ = lean_ctor_get(v___x_481_, 1);
lean_inc(v_a_483_);
lean_dec_ref_known(v___x_481_, 2);
v_one_484_ = lean_unsigned_to_nat(1u);
v_n_485_ = lean_nat_sub(v_x_475_, v_one_484_);
lean_dec(v_x_475_);
v___x_486_ = lean_array_push(v_x_476_, v_a_482_);
v_x_475_ = v_n_485_;
v_x_476_ = v___x_486_;
v_a_477_ = v_a_483_;
goto _start;
}
else
{
lean_object* v_a_488_; lean_object* v_a_489_; lean_object* v___x_491_; uint8_t v_isShared_492_; uint8_t v_isSharedCheck_496_; 
lean_dec_ref(v_x_476_);
lean_dec(v_x_475_);
lean_dec_ref(v_read_474_);
v_a_488_ = lean_ctor_get(v___x_481_, 0);
v_a_489_ = lean_ctor_get(v___x_481_, 1);
v_isSharedCheck_496_ = !lean_is_exclusive(v___x_481_);
if (v_isSharedCheck_496_ == 0)
{
v___x_491_ = v___x_481_;
v_isShared_492_ = v_isSharedCheck_496_;
goto v_resetjp_490_;
}
else
{
lean_inc(v_a_489_);
lean_inc(v_a_488_);
lean_dec(v___x_481_);
v___x_491_ = lean_box(0);
v_isShared_492_ = v_isSharedCheck_496_;
goto v_resetjp_490_;
}
v_resetjp_490_:
{
lean_object* v___x_494_; 
if (v_isShared_492_ == 0)
{
v___x_494_ = v___x_491_;
goto v_reusejp_493_;
}
else
{
lean_object* v_reuseFailAlloc_495_; 
v_reuseFailAlloc_495_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_495_, 0, v_a_488_);
lean_ctor_set(v_reuseFailAlloc_495_, 1, v_a_489_);
v___x_494_ = v_reuseFailAlloc_495_;
goto v_reusejp_493_;
}
v_reusejp_493_:
{
return v___x_494_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArrayExact(lean_object* v_00_u03b1_497_, lean_object* v_read_498_, lean_object* v_x_499_, lean_object* v_x_500_, lean_object* v_a_501_){
_start:
{
lean_object* v___x_502_; 
v___x_502_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArrayExact___redArg(v_read_498_, v_x_499_, v_x_500_, v_a_501_);
return v___x_502_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(lean_object* v_read_503_, lean_object* v_a_504_){
_start:
{
lean_object* v___x_505_; 
v___x_505_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readNat(v_a_504_);
if (lean_obj_tag(v___x_505_) == 0)
{
lean_object* v_a_506_; lean_object* v_a_507_; lean_object* v___x_508_; lean_object* v___x_509_; 
v_a_506_ = lean_ctor_get(v___x_505_, 0);
lean_inc(v_a_506_);
v_a_507_ = lean_ctor_get(v___x_505_, 1);
lean_inc(v_a_507_);
lean_dec_ref_known(v___x_505_, 2);
v___x_508_ = lean_mk_empty_array_with_capacity(v_a_506_);
v___x_509_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArrayExact___redArg(v_read_503_, v_a_506_, v___x_508_, v_a_507_);
return v___x_509_;
}
else
{
lean_object* v_a_510_; lean_object* v_a_511_; lean_object* v___x_513_; uint8_t v_isShared_514_; uint8_t v_isSharedCheck_518_; 
lean_dec_ref(v_read_503_);
v_a_510_ = lean_ctor_get(v___x_505_, 0);
v_a_511_ = lean_ctor_get(v___x_505_, 1);
v_isSharedCheck_518_ = !lean_is_exclusive(v___x_505_);
if (v_isSharedCheck_518_ == 0)
{
v___x_513_ = v___x_505_;
v_isShared_514_ = v_isSharedCheck_518_;
goto v_resetjp_512_;
}
else
{
lean_inc(v_a_511_);
lean_inc(v_a_510_);
lean_dec(v___x_505_);
v___x_513_ = lean_box(0);
v_isShared_514_ = v_isSharedCheck_518_;
goto v_resetjp_512_;
}
v_resetjp_512_:
{
lean_object* v___x_516_; 
if (v_isShared_514_ == 0)
{
v___x_516_ = v___x_513_;
goto v_reusejp_515_;
}
else
{
lean_object* v_reuseFailAlloc_517_; 
v_reuseFailAlloc_517_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_517_, 0, v_a_510_);
lean_ctor_set(v_reuseFailAlloc_517_, 1, v_a_511_);
v___x_516_ = v_reuseFailAlloc_517_;
goto v_reusejp_515_;
}
v_reusejp_515_:
{
return v___x_516_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr(lean_object* v_00_u03b1_519_, lean_object* v_read_520_, lean_object* v_a_521_){
_start:
{
lean_object* v___x_522_; 
v___x_522_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v_read_520_, v_a_521_);
return v___x_522_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readVectorN___redArg(lean_object* v_read_525_, lean_object* v_x_526_, lean_object* v_a_527_){
_start:
{
lean_object* v_zero_528_; uint8_t v_isZero_529_; 
v_zero_528_ = lean_unsigned_to_nat(0u);
v_isZero_529_ = lean_nat_dec_eq(v_x_526_, v_zero_528_);
if (v_isZero_529_ == 1)
{
lean_object* v___x_530_; lean_object* v___x_531_; 
lean_dec_ref(v_read_525_);
v___x_530_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readVectorN___redArg___closed__0));
v___x_531_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_531_, 0, v___x_530_);
lean_ctor_set(v___x_531_, 1, v_a_527_);
return v___x_531_;
}
else
{
lean_object* v_one_532_; lean_object* v_n_533_; lean_object* v___x_534_; 
v_one_532_ = lean_unsigned_to_nat(1u);
v_n_533_ = lean_nat_sub(v_x_526_, v_one_532_);
lean_inc_ref(v_read_525_);
v___x_534_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readVectorN___redArg(v_read_525_, v_n_533_, v_a_527_);
lean_dec(v_n_533_);
if (lean_obj_tag(v___x_534_) == 0)
{
lean_object* v_a_535_; lean_object* v_a_536_; lean_object* v___x_537_; 
v_a_535_ = lean_ctor_get(v___x_534_, 0);
lean_inc(v_a_535_);
v_a_536_ = lean_ctor_get(v___x_534_, 1);
lean_inc(v_a_536_);
lean_dec_ref_known(v___x_534_, 2);
v___x_537_ = lean_apply_1(v_read_525_, v_a_536_);
if (lean_obj_tag(v___x_537_) == 0)
{
lean_object* v_a_538_; lean_object* v_a_539_; lean_object* v___x_541_; uint8_t v_isShared_542_; uint8_t v_isSharedCheck_547_; 
v_a_538_ = lean_ctor_get(v___x_537_, 0);
v_a_539_ = lean_ctor_get(v___x_537_, 1);
v_isSharedCheck_547_ = !lean_is_exclusive(v___x_537_);
if (v_isSharedCheck_547_ == 0)
{
v___x_541_ = v___x_537_;
v_isShared_542_ = v_isSharedCheck_547_;
goto v_resetjp_540_;
}
else
{
lean_inc(v_a_539_);
lean_inc(v_a_538_);
lean_dec(v___x_537_);
v___x_541_ = lean_box(0);
v_isShared_542_ = v_isSharedCheck_547_;
goto v_resetjp_540_;
}
v_resetjp_540_:
{
lean_object* v___x_543_; lean_object* v___x_545_; 
v___x_543_ = lean_array_push(v_a_535_, v_a_538_);
if (v_isShared_542_ == 0)
{
lean_ctor_set(v___x_541_, 0, v___x_543_);
v___x_545_ = v___x_541_;
goto v_reusejp_544_;
}
else
{
lean_object* v_reuseFailAlloc_546_; 
v_reuseFailAlloc_546_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_546_, 0, v___x_543_);
lean_ctor_set(v_reuseFailAlloc_546_, 1, v_a_539_);
v___x_545_ = v_reuseFailAlloc_546_;
goto v_reusejp_544_;
}
v_reusejp_544_:
{
return v___x_545_;
}
}
}
else
{
lean_object* v_a_548_; lean_object* v_a_549_; lean_object* v___x_551_; uint8_t v_isShared_552_; uint8_t v_isSharedCheck_556_; 
lean_dec(v_a_535_);
v_a_548_ = lean_ctor_get(v___x_537_, 0);
v_a_549_ = lean_ctor_get(v___x_537_, 1);
v_isSharedCheck_556_ = !lean_is_exclusive(v___x_537_);
if (v_isSharedCheck_556_ == 0)
{
v___x_551_ = v___x_537_;
v_isShared_552_ = v_isSharedCheck_556_;
goto v_resetjp_550_;
}
else
{
lean_inc(v_a_549_);
lean_inc(v_a_548_);
lean_dec(v___x_537_);
v___x_551_ = lean_box(0);
v_isShared_552_ = v_isSharedCheck_556_;
goto v_resetjp_550_;
}
v_resetjp_550_:
{
lean_object* v___x_554_; 
if (v_isShared_552_ == 0)
{
v___x_554_ = v___x_551_;
goto v_reusejp_553_;
}
else
{
lean_object* v_reuseFailAlloc_555_; 
v_reuseFailAlloc_555_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_555_, 0, v_a_548_);
lean_ctor_set(v_reuseFailAlloc_555_, 1, v_a_549_);
v___x_554_ = v_reuseFailAlloc_555_;
goto v_reusejp_553_;
}
v_reusejp_553_:
{
return v___x_554_;
}
}
}
}
else
{
lean_dec_ref(v_read_525_);
return v___x_534_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readVectorN___redArg___boxed(lean_object* v_read_557_, lean_object* v_x_558_, lean_object* v_a_559_){
_start:
{
lean_object* v_res_560_; 
v_res_560_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readVectorN___redArg(v_read_557_, v_x_558_, v_a_559_);
lean_dec(v_x_558_);
return v_res_560_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readVectorN(lean_object* v_00_u03b1_561_, lean_object* v_read_562_, lean_object* v_x_563_, lean_object* v_a_564_){
_start:
{
lean_object* v___x_565_; 
v___x_565_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readVectorN___redArg(v_read_562_, v_x_563_, v_a_564_);
return v___x_565_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readVectorN___boxed(lean_object* v_00_u03b1_566_, lean_object* v_read_567_, lean_object* v_x_568_, lean_object* v_a_569_){
_start:
{
lean_object* v_res_570_; 
v_res_570_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readVectorN(v_00_u03b1_566_, v_read_567_, v_x_568_, v_a_569_);
lean_dec(v_x_568_);
return v_res_570_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readDigest(lean_object* v_a_571_){
_start:
{
lean_object* v___x_572_; lean_object* v___x_573_; lean_object* v___x_574_; 
v___x_572_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readCanonicalFBB), 1, 0);
v___x_573_ = lean_unsigned_to_nat(8u);
v___x_574_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readVectorN___redArg(v___x_572_, v___x_573_, v_a_571_);
return v___x_574_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readExt4(lean_object* v_a_575_){
_start:
{
lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; 
v___x_576_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readCanonicalFBB), 1, 0);
v___x_577_ = lean_unsigned_to_nat(4u);
v___x_578_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readVectorN___redArg(v___x_576_, v___x_577_, v_a_575_);
return v___x_578_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_asciiMagic(uint8_t v_a_579_, uint8_t v_b_580_, uint8_t v_c_581_, uint8_t v_d_582_){
_start:
{
lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_593_; 
v___x_583_ = lean_unsigned_to_nat(4u);
v___x_584_ = lean_mk_empty_array_with_capacity(v___x_583_);
v___x_585_ = lean_box(v_a_579_);
v___x_586_ = lean_array_push(v___x_584_, v___x_585_);
v___x_587_ = lean_box(v_b_580_);
v___x_588_ = lean_array_push(v___x_586_, v___x_587_);
v___x_589_ = lean_box(v_c_581_);
v___x_590_ = lean_array_push(v___x_588_, v___x_589_);
v___x_591_ = lean_box(v_d_582_);
v___x_592_ = lean_array_push(v___x_590_, v___x_591_);
v___x_593_ = lean_byte_array_mk(v___x_592_);
return v___x_593_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_asciiMagic___boxed(lean_object* v_a_594_, lean_object* v_b_595_, lean_object* v_c_596_, lean_object* v_d_597_){
_start:
{
uint8_t v_a_boxed_598_; uint8_t v_b_boxed_599_; uint8_t v_c_boxed_600_; uint8_t v_d_boxed_601_; lean_object* v_res_602_; 
v_a_boxed_598_ = lean_unbox(v_a_594_);
v_b_boxed_599_ = lean_unbox(v_b_595_);
v_c_boxed_600_ = lean_unbox(v_c_596_);
v_d_boxed_601_ = lean_unbox(v_d_597_);
v_res_602_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_asciiMagic(v_a_boxed_598_, v_b_boxed_599_, v_c_boxed_600_, v_d_boxed_601_);
return v_res_602_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicProof___closed__0(void){
_start:
{
uint8_t v___x_603_; uint8_t v___x_604_; uint8_t v___x_605_; uint8_t v___x_606_; lean_object* v___x_607_; 
v___x_603_ = 70;
v___x_604_ = 79;
v___x_605_ = 82;
v___x_606_ = 80;
v___x_607_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_asciiMagic(v___x_606_, v___x_605_, v___x_604_, v___x_603_);
return v___x_607_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicProof(void){
_start:
{
lean_object* v___x_608_; 
v___x_608_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicProof___closed__0, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicProof___closed__0_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicProof___closed__0);
return v___x_608_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicVk___closed__0(void){
_start:
{
uint8_t v___x_609_; uint8_t v___x_610_; uint8_t v___x_611_; uint8_t v___x_612_; lean_object* v___x_613_; 
v___x_609_ = 89;
v___x_610_ = 75;
v___x_611_ = 86;
v___x_612_ = 83;
v___x_613_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_asciiMagic(v___x_612_, v___x_611_, v___x_610_, v___x_609_);
return v___x_613_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicVk(void){
_start:
{
lean_object* v___x_614_; 
v___x_614_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicVk___closed__0, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicVk___closed__0_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicVk___closed__0);
return v___x_614_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicPv___closed__0(void){
_start:
{
uint8_t v___x_615_; uint8_t v___x_616_; uint8_t v___x_617_; uint8_t v___x_618_; lean_object* v___x_619_; 
v___x_615_ = 86;
v___x_616_ = 66;
v___x_617_ = 85;
v___x_618_ = 80;
v___x_619_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_asciiMagic(v___x_618_, v___x_617_, v___x_616_, v___x_615_);
return v___x_619_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicPv(void){
_start:
{
lean_object* v___x_620_; 
v___x_620_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicPv___closed__0, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicPv___closed__0_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicPv___closed__0);
return v___x_620_;
}
}
static uint32_t _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_wireVersion(void){
_start:
{
uint32_t v___x_621_; 
v___x_621_ = 1;
return v___x_621_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readHeader(lean_object* v_expected_622_, lean_object* v_a_623_){
_start:
{
lean_object* v___x_624_; 
v___x_624_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readByte(v_a_623_);
if (lean_obj_tag(v___x_624_) == 0)
{
lean_object* v_a_625_; lean_object* v_a_626_; lean_object* v___x_627_; 
v_a_625_ = lean_ctor_get(v___x_624_, 0);
lean_inc(v_a_625_);
v_a_626_ = lean_ctor_get(v___x_624_, 1);
lean_inc(v_a_626_);
lean_dec_ref_known(v___x_624_, 2);
v___x_627_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readByte(v_a_626_);
if (lean_obj_tag(v___x_627_) == 0)
{
lean_object* v_a_628_; lean_object* v_a_629_; lean_object* v___x_630_; 
v_a_628_ = lean_ctor_get(v___x_627_, 0);
lean_inc(v_a_628_);
v_a_629_ = lean_ctor_get(v___x_627_, 1);
lean_inc(v_a_629_);
lean_dec_ref_known(v___x_627_, 2);
v___x_630_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readByte(v_a_629_);
if (lean_obj_tag(v___x_630_) == 0)
{
lean_object* v_a_631_; lean_object* v_a_632_; lean_object* v___x_633_; 
v_a_631_ = lean_ctor_get(v___x_630_, 0);
lean_inc(v_a_631_);
v_a_632_ = lean_ctor_get(v___x_630_, 1);
lean_inc(v_a_632_);
lean_dec_ref_known(v___x_630_, 2);
v___x_633_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readByte(v_a_632_);
if (lean_obj_tag(v___x_633_) == 0)
{
lean_object* v_a_634_; lean_object* v_a_635_; lean_object* v___x_637_; uint8_t v_isShared_638_; uint8_t v_isSharedCheck_679_; 
v_a_634_ = lean_ctor_get(v___x_633_, 0);
v_a_635_ = lean_ctor_get(v___x_633_, 1);
v_isSharedCheck_679_ = !lean_is_exclusive(v___x_633_);
if (v_isSharedCheck_679_ == 0)
{
v___x_637_ = v___x_633_;
v_isShared_638_ = v_isSharedCheck_679_;
goto v_resetjp_636_;
}
else
{
lean_inc(v_a_635_);
lean_inc(v_a_634_);
lean_dec(v___x_633_);
v___x_637_ = lean_box(0);
v_isShared_638_ = v_isSharedCheck_679_;
goto v_resetjp_636_;
}
v_resetjp_636_:
{
lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; uint8_t v___x_646_; 
v___x_639_ = lean_unsigned_to_nat(4u);
v___x_640_ = lean_mk_empty_array_with_capacity(v___x_639_);
v___x_641_ = lean_array_push(v___x_640_, v_a_625_);
v___x_642_ = lean_array_push(v___x_641_, v_a_628_);
v___x_643_ = lean_array_push(v___x_642_, v_a_631_);
v___x_644_ = lean_array_push(v___x_643_, v_a_634_);
v___x_645_ = lean_byte_array_mk(v___x_644_);
v___x_646_ = lean_sarray_dec_eq(v___x_645_, v_expected_622_);
if (v___x_646_ == 0)
{
lean_object* v___x_647_; lean_object* v___x_649_; 
v___x_647_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_647_, 0, v_expected_622_);
lean_ctor_set(v___x_647_, 1, v___x_645_);
if (v_isShared_638_ == 0)
{
lean_ctor_set_tag(v___x_637_, 1);
lean_ctor_set(v___x_637_, 0, v___x_647_);
v___x_649_ = v___x_637_;
goto v_reusejp_648_;
}
else
{
lean_object* v_reuseFailAlloc_650_; 
v_reuseFailAlloc_650_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_650_, 0, v___x_647_);
lean_ctor_set(v_reuseFailAlloc_650_, 1, v_a_635_);
v___x_649_ = v_reuseFailAlloc_650_;
goto v_reusejp_648_;
}
v_reusejp_648_:
{
return v___x_649_;
}
}
else
{
lean_object* v___x_651_; 
lean_dec_ref(v___x_645_);
lean_del_object(v___x_637_);
lean_dec_ref(v_expected_622_);
v___x_651_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_635_);
if (lean_obj_tag(v___x_651_) == 0)
{
lean_object* v_a_652_; lean_object* v_a_653_; lean_object* v___x_655_; uint8_t v_isShared_656_; uint8_t v_isSharedCheck_669_; 
v_a_652_ = lean_ctor_get(v___x_651_, 0);
v_a_653_ = lean_ctor_get(v___x_651_, 1);
v_isSharedCheck_669_ = !lean_is_exclusive(v___x_651_);
if (v_isSharedCheck_669_ == 0)
{
v___x_655_ = v___x_651_;
v_isShared_656_ = v_isSharedCheck_669_;
goto v_resetjp_654_;
}
else
{
lean_inc(v_a_653_);
lean_inc(v_a_652_);
lean_dec(v___x_651_);
v___x_655_ = lean_box(0);
v_isShared_656_ = v_isSharedCheck_669_;
goto v_resetjp_654_;
}
v_resetjp_654_:
{
uint32_t v___x_657_; uint32_t v___x_658_; uint8_t v___x_659_; 
v___x_657_ = 1;
v___x_658_ = lean_unbox_uint32(v_a_652_);
v___x_659_ = lean_uint32_dec_eq(v___x_658_, v___x_657_);
if (v___x_659_ == 0)
{
lean_object* v___x_660_; uint32_t v___x_661_; lean_object* v___x_663_; 
v___x_660_ = lean_alloc_ctor(1, 0, 8);
lean_ctor_set_uint32(v___x_660_, 0, v___x_657_);
v___x_661_ = lean_unbox_uint32(v_a_652_);
lean_dec(v_a_652_);
lean_ctor_set_uint32(v___x_660_, 4, v___x_661_);
if (v_isShared_656_ == 0)
{
lean_ctor_set_tag(v___x_655_, 1);
lean_ctor_set(v___x_655_, 0, v___x_660_);
v___x_663_ = v___x_655_;
goto v_reusejp_662_;
}
else
{
lean_object* v_reuseFailAlloc_664_; 
v_reuseFailAlloc_664_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_664_, 0, v___x_660_);
lean_ctor_set(v_reuseFailAlloc_664_, 1, v_a_653_);
v___x_663_ = v_reuseFailAlloc_664_;
goto v_reusejp_662_;
}
v_reusejp_662_:
{
return v___x_663_;
}
}
else
{
lean_object* v___x_665_; lean_object* v___x_667_; 
lean_dec(v_a_652_);
v___x_665_ = lean_box(0);
if (v_isShared_656_ == 0)
{
lean_ctor_set(v___x_655_, 0, v___x_665_);
v___x_667_ = v___x_655_;
goto v_reusejp_666_;
}
else
{
lean_object* v_reuseFailAlloc_668_; 
v_reuseFailAlloc_668_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_668_, 0, v___x_665_);
lean_ctor_set(v_reuseFailAlloc_668_, 1, v_a_653_);
v___x_667_ = v_reuseFailAlloc_668_;
goto v_reusejp_666_;
}
v_reusejp_666_:
{
return v___x_667_;
}
}
}
}
else
{
lean_object* v_a_670_; lean_object* v_a_671_; lean_object* v___x_673_; uint8_t v_isShared_674_; uint8_t v_isSharedCheck_678_; 
v_a_670_ = lean_ctor_get(v___x_651_, 0);
v_a_671_ = lean_ctor_get(v___x_651_, 1);
v_isSharedCheck_678_ = !lean_is_exclusive(v___x_651_);
if (v_isSharedCheck_678_ == 0)
{
v___x_673_ = v___x_651_;
v_isShared_674_ = v_isSharedCheck_678_;
goto v_resetjp_672_;
}
else
{
lean_inc(v_a_671_);
lean_inc(v_a_670_);
lean_dec(v___x_651_);
v___x_673_ = lean_box(0);
v_isShared_674_ = v_isSharedCheck_678_;
goto v_resetjp_672_;
}
v_resetjp_672_:
{
lean_object* v___x_676_; 
if (v_isShared_674_ == 0)
{
v___x_676_ = v___x_673_;
goto v_reusejp_675_;
}
else
{
lean_object* v_reuseFailAlloc_677_; 
v_reuseFailAlloc_677_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_677_, 0, v_a_670_);
lean_ctor_set(v_reuseFailAlloc_677_, 1, v_a_671_);
v___x_676_ = v_reuseFailAlloc_677_;
goto v_reusejp_675_;
}
v_reusejp_675_:
{
return v___x_676_;
}
}
}
}
}
}
else
{
lean_object* v_a_680_; lean_object* v_a_681_; lean_object* v___x_683_; uint8_t v_isShared_684_; uint8_t v_isSharedCheck_688_; 
lean_dec(v_a_631_);
lean_dec(v_a_628_);
lean_dec(v_a_625_);
lean_dec_ref(v_expected_622_);
v_a_680_ = lean_ctor_get(v___x_633_, 0);
v_a_681_ = lean_ctor_get(v___x_633_, 1);
v_isSharedCheck_688_ = !lean_is_exclusive(v___x_633_);
if (v_isSharedCheck_688_ == 0)
{
v___x_683_ = v___x_633_;
v_isShared_684_ = v_isSharedCheck_688_;
goto v_resetjp_682_;
}
else
{
lean_inc(v_a_681_);
lean_inc(v_a_680_);
lean_dec(v___x_633_);
v___x_683_ = lean_box(0);
v_isShared_684_ = v_isSharedCheck_688_;
goto v_resetjp_682_;
}
v_resetjp_682_:
{
lean_object* v___x_686_; 
if (v_isShared_684_ == 0)
{
v___x_686_ = v___x_683_;
goto v_reusejp_685_;
}
else
{
lean_object* v_reuseFailAlloc_687_; 
v_reuseFailAlloc_687_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_687_, 0, v_a_680_);
lean_ctor_set(v_reuseFailAlloc_687_, 1, v_a_681_);
v___x_686_ = v_reuseFailAlloc_687_;
goto v_reusejp_685_;
}
v_reusejp_685_:
{
return v___x_686_;
}
}
}
}
else
{
lean_object* v_a_689_; lean_object* v_a_690_; lean_object* v___x_692_; uint8_t v_isShared_693_; uint8_t v_isSharedCheck_697_; 
lean_dec(v_a_628_);
lean_dec(v_a_625_);
lean_dec_ref(v_expected_622_);
v_a_689_ = lean_ctor_get(v___x_630_, 0);
v_a_690_ = lean_ctor_get(v___x_630_, 1);
v_isSharedCheck_697_ = !lean_is_exclusive(v___x_630_);
if (v_isSharedCheck_697_ == 0)
{
v___x_692_ = v___x_630_;
v_isShared_693_ = v_isSharedCheck_697_;
goto v_resetjp_691_;
}
else
{
lean_inc(v_a_690_);
lean_inc(v_a_689_);
lean_dec(v___x_630_);
v___x_692_ = lean_box(0);
v_isShared_693_ = v_isSharedCheck_697_;
goto v_resetjp_691_;
}
v_resetjp_691_:
{
lean_object* v___x_695_; 
if (v_isShared_693_ == 0)
{
v___x_695_ = v___x_692_;
goto v_reusejp_694_;
}
else
{
lean_object* v_reuseFailAlloc_696_; 
v_reuseFailAlloc_696_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_696_, 0, v_a_689_);
lean_ctor_set(v_reuseFailAlloc_696_, 1, v_a_690_);
v___x_695_ = v_reuseFailAlloc_696_;
goto v_reusejp_694_;
}
v_reusejp_694_:
{
return v___x_695_;
}
}
}
}
else
{
lean_object* v_a_698_; lean_object* v_a_699_; lean_object* v___x_701_; uint8_t v_isShared_702_; uint8_t v_isSharedCheck_706_; 
lean_dec(v_a_625_);
lean_dec_ref(v_expected_622_);
v_a_698_ = lean_ctor_get(v___x_627_, 0);
v_a_699_ = lean_ctor_get(v___x_627_, 1);
v_isSharedCheck_706_ = !lean_is_exclusive(v___x_627_);
if (v_isSharedCheck_706_ == 0)
{
v___x_701_ = v___x_627_;
v_isShared_702_ = v_isSharedCheck_706_;
goto v_resetjp_700_;
}
else
{
lean_inc(v_a_699_);
lean_inc(v_a_698_);
lean_dec(v___x_627_);
v___x_701_ = lean_box(0);
v_isShared_702_ = v_isSharedCheck_706_;
goto v_resetjp_700_;
}
v_resetjp_700_:
{
lean_object* v___x_704_; 
if (v_isShared_702_ == 0)
{
v___x_704_ = v___x_701_;
goto v_reusejp_703_;
}
else
{
lean_object* v_reuseFailAlloc_705_; 
v_reuseFailAlloc_705_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_705_, 0, v_a_698_);
lean_ctor_set(v_reuseFailAlloc_705_, 1, v_a_699_);
v___x_704_ = v_reuseFailAlloc_705_;
goto v_reusejp_703_;
}
v_reusejp_703_:
{
return v___x_704_;
}
}
}
}
else
{
lean_object* v_a_707_; lean_object* v_a_708_; lean_object* v___x_710_; uint8_t v_isShared_711_; uint8_t v_isSharedCheck_715_; 
lean_dec_ref(v_expected_622_);
v_a_707_ = lean_ctor_get(v___x_624_, 0);
v_a_708_ = lean_ctor_get(v___x_624_, 1);
v_isSharedCheck_715_ = !lean_is_exclusive(v___x_624_);
if (v_isSharedCheck_715_ == 0)
{
v___x_710_ = v___x_624_;
v_isShared_711_ = v_isSharedCheck_715_;
goto v_resetjp_709_;
}
else
{
lean_inc(v_a_708_);
lean_inc(v_a_707_);
lean_dec(v___x_624_);
v___x_710_ = lean_box(0);
v_isShared_711_ = v_isSharedCheck_715_;
goto v_resetjp_709_;
}
v_resetjp_709_:
{
lean_object* v___x_713_; 
if (v_isShared_711_ == 0)
{
v___x_713_ = v___x_710_;
goto v_reusejp_712_;
}
else
{
lean_object* v_reuseFailAlloc_714_; 
v_reuseFailAlloc_714_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_714_, 0, v_a_707_);
lean_ctor_set(v_reuseFailAlloc_714_, 1, v_a_708_);
v___x_713_ = v_reuseFailAlloc_714_;
goto v_reusejp_712_;
}
v_reusejp_712_:
{
return v___x_713_;
}
}
}
}
}
static uint32_t _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirRoundConfig_default(void){
_start:
{
uint32_t v___x_716_; 
v___x_716_ = 0;
return v___x_716_;
}
}
static uint32_t _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirRoundConfig(void){
_start:
{
uint32_t v___x_717_; 
v___x_717_ = 0;
return v___x_717_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_731_; lean_object* v___x_732_; 
v___x_731_ = lean_unsigned_to_nat(14u);
v___x_732_ = lean_nat_to_int(v___x_731_);
return v___x_732_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__9(void){
_start:
{
lean_object* v___x_734_; lean_object* v___x_735_; 
v___x_734_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__0));
v___x_735_ = lean_string_length(v___x_734_);
return v___x_735_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10(void){
_start:
{
lean_object* v___x_736_; lean_object* v___x_737_; 
v___x_736_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__9, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__9_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__9);
v___x_737_ = lean_nat_to_int(v___x_736_);
return v___x_737_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg(uint32_t v_x_742_){
_start:
{
lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; lean_object* v___x_747_; lean_object* v___x_748_; uint8_t v___x_749_; lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; 
v___x_743_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__6));
v___x_744_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__7, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__7_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__7);
v___x_745_ = lean_uint32_to_nat(v_x_742_);
v___x_746_ = l_Nat_reprFast(v___x_745_);
v___x_747_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_747_, 0, v___x_746_);
v___x_748_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_748_, 0, v___x_744_);
lean_ctor_set(v___x_748_, 1, v___x_747_);
v___x_749_ = 0;
v___x_750_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_750_, 0, v___x_748_);
lean_ctor_set_uint8(v___x_750_, sizeof(void*)*1, v___x_749_);
v___x_751_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_751_, 0, v___x_743_);
lean_ctor_set(v___x_751_, 1, v___x_750_);
v___x_752_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10);
v___x_753_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__11));
v___x_754_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_754_, 0, v___x_753_);
lean_ctor_set(v___x_754_, 1, v___x_751_);
v___x_755_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__12));
v___x_756_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_756_, 0, v___x_754_);
lean_ctor_set(v___x_756_, 1, v___x_755_);
v___x_757_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_757_, 0, v___x_752_);
lean_ctor_set(v___x_757_, 1, v___x_756_);
v___x_758_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_758_, 0, v___x_757_);
lean_ctor_set_uint8(v___x_758_, sizeof(void*)*1, v___x_749_);
return v___x_758_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___boxed(lean_object* v_x_759_){
_start:
{
uint32_t v_x_131__boxed_760_; lean_object* v_res_761_; 
v_x_131__boxed_760_ = lean_unbox_uint32(v_x_759_);
lean_dec(v_x_759_);
v_res_761_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg(v_x_131__boxed_760_);
return v_res_761_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr(uint32_t v_x_762_, lean_object* v_prec_763_){
_start:
{
lean_object* v___x_764_; 
v___x_764_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg(v_x_762_);
return v___x_764_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___boxed(lean_object* v_x_765_, lean_object* v_prec_766_){
_start:
{
uint32_t v_x_190__boxed_767_; lean_object* v_res_768_; 
v_x_190__boxed_767_ = lean_unbox_uint32(v_x_765_);
lean_dec(v_x_765_);
v_res_768_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr(v_x_190__boxed_767_, v_prec_766_);
lean_dec(v_prec_766_);
return v_res_768_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_ctorIdx(lean_object* v_x_771_){
_start:
{
switch(lean_obj_tag(v_x_771_))
{
case 0:
{
lean_object* v___x_772_; 
v___x_772_ = lean_unsigned_to_nat(0u);
return v___x_772_;
}
case 1:
{
lean_object* v___x_773_; 
v___x_773_ = lean_unsigned_to_nat(1u);
return v___x_773_;
}
default: 
{
lean_object* v___x_774_; 
v___x_774_ = lean_unsigned_to_nat(2u);
return v___x_774_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_ctorIdx___boxed(lean_object* v_x_775_){
_start:
{
lean_object* v_res_776_; 
v_res_776_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_ctorIdx(v_x_775_);
lean_dec(v_x_775_);
return v_res_776_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_ctorElim___redArg(lean_object* v_t_777_, lean_object* v_k_778_){
_start:
{
switch(lean_obj_tag(v_t_777_))
{
case 0:
{
return v_k_778_;
}
case 1:
{
uint64_t v_m_779_; uint64_t v_listStartRound_780_; lean_object* v___x_781_; lean_object* v___x_782_; lean_object* v___x_783_; 
v_m_779_ = lean_ctor_get_uint64(v_t_777_, 0);
v_listStartRound_780_ = lean_ctor_get_uint64(v_t_777_, 8);
v___x_781_ = lean_box_uint64(v_m_779_);
v___x_782_ = lean_box_uint64(v_listStartRound_780_);
v___x_783_ = lean_apply_2(v_k_778_, v___x_781_, v___x_782_);
return v___x_783_;
}
default: 
{
uint64_t v_m_784_; lean_object* v___x_785_; lean_object* v___x_786_; 
v_m_784_ = lean_ctor_get_uint64(v_t_777_, 0);
v___x_785_ = lean_box_uint64(v_m_784_);
v___x_786_ = lean_apply_1(v_k_778_, v___x_785_);
return v___x_786_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_ctorElim___redArg___boxed(lean_object* v_t_787_, lean_object* v_k_788_){
_start:
{
lean_object* v_res_789_; 
v_res_789_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_ctorElim___redArg(v_t_787_, v_k_788_);
lean_dec(v_t_787_);
return v_res_789_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_ctorElim(lean_object* v_motive_790_, lean_object* v_ctorIdx_791_, lean_object* v_t_792_, lean_object* v_h_793_, lean_object* v_k_794_){
_start:
{
lean_object* v___x_795_; 
v___x_795_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_ctorElim___redArg(v_t_792_, v_k_794_);
return v___x_795_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_ctorElim___boxed(lean_object* v_motive_796_, lean_object* v_ctorIdx_797_, lean_object* v_t_798_, lean_object* v_h_799_, lean_object* v_k_800_){
_start:
{
lean_object* v_res_801_; 
v_res_801_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_ctorElim(v_motive_796_, v_ctorIdx_797_, v_t_798_, v_h_799_, v_k_800_);
lean_dec(v_t_798_);
lean_dec(v_ctorIdx_797_);
return v_res_801_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_uniqueDecoding_elim___redArg(lean_object* v_t_802_, lean_object* v_uniqueDecoding_803_){
_start:
{
lean_object* v___x_804_; 
v___x_804_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_ctorElim___redArg(v_t_802_, v_uniqueDecoding_803_);
return v___x_804_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_uniqueDecoding_elim___redArg___boxed(lean_object* v_t_805_, lean_object* v_uniqueDecoding_806_){
_start:
{
lean_object* v_res_807_; 
v_res_807_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_uniqueDecoding_elim___redArg(v_t_805_, v_uniqueDecoding_806_);
lean_dec(v_t_805_);
return v_res_807_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_uniqueDecoding_elim(lean_object* v_motive_808_, lean_object* v_t_809_, lean_object* v_h_810_, lean_object* v_uniqueDecoding_811_){
_start:
{
lean_object* v___x_812_; 
v___x_812_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_ctorElim___redArg(v_t_809_, v_uniqueDecoding_811_);
return v___x_812_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_uniqueDecoding_elim___boxed(lean_object* v_motive_813_, lean_object* v_t_814_, lean_object* v_h_815_, lean_object* v_uniqueDecoding_816_){
_start:
{
lean_object* v_res_817_; 
v_res_817_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_uniqueDecoding_elim(v_motive_813_, v_t_814_, v_h_815_, v_uniqueDecoding_816_);
lean_dec(v_t_814_);
return v_res_817_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_splitUniqueList_elim___redArg(lean_object* v_t_818_, lean_object* v_splitUniqueList_819_){
_start:
{
lean_object* v___x_820_; 
v___x_820_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_ctorElim___redArg(v_t_818_, v_splitUniqueList_819_);
return v___x_820_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_splitUniqueList_elim___redArg___boxed(lean_object* v_t_821_, lean_object* v_splitUniqueList_822_){
_start:
{
lean_object* v_res_823_; 
v_res_823_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_splitUniqueList_elim___redArg(v_t_821_, v_splitUniqueList_822_);
lean_dec(v_t_821_);
return v_res_823_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_splitUniqueList_elim(lean_object* v_motive_824_, lean_object* v_t_825_, lean_object* v_h_826_, lean_object* v_splitUniqueList_827_){
_start:
{
lean_object* v___x_828_; 
v___x_828_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_ctorElim___redArg(v_t_825_, v_splitUniqueList_827_);
return v___x_828_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_splitUniqueList_elim___boxed(lean_object* v_motive_829_, lean_object* v_t_830_, lean_object* v_h_831_, lean_object* v_splitUniqueList_832_){
_start:
{
lean_object* v_res_833_; 
v_res_833_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_splitUniqueList_elim(v_motive_829_, v_t_830_, v_h_831_, v_splitUniqueList_832_);
lean_dec(v_t_830_);
return v_res_833_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_listDecoding_elim___redArg(lean_object* v_t_834_, lean_object* v_listDecoding_835_){
_start:
{
lean_object* v___x_836_; 
v___x_836_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_ctorElim___redArg(v_t_834_, v_listDecoding_835_);
return v___x_836_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_listDecoding_elim___redArg___boxed(lean_object* v_t_837_, lean_object* v_listDecoding_838_){
_start:
{
lean_object* v_res_839_; 
v_res_839_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_listDecoding_elim___redArg(v_t_837_, v_listDecoding_838_);
lean_dec(v_t_837_);
return v_res_839_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_listDecoding_elim(lean_object* v_motive_840_, lean_object* v_t_841_, lean_object* v_h_842_, lean_object* v_listDecoding_843_){
_start:
{
lean_object* v___x_844_; 
v___x_844_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_ctorElim___redArg(v_t_841_, v_listDecoding_843_);
return v___x_844_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_listDecoding_elim___boxed(lean_object* v_motive_845_, lean_object* v_t_846_, lean_object* v_h_847_, lean_object* v_listDecoding_848_){
_start:
{
lean_object* v_res_849_; 
v_res_849_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawWhirProximityStrategy_listDecoding_elim(v_motive_845_, v_t_846_, v_h_847_, v_listDecoding_848_);
lean_dec(v_t_846_);
return v_res_849_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProximityStrategy_default(void){
_start:
{
lean_object* v___x_850_; 
v___x_850_ = lean_box(0);
return v___x_850_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProximityStrategy(void){
_start:
{
lean_object* v___x_851_; 
v___x_851_ = lean_box(0);
return v___x_851_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2(void){
_start:
{
lean_object* v___x_855_; lean_object* v___x_856_; 
v___x_855_ = lean_unsigned_to_nat(2u);
v___x_856_ = lean_nat_to_int(v___x_855_);
return v___x_856_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3(void){
_start:
{
lean_object* v___x_857_; lean_object* v___x_858_; 
v___x_857_ = lean_unsigned_to_nat(1u);
v___x_858_ = lean_nat_to_int(v___x_857_);
return v___x_858_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr(lean_object* v_x_871_, lean_object* v_prec_872_){
_start:
{
lean_object* v___y_874_; 
switch(lean_obj_tag(v_x_871_))
{
case 0:
{
lean_object* v___x_880_; uint8_t v___x_881_; 
v___x_880_ = lean_unsigned_to_nat(1024u);
v___x_881_ = lean_nat_dec_le(v___x_880_, v_prec_872_);
if (v___x_881_ == 0)
{
lean_object* v___x_882_; 
v___x_882_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2);
v___y_874_ = v___x_882_;
goto v___jp_873_;
}
else
{
lean_object* v___x_883_; 
v___x_883_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3);
v___y_874_ = v___x_883_;
goto v___jp_873_;
}
}
case 1:
{
uint64_t v_m_884_; uint64_t v_listStartRound_885_; lean_object* v___y_887_; lean_object* v___x_903_; uint8_t v___x_904_; 
v_m_884_ = lean_ctor_get_uint64(v_x_871_, 0);
v_listStartRound_885_ = lean_ctor_get_uint64(v_x_871_, 8);
v___x_903_ = lean_unsigned_to_nat(1024u);
v___x_904_ = lean_nat_dec_le(v___x_903_, v_prec_872_);
if (v___x_904_ == 0)
{
lean_object* v___x_905_; 
v___x_905_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2);
v___y_887_ = v___x_905_;
goto v___jp_886_;
}
else
{
lean_object* v___x_906_; 
v___x_906_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3);
v___y_887_ = v___x_906_;
goto v___jp_886_;
}
v___jp_886_:
{
lean_object* v___x_888_; lean_object* v___x_889_; lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; lean_object* v___x_893_; lean_object* v___x_894_; lean_object* v___x_895_; lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; lean_object* v___x_899_; uint8_t v___x_900_; lean_object* v___x_901_; lean_object* v___x_902_; 
v___x_888_ = lean_box(1);
v___x_889_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__6));
v___x_890_ = lean_uint64_to_nat(v_m_884_);
v___x_891_ = l_Nat_reprFast(v___x_890_);
v___x_892_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_892_, 0, v___x_891_);
v___x_893_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_893_, 0, v___x_889_);
lean_ctor_set(v___x_893_, 1, v___x_892_);
v___x_894_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_894_, 0, v___x_893_);
lean_ctor_set(v___x_894_, 1, v___x_888_);
v___x_895_ = lean_uint64_to_nat(v_listStartRound_885_);
v___x_896_ = l_Nat_reprFast(v___x_895_);
v___x_897_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_897_, 0, v___x_896_);
v___x_898_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_898_, 0, v___x_894_);
lean_ctor_set(v___x_898_, 1, v___x_897_);
lean_inc(v___y_887_);
v___x_899_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_899_, 0, v___y_887_);
lean_ctor_set(v___x_899_, 1, v___x_898_);
v___x_900_ = 0;
v___x_901_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_901_, 0, v___x_899_);
lean_ctor_set_uint8(v___x_901_, sizeof(void*)*1, v___x_900_);
v___x_902_ = l_Repr_addAppParen(v___x_901_, v_prec_872_);
return v___x_902_;
}
}
default: 
{
uint64_t v_m_907_; lean_object* v___y_909_; lean_object* v___x_919_; uint8_t v___x_920_; 
v_m_907_ = lean_ctor_get_uint64(v_x_871_, 0);
v___x_919_ = lean_unsigned_to_nat(1024u);
v___x_920_ = lean_nat_dec_le(v___x_919_, v_prec_872_);
if (v___x_920_ == 0)
{
lean_object* v___x_921_; 
v___x_921_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2);
v___y_909_ = v___x_921_;
goto v___jp_908_;
}
else
{
lean_object* v___x_922_; 
v___x_922_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3);
v___y_909_ = v___x_922_;
goto v___jp_908_;
}
v___jp_908_:
{
lean_object* v___x_910_; lean_object* v___x_911_; lean_object* v___x_912_; lean_object* v___x_913_; lean_object* v___x_914_; lean_object* v___x_915_; uint8_t v___x_916_; lean_object* v___x_917_; lean_object* v___x_918_; 
v___x_910_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__9));
v___x_911_ = lean_uint64_to_nat(v_m_907_);
v___x_912_ = l_Nat_reprFast(v___x_911_);
v___x_913_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_913_, 0, v___x_912_);
v___x_914_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_914_, 0, v___x_910_);
lean_ctor_set(v___x_914_, 1, v___x_913_);
lean_inc(v___y_909_);
v___x_915_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_915_, 0, v___y_909_);
lean_ctor_set(v___x_915_, 1, v___x_914_);
v___x_916_ = 0;
v___x_917_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_917_, 0, v___x_915_);
lean_ctor_set_uint8(v___x_917_, sizeof(void*)*1, v___x_916_);
v___x_918_ = l_Repr_addAppParen(v___x_917_, v_prec_872_);
return v___x_918_;
}
}
}
v___jp_873_:
{
lean_object* v___x_875_; lean_object* v___x_876_; uint8_t v___x_877_; lean_object* v___x_878_; lean_object* v___x_879_; 
v___x_875_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__1));
lean_inc(v___y_874_);
v___x_876_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_876_, 0, v___y_874_);
lean_ctor_set(v___x_876_, 1, v___x_875_);
v___x_877_ = 0;
v___x_878_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_878_, 0, v___x_876_);
lean_ctor_set_uint8(v___x_878_, sizeof(void*)*1, v___x_877_);
v___x_879_ = l_Repr_addAppParen(v___x_878_, v_prec_872_);
return v___x_879_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___boxed(lean_object* v_x_923_, lean_object* v_prec_924_){
_start:
{
lean_object* v_res_925_; 
v_res_925_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr(v_x_923_, v_prec_924_);
lean_dec(v_prec_924_);
lean_dec(v_x_923_);
return v_res_925_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0_spec__0_spec__1_spec__2(lean_object* v_x_936_, lean_object* v_x_937_, lean_object* v_x_938_){
_start:
{
if (lean_obj_tag(v_x_938_) == 0)
{
lean_dec(v_x_936_);
return v_x_937_;
}
else
{
lean_object* v_head_939_; lean_object* v_tail_940_; lean_object* v___x_942_; uint8_t v_isShared_943_; uint8_t v_isSharedCheck_951_; 
v_head_939_ = lean_ctor_get(v_x_938_, 0);
v_tail_940_ = lean_ctor_get(v_x_938_, 1);
v_isSharedCheck_951_ = !lean_is_exclusive(v_x_938_);
if (v_isSharedCheck_951_ == 0)
{
v___x_942_ = v_x_938_;
v_isShared_943_ = v_isSharedCheck_951_;
goto v_resetjp_941_;
}
else
{
lean_inc(v_tail_940_);
lean_inc(v_head_939_);
lean_dec(v_x_938_);
v___x_942_ = lean_box(0);
v_isShared_943_ = v_isSharedCheck_951_;
goto v_resetjp_941_;
}
v_resetjp_941_:
{
lean_object* v___x_945_; 
lean_inc(v_x_936_);
if (v_isShared_943_ == 0)
{
lean_ctor_set_tag(v___x_942_, 5);
lean_ctor_set(v___x_942_, 1, v_x_936_);
lean_ctor_set(v___x_942_, 0, v_x_937_);
v___x_945_ = v___x_942_;
goto v_reusejp_944_;
}
else
{
lean_object* v_reuseFailAlloc_950_; 
v_reuseFailAlloc_950_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_950_, 0, v_x_937_);
lean_ctor_set(v_reuseFailAlloc_950_, 1, v_x_936_);
v___x_945_ = v_reuseFailAlloc_950_;
goto v_reusejp_944_;
}
v_reusejp_944_:
{
uint32_t v___x_946_; lean_object* v___x_947_; lean_object* v___x_948_; 
v___x_946_ = lean_unbox_uint32(v_head_939_);
lean_dec(v_head_939_);
v___x_947_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg(v___x_946_);
v___x_948_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_948_, 0, v___x_945_);
lean_ctor_set(v___x_948_, 1, v___x_947_);
v_x_937_ = v___x_948_;
v_x_938_ = v_tail_940_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0_spec__0_spec__1(lean_object* v_x_952_, lean_object* v_x_953_, lean_object* v_x_954_){
_start:
{
if (lean_obj_tag(v_x_954_) == 0)
{
lean_dec(v_x_952_);
return v_x_953_;
}
else
{
lean_object* v_head_955_; lean_object* v_tail_956_; lean_object* v___x_958_; uint8_t v_isShared_959_; uint8_t v_isSharedCheck_967_; 
v_head_955_ = lean_ctor_get(v_x_954_, 0);
v_tail_956_ = lean_ctor_get(v_x_954_, 1);
v_isSharedCheck_967_ = !lean_is_exclusive(v_x_954_);
if (v_isSharedCheck_967_ == 0)
{
v___x_958_ = v_x_954_;
v_isShared_959_ = v_isSharedCheck_967_;
goto v_resetjp_957_;
}
else
{
lean_inc(v_tail_956_);
lean_inc(v_head_955_);
lean_dec(v_x_954_);
v___x_958_ = lean_box(0);
v_isShared_959_ = v_isSharedCheck_967_;
goto v_resetjp_957_;
}
v_resetjp_957_:
{
lean_object* v___x_961_; 
lean_inc(v_x_952_);
if (v_isShared_959_ == 0)
{
lean_ctor_set_tag(v___x_958_, 5);
lean_ctor_set(v___x_958_, 1, v_x_952_);
lean_ctor_set(v___x_958_, 0, v_x_953_);
v___x_961_ = v___x_958_;
goto v_reusejp_960_;
}
else
{
lean_object* v_reuseFailAlloc_966_; 
v_reuseFailAlloc_966_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_966_, 0, v_x_953_);
lean_ctor_set(v_reuseFailAlloc_966_, 1, v_x_952_);
v___x_961_ = v_reuseFailAlloc_966_;
goto v_reusejp_960_;
}
v_reusejp_960_:
{
uint32_t v___x_962_; lean_object* v___x_963_; lean_object* v___x_964_; lean_object* v___x_965_; 
v___x_962_ = lean_unbox_uint32(v_head_955_);
lean_dec(v_head_955_);
v___x_963_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg(v___x_962_);
v___x_964_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_964_, 0, v___x_961_);
lean_ctor_set(v___x_964_, 1, v___x_963_);
v___x_965_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0_spec__0_spec__1_spec__2(v_x_952_, v___x_964_, v_tail_956_);
return v___x_965_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0_spec__0(lean_object* v_x_968_, lean_object* v_x_969_){
_start:
{
if (lean_obj_tag(v_x_968_) == 0)
{
lean_object* v___x_970_; 
lean_dec(v_x_969_);
v___x_970_ = lean_box(0);
return v___x_970_;
}
else
{
lean_object* v_tail_971_; 
v_tail_971_ = lean_ctor_get(v_x_968_, 1);
if (lean_obj_tag(v_tail_971_) == 0)
{
lean_object* v_head_972_; uint32_t v___x_973_; lean_object* v___x_974_; 
lean_dec(v_x_969_);
v_head_972_ = lean_ctor_get(v_x_968_, 0);
lean_inc(v_head_972_);
lean_dec_ref_known(v_x_968_, 2);
v___x_973_ = lean_unbox_uint32(v_head_972_);
lean_dec(v_head_972_);
v___x_974_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg(v___x_973_);
return v___x_974_;
}
else
{
lean_object* v_head_975_; uint32_t v___x_976_; lean_object* v___x_977_; lean_object* v___x_978_; 
lean_inc(v_tail_971_);
v_head_975_ = lean_ctor_get(v_x_968_, 0);
lean_inc(v_head_975_);
lean_dec_ref_known(v_x_968_, 2);
v___x_976_ = lean_unbox_uint32(v_head_975_);
lean_dec(v_head_975_);
v___x_977_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg(v___x_976_);
v___x_978_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0_spec__0_spec__1(v_x_969_, v___x_977_, v_tail_971_);
return v___x_978_;
}
}
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__5(void){
_start:
{
lean_object* v___x_987_; lean_object* v___x_988_; 
v___x_987_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__0));
v___x_988_ = lean_string_length(v___x_987_);
return v___x_988_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__6(void){
_start:
{
lean_object* v___x_989_; lean_object* v___x_990_; 
v___x_989_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__5, &lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__5_once, _init_lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__5);
v___x_990_ = lean_nat_to_int(v___x_989_);
return v___x_990_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0(lean_object* v_xs_998_){
_start:
{
lean_object* v___x_999_; lean_object* v___x_1000_; uint8_t v___x_1001_; 
v___x_999_ = lean_array_get_size(v_xs_998_);
v___x_1000_ = lean_unsigned_to_nat(0u);
v___x_1001_ = lean_nat_dec_eq(v___x_999_, v___x_1000_);
if (v___x_1001_ == 0)
{
lean_object* v___x_1002_; lean_object* v___x_1003_; lean_object* v___x_1004_; lean_object* v___x_1005_; lean_object* v___x_1006_; lean_object* v___x_1007_; lean_object* v___x_1008_; lean_object* v___x_1009_; lean_object* v___x_1010_; lean_object* v___x_1011_; 
v___x_1002_ = lean_array_to_list(v_xs_998_);
v___x_1003_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__3));
v___x_1004_ = lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0_spec__0(v___x_1002_, v___x_1003_);
v___x_1005_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__6, &lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__6_once, _init_lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__6);
v___x_1006_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__7));
v___x_1007_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1007_, 0, v___x_1006_);
lean_ctor_set(v___x_1007_, 1, v___x_1004_);
v___x_1008_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__8));
v___x_1009_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1009_, 0, v___x_1007_);
lean_ctor_set(v___x_1009_, 1, v___x_1008_);
v___x_1010_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1010_, 0, v___x_1005_);
lean_ctor_set(v___x_1010_, 1, v___x_1009_);
v___x_1011_ = l_Std_Format_fill(v___x_1010_);
return v___x_1011_;
}
else
{
lean_object* v___x_1012_; 
lean_dec_ref(v_xs_998_);
v___x_1012_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__10));
return v___x_1012_;
}
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__4(void){
_start:
{
lean_object* v___x_1022_; lean_object* v___x_1023_; 
v___x_1022_ = lean_unsigned_to_nat(5u);
v___x_1023_ = lean_nat_to_int(v___x_1022_);
return v___x_1023_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_1027_; lean_object* v___x_1028_; 
v___x_1027_ = lean_unsigned_to_nat(10u);
v___x_1028_ = lean_nat_to_int(v___x_1027_);
return v___x_1028_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__10(void){
_start:
{
lean_object* v___x_1032_; lean_object* v___x_1033_; 
v___x_1032_ = lean_unsigned_to_nat(13u);
v___x_1033_ = lean_nat_to_int(v___x_1032_);
return v___x_1033_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__13(void){
_start:
{
lean_object* v___x_1037_; lean_object* v___x_1038_; 
v___x_1037_ = lean_unsigned_to_nat(21u);
v___x_1038_ = lean_nat_to_int(v___x_1037_);
return v___x_1038_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__16(void){
_start:
{
lean_object* v___x_1042_; lean_object* v___x_1043_; 
v___x_1042_ = lean_unsigned_to_nat(18u);
v___x_1043_ = lean_nat_to_int(v___x_1042_);
return v___x_1043_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg(lean_object* v_x_1047_){
_start:
{
uint32_t v_k_1048_; lean_object* v_rounds_1049_; uint32_t v_muPowBits_1050_; uint32_t v_queryPhasePowBits_1051_; uint32_t v_foldingPowBits_1052_; lean_object* v_proximity_1053_; lean_object* v___x_1054_; lean_object* v___x_1055_; lean_object* v___x_1056_; lean_object* v___x_1057_; lean_object* v___x_1058_; lean_object* v___x_1059_; lean_object* v___x_1060_; uint8_t v___x_1061_; lean_object* v___x_1062_; lean_object* v___x_1063_; lean_object* v___x_1064_; lean_object* v___x_1065_; lean_object* v___x_1066_; lean_object* v___x_1067_; lean_object* v___x_1068_; lean_object* v___x_1069_; lean_object* v___x_1070_; lean_object* v___x_1071_; lean_object* v___x_1072_; lean_object* v___x_1073_; lean_object* v___x_1074_; lean_object* v___x_1075_; lean_object* v___x_1076_; lean_object* v___x_1077_; lean_object* v___x_1078_; lean_object* v___x_1079_; lean_object* v___x_1080_; lean_object* v___x_1081_; lean_object* v___x_1082_; lean_object* v___x_1083_; lean_object* v___x_1084_; lean_object* v___x_1085_; lean_object* v___x_1086_; lean_object* v___x_1087_; lean_object* v___x_1088_; lean_object* v___x_1089_; lean_object* v___x_1090_; lean_object* v___x_1091_; lean_object* v___x_1092_; lean_object* v___x_1093_; lean_object* v___x_1094_; lean_object* v___x_1095_; lean_object* v___x_1096_; lean_object* v___x_1097_; lean_object* v___x_1098_; lean_object* v___x_1099_; lean_object* v___x_1100_; lean_object* v___x_1101_; lean_object* v___x_1102_; lean_object* v___x_1103_; lean_object* v___x_1104_; lean_object* v___x_1105_; lean_object* v___x_1106_; lean_object* v___x_1107_; lean_object* v___x_1108_; lean_object* v___x_1109_; lean_object* v___x_1110_; lean_object* v___x_1111_; lean_object* v___x_1112_; lean_object* v___x_1113_; lean_object* v___x_1114_; lean_object* v___x_1115_; lean_object* v___x_1116_; lean_object* v___x_1117_; lean_object* v___x_1118_; lean_object* v___x_1119_; lean_object* v___x_1120_; lean_object* v___x_1121_; lean_object* v___x_1122_; lean_object* v___x_1123_; lean_object* v___x_1124_; lean_object* v___x_1125_; lean_object* v___x_1126_; lean_object* v___x_1127_; lean_object* v___x_1128_; 
v_k_1048_ = lean_ctor_get_uint32(v_x_1047_, sizeof(void*)*2);
v_rounds_1049_ = lean_ctor_get(v_x_1047_, 0);
lean_inc_ref(v_rounds_1049_);
v_muPowBits_1050_ = lean_ctor_get_uint32(v_x_1047_, sizeof(void*)*2 + 4);
v_queryPhasePowBits_1051_ = lean_ctor_get_uint32(v_x_1047_, sizeof(void*)*2 + 8);
v_foldingPowBits_1052_ = lean_ctor_get_uint32(v_x_1047_, sizeof(void*)*2 + 12);
v_proximity_1053_ = lean_ctor_get(v_x_1047_, 1);
lean_inc(v_proximity_1053_);
lean_dec_ref(v_x_1047_);
v___x_1054_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__5));
v___x_1055_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__3));
v___x_1056_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__4);
v___x_1057_ = lean_uint32_to_nat(v_k_1048_);
v___x_1058_ = l_Nat_reprFast(v___x_1057_);
v___x_1059_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1059_, 0, v___x_1058_);
v___x_1060_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1060_, 0, v___x_1056_);
lean_ctor_set(v___x_1060_, 1, v___x_1059_);
v___x_1061_ = 0;
v___x_1062_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1062_, 0, v___x_1060_);
lean_ctor_set_uint8(v___x_1062_, sizeof(void*)*1, v___x_1061_);
v___x_1063_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1063_, 0, v___x_1055_);
lean_ctor_set(v___x_1063_, 1, v___x_1062_);
v___x_1064_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__2));
v___x_1065_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1065_, 0, v___x_1063_);
lean_ctor_set(v___x_1065_, 1, v___x_1064_);
v___x_1066_ = lean_box(1);
v___x_1067_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1067_, 0, v___x_1065_);
lean_ctor_set(v___x_1067_, 1, v___x_1066_);
v___x_1068_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__6));
v___x_1069_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1069_, 0, v___x_1067_);
lean_ctor_set(v___x_1069_, 1, v___x_1068_);
v___x_1070_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1070_, 0, v___x_1069_);
lean_ctor_set(v___x_1070_, 1, v___x_1054_);
v___x_1071_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__7, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__7_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__7);
v___x_1072_ = lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0(v_rounds_1049_);
v___x_1073_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1073_, 0, v___x_1071_);
lean_ctor_set(v___x_1073_, 1, v___x_1072_);
v___x_1074_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1074_, 0, v___x_1073_);
lean_ctor_set_uint8(v___x_1074_, sizeof(void*)*1, v___x_1061_);
v___x_1075_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1075_, 0, v___x_1070_);
lean_ctor_set(v___x_1075_, 1, v___x_1074_);
v___x_1076_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1076_, 0, v___x_1075_);
lean_ctor_set(v___x_1076_, 1, v___x_1064_);
v___x_1077_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1077_, 0, v___x_1076_);
lean_ctor_set(v___x_1077_, 1, v___x_1066_);
v___x_1078_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__9));
v___x_1079_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1079_, 0, v___x_1077_);
lean_ctor_set(v___x_1079_, 1, v___x_1078_);
v___x_1080_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1080_, 0, v___x_1079_);
lean_ctor_set(v___x_1080_, 1, v___x_1054_);
v___x_1081_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__10, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__10_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__10);
v___x_1082_ = lean_uint32_to_nat(v_muPowBits_1050_);
v___x_1083_ = l_Nat_reprFast(v___x_1082_);
v___x_1084_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1084_, 0, v___x_1083_);
v___x_1085_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1085_, 0, v___x_1081_);
lean_ctor_set(v___x_1085_, 1, v___x_1084_);
v___x_1086_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1086_, 0, v___x_1085_);
lean_ctor_set_uint8(v___x_1086_, sizeof(void*)*1, v___x_1061_);
v___x_1087_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1087_, 0, v___x_1080_);
lean_ctor_set(v___x_1087_, 1, v___x_1086_);
v___x_1088_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1088_, 0, v___x_1087_);
lean_ctor_set(v___x_1088_, 1, v___x_1064_);
v___x_1089_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1089_, 0, v___x_1088_);
lean_ctor_set(v___x_1089_, 1, v___x_1066_);
v___x_1090_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__12));
v___x_1091_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1091_, 0, v___x_1089_);
lean_ctor_set(v___x_1091_, 1, v___x_1090_);
v___x_1092_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1092_, 0, v___x_1091_);
lean_ctor_set(v___x_1092_, 1, v___x_1054_);
v___x_1093_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__13, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__13_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__13);
v___x_1094_ = lean_uint32_to_nat(v_queryPhasePowBits_1051_);
v___x_1095_ = l_Nat_reprFast(v___x_1094_);
v___x_1096_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1096_, 0, v___x_1095_);
v___x_1097_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1097_, 0, v___x_1093_);
lean_ctor_set(v___x_1097_, 1, v___x_1096_);
v___x_1098_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1098_, 0, v___x_1097_);
lean_ctor_set_uint8(v___x_1098_, sizeof(void*)*1, v___x_1061_);
v___x_1099_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1099_, 0, v___x_1092_);
lean_ctor_set(v___x_1099_, 1, v___x_1098_);
v___x_1100_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1100_, 0, v___x_1099_);
lean_ctor_set(v___x_1100_, 1, v___x_1064_);
v___x_1101_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1101_, 0, v___x_1100_);
lean_ctor_set(v___x_1101_, 1, v___x_1066_);
v___x_1102_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__15));
v___x_1103_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1103_, 0, v___x_1101_);
lean_ctor_set(v___x_1103_, 1, v___x_1102_);
v___x_1104_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1104_, 0, v___x_1103_);
lean_ctor_set(v___x_1104_, 1, v___x_1054_);
v___x_1105_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__16, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__16_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__16);
v___x_1106_ = lean_uint32_to_nat(v_foldingPowBits_1052_);
v___x_1107_ = l_Nat_reprFast(v___x_1106_);
v___x_1108_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1108_, 0, v___x_1107_);
v___x_1109_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1109_, 0, v___x_1105_);
lean_ctor_set(v___x_1109_, 1, v___x_1108_);
v___x_1110_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1110_, 0, v___x_1109_);
lean_ctor_set_uint8(v___x_1110_, sizeof(void*)*1, v___x_1061_);
v___x_1111_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1111_, 0, v___x_1104_);
lean_ctor_set(v___x_1111_, 1, v___x_1110_);
v___x_1112_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1112_, 0, v___x_1111_);
lean_ctor_set(v___x_1112_, 1, v___x_1064_);
v___x_1113_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1113_, 0, v___x_1112_);
lean_ctor_set(v___x_1113_, 1, v___x_1066_);
v___x_1114_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__18));
v___x_1115_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1115_, 0, v___x_1113_);
lean_ctor_set(v___x_1115_, 1, v___x_1114_);
v___x_1116_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1116_, 0, v___x_1115_);
lean_ctor_set(v___x_1116_, 1, v___x_1054_);
v___x_1117_ = lean_unsigned_to_nat(0u);
v___x_1118_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr(v_proximity_1053_, v___x_1117_);
lean_dec(v_proximity_1053_);
v___x_1119_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1119_, 0, v___x_1081_);
lean_ctor_set(v___x_1119_, 1, v___x_1118_);
v___x_1120_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1120_, 0, v___x_1119_);
lean_ctor_set_uint8(v___x_1120_, sizeof(void*)*1, v___x_1061_);
v___x_1121_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1121_, 0, v___x_1116_);
lean_ctor_set(v___x_1121_, 1, v___x_1120_);
v___x_1122_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10);
v___x_1123_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__11));
v___x_1124_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1124_, 0, v___x_1123_);
lean_ctor_set(v___x_1124_, 1, v___x_1121_);
v___x_1125_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__12));
v___x_1126_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1126_, 0, v___x_1124_);
lean_ctor_set(v___x_1126_, 1, v___x_1125_);
v___x_1127_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1127_, 0, v___x_1122_);
lean_ctor_set(v___x_1127_, 1, v___x_1126_);
v___x_1128_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1128_, 0, v___x_1127_);
lean_ctor_set_uint8(v___x_1128_, sizeof(void*)*1, v___x_1061_);
return v___x_1128_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr(lean_object* v_x_1129_, lean_object* v_prec_1130_){
_start:
{
lean_object* v___x_1131_; 
v___x_1131_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg(v_x_1129_);
return v___x_1131_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___boxed(lean_object* v_x_1132_, lean_object* v_prec_1133_){
_start:
{
lean_object* v_res_1134_; 
v_res_1134_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr(v_x_1132_, v_prec_1133_);
lean_dec(v_prec_1133_);
return v_res_1134_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__4(void){
_start:
{
lean_object* v___x_1150_; lean_object* v___x_1151_; 
v___x_1150_ = lean_unsigned_to_nat(23u);
v___x_1151_ = lean_nat_to_int(v___x_1150_);
return v___x_1151_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__9(void){
_start:
{
lean_object* v___x_1158_; lean_object* v___x_1159_; 
v___x_1158_ = lean_unsigned_to_nat(11u);
v___x_1159_ = lean_nat_to_int(v___x_1158_);
return v___x_1159_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg(lean_object* v_x_1160_){
_start:
{
uint32_t v_maxInteractionCount_1161_; uint32_t v_logMaxMessageLength_1162_; uint32_t v_powBits_1163_; lean_object* v___x_1164_; lean_object* v___x_1165_; lean_object* v___x_1166_; lean_object* v___x_1167_; lean_object* v___x_1168_; lean_object* v___x_1169_; lean_object* v___x_1170_; uint8_t v___x_1171_; lean_object* v___x_1172_; lean_object* v___x_1173_; lean_object* v___x_1174_; lean_object* v___x_1175_; lean_object* v___x_1176_; lean_object* v___x_1177_; lean_object* v___x_1178_; lean_object* v___x_1179_; lean_object* v___x_1180_; lean_object* v___x_1181_; lean_object* v___x_1182_; lean_object* v___x_1183_; lean_object* v___x_1184_; lean_object* v___x_1185_; lean_object* v___x_1186_; lean_object* v___x_1187_; lean_object* v___x_1188_; lean_object* v___x_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; lean_object* v___x_1192_; lean_object* v___x_1193_; lean_object* v___x_1194_; lean_object* v___x_1195_; lean_object* v___x_1196_; lean_object* v___x_1197_; lean_object* v___x_1198_; lean_object* v___x_1199_; lean_object* v___x_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; lean_object* v___x_1203_; lean_object* v___x_1204_; lean_object* v___x_1205_; 
v_maxInteractionCount_1161_ = lean_ctor_get_uint32(v_x_1160_, 0);
v_logMaxMessageLength_1162_ = lean_ctor_get_uint32(v_x_1160_, 4);
v_powBits_1163_ = lean_ctor_get_uint32(v_x_1160_, 8);
v___x_1164_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__5));
v___x_1165_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__3));
v___x_1166_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__4);
v___x_1167_ = lean_uint32_to_nat(v_maxInteractionCount_1161_);
v___x_1168_ = l_Nat_reprFast(v___x_1167_);
v___x_1169_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1169_, 0, v___x_1168_);
v___x_1170_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1170_, 0, v___x_1166_);
lean_ctor_set(v___x_1170_, 1, v___x_1169_);
v___x_1171_ = 0;
v___x_1172_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1172_, 0, v___x_1170_);
lean_ctor_set_uint8(v___x_1172_, sizeof(void*)*1, v___x_1171_);
v___x_1173_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1173_, 0, v___x_1165_);
lean_ctor_set(v___x_1173_, 1, v___x_1172_);
v___x_1174_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__2));
v___x_1175_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1175_, 0, v___x_1173_);
lean_ctor_set(v___x_1175_, 1, v___x_1174_);
v___x_1176_ = lean_box(1);
v___x_1177_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1177_, 0, v___x_1175_);
lean_ctor_set(v___x_1177_, 1, v___x_1176_);
v___x_1178_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__6));
v___x_1179_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1179_, 0, v___x_1177_);
lean_ctor_set(v___x_1179_, 1, v___x_1178_);
v___x_1180_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1180_, 0, v___x_1179_);
lean_ctor_set(v___x_1180_, 1, v___x_1164_);
v___x_1181_ = lean_uint32_to_nat(v_logMaxMessageLength_1162_);
v___x_1182_ = l_Nat_reprFast(v___x_1181_);
v___x_1183_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1183_, 0, v___x_1182_);
v___x_1184_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1184_, 0, v___x_1166_);
lean_ctor_set(v___x_1184_, 1, v___x_1183_);
v___x_1185_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1185_, 0, v___x_1184_);
lean_ctor_set_uint8(v___x_1185_, sizeof(void*)*1, v___x_1171_);
v___x_1186_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1186_, 0, v___x_1180_);
lean_ctor_set(v___x_1186_, 1, v___x_1185_);
v___x_1187_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1187_, 0, v___x_1186_);
lean_ctor_set(v___x_1187_, 1, v___x_1174_);
v___x_1188_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1188_, 0, v___x_1187_);
lean_ctor_set(v___x_1188_, 1, v___x_1176_);
v___x_1189_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__8));
v___x_1190_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1190_, 0, v___x_1188_);
lean_ctor_set(v___x_1190_, 1, v___x_1189_);
v___x_1191_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1191_, 0, v___x_1190_);
lean_ctor_set(v___x_1191_, 1, v___x_1164_);
v___x_1192_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__9, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__9_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__9);
v___x_1193_ = lean_uint32_to_nat(v_powBits_1163_);
v___x_1194_ = l_Nat_reprFast(v___x_1193_);
v___x_1195_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1195_, 0, v___x_1194_);
v___x_1196_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1196_, 0, v___x_1192_);
lean_ctor_set(v___x_1196_, 1, v___x_1195_);
v___x_1197_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1197_, 0, v___x_1196_);
lean_ctor_set_uint8(v___x_1197_, sizeof(void*)*1, v___x_1171_);
v___x_1198_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1198_, 0, v___x_1191_);
lean_ctor_set(v___x_1198_, 1, v___x_1197_);
v___x_1199_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10);
v___x_1200_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__11));
v___x_1201_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1201_, 0, v___x_1200_);
lean_ctor_set(v___x_1201_, 1, v___x_1198_);
v___x_1202_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__12));
v___x_1203_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1203_, 0, v___x_1201_);
lean_ctor_set(v___x_1203_, 1, v___x_1202_);
v___x_1204_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1204_, 0, v___x_1199_);
lean_ctor_set(v___x_1204_, 1, v___x_1203_);
v___x_1205_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1205_, 0, v___x_1204_);
lean_ctor_set_uint8(v___x_1205_, sizeof(void*)*1, v___x_1171_);
return v___x_1205_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___boxed(lean_object* v_x_1206_){
_start:
{
lean_object* v_res_1207_; 
v_res_1207_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg(v_x_1206_);
lean_dec_ref(v_x_1206_);
return v_res_1207_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr(lean_object* v_x_1208_, lean_object* v_prec_1209_){
_start:
{
lean_object* v___x_1210_; 
v___x_1210_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg(v_x_1208_);
return v___x_1210_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___boxed(lean_object* v_x_1211_, lean_object* v_prec_1212_){
_start:
{
lean_object* v_res_1213_; 
v_res_1213_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr(v_x_1211_, v_prec_1212_);
lean_dec(v_prec_1212_);
lean_dec_ref(v_x_1211_);
return v_res_1213_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__4(void){
_start:
{
lean_object* v___x_1231_; lean_object* v___x_1232_; 
v___x_1231_ = lean_unsigned_to_nat(9u);
v___x_1232_ = lean_nat_to_int(v___x_1231_);
return v___x_1232_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__13(void){
_start:
{
lean_object* v___x_1245_; lean_object* v___x_1246_; 
v___x_1245_ = lean_unsigned_to_nat(8u);
v___x_1246_ = lean_nat_to_int(v___x_1245_);
return v___x_1246_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg(lean_object* v_x_1253_){
_start:
{
uint32_t v_lSkip_1254_; uint32_t v_nStack_1255_; uint32_t v_wStack_1256_; uint32_t v_logBlowup_1257_; lean_object* v_whir_1258_; lean_object* v_logup_1259_; uint32_t v_maxConstraintDegree_1260_; lean_object* v___x_1261_; lean_object* v___x_1262_; lean_object* v___x_1263_; lean_object* v___x_1264_; lean_object* v___x_1265_; lean_object* v___x_1266_; lean_object* v___x_1267_; uint8_t v___x_1268_; lean_object* v___x_1269_; lean_object* v___x_1270_; lean_object* v___x_1271_; lean_object* v___x_1272_; lean_object* v___x_1273_; lean_object* v___x_1274_; lean_object* v___x_1275_; lean_object* v___x_1276_; lean_object* v___x_1277_; lean_object* v___x_1278_; lean_object* v___x_1279_; lean_object* v___x_1280_; lean_object* v___x_1281_; lean_object* v___x_1282_; lean_object* v___x_1283_; lean_object* v___x_1284_; lean_object* v___x_1285_; lean_object* v___x_1286_; lean_object* v___x_1287_; lean_object* v___x_1288_; lean_object* v___x_1289_; lean_object* v___x_1290_; lean_object* v___x_1291_; lean_object* v___x_1292_; lean_object* v___x_1293_; lean_object* v___x_1294_; lean_object* v___x_1295_; lean_object* v___x_1296_; lean_object* v___x_1297_; lean_object* v___x_1298_; lean_object* v___x_1299_; lean_object* v___x_1300_; lean_object* v___x_1301_; lean_object* v___x_1302_; lean_object* v___x_1303_; lean_object* v___x_1304_; lean_object* v___x_1305_; lean_object* v___x_1306_; lean_object* v___x_1307_; lean_object* v___x_1308_; lean_object* v___x_1309_; lean_object* v___x_1310_; lean_object* v___x_1311_; lean_object* v___x_1312_; lean_object* v___x_1313_; lean_object* v___x_1314_; lean_object* v___x_1315_; lean_object* v___x_1316_; lean_object* v___x_1317_; lean_object* v___x_1318_; lean_object* v___x_1319_; lean_object* v___x_1320_; lean_object* v___x_1321_; lean_object* v___x_1322_; lean_object* v___x_1323_; lean_object* v___x_1324_; lean_object* v___x_1325_; lean_object* v___x_1326_; lean_object* v___x_1327_; lean_object* v___x_1328_; lean_object* v___x_1329_; lean_object* v___x_1330_; lean_object* v___x_1331_; lean_object* v___x_1332_; lean_object* v___x_1333_; lean_object* v___x_1334_; lean_object* v___x_1335_; lean_object* v___x_1336_; lean_object* v___x_1337_; lean_object* v___x_1338_; lean_object* v___x_1339_; lean_object* v___x_1340_; lean_object* v___x_1341_; lean_object* v___x_1342_; lean_object* v___x_1343_; lean_object* v___x_1344_; lean_object* v___x_1345_; 
v_lSkip_1254_ = lean_ctor_get_uint32(v_x_1253_, sizeof(void*)*2);
v_nStack_1255_ = lean_ctor_get_uint32(v_x_1253_, sizeof(void*)*2 + 4);
v_wStack_1256_ = lean_ctor_get_uint32(v_x_1253_, sizeof(void*)*2 + 8);
v_logBlowup_1257_ = lean_ctor_get_uint32(v_x_1253_, sizeof(void*)*2 + 12);
v_whir_1258_ = lean_ctor_get(v_x_1253_, 0);
lean_inc_ref(v_whir_1258_);
v_logup_1259_ = lean_ctor_get(v_x_1253_, 1);
lean_inc_ref(v_logup_1259_);
v_maxConstraintDegree_1260_ = lean_ctor_get_uint32(v_x_1253_, sizeof(void*)*2 + 16);
lean_dec_ref(v_x_1253_);
v___x_1261_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__5));
v___x_1262_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__3));
v___x_1263_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__4);
v___x_1264_ = lean_uint32_to_nat(v_lSkip_1254_);
v___x_1265_ = l_Nat_reprFast(v___x_1264_);
v___x_1266_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1266_, 0, v___x_1265_);
v___x_1267_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1267_, 0, v___x_1263_);
lean_ctor_set(v___x_1267_, 1, v___x_1266_);
v___x_1268_ = 0;
v___x_1269_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1269_, 0, v___x_1267_);
lean_ctor_set_uint8(v___x_1269_, sizeof(void*)*1, v___x_1268_);
v___x_1270_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1270_, 0, v___x_1262_);
lean_ctor_set(v___x_1270_, 1, v___x_1269_);
v___x_1271_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__2));
v___x_1272_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1272_, 0, v___x_1270_);
lean_ctor_set(v___x_1272_, 1, v___x_1271_);
v___x_1273_ = lean_box(1);
v___x_1274_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1274_, 0, v___x_1272_);
lean_ctor_set(v___x_1274_, 1, v___x_1273_);
v___x_1275_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__6));
v___x_1276_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1276_, 0, v___x_1274_);
lean_ctor_set(v___x_1276_, 1, v___x_1275_);
v___x_1277_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1277_, 0, v___x_1276_);
lean_ctor_set(v___x_1277_, 1, v___x_1261_);
v___x_1278_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__7, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__7_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__7);
v___x_1279_ = lean_uint32_to_nat(v_nStack_1255_);
v___x_1280_ = l_Nat_reprFast(v___x_1279_);
v___x_1281_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1281_, 0, v___x_1280_);
v___x_1282_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1282_, 0, v___x_1278_);
lean_ctor_set(v___x_1282_, 1, v___x_1281_);
v___x_1283_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1283_, 0, v___x_1282_);
lean_ctor_set_uint8(v___x_1283_, sizeof(void*)*1, v___x_1268_);
v___x_1284_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1284_, 0, v___x_1277_);
lean_ctor_set(v___x_1284_, 1, v___x_1283_);
v___x_1285_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1285_, 0, v___x_1284_);
lean_ctor_set(v___x_1285_, 1, v___x_1271_);
v___x_1286_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1286_, 0, v___x_1285_);
lean_ctor_set(v___x_1286_, 1, v___x_1273_);
v___x_1287_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__8));
v___x_1288_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1288_, 0, v___x_1286_);
lean_ctor_set(v___x_1288_, 1, v___x_1287_);
v___x_1289_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1289_, 0, v___x_1288_);
lean_ctor_set(v___x_1289_, 1, v___x_1261_);
v___x_1290_ = lean_uint32_to_nat(v_wStack_1256_);
v___x_1291_ = l_Nat_reprFast(v___x_1290_);
v___x_1292_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1292_, 0, v___x_1291_);
v___x_1293_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1293_, 0, v___x_1278_);
lean_ctor_set(v___x_1293_, 1, v___x_1292_);
v___x_1294_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1294_, 0, v___x_1293_);
lean_ctor_set_uint8(v___x_1294_, sizeof(void*)*1, v___x_1268_);
v___x_1295_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1295_, 0, v___x_1289_);
lean_ctor_set(v___x_1295_, 1, v___x_1294_);
v___x_1296_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1296_, 0, v___x_1295_);
lean_ctor_set(v___x_1296_, 1, v___x_1271_);
v___x_1297_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1297_, 0, v___x_1296_);
lean_ctor_set(v___x_1297_, 1, v___x_1273_);
v___x_1298_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__10));
v___x_1299_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1299_, 0, v___x_1297_);
lean_ctor_set(v___x_1299_, 1, v___x_1298_);
v___x_1300_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1300_, 0, v___x_1299_);
lean_ctor_set(v___x_1300_, 1, v___x_1261_);
v___x_1301_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__10, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__10_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__10);
v___x_1302_ = lean_uint32_to_nat(v_logBlowup_1257_);
v___x_1303_ = l_Nat_reprFast(v___x_1302_);
v___x_1304_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1304_, 0, v___x_1303_);
v___x_1305_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1305_, 0, v___x_1301_);
lean_ctor_set(v___x_1305_, 1, v___x_1304_);
v___x_1306_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1306_, 0, v___x_1305_);
lean_ctor_set_uint8(v___x_1306_, sizeof(void*)*1, v___x_1268_);
v___x_1307_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1307_, 0, v___x_1300_);
lean_ctor_set(v___x_1307_, 1, v___x_1306_);
v___x_1308_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1308_, 0, v___x_1307_);
lean_ctor_set(v___x_1308_, 1, v___x_1271_);
v___x_1309_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1309_, 0, v___x_1308_);
lean_ctor_set(v___x_1309_, 1, v___x_1273_);
v___x_1310_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__12));
v___x_1311_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1311_, 0, v___x_1309_);
lean_ctor_set(v___x_1311_, 1, v___x_1310_);
v___x_1312_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1312_, 0, v___x_1311_);
lean_ctor_set(v___x_1312_, 1, v___x_1261_);
v___x_1313_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__13, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__13_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__13);
v___x_1314_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg(v_whir_1258_);
v___x_1315_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1315_, 0, v___x_1313_);
lean_ctor_set(v___x_1315_, 1, v___x_1314_);
v___x_1316_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1316_, 0, v___x_1315_);
lean_ctor_set_uint8(v___x_1316_, sizeof(void*)*1, v___x_1268_);
v___x_1317_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1317_, 0, v___x_1312_);
lean_ctor_set(v___x_1317_, 1, v___x_1316_);
v___x_1318_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1318_, 0, v___x_1317_);
lean_ctor_set(v___x_1318_, 1, v___x_1271_);
v___x_1319_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1319_, 0, v___x_1318_);
lean_ctor_set(v___x_1319_, 1, v___x_1273_);
v___x_1320_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__15));
v___x_1321_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1321_, 0, v___x_1319_);
lean_ctor_set(v___x_1321_, 1, v___x_1320_);
v___x_1322_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1322_, 0, v___x_1321_);
lean_ctor_set(v___x_1322_, 1, v___x_1261_);
v___x_1323_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg(v_logup_1259_);
lean_dec_ref(v_logup_1259_);
v___x_1324_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1324_, 0, v___x_1263_);
lean_ctor_set(v___x_1324_, 1, v___x_1323_);
v___x_1325_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1325_, 0, v___x_1324_);
lean_ctor_set_uint8(v___x_1325_, sizeof(void*)*1, v___x_1268_);
v___x_1326_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1326_, 0, v___x_1322_);
lean_ctor_set(v___x_1326_, 1, v___x_1325_);
v___x_1327_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1327_, 0, v___x_1326_);
lean_ctor_set(v___x_1327_, 1, v___x_1271_);
v___x_1328_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1328_, 0, v___x_1327_);
lean_ctor_set(v___x_1328_, 1, v___x_1273_);
v___x_1329_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__17));
v___x_1330_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1330_, 0, v___x_1328_);
lean_ctor_set(v___x_1330_, 1, v___x_1329_);
v___x_1331_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1331_, 0, v___x_1330_);
lean_ctor_set(v___x_1331_, 1, v___x_1261_);
v___x_1332_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__4);
v___x_1333_ = lean_uint32_to_nat(v_maxConstraintDegree_1260_);
v___x_1334_ = l_Nat_reprFast(v___x_1333_);
v___x_1335_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1335_, 0, v___x_1334_);
v___x_1336_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1336_, 0, v___x_1332_);
lean_ctor_set(v___x_1336_, 1, v___x_1335_);
v___x_1337_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1337_, 0, v___x_1336_);
lean_ctor_set_uint8(v___x_1337_, sizeof(void*)*1, v___x_1268_);
v___x_1338_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1338_, 0, v___x_1331_);
lean_ctor_set(v___x_1338_, 1, v___x_1337_);
v___x_1339_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10);
v___x_1340_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__11));
v___x_1341_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1341_, 0, v___x_1340_);
lean_ctor_set(v___x_1341_, 1, v___x_1338_);
v___x_1342_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__12));
v___x_1343_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1343_, 0, v___x_1341_);
lean_ctor_set(v___x_1343_, 1, v___x_1342_);
v___x_1344_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1344_, 0, v___x_1339_);
lean_ctor_set(v___x_1344_, 1, v___x_1343_);
v___x_1345_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1345_, 0, v___x_1344_);
lean_ctor_set_uint8(v___x_1345_, sizeof(void*)*1, v___x_1268_);
return v___x_1345_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr(lean_object* v_x_1346_, lean_object* v_prec_1347_){
_start:
{
lean_object* v___x_1348_; 
v___x_1348_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg(v_x_1346_);
return v___x_1348_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___boxed(lean_object* v_x_1349_, lean_object* v_prec_1350_){
_start:
{
lean_object* v_res_1351_; 
v_res_1351_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr(v_x_1349_, v_prec_1350_);
lean_dec(v_prec_1350_);
return v_res_1351_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Option_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__0(lean_object* v_x_1366_, lean_object* v_x_1367_){
_start:
{
if (lean_obj_tag(v_x_1366_) == 0)
{
lean_object* v___x_1368_; 
v___x_1368_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Option_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__0___closed__1));
return v___x_1368_;
}
else
{
lean_object* v_val_1369_; lean_object* v___x_1371_; uint8_t v_isShared_1372_; uint8_t v_isSharedCheck_1382_; 
v_val_1369_ = lean_ctor_get(v_x_1366_, 0);
v_isSharedCheck_1382_ = !lean_is_exclusive(v_x_1366_);
if (v_isSharedCheck_1382_ == 0)
{
v___x_1371_ = v_x_1366_;
v_isShared_1372_ = v_isSharedCheck_1382_;
goto v_resetjp_1370_;
}
else
{
lean_inc(v_val_1369_);
lean_dec(v_x_1366_);
v___x_1371_ = lean_box(0);
v_isShared_1372_ = v_isSharedCheck_1382_;
goto v_resetjp_1370_;
}
v_resetjp_1370_:
{
lean_object* v___x_1373_; uint32_t v___x_1374_; lean_object* v___x_1375_; lean_object* v___x_1376_; lean_object* v___x_1378_; 
v___x_1373_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Option_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__0___closed__3));
v___x_1374_ = lean_unbox_uint32(v_val_1369_);
lean_dec(v_val_1369_);
v___x_1375_ = lean_uint32_to_nat(v___x_1374_);
v___x_1376_ = l_Nat_reprFast(v___x_1375_);
if (v_isShared_1372_ == 0)
{
lean_ctor_set_tag(v___x_1371_, 3);
lean_ctor_set(v___x_1371_, 0, v___x_1376_);
v___x_1378_ = v___x_1371_;
goto v_reusejp_1377_;
}
else
{
lean_object* v_reuseFailAlloc_1381_; 
v_reuseFailAlloc_1381_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1381_, 0, v___x_1376_);
v___x_1378_ = v_reuseFailAlloc_1381_;
goto v_reusejp_1377_;
}
v_reusejp_1377_:
{
lean_object* v___x_1379_; lean_object* v___x_1380_; 
v___x_1379_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1379_, 0, v___x_1373_);
lean_ctor_set(v___x_1379_, 1, v___x_1378_);
v___x_1380_ = l_Repr_addAppParen(v___x_1379_, v_x_1367_);
return v___x_1380_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Option_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__0___boxed(lean_object* v_x_1383_, lean_object* v_x_1384_){
_start:
{
lean_object* v_res_1385_; 
v_res_1385_ = lp_swirl_x2drbr_x2dformal_Option_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__0(v_x_1383_, v_x_1384_);
lean_dec(v_x_1384_);
return v_res_1385_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__1_spec__1___lam__0(uint32_t v___y_1386_){
_start:
{
lean_object* v___x_1387_; lean_object* v___x_1388_; lean_object* v___x_1389_; 
v___x_1387_ = lean_uint32_to_nat(v___y_1386_);
v___x_1388_ = l_Nat_reprFast(v___x_1387_);
v___x_1389_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1389_, 0, v___x_1388_);
return v___x_1389_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__1_spec__1___lam__0___boxed(lean_object* v___y_1390_){
_start:
{
uint32_t v___y_474__boxed_1391_; lean_object* v_res_1392_; 
v___y_474__boxed_1391_ = lean_unbox_uint32(v___y_1390_);
lean_dec(v___y_1390_);
v_res_1392_ = lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__1_spec__1___lam__0(v___y_474__boxed_1391_);
return v_res_1392_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__1_spec__1_spec__2_spec__3(lean_object* v_x_1393_, lean_object* v_x_1394_, lean_object* v_x_1395_){
_start:
{
if (lean_obj_tag(v_x_1395_) == 0)
{
lean_dec(v_x_1393_);
return v_x_1394_;
}
else
{
lean_object* v_head_1396_; lean_object* v_tail_1397_; lean_object* v___x_1399_; uint8_t v_isShared_1400_; uint8_t v_isSharedCheck_1410_; 
v_head_1396_ = lean_ctor_get(v_x_1395_, 0);
v_tail_1397_ = lean_ctor_get(v_x_1395_, 1);
v_isSharedCheck_1410_ = !lean_is_exclusive(v_x_1395_);
if (v_isSharedCheck_1410_ == 0)
{
v___x_1399_ = v_x_1395_;
v_isShared_1400_ = v_isSharedCheck_1410_;
goto v_resetjp_1398_;
}
else
{
lean_inc(v_tail_1397_);
lean_inc(v_head_1396_);
lean_dec(v_x_1395_);
v___x_1399_ = lean_box(0);
v_isShared_1400_ = v_isSharedCheck_1410_;
goto v_resetjp_1398_;
}
v_resetjp_1398_:
{
lean_object* v___x_1402_; 
lean_inc(v_x_1393_);
if (v_isShared_1400_ == 0)
{
lean_ctor_set_tag(v___x_1399_, 5);
lean_ctor_set(v___x_1399_, 1, v_x_1393_);
lean_ctor_set(v___x_1399_, 0, v_x_1394_);
v___x_1402_ = v___x_1399_;
goto v_reusejp_1401_;
}
else
{
lean_object* v_reuseFailAlloc_1409_; 
v_reuseFailAlloc_1409_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1409_, 0, v_x_1394_);
lean_ctor_set(v_reuseFailAlloc_1409_, 1, v_x_1393_);
v___x_1402_ = v_reuseFailAlloc_1409_;
goto v_reusejp_1401_;
}
v_reusejp_1401_:
{
uint32_t v___x_1403_; lean_object* v___x_1404_; lean_object* v___x_1405_; lean_object* v___x_1406_; lean_object* v___x_1407_; 
v___x_1403_ = lean_unbox_uint32(v_head_1396_);
lean_dec(v_head_1396_);
v___x_1404_ = lean_uint32_to_nat(v___x_1403_);
v___x_1405_ = l_Nat_reprFast(v___x_1404_);
v___x_1406_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1406_, 0, v___x_1405_);
v___x_1407_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1407_, 0, v___x_1402_);
lean_ctor_set(v___x_1407_, 1, v___x_1406_);
v_x_1394_ = v___x_1407_;
v_x_1395_ = v_tail_1397_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__1_spec__1_spec__2(lean_object* v_x_1411_, lean_object* v_x_1412_, lean_object* v_x_1413_){
_start:
{
if (lean_obj_tag(v_x_1413_) == 0)
{
lean_dec(v_x_1411_);
return v_x_1412_;
}
else
{
lean_object* v_head_1414_; lean_object* v_tail_1415_; lean_object* v___x_1417_; uint8_t v_isShared_1418_; uint8_t v_isSharedCheck_1428_; 
v_head_1414_ = lean_ctor_get(v_x_1413_, 0);
v_tail_1415_ = lean_ctor_get(v_x_1413_, 1);
v_isSharedCheck_1428_ = !lean_is_exclusive(v_x_1413_);
if (v_isSharedCheck_1428_ == 0)
{
v___x_1417_ = v_x_1413_;
v_isShared_1418_ = v_isSharedCheck_1428_;
goto v_resetjp_1416_;
}
else
{
lean_inc(v_tail_1415_);
lean_inc(v_head_1414_);
lean_dec(v_x_1413_);
v___x_1417_ = lean_box(0);
v_isShared_1418_ = v_isSharedCheck_1428_;
goto v_resetjp_1416_;
}
v_resetjp_1416_:
{
lean_object* v___x_1420_; 
lean_inc(v_x_1411_);
if (v_isShared_1418_ == 0)
{
lean_ctor_set_tag(v___x_1417_, 5);
lean_ctor_set(v___x_1417_, 1, v_x_1411_);
lean_ctor_set(v___x_1417_, 0, v_x_1412_);
v___x_1420_ = v___x_1417_;
goto v_reusejp_1419_;
}
else
{
lean_object* v_reuseFailAlloc_1427_; 
v_reuseFailAlloc_1427_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1427_, 0, v_x_1412_);
lean_ctor_set(v_reuseFailAlloc_1427_, 1, v_x_1411_);
v___x_1420_ = v_reuseFailAlloc_1427_;
goto v_reusejp_1419_;
}
v_reusejp_1419_:
{
uint32_t v___x_1421_; lean_object* v___x_1422_; lean_object* v___x_1423_; lean_object* v___x_1424_; lean_object* v___x_1425_; lean_object* v___x_1426_; 
v___x_1421_ = lean_unbox_uint32(v_head_1414_);
lean_dec(v_head_1414_);
v___x_1422_ = lean_uint32_to_nat(v___x_1421_);
v___x_1423_ = l_Nat_reprFast(v___x_1422_);
v___x_1424_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1424_, 0, v___x_1423_);
v___x_1425_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1425_, 0, v___x_1420_);
lean_ctor_set(v___x_1425_, 1, v___x_1424_);
v___x_1426_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__1_spec__1_spec__2_spec__3(v_x_1411_, v___x_1425_, v_tail_1415_);
return v___x_1426_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__1_spec__1(lean_object* v_x_1429_, lean_object* v_x_1430_){
_start:
{
if (lean_obj_tag(v_x_1429_) == 0)
{
lean_object* v___x_1431_; 
lean_dec(v_x_1430_);
v___x_1431_ = lean_box(0);
return v___x_1431_;
}
else
{
lean_object* v_tail_1432_; 
v_tail_1432_ = lean_ctor_get(v_x_1429_, 1);
if (lean_obj_tag(v_tail_1432_) == 0)
{
lean_object* v_head_1433_; uint32_t v___x_1434_; lean_object* v___x_1435_; 
lean_dec(v_x_1430_);
v_head_1433_ = lean_ctor_get(v_x_1429_, 0);
lean_inc(v_head_1433_);
lean_dec_ref_known(v_x_1429_, 2);
v___x_1434_ = lean_unbox_uint32(v_head_1433_);
lean_dec(v_head_1433_);
v___x_1435_ = lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__1_spec__1___lam__0(v___x_1434_);
return v___x_1435_;
}
else
{
lean_object* v_head_1436_; uint32_t v___x_1437_; lean_object* v___x_1438_; lean_object* v___x_1439_; 
lean_inc(v_tail_1432_);
v_head_1436_ = lean_ctor_get(v_x_1429_, 0);
lean_inc(v_head_1436_);
lean_dec_ref_known(v_x_1429_, 2);
v___x_1437_ = lean_unbox_uint32(v_head_1436_);
lean_dec(v_head_1436_);
v___x_1438_ = lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__1_spec__1___lam__0(v___x_1437_);
v___x_1439_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__1_spec__1_spec__2(v_x_1430_, v___x_1438_, v_tail_1432_);
return v___x_1439_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__1(lean_object* v_xs_1440_){
_start:
{
lean_object* v___x_1441_; lean_object* v___x_1442_; uint8_t v___x_1443_; 
v___x_1441_ = lean_array_get_size(v_xs_1440_);
v___x_1442_ = lean_unsigned_to_nat(0u);
v___x_1443_ = lean_nat_dec_eq(v___x_1441_, v___x_1442_);
if (v___x_1443_ == 0)
{
lean_object* v___x_1444_; lean_object* v___x_1445_; lean_object* v___x_1446_; lean_object* v___x_1447_; lean_object* v___x_1448_; lean_object* v___x_1449_; lean_object* v___x_1450_; lean_object* v___x_1451_; lean_object* v___x_1452_; lean_object* v___x_1453_; 
v___x_1444_ = lean_array_to_list(v_xs_1440_);
v___x_1445_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__3));
v___x_1446_ = lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__1_spec__1(v___x_1444_, v___x_1445_);
v___x_1447_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__6, &lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__6_once, _init_lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__6);
v___x_1448_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__7));
v___x_1449_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1449_, 0, v___x_1448_);
lean_ctor_set(v___x_1449_, 1, v___x_1446_);
v___x_1450_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__8));
v___x_1451_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1451_, 0, v___x_1449_);
lean_ctor_set(v___x_1451_, 1, v___x_1450_);
v___x_1452_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1452_, 0, v___x_1447_);
lean_ctor_set(v___x_1452_, 1, v___x_1451_);
v___x_1453_ = l_Std_Format_fill(v___x_1452_);
return v___x_1453_;
}
else
{
lean_object* v___x_1454_; 
lean_dec_ref(v_xs_1440_);
v___x_1454_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__10));
return v___x_1454_;
}
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__4(void){
_start:
{
lean_object* v___x_1464_; lean_object* v___x_1465_; 
v___x_1464_ = lean_unsigned_to_nat(16u);
v___x_1465_ = lean_nat_to_int(v___x_1464_);
return v___x_1465_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__7(void){
_start:
{
lean_object* v___x_1469_; lean_object* v___x_1470_; 
v___x_1469_ = lean_unsigned_to_nat(15u);
v___x_1470_ = lean_nat_to_int(v___x_1469_);
return v___x_1470_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg(lean_object* v_x_1474_){
_start:
{
lean_object* v_preprocessed_1475_; lean_object* v_cachedMains_1476_; uint32_t v_commonMain_1477_; lean_object* v___x_1478_; lean_object* v___x_1479_; lean_object* v___x_1480_; lean_object* v___x_1481_; lean_object* v___x_1482_; lean_object* v___x_1483_; uint8_t v___x_1484_; lean_object* v___x_1485_; lean_object* v___x_1486_; lean_object* v___x_1487_; lean_object* v___x_1488_; lean_object* v___x_1489_; lean_object* v___x_1490_; lean_object* v___x_1491_; lean_object* v___x_1492_; lean_object* v___x_1493_; lean_object* v___x_1494_; lean_object* v___x_1495_; lean_object* v___x_1496_; lean_object* v___x_1497_; lean_object* v___x_1498_; lean_object* v___x_1499_; lean_object* v___x_1500_; lean_object* v___x_1501_; lean_object* v___x_1502_; lean_object* v___x_1503_; lean_object* v___x_1504_; lean_object* v___x_1505_; lean_object* v___x_1506_; lean_object* v___x_1507_; lean_object* v___x_1508_; lean_object* v___x_1509_; lean_object* v___x_1510_; lean_object* v___x_1511_; lean_object* v___x_1512_; lean_object* v___x_1513_; lean_object* v___x_1514_; lean_object* v___x_1515_; lean_object* v___x_1516_; lean_object* v___x_1517_; 
v_preprocessed_1475_ = lean_ctor_get(v_x_1474_, 0);
lean_inc(v_preprocessed_1475_);
v_cachedMains_1476_ = lean_ctor_get(v_x_1474_, 1);
lean_inc_ref(v_cachedMains_1476_);
v_commonMain_1477_ = lean_ctor_get_uint32(v_x_1474_, sizeof(void*)*2);
lean_dec_ref(v_x_1474_);
v___x_1478_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__5));
v___x_1479_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__3));
v___x_1480_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__4);
v___x_1481_ = lean_unsigned_to_nat(0u);
v___x_1482_ = lp_swirl_x2drbr_x2dformal_Option_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__0(v_preprocessed_1475_, v___x_1481_);
v___x_1483_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1483_, 0, v___x_1480_);
lean_ctor_set(v___x_1483_, 1, v___x_1482_);
v___x_1484_ = 0;
v___x_1485_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1485_, 0, v___x_1483_);
lean_ctor_set_uint8(v___x_1485_, sizeof(void*)*1, v___x_1484_);
v___x_1486_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1486_, 0, v___x_1479_);
lean_ctor_set(v___x_1486_, 1, v___x_1485_);
v___x_1487_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__2));
v___x_1488_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1488_, 0, v___x_1486_);
lean_ctor_set(v___x_1488_, 1, v___x_1487_);
v___x_1489_ = lean_box(1);
v___x_1490_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1490_, 0, v___x_1488_);
lean_ctor_set(v___x_1490_, 1, v___x_1489_);
v___x_1491_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__6));
v___x_1492_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1492_, 0, v___x_1490_);
lean_ctor_set(v___x_1492_, 1, v___x_1491_);
v___x_1493_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1493_, 0, v___x_1492_);
lean_ctor_set(v___x_1493_, 1, v___x_1478_);
v___x_1494_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__7, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__7_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__7);
v___x_1495_ = lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__1(v_cachedMains_1476_);
v___x_1496_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1496_, 0, v___x_1494_);
lean_ctor_set(v___x_1496_, 1, v___x_1495_);
v___x_1497_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1497_, 0, v___x_1496_);
lean_ctor_set_uint8(v___x_1497_, sizeof(void*)*1, v___x_1484_);
v___x_1498_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1498_, 0, v___x_1493_);
lean_ctor_set(v___x_1498_, 1, v___x_1497_);
v___x_1499_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1499_, 0, v___x_1498_);
lean_ctor_set(v___x_1499_, 1, v___x_1487_);
v___x_1500_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1500_, 0, v___x_1499_);
lean_ctor_set(v___x_1500_, 1, v___x_1489_);
v___x_1501_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__9));
v___x_1502_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1502_, 0, v___x_1500_);
lean_ctor_set(v___x_1502_, 1, v___x_1501_);
v___x_1503_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1503_, 0, v___x_1502_);
lean_ctor_set(v___x_1503_, 1, v___x_1478_);
v___x_1504_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__7, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__7_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__7);
v___x_1505_ = lean_uint32_to_nat(v_commonMain_1477_);
v___x_1506_ = l_Nat_reprFast(v___x_1505_);
v___x_1507_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1507_, 0, v___x_1506_);
v___x_1508_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1508_, 0, v___x_1504_);
lean_ctor_set(v___x_1508_, 1, v___x_1507_);
v___x_1509_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1509_, 0, v___x_1508_);
lean_ctor_set_uint8(v___x_1509_, sizeof(void*)*1, v___x_1484_);
v___x_1510_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1510_, 0, v___x_1503_);
lean_ctor_set(v___x_1510_, 1, v___x_1509_);
v___x_1511_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10);
v___x_1512_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__11));
v___x_1513_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1513_, 0, v___x_1512_);
lean_ctor_set(v___x_1513_, 1, v___x_1510_);
v___x_1514_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__12));
v___x_1515_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1515_, 0, v___x_1513_);
lean_ctor_set(v___x_1515_, 1, v___x_1514_);
v___x_1516_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1516_, 0, v___x_1511_);
lean_ctor_set(v___x_1516_, 1, v___x_1515_);
v___x_1517_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1517_, 0, v___x_1516_);
lean_ctor_set_uint8(v___x_1517_, sizeof(void*)*1, v___x_1484_);
return v___x_1517_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr(lean_object* v_x_1518_, lean_object* v_prec_1519_){
_start:
{
lean_object* v___x_1520_; 
v___x_1520_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg(v_x_1518_);
return v___x_1520_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___boxed(lean_object* v_x_1521_, lean_object* v_prec_1522_){
_start:
{
lean_object* v_res_1523_; 
v_res_1523_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr(v_x_1521_, v_prec_1522_);
lean_dec(v_prec_1522_);
return v_res_1523_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Swirl_Protocol_Noninteractive_Wire_Raw_RawTraceWidth_totalWidth_spec__0(lean_object* v_as_1526_, size_t v_i_1527_, size_t v_stop_1528_, lean_object* v_b_1529_){
_start:
{
uint8_t v___x_1530_; 
v___x_1530_ = lean_usize_dec_eq(v_i_1527_, v_stop_1528_);
if (v___x_1530_ == 0)
{
lean_object* v___x_1531_; uint32_t v___x_1532_; lean_object* v___x_1533_; lean_object* v___x_1534_; size_t v___x_1535_; size_t v___x_1536_; 
v___x_1531_ = lean_array_uget_borrowed(v_as_1526_, v_i_1527_);
v___x_1532_ = lean_unbox_uint32(v___x_1531_);
v___x_1533_ = lean_uint32_to_nat(v___x_1532_);
v___x_1534_ = lean_nat_add(v_b_1529_, v___x_1533_);
lean_dec(v___x_1533_);
lean_dec(v_b_1529_);
v___x_1535_ = ((size_t)1ULL);
v___x_1536_ = lean_usize_add(v_i_1527_, v___x_1535_);
v_i_1527_ = v___x_1536_;
v_b_1529_ = v___x_1534_;
goto _start;
}
else
{
return v_b_1529_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Swirl_Protocol_Noninteractive_Wire_Raw_RawTraceWidth_totalWidth_spec__0___boxed(lean_object* v_as_1538_, lean_object* v_i_1539_, lean_object* v_stop_1540_, lean_object* v_b_1541_){
_start:
{
size_t v_i_boxed_1542_; size_t v_stop_boxed_1543_; lean_object* v_res_1544_; 
v_i_boxed_1542_ = lean_unbox_usize(v_i_1539_);
lean_dec(v_i_1539_);
v_stop_boxed_1543_ = lean_unbox_usize(v_stop_1540_);
lean_dec(v_stop_1540_);
v_res_1544_ = lp_swirl_x2drbr_x2dformal___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Swirl_Protocol_Noninteractive_Wire_Raw_RawTraceWidth_totalWidth_spec__0(v_as_1538_, v_i_boxed_1542_, v_stop_boxed_1543_, v_b_1541_);
lean_dec_ref(v_as_1538_);
return v_res_1544_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawTraceWidth_totalWidth(lean_object* v_w_1545_){
_start:
{
lean_object* v_preprocessed_1546_; lean_object* v_cachedMains_1547_; uint32_t v_commonMain_1548_; uint32_t v___y_1550_; 
v_preprocessed_1546_ = lean_ctor_get(v_w_1545_, 0);
v_cachedMains_1547_ = lean_ctor_get(v_w_1545_, 1);
v_commonMain_1548_ = lean_ctor_get_uint32(v_w_1545_, sizeof(void*)*2);
if (lean_obj_tag(v_preprocessed_1546_) == 0)
{
uint32_t v___x_1566_; 
v___x_1566_ = 0;
v___y_1550_ = v___x_1566_;
goto v___jp_1549_;
}
else
{
lean_object* v_val_1567_; uint32_t v___x_1568_; 
v_val_1567_ = lean_ctor_get(v_preprocessed_1546_, 0);
v___x_1568_ = lean_unbox_uint32(v_val_1567_);
v___y_1550_ = v___x_1568_;
goto v___jp_1549_;
}
v___jp_1549_:
{
lean_object* v___x_1551_; lean_object* v___x_1552_; lean_object* v___x_1553_; lean_object* v___x_1554_; lean_object* v___x_1555_; uint8_t v___x_1556_; 
v___x_1551_ = lean_uint32_to_nat(v___y_1550_);
v___x_1552_ = lean_uint32_to_nat(v_commonMain_1548_);
v___x_1553_ = lean_nat_add(v___x_1551_, v___x_1552_);
lean_dec(v___x_1552_);
lean_dec(v___x_1551_);
v___x_1554_ = lean_unsigned_to_nat(0u);
v___x_1555_ = lean_array_get_size(v_cachedMains_1547_);
v___x_1556_ = lean_nat_dec_lt(v___x_1554_, v___x_1555_);
if (v___x_1556_ == 0)
{
return v___x_1553_;
}
else
{
uint8_t v___x_1557_; 
v___x_1557_ = lean_nat_dec_le(v___x_1555_, v___x_1555_);
if (v___x_1557_ == 0)
{
if (v___x_1556_ == 0)
{
return v___x_1553_;
}
else
{
size_t v___x_1558_; size_t v___x_1559_; lean_object* v___x_1560_; lean_object* v___x_1561_; 
v___x_1558_ = ((size_t)0ULL);
v___x_1559_ = lean_usize_of_nat(v___x_1555_);
v___x_1560_ = lp_swirl_x2drbr_x2dformal___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Swirl_Protocol_Noninteractive_Wire_Raw_RawTraceWidth_totalWidth_spec__0(v_cachedMains_1547_, v___x_1558_, v___x_1559_, v___x_1554_);
v___x_1561_ = lean_nat_add(v___x_1553_, v___x_1560_);
lean_dec(v___x_1560_);
lean_dec(v___x_1553_);
return v___x_1561_;
}
}
else
{
size_t v___x_1562_; size_t v___x_1563_; lean_object* v___x_1564_; lean_object* v___x_1565_; 
v___x_1562_ = ((size_t)0ULL);
v___x_1563_ = lean_usize_of_nat(v___x_1555_);
v___x_1564_ = lp_swirl_x2drbr_x2dformal___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Swirl_Protocol_Noninteractive_Wire_Raw_RawTraceWidth_totalWidth_spec__0(v_cachedMains_1547_, v___x_1562_, v___x_1563_, v___x_1554_);
v___x_1565_ = lean_nat_add(v___x_1553_, v___x_1564_);
lean_dec(v___x_1564_);
lean_dec(v___x_1553_);
return v___x_1565_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawTraceWidth_totalWidth___boxed(lean_object* v_w_1569_){
_start:
{
lean_object* v_res_1570_; 
v_res_1570_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawTraceWidth_totalWidth(v_w_1569_);
lean_dec_ref(v_w_1569_);
return v_res_1570_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__6(void){
_start:
{
lean_object* v___x_1589_; lean_object* v___x_1590_; 
v___x_1589_ = lean_unsigned_to_nat(19u);
v___x_1590_ = lean_nat_to_int(v___x_1589_);
return v___x_1590_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg(lean_object* v_x_1594_){
_start:
{
lean_object* v_width_1595_; uint32_t v_numPublicValues_1596_; uint8_t v_needRot_1597_; lean_object* v___x_1598_; lean_object* v___x_1599_; lean_object* v___x_1600_; lean_object* v___x_1601_; lean_object* v___x_1602_; uint8_t v___x_1603_; lean_object* v___x_1604_; lean_object* v___x_1605_; lean_object* v___x_1606_; lean_object* v___x_1607_; lean_object* v___x_1608_; lean_object* v___x_1609_; lean_object* v___x_1610_; lean_object* v___x_1611_; lean_object* v___x_1612_; lean_object* v___x_1613_; lean_object* v___x_1614_; lean_object* v___x_1615_; lean_object* v___x_1616_; lean_object* v___x_1617_; lean_object* v___x_1618_; lean_object* v___x_1619_; lean_object* v___x_1620_; lean_object* v___x_1621_; lean_object* v___x_1622_; lean_object* v___x_1623_; lean_object* v___x_1624_; lean_object* v___x_1625_; lean_object* v___x_1626_; lean_object* v___x_1627_; lean_object* v___x_1628_; lean_object* v___x_1629_; lean_object* v___x_1630_; lean_object* v___x_1631_; lean_object* v___x_1632_; lean_object* v___x_1633_; lean_object* v___x_1634_; lean_object* v___x_1635_; lean_object* v___x_1636_; 
v_width_1595_ = lean_ctor_get(v_x_1594_, 0);
lean_inc_ref(v_width_1595_);
v_numPublicValues_1596_ = lean_ctor_get_uint32(v_x_1594_, sizeof(void*)*1);
v_needRot_1597_ = lean_ctor_get_uint8(v_x_1594_, sizeof(void*)*1 + 4);
lean_dec_ref(v_x_1594_);
v___x_1598_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__5));
v___x_1599_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__3));
v___x_1600_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__4);
v___x_1601_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg(v_width_1595_);
v___x_1602_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1602_, 0, v___x_1600_);
lean_ctor_set(v___x_1602_, 1, v___x_1601_);
v___x_1603_ = 0;
v___x_1604_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1604_, 0, v___x_1602_);
lean_ctor_set_uint8(v___x_1604_, sizeof(void*)*1, v___x_1603_);
v___x_1605_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1605_, 0, v___x_1599_);
lean_ctor_set(v___x_1605_, 1, v___x_1604_);
v___x_1606_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__2));
v___x_1607_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1607_, 0, v___x_1605_);
lean_ctor_set(v___x_1607_, 1, v___x_1606_);
v___x_1608_ = lean_box(1);
v___x_1609_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1609_, 0, v___x_1607_);
lean_ctor_set(v___x_1609_, 1, v___x_1608_);
v___x_1610_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__5));
v___x_1611_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1611_, 0, v___x_1609_);
lean_ctor_set(v___x_1611_, 1, v___x_1610_);
v___x_1612_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1612_, 0, v___x_1611_);
lean_ctor_set(v___x_1612_, 1, v___x_1598_);
v___x_1613_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__6, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__6_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__6);
v___x_1614_ = lean_uint32_to_nat(v_numPublicValues_1596_);
v___x_1615_ = l_Nat_reprFast(v___x_1614_);
v___x_1616_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1616_, 0, v___x_1615_);
v___x_1617_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1617_, 0, v___x_1613_);
lean_ctor_set(v___x_1617_, 1, v___x_1616_);
v___x_1618_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1618_, 0, v___x_1617_);
lean_ctor_set_uint8(v___x_1618_, sizeof(void*)*1, v___x_1603_);
v___x_1619_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1619_, 0, v___x_1612_);
lean_ctor_set(v___x_1619_, 1, v___x_1618_);
v___x_1620_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1620_, 0, v___x_1619_);
lean_ctor_set(v___x_1620_, 1, v___x_1606_);
v___x_1621_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1621_, 0, v___x_1620_);
lean_ctor_set(v___x_1621_, 1, v___x_1608_);
v___x_1622_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg___closed__8));
v___x_1623_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1623_, 0, v___x_1621_);
lean_ctor_set(v___x_1623_, 1, v___x_1622_);
v___x_1624_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1624_, 0, v___x_1623_);
lean_ctor_set(v___x_1624_, 1, v___x_1598_);
v___x_1625_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__9, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__9_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__9);
v___x_1626_ = l_Bool_repr___redArg(v_needRot_1597_);
v___x_1627_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1627_, 0, v___x_1625_);
lean_ctor_set(v___x_1627_, 1, v___x_1626_);
v___x_1628_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1628_, 0, v___x_1627_);
lean_ctor_set_uint8(v___x_1628_, sizeof(void*)*1, v___x_1603_);
v___x_1629_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1629_, 0, v___x_1624_);
lean_ctor_set(v___x_1629_, 1, v___x_1628_);
v___x_1630_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10);
v___x_1631_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__11));
v___x_1632_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1632_, 0, v___x_1631_);
lean_ctor_set(v___x_1632_, 1, v___x_1629_);
v___x_1633_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__12));
v___x_1634_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1634_, 0, v___x_1632_);
lean_ctor_set(v___x_1634_, 1, v___x_1633_);
v___x_1635_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1635_, 0, v___x_1630_);
lean_ctor_set(v___x_1635_, 1, v___x_1634_);
v___x_1636_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1636_, 0, v___x_1635_);
lean_ctor_set_uint8(v___x_1636_, sizeof(void*)*1, v___x_1603_);
return v___x_1636_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr(lean_object* v_x_1637_, lean_object* v_prec_1638_){
_start:
{
lean_object* v___x_1639_; 
v___x_1639_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___redArg(v_x_1637_);
return v___x_1639_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr___boxed(lean_object* v_x_1640_, lean_object* v_prec_1641_){
_start:
{
lean_object* v_res_1642_; 
v_res_1642_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawStarkVerifyingParams_repr(v_x_1640_, v_prec_1641_);
lean_dec(v_prec_1641_);
return v_res_1642_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__0___boxed__const__1(void){
_start:
{
uint32_t v___x_1645_; lean_object* v___x_1646_; 
v___x_1645_ = 0;
v___x_1646_ = lean_box_uint32(v___x_1645_);
return v___x_1646_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__0(void){
_start:
{
lean_object* v___x_1647_; lean_object* v___x_1648_; lean_object* v___x_1649_; 
v___x_1647_ = lean_unsigned_to_nat(8u);
v___x_1648_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__0___boxed__const__1;
v___x_1649_ = lean_mk_array(v___x_1647_, v___x_1648_);
return v___x_1649_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__1(void){
_start:
{
lean_object* v___x_1650_; lean_object* v___x_1651_; 
v___x_1650_ = lean_unsigned_to_nat(0u);
v___x_1651_ = lean_nat_to_int(v___x_1650_);
return v___x_1651_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__2(void){
_start:
{
uint32_t v___x_1652_; lean_object* v___x_1653_; lean_object* v___x_1654_; lean_object* v___x_1655_; 
v___x_1652_ = 0;
v___x_1653_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__1, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__1_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__1);
v___x_1654_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__0, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__0_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__0);
v___x_1655_ = lean_alloc_ctor(0, 2, 4);
lean_ctor_set(v___x_1655_, 0, v___x_1654_);
lean_ctor_set(v___x_1655_, 1, v___x_1653_);
lean_ctor_set_uint32(v___x_1655_, sizeof(void*)*2, v___x_1652_);
return v___x_1655_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default(void){
_start:
{
lean_object* v___x_1656_; 
v___x_1656_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__2, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__2_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__2);
return v___x_1656_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData(void){
_start:
{
lean_object* v___x_1657_; 
v___x_1657_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default;
return v___x_1657_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg(lean_object* v_x_1675_){
_start:
{
lean_object* v_coefficients_1676_; uint32_t v_threshold_1677_; lean_object* v___x_1678_; lean_object* v___x_1679_; lean_object* v___x_1680_; lean_object* v___x_1681_; lean_object* v___x_1682_; uint8_t v___x_1683_; lean_object* v___x_1684_; lean_object* v___x_1685_; lean_object* v___x_1686_; lean_object* v___x_1687_; lean_object* v___x_1688_; lean_object* v___x_1689_; lean_object* v___x_1690_; lean_object* v___x_1691_; lean_object* v___x_1692_; lean_object* v___x_1693_; lean_object* v___x_1694_; lean_object* v___x_1695_; lean_object* v___x_1696_; lean_object* v___x_1697_; lean_object* v___x_1698_; lean_object* v___x_1699_; lean_object* v___x_1700_; lean_object* v___x_1701_; lean_object* v___x_1702_; lean_object* v___x_1703_; lean_object* v___x_1704_; lean_object* v___x_1705_; lean_object* v___x_1706_; 
v_coefficients_1676_ = lean_ctor_get(v_x_1675_, 0);
lean_inc_ref(v_coefficients_1676_);
v_threshold_1677_ = lean_ctor_get_uint32(v_x_1675_, sizeof(void*)*1);
lean_dec_ref(v_x_1675_);
v___x_1678_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__5));
v___x_1679_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg___closed__3));
v___x_1680_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__4);
v___x_1681_ = lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__1(v_coefficients_1676_);
v___x_1682_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1682_, 0, v___x_1680_);
lean_ctor_set(v___x_1682_, 1, v___x_1681_);
v___x_1683_ = 0;
v___x_1684_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1684_, 0, v___x_1682_);
lean_ctor_set_uint8(v___x_1684_, sizeof(void*)*1, v___x_1683_);
v___x_1685_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1685_, 0, v___x_1679_);
lean_ctor_set(v___x_1685_, 1, v___x_1684_);
v___x_1686_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__2));
v___x_1687_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1687_, 0, v___x_1685_);
lean_ctor_set(v___x_1687_, 1, v___x_1686_);
v___x_1688_ = lean_box(1);
v___x_1689_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1689_, 0, v___x_1687_);
lean_ctor_set(v___x_1689_, 1, v___x_1688_);
v___x_1690_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg___closed__5));
v___x_1691_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1691_, 0, v___x_1689_);
lean_ctor_set(v___x_1691_, 1, v___x_1690_);
v___x_1692_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1692_, 0, v___x_1691_);
lean_ctor_set(v___x_1692_, 1, v___x_1678_);
v___x_1693_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__10, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__10_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr___redArg___closed__10);
v___x_1694_ = lean_uint32_to_nat(v_threshold_1677_);
v___x_1695_ = l_Nat_reprFast(v___x_1694_);
v___x_1696_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1696_, 0, v___x_1695_);
v___x_1697_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1697_, 0, v___x_1693_);
lean_ctor_set(v___x_1697_, 1, v___x_1696_);
v___x_1698_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1698_, 0, v___x_1697_);
lean_ctor_set_uint8(v___x_1698_, sizeof(void*)*1, v___x_1683_);
v___x_1699_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1699_, 0, v___x_1692_);
lean_ctor_set(v___x_1699_, 1, v___x_1698_);
v___x_1700_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10);
v___x_1701_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__11));
v___x_1702_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1702_, 0, v___x_1701_);
lean_ctor_set(v___x_1702_, 1, v___x_1699_);
v___x_1703_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__12));
v___x_1704_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1704_, 0, v___x_1702_);
lean_ctor_set(v___x_1704_, 1, v___x_1703_);
v___x_1705_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1705_, 0, v___x_1700_);
lean_ctor_set(v___x_1705_, 1, v___x_1704_);
v___x_1706_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1706_, 0, v___x_1705_);
lean_ctor_set_uint8(v___x_1706_, sizeof(void*)*1, v___x_1683_);
return v___x_1706_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr(lean_object* v_x_1707_, lean_object* v_prec_1708_){
_start:
{
lean_object* v___x_1709_; 
v___x_1709_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___redArg(v_x_1707_);
return v___x_1709_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr___boxed(lean_object* v_x_1710_, lean_object* v_prec_1711_){
_start:
{
lean_object* v_res_1712_; 
v_res_1712_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLinearConstraint_repr(v_x_1710_, v_prec_1711_);
lean_dec(v_prec_1711_);
return v_res_1712_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_ctorIdx(lean_object* v_x_1715_){
_start:
{
switch(lean_obj_tag(v_x_1715_))
{
case 0:
{
lean_object* v___x_1716_; 
v___x_1716_ = lean_unsigned_to_nat(0u);
return v___x_1716_;
}
case 1:
{
lean_object* v___x_1717_; 
v___x_1717_ = lean_unsigned_to_nat(1u);
return v___x_1717_;
}
case 2:
{
lean_object* v___x_1718_; 
v___x_1718_ = lean_unsigned_to_nat(2u);
return v___x_1718_;
}
default: 
{
lean_object* v___x_1719_; 
v___x_1719_ = lean_unsigned_to_nat(3u);
return v___x_1719_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_ctorIdx___boxed(lean_object* v_x_1720_){
_start:
{
lean_object* v_res_1721_; 
v_res_1721_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_ctorIdx(v_x_1720_);
lean_dec(v_x_1720_);
return v_res_1721_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_ctorElim___redArg(lean_object* v_t_1722_, lean_object* v_k_1723_){
_start:
{
switch(lean_obj_tag(v_t_1722_))
{
case 0:
{
uint32_t v_offset_1724_; lean_object* v___x_1725_; lean_object* v___x_1726_; 
v_offset_1724_ = lean_ctor_get_uint32(v_t_1722_, 0);
v___x_1725_ = lean_box_uint32(v_offset_1724_);
v___x_1726_ = lean_apply_1(v_k_1723_, v___x_1725_);
return v___x_1726_;
}
case 1:
{
uint32_t v_partIndex_1727_; uint32_t v_offset_1728_; lean_object* v___x_1729_; lean_object* v___x_1730_; lean_object* v___x_1731_; 
v_partIndex_1727_ = lean_ctor_get_uint32(v_t_1722_, 0);
v_offset_1728_ = lean_ctor_get_uint32(v_t_1722_, 4);
v___x_1729_ = lean_box_uint32(v_partIndex_1727_);
v___x_1730_ = lean_box_uint32(v_offset_1728_);
v___x_1731_ = lean_apply_2(v_k_1723_, v___x_1729_, v___x_1730_);
return v___x_1731_;
}
default: 
{
return v_k_1723_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_ctorElim___redArg___boxed(lean_object* v_t_1732_, lean_object* v_k_1733_){
_start:
{
lean_object* v_res_1734_; 
v_res_1734_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_ctorElim___redArg(v_t_1732_, v_k_1733_);
lean_dec(v_t_1732_);
return v_res_1734_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_ctorElim(lean_object* v_motive_1735_, lean_object* v_ctorIdx_1736_, lean_object* v_t_1737_, lean_object* v_h_1738_, lean_object* v_k_1739_){
_start:
{
lean_object* v___x_1740_; 
v___x_1740_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_ctorElim___redArg(v_t_1737_, v_k_1739_);
return v___x_1740_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_ctorElim___boxed(lean_object* v_motive_1741_, lean_object* v_ctorIdx_1742_, lean_object* v_t_1743_, lean_object* v_h_1744_, lean_object* v_k_1745_){
_start:
{
lean_object* v_res_1746_; 
v_res_1746_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_ctorElim(v_motive_1741_, v_ctorIdx_1742_, v_t_1743_, v_h_1744_, v_k_1745_);
lean_dec(v_t_1743_);
lean_dec(v_ctorIdx_1742_);
return v_res_1746_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_preprocessed_elim___redArg(lean_object* v_t_1747_, lean_object* v_preprocessed_1748_){
_start:
{
lean_object* v___x_1749_; 
v___x_1749_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_ctorElim___redArg(v_t_1747_, v_preprocessed_1748_);
return v___x_1749_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_preprocessed_elim___redArg___boxed(lean_object* v_t_1750_, lean_object* v_preprocessed_1751_){
_start:
{
lean_object* v_res_1752_; 
v_res_1752_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_preprocessed_elim___redArg(v_t_1750_, v_preprocessed_1751_);
lean_dec(v_t_1750_);
return v_res_1752_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_preprocessed_elim(lean_object* v_motive_1753_, lean_object* v_t_1754_, lean_object* v_h_1755_, lean_object* v_preprocessed_1756_){
_start:
{
lean_object* v___x_1757_; 
v___x_1757_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_ctorElim___redArg(v_t_1754_, v_preprocessed_1756_);
return v___x_1757_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_preprocessed_elim___boxed(lean_object* v_motive_1758_, lean_object* v_t_1759_, lean_object* v_h_1760_, lean_object* v_preprocessed_1761_){
_start:
{
lean_object* v_res_1762_; 
v_res_1762_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_preprocessed_elim(v_motive_1758_, v_t_1759_, v_h_1760_, v_preprocessed_1761_);
lean_dec(v_t_1759_);
return v_res_1762_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_main_elim___redArg(lean_object* v_t_1763_, lean_object* v_main_1764_){
_start:
{
lean_object* v___x_1765_; 
v___x_1765_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_ctorElim___redArg(v_t_1763_, v_main_1764_);
return v___x_1765_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_main_elim___redArg___boxed(lean_object* v_t_1766_, lean_object* v_main_1767_){
_start:
{
lean_object* v_res_1768_; 
v_res_1768_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_main_elim___redArg(v_t_1766_, v_main_1767_);
lean_dec(v_t_1766_);
return v_res_1768_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_main_elim(lean_object* v_motive_1769_, lean_object* v_t_1770_, lean_object* v_h_1771_, lean_object* v_main_1772_){
_start:
{
lean_object* v___x_1773_; 
v___x_1773_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_ctorElim___redArg(v_t_1770_, v_main_1772_);
return v___x_1773_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_main_elim___boxed(lean_object* v_motive_1774_, lean_object* v_t_1775_, lean_object* v_h_1776_, lean_object* v_main_1777_){
_start:
{
lean_object* v_res_1778_; 
v_res_1778_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_main_elim(v_motive_1774_, v_t_1775_, v_h_1776_, v_main_1777_);
lean_dec(v_t_1775_);
return v_res_1778_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_publicInput_elim___redArg(lean_object* v_t_1779_, lean_object* v_publicInput_1780_){
_start:
{
lean_object* v___x_1781_; 
v___x_1781_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_ctorElim___redArg(v_t_1779_, v_publicInput_1780_);
return v___x_1781_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_publicInput_elim___redArg___boxed(lean_object* v_t_1782_, lean_object* v_publicInput_1783_){
_start:
{
lean_object* v_res_1784_; 
v_res_1784_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_publicInput_elim___redArg(v_t_1782_, v_publicInput_1783_);
lean_dec(v_t_1782_);
return v_res_1784_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_publicInput_elim(lean_object* v_motive_1785_, lean_object* v_t_1786_, lean_object* v_h_1787_, lean_object* v_publicInput_1788_){
_start:
{
lean_object* v___x_1789_; 
v___x_1789_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_ctorElim___redArg(v_t_1786_, v_publicInput_1788_);
return v___x_1789_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_publicInput_elim___boxed(lean_object* v_motive_1790_, lean_object* v_t_1791_, lean_object* v_h_1792_, lean_object* v_publicInput_1793_){
_start:
{
lean_object* v_res_1794_; 
v_res_1794_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_publicInput_elim(v_motive_1790_, v_t_1791_, v_h_1792_, v_publicInput_1793_);
lean_dec(v_t_1791_);
return v_res_1794_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_challenge_elim___redArg(lean_object* v_t_1795_, lean_object* v_challenge_1796_){
_start:
{
lean_object* v___x_1797_; 
v___x_1797_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_ctorElim___redArg(v_t_1795_, v_challenge_1796_);
return v___x_1797_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_challenge_elim___redArg___boxed(lean_object* v_t_1798_, lean_object* v_challenge_1799_){
_start:
{
lean_object* v_res_1800_; 
v_res_1800_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_challenge_elim___redArg(v_t_1798_, v_challenge_1799_);
lean_dec(v_t_1798_);
return v_res_1800_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_challenge_elim(lean_object* v_motive_1801_, lean_object* v_t_1802_, lean_object* v_h_1803_, lean_object* v_challenge_1804_){
_start:
{
lean_object* v___x_1805_; 
v___x_1805_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_ctorElim___redArg(v_t_1802_, v_challenge_1804_);
return v___x_1805_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_challenge_elim___boxed(lean_object* v_motive_1806_, lean_object* v_t_1807_, lean_object* v_h_1808_, lean_object* v_challenge_1809_){
_start:
{
lean_object* v_res_1810_; 
v_res_1810_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawEntry_challenge_elim(v_motive_1806_, v_t_1807_, v_h_1808_, v_challenge_1809_);
lean_dec(v_t_1807_);
return v_res_1810_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr(lean_object* v_x_1833_, lean_object* v_prec_1834_){
_start:
{
lean_object* v___y_1836_; lean_object* v___y_1843_; 
switch(lean_obj_tag(v_x_1833_))
{
case 0:
{
uint32_t v_offset_1849_; lean_object* v___y_1851_; lean_object* v___x_1861_; uint8_t v___x_1862_; 
v_offset_1849_ = lean_ctor_get_uint32(v_x_1833_, 0);
v___x_1861_ = lean_unsigned_to_nat(1024u);
v___x_1862_ = lean_nat_dec_le(v___x_1861_, v_prec_1834_);
if (v___x_1862_ == 0)
{
lean_object* v___x_1863_; 
v___x_1863_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2);
v___y_1851_ = v___x_1863_;
goto v___jp_1850_;
}
else
{
lean_object* v___x_1864_; 
v___x_1864_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3);
v___y_1851_ = v___x_1864_;
goto v___jp_1850_;
}
v___jp_1850_:
{
lean_object* v___x_1852_; lean_object* v___x_1853_; lean_object* v___x_1854_; lean_object* v___x_1855_; lean_object* v___x_1856_; lean_object* v___x_1857_; uint8_t v___x_1858_; lean_object* v___x_1859_; lean_object* v___x_1860_; 
v___x_1852_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__6));
v___x_1853_ = lean_uint32_to_nat(v_offset_1849_);
v___x_1854_ = l_Nat_reprFast(v___x_1853_);
v___x_1855_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1855_, 0, v___x_1854_);
v___x_1856_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1856_, 0, v___x_1852_);
lean_ctor_set(v___x_1856_, 1, v___x_1855_);
lean_inc(v___y_1851_);
v___x_1857_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1857_, 0, v___y_1851_);
lean_ctor_set(v___x_1857_, 1, v___x_1856_);
v___x_1858_ = 0;
v___x_1859_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1859_, 0, v___x_1857_);
lean_ctor_set_uint8(v___x_1859_, sizeof(void*)*1, v___x_1858_);
v___x_1860_ = l_Repr_addAppParen(v___x_1859_, v_prec_1834_);
return v___x_1860_;
}
}
case 1:
{
uint32_t v_partIndex_1865_; uint32_t v_offset_1866_; lean_object* v___y_1868_; lean_object* v___x_1884_; uint8_t v___x_1885_; 
v_partIndex_1865_ = lean_ctor_get_uint32(v_x_1833_, 0);
v_offset_1866_ = lean_ctor_get_uint32(v_x_1833_, 4);
v___x_1884_ = lean_unsigned_to_nat(1024u);
v___x_1885_ = lean_nat_dec_le(v___x_1884_, v_prec_1834_);
if (v___x_1885_ == 0)
{
lean_object* v___x_1886_; 
v___x_1886_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2);
v___y_1868_ = v___x_1886_;
goto v___jp_1867_;
}
else
{
lean_object* v___x_1887_; 
v___x_1887_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3);
v___y_1868_ = v___x_1887_;
goto v___jp_1867_;
}
v___jp_1867_:
{
lean_object* v___x_1869_; lean_object* v___x_1870_; lean_object* v___x_1871_; lean_object* v___x_1872_; lean_object* v___x_1873_; lean_object* v___x_1874_; lean_object* v___x_1875_; lean_object* v___x_1876_; lean_object* v___x_1877_; lean_object* v___x_1878_; lean_object* v___x_1879_; lean_object* v___x_1880_; uint8_t v___x_1881_; lean_object* v___x_1882_; lean_object* v___x_1883_; 
v___x_1869_ = lean_box(1);
v___x_1870_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__9));
v___x_1871_ = lean_uint32_to_nat(v_partIndex_1865_);
v___x_1872_ = l_Nat_reprFast(v___x_1871_);
v___x_1873_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1873_, 0, v___x_1872_);
v___x_1874_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1874_, 0, v___x_1870_);
lean_ctor_set(v___x_1874_, 1, v___x_1873_);
v___x_1875_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1875_, 0, v___x_1874_);
lean_ctor_set(v___x_1875_, 1, v___x_1869_);
v___x_1876_ = lean_uint32_to_nat(v_offset_1866_);
v___x_1877_ = l_Nat_reprFast(v___x_1876_);
v___x_1878_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1878_, 0, v___x_1877_);
v___x_1879_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1879_, 0, v___x_1875_);
lean_ctor_set(v___x_1879_, 1, v___x_1878_);
lean_inc(v___y_1868_);
v___x_1880_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1880_, 0, v___y_1868_);
lean_ctor_set(v___x_1880_, 1, v___x_1879_);
v___x_1881_ = 0;
v___x_1882_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1882_, 0, v___x_1880_);
lean_ctor_set_uint8(v___x_1882_, sizeof(void*)*1, v___x_1881_);
v___x_1883_ = l_Repr_addAppParen(v___x_1882_, v_prec_1834_);
return v___x_1883_;
}
}
case 2:
{
lean_object* v___x_1888_; uint8_t v___x_1889_; 
v___x_1888_ = lean_unsigned_to_nat(1024u);
v___x_1889_ = lean_nat_dec_le(v___x_1888_, v_prec_1834_);
if (v___x_1889_ == 0)
{
lean_object* v___x_1890_; 
v___x_1890_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2);
v___y_1836_ = v___x_1890_;
goto v___jp_1835_;
}
else
{
lean_object* v___x_1891_; 
v___x_1891_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3);
v___y_1836_ = v___x_1891_;
goto v___jp_1835_;
}
}
default: 
{
lean_object* v___x_1892_; uint8_t v___x_1893_; 
v___x_1892_ = lean_unsigned_to_nat(1024u);
v___x_1893_ = lean_nat_dec_le(v___x_1892_, v_prec_1834_);
if (v___x_1893_ == 0)
{
lean_object* v___x_1894_; 
v___x_1894_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2);
v___y_1843_ = v___x_1894_;
goto v___jp_1842_;
}
else
{
lean_object* v___x_1895_; 
v___x_1895_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3);
v___y_1843_ = v___x_1895_;
goto v___jp_1842_;
}
}
}
v___jp_1835_:
{
lean_object* v___x_1837_; lean_object* v___x_1838_; uint8_t v___x_1839_; lean_object* v___x_1840_; lean_object* v___x_1841_; 
v___x_1837_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__1));
lean_inc(v___y_1836_);
v___x_1838_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1838_, 0, v___y_1836_);
lean_ctor_set(v___x_1838_, 1, v___x_1837_);
v___x_1839_ = 0;
v___x_1840_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1840_, 0, v___x_1838_);
lean_ctor_set_uint8(v___x_1840_, sizeof(void*)*1, v___x_1839_);
v___x_1841_ = l_Repr_addAppParen(v___x_1840_, v_prec_1834_);
return v___x_1841_;
}
v___jp_1842_:
{
lean_object* v___x_1844_; lean_object* v___x_1845_; uint8_t v___x_1846_; lean_object* v___x_1847_; lean_object* v___x_1848_; 
v___x_1844_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___closed__3));
lean_inc(v___y_1843_);
v___x_1845_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1845_, 0, v___y_1843_);
lean_ctor_set(v___x_1845_, 1, v___x_1844_);
v___x_1846_ = 0;
v___x_1847_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1847_, 0, v___x_1845_);
lean_ctor_set_uint8(v___x_1847_, sizeof(void*)*1, v___x_1846_);
v___x_1848_ = l_Repr_addAppParen(v___x_1847_, v_prec_1834_);
return v___x_1848_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr___boxed(lean_object* v_x_1896_, lean_object* v_prec_1897_){
_start:
{
lean_object* v_res_1898_; 
v_res_1898_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr(v_x_1896_, v_prec_1897_);
lean_dec(v_prec_1897_);
lean_dec(v_x_1896_);
return v_res_1898_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg(lean_object* v_x_1918_){
_start:
{
lean_object* v_entry_1919_; uint32_t v_index_1920_; lean_object* v___x_1921_; lean_object* v___x_1922_; lean_object* v___x_1923_; lean_object* v___x_1924_; lean_object* v___x_1925_; lean_object* v___x_1926_; uint8_t v___x_1927_; lean_object* v___x_1928_; lean_object* v___x_1929_; lean_object* v___x_1930_; lean_object* v___x_1931_; lean_object* v___x_1932_; lean_object* v___x_1933_; lean_object* v___x_1934_; lean_object* v___x_1935_; lean_object* v___x_1936_; lean_object* v___x_1937_; lean_object* v___x_1938_; lean_object* v___x_1939_; lean_object* v___x_1940_; lean_object* v___x_1941_; lean_object* v___x_1942_; lean_object* v___x_1943_; lean_object* v___x_1944_; lean_object* v___x_1945_; lean_object* v___x_1946_; lean_object* v___x_1947_; lean_object* v___x_1948_; lean_object* v___x_1949_; 
v_entry_1919_ = lean_ctor_get(v_x_1918_, 0);
v_index_1920_ = lean_ctor_get_uint32(v_x_1918_, sizeof(void*)*1);
v___x_1921_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__5));
v___x_1922_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___closed__3));
v___x_1923_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__4);
v___x_1924_ = lean_unsigned_to_nat(0u);
v___x_1925_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawEntry_repr(v_entry_1919_, v___x_1924_);
v___x_1926_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1926_, 0, v___x_1923_);
lean_ctor_set(v___x_1926_, 1, v___x_1925_);
v___x_1927_ = 0;
v___x_1928_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1928_, 0, v___x_1926_);
lean_ctor_set_uint8(v___x_1928_, sizeof(void*)*1, v___x_1927_);
v___x_1929_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1929_, 0, v___x_1922_);
lean_ctor_set(v___x_1929_, 1, v___x_1928_);
v___x_1930_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__2));
v___x_1931_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1931_, 0, v___x_1929_);
lean_ctor_set(v___x_1931_, 1, v___x_1930_);
v___x_1932_ = lean_box(1);
v___x_1933_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1933_, 0, v___x_1931_);
lean_ctor_set(v___x_1933_, 1, v___x_1932_);
v___x_1934_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___closed__5));
v___x_1935_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1935_, 0, v___x_1933_);
lean_ctor_set(v___x_1935_, 1, v___x_1934_);
v___x_1936_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1936_, 0, v___x_1935_);
lean_ctor_set(v___x_1936_, 1, v___x_1921_);
v___x_1937_ = lean_uint32_to_nat(v_index_1920_);
v___x_1938_ = l_Nat_reprFast(v___x_1937_);
v___x_1939_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1939_, 0, v___x_1938_);
v___x_1940_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1940_, 0, v___x_1923_);
lean_ctor_set(v___x_1940_, 1, v___x_1939_);
v___x_1941_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1941_, 0, v___x_1940_);
lean_ctor_set_uint8(v___x_1941_, sizeof(void*)*1, v___x_1927_);
v___x_1942_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1942_, 0, v___x_1936_);
lean_ctor_set(v___x_1942_, 1, v___x_1941_);
v___x_1943_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10);
v___x_1944_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__11));
v___x_1945_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1945_, 0, v___x_1944_);
lean_ctor_set(v___x_1945_, 1, v___x_1942_);
v___x_1946_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__12));
v___x_1947_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1947_, 0, v___x_1945_);
lean_ctor_set(v___x_1947_, 1, v___x_1946_);
v___x_1948_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1948_, 0, v___x_1943_);
lean_ctor_set(v___x_1948_, 1, v___x_1947_);
v___x_1949_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1949_, 0, v___x_1948_);
lean_ctor_set_uint8(v___x_1949_, sizeof(void*)*1, v___x_1927_);
return v___x_1949_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg___boxed(lean_object* v_x_1950_){
_start:
{
lean_object* v_res_1951_; 
v_res_1951_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg(v_x_1950_);
lean_dec_ref(v_x_1950_);
return v_res_1951_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr(lean_object* v_x_1952_, lean_object* v_prec_1953_){
_start:
{
lean_object* v___x_1954_; 
v___x_1954_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg(v_x_1952_);
return v___x_1954_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___boxed(lean_object* v_x_1955_, lean_object* v_prec_1956_){
_start:
{
lean_object* v_res_1957_; 
v_res_1957_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr(v_x_1955_, v_prec_1956_);
lean_dec(v_prec_1956_);
lean_dec_ref(v_x_1955_);
return v_res_1957_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorIdx(lean_object* v_x_1960_){
_start:
{
switch(lean_obj_tag(v_x_1960_))
{
case 0:
{
lean_object* v___x_1961_; 
v___x_1961_ = lean_unsigned_to_nat(0u);
return v___x_1961_;
}
case 1:
{
lean_object* v___x_1962_; 
v___x_1962_ = lean_unsigned_to_nat(1u);
return v___x_1962_;
}
case 2:
{
lean_object* v___x_1963_; 
v___x_1963_ = lean_unsigned_to_nat(2u);
return v___x_1963_;
}
case 3:
{
lean_object* v___x_1964_; 
v___x_1964_ = lean_unsigned_to_nat(3u);
return v___x_1964_;
}
case 4:
{
lean_object* v___x_1965_; 
v___x_1965_ = lean_unsigned_to_nat(4u);
return v___x_1965_;
}
case 5:
{
lean_object* v___x_1966_; 
v___x_1966_ = lean_unsigned_to_nat(5u);
return v___x_1966_;
}
case 6:
{
lean_object* v___x_1967_; 
v___x_1967_ = lean_unsigned_to_nat(6u);
return v___x_1967_;
}
case 7:
{
lean_object* v___x_1968_; 
v___x_1968_ = lean_unsigned_to_nat(7u);
return v___x_1968_;
}
default: 
{
lean_object* v___x_1969_; 
v___x_1969_ = lean_unsigned_to_nat(8u);
return v___x_1969_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorIdx___boxed(lean_object* v_x_1970_){
_start:
{
lean_object* v_res_1971_; 
v_res_1971_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorIdx(v_x_1970_);
lean_dec(v_x_1970_);
return v_res_1971_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim___redArg(lean_object* v_t_1972_, lean_object* v_k_1973_){
_start:
{
switch(lean_obj_tag(v_t_1972_))
{
case 0:
{
lean_object* v_v_1974_; lean_object* v___x_1975_; 
v_v_1974_ = lean_ctor_get(v_t_1972_, 0);
lean_inc_ref(v_v_1974_);
lean_dec_ref_known(v_t_1972_, 1);
v___x_1975_ = lean_apply_1(v_k_1973_, v_v_1974_);
return v___x_1975_;
}
case 4:
{
uint32_t v_c_1976_; lean_object* v___x_1977_; lean_object* v___x_1978_; 
v_c_1976_ = lean_ctor_get_uint32(v_t_1972_, 0);
lean_dec_ref_known(v_t_1972_, 0);
v___x_1977_ = lean_box_uint32(v_c_1976_);
v___x_1978_ = lean_apply_1(v_k_1973_, v___x_1977_);
return v___x_1978_;
}
case 5:
{
uint32_t v_leftIdx_1979_; uint32_t v_rightIdx_1980_; uint64_t v_degreeMultiple_1981_; lean_object* v___x_1982_; lean_object* v___x_1983_; lean_object* v___x_1984_; lean_object* v___x_1985_; 
v_leftIdx_1979_ = lean_ctor_get_uint32(v_t_1972_, 8);
v_rightIdx_1980_ = lean_ctor_get_uint32(v_t_1972_, 12);
v_degreeMultiple_1981_ = lean_ctor_get_uint64(v_t_1972_, 0);
lean_dec_ref_known(v_t_1972_, 0);
v___x_1982_ = lean_box_uint32(v_leftIdx_1979_);
v___x_1983_ = lean_box_uint32(v_rightIdx_1980_);
v___x_1984_ = lean_box_uint64(v_degreeMultiple_1981_);
v___x_1985_ = lean_apply_3(v_k_1973_, v___x_1982_, v___x_1983_, v___x_1984_);
return v___x_1985_;
}
case 6:
{
uint32_t v_leftIdx_1986_; uint32_t v_rightIdx_1987_; uint64_t v_degreeMultiple_1988_; lean_object* v___x_1989_; lean_object* v___x_1990_; lean_object* v___x_1991_; lean_object* v___x_1992_; 
v_leftIdx_1986_ = lean_ctor_get_uint32(v_t_1972_, 8);
v_rightIdx_1987_ = lean_ctor_get_uint32(v_t_1972_, 12);
v_degreeMultiple_1988_ = lean_ctor_get_uint64(v_t_1972_, 0);
lean_dec_ref_known(v_t_1972_, 0);
v___x_1989_ = lean_box_uint32(v_leftIdx_1986_);
v___x_1990_ = lean_box_uint32(v_rightIdx_1987_);
v___x_1991_ = lean_box_uint64(v_degreeMultiple_1988_);
v___x_1992_ = lean_apply_3(v_k_1973_, v___x_1989_, v___x_1990_, v___x_1991_);
return v___x_1992_;
}
case 7:
{
uint32_t v_idx_1993_; uint64_t v_degreeMultiple_1994_; lean_object* v___x_1995_; lean_object* v___x_1996_; lean_object* v___x_1997_; 
v_idx_1993_ = lean_ctor_get_uint32(v_t_1972_, 8);
v_degreeMultiple_1994_ = lean_ctor_get_uint64(v_t_1972_, 0);
lean_dec_ref_known(v_t_1972_, 0);
v___x_1995_ = lean_box_uint32(v_idx_1993_);
v___x_1996_ = lean_box_uint64(v_degreeMultiple_1994_);
v___x_1997_ = lean_apply_2(v_k_1973_, v___x_1995_, v___x_1996_);
return v___x_1997_;
}
case 8:
{
uint32_t v_leftIdx_1998_; uint32_t v_rightIdx_1999_; uint64_t v_degreeMultiple_2000_; lean_object* v___x_2001_; lean_object* v___x_2002_; lean_object* v___x_2003_; lean_object* v___x_2004_; 
v_leftIdx_1998_ = lean_ctor_get_uint32(v_t_1972_, 8);
v_rightIdx_1999_ = lean_ctor_get_uint32(v_t_1972_, 12);
v_degreeMultiple_2000_ = lean_ctor_get_uint64(v_t_1972_, 0);
lean_dec_ref_known(v_t_1972_, 0);
v___x_2001_ = lean_box_uint32(v_leftIdx_1998_);
v___x_2002_ = lean_box_uint32(v_rightIdx_1999_);
v___x_2003_ = lean_box_uint64(v_degreeMultiple_2000_);
v___x_2004_ = lean_apply_3(v_k_1973_, v___x_2001_, v___x_2002_, v___x_2003_);
return v___x_2004_;
}
default: 
{
lean_dec(v_t_1972_);
return v_k_1973_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim(lean_object* v_motive_2005_, lean_object* v_ctorIdx_2006_, lean_object* v_t_2007_, lean_object* v_h_2008_, lean_object* v_k_2009_){
_start:
{
lean_object* v___x_2010_; 
v___x_2010_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim___redArg(v_t_2007_, v_k_2009_);
return v___x_2010_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim___boxed(lean_object* v_motive_2011_, lean_object* v_ctorIdx_2012_, lean_object* v_t_2013_, lean_object* v_h_2014_, lean_object* v_k_2015_){
_start:
{
lean_object* v_res_2016_; 
v_res_2016_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim(v_motive_2011_, v_ctorIdx_2012_, v_t_2013_, v_h_2014_, v_k_2015_);
lean_dec(v_ctorIdx_2012_);
return v_res_2016_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_variable_elim___redArg(lean_object* v_t_2017_, lean_object* v_variable_2018_){
_start:
{
lean_object* v___x_2019_; 
v___x_2019_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim___redArg(v_t_2017_, v_variable_2018_);
return v___x_2019_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_variable_elim(lean_object* v_motive_2020_, lean_object* v_t_2021_, lean_object* v_h_2022_, lean_object* v_variable_2023_){
_start:
{
lean_object* v___x_2024_; 
v___x_2024_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim___redArg(v_t_2021_, v_variable_2023_);
return v___x_2024_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_isFirstRow_elim___redArg(lean_object* v_t_2025_, lean_object* v_isFirstRow_2026_){
_start:
{
lean_object* v___x_2027_; 
v___x_2027_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim___redArg(v_t_2025_, v_isFirstRow_2026_);
return v___x_2027_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_isFirstRow_elim(lean_object* v_motive_2028_, lean_object* v_t_2029_, lean_object* v_h_2030_, lean_object* v_isFirstRow_2031_){
_start:
{
lean_object* v___x_2032_; 
v___x_2032_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim___redArg(v_t_2029_, v_isFirstRow_2031_);
return v___x_2032_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_isLastRow_elim___redArg(lean_object* v_t_2033_, lean_object* v_isLastRow_2034_){
_start:
{
lean_object* v___x_2035_; 
v___x_2035_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim___redArg(v_t_2033_, v_isLastRow_2034_);
return v___x_2035_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_isLastRow_elim(lean_object* v_motive_2036_, lean_object* v_t_2037_, lean_object* v_h_2038_, lean_object* v_isLastRow_2039_){
_start:
{
lean_object* v___x_2040_; 
v___x_2040_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim___redArg(v_t_2037_, v_isLastRow_2039_);
return v___x_2040_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_isTransition_elim___redArg(lean_object* v_t_2041_, lean_object* v_isTransition_2042_){
_start:
{
lean_object* v___x_2043_; 
v___x_2043_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim___redArg(v_t_2041_, v_isTransition_2042_);
return v___x_2043_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_isTransition_elim(lean_object* v_motive_2044_, lean_object* v_t_2045_, lean_object* v_h_2046_, lean_object* v_isTransition_2047_){
_start:
{
lean_object* v___x_2048_; 
v___x_2048_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim___redArg(v_t_2045_, v_isTransition_2047_);
return v___x_2048_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_constant_elim___redArg(lean_object* v_t_2049_, lean_object* v_constant_2050_){
_start:
{
lean_object* v___x_2051_; 
v___x_2051_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim___redArg(v_t_2049_, v_constant_2050_);
return v___x_2051_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_constant_elim(lean_object* v_motive_2052_, lean_object* v_t_2053_, lean_object* v_h_2054_, lean_object* v_constant_2055_){
_start:
{
lean_object* v___x_2056_; 
v___x_2056_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim___redArg(v_t_2053_, v_constant_2055_);
return v___x_2056_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_add_elim___redArg(lean_object* v_t_2057_, lean_object* v_add_2058_){
_start:
{
lean_object* v___x_2059_; 
v___x_2059_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim___redArg(v_t_2057_, v_add_2058_);
return v___x_2059_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_add_elim(lean_object* v_motive_2060_, lean_object* v_t_2061_, lean_object* v_h_2062_, lean_object* v_add_2063_){
_start:
{
lean_object* v___x_2064_; 
v___x_2064_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim___redArg(v_t_2061_, v_add_2063_);
return v___x_2064_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_sub_elim___redArg(lean_object* v_t_2065_, lean_object* v_sub_2066_){
_start:
{
lean_object* v___x_2067_; 
v___x_2067_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim___redArg(v_t_2065_, v_sub_2066_);
return v___x_2067_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_sub_elim(lean_object* v_motive_2068_, lean_object* v_t_2069_, lean_object* v_h_2070_, lean_object* v_sub_2071_){
_start:
{
lean_object* v___x_2072_; 
v___x_2072_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim___redArg(v_t_2069_, v_sub_2071_);
return v___x_2072_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_neg_elim___redArg(lean_object* v_t_2073_, lean_object* v_neg_2074_){
_start:
{
lean_object* v___x_2075_; 
v___x_2075_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim___redArg(v_t_2073_, v_neg_2074_);
return v___x_2075_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_neg_elim(lean_object* v_motive_2076_, lean_object* v_t_2077_, lean_object* v_h_2078_, lean_object* v_neg_2079_){
_start:
{
lean_object* v___x_2080_; 
v___x_2080_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim___redArg(v_t_2077_, v_neg_2079_);
return v___x_2080_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_mul_elim___redArg(lean_object* v_t_2081_, lean_object* v_mul_2082_){
_start:
{
lean_object* v___x_2083_; 
v___x_2083_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim___redArg(v_t_2081_, v_mul_2082_);
return v___x_2083_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_mul_elim(lean_object* v_motive_2084_, lean_object* v_t_2085_, lean_object* v_h_2086_, lean_object* v_mul_2087_){
_start:
{
lean_object* v___x_2088_; 
v___x_2088_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawSymbolicExpressionNode_ctorElim___redArg(v_t_2085_, v_mul_2087_);
return v___x_2088_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr(lean_object* v_x_2138_, lean_object* v_prec_2139_){
_start:
{
lean_object* v___y_2141_; lean_object* v___y_2148_; lean_object* v___y_2155_; 
switch(lean_obj_tag(v_x_2138_))
{
case 0:
{
lean_object* v_v_2161_; lean_object* v___y_2163_; lean_object* v___x_2171_; uint8_t v___x_2172_; 
v_v_2161_ = lean_ctor_get(v_x_2138_, 0);
v___x_2171_ = lean_unsigned_to_nat(1024u);
v___x_2172_ = lean_nat_dec_le(v___x_2171_, v_prec_2139_);
if (v___x_2172_ == 0)
{
lean_object* v___x_2173_; 
v___x_2173_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2);
v___y_2163_ = v___x_2173_;
goto v___jp_2162_;
}
else
{
lean_object* v___x_2174_; 
v___x_2174_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3);
v___y_2163_ = v___x_2174_;
goto v___jp_2162_;
}
v___jp_2162_:
{
lean_object* v___x_2164_; lean_object* v___x_2165_; lean_object* v___x_2166_; lean_object* v___x_2167_; uint8_t v___x_2168_; lean_object* v___x_2169_; lean_object* v___x_2170_; 
v___x_2164_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__8));
v___x_2165_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicVariable_repr___redArg(v_v_2161_);
v___x_2166_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2166_, 0, v___x_2164_);
lean_ctor_set(v___x_2166_, 1, v___x_2165_);
lean_inc(v___y_2163_);
v___x_2167_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2167_, 0, v___y_2163_);
lean_ctor_set(v___x_2167_, 1, v___x_2166_);
v___x_2168_ = 0;
v___x_2169_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2169_, 0, v___x_2167_);
lean_ctor_set_uint8(v___x_2169_, sizeof(void*)*1, v___x_2168_);
v___x_2170_ = l_Repr_addAppParen(v___x_2169_, v_prec_2139_);
return v___x_2170_;
}
}
case 1:
{
lean_object* v___x_2175_; uint8_t v___x_2176_; 
v___x_2175_ = lean_unsigned_to_nat(1024u);
v___x_2176_ = lean_nat_dec_le(v___x_2175_, v_prec_2139_);
if (v___x_2176_ == 0)
{
lean_object* v___x_2177_; 
v___x_2177_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2);
v___y_2155_ = v___x_2177_;
goto v___jp_2154_;
}
else
{
lean_object* v___x_2178_; 
v___x_2178_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3);
v___y_2155_ = v___x_2178_;
goto v___jp_2154_;
}
}
case 2:
{
lean_object* v___x_2179_; uint8_t v___x_2180_; 
v___x_2179_ = lean_unsigned_to_nat(1024u);
v___x_2180_ = lean_nat_dec_le(v___x_2179_, v_prec_2139_);
if (v___x_2180_ == 0)
{
lean_object* v___x_2181_; 
v___x_2181_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2);
v___y_2148_ = v___x_2181_;
goto v___jp_2147_;
}
else
{
lean_object* v___x_2182_; 
v___x_2182_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3);
v___y_2148_ = v___x_2182_;
goto v___jp_2147_;
}
}
case 3:
{
lean_object* v___x_2183_; uint8_t v___x_2184_; 
v___x_2183_ = lean_unsigned_to_nat(1024u);
v___x_2184_ = lean_nat_dec_le(v___x_2183_, v_prec_2139_);
if (v___x_2184_ == 0)
{
lean_object* v___x_2185_; 
v___x_2185_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2);
v___y_2141_ = v___x_2185_;
goto v___jp_2140_;
}
else
{
lean_object* v___x_2186_; 
v___x_2186_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3);
v___y_2141_ = v___x_2186_;
goto v___jp_2140_;
}
}
case 4:
{
uint32_t v_c_2187_; lean_object* v___y_2189_; lean_object* v___x_2199_; uint8_t v___x_2200_; 
v_c_2187_ = lean_ctor_get_uint32(v_x_2138_, 0);
v___x_2199_ = lean_unsigned_to_nat(1024u);
v___x_2200_ = lean_nat_dec_le(v___x_2199_, v_prec_2139_);
if (v___x_2200_ == 0)
{
lean_object* v___x_2201_; 
v___x_2201_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2);
v___y_2189_ = v___x_2201_;
goto v___jp_2188_;
}
else
{
lean_object* v___x_2202_; 
v___x_2202_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3);
v___y_2189_ = v___x_2202_;
goto v___jp_2188_;
}
v___jp_2188_:
{
lean_object* v___x_2190_; lean_object* v___x_2191_; lean_object* v___x_2192_; lean_object* v___x_2193_; lean_object* v___x_2194_; lean_object* v___x_2195_; uint8_t v___x_2196_; lean_object* v___x_2197_; lean_object* v___x_2198_; 
v___x_2190_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__11));
v___x_2191_ = lean_uint32_to_nat(v_c_2187_);
v___x_2192_ = l_Nat_reprFast(v___x_2191_);
v___x_2193_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2193_, 0, v___x_2192_);
v___x_2194_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2194_, 0, v___x_2190_);
lean_ctor_set(v___x_2194_, 1, v___x_2193_);
lean_inc(v___y_2189_);
v___x_2195_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2195_, 0, v___y_2189_);
lean_ctor_set(v___x_2195_, 1, v___x_2194_);
v___x_2196_ = 0;
v___x_2197_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2197_, 0, v___x_2195_);
lean_ctor_set_uint8(v___x_2197_, sizeof(void*)*1, v___x_2196_);
v___x_2198_ = l_Repr_addAppParen(v___x_2197_, v_prec_2139_);
return v___x_2198_;
}
}
case 5:
{
uint32_t v_leftIdx_2203_; uint32_t v_rightIdx_2204_; uint64_t v_degreeMultiple_2205_; lean_object* v___y_2207_; lean_object* v___x_2228_; uint8_t v___x_2229_; 
v_leftIdx_2203_ = lean_ctor_get_uint32(v_x_2138_, 8);
v_rightIdx_2204_ = lean_ctor_get_uint32(v_x_2138_, 12);
v_degreeMultiple_2205_ = lean_ctor_get_uint64(v_x_2138_, 0);
v___x_2228_ = lean_unsigned_to_nat(1024u);
v___x_2229_ = lean_nat_dec_le(v___x_2228_, v_prec_2139_);
if (v___x_2229_ == 0)
{
lean_object* v___x_2230_; 
v___x_2230_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2);
v___y_2207_ = v___x_2230_;
goto v___jp_2206_;
}
else
{
lean_object* v___x_2231_; 
v___x_2231_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3);
v___y_2207_ = v___x_2231_;
goto v___jp_2206_;
}
v___jp_2206_:
{
lean_object* v___x_2208_; lean_object* v___x_2209_; lean_object* v___x_2210_; lean_object* v___x_2211_; lean_object* v___x_2212_; lean_object* v___x_2213_; lean_object* v___x_2214_; lean_object* v___x_2215_; lean_object* v___x_2216_; lean_object* v___x_2217_; lean_object* v___x_2218_; lean_object* v___x_2219_; lean_object* v___x_2220_; lean_object* v___x_2221_; lean_object* v___x_2222_; lean_object* v___x_2223_; lean_object* v___x_2224_; uint8_t v___x_2225_; lean_object* v___x_2226_; lean_object* v___x_2227_; 
v___x_2208_ = lean_box(1);
v___x_2209_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__14));
v___x_2210_ = lean_uint32_to_nat(v_leftIdx_2203_);
v___x_2211_ = l_Nat_reprFast(v___x_2210_);
v___x_2212_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2212_, 0, v___x_2211_);
v___x_2213_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2213_, 0, v___x_2209_);
lean_ctor_set(v___x_2213_, 1, v___x_2212_);
v___x_2214_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2214_, 0, v___x_2213_);
lean_ctor_set(v___x_2214_, 1, v___x_2208_);
v___x_2215_ = lean_uint32_to_nat(v_rightIdx_2204_);
v___x_2216_ = l_Nat_reprFast(v___x_2215_);
v___x_2217_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2217_, 0, v___x_2216_);
v___x_2218_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2218_, 0, v___x_2214_);
lean_ctor_set(v___x_2218_, 1, v___x_2217_);
v___x_2219_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2219_, 0, v___x_2218_);
lean_ctor_set(v___x_2219_, 1, v___x_2208_);
v___x_2220_ = lean_uint64_to_nat(v_degreeMultiple_2205_);
v___x_2221_ = l_Nat_reprFast(v___x_2220_);
v___x_2222_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2222_, 0, v___x_2221_);
v___x_2223_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2223_, 0, v___x_2219_);
lean_ctor_set(v___x_2223_, 1, v___x_2222_);
lean_inc(v___y_2207_);
v___x_2224_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2224_, 0, v___y_2207_);
lean_ctor_set(v___x_2224_, 1, v___x_2223_);
v___x_2225_ = 0;
v___x_2226_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2226_, 0, v___x_2224_);
lean_ctor_set_uint8(v___x_2226_, sizeof(void*)*1, v___x_2225_);
v___x_2227_ = l_Repr_addAppParen(v___x_2226_, v_prec_2139_);
return v___x_2227_;
}
}
case 6:
{
uint32_t v_leftIdx_2232_; uint32_t v_rightIdx_2233_; uint64_t v_degreeMultiple_2234_; lean_object* v___y_2236_; lean_object* v___x_2257_; uint8_t v___x_2258_; 
v_leftIdx_2232_ = lean_ctor_get_uint32(v_x_2138_, 8);
v_rightIdx_2233_ = lean_ctor_get_uint32(v_x_2138_, 12);
v_degreeMultiple_2234_ = lean_ctor_get_uint64(v_x_2138_, 0);
v___x_2257_ = lean_unsigned_to_nat(1024u);
v___x_2258_ = lean_nat_dec_le(v___x_2257_, v_prec_2139_);
if (v___x_2258_ == 0)
{
lean_object* v___x_2259_; 
v___x_2259_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2);
v___y_2236_ = v___x_2259_;
goto v___jp_2235_;
}
else
{
lean_object* v___x_2260_; 
v___x_2260_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3);
v___y_2236_ = v___x_2260_;
goto v___jp_2235_;
}
v___jp_2235_:
{
lean_object* v___x_2237_; lean_object* v___x_2238_; lean_object* v___x_2239_; lean_object* v___x_2240_; lean_object* v___x_2241_; lean_object* v___x_2242_; lean_object* v___x_2243_; lean_object* v___x_2244_; lean_object* v___x_2245_; lean_object* v___x_2246_; lean_object* v___x_2247_; lean_object* v___x_2248_; lean_object* v___x_2249_; lean_object* v___x_2250_; lean_object* v___x_2251_; lean_object* v___x_2252_; lean_object* v___x_2253_; uint8_t v___x_2254_; lean_object* v___x_2255_; lean_object* v___x_2256_; 
v___x_2237_ = lean_box(1);
v___x_2238_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__17));
v___x_2239_ = lean_uint32_to_nat(v_leftIdx_2232_);
v___x_2240_ = l_Nat_reprFast(v___x_2239_);
v___x_2241_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2241_, 0, v___x_2240_);
v___x_2242_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2242_, 0, v___x_2238_);
lean_ctor_set(v___x_2242_, 1, v___x_2241_);
v___x_2243_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2243_, 0, v___x_2242_);
lean_ctor_set(v___x_2243_, 1, v___x_2237_);
v___x_2244_ = lean_uint32_to_nat(v_rightIdx_2233_);
v___x_2245_ = l_Nat_reprFast(v___x_2244_);
v___x_2246_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2246_, 0, v___x_2245_);
v___x_2247_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2247_, 0, v___x_2243_);
lean_ctor_set(v___x_2247_, 1, v___x_2246_);
v___x_2248_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2248_, 0, v___x_2247_);
lean_ctor_set(v___x_2248_, 1, v___x_2237_);
v___x_2249_ = lean_uint64_to_nat(v_degreeMultiple_2234_);
v___x_2250_ = l_Nat_reprFast(v___x_2249_);
v___x_2251_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2251_, 0, v___x_2250_);
v___x_2252_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2252_, 0, v___x_2248_);
lean_ctor_set(v___x_2252_, 1, v___x_2251_);
lean_inc(v___y_2236_);
v___x_2253_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2253_, 0, v___y_2236_);
lean_ctor_set(v___x_2253_, 1, v___x_2252_);
v___x_2254_ = 0;
v___x_2255_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2255_, 0, v___x_2253_);
lean_ctor_set_uint8(v___x_2255_, sizeof(void*)*1, v___x_2254_);
v___x_2256_ = l_Repr_addAppParen(v___x_2255_, v_prec_2139_);
return v___x_2256_;
}
}
case 7:
{
uint32_t v_idx_2261_; uint64_t v_degreeMultiple_2262_; lean_object* v___y_2264_; lean_object* v___x_2280_; uint8_t v___x_2281_; 
v_idx_2261_ = lean_ctor_get_uint32(v_x_2138_, 8);
v_degreeMultiple_2262_ = lean_ctor_get_uint64(v_x_2138_, 0);
v___x_2280_ = lean_unsigned_to_nat(1024u);
v___x_2281_ = lean_nat_dec_le(v___x_2280_, v_prec_2139_);
if (v___x_2281_ == 0)
{
lean_object* v___x_2282_; 
v___x_2282_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2);
v___y_2264_ = v___x_2282_;
goto v___jp_2263_;
}
else
{
lean_object* v___x_2283_; 
v___x_2283_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3);
v___y_2264_ = v___x_2283_;
goto v___jp_2263_;
}
v___jp_2263_:
{
lean_object* v___x_2265_; lean_object* v___x_2266_; lean_object* v___x_2267_; lean_object* v___x_2268_; lean_object* v___x_2269_; lean_object* v___x_2270_; lean_object* v___x_2271_; lean_object* v___x_2272_; lean_object* v___x_2273_; lean_object* v___x_2274_; lean_object* v___x_2275_; lean_object* v___x_2276_; uint8_t v___x_2277_; lean_object* v___x_2278_; lean_object* v___x_2279_; 
v___x_2265_ = lean_box(1);
v___x_2266_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__20));
v___x_2267_ = lean_uint32_to_nat(v_idx_2261_);
v___x_2268_ = l_Nat_reprFast(v___x_2267_);
v___x_2269_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2269_, 0, v___x_2268_);
v___x_2270_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2270_, 0, v___x_2266_);
lean_ctor_set(v___x_2270_, 1, v___x_2269_);
v___x_2271_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2271_, 0, v___x_2270_);
lean_ctor_set(v___x_2271_, 1, v___x_2265_);
v___x_2272_ = lean_uint64_to_nat(v_degreeMultiple_2262_);
v___x_2273_ = l_Nat_reprFast(v___x_2272_);
v___x_2274_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2274_, 0, v___x_2273_);
v___x_2275_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2275_, 0, v___x_2271_);
lean_ctor_set(v___x_2275_, 1, v___x_2274_);
lean_inc(v___y_2264_);
v___x_2276_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2276_, 0, v___y_2264_);
lean_ctor_set(v___x_2276_, 1, v___x_2275_);
v___x_2277_ = 0;
v___x_2278_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2278_, 0, v___x_2276_);
lean_ctor_set_uint8(v___x_2278_, sizeof(void*)*1, v___x_2277_);
v___x_2279_ = l_Repr_addAppParen(v___x_2278_, v_prec_2139_);
return v___x_2279_;
}
}
default: 
{
uint32_t v_leftIdx_2284_; uint32_t v_rightIdx_2285_; uint64_t v_degreeMultiple_2286_; lean_object* v___y_2288_; lean_object* v___x_2309_; uint8_t v___x_2310_; 
v_leftIdx_2284_ = lean_ctor_get_uint32(v_x_2138_, 8);
v_rightIdx_2285_ = lean_ctor_get_uint32(v_x_2138_, 12);
v_degreeMultiple_2286_ = lean_ctor_get_uint64(v_x_2138_, 0);
v___x_2309_ = lean_unsigned_to_nat(1024u);
v___x_2310_ = lean_nat_dec_le(v___x_2309_, v_prec_2139_);
if (v___x_2310_ == 0)
{
lean_object* v___x_2311_; 
v___x_2311_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__2);
v___y_2288_ = v___x_2311_;
goto v___jp_2287_;
}
else
{
lean_object* v___x_2312_; 
v___x_2312_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirProximityStrategy_repr___closed__3);
v___y_2288_ = v___x_2312_;
goto v___jp_2287_;
}
v___jp_2287_:
{
lean_object* v___x_2289_; lean_object* v___x_2290_; lean_object* v___x_2291_; lean_object* v___x_2292_; lean_object* v___x_2293_; lean_object* v___x_2294_; lean_object* v___x_2295_; lean_object* v___x_2296_; lean_object* v___x_2297_; lean_object* v___x_2298_; lean_object* v___x_2299_; lean_object* v___x_2300_; lean_object* v___x_2301_; lean_object* v___x_2302_; lean_object* v___x_2303_; lean_object* v___x_2304_; lean_object* v___x_2305_; uint8_t v___x_2306_; lean_object* v___x_2307_; lean_object* v___x_2308_; 
v___x_2289_ = lean_box(1);
v___x_2290_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__23));
v___x_2291_ = lean_uint32_to_nat(v_leftIdx_2284_);
v___x_2292_ = l_Nat_reprFast(v___x_2291_);
v___x_2293_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2293_, 0, v___x_2292_);
v___x_2294_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2294_, 0, v___x_2290_);
lean_ctor_set(v___x_2294_, 1, v___x_2293_);
v___x_2295_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2295_, 0, v___x_2294_);
lean_ctor_set(v___x_2295_, 1, v___x_2289_);
v___x_2296_ = lean_uint32_to_nat(v_rightIdx_2285_);
v___x_2297_ = l_Nat_reprFast(v___x_2296_);
v___x_2298_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2298_, 0, v___x_2297_);
v___x_2299_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2299_, 0, v___x_2295_);
lean_ctor_set(v___x_2299_, 1, v___x_2298_);
v___x_2300_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2300_, 0, v___x_2299_);
lean_ctor_set(v___x_2300_, 1, v___x_2289_);
v___x_2301_ = lean_uint64_to_nat(v_degreeMultiple_2286_);
v___x_2302_ = l_Nat_reprFast(v___x_2301_);
v___x_2303_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2303_, 0, v___x_2302_);
v___x_2304_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2304_, 0, v___x_2300_);
lean_ctor_set(v___x_2304_, 1, v___x_2303_);
lean_inc(v___y_2288_);
v___x_2305_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2305_, 0, v___y_2288_);
lean_ctor_set(v___x_2305_, 1, v___x_2304_);
v___x_2306_ = 0;
v___x_2307_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2307_, 0, v___x_2305_);
lean_ctor_set_uint8(v___x_2307_, sizeof(void*)*1, v___x_2306_);
v___x_2308_ = l_Repr_addAppParen(v___x_2307_, v_prec_2139_);
return v___x_2308_;
}
}
}
v___jp_2140_:
{
lean_object* v___x_2142_; lean_object* v___x_2143_; uint8_t v___x_2144_; lean_object* v___x_2145_; lean_object* v___x_2146_; 
v___x_2142_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__1));
lean_inc(v___y_2141_);
v___x_2143_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2143_, 0, v___y_2141_);
lean_ctor_set(v___x_2143_, 1, v___x_2142_);
v___x_2144_ = 0;
v___x_2145_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2145_, 0, v___x_2143_);
lean_ctor_set_uint8(v___x_2145_, sizeof(void*)*1, v___x_2144_);
v___x_2146_ = l_Repr_addAppParen(v___x_2145_, v_prec_2139_);
return v___x_2146_;
}
v___jp_2147_:
{
lean_object* v___x_2149_; lean_object* v___x_2150_; uint8_t v___x_2151_; lean_object* v___x_2152_; lean_object* v___x_2153_; 
v___x_2149_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__3));
lean_inc(v___y_2148_);
v___x_2150_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2150_, 0, v___y_2148_);
lean_ctor_set(v___x_2150_, 1, v___x_2149_);
v___x_2151_ = 0;
v___x_2152_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2152_, 0, v___x_2150_);
lean_ctor_set_uint8(v___x_2152_, sizeof(void*)*1, v___x_2151_);
v___x_2153_ = l_Repr_addAppParen(v___x_2152_, v_prec_2139_);
return v___x_2153_;
}
v___jp_2154_:
{
lean_object* v___x_2156_; lean_object* v___x_2157_; uint8_t v___x_2158_; lean_object* v___x_2159_; lean_object* v___x_2160_; 
v___x_2156_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___closed__5));
lean_inc(v___y_2155_);
v___x_2157_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2157_, 0, v___y_2155_);
lean_ctor_set(v___x_2157_, 1, v___x_2156_);
v___x_2158_ = 0;
v___x_2159_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2159_, 0, v___x_2157_);
lean_ctor_set_uint8(v___x_2159_, sizeof(void*)*1, v___x_2158_);
v___x_2160_ = l_Repr_addAppParen(v___x_2159_, v_prec_2139_);
return v___x_2160_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr___boxed(lean_object* v_x_2313_, lean_object* v_prec_2314_){
_start:
{
lean_object* v_res_2315_; 
v_res_2315_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr(v_x_2313_, v_prec_2314_);
lean_dec(v_prec_2314_);
lean_dec(v_x_2313_);
return v_res_2315_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr_spec__0_spec__0___lam__0(lean_object* v___y_2324_){
_start:
{
lean_object* v___x_2325_; lean_object* v___x_2326_; 
v___x_2325_ = lean_unsigned_to_nat(0u);
v___x_2326_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr(v___y_2324_, v___x_2325_);
return v___x_2326_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr_spec__0_spec__0___lam__0___boxed(lean_object* v___y_2327_){
_start:
{
lean_object* v_res_2328_; 
v_res_2328_ = lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr_spec__0_spec__0___lam__0(v___y_2327_);
lean_dec(v___y_2327_);
return v_res_2328_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr_spec__0_spec__0_spec__1_spec__2(lean_object* v_x_2329_, lean_object* v_x_2330_, lean_object* v_x_2331_){
_start:
{
if (lean_obj_tag(v_x_2331_) == 0)
{
lean_dec(v_x_2329_);
return v_x_2330_;
}
else
{
lean_object* v_head_2332_; lean_object* v_tail_2333_; lean_object* v___x_2335_; uint8_t v_isShared_2336_; uint8_t v_isSharedCheck_2344_; 
v_head_2332_ = lean_ctor_get(v_x_2331_, 0);
v_tail_2333_ = lean_ctor_get(v_x_2331_, 1);
v_isSharedCheck_2344_ = !lean_is_exclusive(v_x_2331_);
if (v_isSharedCheck_2344_ == 0)
{
v___x_2335_ = v_x_2331_;
v_isShared_2336_ = v_isSharedCheck_2344_;
goto v_resetjp_2334_;
}
else
{
lean_inc(v_tail_2333_);
lean_inc(v_head_2332_);
lean_dec(v_x_2331_);
v___x_2335_ = lean_box(0);
v_isShared_2336_ = v_isSharedCheck_2344_;
goto v_resetjp_2334_;
}
v_resetjp_2334_:
{
lean_object* v___x_2338_; 
lean_inc(v_x_2329_);
if (v_isShared_2336_ == 0)
{
lean_ctor_set_tag(v___x_2335_, 5);
lean_ctor_set(v___x_2335_, 1, v_x_2329_);
lean_ctor_set(v___x_2335_, 0, v_x_2330_);
v___x_2338_ = v___x_2335_;
goto v_reusejp_2337_;
}
else
{
lean_object* v_reuseFailAlloc_2343_; 
v_reuseFailAlloc_2343_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2343_, 0, v_x_2330_);
lean_ctor_set(v_reuseFailAlloc_2343_, 1, v_x_2329_);
v___x_2338_ = v_reuseFailAlloc_2343_;
goto v_reusejp_2337_;
}
v_reusejp_2337_:
{
lean_object* v___x_2339_; lean_object* v___x_2340_; lean_object* v___x_2341_; 
v___x_2339_ = lean_unsigned_to_nat(0u);
v___x_2340_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr(v_head_2332_, v___x_2339_);
lean_dec(v_head_2332_);
v___x_2341_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2341_, 0, v___x_2338_);
lean_ctor_set(v___x_2341_, 1, v___x_2340_);
v_x_2330_ = v___x_2341_;
v_x_2331_ = v_tail_2333_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr_spec__0_spec__0_spec__1(lean_object* v_x_2345_, lean_object* v_x_2346_, lean_object* v_x_2347_){
_start:
{
if (lean_obj_tag(v_x_2347_) == 0)
{
lean_dec(v_x_2345_);
return v_x_2346_;
}
else
{
lean_object* v_head_2348_; lean_object* v_tail_2349_; lean_object* v___x_2351_; uint8_t v_isShared_2352_; uint8_t v_isSharedCheck_2360_; 
v_head_2348_ = lean_ctor_get(v_x_2347_, 0);
v_tail_2349_ = lean_ctor_get(v_x_2347_, 1);
v_isSharedCheck_2360_ = !lean_is_exclusive(v_x_2347_);
if (v_isSharedCheck_2360_ == 0)
{
v___x_2351_ = v_x_2347_;
v_isShared_2352_ = v_isSharedCheck_2360_;
goto v_resetjp_2350_;
}
else
{
lean_inc(v_tail_2349_);
lean_inc(v_head_2348_);
lean_dec(v_x_2347_);
v___x_2351_ = lean_box(0);
v_isShared_2352_ = v_isSharedCheck_2360_;
goto v_resetjp_2350_;
}
v_resetjp_2350_:
{
lean_object* v___x_2354_; 
lean_inc(v_x_2345_);
if (v_isShared_2352_ == 0)
{
lean_ctor_set_tag(v___x_2351_, 5);
lean_ctor_set(v___x_2351_, 1, v_x_2345_);
lean_ctor_set(v___x_2351_, 0, v_x_2346_);
v___x_2354_ = v___x_2351_;
goto v_reusejp_2353_;
}
else
{
lean_object* v_reuseFailAlloc_2359_; 
v_reuseFailAlloc_2359_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2359_, 0, v_x_2346_);
lean_ctor_set(v_reuseFailAlloc_2359_, 1, v_x_2345_);
v___x_2354_ = v_reuseFailAlloc_2359_;
goto v_reusejp_2353_;
}
v_reusejp_2353_:
{
lean_object* v___x_2355_; lean_object* v___x_2356_; lean_object* v___x_2357_; lean_object* v___x_2358_; 
v___x_2355_ = lean_unsigned_to_nat(0u);
v___x_2356_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionNode_repr(v_head_2348_, v___x_2355_);
lean_dec(v_head_2348_);
v___x_2357_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2357_, 0, v___x_2354_);
lean_ctor_set(v___x_2357_, 1, v___x_2356_);
v___x_2358_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr_spec__0_spec__0_spec__1_spec__2(v_x_2345_, v___x_2357_, v_tail_2349_);
return v___x_2358_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr_spec__0_spec__0(lean_object* v_x_2361_, lean_object* v_x_2362_){
_start:
{
if (lean_obj_tag(v_x_2361_) == 0)
{
lean_object* v___x_2363_; 
lean_dec(v_x_2362_);
v___x_2363_ = lean_box(0);
return v___x_2363_;
}
else
{
lean_object* v_tail_2364_; 
v_tail_2364_ = lean_ctor_get(v_x_2361_, 1);
if (lean_obj_tag(v_tail_2364_) == 0)
{
lean_object* v_head_2365_; lean_object* v___x_2366_; 
lean_dec(v_x_2362_);
v_head_2365_ = lean_ctor_get(v_x_2361_, 0);
lean_inc(v_head_2365_);
lean_dec_ref_known(v_x_2361_, 2);
v___x_2366_ = lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr_spec__0_spec__0___lam__0(v_head_2365_);
lean_dec(v_head_2365_);
return v___x_2366_;
}
else
{
lean_object* v_head_2367_; lean_object* v___x_2368_; lean_object* v___x_2369_; 
lean_inc(v_tail_2364_);
v_head_2367_ = lean_ctor_get(v_x_2361_, 0);
lean_inc(v_head_2367_);
lean_dec_ref_known(v_x_2361_, 2);
v___x_2368_ = lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr_spec__0_spec__0___lam__0(v_head_2367_);
lean_dec(v_head_2367_);
v___x_2369_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr_spec__0_spec__0_spec__1(v_x_2362_, v___x_2368_, v_tail_2364_);
return v___x_2369_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr_spec__0(lean_object* v_xs_2370_){
_start:
{
lean_object* v___x_2371_; lean_object* v___x_2372_; uint8_t v___x_2373_; 
v___x_2371_ = lean_array_get_size(v_xs_2370_);
v___x_2372_ = lean_unsigned_to_nat(0u);
v___x_2373_ = lean_nat_dec_eq(v___x_2371_, v___x_2372_);
if (v___x_2373_ == 0)
{
lean_object* v___x_2374_; lean_object* v___x_2375_; lean_object* v___x_2376_; lean_object* v___x_2377_; lean_object* v___x_2378_; lean_object* v___x_2379_; lean_object* v___x_2380_; lean_object* v___x_2381_; lean_object* v___x_2382_; lean_object* v___x_2383_; 
v___x_2374_ = lean_array_to_list(v_xs_2370_);
v___x_2375_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__3));
v___x_2376_ = lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr_spec__0_spec__0(v___x_2374_, v___x_2375_);
v___x_2377_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__6, &lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__6_once, _init_lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__6);
v___x_2378_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__7));
v___x_2379_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2379_, 0, v___x_2378_);
lean_ctor_set(v___x_2379_, 1, v___x_2376_);
v___x_2380_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__8));
v___x_2381_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2381_, 0, v___x_2379_);
lean_ctor_set(v___x_2381_, 1, v___x_2380_);
v___x_2382_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2382_, 0, v___x_2377_);
lean_ctor_set(v___x_2382_, 1, v___x_2381_);
v___x_2383_ = l_Std_Format_fill(v___x_2382_);
return v___x_2383_;
}
else
{
lean_object* v___x_2384_; 
lean_dec_ref(v_xs_2370_);
v___x_2384_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__10));
return v___x_2384_;
}
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__6(void){
_start:
{
lean_object* v___x_2397_; lean_object* v___x_2398_; 
v___x_2397_ = lean_unsigned_to_nat(17u);
v___x_2398_ = lean_nat_to_int(v___x_2397_);
return v___x_2398_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg(lean_object* v_x_2399_){
_start:
{
lean_object* v_nodes_2400_; lean_object* v_constraintIdx_2401_; lean_object* v___x_2403_; uint8_t v_isShared_2404_; uint8_t v_isSharedCheck_2434_; 
v_nodes_2400_ = lean_ctor_get(v_x_2399_, 0);
v_constraintIdx_2401_ = lean_ctor_get(v_x_2399_, 1);
v_isSharedCheck_2434_ = !lean_is_exclusive(v_x_2399_);
if (v_isSharedCheck_2434_ == 0)
{
v___x_2403_ = v_x_2399_;
v_isShared_2404_ = v_isSharedCheck_2434_;
goto v_resetjp_2402_;
}
else
{
lean_inc(v_constraintIdx_2401_);
lean_inc(v_nodes_2400_);
lean_dec(v_x_2399_);
v___x_2403_ = lean_box(0);
v_isShared_2404_ = v_isSharedCheck_2434_;
goto v_resetjp_2402_;
}
v_resetjp_2402_:
{
lean_object* v___x_2405_; lean_object* v___x_2406_; lean_object* v___x_2407_; lean_object* v___x_2408_; lean_object* v___x_2410_; 
v___x_2405_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__5));
v___x_2406_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__3));
v___x_2407_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__4);
v___x_2408_ = lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr_spec__0(v_nodes_2400_);
if (v_isShared_2404_ == 0)
{
lean_ctor_set_tag(v___x_2403_, 4);
lean_ctor_set(v___x_2403_, 1, v___x_2408_);
lean_ctor_set(v___x_2403_, 0, v___x_2407_);
v___x_2410_ = v___x_2403_;
goto v_reusejp_2409_;
}
else
{
lean_object* v_reuseFailAlloc_2433_; 
v_reuseFailAlloc_2433_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2433_, 0, v___x_2407_);
lean_ctor_set(v_reuseFailAlloc_2433_, 1, v___x_2408_);
v___x_2410_ = v_reuseFailAlloc_2433_;
goto v_reusejp_2409_;
}
v_reusejp_2409_:
{
uint8_t v___x_2411_; lean_object* v___x_2412_; lean_object* v___x_2413_; lean_object* v___x_2414_; lean_object* v___x_2415_; lean_object* v___x_2416_; lean_object* v___x_2417_; lean_object* v___x_2418_; lean_object* v___x_2419_; lean_object* v___x_2420_; lean_object* v___x_2421_; lean_object* v___x_2422_; lean_object* v___x_2423_; lean_object* v___x_2424_; lean_object* v___x_2425_; lean_object* v___x_2426_; lean_object* v___x_2427_; lean_object* v___x_2428_; lean_object* v___x_2429_; lean_object* v___x_2430_; lean_object* v___x_2431_; lean_object* v___x_2432_; 
v___x_2411_ = 0;
v___x_2412_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2412_, 0, v___x_2410_);
lean_ctor_set_uint8(v___x_2412_, sizeof(void*)*1, v___x_2411_);
v___x_2413_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2413_, 0, v___x_2406_);
lean_ctor_set(v___x_2413_, 1, v___x_2412_);
v___x_2414_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__2));
v___x_2415_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2415_, 0, v___x_2413_);
lean_ctor_set(v___x_2415_, 1, v___x_2414_);
v___x_2416_ = lean_box(1);
v___x_2417_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2417_, 0, v___x_2415_);
lean_ctor_set(v___x_2417_, 1, v___x_2416_);
v___x_2418_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__5));
v___x_2419_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2419_, 0, v___x_2417_);
lean_ctor_set(v___x_2419_, 1, v___x_2418_);
v___x_2420_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2420_, 0, v___x_2419_);
lean_ctor_set(v___x_2420_, 1, v___x_2405_);
v___x_2421_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__6, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__6_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg___closed__6);
v___x_2422_ = lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__1(v_constraintIdx_2401_);
v___x_2423_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2423_, 0, v___x_2421_);
lean_ctor_set(v___x_2423_, 1, v___x_2422_);
v___x_2424_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2424_, 0, v___x_2423_);
lean_ctor_set_uint8(v___x_2424_, sizeof(void*)*1, v___x_2411_);
v___x_2425_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2425_, 0, v___x_2420_);
lean_ctor_set(v___x_2425_, 1, v___x_2424_);
v___x_2426_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10);
v___x_2427_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__11));
v___x_2428_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2428_, 0, v___x_2427_);
lean_ctor_set(v___x_2428_, 1, v___x_2425_);
v___x_2429_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__12));
v___x_2430_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2430_, 0, v___x_2428_);
lean_ctor_set(v___x_2430_, 1, v___x_2429_);
v___x_2431_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2431_, 0, v___x_2426_);
lean_ctor_set(v___x_2431_, 1, v___x_2430_);
v___x_2432_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2432_, 0, v___x_2431_);
lean_ctor_set_uint8(v___x_2432_, sizeof(void*)*1, v___x_2411_);
return v___x_2432_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr(lean_object* v_x_2435_, lean_object* v_prec_2436_){
_start:
{
lean_object* v___x_2437_; 
v___x_2437_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg(v_x_2435_);
return v___x_2437_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___boxed(lean_object* v_x_2438_, lean_object* v_prec_2439_){
_start:
{
lean_object* v_res_2440_; 
v_res_2440_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr(v_x_2438_, v_prec_2439_);
lean_dec(v_prec_2439_);
return v_res_2440_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__8(void){
_start:
{
lean_object* v___x_2463_; lean_object* v___x_2464_; 
v___x_2463_ = lean_unsigned_to_nat(12u);
v___x_2464_ = lean_nat_to_int(v___x_2463_);
return v___x_2464_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg(lean_object* v_x_2468_){
_start:
{
lean_object* v_message_2469_; uint32_t v_count_2470_; uint32_t v_busIndex_2471_; uint32_t v_countWeight_2472_; lean_object* v___x_2473_; lean_object* v___x_2474_; lean_object* v___x_2475_; lean_object* v___x_2476_; lean_object* v___x_2477_; uint8_t v___x_2478_; lean_object* v___x_2479_; lean_object* v___x_2480_; lean_object* v___x_2481_; lean_object* v___x_2482_; lean_object* v___x_2483_; lean_object* v___x_2484_; lean_object* v___x_2485_; lean_object* v___x_2486_; lean_object* v___x_2487_; lean_object* v___x_2488_; lean_object* v___x_2489_; lean_object* v___x_2490_; lean_object* v___x_2491_; lean_object* v___x_2492_; lean_object* v___x_2493_; lean_object* v___x_2494_; lean_object* v___x_2495_; lean_object* v___x_2496_; lean_object* v___x_2497_; lean_object* v___x_2498_; lean_object* v___x_2499_; lean_object* v___x_2500_; lean_object* v___x_2501_; lean_object* v___x_2502_; lean_object* v___x_2503_; lean_object* v___x_2504_; lean_object* v___x_2505_; lean_object* v___x_2506_; lean_object* v___x_2507_; lean_object* v___x_2508_; lean_object* v___x_2509_; lean_object* v___x_2510_; lean_object* v___x_2511_; lean_object* v___x_2512_; lean_object* v___x_2513_; lean_object* v___x_2514_; lean_object* v___x_2515_; lean_object* v___x_2516_; lean_object* v___x_2517_; lean_object* v___x_2518_; lean_object* v___x_2519_; lean_object* v___x_2520_; lean_object* v___x_2521_; lean_object* v___x_2522_; lean_object* v___x_2523_; lean_object* v___x_2524_; lean_object* v___x_2525_; 
v_message_2469_ = lean_ctor_get(v_x_2468_, 0);
lean_inc_ref(v_message_2469_);
v_count_2470_ = lean_ctor_get_uint32(v_x_2468_, sizeof(void*)*1);
v_busIndex_2471_ = lean_ctor_get_uint32(v_x_2468_, sizeof(void*)*1 + 4);
v_countWeight_2472_ = lean_ctor_get_uint32(v_x_2468_, sizeof(void*)*1 + 8);
lean_dec_ref(v_x_2468_);
v___x_2473_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__5));
v___x_2474_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__3));
v___x_2475_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__9, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__9_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawLogUpSecurityParameters_repr___redArg___closed__9);
v___x_2476_ = lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr_spec__1(v_message_2469_);
v___x_2477_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2477_, 0, v___x_2475_);
lean_ctor_set(v___x_2477_, 1, v___x_2476_);
v___x_2478_ = 0;
v___x_2479_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2479_, 0, v___x_2477_);
lean_ctor_set_uint8(v___x_2479_, sizeof(void*)*1, v___x_2478_);
v___x_2480_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2480_, 0, v___x_2474_);
lean_ctor_set(v___x_2480_, 1, v___x_2479_);
v___x_2481_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__2));
v___x_2482_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2482_, 0, v___x_2480_);
lean_ctor_set(v___x_2482_, 1, v___x_2481_);
v___x_2483_ = lean_box(1);
v___x_2484_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2484_, 0, v___x_2482_);
lean_ctor_set(v___x_2484_, 1, v___x_2483_);
v___x_2485_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__5));
v___x_2486_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2486_, 0, v___x_2484_);
lean_ctor_set(v___x_2486_, 1, v___x_2485_);
v___x_2487_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2487_, 0, v___x_2486_);
lean_ctor_set(v___x_2487_, 1, v___x_2473_);
v___x_2488_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSystemParams_repr___redArg___closed__4);
v___x_2489_ = lean_uint32_to_nat(v_count_2470_);
v___x_2490_ = l_Nat_reprFast(v___x_2489_);
v___x_2491_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2491_, 0, v___x_2490_);
v___x_2492_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2492_, 0, v___x_2488_);
lean_ctor_set(v___x_2492_, 1, v___x_2491_);
v___x_2493_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2493_, 0, v___x_2492_);
lean_ctor_set_uint8(v___x_2493_, sizeof(void*)*1, v___x_2478_);
v___x_2494_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2494_, 0, v___x_2487_);
lean_ctor_set(v___x_2494_, 1, v___x_2493_);
v___x_2495_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2495_, 0, v___x_2494_);
lean_ctor_set(v___x_2495_, 1, v___x_2481_);
v___x_2496_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2496_, 0, v___x_2495_);
lean_ctor_set(v___x_2496_, 1, v___x_2483_);
v___x_2497_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__7));
v___x_2498_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2498_, 0, v___x_2496_);
lean_ctor_set(v___x_2498_, 1, v___x_2497_);
v___x_2499_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2499_, 0, v___x_2498_);
lean_ctor_set(v___x_2499_, 1, v___x_2473_);
v___x_2500_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__8, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__8_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__8);
v___x_2501_ = lean_uint32_to_nat(v_busIndex_2471_);
v___x_2502_ = l_Nat_reprFast(v___x_2501_);
v___x_2503_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2503_, 0, v___x_2502_);
v___x_2504_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2504_, 0, v___x_2500_);
lean_ctor_set(v___x_2504_, 1, v___x_2503_);
v___x_2505_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2505_, 0, v___x_2504_);
lean_ctor_set_uint8(v___x_2505_, sizeof(void*)*1, v___x_2478_);
v___x_2506_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2506_, 0, v___x_2499_);
lean_ctor_set(v___x_2506_, 1, v___x_2505_);
v___x_2507_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2507_, 0, v___x_2506_);
lean_ctor_set(v___x_2507_, 1, v___x_2481_);
v___x_2508_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2508_, 0, v___x_2507_);
lean_ctor_set(v___x_2508_, 1, v___x_2483_);
v___x_2509_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg___closed__10));
v___x_2510_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2510_, 0, v___x_2508_);
lean_ctor_set(v___x_2510_, 1, v___x_2509_);
v___x_2511_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2511_, 0, v___x_2510_);
lean_ctor_set(v___x_2511_, 1, v___x_2473_);
v___x_2512_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__7, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__7_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__7);
v___x_2513_ = lean_uint32_to_nat(v_countWeight_2472_);
v___x_2514_ = l_Nat_reprFast(v___x_2513_);
v___x_2515_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2515_, 0, v___x_2514_);
v___x_2516_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2516_, 0, v___x_2512_);
lean_ctor_set(v___x_2516_, 1, v___x_2515_);
v___x_2517_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2517_, 0, v___x_2516_);
lean_ctor_set_uint8(v___x_2517_, sizeof(void*)*1, v___x_2478_);
v___x_2518_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2518_, 0, v___x_2511_);
lean_ctor_set(v___x_2518_, 1, v___x_2517_);
v___x_2519_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10);
v___x_2520_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__11));
v___x_2521_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2521_, 0, v___x_2520_);
lean_ctor_set(v___x_2521_, 1, v___x_2518_);
v___x_2522_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__12));
v___x_2523_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2523_, 0, v___x_2521_);
lean_ctor_set(v___x_2523_, 1, v___x_2522_);
v___x_2524_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2524_, 0, v___x_2519_);
lean_ctor_set(v___x_2524_, 1, v___x_2523_);
v___x_2525_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2525_, 0, v___x_2524_);
lean_ctor_set_uint8(v___x_2525_, sizeof(void*)*1, v___x_2478_);
return v___x_2525_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr(lean_object* v_x_2526_, lean_object* v_prec_2527_){
_start:
{
lean_object* v___x_2528_; 
v___x_2528_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg(v_x_2526_);
return v___x_2528_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___boxed(lean_object* v_x_2529_, lean_object* v_prec_2530_){
_start:
{
lean_object* v_res_2531_; 
v_res_2531_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr(v_x_2529_, v_prec_2530_);
lean_dec(v_prec_2530_);
return v_res_2531_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr_spec__0_spec__0_spec__1_spec__2(lean_object* v_x_2541_, lean_object* v_x_2542_, lean_object* v_x_2543_){
_start:
{
if (lean_obj_tag(v_x_2543_) == 0)
{
lean_dec(v_x_2541_);
return v_x_2542_;
}
else
{
lean_object* v_head_2544_; lean_object* v_tail_2545_; lean_object* v___x_2547_; uint8_t v_isShared_2548_; uint8_t v_isSharedCheck_2555_; 
v_head_2544_ = lean_ctor_get(v_x_2543_, 0);
v_tail_2545_ = lean_ctor_get(v_x_2543_, 1);
v_isSharedCheck_2555_ = !lean_is_exclusive(v_x_2543_);
if (v_isSharedCheck_2555_ == 0)
{
v___x_2547_ = v_x_2543_;
v_isShared_2548_ = v_isSharedCheck_2555_;
goto v_resetjp_2546_;
}
else
{
lean_inc(v_tail_2545_);
lean_inc(v_head_2544_);
lean_dec(v_x_2543_);
v___x_2547_ = lean_box(0);
v_isShared_2548_ = v_isSharedCheck_2555_;
goto v_resetjp_2546_;
}
v_resetjp_2546_:
{
lean_object* v___x_2550_; 
lean_inc(v_x_2541_);
if (v_isShared_2548_ == 0)
{
lean_ctor_set_tag(v___x_2547_, 5);
lean_ctor_set(v___x_2547_, 1, v_x_2541_);
lean_ctor_set(v___x_2547_, 0, v_x_2542_);
v___x_2550_ = v___x_2547_;
goto v_reusejp_2549_;
}
else
{
lean_object* v_reuseFailAlloc_2554_; 
v_reuseFailAlloc_2554_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2554_, 0, v_x_2542_);
lean_ctor_set(v_reuseFailAlloc_2554_, 1, v_x_2541_);
v___x_2550_ = v_reuseFailAlloc_2554_;
goto v_reusejp_2549_;
}
v_reusejp_2549_:
{
lean_object* v___x_2551_; lean_object* v___x_2552_; 
v___x_2551_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg(v_head_2544_);
v___x_2552_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2552_, 0, v___x_2550_);
lean_ctor_set(v___x_2552_, 1, v___x_2551_);
v_x_2542_ = v___x_2552_;
v_x_2543_ = v_tail_2545_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr_spec__0_spec__0_spec__1(lean_object* v_x_2556_, lean_object* v_x_2557_, lean_object* v_x_2558_){
_start:
{
if (lean_obj_tag(v_x_2558_) == 0)
{
lean_dec(v_x_2556_);
return v_x_2557_;
}
else
{
lean_object* v_head_2559_; lean_object* v_tail_2560_; lean_object* v___x_2562_; uint8_t v_isShared_2563_; uint8_t v_isSharedCheck_2570_; 
v_head_2559_ = lean_ctor_get(v_x_2558_, 0);
v_tail_2560_ = lean_ctor_get(v_x_2558_, 1);
v_isSharedCheck_2570_ = !lean_is_exclusive(v_x_2558_);
if (v_isSharedCheck_2570_ == 0)
{
v___x_2562_ = v_x_2558_;
v_isShared_2563_ = v_isSharedCheck_2570_;
goto v_resetjp_2561_;
}
else
{
lean_inc(v_tail_2560_);
lean_inc(v_head_2559_);
lean_dec(v_x_2558_);
v___x_2562_ = lean_box(0);
v_isShared_2563_ = v_isSharedCheck_2570_;
goto v_resetjp_2561_;
}
v_resetjp_2561_:
{
lean_object* v___x_2565_; 
lean_inc(v_x_2556_);
if (v_isShared_2563_ == 0)
{
lean_ctor_set_tag(v___x_2562_, 5);
lean_ctor_set(v___x_2562_, 1, v_x_2556_);
lean_ctor_set(v___x_2562_, 0, v_x_2557_);
v___x_2565_ = v___x_2562_;
goto v_reusejp_2564_;
}
else
{
lean_object* v_reuseFailAlloc_2569_; 
v_reuseFailAlloc_2569_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2569_, 0, v_x_2557_);
lean_ctor_set(v_reuseFailAlloc_2569_, 1, v_x_2556_);
v___x_2565_ = v_reuseFailAlloc_2569_;
goto v_reusejp_2564_;
}
v_reusejp_2564_:
{
lean_object* v___x_2566_; lean_object* v___x_2567_; lean_object* v___x_2568_; 
v___x_2566_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg(v_head_2559_);
v___x_2567_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2567_, 0, v___x_2565_);
lean_ctor_set(v___x_2567_, 1, v___x_2566_);
v___x_2568_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr_spec__0_spec__0_spec__1_spec__2(v_x_2556_, v___x_2567_, v_tail_2560_);
return v___x_2568_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr_spec__0_spec__0(lean_object* v_x_2571_, lean_object* v_x_2572_){
_start:
{
if (lean_obj_tag(v_x_2571_) == 0)
{
lean_object* v___x_2573_; 
lean_dec(v_x_2572_);
v___x_2573_ = lean_box(0);
return v___x_2573_;
}
else
{
lean_object* v_tail_2574_; 
v_tail_2574_ = lean_ctor_get(v_x_2571_, 1);
if (lean_obj_tag(v_tail_2574_) == 0)
{
lean_object* v_head_2575_; lean_object* v___x_2576_; 
lean_dec(v_x_2572_);
v_head_2575_ = lean_ctor_get(v_x_2571_, 0);
lean_inc(v_head_2575_);
lean_dec_ref_known(v_x_2571_, 2);
v___x_2576_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg(v_head_2575_);
return v___x_2576_;
}
else
{
lean_object* v_head_2577_; lean_object* v___x_2578_; lean_object* v___x_2579_; 
lean_inc(v_tail_2574_);
v_head_2577_ = lean_ctor_get(v_x_2571_, 0);
lean_inc(v_head_2577_);
lean_dec_ref_known(v_x_2571_, 2);
v___x_2578_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicInteraction_repr___redArg(v_head_2577_);
v___x_2579_ = lp_swirl_x2drbr_x2dformal_List_foldl___at___00Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr_spec__0_spec__0_spec__1(v_x_2572_, v___x_2578_, v_tail_2574_);
return v___x_2579_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr_spec__0(lean_object* v_xs_2580_){
_start:
{
lean_object* v___x_2581_; lean_object* v___x_2582_; uint8_t v___x_2583_; 
v___x_2581_ = lean_array_get_size(v_xs_2580_);
v___x_2582_ = lean_unsigned_to_nat(0u);
v___x_2583_ = lean_nat_dec_eq(v___x_2581_, v___x_2582_);
if (v___x_2583_ == 0)
{
lean_object* v___x_2584_; lean_object* v___x_2585_; lean_object* v___x_2586_; lean_object* v___x_2587_; lean_object* v___x_2588_; lean_object* v___x_2589_; lean_object* v___x_2590_; lean_object* v___x_2591_; lean_object* v___x_2592_; lean_object* v___x_2593_; 
v___x_2584_ = lean_array_to_list(v_xs_2580_);
v___x_2585_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__3));
v___x_2586_ = lp_swirl_x2drbr_x2dformal_Std_Format_joinSep___at___00Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr_spec__0_spec__0(v___x_2584_, v___x_2585_);
v___x_2587_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__6, &lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__6_once, _init_lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__6);
v___x_2588_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__7));
v___x_2589_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2589_, 0, v___x_2588_);
lean_ctor_set(v___x_2589_, 1, v___x_2586_);
v___x_2590_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__8));
v___x_2591_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2591_, 0, v___x_2589_);
lean_ctor_set(v___x_2591_, 1, v___x_2590_);
v___x_2592_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2592_, 0, v___x_2587_);
lean_ctor_set(v___x_2592_, 1, v___x_2591_);
v___x_2593_ = l_Std_Format_fill(v___x_2592_);
return v___x_2593_;
}
else
{
lean_object* v___x_2594_; 
lean_dec_ref(v_xs_2580_);
v___x_2594_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__10));
return v___x_2594_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg(lean_object* v_x_2607_){
_start:
{
lean_object* v_constraints_2608_; lean_object* v_interactions_2609_; lean_object* v___x_2611_; uint8_t v_isShared_2612_; uint8_t v_isSharedCheck_2642_; 
v_constraints_2608_ = lean_ctor_get(v_x_2607_, 0);
v_interactions_2609_ = lean_ctor_get(v_x_2607_, 1);
v_isSharedCheck_2642_ = !lean_is_exclusive(v_x_2607_);
if (v_isSharedCheck_2642_ == 0)
{
v___x_2611_ = v_x_2607_;
v_isShared_2612_ = v_isSharedCheck_2642_;
goto v_resetjp_2610_;
}
else
{
lean_inc(v_interactions_2609_);
lean_inc(v_constraints_2608_);
lean_dec(v_x_2607_);
v___x_2611_ = lean_box(0);
v_isShared_2612_ = v_isSharedCheck_2642_;
goto v_resetjp_2610_;
}
v_resetjp_2610_:
{
lean_object* v___x_2613_; lean_object* v___x_2614_; lean_object* v___x_2615_; lean_object* v___x_2616_; lean_object* v___x_2618_; 
v___x_2613_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__5));
v___x_2614_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg___closed__3));
v___x_2615_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__7, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__7_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__7);
v___x_2616_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicExpressionDag_repr___redArg(v_constraints_2608_);
if (v_isShared_2612_ == 0)
{
lean_ctor_set_tag(v___x_2611_, 4);
lean_ctor_set(v___x_2611_, 1, v___x_2616_);
lean_ctor_set(v___x_2611_, 0, v___x_2615_);
v___x_2618_ = v___x_2611_;
goto v_reusejp_2617_;
}
else
{
lean_object* v_reuseFailAlloc_2641_; 
v_reuseFailAlloc_2641_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2641_, 0, v___x_2615_);
lean_ctor_set(v_reuseFailAlloc_2641_, 1, v___x_2616_);
v___x_2618_ = v_reuseFailAlloc_2641_;
goto v_reusejp_2617_;
}
v_reusejp_2617_:
{
uint8_t v___x_2619_; lean_object* v___x_2620_; lean_object* v___x_2621_; lean_object* v___x_2622_; lean_object* v___x_2623_; lean_object* v___x_2624_; lean_object* v___x_2625_; lean_object* v___x_2626_; lean_object* v___x_2627_; lean_object* v___x_2628_; lean_object* v___x_2629_; lean_object* v___x_2630_; lean_object* v___x_2631_; lean_object* v___x_2632_; lean_object* v___x_2633_; lean_object* v___x_2634_; lean_object* v___x_2635_; lean_object* v___x_2636_; lean_object* v___x_2637_; lean_object* v___x_2638_; lean_object* v___x_2639_; lean_object* v___x_2640_; 
v___x_2619_ = 0;
v___x_2620_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2620_, 0, v___x_2618_);
lean_ctor_set_uint8(v___x_2620_, sizeof(void*)*1, v___x_2619_);
v___x_2621_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2621_, 0, v___x_2614_);
lean_ctor_set(v___x_2621_, 1, v___x_2620_);
v___x_2622_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirConfig_repr_spec__0___closed__2));
v___x_2623_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2623_, 0, v___x_2621_);
lean_ctor_set(v___x_2623_, 1, v___x_2622_);
v___x_2624_ = lean_box(1);
v___x_2625_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2625_, 0, v___x_2623_);
lean_ctor_set(v___x_2625_, 1, v___x_2624_);
v___x_2626_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg___closed__5));
v___x_2627_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2627_, 0, v___x_2625_);
lean_ctor_set(v___x_2627_, 1, v___x_2626_);
v___x_2628_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2628_, 0, v___x_2627_);
lean_ctor_set(v___x_2628_, 1, v___x_2613_);
v___x_2629_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__4, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__4_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawTraceWidth_repr___redArg___closed__4);
v___x_2630_ = lp_swirl_x2drbr_x2dformal_Array_repr___at___00Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr_spec__0(v_interactions_2609_);
v___x_2631_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2631_, 0, v___x_2629_);
lean_ctor_set(v___x_2631_, 1, v___x_2630_);
v___x_2632_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2632_, 0, v___x_2631_);
lean_ctor_set_uint8(v___x_2632_, sizeof(void*)*1, v___x_2619_);
v___x_2633_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2633_, 0, v___x_2628_);
lean_ctor_set(v___x_2633_, 1, v___x_2632_);
v___x_2634_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__10);
v___x_2635_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__11));
v___x_2636_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2636_, 0, v___x_2635_);
lean_ctor_set(v___x_2636_, 1, v___x_2633_);
v___x_2637_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawWhirRoundConfig_repr___redArg___closed__12));
v___x_2638_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_2638_, 0, v___x_2636_);
lean_ctor_set(v___x_2638_, 1, v___x_2637_);
v___x_2639_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2639_, 0, v___x_2634_);
lean_ctor_set(v___x_2639_, 1, v___x_2638_);
v___x_2640_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_2640_, 0, v___x_2639_);
lean_ctor_set_uint8(v___x_2640_, sizeof(void*)*1, v___x_2619_);
return v___x_2640_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr(lean_object* v_x_2643_, lean_object* v_prec_2644_){
_start:
{
lean_object* v___x_2645_; 
v___x_2645_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___redArg(v_x_2643_);
return v___x_2645_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr___boxed(lean_object* v_x_2646_, lean_object* v_prec_2647_){
_start:
{
lean_object* v_res_2648_; 
v_res_2648_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instReprRawSymbolicConstraintsDag_repr(v_x_2646_, v_prec_2647_);
lean_dec(v_prec_2647_);
return v_res_2648_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVk_default___closed__0(void){
_start:
{
lean_object* v___x_2669_; lean_object* v___x_2670_; lean_object* v___x_2671_; 
v___x_2669_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__0, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__0_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__0);
v___x_2670_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawMultiStarkVerifyingKey0_default));
v___x_2671_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2671_, 0, v___x_2670_);
lean_ctor_set(v___x_2671_, 1, v___x_2669_);
return v___x_2671_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVk_default(void){
_start:
{
lean_object* v___x_2672_; 
v___x_2672_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVk_default___closed__0, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVk_default___closed__0_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVk_default___closed__0);
return v___x_2672_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVk(void){
_start:
{
lean_object* v___x_2673_; 
v___x_2673_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVk_default;
return v___x_2673_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawVk_airCount(lean_object* v_vk_2674_){
_start:
{
lean_object* v_inner_2675_; lean_object* v_perAir_2676_; lean_object* v___x_2677_; 
v_inner_2675_ = lean_ctor_get(v_vk_2674_, 0);
v_perAir_2676_ = lean_ctor_get(v_inner_2675_, 1);
v___x_2677_ = lean_array_get_size(v_perAir_2676_);
return v___x_2677_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawVk_airCount___boxed(lean_object* v_vk_2678_){
_start:
{
lean_object* v_res_2679_; 
v_res_2679_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawVk_airCount(v_vk_2678_);
lean_dec_ref(v_vk_2678_);
return v_res_2679_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims_default___closed__0(void){
_start:
{
lean_object* v___x_2687_; lean_object* v___x_2688_; lean_object* v___x_2689_; 
v___x_2687_ = lean_unsigned_to_nat(4u);
v___x_2688_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__0___boxed__const__1;
v___x_2689_ = lean_mk_array(v___x_2687_, v___x_2688_);
return v___x_2689_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims_default___closed__1(void){
_start:
{
lean_object* v___x_2690_; lean_object* v___x_2691_; 
v___x_2690_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims_default___closed__0, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims_default___closed__0_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims_default___closed__0);
v___x_2691_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2691_, 0, v___x_2690_);
lean_ctor_set(v___x_2691_, 1, v___x_2690_);
lean_ctor_set(v___x_2691_, 2, v___x_2690_);
lean_ctor_set(v___x_2691_, 3, v___x_2690_);
return v___x_2691_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims_default(void){
_start:
{
lean_object* v___x_2692_; 
v___x_2692_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims_default___closed__1, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims_default___closed__1_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims_default___closed__1);
return v___x_2692_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims(void){
_start:
{
lean_object* v___x_2693_; 
v___x_2693_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims_default;
return v___x_2693_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrProof_default___closed__1(void){
_start:
{
lean_object* v___x_2696_; lean_object* v___x_2697_; uint32_t v___x_2698_; lean_object* v___x_2699_; 
v___x_2696_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrProof_default___closed__0));
v___x_2697_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims_default___closed__0, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims_default___closed__0_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims_default___closed__0);
v___x_2698_ = 0;
v___x_2699_ = lean_alloc_ctor(0, 3, 4);
lean_ctor_set(v___x_2699_, 0, v___x_2697_);
lean_ctor_set(v___x_2699_, 1, v___x_2696_);
lean_ctor_set(v___x_2699_, 2, v___x_2696_);
lean_ctor_set_uint32(v___x_2699_, sizeof(void*)*3, v___x_2698_);
return v___x_2699_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrProof_default(void){
_start:
{
lean_object* v___x_2700_; 
v___x_2700_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrProof_default___closed__1, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrProof_default___closed__1_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrProof_default___closed__1);
return v___x_2700_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrProof(void){
_start:
{
lean_object* v___x_2701_; 
v___x_2701_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrProof_default;
return v___x_2701_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawProof_default___closed__1(void){
_start:
{
lean_object* v___x_2719_; lean_object* v___x_2720_; lean_object* v___x_2721_; lean_object* v___x_2722_; lean_object* v___x_2723_; lean_object* v___x_2724_; lean_object* v___x_2725_; 
v___x_2719_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProof_default));
v___x_2720_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawStackingProof_default));
v___x_2721_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawBatchConstraintProof_default));
v___x_2722_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrProof_default;
v___x_2723_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawProof_default___closed__0));
v___x_2724_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__0, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__0_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__0);
v___x_2725_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_2725_, 0, v___x_2724_);
lean_ctor_set(v___x_2725_, 1, v___x_2723_);
lean_ctor_set(v___x_2725_, 2, v___x_2722_);
lean_ctor_set(v___x_2725_, 3, v___x_2721_);
lean_ctor_set(v___x_2725_, 4, v___x_2720_);
lean_ctor_set(v___x_2725_, 5, v___x_2719_);
return v___x_2725_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawProof_default(void){
_start:
{
lean_object* v___x_2726_; 
v___x_2726_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawProof_default___closed__1, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawProof_default___closed__1_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawProof_default___closed__1);
return v___x_2726_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawProof(void){
_start:
{
lean_object* v___x_2727_; 
v___x_2727_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawProof_default;
return v___x_2727_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readEntry(lean_object* v_a_2731_){
_start:
{
lean_object* v___x_2732_; 
lean_inc_ref(v_a_2731_);
v___x_2732_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readByte(v_a_2731_);
if (lean_obj_tag(v___x_2732_) == 0)
{
lean_object* v_a_2733_; lean_object* v_a_2734_; lean_object* v___x_2736_; uint8_t v_isShared_2737_; uint8_t v_isSharedCheck_2830_; 
v_a_2733_ = lean_ctor_get(v___x_2732_, 0);
v_a_2734_ = lean_ctor_get(v___x_2732_, 1);
v_isSharedCheck_2830_ = !lean_is_exclusive(v___x_2732_);
if (v_isSharedCheck_2830_ == 0)
{
v___x_2736_ = v___x_2732_;
v_isShared_2737_ = v_isSharedCheck_2830_;
goto v_resetjp_2735_;
}
else
{
lean_inc(v_a_2734_);
lean_inc(v_a_2733_);
lean_dec(v___x_2732_);
v___x_2736_ = lean_box(0);
v_isShared_2737_ = v_isSharedCheck_2830_;
goto v_resetjp_2735_;
}
v_resetjp_2735_:
{
uint8_t v___x_2738_; uint8_t v___x_2739_; uint8_t v___x_2740_; 
v___x_2738_ = 0;
v___x_2739_ = lean_unbox(v_a_2733_);
v___x_2740_ = lean_uint8_dec_eq(v___x_2739_, v___x_2738_);
if (v___x_2740_ == 0)
{
uint8_t v___x_2741_; uint8_t v___x_2742_; uint8_t v___x_2743_; 
v___x_2741_ = 1;
v___x_2742_ = lean_unbox(v_a_2733_);
v___x_2743_ = lean_uint8_dec_eq(v___x_2742_, v___x_2741_);
if (v___x_2743_ == 0)
{
uint8_t v___x_2744_; uint8_t v___x_2745_; uint8_t v___x_2746_; 
v___x_2744_ = 2;
v___x_2745_ = lean_unbox(v_a_2733_);
v___x_2746_ = lean_uint8_dec_eq(v___x_2745_, v___x_2744_);
if (v___x_2746_ == 0)
{
uint8_t v___x_2747_; uint8_t v___x_2748_; uint8_t v___x_2749_; 
v___x_2747_ = 3;
v___x_2748_ = lean_unbox(v_a_2733_);
v___x_2749_ = lean_uint8_dec_eq(v___x_2748_, v___x_2747_);
if (v___x_2749_ == 0)
{
lean_object* v_offset_2750_; lean_object* v___x_2752_; uint8_t v_isShared_2753_; uint8_t v_isSharedCheck_2765_; 
v_offset_2750_ = lean_ctor_get(v_a_2731_, 1);
v_isSharedCheck_2765_ = !lean_is_exclusive(v_a_2731_);
if (v_isSharedCheck_2765_ == 0)
{
lean_object* v_unused_2766_; 
v_unused_2766_ = lean_ctor_get(v_a_2731_, 0);
lean_dec(v_unused_2766_);
v___x_2752_ = v_a_2731_;
v_isShared_2753_ = v_isSharedCheck_2765_;
goto v_resetjp_2751_;
}
else
{
lean_inc(v_offset_2750_);
lean_dec(v_a_2731_);
v___x_2752_ = lean_box(0);
v_isShared_2753_ = v_isSharedCheck_2765_;
goto v_resetjp_2751_;
}
v_resetjp_2751_:
{
lean_object* v___x_2754_; uint8_t v___x_2755_; lean_object* v___x_2756_; lean_object* v___x_2757_; lean_object* v___x_2758_; lean_object* v___x_2760_; 
v___x_2754_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readEntry___closed__0));
v___x_2755_ = lean_unbox(v_a_2733_);
lean_dec(v_a_2733_);
v___x_2756_ = lean_uint8_to_nat(v___x_2755_);
v___x_2757_ = l_Nat_reprFast(v___x_2756_);
v___x_2758_ = lean_string_append(v___x_2754_, v___x_2757_);
lean_dec_ref(v___x_2757_);
if (v_isShared_2753_ == 0)
{
lean_ctor_set_tag(v___x_2752_, 3);
lean_ctor_set(v___x_2752_, 1, v___x_2758_);
lean_ctor_set(v___x_2752_, 0, v_offset_2750_);
v___x_2760_ = v___x_2752_;
goto v_reusejp_2759_;
}
else
{
lean_object* v_reuseFailAlloc_2764_; 
v_reuseFailAlloc_2764_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2764_, 0, v_offset_2750_);
lean_ctor_set(v_reuseFailAlloc_2764_, 1, v___x_2758_);
v___x_2760_ = v_reuseFailAlloc_2764_;
goto v_reusejp_2759_;
}
v_reusejp_2759_:
{
lean_object* v___x_2762_; 
if (v_isShared_2737_ == 0)
{
lean_ctor_set_tag(v___x_2736_, 1);
lean_ctor_set(v___x_2736_, 0, v___x_2760_);
v___x_2762_ = v___x_2736_;
goto v_reusejp_2761_;
}
else
{
lean_object* v_reuseFailAlloc_2763_; 
v_reuseFailAlloc_2763_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2763_, 0, v___x_2760_);
lean_ctor_set(v_reuseFailAlloc_2763_, 1, v_a_2734_);
v___x_2762_ = v_reuseFailAlloc_2763_;
goto v_reusejp_2761_;
}
v_reusejp_2761_:
{
return v___x_2762_;
}
}
}
}
else
{
lean_object* v___x_2767_; lean_object* v___x_2769_; 
lean_dec(v_a_2733_);
lean_dec_ref(v_a_2731_);
v___x_2767_ = lean_box(3);
if (v_isShared_2737_ == 0)
{
lean_ctor_set(v___x_2736_, 0, v___x_2767_);
v___x_2769_ = v___x_2736_;
goto v_reusejp_2768_;
}
else
{
lean_object* v_reuseFailAlloc_2770_; 
v_reuseFailAlloc_2770_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2770_, 0, v___x_2767_);
lean_ctor_set(v_reuseFailAlloc_2770_, 1, v_a_2734_);
v___x_2769_ = v_reuseFailAlloc_2770_;
goto v_reusejp_2768_;
}
v_reusejp_2768_:
{
return v___x_2769_;
}
}
}
else
{
lean_object* v___x_2771_; lean_object* v___x_2773_; 
lean_dec(v_a_2733_);
lean_dec_ref(v_a_2731_);
v___x_2771_ = lean_box(2);
if (v_isShared_2737_ == 0)
{
lean_ctor_set(v___x_2736_, 0, v___x_2771_);
v___x_2773_ = v___x_2736_;
goto v_reusejp_2772_;
}
else
{
lean_object* v_reuseFailAlloc_2774_; 
v_reuseFailAlloc_2774_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2774_, 0, v___x_2771_);
lean_ctor_set(v_reuseFailAlloc_2774_, 1, v_a_2734_);
v___x_2773_ = v_reuseFailAlloc_2774_;
goto v_reusejp_2772_;
}
v_reusejp_2772_:
{
return v___x_2773_;
}
}
}
else
{
lean_object* v___x_2775_; 
lean_del_object(v___x_2736_);
lean_dec(v_a_2733_);
lean_dec_ref(v_a_2731_);
v___x_2775_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_2734_);
if (lean_obj_tag(v___x_2775_) == 0)
{
lean_object* v_a_2776_; lean_object* v_a_2777_; lean_object* v___x_2778_; 
v_a_2776_ = lean_ctor_get(v___x_2775_, 0);
lean_inc(v_a_2776_);
v_a_2777_ = lean_ctor_get(v___x_2775_, 1);
lean_inc(v_a_2777_);
lean_dec_ref_known(v___x_2775_, 2);
v___x_2778_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_2777_);
if (lean_obj_tag(v___x_2778_) == 0)
{
lean_object* v_a_2779_; lean_object* v_a_2780_; lean_object* v___x_2782_; uint8_t v_isShared_2783_; uint8_t v_isSharedCheck_2790_; 
v_a_2779_ = lean_ctor_get(v___x_2778_, 0);
v_a_2780_ = lean_ctor_get(v___x_2778_, 1);
v_isSharedCheck_2790_ = !lean_is_exclusive(v___x_2778_);
if (v_isSharedCheck_2790_ == 0)
{
v___x_2782_ = v___x_2778_;
v_isShared_2783_ = v_isSharedCheck_2790_;
goto v_resetjp_2781_;
}
else
{
lean_inc(v_a_2780_);
lean_inc(v_a_2779_);
lean_dec(v___x_2778_);
v___x_2782_ = lean_box(0);
v_isShared_2783_ = v_isSharedCheck_2790_;
goto v_resetjp_2781_;
}
v_resetjp_2781_:
{
lean_object* v___x_2784_; uint32_t v___x_2785_; uint32_t v___x_2786_; lean_object* v___x_2788_; 
v___x_2784_ = lean_alloc_ctor(1, 0, 8);
v___x_2785_ = lean_unbox_uint32(v_a_2776_);
lean_dec(v_a_2776_);
lean_ctor_set_uint32(v___x_2784_, 0, v___x_2785_);
v___x_2786_ = lean_unbox_uint32(v_a_2779_);
lean_dec(v_a_2779_);
lean_ctor_set_uint32(v___x_2784_, 4, v___x_2786_);
if (v_isShared_2783_ == 0)
{
lean_ctor_set(v___x_2782_, 0, v___x_2784_);
v___x_2788_ = v___x_2782_;
goto v_reusejp_2787_;
}
else
{
lean_object* v_reuseFailAlloc_2789_; 
v_reuseFailAlloc_2789_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2789_, 0, v___x_2784_);
lean_ctor_set(v_reuseFailAlloc_2789_, 1, v_a_2780_);
v___x_2788_ = v_reuseFailAlloc_2789_;
goto v_reusejp_2787_;
}
v_reusejp_2787_:
{
return v___x_2788_;
}
}
}
else
{
lean_object* v_a_2791_; lean_object* v_a_2792_; lean_object* v___x_2794_; uint8_t v_isShared_2795_; uint8_t v_isSharedCheck_2799_; 
lean_dec(v_a_2776_);
v_a_2791_ = lean_ctor_get(v___x_2778_, 0);
v_a_2792_ = lean_ctor_get(v___x_2778_, 1);
v_isSharedCheck_2799_ = !lean_is_exclusive(v___x_2778_);
if (v_isSharedCheck_2799_ == 0)
{
v___x_2794_ = v___x_2778_;
v_isShared_2795_ = v_isSharedCheck_2799_;
goto v_resetjp_2793_;
}
else
{
lean_inc(v_a_2792_);
lean_inc(v_a_2791_);
lean_dec(v___x_2778_);
v___x_2794_ = lean_box(0);
v_isShared_2795_ = v_isSharedCheck_2799_;
goto v_resetjp_2793_;
}
v_resetjp_2793_:
{
lean_object* v___x_2797_; 
if (v_isShared_2795_ == 0)
{
v___x_2797_ = v___x_2794_;
goto v_reusejp_2796_;
}
else
{
lean_object* v_reuseFailAlloc_2798_; 
v_reuseFailAlloc_2798_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2798_, 0, v_a_2791_);
lean_ctor_set(v_reuseFailAlloc_2798_, 1, v_a_2792_);
v___x_2797_ = v_reuseFailAlloc_2798_;
goto v_reusejp_2796_;
}
v_reusejp_2796_:
{
return v___x_2797_;
}
}
}
}
else
{
lean_object* v_a_2800_; lean_object* v_a_2801_; lean_object* v___x_2803_; uint8_t v_isShared_2804_; uint8_t v_isSharedCheck_2808_; 
v_a_2800_ = lean_ctor_get(v___x_2775_, 0);
v_a_2801_ = lean_ctor_get(v___x_2775_, 1);
v_isSharedCheck_2808_ = !lean_is_exclusive(v___x_2775_);
if (v_isSharedCheck_2808_ == 0)
{
v___x_2803_ = v___x_2775_;
v_isShared_2804_ = v_isSharedCheck_2808_;
goto v_resetjp_2802_;
}
else
{
lean_inc(v_a_2801_);
lean_inc(v_a_2800_);
lean_dec(v___x_2775_);
v___x_2803_ = lean_box(0);
v_isShared_2804_ = v_isSharedCheck_2808_;
goto v_resetjp_2802_;
}
v_resetjp_2802_:
{
lean_object* v___x_2806_; 
if (v_isShared_2804_ == 0)
{
v___x_2806_ = v___x_2803_;
goto v_reusejp_2805_;
}
else
{
lean_object* v_reuseFailAlloc_2807_; 
v_reuseFailAlloc_2807_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2807_, 0, v_a_2800_);
lean_ctor_set(v_reuseFailAlloc_2807_, 1, v_a_2801_);
v___x_2806_ = v_reuseFailAlloc_2807_;
goto v_reusejp_2805_;
}
v_reusejp_2805_:
{
return v___x_2806_;
}
}
}
}
}
else
{
lean_object* v___x_2809_; 
lean_del_object(v___x_2736_);
lean_dec(v_a_2733_);
lean_dec_ref(v_a_2731_);
v___x_2809_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_2734_);
if (lean_obj_tag(v___x_2809_) == 0)
{
lean_object* v_a_2810_; lean_object* v_a_2811_; lean_object* v___x_2813_; uint8_t v_isShared_2814_; uint8_t v_isSharedCheck_2820_; 
v_a_2810_ = lean_ctor_get(v___x_2809_, 0);
v_a_2811_ = lean_ctor_get(v___x_2809_, 1);
v_isSharedCheck_2820_ = !lean_is_exclusive(v___x_2809_);
if (v_isSharedCheck_2820_ == 0)
{
v___x_2813_ = v___x_2809_;
v_isShared_2814_ = v_isSharedCheck_2820_;
goto v_resetjp_2812_;
}
else
{
lean_inc(v_a_2811_);
lean_inc(v_a_2810_);
lean_dec(v___x_2809_);
v___x_2813_ = lean_box(0);
v_isShared_2814_ = v_isSharedCheck_2820_;
goto v_resetjp_2812_;
}
v_resetjp_2812_:
{
lean_object* v___x_2815_; uint32_t v___x_2816_; lean_object* v___x_2818_; 
v___x_2815_ = lean_alloc_ctor(0, 0, 4);
v___x_2816_ = lean_unbox_uint32(v_a_2810_);
lean_dec(v_a_2810_);
lean_ctor_set_uint32(v___x_2815_, 0, v___x_2816_);
if (v_isShared_2814_ == 0)
{
lean_ctor_set(v___x_2813_, 0, v___x_2815_);
v___x_2818_ = v___x_2813_;
goto v_reusejp_2817_;
}
else
{
lean_object* v_reuseFailAlloc_2819_; 
v_reuseFailAlloc_2819_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2819_, 0, v___x_2815_);
lean_ctor_set(v_reuseFailAlloc_2819_, 1, v_a_2811_);
v___x_2818_ = v_reuseFailAlloc_2819_;
goto v_reusejp_2817_;
}
v_reusejp_2817_:
{
return v___x_2818_;
}
}
}
else
{
lean_object* v_a_2821_; lean_object* v_a_2822_; lean_object* v___x_2824_; uint8_t v_isShared_2825_; uint8_t v_isSharedCheck_2829_; 
v_a_2821_ = lean_ctor_get(v___x_2809_, 0);
v_a_2822_ = lean_ctor_get(v___x_2809_, 1);
v_isSharedCheck_2829_ = !lean_is_exclusive(v___x_2809_);
if (v_isSharedCheck_2829_ == 0)
{
v___x_2824_ = v___x_2809_;
v_isShared_2825_ = v_isSharedCheck_2829_;
goto v_resetjp_2823_;
}
else
{
lean_inc(v_a_2822_);
lean_inc(v_a_2821_);
lean_dec(v___x_2809_);
v___x_2824_ = lean_box(0);
v_isShared_2825_ = v_isSharedCheck_2829_;
goto v_resetjp_2823_;
}
v_resetjp_2823_:
{
lean_object* v___x_2827_; 
if (v_isShared_2825_ == 0)
{
v___x_2827_ = v___x_2824_;
goto v_reusejp_2826_;
}
else
{
lean_object* v_reuseFailAlloc_2828_; 
v_reuseFailAlloc_2828_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2828_, 0, v_a_2821_);
lean_ctor_set(v_reuseFailAlloc_2828_, 1, v_a_2822_);
v___x_2827_ = v_reuseFailAlloc_2828_;
goto v_reusejp_2826_;
}
v_reusejp_2826_:
{
return v___x_2827_;
}
}
}
}
}
}
else
{
lean_object* v_a_2831_; lean_object* v_a_2832_; lean_object* v___x_2834_; uint8_t v_isShared_2835_; uint8_t v_isSharedCheck_2839_; 
lean_dec_ref(v_a_2731_);
v_a_2831_ = lean_ctor_get(v___x_2732_, 0);
v_a_2832_ = lean_ctor_get(v___x_2732_, 1);
v_isSharedCheck_2839_ = !lean_is_exclusive(v___x_2732_);
if (v_isSharedCheck_2839_ == 0)
{
v___x_2834_ = v___x_2732_;
v_isShared_2835_ = v_isSharedCheck_2839_;
goto v_resetjp_2833_;
}
else
{
lean_inc(v_a_2832_);
lean_inc(v_a_2831_);
lean_dec(v___x_2732_);
v___x_2834_ = lean_box(0);
v_isShared_2835_ = v_isSharedCheck_2839_;
goto v_resetjp_2833_;
}
v_resetjp_2833_:
{
lean_object* v___x_2837_; 
if (v_isShared_2835_ == 0)
{
v___x_2837_ = v___x_2834_;
goto v_reusejp_2836_;
}
else
{
lean_object* v_reuseFailAlloc_2838_; 
v_reuseFailAlloc_2838_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2838_, 0, v_a_2831_);
lean_ctor_set(v_reuseFailAlloc_2838_, 1, v_a_2832_);
v___x_2837_ = v_reuseFailAlloc_2838_;
goto v_reusejp_2836_;
}
v_reusejp_2836_:
{
return v___x_2837_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readSymbolicVariable(lean_object* v_a_2840_){
_start:
{
lean_object* v___x_2841_; 
v___x_2841_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readEntry(v_a_2840_);
if (lean_obj_tag(v___x_2841_) == 0)
{
lean_object* v_a_2842_; lean_object* v_a_2843_; lean_object* v___x_2844_; 
v_a_2842_ = lean_ctor_get(v___x_2841_, 0);
lean_inc(v_a_2842_);
v_a_2843_ = lean_ctor_get(v___x_2841_, 1);
lean_inc(v_a_2843_);
lean_dec_ref_known(v___x_2841_, 2);
v___x_2844_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_2843_);
if (lean_obj_tag(v___x_2844_) == 0)
{
lean_object* v_a_2845_; lean_object* v_a_2846_; lean_object* v___x_2848_; uint8_t v_isShared_2849_; uint8_t v_isSharedCheck_2855_; 
v_a_2845_ = lean_ctor_get(v___x_2844_, 0);
v_a_2846_ = lean_ctor_get(v___x_2844_, 1);
v_isSharedCheck_2855_ = !lean_is_exclusive(v___x_2844_);
if (v_isSharedCheck_2855_ == 0)
{
v___x_2848_ = v___x_2844_;
v_isShared_2849_ = v_isSharedCheck_2855_;
goto v_resetjp_2847_;
}
else
{
lean_inc(v_a_2846_);
lean_inc(v_a_2845_);
lean_dec(v___x_2844_);
v___x_2848_ = lean_box(0);
v_isShared_2849_ = v_isSharedCheck_2855_;
goto v_resetjp_2847_;
}
v_resetjp_2847_:
{
lean_object* v___x_2850_; uint32_t v___x_2851_; lean_object* v___x_2853_; 
v___x_2850_ = lean_alloc_ctor(0, 1, 4);
lean_ctor_set(v___x_2850_, 0, v_a_2842_);
v___x_2851_ = lean_unbox_uint32(v_a_2845_);
lean_dec(v_a_2845_);
lean_ctor_set_uint32(v___x_2850_, sizeof(void*)*1, v___x_2851_);
if (v_isShared_2849_ == 0)
{
lean_ctor_set(v___x_2848_, 0, v___x_2850_);
v___x_2853_ = v___x_2848_;
goto v_reusejp_2852_;
}
else
{
lean_object* v_reuseFailAlloc_2854_; 
v_reuseFailAlloc_2854_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2854_, 0, v___x_2850_);
lean_ctor_set(v_reuseFailAlloc_2854_, 1, v_a_2846_);
v___x_2853_ = v_reuseFailAlloc_2854_;
goto v_reusejp_2852_;
}
v_reusejp_2852_:
{
return v___x_2853_;
}
}
}
else
{
lean_object* v_a_2856_; lean_object* v_a_2857_; lean_object* v___x_2859_; uint8_t v_isShared_2860_; uint8_t v_isSharedCheck_2864_; 
lean_dec(v_a_2842_);
v_a_2856_ = lean_ctor_get(v___x_2844_, 0);
v_a_2857_ = lean_ctor_get(v___x_2844_, 1);
v_isSharedCheck_2864_ = !lean_is_exclusive(v___x_2844_);
if (v_isSharedCheck_2864_ == 0)
{
v___x_2859_ = v___x_2844_;
v_isShared_2860_ = v_isSharedCheck_2864_;
goto v_resetjp_2858_;
}
else
{
lean_inc(v_a_2857_);
lean_inc(v_a_2856_);
lean_dec(v___x_2844_);
v___x_2859_ = lean_box(0);
v_isShared_2860_ = v_isSharedCheck_2864_;
goto v_resetjp_2858_;
}
v_resetjp_2858_:
{
lean_object* v___x_2862_; 
if (v_isShared_2860_ == 0)
{
v___x_2862_ = v___x_2859_;
goto v_reusejp_2861_;
}
else
{
lean_object* v_reuseFailAlloc_2863_; 
v_reuseFailAlloc_2863_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2863_, 0, v_a_2856_);
lean_ctor_set(v_reuseFailAlloc_2863_, 1, v_a_2857_);
v___x_2862_ = v_reuseFailAlloc_2863_;
goto v_reusejp_2861_;
}
v_reusejp_2861_:
{
return v___x_2862_;
}
}
}
}
else
{
lean_object* v_a_2865_; lean_object* v_a_2866_; lean_object* v___x_2868_; uint8_t v_isShared_2869_; uint8_t v_isSharedCheck_2873_; 
v_a_2865_ = lean_ctor_get(v___x_2841_, 0);
v_a_2866_ = lean_ctor_get(v___x_2841_, 1);
v_isSharedCheck_2873_ = !lean_is_exclusive(v___x_2841_);
if (v_isSharedCheck_2873_ == 0)
{
v___x_2868_ = v___x_2841_;
v_isShared_2869_ = v_isSharedCheck_2873_;
goto v_resetjp_2867_;
}
else
{
lean_inc(v_a_2866_);
lean_inc(v_a_2865_);
lean_dec(v___x_2841_);
v___x_2868_ = lean_box(0);
v_isShared_2869_ = v_isSharedCheck_2873_;
goto v_resetjp_2867_;
}
v_resetjp_2867_:
{
lean_object* v___x_2871_; 
if (v_isShared_2869_ == 0)
{
v___x_2871_ = v___x_2868_;
goto v_reusejp_2870_;
}
else
{
lean_object* v_reuseFailAlloc_2872_; 
v_reuseFailAlloc_2872_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2872_, 0, v_a_2865_);
lean_ctor_set(v_reuseFailAlloc_2872_, 1, v_a_2866_);
v___x_2871_ = v_reuseFailAlloc_2872_;
goto v_reusejp_2870_;
}
v_reusejp_2870_:
{
return v___x_2871_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readSymbolicExpressionNode(lean_object* v_a_2875_){
_start:
{
lean_object* v___x_2876_; 
lean_inc_ref(v_a_2875_);
v___x_2876_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readByte(v_a_2875_);
if (lean_obj_tag(v___x_2876_) == 0)
{
lean_object* v_a_2877_; lean_object* v_a_2878_; lean_object* v___x_2880_; uint8_t v_isShared_2881_; uint8_t v_isSharedCheck_3154_; 
v_a_2877_ = lean_ctor_get(v___x_2876_, 0);
v_a_2878_ = lean_ctor_get(v___x_2876_, 1);
v_isSharedCheck_3154_ = !lean_is_exclusive(v___x_2876_);
if (v_isSharedCheck_3154_ == 0)
{
v___x_2880_ = v___x_2876_;
v_isShared_2881_ = v_isSharedCheck_3154_;
goto v_resetjp_2879_;
}
else
{
lean_inc(v_a_2878_);
lean_inc(v_a_2877_);
lean_dec(v___x_2876_);
v___x_2880_ = lean_box(0);
v_isShared_2881_ = v_isSharedCheck_3154_;
goto v_resetjp_2879_;
}
v_resetjp_2879_:
{
uint8_t v___x_2882_; uint8_t v___x_2883_; uint8_t v___x_2884_; 
v___x_2882_ = 0;
v___x_2883_ = lean_unbox(v_a_2877_);
v___x_2884_ = lean_uint8_dec_eq(v___x_2883_, v___x_2882_);
if (v___x_2884_ == 0)
{
uint8_t v___x_2885_; uint8_t v___x_2886_; uint8_t v___x_2887_; 
v___x_2885_ = 1;
v___x_2886_ = lean_unbox(v_a_2877_);
v___x_2887_ = lean_uint8_dec_eq(v___x_2886_, v___x_2885_);
if (v___x_2887_ == 0)
{
uint8_t v___x_2888_; uint8_t v___x_2889_; uint8_t v___x_2890_; 
v___x_2888_ = 2;
v___x_2889_ = lean_unbox(v_a_2877_);
v___x_2890_ = lean_uint8_dec_eq(v___x_2889_, v___x_2888_);
if (v___x_2890_ == 0)
{
uint8_t v___x_2891_; uint8_t v___x_2892_; uint8_t v___x_2893_; 
v___x_2891_ = 3;
v___x_2892_ = lean_unbox(v_a_2877_);
v___x_2893_ = lean_uint8_dec_eq(v___x_2892_, v___x_2891_);
if (v___x_2893_ == 0)
{
uint8_t v___x_2894_; uint8_t v___x_2895_; uint8_t v___x_2896_; 
v___x_2894_ = 4;
v___x_2895_ = lean_unbox(v_a_2877_);
v___x_2896_ = lean_uint8_dec_eq(v___x_2895_, v___x_2894_);
if (v___x_2896_ == 0)
{
uint8_t v___x_2897_; uint8_t v___x_2898_; uint8_t v___x_2899_; 
v___x_2897_ = 5;
v___x_2898_ = lean_unbox(v_a_2877_);
v___x_2899_ = lean_uint8_dec_eq(v___x_2898_, v___x_2897_);
if (v___x_2899_ == 0)
{
uint8_t v___x_2900_; uint8_t v___x_2901_; uint8_t v___x_2902_; 
v___x_2900_ = 6;
v___x_2901_ = lean_unbox(v_a_2877_);
v___x_2902_ = lean_uint8_dec_eq(v___x_2901_, v___x_2900_);
if (v___x_2902_ == 0)
{
uint8_t v___x_2903_; uint8_t v___x_2904_; uint8_t v___x_2905_; 
v___x_2903_ = 7;
v___x_2904_ = lean_unbox(v_a_2877_);
v___x_2905_ = lean_uint8_dec_eq(v___x_2904_, v___x_2903_);
if (v___x_2905_ == 0)
{
uint8_t v___x_2906_; uint8_t v___x_2907_; uint8_t v___x_2908_; 
v___x_2906_ = 8;
v___x_2907_ = lean_unbox(v_a_2877_);
v___x_2908_ = lean_uint8_dec_eq(v___x_2907_, v___x_2906_);
if (v___x_2908_ == 0)
{
lean_object* v_offset_2909_; lean_object* v___x_2911_; uint8_t v_isShared_2912_; uint8_t v_isSharedCheck_2924_; 
v_offset_2909_ = lean_ctor_get(v_a_2875_, 1);
v_isSharedCheck_2924_ = !lean_is_exclusive(v_a_2875_);
if (v_isSharedCheck_2924_ == 0)
{
lean_object* v_unused_2925_; 
v_unused_2925_ = lean_ctor_get(v_a_2875_, 0);
lean_dec(v_unused_2925_);
v___x_2911_ = v_a_2875_;
v_isShared_2912_ = v_isSharedCheck_2924_;
goto v_resetjp_2910_;
}
else
{
lean_inc(v_offset_2909_);
lean_dec(v_a_2875_);
v___x_2911_ = lean_box(0);
v_isShared_2912_ = v_isSharedCheck_2924_;
goto v_resetjp_2910_;
}
v_resetjp_2910_:
{
lean_object* v___x_2913_; uint8_t v___x_2914_; lean_object* v___x_2915_; lean_object* v___x_2916_; lean_object* v___x_2917_; lean_object* v___x_2919_; 
v___x_2913_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readSymbolicExpressionNode___closed__0));
v___x_2914_ = lean_unbox(v_a_2877_);
lean_dec(v_a_2877_);
v___x_2915_ = lean_uint8_to_nat(v___x_2914_);
v___x_2916_ = l_Nat_reprFast(v___x_2915_);
v___x_2917_ = lean_string_append(v___x_2913_, v___x_2916_);
lean_dec_ref(v___x_2916_);
if (v_isShared_2912_ == 0)
{
lean_ctor_set_tag(v___x_2911_, 3);
lean_ctor_set(v___x_2911_, 1, v___x_2917_);
lean_ctor_set(v___x_2911_, 0, v_offset_2909_);
v___x_2919_ = v___x_2911_;
goto v_reusejp_2918_;
}
else
{
lean_object* v_reuseFailAlloc_2923_; 
v_reuseFailAlloc_2923_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2923_, 0, v_offset_2909_);
lean_ctor_set(v_reuseFailAlloc_2923_, 1, v___x_2917_);
v___x_2919_ = v_reuseFailAlloc_2923_;
goto v_reusejp_2918_;
}
v_reusejp_2918_:
{
lean_object* v___x_2921_; 
if (v_isShared_2881_ == 0)
{
lean_ctor_set_tag(v___x_2880_, 1);
lean_ctor_set(v___x_2880_, 0, v___x_2919_);
v___x_2921_ = v___x_2880_;
goto v_reusejp_2920_;
}
else
{
lean_object* v_reuseFailAlloc_2922_; 
v_reuseFailAlloc_2922_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2922_, 0, v___x_2919_);
lean_ctor_set(v_reuseFailAlloc_2922_, 1, v_a_2878_);
v___x_2921_ = v_reuseFailAlloc_2922_;
goto v_reusejp_2920_;
}
v_reusejp_2920_:
{
return v___x_2921_;
}
}
}
}
else
{
lean_object* v___x_2926_; 
lean_del_object(v___x_2880_);
lean_dec(v_a_2877_);
lean_dec_ref(v_a_2875_);
v___x_2926_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_2878_);
if (lean_obj_tag(v___x_2926_) == 0)
{
lean_object* v_a_2927_; lean_object* v_a_2928_; lean_object* v___x_2929_; 
v_a_2927_ = lean_ctor_get(v___x_2926_, 0);
lean_inc(v_a_2927_);
v_a_2928_ = lean_ctor_get(v___x_2926_, 1);
lean_inc(v_a_2928_);
lean_dec_ref_known(v___x_2926_, 2);
v___x_2929_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_2928_);
if (lean_obj_tag(v___x_2929_) == 0)
{
lean_object* v_a_2930_; lean_object* v_a_2931_; lean_object* v___x_2932_; 
v_a_2930_ = lean_ctor_get(v___x_2929_, 0);
lean_inc(v_a_2930_);
v_a_2931_ = lean_ctor_get(v___x_2929_, 1);
lean_inc(v_a_2931_);
lean_dec_ref_known(v___x_2929_, 2);
v___x_2932_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt64(v_a_2931_);
if (lean_obj_tag(v___x_2932_) == 0)
{
lean_object* v_a_2933_; lean_object* v_a_2934_; lean_object* v___x_2936_; uint8_t v_isShared_2937_; uint8_t v_isSharedCheck_2945_; 
v_a_2933_ = lean_ctor_get(v___x_2932_, 0);
v_a_2934_ = lean_ctor_get(v___x_2932_, 1);
v_isSharedCheck_2945_ = !lean_is_exclusive(v___x_2932_);
if (v_isSharedCheck_2945_ == 0)
{
v___x_2936_ = v___x_2932_;
v_isShared_2937_ = v_isSharedCheck_2945_;
goto v_resetjp_2935_;
}
else
{
lean_inc(v_a_2934_);
lean_inc(v_a_2933_);
lean_dec(v___x_2932_);
v___x_2936_ = lean_box(0);
v_isShared_2937_ = v_isSharedCheck_2945_;
goto v_resetjp_2935_;
}
v_resetjp_2935_:
{
lean_object* v___x_2938_; uint32_t v___x_2939_; uint32_t v___x_2940_; uint64_t v___x_2941_; lean_object* v___x_2943_; 
v___x_2938_ = lean_alloc_ctor(8, 0, 16);
v___x_2939_ = lean_unbox_uint32(v_a_2927_);
lean_dec(v_a_2927_);
lean_ctor_set_uint32(v___x_2938_, 8, v___x_2939_);
v___x_2940_ = lean_unbox_uint32(v_a_2930_);
lean_dec(v_a_2930_);
lean_ctor_set_uint32(v___x_2938_, 12, v___x_2940_);
v___x_2941_ = lean_unbox_uint64(v_a_2933_);
lean_dec(v_a_2933_);
lean_ctor_set_uint64(v___x_2938_, 0, v___x_2941_);
if (v_isShared_2937_ == 0)
{
lean_ctor_set(v___x_2936_, 0, v___x_2938_);
v___x_2943_ = v___x_2936_;
goto v_reusejp_2942_;
}
else
{
lean_object* v_reuseFailAlloc_2944_; 
v_reuseFailAlloc_2944_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2944_, 0, v___x_2938_);
lean_ctor_set(v_reuseFailAlloc_2944_, 1, v_a_2934_);
v___x_2943_ = v_reuseFailAlloc_2944_;
goto v_reusejp_2942_;
}
v_reusejp_2942_:
{
return v___x_2943_;
}
}
}
else
{
lean_object* v_a_2946_; lean_object* v_a_2947_; lean_object* v___x_2949_; uint8_t v_isShared_2950_; uint8_t v_isSharedCheck_2954_; 
lean_dec(v_a_2930_);
lean_dec(v_a_2927_);
v_a_2946_ = lean_ctor_get(v___x_2932_, 0);
v_a_2947_ = lean_ctor_get(v___x_2932_, 1);
v_isSharedCheck_2954_ = !lean_is_exclusive(v___x_2932_);
if (v_isSharedCheck_2954_ == 0)
{
v___x_2949_ = v___x_2932_;
v_isShared_2950_ = v_isSharedCheck_2954_;
goto v_resetjp_2948_;
}
else
{
lean_inc(v_a_2947_);
lean_inc(v_a_2946_);
lean_dec(v___x_2932_);
v___x_2949_ = lean_box(0);
v_isShared_2950_ = v_isSharedCheck_2954_;
goto v_resetjp_2948_;
}
v_resetjp_2948_:
{
lean_object* v___x_2952_; 
if (v_isShared_2950_ == 0)
{
v___x_2952_ = v___x_2949_;
goto v_reusejp_2951_;
}
else
{
lean_object* v_reuseFailAlloc_2953_; 
v_reuseFailAlloc_2953_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2953_, 0, v_a_2946_);
lean_ctor_set(v_reuseFailAlloc_2953_, 1, v_a_2947_);
v___x_2952_ = v_reuseFailAlloc_2953_;
goto v_reusejp_2951_;
}
v_reusejp_2951_:
{
return v___x_2952_;
}
}
}
}
else
{
lean_object* v_a_2955_; lean_object* v_a_2956_; lean_object* v___x_2958_; uint8_t v_isShared_2959_; uint8_t v_isSharedCheck_2963_; 
lean_dec(v_a_2927_);
v_a_2955_ = lean_ctor_get(v___x_2929_, 0);
v_a_2956_ = lean_ctor_get(v___x_2929_, 1);
v_isSharedCheck_2963_ = !lean_is_exclusive(v___x_2929_);
if (v_isSharedCheck_2963_ == 0)
{
v___x_2958_ = v___x_2929_;
v_isShared_2959_ = v_isSharedCheck_2963_;
goto v_resetjp_2957_;
}
else
{
lean_inc(v_a_2956_);
lean_inc(v_a_2955_);
lean_dec(v___x_2929_);
v___x_2958_ = lean_box(0);
v_isShared_2959_ = v_isSharedCheck_2963_;
goto v_resetjp_2957_;
}
v_resetjp_2957_:
{
lean_object* v___x_2961_; 
if (v_isShared_2959_ == 0)
{
v___x_2961_ = v___x_2958_;
goto v_reusejp_2960_;
}
else
{
lean_object* v_reuseFailAlloc_2962_; 
v_reuseFailAlloc_2962_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2962_, 0, v_a_2955_);
lean_ctor_set(v_reuseFailAlloc_2962_, 1, v_a_2956_);
v___x_2961_ = v_reuseFailAlloc_2962_;
goto v_reusejp_2960_;
}
v_reusejp_2960_:
{
return v___x_2961_;
}
}
}
}
else
{
lean_object* v_a_2964_; lean_object* v_a_2965_; lean_object* v___x_2967_; uint8_t v_isShared_2968_; uint8_t v_isSharedCheck_2972_; 
v_a_2964_ = lean_ctor_get(v___x_2926_, 0);
v_a_2965_ = lean_ctor_get(v___x_2926_, 1);
v_isSharedCheck_2972_ = !lean_is_exclusive(v___x_2926_);
if (v_isSharedCheck_2972_ == 0)
{
v___x_2967_ = v___x_2926_;
v_isShared_2968_ = v_isSharedCheck_2972_;
goto v_resetjp_2966_;
}
else
{
lean_inc(v_a_2965_);
lean_inc(v_a_2964_);
lean_dec(v___x_2926_);
v___x_2967_ = lean_box(0);
v_isShared_2968_ = v_isSharedCheck_2972_;
goto v_resetjp_2966_;
}
v_resetjp_2966_:
{
lean_object* v___x_2970_; 
if (v_isShared_2968_ == 0)
{
v___x_2970_ = v___x_2967_;
goto v_reusejp_2969_;
}
else
{
lean_object* v_reuseFailAlloc_2971_; 
v_reuseFailAlloc_2971_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2971_, 0, v_a_2964_);
lean_ctor_set(v_reuseFailAlloc_2971_, 1, v_a_2965_);
v___x_2970_ = v_reuseFailAlloc_2971_;
goto v_reusejp_2969_;
}
v_reusejp_2969_:
{
return v___x_2970_;
}
}
}
}
}
else
{
lean_object* v___x_2973_; 
lean_del_object(v___x_2880_);
lean_dec(v_a_2877_);
lean_dec_ref(v_a_2875_);
v___x_2973_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_2878_);
if (lean_obj_tag(v___x_2973_) == 0)
{
lean_object* v_a_2974_; lean_object* v_a_2975_; lean_object* v___x_2976_; 
v_a_2974_ = lean_ctor_get(v___x_2973_, 0);
lean_inc(v_a_2974_);
v_a_2975_ = lean_ctor_get(v___x_2973_, 1);
lean_inc(v_a_2975_);
lean_dec_ref_known(v___x_2973_, 2);
v___x_2976_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt64(v_a_2975_);
if (lean_obj_tag(v___x_2976_) == 0)
{
lean_object* v_a_2977_; lean_object* v_a_2978_; lean_object* v___x_2980_; uint8_t v_isShared_2981_; uint8_t v_isSharedCheck_2988_; 
v_a_2977_ = lean_ctor_get(v___x_2976_, 0);
v_a_2978_ = lean_ctor_get(v___x_2976_, 1);
v_isSharedCheck_2988_ = !lean_is_exclusive(v___x_2976_);
if (v_isSharedCheck_2988_ == 0)
{
v___x_2980_ = v___x_2976_;
v_isShared_2981_ = v_isSharedCheck_2988_;
goto v_resetjp_2979_;
}
else
{
lean_inc(v_a_2978_);
lean_inc(v_a_2977_);
lean_dec(v___x_2976_);
v___x_2980_ = lean_box(0);
v_isShared_2981_ = v_isSharedCheck_2988_;
goto v_resetjp_2979_;
}
v_resetjp_2979_:
{
lean_object* v___x_2982_; uint32_t v___x_2983_; uint64_t v___x_2984_; lean_object* v___x_2986_; 
v___x_2982_ = lean_alloc_ctor(7, 0, 12);
v___x_2983_ = lean_unbox_uint32(v_a_2974_);
lean_dec(v_a_2974_);
lean_ctor_set_uint32(v___x_2982_, 8, v___x_2983_);
v___x_2984_ = lean_unbox_uint64(v_a_2977_);
lean_dec(v_a_2977_);
lean_ctor_set_uint64(v___x_2982_, 0, v___x_2984_);
if (v_isShared_2981_ == 0)
{
lean_ctor_set(v___x_2980_, 0, v___x_2982_);
v___x_2986_ = v___x_2980_;
goto v_reusejp_2985_;
}
else
{
lean_object* v_reuseFailAlloc_2987_; 
v_reuseFailAlloc_2987_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2987_, 0, v___x_2982_);
lean_ctor_set(v_reuseFailAlloc_2987_, 1, v_a_2978_);
v___x_2986_ = v_reuseFailAlloc_2987_;
goto v_reusejp_2985_;
}
v_reusejp_2985_:
{
return v___x_2986_;
}
}
}
else
{
lean_object* v_a_2989_; lean_object* v_a_2990_; lean_object* v___x_2992_; uint8_t v_isShared_2993_; uint8_t v_isSharedCheck_2997_; 
lean_dec(v_a_2974_);
v_a_2989_ = lean_ctor_get(v___x_2976_, 0);
v_a_2990_ = lean_ctor_get(v___x_2976_, 1);
v_isSharedCheck_2997_ = !lean_is_exclusive(v___x_2976_);
if (v_isSharedCheck_2997_ == 0)
{
v___x_2992_ = v___x_2976_;
v_isShared_2993_ = v_isSharedCheck_2997_;
goto v_resetjp_2991_;
}
else
{
lean_inc(v_a_2990_);
lean_inc(v_a_2989_);
lean_dec(v___x_2976_);
v___x_2992_ = lean_box(0);
v_isShared_2993_ = v_isSharedCheck_2997_;
goto v_resetjp_2991_;
}
v_resetjp_2991_:
{
lean_object* v___x_2995_; 
if (v_isShared_2993_ == 0)
{
v___x_2995_ = v___x_2992_;
goto v_reusejp_2994_;
}
else
{
lean_object* v_reuseFailAlloc_2996_; 
v_reuseFailAlloc_2996_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2996_, 0, v_a_2989_);
lean_ctor_set(v_reuseFailAlloc_2996_, 1, v_a_2990_);
v___x_2995_ = v_reuseFailAlloc_2996_;
goto v_reusejp_2994_;
}
v_reusejp_2994_:
{
return v___x_2995_;
}
}
}
}
else
{
lean_object* v_a_2998_; lean_object* v_a_2999_; lean_object* v___x_3001_; uint8_t v_isShared_3002_; uint8_t v_isSharedCheck_3006_; 
v_a_2998_ = lean_ctor_get(v___x_2973_, 0);
v_a_2999_ = lean_ctor_get(v___x_2973_, 1);
v_isSharedCheck_3006_ = !lean_is_exclusive(v___x_2973_);
if (v_isSharedCheck_3006_ == 0)
{
v___x_3001_ = v___x_2973_;
v_isShared_3002_ = v_isSharedCheck_3006_;
goto v_resetjp_3000_;
}
else
{
lean_inc(v_a_2999_);
lean_inc(v_a_2998_);
lean_dec(v___x_2973_);
v___x_3001_ = lean_box(0);
v_isShared_3002_ = v_isSharedCheck_3006_;
goto v_resetjp_3000_;
}
v_resetjp_3000_:
{
lean_object* v___x_3004_; 
if (v_isShared_3002_ == 0)
{
v___x_3004_ = v___x_3001_;
goto v_reusejp_3003_;
}
else
{
lean_object* v_reuseFailAlloc_3005_; 
v_reuseFailAlloc_3005_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3005_, 0, v_a_2998_);
lean_ctor_set(v_reuseFailAlloc_3005_, 1, v_a_2999_);
v___x_3004_ = v_reuseFailAlloc_3005_;
goto v_reusejp_3003_;
}
v_reusejp_3003_:
{
return v___x_3004_;
}
}
}
}
}
else
{
lean_object* v___x_3007_; 
lean_del_object(v___x_2880_);
lean_dec(v_a_2877_);
lean_dec_ref(v_a_2875_);
v___x_3007_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_2878_);
if (lean_obj_tag(v___x_3007_) == 0)
{
lean_object* v_a_3008_; lean_object* v_a_3009_; lean_object* v___x_3010_; 
v_a_3008_ = lean_ctor_get(v___x_3007_, 0);
lean_inc(v_a_3008_);
v_a_3009_ = lean_ctor_get(v___x_3007_, 1);
lean_inc(v_a_3009_);
lean_dec_ref_known(v___x_3007_, 2);
v___x_3010_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_3009_);
if (lean_obj_tag(v___x_3010_) == 0)
{
lean_object* v_a_3011_; lean_object* v_a_3012_; lean_object* v___x_3013_; 
v_a_3011_ = lean_ctor_get(v___x_3010_, 0);
lean_inc(v_a_3011_);
v_a_3012_ = lean_ctor_get(v___x_3010_, 1);
lean_inc(v_a_3012_);
lean_dec_ref_known(v___x_3010_, 2);
v___x_3013_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt64(v_a_3012_);
if (lean_obj_tag(v___x_3013_) == 0)
{
lean_object* v_a_3014_; lean_object* v_a_3015_; lean_object* v___x_3017_; uint8_t v_isShared_3018_; uint8_t v_isSharedCheck_3026_; 
v_a_3014_ = lean_ctor_get(v___x_3013_, 0);
v_a_3015_ = lean_ctor_get(v___x_3013_, 1);
v_isSharedCheck_3026_ = !lean_is_exclusive(v___x_3013_);
if (v_isSharedCheck_3026_ == 0)
{
v___x_3017_ = v___x_3013_;
v_isShared_3018_ = v_isSharedCheck_3026_;
goto v_resetjp_3016_;
}
else
{
lean_inc(v_a_3015_);
lean_inc(v_a_3014_);
lean_dec(v___x_3013_);
v___x_3017_ = lean_box(0);
v_isShared_3018_ = v_isSharedCheck_3026_;
goto v_resetjp_3016_;
}
v_resetjp_3016_:
{
lean_object* v___x_3019_; uint32_t v___x_3020_; uint32_t v___x_3021_; uint64_t v___x_3022_; lean_object* v___x_3024_; 
v___x_3019_ = lean_alloc_ctor(6, 0, 16);
v___x_3020_ = lean_unbox_uint32(v_a_3008_);
lean_dec(v_a_3008_);
lean_ctor_set_uint32(v___x_3019_, 8, v___x_3020_);
v___x_3021_ = lean_unbox_uint32(v_a_3011_);
lean_dec(v_a_3011_);
lean_ctor_set_uint32(v___x_3019_, 12, v___x_3021_);
v___x_3022_ = lean_unbox_uint64(v_a_3014_);
lean_dec(v_a_3014_);
lean_ctor_set_uint64(v___x_3019_, 0, v___x_3022_);
if (v_isShared_3018_ == 0)
{
lean_ctor_set(v___x_3017_, 0, v___x_3019_);
v___x_3024_ = v___x_3017_;
goto v_reusejp_3023_;
}
else
{
lean_object* v_reuseFailAlloc_3025_; 
v_reuseFailAlloc_3025_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3025_, 0, v___x_3019_);
lean_ctor_set(v_reuseFailAlloc_3025_, 1, v_a_3015_);
v___x_3024_ = v_reuseFailAlloc_3025_;
goto v_reusejp_3023_;
}
v_reusejp_3023_:
{
return v___x_3024_;
}
}
}
else
{
lean_object* v_a_3027_; lean_object* v_a_3028_; lean_object* v___x_3030_; uint8_t v_isShared_3031_; uint8_t v_isSharedCheck_3035_; 
lean_dec(v_a_3011_);
lean_dec(v_a_3008_);
v_a_3027_ = lean_ctor_get(v___x_3013_, 0);
v_a_3028_ = lean_ctor_get(v___x_3013_, 1);
v_isSharedCheck_3035_ = !lean_is_exclusive(v___x_3013_);
if (v_isSharedCheck_3035_ == 0)
{
v___x_3030_ = v___x_3013_;
v_isShared_3031_ = v_isSharedCheck_3035_;
goto v_resetjp_3029_;
}
else
{
lean_inc(v_a_3028_);
lean_inc(v_a_3027_);
lean_dec(v___x_3013_);
v___x_3030_ = lean_box(0);
v_isShared_3031_ = v_isSharedCheck_3035_;
goto v_resetjp_3029_;
}
v_resetjp_3029_:
{
lean_object* v___x_3033_; 
if (v_isShared_3031_ == 0)
{
v___x_3033_ = v___x_3030_;
goto v_reusejp_3032_;
}
else
{
lean_object* v_reuseFailAlloc_3034_; 
v_reuseFailAlloc_3034_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3034_, 0, v_a_3027_);
lean_ctor_set(v_reuseFailAlloc_3034_, 1, v_a_3028_);
v___x_3033_ = v_reuseFailAlloc_3034_;
goto v_reusejp_3032_;
}
v_reusejp_3032_:
{
return v___x_3033_;
}
}
}
}
else
{
lean_object* v_a_3036_; lean_object* v_a_3037_; lean_object* v___x_3039_; uint8_t v_isShared_3040_; uint8_t v_isSharedCheck_3044_; 
lean_dec(v_a_3008_);
v_a_3036_ = lean_ctor_get(v___x_3010_, 0);
v_a_3037_ = lean_ctor_get(v___x_3010_, 1);
v_isSharedCheck_3044_ = !lean_is_exclusive(v___x_3010_);
if (v_isSharedCheck_3044_ == 0)
{
v___x_3039_ = v___x_3010_;
v_isShared_3040_ = v_isSharedCheck_3044_;
goto v_resetjp_3038_;
}
else
{
lean_inc(v_a_3037_);
lean_inc(v_a_3036_);
lean_dec(v___x_3010_);
v___x_3039_ = lean_box(0);
v_isShared_3040_ = v_isSharedCheck_3044_;
goto v_resetjp_3038_;
}
v_resetjp_3038_:
{
lean_object* v___x_3042_; 
if (v_isShared_3040_ == 0)
{
v___x_3042_ = v___x_3039_;
goto v_reusejp_3041_;
}
else
{
lean_object* v_reuseFailAlloc_3043_; 
v_reuseFailAlloc_3043_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3043_, 0, v_a_3036_);
lean_ctor_set(v_reuseFailAlloc_3043_, 1, v_a_3037_);
v___x_3042_ = v_reuseFailAlloc_3043_;
goto v_reusejp_3041_;
}
v_reusejp_3041_:
{
return v___x_3042_;
}
}
}
}
else
{
lean_object* v_a_3045_; lean_object* v_a_3046_; lean_object* v___x_3048_; uint8_t v_isShared_3049_; uint8_t v_isSharedCheck_3053_; 
v_a_3045_ = lean_ctor_get(v___x_3007_, 0);
v_a_3046_ = lean_ctor_get(v___x_3007_, 1);
v_isSharedCheck_3053_ = !lean_is_exclusive(v___x_3007_);
if (v_isSharedCheck_3053_ == 0)
{
v___x_3048_ = v___x_3007_;
v_isShared_3049_ = v_isSharedCheck_3053_;
goto v_resetjp_3047_;
}
else
{
lean_inc(v_a_3046_);
lean_inc(v_a_3045_);
lean_dec(v___x_3007_);
v___x_3048_ = lean_box(0);
v_isShared_3049_ = v_isSharedCheck_3053_;
goto v_resetjp_3047_;
}
v_resetjp_3047_:
{
lean_object* v___x_3051_; 
if (v_isShared_3049_ == 0)
{
v___x_3051_ = v___x_3048_;
goto v_reusejp_3050_;
}
else
{
lean_object* v_reuseFailAlloc_3052_; 
v_reuseFailAlloc_3052_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3052_, 0, v_a_3045_);
lean_ctor_set(v_reuseFailAlloc_3052_, 1, v_a_3046_);
v___x_3051_ = v_reuseFailAlloc_3052_;
goto v_reusejp_3050_;
}
v_reusejp_3050_:
{
return v___x_3051_;
}
}
}
}
}
else
{
lean_object* v___x_3054_; 
lean_del_object(v___x_2880_);
lean_dec(v_a_2877_);
lean_dec_ref(v_a_2875_);
v___x_3054_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_2878_);
if (lean_obj_tag(v___x_3054_) == 0)
{
lean_object* v_a_3055_; lean_object* v_a_3056_; lean_object* v___x_3057_; 
v_a_3055_ = lean_ctor_get(v___x_3054_, 0);
lean_inc(v_a_3055_);
v_a_3056_ = lean_ctor_get(v___x_3054_, 1);
lean_inc(v_a_3056_);
lean_dec_ref_known(v___x_3054_, 2);
v___x_3057_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_3056_);
if (lean_obj_tag(v___x_3057_) == 0)
{
lean_object* v_a_3058_; lean_object* v_a_3059_; lean_object* v___x_3060_; 
v_a_3058_ = lean_ctor_get(v___x_3057_, 0);
lean_inc(v_a_3058_);
v_a_3059_ = lean_ctor_get(v___x_3057_, 1);
lean_inc(v_a_3059_);
lean_dec_ref_known(v___x_3057_, 2);
v___x_3060_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt64(v_a_3059_);
if (lean_obj_tag(v___x_3060_) == 0)
{
lean_object* v_a_3061_; lean_object* v_a_3062_; lean_object* v___x_3064_; uint8_t v_isShared_3065_; uint8_t v_isSharedCheck_3073_; 
v_a_3061_ = lean_ctor_get(v___x_3060_, 0);
v_a_3062_ = lean_ctor_get(v___x_3060_, 1);
v_isSharedCheck_3073_ = !lean_is_exclusive(v___x_3060_);
if (v_isSharedCheck_3073_ == 0)
{
v___x_3064_ = v___x_3060_;
v_isShared_3065_ = v_isSharedCheck_3073_;
goto v_resetjp_3063_;
}
else
{
lean_inc(v_a_3062_);
lean_inc(v_a_3061_);
lean_dec(v___x_3060_);
v___x_3064_ = lean_box(0);
v_isShared_3065_ = v_isSharedCheck_3073_;
goto v_resetjp_3063_;
}
v_resetjp_3063_:
{
lean_object* v___x_3066_; uint32_t v___x_3067_; uint32_t v___x_3068_; uint64_t v___x_3069_; lean_object* v___x_3071_; 
v___x_3066_ = lean_alloc_ctor(5, 0, 16);
v___x_3067_ = lean_unbox_uint32(v_a_3055_);
lean_dec(v_a_3055_);
lean_ctor_set_uint32(v___x_3066_, 8, v___x_3067_);
v___x_3068_ = lean_unbox_uint32(v_a_3058_);
lean_dec(v_a_3058_);
lean_ctor_set_uint32(v___x_3066_, 12, v___x_3068_);
v___x_3069_ = lean_unbox_uint64(v_a_3061_);
lean_dec(v_a_3061_);
lean_ctor_set_uint64(v___x_3066_, 0, v___x_3069_);
if (v_isShared_3065_ == 0)
{
lean_ctor_set(v___x_3064_, 0, v___x_3066_);
v___x_3071_ = v___x_3064_;
goto v_reusejp_3070_;
}
else
{
lean_object* v_reuseFailAlloc_3072_; 
v_reuseFailAlloc_3072_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3072_, 0, v___x_3066_);
lean_ctor_set(v_reuseFailAlloc_3072_, 1, v_a_3062_);
v___x_3071_ = v_reuseFailAlloc_3072_;
goto v_reusejp_3070_;
}
v_reusejp_3070_:
{
return v___x_3071_;
}
}
}
else
{
lean_object* v_a_3074_; lean_object* v_a_3075_; lean_object* v___x_3077_; uint8_t v_isShared_3078_; uint8_t v_isSharedCheck_3082_; 
lean_dec(v_a_3058_);
lean_dec(v_a_3055_);
v_a_3074_ = lean_ctor_get(v___x_3060_, 0);
v_a_3075_ = lean_ctor_get(v___x_3060_, 1);
v_isSharedCheck_3082_ = !lean_is_exclusive(v___x_3060_);
if (v_isSharedCheck_3082_ == 0)
{
v___x_3077_ = v___x_3060_;
v_isShared_3078_ = v_isSharedCheck_3082_;
goto v_resetjp_3076_;
}
else
{
lean_inc(v_a_3075_);
lean_inc(v_a_3074_);
lean_dec(v___x_3060_);
v___x_3077_ = lean_box(0);
v_isShared_3078_ = v_isSharedCheck_3082_;
goto v_resetjp_3076_;
}
v_resetjp_3076_:
{
lean_object* v___x_3080_; 
if (v_isShared_3078_ == 0)
{
v___x_3080_ = v___x_3077_;
goto v_reusejp_3079_;
}
else
{
lean_object* v_reuseFailAlloc_3081_; 
v_reuseFailAlloc_3081_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3081_, 0, v_a_3074_);
lean_ctor_set(v_reuseFailAlloc_3081_, 1, v_a_3075_);
v___x_3080_ = v_reuseFailAlloc_3081_;
goto v_reusejp_3079_;
}
v_reusejp_3079_:
{
return v___x_3080_;
}
}
}
}
else
{
lean_object* v_a_3083_; lean_object* v_a_3084_; lean_object* v___x_3086_; uint8_t v_isShared_3087_; uint8_t v_isSharedCheck_3091_; 
lean_dec(v_a_3055_);
v_a_3083_ = lean_ctor_get(v___x_3057_, 0);
v_a_3084_ = lean_ctor_get(v___x_3057_, 1);
v_isSharedCheck_3091_ = !lean_is_exclusive(v___x_3057_);
if (v_isSharedCheck_3091_ == 0)
{
v___x_3086_ = v___x_3057_;
v_isShared_3087_ = v_isSharedCheck_3091_;
goto v_resetjp_3085_;
}
else
{
lean_inc(v_a_3084_);
lean_inc(v_a_3083_);
lean_dec(v___x_3057_);
v___x_3086_ = lean_box(0);
v_isShared_3087_ = v_isSharedCheck_3091_;
goto v_resetjp_3085_;
}
v_resetjp_3085_:
{
lean_object* v___x_3089_; 
if (v_isShared_3087_ == 0)
{
v___x_3089_ = v___x_3086_;
goto v_reusejp_3088_;
}
else
{
lean_object* v_reuseFailAlloc_3090_; 
v_reuseFailAlloc_3090_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3090_, 0, v_a_3083_);
lean_ctor_set(v_reuseFailAlloc_3090_, 1, v_a_3084_);
v___x_3089_ = v_reuseFailAlloc_3090_;
goto v_reusejp_3088_;
}
v_reusejp_3088_:
{
return v___x_3089_;
}
}
}
}
else
{
lean_object* v_a_3092_; lean_object* v_a_3093_; lean_object* v___x_3095_; uint8_t v_isShared_3096_; uint8_t v_isSharedCheck_3100_; 
v_a_3092_ = lean_ctor_get(v___x_3054_, 0);
v_a_3093_ = lean_ctor_get(v___x_3054_, 1);
v_isSharedCheck_3100_ = !lean_is_exclusive(v___x_3054_);
if (v_isSharedCheck_3100_ == 0)
{
v___x_3095_ = v___x_3054_;
v_isShared_3096_ = v_isSharedCheck_3100_;
goto v_resetjp_3094_;
}
else
{
lean_inc(v_a_3093_);
lean_inc(v_a_3092_);
lean_dec(v___x_3054_);
v___x_3095_ = lean_box(0);
v_isShared_3096_ = v_isSharedCheck_3100_;
goto v_resetjp_3094_;
}
v_resetjp_3094_:
{
lean_object* v___x_3098_; 
if (v_isShared_3096_ == 0)
{
v___x_3098_ = v___x_3095_;
goto v_reusejp_3097_;
}
else
{
lean_object* v_reuseFailAlloc_3099_; 
v_reuseFailAlloc_3099_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3099_, 0, v_a_3092_);
lean_ctor_set(v_reuseFailAlloc_3099_, 1, v_a_3093_);
v___x_3098_ = v_reuseFailAlloc_3099_;
goto v_reusejp_3097_;
}
v_reusejp_3097_:
{
return v___x_3098_;
}
}
}
}
}
else
{
lean_object* v___x_3101_; 
lean_del_object(v___x_2880_);
lean_dec(v_a_2877_);
lean_dec_ref(v_a_2875_);
v___x_3101_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readCanonicalFBB(v_a_2878_);
if (lean_obj_tag(v___x_3101_) == 0)
{
lean_object* v_a_3102_; lean_object* v_a_3103_; lean_object* v___x_3105_; uint8_t v_isShared_3106_; uint8_t v_isSharedCheck_3112_; 
v_a_3102_ = lean_ctor_get(v___x_3101_, 0);
v_a_3103_ = lean_ctor_get(v___x_3101_, 1);
v_isSharedCheck_3112_ = !lean_is_exclusive(v___x_3101_);
if (v_isSharedCheck_3112_ == 0)
{
v___x_3105_ = v___x_3101_;
v_isShared_3106_ = v_isSharedCheck_3112_;
goto v_resetjp_3104_;
}
else
{
lean_inc(v_a_3103_);
lean_inc(v_a_3102_);
lean_dec(v___x_3101_);
v___x_3105_ = lean_box(0);
v_isShared_3106_ = v_isSharedCheck_3112_;
goto v_resetjp_3104_;
}
v_resetjp_3104_:
{
lean_object* v___x_3107_; uint32_t v___x_3108_; lean_object* v___x_3110_; 
v___x_3107_ = lean_alloc_ctor(4, 0, 4);
v___x_3108_ = lean_unbox_uint32(v_a_3102_);
lean_dec(v_a_3102_);
lean_ctor_set_uint32(v___x_3107_, 0, v___x_3108_);
if (v_isShared_3106_ == 0)
{
lean_ctor_set(v___x_3105_, 0, v___x_3107_);
v___x_3110_ = v___x_3105_;
goto v_reusejp_3109_;
}
else
{
lean_object* v_reuseFailAlloc_3111_; 
v_reuseFailAlloc_3111_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3111_, 0, v___x_3107_);
lean_ctor_set(v_reuseFailAlloc_3111_, 1, v_a_3103_);
v___x_3110_ = v_reuseFailAlloc_3111_;
goto v_reusejp_3109_;
}
v_reusejp_3109_:
{
return v___x_3110_;
}
}
}
else
{
lean_object* v_a_3113_; lean_object* v_a_3114_; lean_object* v___x_3116_; uint8_t v_isShared_3117_; uint8_t v_isSharedCheck_3121_; 
v_a_3113_ = lean_ctor_get(v___x_3101_, 0);
v_a_3114_ = lean_ctor_get(v___x_3101_, 1);
v_isSharedCheck_3121_ = !lean_is_exclusive(v___x_3101_);
if (v_isSharedCheck_3121_ == 0)
{
v___x_3116_ = v___x_3101_;
v_isShared_3117_ = v_isSharedCheck_3121_;
goto v_resetjp_3115_;
}
else
{
lean_inc(v_a_3114_);
lean_inc(v_a_3113_);
lean_dec(v___x_3101_);
v___x_3116_ = lean_box(0);
v_isShared_3117_ = v_isSharedCheck_3121_;
goto v_resetjp_3115_;
}
v_resetjp_3115_:
{
lean_object* v___x_3119_; 
if (v_isShared_3117_ == 0)
{
v___x_3119_ = v___x_3116_;
goto v_reusejp_3118_;
}
else
{
lean_object* v_reuseFailAlloc_3120_; 
v_reuseFailAlloc_3120_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3120_, 0, v_a_3113_);
lean_ctor_set(v_reuseFailAlloc_3120_, 1, v_a_3114_);
v___x_3119_ = v_reuseFailAlloc_3120_;
goto v_reusejp_3118_;
}
v_reusejp_3118_:
{
return v___x_3119_;
}
}
}
}
}
else
{
lean_object* v___x_3122_; lean_object* v___x_3124_; 
lean_dec(v_a_2877_);
lean_dec_ref(v_a_2875_);
v___x_3122_ = lean_box(3);
if (v_isShared_2881_ == 0)
{
lean_ctor_set(v___x_2880_, 0, v___x_3122_);
v___x_3124_ = v___x_2880_;
goto v_reusejp_3123_;
}
else
{
lean_object* v_reuseFailAlloc_3125_; 
v_reuseFailAlloc_3125_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3125_, 0, v___x_3122_);
lean_ctor_set(v_reuseFailAlloc_3125_, 1, v_a_2878_);
v___x_3124_ = v_reuseFailAlloc_3125_;
goto v_reusejp_3123_;
}
v_reusejp_3123_:
{
return v___x_3124_;
}
}
}
else
{
lean_object* v___x_3126_; lean_object* v___x_3128_; 
lean_dec(v_a_2877_);
lean_dec_ref(v_a_2875_);
v___x_3126_ = lean_box(2);
if (v_isShared_2881_ == 0)
{
lean_ctor_set(v___x_2880_, 0, v___x_3126_);
v___x_3128_ = v___x_2880_;
goto v_reusejp_3127_;
}
else
{
lean_object* v_reuseFailAlloc_3129_; 
v_reuseFailAlloc_3129_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3129_, 0, v___x_3126_);
lean_ctor_set(v_reuseFailAlloc_3129_, 1, v_a_2878_);
v___x_3128_ = v_reuseFailAlloc_3129_;
goto v_reusejp_3127_;
}
v_reusejp_3127_:
{
return v___x_3128_;
}
}
}
else
{
lean_object* v___x_3130_; lean_object* v___x_3132_; 
lean_dec(v_a_2877_);
lean_dec_ref(v_a_2875_);
v___x_3130_ = lean_box(1);
if (v_isShared_2881_ == 0)
{
lean_ctor_set(v___x_2880_, 0, v___x_3130_);
v___x_3132_ = v___x_2880_;
goto v_reusejp_3131_;
}
else
{
lean_object* v_reuseFailAlloc_3133_; 
v_reuseFailAlloc_3133_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3133_, 0, v___x_3130_);
lean_ctor_set(v_reuseFailAlloc_3133_, 1, v_a_2878_);
v___x_3132_ = v_reuseFailAlloc_3133_;
goto v_reusejp_3131_;
}
v_reusejp_3131_:
{
return v___x_3132_;
}
}
}
else
{
lean_object* v___x_3134_; 
lean_del_object(v___x_2880_);
lean_dec(v_a_2877_);
lean_dec_ref(v_a_2875_);
v___x_3134_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readSymbolicVariable(v_a_2878_);
if (lean_obj_tag(v___x_3134_) == 0)
{
lean_object* v_a_3135_; lean_object* v_a_3136_; lean_object* v___x_3138_; uint8_t v_isShared_3139_; uint8_t v_isSharedCheck_3144_; 
v_a_3135_ = lean_ctor_get(v___x_3134_, 0);
v_a_3136_ = lean_ctor_get(v___x_3134_, 1);
v_isSharedCheck_3144_ = !lean_is_exclusive(v___x_3134_);
if (v_isSharedCheck_3144_ == 0)
{
v___x_3138_ = v___x_3134_;
v_isShared_3139_ = v_isSharedCheck_3144_;
goto v_resetjp_3137_;
}
else
{
lean_inc(v_a_3136_);
lean_inc(v_a_3135_);
lean_dec(v___x_3134_);
v___x_3138_ = lean_box(0);
v_isShared_3139_ = v_isSharedCheck_3144_;
goto v_resetjp_3137_;
}
v_resetjp_3137_:
{
lean_object* v___x_3140_; lean_object* v___x_3142_; 
v___x_3140_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3140_, 0, v_a_3135_);
if (v_isShared_3139_ == 0)
{
lean_ctor_set(v___x_3138_, 0, v___x_3140_);
v___x_3142_ = v___x_3138_;
goto v_reusejp_3141_;
}
else
{
lean_object* v_reuseFailAlloc_3143_; 
v_reuseFailAlloc_3143_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3143_, 0, v___x_3140_);
lean_ctor_set(v_reuseFailAlloc_3143_, 1, v_a_3136_);
v___x_3142_ = v_reuseFailAlloc_3143_;
goto v_reusejp_3141_;
}
v_reusejp_3141_:
{
return v___x_3142_;
}
}
}
else
{
lean_object* v_a_3145_; lean_object* v_a_3146_; lean_object* v___x_3148_; uint8_t v_isShared_3149_; uint8_t v_isSharedCheck_3153_; 
v_a_3145_ = lean_ctor_get(v___x_3134_, 0);
v_a_3146_ = lean_ctor_get(v___x_3134_, 1);
v_isSharedCheck_3153_ = !lean_is_exclusive(v___x_3134_);
if (v_isSharedCheck_3153_ == 0)
{
v___x_3148_ = v___x_3134_;
v_isShared_3149_ = v_isSharedCheck_3153_;
goto v_resetjp_3147_;
}
else
{
lean_inc(v_a_3146_);
lean_inc(v_a_3145_);
lean_dec(v___x_3134_);
v___x_3148_ = lean_box(0);
v_isShared_3149_ = v_isSharedCheck_3153_;
goto v_resetjp_3147_;
}
v_resetjp_3147_:
{
lean_object* v___x_3151_; 
if (v_isShared_3149_ == 0)
{
v___x_3151_ = v___x_3148_;
goto v_reusejp_3150_;
}
else
{
lean_object* v_reuseFailAlloc_3152_; 
v_reuseFailAlloc_3152_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3152_, 0, v_a_3145_);
lean_ctor_set(v_reuseFailAlloc_3152_, 1, v_a_3146_);
v___x_3151_ = v_reuseFailAlloc_3152_;
goto v_reusejp_3150_;
}
v_reusejp_3150_:
{
return v___x_3151_;
}
}
}
}
}
}
else
{
lean_object* v_a_3155_; lean_object* v_a_3156_; lean_object* v___x_3158_; uint8_t v_isShared_3159_; uint8_t v_isSharedCheck_3163_; 
lean_dec_ref(v_a_2875_);
v_a_3155_ = lean_ctor_get(v___x_2876_, 0);
v_a_3156_ = lean_ctor_get(v___x_2876_, 1);
v_isSharedCheck_3163_ = !lean_is_exclusive(v___x_2876_);
if (v_isSharedCheck_3163_ == 0)
{
v___x_3158_ = v___x_2876_;
v_isShared_3159_ = v_isSharedCheck_3163_;
goto v_resetjp_3157_;
}
else
{
lean_inc(v_a_3156_);
lean_inc(v_a_3155_);
lean_dec(v___x_2876_);
v___x_3158_ = lean_box(0);
v_isShared_3159_ = v_isSharedCheck_3163_;
goto v_resetjp_3157_;
}
v_resetjp_3157_:
{
lean_object* v___x_3161_; 
if (v_isShared_3159_ == 0)
{
v___x_3161_ = v___x_3158_;
goto v_reusejp_3160_;
}
else
{
lean_object* v_reuseFailAlloc_3162_; 
v_reuseFailAlloc_3162_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3162_, 0, v_a_3155_);
lean_ctor_set(v_reuseFailAlloc_3162_, 1, v_a_3156_);
v___x_3161_ = v_reuseFailAlloc_3162_;
goto v_reusejp_3160_;
}
v_reusejp_3160_:
{
return v___x_3161_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readSymbolicExpressionDag(lean_object* v_a_3164_){
_start:
{
lean_object* v___x_3165_; lean_object* v___x_3166_; 
v___x_3165_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readSymbolicExpressionNode), 1, 0);
v___x_3166_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_3165_, v_a_3164_);
if (lean_obj_tag(v___x_3166_) == 0)
{
lean_object* v_a_3167_; lean_object* v_a_3168_; lean_object* v___x_3169_; lean_object* v___x_3170_; 
v_a_3167_ = lean_ctor_get(v___x_3166_, 0);
lean_inc(v_a_3167_);
v_a_3168_ = lean_ctor_get(v___x_3166_, 1);
lean_inc(v_a_3168_);
lean_dec_ref_known(v___x_3166_, 2);
v___x_3169_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32), 1, 0);
v___x_3170_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_3169_, v_a_3168_);
if (lean_obj_tag(v___x_3170_) == 0)
{
lean_object* v_a_3171_; lean_object* v_a_3172_; lean_object* v___x_3174_; uint8_t v_isShared_3175_; uint8_t v_isSharedCheck_3180_; 
v_a_3171_ = lean_ctor_get(v___x_3170_, 0);
v_a_3172_ = lean_ctor_get(v___x_3170_, 1);
v_isSharedCheck_3180_ = !lean_is_exclusive(v___x_3170_);
if (v_isSharedCheck_3180_ == 0)
{
v___x_3174_ = v___x_3170_;
v_isShared_3175_ = v_isSharedCheck_3180_;
goto v_resetjp_3173_;
}
else
{
lean_inc(v_a_3172_);
lean_inc(v_a_3171_);
lean_dec(v___x_3170_);
v___x_3174_ = lean_box(0);
v_isShared_3175_ = v_isSharedCheck_3180_;
goto v_resetjp_3173_;
}
v_resetjp_3173_:
{
lean_object* v___x_3176_; lean_object* v___x_3178_; 
v___x_3176_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3176_, 0, v_a_3167_);
lean_ctor_set(v___x_3176_, 1, v_a_3171_);
if (v_isShared_3175_ == 0)
{
lean_ctor_set(v___x_3174_, 0, v___x_3176_);
v___x_3178_ = v___x_3174_;
goto v_reusejp_3177_;
}
else
{
lean_object* v_reuseFailAlloc_3179_; 
v_reuseFailAlloc_3179_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3179_, 0, v___x_3176_);
lean_ctor_set(v_reuseFailAlloc_3179_, 1, v_a_3172_);
v___x_3178_ = v_reuseFailAlloc_3179_;
goto v_reusejp_3177_;
}
v_reusejp_3177_:
{
return v___x_3178_;
}
}
}
else
{
lean_object* v_a_3181_; lean_object* v_a_3182_; lean_object* v___x_3184_; uint8_t v_isShared_3185_; uint8_t v_isSharedCheck_3189_; 
lean_dec(v_a_3167_);
v_a_3181_ = lean_ctor_get(v___x_3170_, 0);
v_a_3182_ = lean_ctor_get(v___x_3170_, 1);
v_isSharedCheck_3189_ = !lean_is_exclusive(v___x_3170_);
if (v_isSharedCheck_3189_ == 0)
{
v___x_3184_ = v___x_3170_;
v_isShared_3185_ = v_isSharedCheck_3189_;
goto v_resetjp_3183_;
}
else
{
lean_inc(v_a_3182_);
lean_inc(v_a_3181_);
lean_dec(v___x_3170_);
v___x_3184_ = lean_box(0);
v_isShared_3185_ = v_isSharedCheck_3189_;
goto v_resetjp_3183_;
}
v_resetjp_3183_:
{
lean_object* v___x_3187_; 
if (v_isShared_3185_ == 0)
{
v___x_3187_ = v___x_3184_;
goto v_reusejp_3186_;
}
else
{
lean_object* v_reuseFailAlloc_3188_; 
v_reuseFailAlloc_3188_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3188_, 0, v_a_3181_);
lean_ctor_set(v_reuseFailAlloc_3188_, 1, v_a_3182_);
v___x_3187_ = v_reuseFailAlloc_3188_;
goto v_reusejp_3186_;
}
v_reusejp_3186_:
{
return v___x_3187_;
}
}
}
}
else
{
lean_object* v_a_3190_; lean_object* v_a_3191_; lean_object* v___x_3193_; uint8_t v_isShared_3194_; uint8_t v_isSharedCheck_3198_; 
v_a_3190_ = lean_ctor_get(v___x_3166_, 0);
v_a_3191_ = lean_ctor_get(v___x_3166_, 1);
v_isSharedCheck_3198_ = !lean_is_exclusive(v___x_3166_);
if (v_isSharedCheck_3198_ == 0)
{
v___x_3193_ = v___x_3166_;
v_isShared_3194_ = v_isSharedCheck_3198_;
goto v_resetjp_3192_;
}
else
{
lean_inc(v_a_3191_);
lean_inc(v_a_3190_);
lean_dec(v___x_3166_);
v___x_3193_ = lean_box(0);
v_isShared_3194_ = v_isSharedCheck_3198_;
goto v_resetjp_3192_;
}
v_resetjp_3192_:
{
lean_object* v___x_3196_; 
if (v_isShared_3194_ == 0)
{
v___x_3196_ = v___x_3193_;
goto v_reusejp_3195_;
}
else
{
lean_object* v_reuseFailAlloc_3197_; 
v_reuseFailAlloc_3197_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3197_, 0, v_a_3190_);
lean_ctor_set(v_reuseFailAlloc_3197_, 1, v_a_3191_);
v___x_3196_ = v_reuseFailAlloc_3197_;
goto v_reusejp_3195_;
}
v_reusejp_3195_:
{
return v___x_3196_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readSymbolicInteraction(lean_object* v_a_3199_){
_start:
{
lean_object* v___x_3200_; lean_object* v___x_3201_; 
v___x_3200_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32), 1, 0);
v___x_3201_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_3200_, v_a_3199_);
if (lean_obj_tag(v___x_3201_) == 0)
{
lean_object* v_a_3202_; lean_object* v_a_3203_; lean_object* v___x_3204_; 
v_a_3202_ = lean_ctor_get(v___x_3201_, 0);
lean_inc(v_a_3202_);
v_a_3203_ = lean_ctor_get(v___x_3201_, 1);
lean_inc(v_a_3203_);
lean_dec_ref_known(v___x_3201_, 2);
v___x_3204_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_3203_);
if (lean_obj_tag(v___x_3204_) == 0)
{
lean_object* v_a_3205_; lean_object* v_a_3206_; lean_object* v___x_3207_; 
v_a_3205_ = lean_ctor_get(v___x_3204_, 0);
lean_inc(v_a_3205_);
v_a_3206_ = lean_ctor_get(v___x_3204_, 1);
lean_inc(v_a_3206_);
lean_dec_ref_known(v___x_3204_, 2);
v___x_3207_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_3206_);
if (lean_obj_tag(v___x_3207_) == 0)
{
lean_object* v_a_3208_; lean_object* v_a_3209_; lean_object* v___x_3210_; 
v_a_3208_ = lean_ctor_get(v___x_3207_, 0);
lean_inc(v_a_3208_);
v_a_3209_ = lean_ctor_get(v___x_3207_, 1);
lean_inc(v_a_3209_);
lean_dec_ref_known(v___x_3207_, 2);
v___x_3210_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_3209_);
if (lean_obj_tag(v___x_3210_) == 0)
{
lean_object* v_a_3211_; lean_object* v_a_3212_; lean_object* v___x_3214_; uint8_t v_isShared_3215_; uint8_t v_isSharedCheck_3223_; 
v_a_3211_ = lean_ctor_get(v___x_3210_, 0);
v_a_3212_ = lean_ctor_get(v___x_3210_, 1);
v_isSharedCheck_3223_ = !lean_is_exclusive(v___x_3210_);
if (v_isSharedCheck_3223_ == 0)
{
v___x_3214_ = v___x_3210_;
v_isShared_3215_ = v_isSharedCheck_3223_;
goto v_resetjp_3213_;
}
else
{
lean_inc(v_a_3212_);
lean_inc(v_a_3211_);
lean_dec(v___x_3210_);
v___x_3214_ = lean_box(0);
v_isShared_3215_ = v_isSharedCheck_3223_;
goto v_resetjp_3213_;
}
v_resetjp_3213_:
{
lean_object* v___x_3216_; uint32_t v___x_3217_; uint32_t v___x_3218_; uint32_t v___x_3219_; lean_object* v___x_3221_; 
v___x_3216_ = lean_alloc_ctor(0, 1, 12);
lean_ctor_set(v___x_3216_, 0, v_a_3202_);
v___x_3217_ = lean_unbox_uint32(v_a_3205_);
lean_dec(v_a_3205_);
lean_ctor_set_uint32(v___x_3216_, sizeof(void*)*1, v___x_3217_);
v___x_3218_ = lean_unbox_uint32(v_a_3208_);
lean_dec(v_a_3208_);
lean_ctor_set_uint32(v___x_3216_, sizeof(void*)*1 + 4, v___x_3218_);
v___x_3219_ = lean_unbox_uint32(v_a_3211_);
lean_dec(v_a_3211_);
lean_ctor_set_uint32(v___x_3216_, sizeof(void*)*1 + 8, v___x_3219_);
if (v_isShared_3215_ == 0)
{
lean_ctor_set(v___x_3214_, 0, v___x_3216_);
v___x_3221_ = v___x_3214_;
goto v_reusejp_3220_;
}
else
{
lean_object* v_reuseFailAlloc_3222_; 
v_reuseFailAlloc_3222_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3222_, 0, v___x_3216_);
lean_ctor_set(v_reuseFailAlloc_3222_, 1, v_a_3212_);
v___x_3221_ = v_reuseFailAlloc_3222_;
goto v_reusejp_3220_;
}
v_reusejp_3220_:
{
return v___x_3221_;
}
}
}
else
{
lean_object* v_a_3224_; lean_object* v_a_3225_; lean_object* v___x_3227_; uint8_t v_isShared_3228_; uint8_t v_isSharedCheck_3232_; 
lean_dec(v_a_3208_);
lean_dec(v_a_3205_);
lean_dec(v_a_3202_);
v_a_3224_ = lean_ctor_get(v___x_3210_, 0);
v_a_3225_ = lean_ctor_get(v___x_3210_, 1);
v_isSharedCheck_3232_ = !lean_is_exclusive(v___x_3210_);
if (v_isSharedCheck_3232_ == 0)
{
v___x_3227_ = v___x_3210_;
v_isShared_3228_ = v_isSharedCheck_3232_;
goto v_resetjp_3226_;
}
else
{
lean_inc(v_a_3225_);
lean_inc(v_a_3224_);
lean_dec(v___x_3210_);
v___x_3227_ = lean_box(0);
v_isShared_3228_ = v_isSharedCheck_3232_;
goto v_resetjp_3226_;
}
v_resetjp_3226_:
{
lean_object* v___x_3230_; 
if (v_isShared_3228_ == 0)
{
v___x_3230_ = v___x_3227_;
goto v_reusejp_3229_;
}
else
{
lean_object* v_reuseFailAlloc_3231_; 
v_reuseFailAlloc_3231_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3231_, 0, v_a_3224_);
lean_ctor_set(v_reuseFailAlloc_3231_, 1, v_a_3225_);
v___x_3230_ = v_reuseFailAlloc_3231_;
goto v_reusejp_3229_;
}
v_reusejp_3229_:
{
return v___x_3230_;
}
}
}
}
else
{
lean_object* v_a_3233_; lean_object* v_a_3234_; lean_object* v___x_3236_; uint8_t v_isShared_3237_; uint8_t v_isSharedCheck_3241_; 
lean_dec(v_a_3205_);
lean_dec(v_a_3202_);
v_a_3233_ = lean_ctor_get(v___x_3207_, 0);
v_a_3234_ = lean_ctor_get(v___x_3207_, 1);
v_isSharedCheck_3241_ = !lean_is_exclusive(v___x_3207_);
if (v_isSharedCheck_3241_ == 0)
{
v___x_3236_ = v___x_3207_;
v_isShared_3237_ = v_isSharedCheck_3241_;
goto v_resetjp_3235_;
}
else
{
lean_inc(v_a_3234_);
lean_inc(v_a_3233_);
lean_dec(v___x_3207_);
v___x_3236_ = lean_box(0);
v_isShared_3237_ = v_isSharedCheck_3241_;
goto v_resetjp_3235_;
}
v_resetjp_3235_:
{
lean_object* v___x_3239_; 
if (v_isShared_3237_ == 0)
{
v___x_3239_ = v___x_3236_;
goto v_reusejp_3238_;
}
else
{
lean_object* v_reuseFailAlloc_3240_; 
v_reuseFailAlloc_3240_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3240_, 0, v_a_3233_);
lean_ctor_set(v_reuseFailAlloc_3240_, 1, v_a_3234_);
v___x_3239_ = v_reuseFailAlloc_3240_;
goto v_reusejp_3238_;
}
v_reusejp_3238_:
{
return v___x_3239_;
}
}
}
}
else
{
lean_object* v_a_3242_; lean_object* v_a_3243_; lean_object* v___x_3245_; uint8_t v_isShared_3246_; uint8_t v_isSharedCheck_3250_; 
lean_dec(v_a_3202_);
v_a_3242_ = lean_ctor_get(v___x_3204_, 0);
v_a_3243_ = lean_ctor_get(v___x_3204_, 1);
v_isSharedCheck_3250_ = !lean_is_exclusive(v___x_3204_);
if (v_isSharedCheck_3250_ == 0)
{
v___x_3245_ = v___x_3204_;
v_isShared_3246_ = v_isSharedCheck_3250_;
goto v_resetjp_3244_;
}
else
{
lean_inc(v_a_3243_);
lean_inc(v_a_3242_);
lean_dec(v___x_3204_);
v___x_3245_ = lean_box(0);
v_isShared_3246_ = v_isSharedCheck_3250_;
goto v_resetjp_3244_;
}
v_resetjp_3244_:
{
lean_object* v___x_3248_; 
if (v_isShared_3246_ == 0)
{
v___x_3248_ = v___x_3245_;
goto v_reusejp_3247_;
}
else
{
lean_object* v_reuseFailAlloc_3249_; 
v_reuseFailAlloc_3249_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3249_, 0, v_a_3242_);
lean_ctor_set(v_reuseFailAlloc_3249_, 1, v_a_3243_);
v___x_3248_ = v_reuseFailAlloc_3249_;
goto v_reusejp_3247_;
}
v_reusejp_3247_:
{
return v___x_3248_;
}
}
}
}
else
{
lean_object* v_a_3251_; lean_object* v_a_3252_; lean_object* v___x_3254_; uint8_t v_isShared_3255_; uint8_t v_isSharedCheck_3259_; 
v_a_3251_ = lean_ctor_get(v___x_3201_, 0);
v_a_3252_ = lean_ctor_get(v___x_3201_, 1);
v_isSharedCheck_3259_ = !lean_is_exclusive(v___x_3201_);
if (v_isSharedCheck_3259_ == 0)
{
v___x_3254_ = v___x_3201_;
v_isShared_3255_ = v_isSharedCheck_3259_;
goto v_resetjp_3253_;
}
else
{
lean_inc(v_a_3252_);
lean_inc(v_a_3251_);
lean_dec(v___x_3201_);
v___x_3254_ = lean_box(0);
v_isShared_3255_ = v_isSharedCheck_3259_;
goto v_resetjp_3253_;
}
v_resetjp_3253_:
{
lean_object* v___x_3257_; 
if (v_isShared_3255_ == 0)
{
v___x_3257_ = v___x_3254_;
goto v_reusejp_3256_;
}
else
{
lean_object* v_reuseFailAlloc_3258_; 
v_reuseFailAlloc_3258_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3258_, 0, v_a_3251_);
lean_ctor_set(v_reuseFailAlloc_3258_, 1, v_a_3252_);
v___x_3257_ = v_reuseFailAlloc_3258_;
goto v_reusejp_3256_;
}
v_reusejp_3256_:
{
return v___x_3257_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readSymbolicConstraintsDag(lean_object* v_a_3260_){
_start:
{
lean_object* v___x_3261_; 
v___x_3261_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readSymbolicExpressionDag(v_a_3260_);
if (lean_obj_tag(v___x_3261_) == 0)
{
lean_object* v_a_3262_; lean_object* v_a_3263_; lean_object* v___x_3264_; lean_object* v___x_3265_; 
v_a_3262_ = lean_ctor_get(v___x_3261_, 0);
lean_inc(v_a_3262_);
v_a_3263_ = lean_ctor_get(v___x_3261_, 1);
lean_inc(v_a_3263_);
lean_dec_ref_known(v___x_3261_, 2);
v___x_3264_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readSymbolicInteraction), 1, 0);
v___x_3265_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_3264_, v_a_3263_);
if (lean_obj_tag(v___x_3265_) == 0)
{
lean_object* v_a_3266_; lean_object* v_a_3267_; lean_object* v___x_3269_; uint8_t v_isShared_3270_; uint8_t v_isSharedCheck_3275_; 
v_a_3266_ = lean_ctor_get(v___x_3265_, 0);
v_a_3267_ = lean_ctor_get(v___x_3265_, 1);
v_isSharedCheck_3275_ = !lean_is_exclusive(v___x_3265_);
if (v_isSharedCheck_3275_ == 0)
{
v___x_3269_ = v___x_3265_;
v_isShared_3270_ = v_isSharedCheck_3275_;
goto v_resetjp_3268_;
}
else
{
lean_inc(v_a_3267_);
lean_inc(v_a_3266_);
lean_dec(v___x_3265_);
v___x_3269_ = lean_box(0);
v_isShared_3270_ = v_isSharedCheck_3275_;
goto v_resetjp_3268_;
}
v_resetjp_3268_:
{
lean_object* v___x_3271_; lean_object* v___x_3273_; 
v___x_3271_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3271_, 0, v_a_3262_);
lean_ctor_set(v___x_3271_, 1, v_a_3266_);
if (v_isShared_3270_ == 0)
{
lean_ctor_set(v___x_3269_, 0, v___x_3271_);
v___x_3273_ = v___x_3269_;
goto v_reusejp_3272_;
}
else
{
lean_object* v_reuseFailAlloc_3274_; 
v_reuseFailAlloc_3274_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3274_, 0, v___x_3271_);
lean_ctor_set(v_reuseFailAlloc_3274_, 1, v_a_3267_);
v___x_3273_ = v_reuseFailAlloc_3274_;
goto v_reusejp_3272_;
}
v_reusejp_3272_:
{
return v___x_3273_;
}
}
}
else
{
lean_object* v_a_3276_; lean_object* v_a_3277_; lean_object* v___x_3279_; uint8_t v_isShared_3280_; uint8_t v_isSharedCheck_3284_; 
lean_dec(v_a_3262_);
v_a_3276_ = lean_ctor_get(v___x_3265_, 0);
v_a_3277_ = lean_ctor_get(v___x_3265_, 1);
v_isSharedCheck_3284_ = !lean_is_exclusive(v___x_3265_);
if (v_isSharedCheck_3284_ == 0)
{
v___x_3279_ = v___x_3265_;
v_isShared_3280_ = v_isSharedCheck_3284_;
goto v_resetjp_3278_;
}
else
{
lean_inc(v_a_3277_);
lean_inc(v_a_3276_);
lean_dec(v___x_3265_);
v___x_3279_ = lean_box(0);
v_isShared_3280_ = v_isSharedCheck_3284_;
goto v_resetjp_3278_;
}
v_resetjp_3278_:
{
lean_object* v___x_3282_; 
if (v_isShared_3280_ == 0)
{
v___x_3282_ = v___x_3279_;
goto v_reusejp_3281_;
}
else
{
lean_object* v_reuseFailAlloc_3283_; 
v_reuseFailAlloc_3283_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3283_, 0, v_a_3276_);
lean_ctor_set(v_reuseFailAlloc_3283_, 1, v_a_3277_);
v___x_3282_ = v_reuseFailAlloc_3283_;
goto v_reusejp_3281_;
}
v_reusejp_3281_:
{
return v___x_3282_;
}
}
}
}
else
{
lean_object* v_a_3285_; lean_object* v_a_3286_; lean_object* v___x_3288_; uint8_t v_isShared_3289_; uint8_t v_isSharedCheck_3293_; 
v_a_3285_ = lean_ctor_get(v___x_3261_, 0);
v_a_3286_ = lean_ctor_get(v___x_3261_, 1);
v_isSharedCheck_3293_ = !lean_is_exclusive(v___x_3261_);
if (v_isSharedCheck_3293_ == 0)
{
v___x_3288_ = v___x_3261_;
v_isShared_3289_ = v_isSharedCheck_3293_;
goto v_resetjp_3287_;
}
else
{
lean_inc(v_a_3286_);
lean_inc(v_a_3285_);
lean_dec(v___x_3261_);
v___x_3288_ = lean_box(0);
v_isShared_3289_ = v_isSharedCheck_3293_;
goto v_resetjp_3287_;
}
v_resetjp_3287_:
{
lean_object* v___x_3291_; 
if (v_isShared_3289_ == 0)
{
v___x_3291_ = v___x_3288_;
goto v_reusejp_3290_;
}
else
{
lean_object* v_reuseFailAlloc_3292_; 
v_reuseFailAlloc_3292_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3292_, 0, v_a_3285_);
lean_ctor_set(v_reuseFailAlloc_3292_, 1, v_a_3286_);
v___x_3291_ = v_reuseFailAlloc_3292_;
goto v_reusejp_3290_;
}
v_reusejp_3290_:
{
return v___x_3291_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirRoundConfig(lean_object* v_a_3294_){
_start:
{
lean_object* v___x_3295_; 
v___x_3295_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_3294_);
if (lean_obj_tag(v___x_3295_) == 0)
{
lean_object* v_a_3296_; lean_object* v_a_3297_; lean_object* v___x_3299_; uint8_t v_isShared_3300_; uint8_t v_isSharedCheck_3304_; 
v_a_3296_ = lean_ctor_get(v___x_3295_, 0);
v_a_3297_ = lean_ctor_get(v___x_3295_, 1);
v_isSharedCheck_3304_ = !lean_is_exclusive(v___x_3295_);
if (v_isSharedCheck_3304_ == 0)
{
v___x_3299_ = v___x_3295_;
v_isShared_3300_ = v_isSharedCheck_3304_;
goto v_resetjp_3298_;
}
else
{
lean_inc(v_a_3297_);
lean_inc(v_a_3296_);
lean_dec(v___x_3295_);
v___x_3299_ = lean_box(0);
v_isShared_3300_ = v_isSharedCheck_3304_;
goto v_resetjp_3298_;
}
v_resetjp_3298_:
{
lean_object* v___x_3302_; 
if (v_isShared_3300_ == 0)
{
v___x_3302_ = v___x_3299_;
goto v_reusejp_3301_;
}
else
{
lean_object* v_reuseFailAlloc_3303_; 
v_reuseFailAlloc_3303_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3303_, 0, v_a_3296_);
lean_ctor_set(v_reuseFailAlloc_3303_, 1, v_a_3297_);
v___x_3302_ = v_reuseFailAlloc_3303_;
goto v_reusejp_3301_;
}
v_reusejp_3301_:
{
return v___x_3302_;
}
}
}
else
{
lean_object* v_a_3305_; lean_object* v_a_3306_; lean_object* v___x_3308_; uint8_t v_isShared_3309_; uint8_t v_isSharedCheck_3313_; 
v_a_3305_ = lean_ctor_get(v___x_3295_, 0);
v_a_3306_ = lean_ctor_get(v___x_3295_, 1);
v_isSharedCheck_3313_ = !lean_is_exclusive(v___x_3295_);
if (v_isSharedCheck_3313_ == 0)
{
v___x_3308_ = v___x_3295_;
v_isShared_3309_ = v_isSharedCheck_3313_;
goto v_resetjp_3307_;
}
else
{
lean_inc(v_a_3306_);
lean_inc(v_a_3305_);
lean_dec(v___x_3295_);
v___x_3308_ = lean_box(0);
v_isShared_3309_ = v_isSharedCheck_3313_;
goto v_resetjp_3307_;
}
v_resetjp_3307_:
{
lean_object* v___x_3311_; 
if (v_isShared_3309_ == 0)
{
v___x_3311_ = v___x_3308_;
goto v_reusejp_3310_;
}
else
{
lean_object* v_reuseFailAlloc_3312_; 
v_reuseFailAlloc_3312_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3312_, 0, v_a_3305_);
lean_ctor_set(v_reuseFailAlloc_3312_, 1, v_a_3306_);
v___x_3311_ = v_reuseFailAlloc_3312_;
goto v_reusejp_3310_;
}
v_reusejp_3310_:
{
return v___x_3311_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProximityStrategy(lean_object* v_a_3315_){
_start:
{
lean_object* v___x_3316_; 
lean_inc_ref(v_a_3315_);
v___x_3316_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readByte(v_a_3315_);
if (lean_obj_tag(v___x_3316_) == 0)
{
lean_object* v_a_3317_; lean_object* v_a_3318_; lean_object* v___x_3320_; uint8_t v_isShared_3321_; uint8_t v_isSharedCheck_3407_; 
v_a_3317_ = lean_ctor_get(v___x_3316_, 0);
v_a_3318_ = lean_ctor_get(v___x_3316_, 1);
v_isSharedCheck_3407_ = !lean_is_exclusive(v___x_3316_);
if (v_isSharedCheck_3407_ == 0)
{
v___x_3320_ = v___x_3316_;
v_isShared_3321_ = v_isSharedCheck_3407_;
goto v_resetjp_3319_;
}
else
{
lean_inc(v_a_3318_);
lean_inc(v_a_3317_);
lean_dec(v___x_3316_);
v___x_3320_ = lean_box(0);
v_isShared_3321_ = v_isSharedCheck_3407_;
goto v_resetjp_3319_;
}
v_resetjp_3319_:
{
uint8_t v___x_3322_; uint8_t v___x_3323_; uint8_t v___x_3324_; 
v___x_3322_ = 0;
v___x_3323_ = lean_unbox(v_a_3317_);
v___x_3324_ = lean_uint8_dec_eq(v___x_3323_, v___x_3322_);
if (v___x_3324_ == 0)
{
uint8_t v___x_3325_; uint8_t v___x_3326_; uint8_t v___x_3327_; 
v___x_3325_ = 1;
v___x_3326_ = lean_unbox(v_a_3317_);
v___x_3327_ = lean_uint8_dec_eq(v___x_3326_, v___x_3325_);
if (v___x_3327_ == 0)
{
uint8_t v___x_3328_; uint8_t v___x_3329_; uint8_t v___x_3330_; 
v___x_3328_ = 2;
v___x_3329_ = lean_unbox(v_a_3317_);
v___x_3330_ = lean_uint8_dec_eq(v___x_3329_, v___x_3328_);
if (v___x_3330_ == 0)
{
lean_object* v_offset_3331_; lean_object* v___x_3333_; uint8_t v_isShared_3334_; uint8_t v_isSharedCheck_3346_; 
v_offset_3331_ = lean_ctor_get(v_a_3315_, 1);
v_isSharedCheck_3346_ = !lean_is_exclusive(v_a_3315_);
if (v_isSharedCheck_3346_ == 0)
{
lean_object* v_unused_3347_; 
v_unused_3347_ = lean_ctor_get(v_a_3315_, 0);
lean_dec(v_unused_3347_);
v___x_3333_ = v_a_3315_;
v_isShared_3334_ = v_isSharedCheck_3346_;
goto v_resetjp_3332_;
}
else
{
lean_inc(v_offset_3331_);
lean_dec(v_a_3315_);
v___x_3333_ = lean_box(0);
v_isShared_3334_ = v_isSharedCheck_3346_;
goto v_resetjp_3332_;
}
v_resetjp_3332_:
{
lean_object* v___x_3335_; uint8_t v___x_3336_; lean_object* v___x_3337_; lean_object* v___x_3338_; lean_object* v___x_3339_; lean_object* v___x_3341_; 
v___x_3335_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProximityStrategy___closed__0));
v___x_3336_ = lean_unbox(v_a_3317_);
lean_dec(v_a_3317_);
v___x_3337_ = lean_uint8_to_nat(v___x_3336_);
v___x_3338_ = l_Nat_reprFast(v___x_3337_);
v___x_3339_ = lean_string_append(v___x_3335_, v___x_3338_);
lean_dec_ref(v___x_3338_);
if (v_isShared_3334_ == 0)
{
lean_ctor_set_tag(v___x_3333_, 3);
lean_ctor_set(v___x_3333_, 1, v___x_3339_);
lean_ctor_set(v___x_3333_, 0, v_offset_3331_);
v___x_3341_ = v___x_3333_;
goto v_reusejp_3340_;
}
else
{
lean_object* v_reuseFailAlloc_3345_; 
v_reuseFailAlloc_3345_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3345_, 0, v_offset_3331_);
lean_ctor_set(v_reuseFailAlloc_3345_, 1, v___x_3339_);
v___x_3341_ = v_reuseFailAlloc_3345_;
goto v_reusejp_3340_;
}
v_reusejp_3340_:
{
lean_object* v___x_3343_; 
if (v_isShared_3321_ == 0)
{
lean_ctor_set_tag(v___x_3320_, 1);
lean_ctor_set(v___x_3320_, 0, v___x_3341_);
v___x_3343_ = v___x_3320_;
goto v_reusejp_3342_;
}
else
{
lean_object* v_reuseFailAlloc_3344_; 
v_reuseFailAlloc_3344_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3344_, 0, v___x_3341_);
lean_ctor_set(v_reuseFailAlloc_3344_, 1, v_a_3318_);
v___x_3343_ = v_reuseFailAlloc_3344_;
goto v_reusejp_3342_;
}
v_reusejp_3342_:
{
return v___x_3343_;
}
}
}
}
else
{
lean_object* v___x_3348_; 
lean_del_object(v___x_3320_);
lean_dec(v_a_3317_);
lean_dec_ref(v_a_3315_);
v___x_3348_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt64(v_a_3318_);
if (lean_obj_tag(v___x_3348_) == 0)
{
lean_object* v_a_3349_; lean_object* v_a_3350_; lean_object* v___x_3352_; uint8_t v_isShared_3353_; uint8_t v_isSharedCheck_3359_; 
v_a_3349_ = lean_ctor_get(v___x_3348_, 0);
v_a_3350_ = lean_ctor_get(v___x_3348_, 1);
v_isSharedCheck_3359_ = !lean_is_exclusive(v___x_3348_);
if (v_isSharedCheck_3359_ == 0)
{
v___x_3352_ = v___x_3348_;
v_isShared_3353_ = v_isSharedCheck_3359_;
goto v_resetjp_3351_;
}
else
{
lean_inc(v_a_3350_);
lean_inc(v_a_3349_);
lean_dec(v___x_3348_);
v___x_3352_ = lean_box(0);
v_isShared_3353_ = v_isSharedCheck_3359_;
goto v_resetjp_3351_;
}
v_resetjp_3351_:
{
lean_object* v___x_3354_; uint64_t v___x_3355_; lean_object* v___x_3357_; 
v___x_3354_ = lean_alloc_ctor(2, 0, 8);
v___x_3355_ = lean_unbox_uint64(v_a_3349_);
lean_dec(v_a_3349_);
lean_ctor_set_uint64(v___x_3354_, 0, v___x_3355_);
if (v_isShared_3353_ == 0)
{
lean_ctor_set(v___x_3352_, 0, v___x_3354_);
v___x_3357_ = v___x_3352_;
goto v_reusejp_3356_;
}
else
{
lean_object* v_reuseFailAlloc_3358_; 
v_reuseFailAlloc_3358_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3358_, 0, v___x_3354_);
lean_ctor_set(v_reuseFailAlloc_3358_, 1, v_a_3350_);
v___x_3357_ = v_reuseFailAlloc_3358_;
goto v_reusejp_3356_;
}
v_reusejp_3356_:
{
return v___x_3357_;
}
}
}
else
{
lean_object* v_a_3360_; lean_object* v_a_3361_; lean_object* v___x_3363_; uint8_t v_isShared_3364_; uint8_t v_isSharedCheck_3368_; 
v_a_3360_ = lean_ctor_get(v___x_3348_, 0);
v_a_3361_ = lean_ctor_get(v___x_3348_, 1);
v_isSharedCheck_3368_ = !lean_is_exclusive(v___x_3348_);
if (v_isSharedCheck_3368_ == 0)
{
v___x_3363_ = v___x_3348_;
v_isShared_3364_ = v_isSharedCheck_3368_;
goto v_resetjp_3362_;
}
else
{
lean_inc(v_a_3361_);
lean_inc(v_a_3360_);
lean_dec(v___x_3348_);
v___x_3363_ = lean_box(0);
v_isShared_3364_ = v_isSharedCheck_3368_;
goto v_resetjp_3362_;
}
v_resetjp_3362_:
{
lean_object* v___x_3366_; 
if (v_isShared_3364_ == 0)
{
v___x_3366_ = v___x_3363_;
goto v_reusejp_3365_;
}
else
{
lean_object* v_reuseFailAlloc_3367_; 
v_reuseFailAlloc_3367_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3367_, 0, v_a_3360_);
lean_ctor_set(v_reuseFailAlloc_3367_, 1, v_a_3361_);
v___x_3366_ = v_reuseFailAlloc_3367_;
goto v_reusejp_3365_;
}
v_reusejp_3365_:
{
return v___x_3366_;
}
}
}
}
}
else
{
lean_object* v___x_3369_; 
lean_del_object(v___x_3320_);
lean_dec(v_a_3317_);
lean_dec_ref(v_a_3315_);
v___x_3369_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt64(v_a_3318_);
if (lean_obj_tag(v___x_3369_) == 0)
{
lean_object* v_a_3370_; lean_object* v_a_3371_; lean_object* v___x_3372_; 
v_a_3370_ = lean_ctor_get(v___x_3369_, 0);
lean_inc(v_a_3370_);
v_a_3371_ = lean_ctor_get(v___x_3369_, 1);
lean_inc(v_a_3371_);
lean_dec_ref_known(v___x_3369_, 2);
v___x_3372_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt64(v_a_3371_);
if (lean_obj_tag(v___x_3372_) == 0)
{
lean_object* v_a_3373_; lean_object* v_a_3374_; lean_object* v___x_3376_; uint8_t v_isShared_3377_; uint8_t v_isSharedCheck_3384_; 
v_a_3373_ = lean_ctor_get(v___x_3372_, 0);
v_a_3374_ = lean_ctor_get(v___x_3372_, 1);
v_isSharedCheck_3384_ = !lean_is_exclusive(v___x_3372_);
if (v_isSharedCheck_3384_ == 0)
{
v___x_3376_ = v___x_3372_;
v_isShared_3377_ = v_isSharedCheck_3384_;
goto v_resetjp_3375_;
}
else
{
lean_inc(v_a_3374_);
lean_inc(v_a_3373_);
lean_dec(v___x_3372_);
v___x_3376_ = lean_box(0);
v_isShared_3377_ = v_isSharedCheck_3384_;
goto v_resetjp_3375_;
}
v_resetjp_3375_:
{
lean_object* v___x_3378_; uint64_t v___x_3379_; uint64_t v___x_3380_; lean_object* v___x_3382_; 
v___x_3378_ = lean_alloc_ctor(1, 0, 16);
v___x_3379_ = lean_unbox_uint64(v_a_3370_);
lean_dec(v_a_3370_);
lean_ctor_set_uint64(v___x_3378_, 0, v___x_3379_);
v___x_3380_ = lean_unbox_uint64(v_a_3373_);
lean_dec(v_a_3373_);
lean_ctor_set_uint64(v___x_3378_, 8, v___x_3380_);
if (v_isShared_3377_ == 0)
{
lean_ctor_set(v___x_3376_, 0, v___x_3378_);
v___x_3382_ = v___x_3376_;
goto v_reusejp_3381_;
}
else
{
lean_object* v_reuseFailAlloc_3383_; 
v_reuseFailAlloc_3383_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3383_, 0, v___x_3378_);
lean_ctor_set(v_reuseFailAlloc_3383_, 1, v_a_3374_);
v___x_3382_ = v_reuseFailAlloc_3383_;
goto v_reusejp_3381_;
}
v_reusejp_3381_:
{
return v___x_3382_;
}
}
}
else
{
lean_object* v_a_3385_; lean_object* v_a_3386_; lean_object* v___x_3388_; uint8_t v_isShared_3389_; uint8_t v_isSharedCheck_3393_; 
lean_dec(v_a_3370_);
v_a_3385_ = lean_ctor_get(v___x_3372_, 0);
v_a_3386_ = lean_ctor_get(v___x_3372_, 1);
v_isSharedCheck_3393_ = !lean_is_exclusive(v___x_3372_);
if (v_isSharedCheck_3393_ == 0)
{
v___x_3388_ = v___x_3372_;
v_isShared_3389_ = v_isSharedCheck_3393_;
goto v_resetjp_3387_;
}
else
{
lean_inc(v_a_3386_);
lean_inc(v_a_3385_);
lean_dec(v___x_3372_);
v___x_3388_ = lean_box(0);
v_isShared_3389_ = v_isSharedCheck_3393_;
goto v_resetjp_3387_;
}
v_resetjp_3387_:
{
lean_object* v___x_3391_; 
if (v_isShared_3389_ == 0)
{
v___x_3391_ = v___x_3388_;
goto v_reusejp_3390_;
}
else
{
lean_object* v_reuseFailAlloc_3392_; 
v_reuseFailAlloc_3392_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3392_, 0, v_a_3385_);
lean_ctor_set(v_reuseFailAlloc_3392_, 1, v_a_3386_);
v___x_3391_ = v_reuseFailAlloc_3392_;
goto v_reusejp_3390_;
}
v_reusejp_3390_:
{
return v___x_3391_;
}
}
}
}
else
{
lean_object* v_a_3394_; lean_object* v_a_3395_; lean_object* v___x_3397_; uint8_t v_isShared_3398_; uint8_t v_isSharedCheck_3402_; 
v_a_3394_ = lean_ctor_get(v___x_3369_, 0);
v_a_3395_ = lean_ctor_get(v___x_3369_, 1);
v_isSharedCheck_3402_ = !lean_is_exclusive(v___x_3369_);
if (v_isSharedCheck_3402_ == 0)
{
v___x_3397_ = v___x_3369_;
v_isShared_3398_ = v_isSharedCheck_3402_;
goto v_resetjp_3396_;
}
else
{
lean_inc(v_a_3395_);
lean_inc(v_a_3394_);
lean_dec(v___x_3369_);
v___x_3397_ = lean_box(0);
v_isShared_3398_ = v_isSharedCheck_3402_;
goto v_resetjp_3396_;
}
v_resetjp_3396_:
{
lean_object* v___x_3400_; 
if (v_isShared_3398_ == 0)
{
v___x_3400_ = v___x_3397_;
goto v_reusejp_3399_;
}
else
{
lean_object* v_reuseFailAlloc_3401_; 
v_reuseFailAlloc_3401_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3401_, 0, v_a_3394_);
lean_ctor_set(v_reuseFailAlloc_3401_, 1, v_a_3395_);
v___x_3400_ = v_reuseFailAlloc_3401_;
goto v_reusejp_3399_;
}
v_reusejp_3399_:
{
return v___x_3400_;
}
}
}
}
}
else
{
lean_object* v___x_3403_; lean_object* v___x_3405_; 
lean_dec(v_a_3317_);
lean_dec_ref(v_a_3315_);
v___x_3403_ = lean_box(0);
if (v_isShared_3321_ == 0)
{
lean_ctor_set(v___x_3320_, 0, v___x_3403_);
v___x_3405_ = v___x_3320_;
goto v_reusejp_3404_;
}
else
{
lean_object* v_reuseFailAlloc_3406_; 
v_reuseFailAlloc_3406_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3406_, 0, v___x_3403_);
lean_ctor_set(v_reuseFailAlloc_3406_, 1, v_a_3318_);
v___x_3405_ = v_reuseFailAlloc_3406_;
goto v_reusejp_3404_;
}
v_reusejp_3404_:
{
return v___x_3405_;
}
}
}
}
else
{
lean_object* v_a_3408_; lean_object* v_a_3409_; lean_object* v___x_3411_; uint8_t v_isShared_3412_; uint8_t v_isSharedCheck_3416_; 
lean_dec_ref(v_a_3315_);
v_a_3408_ = lean_ctor_get(v___x_3316_, 0);
v_a_3409_ = lean_ctor_get(v___x_3316_, 1);
v_isSharedCheck_3416_ = !lean_is_exclusive(v___x_3316_);
if (v_isSharedCheck_3416_ == 0)
{
v___x_3411_ = v___x_3316_;
v_isShared_3412_ = v_isSharedCheck_3416_;
goto v_resetjp_3410_;
}
else
{
lean_inc(v_a_3409_);
lean_inc(v_a_3408_);
lean_dec(v___x_3316_);
v___x_3411_ = lean_box(0);
v_isShared_3412_ = v_isSharedCheck_3416_;
goto v_resetjp_3410_;
}
v_resetjp_3410_:
{
lean_object* v___x_3414_; 
if (v_isShared_3412_ == 0)
{
v___x_3414_ = v___x_3411_;
goto v_reusejp_3413_;
}
else
{
lean_object* v_reuseFailAlloc_3415_; 
v_reuseFailAlloc_3415_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3415_, 0, v_a_3408_);
lean_ctor_set(v_reuseFailAlloc_3415_, 1, v_a_3409_);
v___x_3414_ = v_reuseFailAlloc_3415_;
goto v_reusejp_3413_;
}
v_reusejp_3413_:
{
return v___x_3414_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirConfig(lean_object* v_a_3417_){
_start:
{
lean_object* v___x_3418_; 
v___x_3418_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_3417_);
if (lean_obj_tag(v___x_3418_) == 0)
{
lean_object* v_a_3419_; lean_object* v_a_3420_; lean_object* v___x_3421_; lean_object* v___x_3422_; 
v_a_3419_ = lean_ctor_get(v___x_3418_, 0);
lean_inc(v_a_3419_);
v_a_3420_ = lean_ctor_get(v___x_3418_, 1);
lean_inc(v_a_3420_);
lean_dec_ref_known(v___x_3418_, 2);
v___x_3421_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirRoundConfig), 1, 0);
v___x_3422_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_3421_, v_a_3420_);
if (lean_obj_tag(v___x_3422_) == 0)
{
lean_object* v_a_3423_; lean_object* v_a_3424_; lean_object* v___x_3425_; 
v_a_3423_ = lean_ctor_get(v___x_3422_, 0);
lean_inc(v_a_3423_);
v_a_3424_ = lean_ctor_get(v___x_3422_, 1);
lean_inc(v_a_3424_);
lean_dec_ref_known(v___x_3422_, 2);
v___x_3425_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_3424_);
if (lean_obj_tag(v___x_3425_) == 0)
{
lean_object* v_a_3426_; lean_object* v_a_3427_; lean_object* v___x_3428_; 
v_a_3426_ = lean_ctor_get(v___x_3425_, 0);
lean_inc(v_a_3426_);
v_a_3427_ = lean_ctor_get(v___x_3425_, 1);
lean_inc(v_a_3427_);
lean_dec_ref_known(v___x_3425_, 2);
v___x_3428_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_3427_);
if (lean_obj_tag(v___x_3428_) == 0)
{
lean_object* v_a_3429_; lean_object* v_a_3430_; lean_object* v___x_3431_; 
v_a_3429_ = lean_ctor_get(v___x_3428_, 0);
lean_inc(v_a_3429_);
v_a_3430_ = lean_ctor_get(v___x_3428_, 1);
lean_inc(v_a_3430_);
lean_dec_ref_known(v___x_3428_, 2);
v___x_3431_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_3430_);
if (lean_obj_tag(v___x_3431_) == 0)
{
lean_object* v_a_3432_; lean_object* v_a_3433_; lean_object* v___x_3434_; 
v_a_3432_ = lean_ctor_get(v___x_3431_, 0);
lean_inc(v_a_3432_);
v_a_3433_ = lean_ctor_get(v___x_3431_, 1);
lean_inc(v_a_3433_);
lean_dec_ref_known(v___x_3431_, 2);
v___x_3434_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProximityStrategy(v_a_3433_);
if (lean_obj_tag(v___x_3434_) == 0)
{
lean_object* v_a_3435_; lean_object* v_a_3436_; lean_object* v___x_3438_; uint8_t v_isShared_3439_; uint8_t v_isSharedCheck_3448_; 
v_a_3435_ = lean_ctor_get(v___x_3434_, 0);
v_a_3436_ = lean_ctor_get(v___x_3434_, 1);
v_isSharedCheck_3448_ = !lean_is_exclusive(v___x_3434_);
if (v_isSharedCheck_3448_ == 0)
{
v___x_3438_ = v___x_3434_;
v_isShared_3439_ = v_isSharedCheck_3448_;
goto v_resetjp_3437_;
}
else
{
lean_inc(v_a_3436_);
lean_inc(v_a_3435_);
lean_dec(v___x_3434_);
v___x_3438_ = lean_box(0);
v_isShared_3439_ = v_isSharedCheck_3448_;
goto v_resetjp_3437_;
}
v_resetjp_3437_:
{
lean_object* v___x_3440_; uint32_t v___x_3441_; uint32_t v___x_3442_; uint32_t v___x_3443_; uint32_t v___x_3444_; lean_object* v___x_3446_; 
v___x_3440_ = lean_alloc_ctor(0, 2, 16);
lean_ctor_set(v___x_3440_, 0, v_a_3423_);
lean_ctor_set(v___x_3440_, 1, v_a_3435_);
v___x_3441_ = lean_unbox_uint32(v_a_3419_);
lean_dec(v_a_3419_);
lean_ctor_set_uint32(v___x_3440_, sizeof(void*)*2, v___x_3441_);
v___x_3442_ = lean_unbox_uint32(v_a_3426_);
lean_dec(v_a_3426_);
lean_ctor_set_uint32(v___x_3440_, sizeof(void*)*2 + 4, v___x_3442_);
v___x_3443_ = lean_unbox_uint32(v_a_3429_);
lean_dec(v_a_3429_);
lean_ctor_set_uint32(v___x_3440_, sizeof(void*)*2 + 8, v___x_3443_);
v___x_3444_ = lean_unbox_uint32(v_a_3432_);
lean_dec(v_a_3432_);
lean_ctor_set_uint32(v___x_3440_, sizeof(void*)*2 + 12, v___x_3444_);
if (v_isShared_3439_ == 0)
{
lean_ctor_set(v___x_3438_, 0, v___x_3440_);
v___x_3446_ = v___x_3438_;
goto v_reusejp_3445_;
}
else
{
lean_object* v_reuseFailAlloc_3447_; 
v_reuseFailAlloc_3447_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3447_, 0, v___x_3440_);
lean_ctor_set(v_reuseFailAlloc_3447_, 1, v_a_3436_);
v___x_3446_ = v_reuseFailAlloc_3447_;
goto v_reusejp_3445_;
}
v_reusejp_3445_:
{
return v___x_3446_;
}
}
}
else
{
lean_object* v_a_3449_; lean_object* v_a_3450_; lean_object* v___x_3452_; uint8_t v_isShared_3453_; uint8_t v_isSharedCheck_3457_; 
lean_dec(v_a_3432_);
lean_dec(v_a_3429_);
lean_dec(v_a_3426_);
lean_dec(v_a_3423_);
lean_dec(v_a_3419_);
v_a_3449_ = lean_ctor_get(v___x_3434_, 0);
v_a_3450_ = lean_ctor_get(v___x_3434_, 1);
v_isSharedCheck_3457_ = !lean_is_exclusive(v___x_3434_);
if (v_isSharedCheck_3457_ == 0)
{
v___x_3452_ = v___x_3434_;
v_isShared_3453_ = v_isSharedCheck_3457_;
goto v_resetjp_3451_;
}
else
{
lean_inc(v_a_3450_);
lean_inc(v_a_3449_);
lean_dec(v___x_3434_);
v___x_3452_ = lean_box(0);
v_isShared_3453_ = v_isSharedCheck_3457_;
goto v_resetjp_3451_;
}
v_resetjp_3451_:
{
lean_object* v___x_3455_; 
if (v_isShared_3453_ == 0)
{
v___x_3455_ = v___x_3452_;
goto v_reusejp_3454_;
}
else
{
lean_object* v_reuseFailAlloc_3456_; 
v_reuseFailAlloc_3456_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3456_, 0, v_a_3449_);
lean_ctor_set(v_reuseFailAlloc_3456_, 1, v_a_3450_);
v___x_3455_ = v_reuseFailAlloc_3456_;
goto v_reusejp_3454_;
}
v_reusejp_3454_:
{
return v___x_3455_;
}
}
}
}
else
{
lean_object* v_a_3458_; lean_object* v_a_3459_; lean_object* v___x_3461_; uint8_t v_isShared_3462_; uint8_t v_isSharedCheck_3466_; 
lean_dec(v_a_3429_);
lean_dec(v_a_3426_);
lean_dec(v_a_3423_);
lean_dec(v_a_3419_);
v_a_3458_ = lean_ctor_get(v___x_3431_, 0);
v_a_3459_ = lean_ctor_get(v___x_3431_, 1);
v_isSharedCheck_3466_ = !lean_is_exclusive(v___x_3431_);
if (v_isSharedCheck_3466_ == 0)
{
v___x_3461_ = v___x_3431_;
v_isShared_3462_ = v_isSharedCheck_3466_;
goto v_resetjp_3460_;
}
else
{
lean_inc(v_a_3459_);
lean_inc(v_a_3458_);
lean_dec(v___x_3431_);
v___x_3461_ = lean_box(0);
v_isShared_3462_ = v_isSharedCheck_3466_;
goto v_resetjp_3460_;
}
v_resetjp_3460_:
{
lean_object* v___x_3464_; 
if (v_isShared_3462_ == 0)
{
v___x_3464_ = v___x_3461_;
goto v_reusejp_3463_;
}
else
{
lean_object* v_reuseFailAlloc_3465_; 
v_reuseFailAlloc_3465_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3465_, 0, v_a_3458_);
lean_ctor_set(v_reuseFailAlloc_3465_, 1, v_a_3459_);
v___x_3464_ = v_reuseFailAlloc_3465_;
goto v_reusejp_3463_;
}
v_reusejp_3463_:
{
return v___x_3464_;
}
}
}
}
else
{
lean_object* v_a_3467_; lean_object* v_a_3468_; lean_object* v___x_3470_; uint8_t v_isShared_3471_; uint8_t v_isSharedCheck_3475_; 
lean_dec(v_a_3426_);
lean_dec(v_a_3423_);
lean_dec(v_a_3419_);
v_a_3467_ = lean_ctor_get(v___x_3428_, 0);
v_a_3468_ = lean_ctor_get(v___x_3428_, 1);
v_isSharedCheck_3475_ = !lean_is_exclusive(v___x_3428_);
if (v_isSharedCheck_3475_ == 0)
{
v___x_3470_ = v___x_3428_;
v_isShared_3471_ = v_isSharedCheck_3475_;
goto v_resetjp_3469_;
}
else
{
lean_inc(v_a_3468_);
lean_inc(v_a_3467_);
lean_dec(v___x_3428_);
v___x_3470_ = lean_box(0);
v_isShared_3471_ = v_isSharedCheck_3475_;
goto v_resetjp_3469_;
}
v_resetjp_3469_:
{
lean_object* v___x_3473_; 
if (v_isShared_3471_ == 0)
{
v___x_3473_ = v___x_3470_;
goto v_reusejp_3472_;
}
else
{
lean_object* v_reuseFailAlloc_3474_; 
v_reuseFailAlloc_3474_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3474_, 0, v_a_3467_);
lean_ctor_set(v_reuseFailAlloc_3474_, 1, v_a_3468_);
v___x_3473_ = v_reuseFailAlloc_3474_;
goto v_reusejp_3472_;
}
v_reusejp_3472_:
{
return v___x_3473_;
}
}
}
}
else
{
lean_object* v_a_3476_; lean_object* v_a_3477_; lean_object* v___x_3479_; uint8_t v_isShared_3480_; uint8_t v_isSharedCheck_3484_; 
lean_dec(v_a_3423_);
lean_dec(v_a_3419_);
v_a_3476_ = lean_ctor_get(v___x_3425_, 0);
v_a_3477_ = lean_ctor_get(v___x_3425_, 1);
v_isSharedCheck_3484_ = !lean_is_exclusive(v___x_3425_);
if (v_isSharedCheck_3484_ == 0)
{
v___x_3479_ = v___x_3425_;
v_isShared_3480_ = v_isSharedCheck_3484_;
goto v_resetjp_3478_;
}
else
{
lean_inc(v_a_3477_);
lean_inc(v_a_3476_);
lean_dec(v___x_3425_);
v___x_3479_ = lean_box(0);
v_isShared_3480_ = v_isSharedCheck_3484_;
goto v_resetjp_3478_;
}
v_resetjp_3478_:
{
lean_object* v___x_3482_; 
if (v_isShared_3480_ == 0)
{
v___x_3482_ = v___x_3479_;
goto v_reusejp_3481_;
}
else
{
lean_object* v_reuseFailAlloc_3483_; 
v_reuseFailAlloc_3483_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3483_, 0, v_a_3476_);
lean_ctor_set(v_reuseFailAlloc_3483_, 1, v_a_3477_);
v___x_3482_ = v_reuseFailAlloc_3483_;
goto v_reusejp_3481_;
}
v_reusejp_3481_:
{
return v___x_3482_;
}
}
}
}
else
{
lean_object* v_a_3485_; lean_object* v_a_3486_; lean_object* v___x_3488_; uint8_t v_isShared_3489_; uint8_t v_isSharedCheck_3493_; 
lean_dec(v_a_3419_);
v_a_3485_ = lean_ctor_get(v___x_3422_, 0);
v_a_3486_ = lean_ctor_get(v___x_3422_, 1);
v_isSharedCheck_3493_ = !lean_is_exclusive(v___x_3422_);
if (v_isSharedCheck_3493_ == 0)
{
v___x_3488_ = v___x_3422_;
v_isShared_3489_ = v_isSharedCheck_3493_;
goto v_resetjp_3487_;
}
else
{
lean_inc(v_a_3486_);
lean_inc(v_a_3485_);
lean_dec(v___x_3422_);
v___x_3488_ = lean_box(0);
v_isShared_3489_ = v_isSharedCheck_3493_;
goto v_resetjp_3487_;
}
v_resetjp_3487_:
{
lean_object* v___x_3491_; 
if (v_isShared_3489_ == 0)
{
v___x_3491_ = v___x_3488_;
goto v_reusejp_3490_;
}
else
{
lean_object* v_reuseFailAlloc_3492_; 
v_reuseFailAlloc_3492_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3492_, 0, v_a_3485_);
lean_ctor_set(v_reuseFailAlloc_3492_, 1, v_a_3486_);
v___x_3491_ = v_reuseFailAlloc_3492_;
goto v_reusejp_3490_;
}
v_reusejp_3490_:
{
return v___x_3491_;
}
}
}
}
else
{
lean_object* v_a_3494_; lean_object* v_a_3495_; lean_object* v___x_3497_; uint8_t v_isShared_3498_; uint8_t v_isSharedCheck_3502_; 
v_a_3494_ = lean_ctor_get(v___x_3418_, 0);
v_a_3495_ = lean_ctor_get(v___x_3418_, 1);
v_isSharedCheck_3502_ = !lean_is_exclusive(v___x_3418_);
if (v_isSharedCheck_3502_ == 0)
{
v___x_3497_ = v___x_3418_;
v_isShared_3498_ = v_isSharedCheck_3502_;
goto v_resetjp_3496_;
}
else
{
lean_inc(v_a_3495_);
lean_inc(v_a_3494_);
lean_dec(v___x_3418_);
v___x_3497_ = lean_box(0);
v_isShared_3498_ = v_isSharedCheck_3502_;
goto v_resetjp_3496_;
}
v_resetjp_3496_:
{
lean_object* v___x_3500_; 
if (v_isShared_3498_ == 0)
{
v___x_3500_ = v___x_3497_;
goto v_reusejp_3499_;
}
else
{
lean_object* v_reuseFailAlloc_3501_; 
v_reuseFailAlloc_3501_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3501_, 0, v_a_3494_);
lean_ctor_set(v_reuseFailAlloc_3501_, 1, v_a_3495_);
v___x_3500_ = v_reuseFailAlloc_3501_;
goto v_reusejp_3499_;
}
v_reusejp_3499_:
{
return v___x_3500_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readLogUpSecurityParameters(lean_object* v_a_3503_){
_start:
{
lean_object* v___x_3504_; 
v___x_3504_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_3503_);
if (lean_obj_tag(v___x_3504_) == 0)
{
lean_object* v_a_3505_; lean_object* v_a_3506_; lean_object* v___x_3507_; 
v_a_3505_ = lean_ctor_get(v___x_3504_, 0);
lean_inc(v_a_3505_);
v_a_3506_ = lean_ctor_get(v___x_3504_, 1);
lean_inc(v_a_3506_);
lean_dec_ref_known(v___x_3504_, 2);
v___x_3507_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_3506_);
if (lean_obj_tag(v___x_3507_) == 0)
{
lean_object* v_a_3508_; lean_object* v_a_3509_; lean_object* v___x_3510_; 
v_a_3508_ = lean_ctor_get(v___x_3507_, 0);
lean_inc(v_a_3508_);
v_a_3509_ = lean_ctor_get(v___x_3507_, 1);
lean_inc(v_a_3509_);
lean_dec_ref_known(v___x_3507_, 2);
v___x_3510_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_3509_);
if (lean_obj_tag(v___x_3510_) == 0)
{
lean_object* v_a_3511_; lean_object* v_a_3512_; lean_object* v___x_3514_; uint8_t v_isShared_3515_; uint8_t v_isSharedCheck_3523_; 
v_a_3511_ = lean_ctor_get(v___x_3510_, 0);
v_a_3512_ = lean_ctor_get(v___x_3510_, 1);
v_isSharedCheck_3523_ = !lean_is_exclusive(v___x_3510_);
if (v_isSharedCheck_3523_ == 0)
{
v___x_3514_ = v___x_3510_;
v_isShared_3515_ = v_isSharedCheck_3523_;
goto v_resetjp_3513_;
}
else
{
lean_inc(v_a_3512_);
lean_inc(v_a_3511_);
lean_dec(v___x_3510_);
v___x_3514_ = lean_box(0);
v_isShared_3515_ = v_isSharedCheck_3523_;
goto v_resetjp_3513_;
}
v_resetjp_3513_:
{
lean_object* v___x_3516_; uint32_t v___x_3517_; uint32_t v___x_3518_; uint32_t v___x_3519_; lean_object* v___x_3521_; 
v___x_3516_ = lean_alloc_ctor(0, 0, 12);
v___x_3517_ = lean_unbox_uint32(v_a_3505_);
lean_dec(v_a_3505_);
lean_ctor_set_uint32(v___x_3516_, 0, v___x_3517_);
v___x_3518_ = lean_unbox_uint32(v_a_3508_);
lean_dec(v_a_3508_);
lean_ctor_set_uint32(v___x_3516_, 4, v___x_3518_);
v___x_3519_ = lean_unbox_uint32(v_a_3511_);
lean_dec(v_a_3511_);
lean_ctor_set_uint32(v___x_3516_, 8, v___x_3519_);
if (v_isShared_3515_ == 0)
{
lean_ctor_set(v___x_3514_, 0, v___x_3516_);
v___x_3521_ = v___x_3514_;
goto v_reusejp_3520_;
}
else
{
lean_object* v_reuseFailAlloc_3522_; 
v_reuseFailAlloc_3522_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3522_, 0, v___x_3516_);
lean_ctor_set(v_reuseFailAlloc_3522_, 1, v_a_3512_);
v___x_3521_ = v_reuseFailAlloc_3522_;
goto v_reusejp_3520_;
}
v_reusejp_3520_:
{
return v___x_3521_;
}
}
}
else
{
lean_object* v_a_3524_; lean_object* v_a_3525_; lean_object* v___x_3527_; uint8_t v_isShared_3528_; uint8_t v_isSharedCheck_3532_; 
lean_dec(v_a_3508_);
lean_dec(v_a_3505_);
v_a_3524_ = lean_ctor_get(v___x_3510_, 0);
v_a_3525_ = lean_ctor_get(v___x_3510_, 1);
v_isSharedCheck_3532_ = !lean_is_exclusive(v___x_3510_);
if (v_isSharedCheck_3532_ == 0)
{
v___x_3527_ = v___x_3510_;
v_isShared_3528_ = v_isSharedCheck_3532_;
goto v_resetjp_3526_;
}
else
{
lean_inc(v_a_3525_);
lean_inc(v_a_3524_);
lean_dec(v___x_3510_);
v___x_3527_ = lean_box(0);
v_isShared_3528_ = v_isSharedCheck_3532_;
goto v_resetjp_3526_;
}
v_resetjp_3526_:
{
lean_object* v___x_3530_; 
if (v_isShared_3528_ == 0)
{
v___x_3530_ = v___x_3527_;
goto v_reusejp_3529_;
}
else
{
lean_object* v_reuseFailAlloc_3531_; 
v_reuseFailAlloc_3531_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3531_, 0, v_a_3524_);
lean_ctor_set(v_reuseFailAlloc_3531_, 1, v_a_3525_);
v___x_3530_ = v_reuseFailAlloc_3531_;
goto v_reusejp_3529_;
}
v_reusejp_3529_:
{
return v___x_3530_;
}
}
}
}
else
{
lean_object* v_a_3533_; lean_object* v_a_3534_; lean_object* v___x_3536_; uint8_t v_isShared_3537_; uint8_t v_isSharedCheck_3541_; 
lean_dec(v_a_3505_);
v_a_3533_ = lean_ctor_get(v___x_3507_, 0);
v_a_3534_ = lean_ctor_get(v___x_3507_, 1);
v_isSharedCheck_3541_ = !lean_is_exclusive(v___x_3507_);
if (v_isSharedCheck_3541_ == 0)
{
v___x_3536_ = v___x_3507_;
v_isShared_3537_ = v_isSharedCheck_3541_;
goto v_resetjp_3535_;
}
else
{
lean_inc(v_a_3534_);
lean_inc(v_a_3533_);
lean_dec(v___x_3507_);
v___x_3536_ = lean_box(0);
v_isShared_3537_ = v_isSharedCheck_3541_;
goto v_resetjp_3535_;
}
v_resetjp_3535_:
{
lean_object* v___x_3539_; 
if (v_isShared_3537_ == 0)
{
v___x_3539_ = v___x_3536_;
goto v_reusejp_3538_;
}
else
{
lean_object* v_reuseFailAlloc_3540_; 
v_reuseFailAlloc_3540_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3540_, 0, v_a_3533_);
lean_ctor_set(v_reuseFailAlloc_3540_, 1, v_a_3534_);
v___x_3539_ = v_reuseFailAlloc_3540_;
goto v_reusejp_3538_;
}
v_reusejp_3538_:
{
return v___x_3539_;
}
}
}
}
else
{
lean_object* v_a_3542_; lean_object* v_a_3543_; lean_object* v___x_3545_; uint8_t v_isShared_3546_; uint8_t v_isSharedCheck_3550_; 
v_a_3542_ = lean_ctor_get(v___x_3504_, 0);
v_a_3543_ = lean_ctor_get(v___x_3504_, 1);
v_isSharedCheck_3550_ = !lean_is_exclusive(v___x_3504_);
if (v_isSharedCheck_3550_ == 0)
{
v___x_3545_ = v___x_3504_;
v_isShared_3546_ = v_isSharedCheck_3550_;
goto v_resetjp_3544_;
}
else
{
lean_inc(v_a_3543_);
lean_inc(v_a_3542_);
lean_dec(v___x_3504_);
v___x_3545_ = lean_box(0);
v_isShared_3546_ = v_isSharedCheck_3550_;
goto v_resetjp_3544_;
}
v_resetjp_3544_:
{
lean_object* v___x_3548_; 
if (v_isShared_3546_ == 0)
{
v___x_3548_ = v___x_3545_;
goto v_reusejp_3547_;
}
else
{
lean_object* v_reuseFailAlloc_3549_; 
v_reuseFailAlloc_3549_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3549_, 0, v_a_3542_);
lean_ctor_set(v_reuseFailAlloc_3549_, 1, v_a_3543_);
v___x_3548_ = v_reuseFailAlloc_3549_;
goto v_reusejp_3547_;
}
v_reusejp_3547_:
{
return v___x_3548_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readSystemParams(lean_object* v_a_3551_){
_start:
{
lean_object* v___x_3552_; 
v___x_3552_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_3551_);
if (lean_obj_tag(v___x_3552_) == 0)
{
lean_object* v_a_3553_; lean_object* v_a_3554_; lean_object* v___x_3555_; 
v_a_3553_ = lean_ctor_get(v___x_3552_, 0);
lean_inc(v_a_3553_);
v_a_3554_ = lean_ctor_get(v___x_3552_, 1);
lean_inc(v_a_3554_);
lean_dec_ref_known(v___x_3552_, 2);
v___x_3555_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_3554_);
if (lean_obj_tag(v___x_3555_) == 0)
{
lean_object* v_a_3556_; lean_object* v_a_3557_; lean_object* v___x_3558_; 
v_a_3556_ = lean_ctor_get(v___x_3555_, 0);
lean_inc(v_a_3556_);
v_a_3557_ = lean_ctor_get(v___x_3555_, 1);
lean_inc(v_a_3557_);
lean_dec_ref_known(v___x_3555_, 2);
v___x_3558_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_3557_);
if (lean_obj_tag(v___x_3558_) == 0)
{
lean_object* v_a_3559_; lean_object* v_a_3560_; lean_object* v___x_3561_; 
v_a_3559_ = lean_ctor_get(v___x_3558_, 0);
lean_inc(v_a_3559_);
v_a_3560_ = lean_ctor_get(v___x_3558_, 1);
lean_inc(v_a_3560_);
lean_dec_ref_known(v___x_3558_, 2);
v___x_3561_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_3560_);
if (lean_obj_tag(v___x_3561_) == 0)
{
lean_object* v_a_3562_; lean_object* v_a_3563_; lean_object* v___x_3564_; 
v_a_3562_ = lean_ctor_get(v___x_3561_, 0);
lean_inc(v_a_3562_);
v_a_3563_ = lean_ctor_get(v___x_3561_, 1);
lean_inc(v_a_3563_);
lean_dec_ref_known(v___x_3561_, 2);
v___x_3564_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirConfig(v_a_3563_);
if (lean_obj_tag(v___x_3564_) == 0)
{
lean_object* v_a_3565_; lean_object* v_a_3566_; lean_object* v___x_3567_; 
v_a_3565_ = lean_ctor_get(v___x_3564_, 0);
lean_inc(v_a_3565_);
v_a_3566_ = lean_ctor_get(v___x_3564_, 1);
lean_inc(v_a_3566_);
lean_dec_ref_known(v___x_3564_, 2);
v___x_3567_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readLogUpSecurityParameters(v_a_3566_);
if (lean_obj_tag(v___x_3567_) == 0)
{
lean_object* v_a_3568_; lean_object* v_a_3569_; lean_object* v___x_3570_; 
v_a_3568_ = lean_ctor_get(v___x_3567_, 0);
lean_inc(v_a_3568_);
v_a_3569_ = lean_ctor_get(v___x_3567_, 1);
lean_inc(v_a_3569_);
lean_dec_ref_known(v___x_3567_, 2);
v___x_3570_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_3569_);
if (lean_obj_tag(v___x_3570_) == 0)
{
lean_object* v_a_3571_; lean_object* v_a_3572_; lean_object* v___x_3574_; uint8_t v_isShared_3575_; uint8_t v_isSharedCheck_3585_; 
v_a_3571_ = lean_ctor_get(v___x_3570_, 0);
v_a_3572_ = lean_ctor_get(v___x_3570_, 1);
v_isSharedCheck_3585_ = !lean_is_exclusive(v___x_3570_);
if (v_isSharedCheck_3585_ == 0)
{
v___x_3574_ = v___x_3570_;
v_isShared_3575_ = v_isSharedCheck_3585_;
goto v_resetjp_3573_;
}
else
{
lean_inc(v_a_3572_);
lean_inc(v_a_3571_);
lean_dec(v___x_3570_);
v___x_3574_ = lean_box(0);
v_isShared_3575_ = v_isSharedCheck_3585_;
goto v_resetjp_3573_;
}
v_resetjp_3573_:
{
lean_object* v___x_3576_; uint32_t v___x_3577_; uint32_t v___x_3578_; uint32_t v___x_3579_; uint32_t v___x_3580_; uint32_t v___x_3581_; lean_object* v___x_3583_; 
v___x_3576_ = lean_alloc_ctor(0, 2, 20);
lean_ctor_set(v___x_3576_, 0, v_a_3565_);
lean_ctor_set(v___x_3576_, 1, v_a_3568_);
v___x_3577_ = lean_unbox_uint32(v_a_3553_);
lean_dec(v_a_3553_);
lean_ctor_set_uint32(v___x_3576_, sizeof(void*)*2, v___x_3577_);
v___x_3578_ = lean_unbox_uint32(v_a_3556_);
lean_dec(v_a_3556_);
lean_ctor_set_uint32(v___x_3576_, sizeof(void*)*2 + 4, v___x_3578_);
v___x_3579_ = lean_unbox_uint32(v_a_3559_);
lean_dec(v_a_3559_);
lean_ctor_set_uint32(v___x_3576_, sizeof(void*)*2 + 8, v___x_3579_);
v___x_3580_ = lean_unbox_uint32(v_a_3562_);
lean_dec(v_a_3562_);
lean_ctor_set_uint32(v___x_3576_, sizeof(void*)*2 + 12, v___x_3580_);
v___x_3581_ = lean_unbox_uint32(v_a_3571_);
lean_dec(v_a_3571_);
lean_ctor_set_uint32(v___x_3576_, sizeof(void*)*2 + 16, v___x_3581_);
if (v_isShared_3575_ == 0)
{
lean_ctor_set(v___x_3574_, 0, v___x_3576_);
v___x_3583_ = v___x_3574_;
goto v_reusejp_3582_;
}
else
{
lean_object* v_reuseFailAlloc_3584_; 
v_reuseFailAlloc_3584_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3584_, 0, v___x_3576_);
lean_ctor_set(v_reuseFailAlloc_3584_, 1, v_a_3572_);
v___x_3583_ = v_reuseFailAlloc_3584_;
goto v_reusejp_3582_;
}
v_reusejp_3582_:
{
return v___x_3583_;
}
}
}
else
{
lean_object* v_a_3586_; lean_object* v_a_3587_; lean_object* v___x_3589_; uint8_t v_isShared_3590_; uint8_t v_isSharedCheck_3594_; 
lean_dec(v_a_3568_);
lean_dec(v_a_3565_);
lean_dec(v_a_3562_);
lean_dec(v_a_3559_);
lean_dec(v_a_3556_);
lean_dec(v_a_3553_);
v_a_3586_ = lean_ctor_get(v___x_3570_, 0);
v_a_3587_ = lean_ctor_get(v___x_3570_, 1);
v_isSharedCheck_3594_ = !lean_is_exclusive(v___x_3570_);
if (v_isSharedCheck_3594_ == 0)
{
v___x_3589_ = v___x_3570_;
v_isShared_3590_ = v_isSharedCheck_3594_;
goto v_resetjp_3588_;
}
else
{
lean_inc(v_a_3587_);
lean_inc(v_a_3586_);
lean_dec(v___x_3570_);
v___x_3589_ = lean_box(0);
v_isShared_3590_ = v_isSharedCheck_3594_;
goto v_resetjp_3588_;
}
v_resetjp_3588_:
{
lean_object* v___x_3592_; 
if (v_isShared_3590_ == 0)
{
v___x_3592_ = v___x_3589_;
goto v_reusejp_3591_;
}
else
{
lean_object* v_reuseFailAlloc_3593_; 
v_reuseFailAlloc_3593_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3593_, 0, v_a_3586_);
lean_ctor_set(v_reuseFailAlloc_3593_, 1, v_a_3587_);
v___x_3592_ = v_reuseFailAlloc_3593_;
goto v_reusejp_3591_;
}
v_reusejp_3591_:
{
return v___x_3592_;
}
}
}
}
else
{
lean_object* v_a_3595_; lean_object* v_a_3596_; lean_object* v___x_3598_; uint8_t v_isShared_3599_; uint8_t v_isSharedCheck_3603_; 
lean_dec(v_a_3565_);
lean_dec(v_a_3562_);
lean_dec(v_a_3559_);
lean_dec(v_a_3556_);
lean_dec(v_a_3553_);
v_a_3595_ = lean_ctor_get(v___x_3567_, 0);
v_a_3596_ = lean_ctor_get(v___x_3567_, 1);
v_isSharedCheck_3603_ = !lean_is_exclusive(v___x_3567_);
if (v_isSharedCheck_3603_ == 0)
{
v___x_3598_ = v___x_3567_;
v_isShared_3599_ = v_isSharedCheck_3603_;
goto v_resetjp_3597_;
}
else
{
lean_inc(v_a_3596_);
lean_inc(v_a_3595_);
lean_dec(v___x_3567_);
v___x_3598_ = lean_box(0);
v_isShared_3599_ = v_isSharedCheck_3603_;
goto v_resetjp_3597_;
}
v_resetjp_3597_:
{
lean_object* v___x_3601_; 
if (v_isShared_3599_ == 0)
{
v___x_3601_ = v___x_3598_;
goto v_reusejp_3600_;
}
else
{
lean_object* v_reuseFailAlloc_3602_; 
v_reuseFailAlloc_3602_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3602_, 0, v_a_3595_);
lean_ctor_set(v_reuseFailAlloc_3602_, 1, v_a_3596_);
v___x_3601_ = v_reuseFailAlloc_3602_;
goto v_reusejp_3600_;
}
v_reusejp_3600_:
{
return v___x_3601_;
}
}
}
}
else
{
lean_object* v_a_3604_; lean_object* v_a_3605_; lean_object* v___x_3607_; uint8_t v_isShared_3608_; uint8_t v_isSharedCheck_3612_; 
lean_dec(v_a_3562_);
lean_dec(v_a_3559_);
lean_dec(v_a_3556_);
lean_dec(v_a_3553_);
v_a_3604_ = lean_ctor_get(v___x_3564_, 0);
v_a_3605_ = lean_ctor_get(v___x_3564_, 1);
v_isSharedCheck_3612_ = !lean_is_exclusive(v___x_3564_);
if (v_isSharedCheck_3612_ == 0)
{
v___x_3607_ = v___x_3564_;
v_isShared_3608_ = v_isSharedCheck_3612_;
goto v_resetjp_3606_;
}
else
{
lean_inc(v_a_3605_);
lean_inc(v_a_3604_);
lean_dec(v___x_3564_);
v___x_3607_ = lean_box(0);
v_isShared_3608_ = v_isSharedCheck_3612_;
goto v_resetjp_3606_;
}
v_resetjp_3606_:
{
lean_object* v___x_3610_; 
if (v_isShared_3608_ == 0)
{
v___x_3610_ = v___x_3607_;
goto v_reusejp_3609_;
}
else
{
lean_object* v_reuseFailAlloc_3611_; 
v_reuseFailAlloc_3611_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3611_, 0, v_a_3604_);
lean_ctor_set(v_reuseFailAlloc_3611_, 1, v_a_3605_);
v___x_3610_ = v_reuseFailAlloc_3611_;
goto v_reusejp_3609_;
}
v_reusejp_3609_:
{
return v___x_3610_;
}
}
}
}
else
{
lean_object* v_a_3613_; lean_object* v_a_3614_; lean_object* v___x_3616_; uint8_t v_isShared_3617_; uint8_t v_isSharedCheck_3621_; 
lean_dec(v_a_3559_);
lean_dec(v_a_3556_);
lean_dec(v_a_3553_);
v_a_3613_ = lean_ctor_get(v___x_3561_, 0);
v_a_3614_ = lean_ctor_get(v___x_3561_, 1);
v_isSharedCheck_3621_ = !lean_is_exclusive(v___x_3561_);
if (v_isSharedCheck_3621_ == 0)
{
v___x_3616_ = v___x_3561_;
v_isShared_3617_ = v_isSharedCheck_3621_;
goto v_resetjp_3615_;
}
else
{
lean_inc(v_a_3614_);
lean_inc(v_a_3613_);
lean_dec(v___x_3561_);
v___x_3616_ = lean_box(0);
v_isShared_3617_ = v_isSharedCheck_3621_;
goto v_resetjp_3615_;
}
v_resetjp_3615_:
{
lean_object* v___x_3619_; 
if (v_isShared_3617_ == 0)
{
v___x_3619_ = v___x_3616_;
goto v_reusejp_3618_;
}
else
{
lean_object* v_reuseFailAlloc_3620_; 
v_reuseFailAlloc_3620_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3620_, 0, v_a_3613_);
lean_ctor_set(v_reuseFailAlloc_3620_, 1, v_a_3614_);
v___x_3619_ = v_reuseFailAlloc_3620_;
goto v_reusejp_3618_;
}
v_reusejp_3618_:
{
return v___x_3619_;
}
}
}
}
else
{
lean_object* v_a_3622_; lean_object* v_a_3623_; lean_object* v___x_3625_; uint8_t v_isShared_3626_; uint8_t v_isSharedCheck_3630_; 
lean_dec(v_a_3556_);
lean_dec(v_a_3553_);
v_a_3622_ = lean_ctor_get(v___x_3558_, 0);
v_a_3623_ = lean_ctor_get(v___x_3558_, 1);
v_isSharedCheck_3630_ = !lean_is_exclusive(v___x_3558_);
if (v_isSharedCheck_3630_ == 0)
{
v___x_3625_ = v___x_3558_;
v_isShared_3626_ = v_isSharedCheck_3630_;
goto v_resetjp_3624_;
}
else
{
lean_inc(v_a_3623_);
lean_inc(v_a_3622_);
lean_dec(v___x_3558_);
v___x_3625_ = lean_box(0);
v_isShared_3626_ = v_isSharedCheck_3630_;
goto v_resetjp_3624_;
}
v_resetjp_3624_:
{
lean_object* v___x_3628_; 
if (v_isShared_3626_ == 0)
{
v___x_3628_ = v___x_3625_;
goto v_reusejp_3627_;
}
else
{
lean_object* v_reuseFailAlloc_3629_; 
v_reuseFailAlloc_3629_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3629_, 0, v_a_3622_);
lean_ctor_set(v_reuseFailAlloc_3629_, 1, v_a_3623_);
v___x_3628_ = v_reuseFailAlloc_3629_;
goto v_reusejp_3627_;
}
v_reusejp_3627_:
{
return v___x_3628_;
}
}
}
}
else
{
lean_object* v_a_3631_; lean_object* v_a_3632_; lean_object* v___x_3634_; uint8_t v_isShared_3635_; uint8_t v_isSharedCheck_3639_; 
lean_dec(v_a_3553_);
v_a_3631_ = lean_ctor_get(v___x_3555_, 0);
v_a_3632_ = lean_ctor_get(v___x_3555_, 1);
v_isSharedCheck_3639_ = !lean_is_exclusive(v___x_3555_);
if (v_isSharedCheck_3639_ == 0)
{
v___x_3634_ = v___x_3555_;
v_isShared_3635_ = v_isSharedCheck_3639_;
goto v_resetjp_3633_;
}
else
{
lean_inc(v_a_3632_);
lean_inc(v_a_3631_);
lean_dec(v___x_3555_);
v___x_3634_ = lean_box(0);
v_isShared_3635_ = v_isSharedCheck_3639_;
goto v_resetjp_3633_;
}
v_resetjp_3633_:
{
lean_object* v___x_3637_; 
if (v_isShared_3635_ == 0)
{
v___x_3637_ = v___x_3634_;
goto v_reusejp_3636_;
}
else
{
lean_object* v_reuseFailAlloc_3638_; 
v_reuseFailAlloc_3638_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3638_, 0, v_a_3631_);
lean_ctor_set(v_reuseFailAlloc_3638_, 1, v_a_3632_);
v___x_3637_ = v_reuseFailAlloc_3638_;
goto v_reusejp_3636_;
}
v_reusejp_3636_:
{
return v___x_3637_;
}
}
}
}
else
{
lean_object* v_a_3640_; lean_object* v_a_3641_; lean_object* v___x_3643_; uint8_t v_isShared_3644_; uint8_t v_isSharedCheck_3648_; 
v_a_3640_ = lean_ctor_get(v___x_3552_, 0);
v_a_3641_ = lean_ctor_get(v___x_3552_, 1);
v_isSharedCheck_3648_ = !lean_is_exclusive(v___x_3552_);
if (v_isSharedCheck_3648_ == 0)
{
v___x_3643_ = v___x_3552_;
v_isShared_3644_ = v_isSharedCheck_3648_;
goto v_resetjp_3642_;
}
else
{
lean_inc(v_a_3641_);
lean_inc(v_a_3640_);
lean_dec(v___x_3552_);
v___x_3643_ = lean_box(0);
v_isShared_3644_ = v_isSharedCheck_3648_;
goto v_resetjp_3642_;
}
v_resetjp_3642_:
{
lean_object* v___x_3646_; 
if (v_isShared_3644_ == 0)
{
v___x_3646_ = v___x_3643_;
goto v_reusejp_3645_;
}
else
{
lean_object* v_reuseFailAlloc_3647_; 
v_reuseFailAlloc_3647_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3647_, 0, v_a_3640_);
lean_ctor_set(v_reuseFailAlloc_3647_, 1, v_a_3641_);
v___x_3646_ = v_reuseFailAlloc_3647_;
goto v_reusejp_3645_;
}
v_reusejp_3645_:
{
return v___x_3646_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readVerifierSinglePreprocessedData(lean_object* v_a_3649_){
_start:
{
lean_object* v___x_3650_; 
v___x_3650_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readDigest(v_a_3649_);
if (lean_obj_tag(v___x_3650_) == 0)
{
lean_object* v_a_3651_; lean_object* v_a_3652_; lean_object* v___x_3653_; 
v_a_3651_ = lean_ctor_get(v___x_3650_, 0);
lean_inc(v_a_3651_);
v_a_3652_ = lean_ctor_get(v___x_3650_, 1);
lean_inc(v_a_3652_);
lean_dec_ref_known(v___x_3650_, 2);
v___x_3653_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readInt32(v_a_3652_);
if (lean_obj_tag(v___x_3653_) == 0)
{
lean_object* v_a_3654_; lean_object* v_a_3655_; lean_object* v___x_3656_; 
v_a_3654_ = lean_ctor_get(v___x_3653_, 0);
lean_inc(v_a_3654_);
v_a_3655_ = lean_ctor_get(v___x_3653_, 1);
lean_inc(v_a_3655_);
lean_dec_ref_known(v___x_3653_, 2);
v___x_3656_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_3655_);
if (lean_obj_tag(v___x_3656_) == 0)
{
lean_object* v_a_3657_; lean_object* v_a_3658_; lean_object* v___x_3660_; uint8_t v_isShared_3661_; uint8_t v_isSharedCheck_3667_; 
v_a_3657_ = lean_ctor_get(v___x_3656_, 0);
v_a_3658_ = lean_ctor_get(v___x_3656_, 1);
v_isSharedCheck_3667_ = !lean_is_exclusive(v___x_3656_);
if (v_isSharedCheck_3667_ == 0)
{
v___x_3660_ = v___x_3656_;
v_isShared_3661_ = v_isSharedCheck_3667_;
goto v_resetjp_3659_;
}
else
{
lean_inc(v_a_3658_);
lean_inc(v_a_3657_);
lean_dec(v___x_3656_);
v___x_3660_ = lean_box(0);
v_isShared_3661_ = v_isSharedCheck_3667_;
goto v_resetjp_3659_;
}
v_resetjp_3659_:
{
lean_object* v___x_3662_; uint32_t v___x_3663_; lean_object* v___x_3665_; 
v___x_3662_ = lean_alloc_ctor(0, 2, 4);
lean_ctor_set(v___x_3662_, 0, v_a_3651_);
lean_ctor_set(v___x_3662_, 1, v_a_3654_);
v___x_3663_ = lean_unbox_uint32(v_a_3657_);
lean_dec(v_a_3657_);
lean_ctor_set_uint32(v___x_3662_, sizeof(void*)*2, v___x_3663_);
if (v_isShared_3661_ == 0)
{
lean_ctor_set(v___x_3660_, 0, v___x_3662_);
v___x_3665_ = v___x_3660_;
goto v_reusejp_3664_;
}
else
{
lean_object* v_reuseFailAlloc_3666_; 
v_reuseFailAlloc_3666_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3666_, 0, v___x_3662_);
lean_ctor_set(v_reuseFailAlloc_3666_, 1, v_a_3658_);
v___x_3665_ = v_reuseFailAlloc_3666_;
goto v_reusejp_3664_;
}
v_reusejp_3664_:
{
return v___x_3665_;
}
}
}
else
{
lean_object* v_a_3668_; lean_object* v_a_3669_; lean_object* v___x_3671_; uint8_t v_isShared_3672_; uint8_t v_isSharedCheck_3676_; 
lean_dec(v_a_3654_);
lean_dec(v_a_3651_);
v_a_3668_ = lean_ctor_get(v___x_3656_, 0);
v_a_3669_ = lean_ctor_get(v___x_3656_, 1);
v_isSharedCheck_3676_ = !lean_is_exclusive(v___x_3656_);
if (v_isSharedCheck_3676_ == 0)
{
v___x_3671_ = v___x_3656_;
v_isShared_3672_ = v_isSharedCheck_3676_;
goto v_resetjp_3670_;
}
else
{
lean_inc(v_a_3669_);
lean_inc(v_a_3668_);
lean_dec(v___x_3656_);
v___x_3671_ = lean_box(0);
v_isShared_3672_ = v_isSharedCheck_3676_;
goto v_resetjp_3670_;
}
v_resetjp_3670_:
{
lean_object* v___x_3674_; 
if (v_isShared_3672_ == 0)
{
v___x_3674_ = v___x_3671_;
goto v_reusejp_3673_;
}
else
{
lean_object* v_reuseFailAlloc_3675_; 
v_reuseFailAlloc_3675_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3675_, 0, v_a_3668_);
lean_ctor_set(v_reuseFailAlloc_3675_, 1, v_a_3669_);
v___x_3674_ = v_reuseFailAlloc_3675_;
goto v_reusejp_3673_;
}
v_reusejp_3673_:
{
return v___x_3674_;
}
}
}
}
else
{
lean_object* v_a_3677_; lean_object* v_a_3678_; lean_object* v___x_3680_; uint8_t v_isShared_3681_; uint8_t v_isSharedCheck_3685_; 
lean_dec(v_a_3651_);
v_a_3677_ = lean_ctor_get(v___x_3653_, 0);
v_a_3678_ = lean_ctor_get(v___x_3653_, 1);
v_isSharedCheck_3685_ = !lean_is_exclusive(v___x_3653_);
if (v_isSharedCheck_3685_ == 0)
{
v___x_3680_ = v___x_3653_;
v_isShared_3681_ = v_isSharedCheck_3685_;
goto v_resetjp_3679_;
}
else
{
lean_inc(v_a_3678_);
lean_inc(v_a_3677_);
lean_dec(v___x_3653_);
v___x_3680_ = lean_box(0);
v_isShared_3681_ = v_isSharedCheck_3685_;
goto v_resetjp_3679_;
}
v_resetjp_3679_:
{
lean_object* v___x_3683_; 
if (v_isShared_3681_ == 0)
{
v___x_3683_ = v___x_3680_;
goto v_reusejp_3682_;
}
else
{
lean_object* v_reuseFailAlloc_3684_; 
v_reuseFailAlloc_3684_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3684_, 0, v_a_3677_);
lean_ctor_set(v_reuseFailAlloc_3684_, 1, v_a_3678_);
v___x_3683_ = v_reuseFailAlloc_3684_;
goto v_reusejp_3682_;
}
v_reusejp_3682_:
{
return v___x_3683_;
}
}
}
}
else
{
lean_object* v_a_3686_; lean_object* v_a_3687_; lean_object* v___x_3689_; uint8_t v_isShared_3690_; uint8_t v_isSharedCheck_3694_; 
v_a_3686_ = lean_ctor_get(v___x_3650_, 0);
v_a_3687_ = lean_ctor_get(v___x_3650_, 1);
v_isSharedCheck_3694_ = !lean_is_exclusive(v___x_3650_);
if (v_isSharedCheck_3694_ == 0)
{
v___x_3689_ = v___x_3650_;
v_isShared_3690_ = v_isSharedCheck_3694_;
goto v_resetjp_3688_;
}
else
{
lean_inc(v_a_3687_);
lean_inc(v_a_3686_);
lean_dec(v___x_3650_);
v___x_3689_ = lean_box(0);
v_isShared_3690_ = v_isSharedCheck_3694_;
goto v_resetjp_3688_;
}
v_resetjp_3688_:
{
lean_object* v___x_3692_; 
if (v_isShared_3690_ == 0)
{
v___x_3692_ = v___x_3689_;
goto v_reusejp_3691_;
}
else
{
lean_object* v_reuseFailAlloc_3693_; 
v_reuseFailAlloc_3693_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3693_, 0, v_a_3686_);
lean_ctor_set(v_reuseFailAlloc_3693_, 1, v_a_3687_);
v___x_3692_ = v_reuseFailAlloc_3693_;
goto v_reusejp_3691_;
}
v_reusejp_3691_:
{
return v___x_3692_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readTraceWidth(lean_object* v_a_3695_){
_start:
{
lean_object* v___x_3696_; lean_object* v___x_3697_; 
v___x_3696_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32), 1, 0);
lean_inc_ref(v___x_3696_);
v___x_3697_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readOption___redArg(v___x_3696_, v_a_3695_);
if (lean_obj_tag(v___x_3697_) == 0)
{
lean_object* v_a_3698_; lean_object* v_a_3699_; lean_object* v___x_3700_; 
v_a_3698_ = lean_ctor_get(v___x_3697_, 0);
lean_inc(v_a_3698_);
v_a_3699_ = lean_ctor_get(v___x_3697_, 1);
lean_inc(v_a_3699_);
lean_dec_ref_known(v___x_3697_, 2);
v___x_3700_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_3696_, v_a_3699_);
if (lean_obj_tag(v___x_3700_) == 0)
{
lean_object* v_a_3701_; lean_object* v_a_3702_; lean_object* v___x_3703_; 
v_a_3701_ = lean_ctor_get(v___x_3700_, 0);
lean_inc(v_a_3701_);
v_a_3702_ = lean_ctor_get(v___x_3700_, 1);
lean_inc(v_a_3702_);
lean_dec_ref_known(v___x_3700_, 2);
v___x_3703_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_3702_);
if (lean_obj_tag(v___x_3703_) == 0)
{
lean_object* v_a_3704_; lean_object* v_a_3705_; lean_object* v___x_3707_; uint8_t v_isShared_3708_; uint8_t v_isSharedCheck_3714_; 
v_a_3704_ = lean_ctor_get(v___x_3703_, 0);
v_a_3705_ = lean_ctor_get(v___x_3703_, 1);
v_isSharedCheck_3714_ = !lean_is_exclusive(v___x_3703_);
if (v_isSharedCheck_3714_ == 0)
{
v___x_3707_ = v___x_3703_;
v_isShared_3708_ = v_isSharedCheck_3714_;
goto v_resetjp_3706_;
}
else
{
lean_inc(v_a_3705_);
lean_inc(v_a_3704_);
lean_dec(v___x_3703_);
v___x_3707_ = lean_box(0);
v_isShared_3708_ = v_isSharedCheck_3714_;
goto v_resetjp_3706_;
}
v_resetjp_3706_:
{
lean_object* v___x_3709_; uint32_t v___x_3710_; lean_object* v___x_3712_; 
v___x_3709_ = lean_alloc_ctor(0, 2, 4);
lean_ctor_set(v___x_3709_, 0, v_a_3698_);
lean_ctor_set(v___x_3709_, 1, v_a_3701_);
v___x_3710_ = lean_unbox_uint32(v_a_3704_);
lean_dec(v_a_3704_);
lean_ctor_set_uint32(v___x_3709_, sizeof(void*)*2, v___x_3710_);
if (v_isShared_3708_ == 0)
{
lean_ctor_set(v___x_3707_, 0, v___x_3709_);
v___x_3712_ = v___x_3707_;
goto v_reusejp_3711_;
}
else
{
lean_object* v_reuseFailAlloc_3713_; 
v_reuseFailAlloc_3713_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3713_, 0, v___x_3709_);
lean_ctor_set(v_reuseFailAlloc_3713_, 1, v_a_3705_);
v___x_3712_ = v_reuseFailAlloc_3713_;
goto v_reusejp_3711_;
}
v_reusejp_3711_:
{
return v___x_3712_;
}
}
}
else
{
lean_object* v_a_3715_; lean_object* v_a_3716_; lean_object* v___x_3718_; uint8_t v_isShared_3719_; uint8_t v_isSharedCheck_3723_; 
lean_dec(v_a_3701_);
lean_dec(v_a_3698_);
v_a_3715_ = lean_ctor_get(v___x_3703_, 0);
v_a_3716_ = lean_ctor_get(v___x_3703_, 1);
v_isSharedCheck_3723_ = !lean_is_exclusive(v___x_3703_);
if (v_isSharedCheck_3723_ == 0)
{
v___x_3718_ = v___x_3703_;
v_isShared_3719_ = v_isSharedCheck_3723_;
goto v_resetjp_3717_;
}
else
{
lean_inc(v_a_3716_);
lean_inc(v_a_3715_);
lean_dec(v___x_3703_);
v___x_3718_ = lean_box(0);
v_isShared_3719_ = v_isSharedCheck_3723_;
goto v_resetjp_3717_;
}
v_resetjp_3717_:
{
lean_object* v___x_3721_; 
if (v_isShared_3719_ == 0)
{
v___x_3721_ = v___x_3718_;
goto v_reusejp_3720_;
}
else
{
lean_object* v_reuseFailAlloc_3722_; 
v_reuseFailAlloc_3722_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3722_, 0, v_a_3715_);
lean_ctor_set(v_reuseFailAlloc_3722_, 1, v_a_3716_);
v___x_3721_ = v_reuseFailAlloc_3722_;
goto v_reusejp_3720_;
}
v_reusejp_3720_:
{
return v___x_3721_;
}
}
}
}
else
{
lean_object* v_a_3724_; lean_object* v_a_3725_; lean_object* v___x_3727_; uint8_t v_isShared_3728_; uint8_t v_isSharedCheck_3732_; 
lean_dec(v_a_3698_);
v_a_3724_ = lean_ctor_get(v___x_3700_, 0);
v_a_3725_ = lean_ctor_get(v___x_3700_, 1);
v_isSharedCheck_3732_ = !lean_is_exclusive(v___x_3700_);
if (v_isSharedCheck_3732_ == 0)
{
v___x_3727_ = v___x_3700_;
v_isShared_3728_ = v_isSharedCheck_3732_;
goto v_resetjp_3726_;
}
else
{
lean_inc(v_a_3725_);
lean_inc(v_a_3724_);
lean_dec(v___x_3700_);
v___x_3727_ = lean_box(0);
v_isShared_3728_ = v_isSharedCheck_3732_;
goto v_resetjp_3726_;
}
v_resetjp_3726_:
{
lean_object* v___x_3730_; 
if (v_isShared_3728_ == 0)
{
v___x_3730_ = v___x_3727_;
goto v_reusejp_3729_;
}
else
{
lean_object* v_reuseFailAlloc_3731_; 
v_reuseFailAlloc_3731_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3731_, 0, v_a_3724_);
lean_ctor_set(v_reuseFailAlloc_3731_, 1, v_a_3725_);
v___x_3730_ = v_reuseFailAlloc_3731_;
goto v_reusejp_3729_;
}
v_reusejp_3729_:
{
return v___x_3730_;
}
}
}
}
else
{
lean_object* v_a_3733_; lean_object* v_a_3734_; lean_object* v___x_3736_; uint8_t v_isShared_3737_; uint8_t v_isSharedCheck_3741_; 
lean_dec_ref(v___x_3696_);
v_a_3733_ = lean_ctor_get(v___x_3697_, 0);
v_a_3734_ = lean_ctor_get(v___x_3697_, 1);
v_isSharedCheck_3741_ = !lean_is_exclusive(v___x_3697_);
if (v_isSharedCheck_3741_ == 0)
{
v___x_3736_ = v___x_3697_;
v_isShared_3737_ = v_isSharedCheck_3741_;
goto v_resetjp_3735_;
}
else
{
lean_inc(v_a_3734_);
lean_inc(v_a_3733_);
lean_dec(v___x_3697_);
v___x_3736_ = lean_box(0);
v_isShared_3737_ = v_isSharedCheck_3741_;
goto v_resetjp_3735_;
}
v_resetjp_3735_:
{
lean_object* v___x_3739_; 
if (v_isShared_3737_ == 0)
{
v___x_3739_ = v___x_3736_;
goto v_reusejp_3738_;
}
else
{
lean_object* v_reuseFailAlloc_3740_; 
v_reuseFailAlloc_3740_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3740_, 0, v_a_3733_);
lean_ctor_set(v_reuseFailAlloc_3740_, 1, v_a_3734_);
v___x_3739_ = v_reuseFailAlloc_3740_;
goto v_reusejp_3738_;
}
v_reusejp_3738_:
{
return v___x_3739_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readStarkVerifyingParams(lean_object* v_a_3742_){
_start:
{
lean_object* v___x_3743_; 
v___x_3743_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readTraceWidth(v_a_3742_);
if (lean_obj_tag(v___x_3743_) == 0)
{
lean_object* v_a_3744_; lean_object* v_a_3745_; lean_object* v___x_3746_; 
v_a_3744_ = lean_ctor_get(v___x_3743_, 0);
lean_inc(v_a_3744_);
v_a_3745_ = lean_ctor_get(v___x_3743_, 1);
lean_inc(v_a_3745_);
lean_dec_ref_known(v___x_3743_, 2);
v___x_3746_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_3745_);
if (lean_obj_tag(v___x_3746_) == 0)
{
lean_object* v_a_3747_; lean_object* v_a_3748_; lean_object* v___x_3749_; 
v_a_3747_ = lean_ctor_get(v___x_3746_, 0);
lean_inc(v_a_3747_);
v_a_3748_ = lean_ctor_get(v___x_3746_, 1);
lean_inc(v_a_3748_);
lean_dec_ref_known(v___x_3746_, 2);
v___x_3749_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBool(v_a_3748_);
if (lean_obj_tag(v___x_3749_) == 0)
{
lean_object* v_a_3750_; lean_object* v_a_3751_; lean_object* v___x_3753_; uint8_t v_isShared_3754_; uint8_t v_isSharedCheck_3761_; 
v_a_3750_ = lean_ctor_get(v___x_3749_, 0);
v_a_3751_ = lean_ctor_get(v___x_3749_, 1);
v_isSharedCheck_3761_ = !lean_is_exclusive(v___x_3749_);
if (v_isSharedCheck_3761_ == 0)
{
v___x_3753_ = v___x_3749_;
v_isShared_3754_ = v_isSharedCheck_3761_;
goto v_resetjp_3752_;
}
else
{
lean_inc(v_a_3751_);
lean_inc(v_a_3750_);
lean_dec(v___x_3749_);
v___x_3753_ = lean_box(0);
v_isShared_3754_ = v_isSharedCheck_3761_;
goto v_resetjp_3752_;
}
v_resetjp_3752_:
{
lean_object* v___x_3755_; uint32_t v___x_3756_; uint8_t v___x_3757_; lean_object* v___x_3759_; 
v___x_3755_ = lean_alloc_ctor(0, 1, 5);
lean_ctor_set(v___x_3755_, 0, v_a_3744_);
v___x_3756_ = lean_unbox_uint32(v_a_3747_);
lean_dec(v_a_3747_);
lean_ctor_set_uint32(v___x_3755_, sizeof(void*)*1, v___x_3756_);
v___x_3757_ = lean_unbox(v_a_3750_);
lean_dec(v_a_3750_);
lean_ctor_set_uint8(v___x_3755_, sizeof(void*)*1 + 4, v___x_3757_);
if (v_isShared_3754_ == 0)
{
lean_ctor_set(v___x_3753_, 0, v___x_3755_);
v___x_3759_ = v___x_3753_;
goto v_reusejp_3758_;
}
else
{
lean_object* v_reuseFailAlloc_3760_; 
v_reuseFailAlloc_3760_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3760_, 0, v___x_3755_);
lean_ctor_set(v_reuseFailAlloc_3760_, 1, v_a_3751_);
v___x_3759_ = v_reuseFailAlloc_3760_;
goto v_reusejp_3758_;
}
v_reusejp_3758_:
{
return v___x_3759_;
}
}
}
else
{
lean_object* v_a_3762_; lean_object* v_a_3763_; lean_object* v___x_3765_; uint8_t v_isShared_3766_; uint8_t v_isSharedCheck_3770_; 
lean_dec(v_a_3747_);
lean_dec(v_a_3744_);
v_a_3762_ = lean_ctor_get(v___x_3749_, 0);
v_a_3763_ = lean_ctor_get(v___x_3749_, 1);
v_isSharedCheck_3770_ = !lean_is_exclusive(v___x_3749_);
if (v_isSharedCheck_3770_ == 0)
{
v___x_3765_ = v___x_3749_;
v_isShared_3766_ = v_isSharedCheck_3770_;
goto v_resetjp_3764_;
}
else
{
lean_inc(v_a_3763_);
lean_inc(v_a_3762_);
lean_dec(v___x_3749_);
v___x_3765_ = lean_box(0);
v_isShared_3766_ = v_isSharedCheck_3770_;
goto v_resetjp_3764_;
}
v_resetjp_3764_:
{
lean_object* v___x_3768_; 
if (v_isShared_3766_ == 0)
{
v___x_3768_ = v___x_3765_;
goto v_reusejp_3767_;
}
else
{
lean_object* v_reuseFailAlloc_3769_; 
v_reuseFailAlloc_3769_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3769_, 0, v_a_3762_);
lean_ctor_set(v_reuseFailAlloc_3769_, 1, v_a_3763_);
v___x_3768_ = v_reuseFailAlloc_3769_;
goto v_reusejp_3767_;
}
v_reusejp_3767_:
{
return v___x_3768_;
}
}
}
}
else
{
lean_object* v_a_3771_; lean_object* v_a_3772_; lean_object* v___x_3774_; uint8_t v_isShared_3775_; uint8_t v_isSharedCheck_3779_; 
lean_dec(v_a_3744_);
v_a_3771_ = lean_ctor_get(v___x_3746_, 0);
v_a_3772_ = lean_ctor_get(v___x_3746_, 1);
v_isSharedCheck_3779_ = !lean_is_exclusive(v___x_3746_);
if (v_isSharedCheck_3779_ == 0)
{
v___x_3774_ = v___x_3746_;
v_isShared_3775_ = v_isSharedCheck_3779_;
goto v_resetjp_3773_;
}
else
{
lean_inc(v_a_3772_);
lean_inc(v_a_3771_);
lean_dec(v___x_3746_);
v___x_3774_ = lean_box(0);
v_isShared_3775_ = v_isSharedCheck_3779_;
goto v_resetjp_3773_;
}
v_resetjp_3773_:
{
lean_object* v___x_3777_; 
if (v_isShared_3775_ == 0)
{
v___x_3777_ = v___x_3774_;
goto v_reusejp_3776_;
}
else
{
lean_object* v_reuseFailAlloc_3778_; 
v_reuseFailAlloc_3778_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3778_, 0, v_a_3771_);
lean_ctor_set(v_reuseFailAlloc_3778_, 1, v_a_3772_);
v___x_3777_ = v_reuseFailAlloc_3778_;
goto v_reusejp_3776_;
}
v_reusejp_3776_:
{
return v___x_3777_;
}
}
}
}
else
{
lean_object* v_a_3780_; lean_object* v_a_3781_; lean_object* v___x_3783_; uint8_t v_isShared_3784_; uint8_t v_isSharedCheck_3788_; 
v_a_3780_ = lean_ctor_get(v___x_3743_, 0);
v_a_3781_ = lean_ctor_get(v___x_3743_, 1);
v_isSharedCheck_3788_ = !lean_is_exclusive(v___x_3743_);
if (v_isSharedCheck_3788_ == 0)
{
v___x_3783_ = v___x_3743_;
v_isShared_3784_ = v_isSharedCheck_3788_;
goto v_resetjp_3782_;
}
else
{
lean_inc(v_a_3781_);
lean_inc(v_a_3780_);
lean_dec(v___x_3743_);
v___x_3783_ = lean_box(0);
v_isShared_3784_ = v_isSharedCheck_3788_;
goto v_resetjp_3782_;
}
v_resetjp_3782_:
{
lean_object* v___x_3786_; 
if (v_isShared_3784_ == 0)
{
v___x_3786_ = v___x_3783_;
goto v_reusejp_3785_;
}
else
{
lean_object* v_reuseFailAlloc_3787_; 
v_reuseFailAlloc_3787_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3787_, 0, v_a_3780_);
lean_ctor_set(v_reuseFailAlloc_3787_, 1, v_a_3781_);
v___x_3786_ = v_reuseFailAlloc_3787_;
goto v_reusejp_3785_;
}
v_reusejp_3785_:
{
return v___x_3786_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readLinearConstraint(lean_object* v_a_3789_){
_start:
{
lean_object* v___x_3790_; lean_object* v___x_3791_; 
v___x_3790_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32), 1, 0);
v___x_3791_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_3790_, v_a_3789_);
if (lean_obj_tag(v___x_3791_) == 0)
{
lean_object* v_a_3792_; lean_object* v_a_3793_; lean_object* v___x_3794_; 
v_a_3792_ = lean_ctor_get(v___x_3791_, 0);
lean_inc(v_a_3792_);
v_a_3793_ = lean_ctor_get(v___x_3791_, 1);
lean_inc(v_a_3793_);
lean_dec_ref_known(v___x_3791_, 2);
v___x_3794_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_3793_);
if (lean_obj_tag(v___x_3794_) == 0)
{
lean_object* v_a_3795_; lean_object* v_a_3796_; lean_object* v___x_3798_; uint8_t v_isShared_3799_; uint8_t v_isSharedCheck_3805_; 
v_a_3795_ = lean_ctor_get(v___x_3794_, 0);
v_a_3796_ = lean_ctor_get(v___x_3794_, 1);
v_isSharedCheck_3805_ = !lean_is_exclusive(v___x_3794_);
if (v_isSharedCheck_3805_ == 0)
{
v___x_3798_ = v___x_3794_;
v_isShared_3799_ = v_isSharedCheck_3805_;
goto v_resetjp_3797_;
}
else
{
lean_inc(v_a_3796_);
lean_inc(v_a_3795_);
lean_dec(v___x_3794_);
v___x_3798_ = lean_box(0);
v_isShared_3799_ = v_isSharedCheck_3805_;
goto v_resetjp_3797_;
}
v_resetjp_3797_:
{
lean_object* v___x_3800_; uint32_t v___x_3801_; lean_object* v___x_3803_; 
v___x_3800_ = lean_alloc_ctor(0, 1, 4);
lean_ctor_set(v___x_3800_, 0, v_a_3792_);
v___x_3801_ = lean_unbox_uint32(v_a_3795_);
lean_dec(v_a_3795_);
lean_ctor_set_uint32(v___x_3800_, sizeof(void*)*1, v___x_3801_);
if (v_isShared_3799_ == 0)
{
lean_ctor_set(v___x_3798_, 0, v___x_3800_);
v___x_3803_ = v___x_3798_;
goto v_reusejp_3802_;
}
else
{
lean_object* v_reuseFailAlloc_3804_; 
v_reuseFailAlloc_3804_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3804_, 0, v___x_3800_);
lean_ctor_set(v_reuseFailAlloc_3804_, 1, v_a_3796_);
v___x_3803_ = v_reuseFailAlloc_3804_;
goto v_reusejp_3802_;
}
v_reusejp_3802_:
{
return v___x_3803_;
}
}
}
else
{
lean_object* v_a_3806_; lean_object* v_a_3807_; lean_object* v___x_3809_; uint8_t v_isShared_3810_; uint8_t v_isSharedCheck_3814_; 
lean_dec(v_a_3792_);
v_a_3806_ = lean_ctor_get(v___x_3794_, 0);
v_a_3807_ = lean_ctor_get(v___x_3794_, 1);
v_isSharedCheck_3814_ = !lean_is_exclusive(v___x_3794_);
if (v_isSharedCheck_3814_ == 0)
{
v___x_3809_ = v___x_3794_;
v_isShared_3810_ = v_isSharedCheck_3814_;
goto v_resetjp_3808_;
}
else
{
lean_inc(v_a_3807_);
lean_inc(v_a_3806_);
lean_dec(v___x_3794_);
v___x_3809_ = lean_box(0);
v_isShared_3810_ = v_isSharedCheck_3814_;
goto v_resetjp_3808_;
}
v_resetjp_3808_:
{
lean_object* v___x_3812_; 
if (v_isShared_3810_ == 0)
{
v___x_3812_ = v___x_3809_;
goto v_reusejp_3811_;
}
else
{
lean_object* v_reuseFailAlloc_3813_; 
v_reuseFailAlloc_3813_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3813_, 0, v_a_3806_);
lean_ctor_set(v_reuseFailAlloc_3813_, 1, v_a_3807_);
v___x_3812_ = v_reuseFailAlloc_3813_;
goto v_reusejp_3811_;
}
v_reusejp_3811_:
{
return v___x_3812_;
}
}
}
}
else
{
lean_object* v_a_3815_; lean_object* v_a_3816_; lean_object* v___x_3818_; uint8_t v_isShared_3819_; uint8_t v_isSharedCheck_3823_; 
v_a_3815_ = lean_ctor_get(v___x_3791_, 0);
v_a_3816_ = lean_ctor_get(v___x_3791_, 1);
v_isSharedCheck_3823_ = !lean_is_exclusive(v___x_3791_);
if (v_isSharedCheck_3823_ == 0)
{
v___x_3818_ = v___x_3791_;
v_isShared_3819_ = v_isSharedCheck_3823_;
goto v_resetjp_3817_;
}
else
{
lean_inc(v_a_3816_);
lean_inc(v_a_3815_);
lean_dec(v___x_3791_);
v___x_3818_ = lean_box(0);
v_isShared_3819_ = v_isSharedCheck_3823_;
goto v_resetjp_3817_;
}
v_resetjp_3817_:
{
lean_object* v___x_3821_; 
if (v_isShared_3819_ == 0)
{
v___x_3821_ = v___x_3818_;
goto v_reusejp_3820_;
}
else
{
lean_object* v_reuseFailAlloc_3822_; 
v_reuseFailAlloc_3822_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3822_, 0, v_a_3815_);
lean_ctor_set(v_reuseFailAlloc_3822_, 1, v_a_3816_);
v___x_3821_ = v_reuseFailAlloc_3822_;
goto v_reusejp_3820_;
}
v_reusejp_3820_:
{
return v___x_3821_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readStarkVerifyingKey(lean_object* v_a_3824_){
_start:
{
lean_object* v___x_3825_; lean_object* v___x_3826_; 
v___x_3825_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readVerifierSinglePreprocessedData), 1, 0);
v___x_3826_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readOption___redArg(v___x_3825_, v_a_3824_);
if (lean_obj_tag(v___x_3826_) == 0)
{
lean_object* v_a_3827_; lean_object* v_a_3828_; lean_object* v___x_3829_; 
v_a_3827_ = lean_ctor_get(v___x_3826_, 0);
lean_inc(v_a_3827_);
v_a_3828_ = lean_ctor_get(v___x_3826_, 1);
lean_inc(v_a_3828_);
lean_dec_ref_known(v___x_3826_, 2);
v___x_3829_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readStarkVerifyingParams(v_a_3828_);
if (lean_obj_tag(v___x_3829_) == 0)
{
lean_object* v_a_3830_; lean_object* v_a_3831_; lean_object* v___x_3832_; 
v_a_3830_ = lean_ctor_get(v___x_3829_, 0);
lean_inc(v_a_3830_);
v_a_3831_ = lean_ctor_get(v___x_3829_, 1);
lean_inc(v_a_3831_);
lean_dec_ref_known(v___x_3829_, 2);
v___x_3832_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readSymbolicConstraintsDag(v_a_3831_);
if (lean_obj_tag(v___x_3832_) == 0)
{
lean_object* v_a_3833_; lean_object* v_a_3834_; lean_object* v___x_3835_; 
v_a_3833_ = lean_ctor_get(v___x_3832_, 0);
lean_inc(v_a_3833_);
v_a_3834_ = lean_ctor_get(v___x_3832_, 1);
lean_inc(v_a_3834_);
lean_dec_ref_known(v___x_3832_, 2);
v___x_3835_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_3834_);
if (lean_obj_tag(v___x_3835_) == 0)
{
lean_object* v_a_3836_; lean_object* v_a_3837_; lean_object* v___x_3838_; 
v_a_3836_ = lean_ctor_get(v___x_3835_, 0);
lean_inc(v_a_3836_);
v_a_3837_ = lean_ctor_get(v___x_3835_, 1);
lean_inc(v_a_3837_);
lean_dec_ref_known(v___x_3835_, 2);
v___x_3838_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBool(v_a_3837_);
if (lean_obj_tag(v___x_3838_) == 0)
{
lean_object* v_a_3839_; lean_object* v_a_3840_; lean_object* v___x_3841_; lean_object* v___x_3842_; 
v_a_3839_ = lean_ctor_get(v___x_3838_, 0);
lean_inc(v_a_3839_);
v_a_3840_ = lean_ctor_get(v___x_3838_, 1);
lean_inc(v_a_3840_);
lean_dec_ref_known(v___x_3838_, 2);
v___x_3841_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readSymbolicVariable), 1, 0);
v___x_3842_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_3841_, v_a_3840_);
if (lean_obj_tag(v___x_3842_) == 0)
{
lean_object* v_a_3843_; lean_object* v_a_3844_; lean_object* v___x_3846_; uint8_t v_isShared_3847_; uint8_t v_isSharedCheck_3854_; 
v_a_3843_ = lean_ctor_get(v___x_3842_, 0);
v_a_3844_ = lean_ctor_get(v___x_3842_, 1);
v_isSharedCheck_3854_ = !lean_is_exclusive(v___x_3842_);
if (v_isSharedCheck_3854_ == 0)
{
v___x_3846_ = v___x_3842_;
v_isShared_3847_ = v_isSharedCheck_3854_;
goto v_resetjp_3845_;
}
else
{
lean_inc(v_a_3844_);
lean_inc(v_a_3843_);
lean_dec(v___x_3842_);
v___x_3846_ = lean_box(0);
v_isShared_3847_ = v_isSharedCheck_3854_;
goto v_resetjp_3845_;
}
v_resetjp_3845_:
{
lean_object* v___x_3848_; uint32_t v___x_3849_; uint8_t v___x_3850_; lean_object* v___x_3852_; 
v___x_3848_ = lean_alloc_ctor(0, 4, 5);
lean_ctor_set(v___x_3848_, 0, v_a_3827_);
lean_ctor_set(v___x_3848_, 1, v_a_3830_);
lean_ctor_set(v___x_3848_, 2, v_a_3833_);
lean_ctor_set(v___x_3848_, 3, v_a_3843_);
v___x_3849_ = lean_unbox_uint32(v_a_3836_);
lean_dec(v_a_3836_);
lean_ctor_set_uint32(v___x_3848_, sizeof(void*)*4, v___x_3849_);
v___x_3850_ = lean_unbox(v_a_3839_);
lean_dec(v_a_3839_);
lean_ctor_set_uint8(v___x_3848_, sizeof(void*)*4 + 4, v___x_3850_);
if (v_isShared_3847_ == 0)
{
lean_ctor_set(v___x_3846_, 0, v___x_3848_);
v___x_3852_ = v___x_3846_;
goto v_reusejp_3851_;
}
else
{
lean_object* v_reuseFailAlloc_3853_; 
v_reuseFailAlloc_3853_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3853_, 0, v___x_3848_);
lean_ctor_set(v_reuseFailAlloc_3853_, 1, v_a_3844_);
v___x_3852_ = v_reuseFailAlloc_3853_;
goto v_reusejp_3851_;
}
v_reusejp_3851_:
{
return v___x_3852_;
}
}
}
else
{
lean_object* v_a_3855_; lean_object* v_a_3856_; lean_object* v___x_3858_; uint8_t v_isShared_3859_; uint8_t v_isSharedCheck_3863_; 
lean_dec(v_a_3839_);
lean_dec(v_a_3836_);
lean_dec(v_a_3833_);
lean_dec(v_a_3830_);
lean_dec(v_a_3827_);
v_a_3855_ = lean_ctor_get(v___x_3842_, 0);
v_a_3856_ = lean_ctor_get(v___x_3842_, 1);
v_isSharedCheck_3863_ = !lean_is_exclusive(v___x_3842_);
if (v_isSharedCheck_3863_ == 0)
{
v___x_3858_ = v___x_3842_;
v_isShared_3859_ = v_isSharedCheck_3863_;
goto v_resetjp_3857_;
}
else
{
lean_inc(v_a_3856_);
lean_inc(v_a_3855_);
lean_dec(v___x_3842_);
v___x_3858_ = lean_box(0);
v_isShared_3859_ = v_isSharedCheck_3863_;
goto v_resetjp_3857_;
}
v_resetjp_3857_:
{
lean_object* v___x_3861_; 
if (v_isShared_3859_ == 0)
{
v___x_3861_ = v___x_3858_;
goto v_reusejp_3860_;
}
else
{
lean_object* v_reuseFailAlloc_3862_; 
v_reuseFailAlloc_3862_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3862_, 0, v_a_3855_);
lean_ctor_set(v_reuseFailAlloc_3862_, 1, v_a_3856_);
v___x_3861_ = v_reuseFailAlloc_3862_;
goto v_reusejp_3860_;
}
v_reusejp_3860_:
{
return v___x_3861_;
}
}
}
}
else
{
lean_object* v_a_3864_; lean_object* v_a_3865_; lean_object* v___x_3867_; uint8_t v_isShared_3868_; uint8_t v_isSharedCheck_3872_; 
lean_dec(v_a_3836_);
lean_dec(v_a_3833_);
lean_dec(v_a_3830_);
lean_dec(v_a_3827_);
v_a_3864_ = lean_ctor_get(v___x_3838_, 0);
v_a_3865_ = lean_ctor_get(v___x_3838_, 1);
v_isSharedCheck_3872_ = !lean_is_exclusive(v___x_3838_);
if (v_isSharedCheck_3872_ == 0)
{
v___x_3867_ = v___x_3838_;
v_isShared_3868_ = v_isSharedCheck_3872_;
goto v_resetjp_3866_;
}
else
{
lean_inc(v_a_3865_);
lean_inc(v_a_3864_);
lean_dec(v___x_3838_);
v___x_3867_ = lean_box(0);
v_isShared_3868_ = v_isSharedCheck_3872_;
goto v_resetjp_3866_;
}
v_resetjp_3866_:
{
lean_object* v___x_3870_; 
if (v_isShared_3868_ == 0)
{
v___x_3870_ = v___x_3867_;
goto v_reusejp_3869_;
}
else
{
lean_object* v_reuseFailAlloc_3871_; 
v_reuseFailAlloc_3871_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3871_, 0, v_a_3864_);
lean_ctor_set(v_reuseFailAlloc_3871_, 1, v_a_3865_);
v___x_3870_ = v_reuseFailAlloc_3871_;
goto v_reusejp_3869_;
}
v_reusejp_3869_:
{
return v___x_3870_;
}
}
}
}
else
{
lean_object* v_a_3873_; lean_object* v_a_3874_; lean_object* v___x_3876_; uint8_t v_isShared_3877_; uint8_t v_isSharedCheck_3881_; 
lean_dec(v_a_3833_);
lean_dec(v_a_3830_);
lean_dec(v_a_3827_);
v_a_3873_ = lean_ctor_get(v___x_3835_, 0);
v_a_3874_ = lean_ctor_get(v___x_3835_, 1);
v_isSharedCheck_3881_ = !lean_is_exclusive(v___x_3835_);
if (v_isSharedCheck_3881_ == 0)
{
v___x_3876_ = v___x_3835_;
v_isShared_3877_ = v_isSharedCheck_3881_;
goto v_resetjp_3875_;
}
else
{
lean_inc(v_a_3874_);
lean_inc(v_a_3873_);
lean_dec(v___x_3835_);
v___x_3876_ = lean_box(0);
v_isShared_3877_ = v_isSharedCheck_3881_;
goto v_resetjp_3875_;
}
v_resetjp_3875_:
{
lean_object* v___x_3879_; 
if (v_isShared_3877_ == 0)
{
v___x_3879_ = v___x_3876_;
goto v_reusejp_3878_;
}
else
{
lean_object* v_reuseFailAlloc_3880_; 
v_reuseFailAlloc_3880_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3880_, 0, v_a_3873_);
lean_ctor_set(v_reuseFailAlloc_3880_, 1, v_a_3874_);
v___x_3879_ = v_reuseFailAlloc_3880_;
goto v_reusejp_3878_;
}
v_reusejp_3878_:
{
return v___x_3879_;
}
}
}
}
else
{
lean_object* v_a_3882_; lean_object* v_a_3883_; lean_object* v___x_3885_; uint8_t v_isShared_3886_; uint8_t v_isSharedCheck_3890_; 
lean_dec(v_a_3830_);
lean_dec(v_a_3827_);
v_a_3882_ = lean_ctor_get(v___x_3832_, 0);
v_a_3883_ = lean_ctor_get(v___x_3832_, 1);
v_isSharedCheck_3890_ = !lean_is_exclusive(v___x_3832_);
if (v_isSharedCheck_3890_ == 0)
{
v___x_3885_ = v___x_3832_;
v_isShared_3886_ = v_isSharedCheck_3890_;
goto v_resetjp_3884_;
}
else
{
lean_inc(v_a_3883_);
lean_inc(v_a_3882_);
lean_dec(v___x_3832_);
v___x_3885_ = lean_box(0);
v_isShared_3886_ = v_isSharedCheck_3890_;
goto v_resetjp_3884_;
}
v_resetjp_3884_:
{
lean_object* v___x_3888_; 
if (v_isShared_3886_ == 0)
{
v___x_3888_ = v___x_3885_;
goto v_reusejp_3887_;
}
else
{
lean_object* v_reuseFailAlloc_3889_; 
v_reuseFailAlloc_3889_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3889_, 0, v_a_3882_);
lean_ctor_set(v_reuseFailAlloc_3889_, 1, v_a_3883_);
v___x_3888_ = v_reuseFailAlloc_3889_;
goto v_reusejp_3887_;
}
v_reusejp_3887_:
{
return v___x_3888_;
}
}
}
}
else
{
lean_object* v_a_3891_; lean_object* v_a_3892_; lean_object* v___x_3894_; uint8_t v_isShared_3895_; uint8_t v_isSharedCheck_3899_; 
lean_dec(v_a_3827_);
v_a_3891_ = lean_ctor_get(v___x_3829_, 0);
v_a_3892_ = lean_ctor_get(v___x_3829_, 1);
v_isSharedCheck_3899_ = !lean_is_exclusive(v___x_3829_);
if (v_isSharedCheck_3899_ == 0)
{
v___x_3894_ = v___x_3829_;
v_isShared_3895_ = v_isSharedCheck_3899_;
goto v_resetjp_3893_;
}
else
{
lean_inc(v_a_3892_);
lean_inc(v_a_3891_);
lean_dec(v___x_3829_);
v___x_3894_ = lean_box(0);
v_isShared_3895_ = v_isSharedCheck_3899_;
goto v_resetjp_3893_;
}
v_resetjp_3893_:
{
lean_object* v___x_3897_; 
if (v_isShared_3895_ == 0)
{
v___x_3897_ = v___x_3894_;
goto v_reusejp_3896_;
}
else
{
lean_object* v_reuseFailAlloc_3898_; 
v_reuseFailAlloc_3898_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3898_, 0, v_a_3891_);
lean_ctor_set(v_reuseFailAlloc_3898_, 1, v_a_3892_);
v___x_3897_ = v_reuseFailAlloc_3898_;
goto v_reusejp_3896_;
}
v_reusejp_3896_:
{
return v___x_3897_;
}
}
}
}
else
{
lean_object* v_a_3900_; lean_object* v_a_3901_; lean_object* v___x_3903_; uint8_t v_isShared_3904_; uint8_t v_isSharedCheck_3908_; 
v_a_3900_ = lean_ctor_get(v___x_3826_, 0);
v_a_3901_ = lean_ctor_get(v___x_3826_, 1);
v_isSharedCheck_3908_ = !lean_is_exclusive(v___x_3826_);
if (v_isSharedCheck_3908_ == 0)
{
v___x_3903_ = v___x_3826_;
v_isShared_3904_ = v_isSharedCheck_3908_;
goto v_resetjp_3902_;
}
else
{
lean_inc(v_a_3901_);
lean_inc(v_a_3900_);
lean_dec(v___x_3826_);
v___x_3903_ = lean_box(0);
v_isShared_3904_ = v_isSharedCheck_3908_;
goto v_resetjp_3902_;
}
v_resetjp_3902_:
{
lean_object* v___x_3906_; 
if (v_isShared_3904_ == 0)
{
v___x_3906_ = v___x_3903_;
goto v_reusejp_3905_;
}
else
{
lean_object* v_reuseFailAlloc_3907_; 
v_reuseFailAlloc_3907_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3907_, 0, v_a_3900_);
lean_ctor_set(v_reuseFailAlloc_3907_, 1, v_a_3901_);
v___x_3906_ = v_reuseFailAlloc_3907_;
goto v_reusejp_3905_;
}
v_reusejp_3905_:
{
return v___x_3906_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readMultiStarkVerifyingKey0(lean_object* v_a_3909_){
_start:
{
lean_object* v___x_3910_; 
v___x_3910_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readSystemParams(v_a_3909_);
if (lean_obj_tag(v___x_3910_) == 0)
{
lean_object* v_a_3911_; lean_object* v_a_3912_; lean_object* v___x_3913_; lean_object* v___x_3914_; 
v_a_3911_ = lean_ctor_get(v___x_3910_, 0);
lean_inc(v_a_3911_);
v_a_3912_ = lean_ctor_get(v___x_3910_, 1);
lean_inc(v_a_3912_);
lean_dec_ref_known(v___x_3910_, 2);
v___x_3913_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readStarkVerifyingKey), 1, 0);
v___x_3914_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_3913_, v_a_3912_);
if (lean_obj_tag(v___x_3914_) == 0)
{
lean_object* v_a_3915_; lean_object* v_a_3916_; lean_object* v___x_3917_; lean_object* v___x_3918_; 
v_a_3915_ = lean_ctor_get(v___x_3914_, 0);
lean_inc(v_a_3915_);
v_a_3916_ = lean_ctor_get(v___x_3914_, 1);
lean_inc(v_a_3916_);
lean_dec_ref_known(v___x_3914_, 2);
v___x_3917_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readLinearConstraint), 1, 0);
v___x_3918_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_3917_, v_a_3916_);
if (lean_obj_tag(v___x_3918_) == 0)
{
lean_object* v_a_3919_; lean_object* v_a_3920_; lean_object* v___x_3922_; uint8_t v_isShared_3923_; uint8_t v_isSharedCheck_3928_; 
v_a_3919_ = lean_ctor_get(v___x_3918_, 0);
v_a_3920_ = lean_ctor_get(v___x_3918_, 1);
v_isSharedCheck_3928_ = !lean_is_exclusive(v___x_3918_);
if (v_isSharedCheck_3928_ == 0)
{
v___x_3922_ = v___x_3918_;
v_isShared_3923_ = v_isSharedCheck_3928_;
goto v_resetjp_3921_;
}
else
{
lean_inc(v_a_3920_);
lean_inc(v_a_3919_);
lean_dec(v___x_3918_);
v___x_3922_ = lean_box(0);
v_isShared_3923_ = v_isSharedCheck_3928_;
goto v_resetjp_3921_;
}
v_resetjp_3921_:
{
lean_object* v___x_3924_; lean_object* v___x_3926_; 
v___x_3924_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3924_, 0, v_a_3911_);
lean_ctor_set(v___x_3924_, 1, v_a_3915_);
lean_ctor_set(v___x_3924_, 2, v_a_3919_);
if (v_isShared_3923_ == 0)
{
lean_ctor_set(v___x_3922_, 0, v___x_3924_);
v___x_3926_ = v___x_3922_;
goto v_reusejp_3925_;
}
else
{
lean_object* v_reuseFailAlloc_3927_; 
v_reuseFailAlloc_3927_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3927_, 0, v___x_3924_);
lean_ctor_set(v_reuseFailAlloc_3927_, 1, v_a_3920_);
v___x_3926_ = v_reuseFailAlloc_3927_;
goto v_reusejp_3925_;
}
v_reusejp_3925_:
{
return v___x_3926_;
}
}
}
else
{
lean_object* v_a_3929_; lean_object* v_a_3930_; lean_object* v___x_3932_; uint8_t v_isShared_3933_; uint8_t v_isSharedCheck_3937_; 
lean_dec(v_a_3915_);
lean_dec(v_a_3911_);
v_a_3929_ = lean_ctor_get(v___x_3918_, 0);
v_a_3930_ = lean_ctor_get(v___x_3918_, 1);
v_isSharedCheck_3937_ = !lean_is_exclusive(v___x_3918_);
if (v_isSharedCheck_3937_ == 0)
{
v___x_3932_ = v___x_3918_;
v_isShared_3933_ = v_isSharedCheck_3937_;
goto v_resetjp_3931_;
}
else
{
lean_inc(v_a_3930_);
lean_inc(v_a_3929_);
lean_dec(v___x_3918_);
v___x_3932_ = lean_box(0);
v_isShared_3933_ = v_isSharedCheck_3937_;
goto v_resetjp_3931_;
}
v_resetjp_3931_:
{
lean_object* v___x_3935_; 
if (v_isShared_3933_ == 0)
{
v___x_3935_ = v___x_3932_;
goto v_reusejp_3934_;
}
else
{
lean_object* v_reuseFailAlloc_3936_; 
v_reuseFailAlloc_3936_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3936_, 0, v_a_3929_);
lean_ctor_set(v_reuseFailAlloc_3936_, 1, v_a_3930_);
v___x_3935_ = v_reuseFailAlloc_3936_;
goto v_reusejp_3934_;
}
v_reusejp_3934_:
{
return v___x_3935_;
}
}
}
}
else
{
lean_object* v_a_3938_; lean_object* v_a_3939_; lean_object* v___x_3941_; uint8_t v_isShared_3942_; uint8_t v_isSharedCheck_3946_; 
lean_dec(v_a_3911_);
v_a_3938_ = lean_ctor_get(v___x_3914_, 0);
v_a_3939_ = lean_ctor_get(v___x_3914_, 1);
v_isSharedCheck_3946_ = !lean_is_exclusive(v___x_3914_);
if (v_isSharedCheck_3946_ == 0)
{
v___x_3941_ = v___x_3914_;
v_isShared_3942_ = v_isSharedCheck_3946_;
goto v_resetjp_3940_;
}
else
{
lean_inc(v_a_3939_);
lean_inc(v_a_3938_);
lean_dec(v___x_3914_);
v___x_3941_ = lean_box(0);
v_isShared_3942_ = v_isSharedCheck_3946_;
goto v_resetjp_3940_;
}
v_resetjp_3940_:
{
lean_object* v___x_3944_; 
if (v_isShared_3942_ == 0)
{
v___x_3944_ = v___x_3941_;
goto v_reusejp_3943_;
}
else
{
lean_object* v_reuseFailAlloc_3945_; 
v_reuseFailAlloc_3945_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3945_, 0, v_a_3938_);
lean_ctor_set(v_reuseFailAlloc_3945_, 1, v_a_3939_);
v___x_3944_ = v_reuseFailAlloc_3945_;
goto v_reusejp_3943_;
}
v_reusejp_3943_:
{
return v___x_3944_;
}
}
}
}
else
{
lean_object* v_a_3947_; lean_object* v_a_3948_; lean_object* v___x_3950_; uint8_t v_isShared_3951_; uint8_t v_isSharedCheck_3955_; 
v_a_3947_ = lean_ctor_get(v___x_3910_, 0);
v_a_3948_ = lean_ctor_get(v___x_3910_, 1);
v_isSharedCheck_3955_ = !lean_is_exclusive(v___x_3910_);
if (v_isSharedCheck_3955_ == 0)
{
v___x_3950_ = v___x_3910_;
v_isShared_3951_ = v_isSharedCheck_3955_;
goto v_resetjp_3949_;
}
else
{
lean_inc(v_a_3948_);
lean_inc(v_a_3947_);
lean_dec(v___x_3910_);
v___x_3950_ = lean_box(0);
v_isShared_3951_ = v_isSharedCheck_3955_;
goto v_resetjp_3949_;
}
v_resetjp_3949_:
{
lean_object* v___x_3953_; 
if (v_isShared_3951_ == 0)
{
v___x_3953_ = v___x_3950_;
goto v_reusejp_3952_;
}
else
{
lean_object* v_reuseFailAlloc_3954_; 
v_reuseFailAlloc_3954_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3954_, 0, v_a_3947_);
lean_ctor_set(v_reuseFailAlloc_3954_, 1, v_a_3948_);
v___x_3953_ = v_reuseFailAlloc_3954_;
goto v_reusejp_3952_;
}
v_reusejp_3952_:
{
return v___x_3953_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readRawVkM(lean_object* v_a_3956_){
_start:
{
lean_object* v___x_3957_; lean_object* v___x_3958_; 
v___x_3957_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicVk;
v___x_3958_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readHeader(v___x_3957_, v_a_3956_);
if (lean_obj_tag(v___x_3958_) == 0)
{
lean_object* v_a_3959_; lean_object* v___x_3960_; 
v_a_3959_ = lean_ctor_get(v___x_3958_, 1);
lean_inc(v_a_3959_);
lean_dec_ref_known(v___x_3958_, 2);
v___x_3960_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readMultiStarkVerifyingKey0(v_a_3959_);
if (lean_obj_tag(v___x_3960_) == 0)
{
lean_object* v_a_3961_; lean_object* v_a_3962_; lean_object* v___x_3963_; 
v_a_3961_ = lean_ctor_get(v___x_3960_, 0);
lean_inc(v_a_3961_);
v_a_3962_ = lean_ctor_get(v___x_3960_, 1);
lean_inc(v_a_3962_);
lean_dec_ref_known(v___x_3960_, 2);
v___x_3963_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readDigest(v_a_3962_);
if (lean_obj_tag(v___x_3963_) == 0)
{
lean_object* v_a_3964_; lean_object* v_a_3965_; lean_object* v___x_3967_; uint8_t v_isShared_3968_; uint8_t v_isSharedCheck_3973_; 
v_a_3964_ = lean_ctor_get(v___x_3963_, 0);
v_a_3965_ = lean_ctor_get(v___x_3963_, 1);
v_isSharedCheck_3973_ = !lean_is_exclusive(v___x_3963_);
if (v_isSharedCheck_3973_ == 0)
{
v___x_3967_ = v___x_3963_;
v_isShared_3968_ = v_isSharedCheck_3973_;
goto v_resetjp_3966_;
}
else
{
lean_inc(v_a_3965_);
lean_inc(v_a_3964_);
lean_dec(v___x_3963_);
v___x_3967_ = lean_box(0);
v_isShared_3968_ = v_isSharedCheck_3973_;
goto v_resetjp_3966_;
}
v_resetjp_3966_:
{
lean_object* v___x_3969_; lean_object* v___x_3971_; 
v___x_3969_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3969_, 0, v_a_3961_);
lean_ctor_set(v___x_3969_, 1, v_a_3964_);
if (v_isShared_3968_ == 0)
{
lean_ctor_set(v___x_3967_, 0, v___x_3969_);
v___x_3971_ = v___x_3967_;
goto v_reusejp_3970_;
}
else
{
lean_object* v_reuseFailAlloc_3972_; 
v_reuseFailAlloc_3972_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3972_, 0, v___x_3969_);
lean_ctor_set(v_reuseFailAlloc_3972_, 1, v_a_3965_);
v___x_3971_ = v_reuseFailAlloc_3972_;
goto v_reusejp_3970_;
}
v_reusejp_3970_:
{
return v___x_3971_;
}
}
}
else
{
lean_object* v_a_3974_; lean_object* v_a_3975_; lean_object* v___x_3977_; uint8_t v_isShared_3978_; uint8_t v_isSharedCheck_3982_; 
lean_dec(v_a_3961_);
v_a_3974_ = lean_ctor_get(v___x_3963_, 0);
v_a_3975_ = lean_ctor_get(v___x_3963_, 1);
v_isSharedCheck_3982_ = !lean_is_exclusive(v___x_3963_);
if (v_isSharedCheck_3982_ == 0)
{
v___x_3977_ = v___x_3963_;
v_isShared_3978_ = v_isSharedCheck_3982_;
goto v_resetjp_3976_;
}
else
{
lean_inc(v_a_3975_);
lean_inc(v_a_3974_);
lean_dec(v___x_3963_);
v___x_3977_ = lean_box(0);
v_isShared_3978_ = v_isSharedCheck_3982_;
goto v_resetjp_3976_;
}
v_resetjp_3976_:
{
lean_object* v___x_3980_; 
if (v_isShared_3978_ == 0)
{
v___x_3980_ = v___x_3977_;
goto v_reusejp_3979_;
}
else
{
lean_object* v_reuseFailAlloc_3981_; 
v_reuseFailAlloc_3981_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3981_, 0, v_a_3974_);
lean_ctor_set(v_reuseFailAlloc_3981_, 1, v_a_3975_);
v___x_3980_ = v_reuseFailAlloc_3981_;
goto v_reusejp_3979_;
}
v_reusejp_3979_:
{
return v___x_3980_;
}
}
}
}
else
{
lean_object* v_a_3983_; lean_object* v_a_3984_; lean_object* v___x_3986_; uint8_t v_isShared_3987_; uint8_t v_isSharedCheck_3991_; 
v_a_3983_ = lean_ctor_get(v___x_3960_, 0);
v_a_3984_ = lean_ctor_get(v___x_3960_, 1);
v_isSharedCheck_3991_ = !lean_is_exclusive(v___x_3960_);
if (v_isSharedCheck_3991_ == 0)
{
v___x_3986_ = v___x_3960_;
v_isShared_3987_ = v_isSharedCheck_3991_;
goto v_resetjp_3985_;
}
else
{
lean_inc(v_a_3984_);
lean_inc(v_a_3983_);
lean_dec(v___x_3960_);
v___x_3986_ = lean_box(0);
v_isShared_3987_ = v_isSharedCheck_3991_;
goto v_resetjp_3985_;
}
v_resetjp_3985_:
{
lean_object* v___x_3989_; 
if (v_isShared_3987_ == 0)
{
v___x_3989_ = v___x_3986_;
goto v_reusejp_3988_;
}
else
{
lean_object* v_reuseFailAlloc_3990_; 
v_reuseFailAlloc_3990_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3990_, 0, v_a_3983_);
lean_ctor_set(v_reuseFailAlloc_3990_, 1, v_a_3984_);
v___x_3989_ = v_reuseFailAlloc_3990_;
goto v_reusejp_3988_;
}
v_reusejp_3988_:
{
return v___x_3989_;
}
}
}
}
else
{
lean_object* v_a_3992_; lean_object* v_a_3993_; lean_object* v___x_3995_; uint8_t v_isShared_3996_; uint8_t v_isSharedCheck_4000_; 
v_a_3992_ = lean_ctor_get(v___x_3958_, 0);
v_a_3993_ = lean_ctor_get(v___x_3958_, 1);
v_isSharedCheck_4000_ = !lean_is_exclusive(v___x_3958_);
if (v_isSharedCheck_4000_ == 0)
{
v___x_3995_ = v___x_3958_;
v_isShared_3996_ = v_isSharedCheck_4000_;
goto v_resetjp_3994_;
}
else
{
lean_inc(v_a_3993_);
lean_inc(v_a_3992_);
lean_dec(v___x_3958_);
v___x_3995_ = lean_box(0);
v_isShared_3996_ = v_isSharedCheck_4000_;
goto v_resetjp_3994_;
}
v_resetjp_3994_:
{
lean_object* v___x_3998_; 
if (v_isShared_3996_ == 0)
{
v___x_3998_ = v___x_3995_;
goto v_reusejp_3997_;
}
else
{
lean_object* v_reuseFailAlloc_3999_; 
v_reuseFailAlloc_3999_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3999_, 0, v_a_3992_);
lean_ctor_set(v_reuseFailAlloc_3999_, 1, v_a_3993_);
v___x_3998_ = v_reuseFailAlloc_3999_;
goto v_reusejp_3997_;
}
v_reusejp_3997_:
{
return v___x_3998_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readRawVk(lean_object* v_data_4001_){
_start:
{
lean_object* v___x_4002_; lean_object* v___x_4003_; 
v___x_4002_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readRawVkM), 1, 0);
v___x_4003_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_runParser___redArg(v___x_4002_, v_data_4001_);
return v___x_4003_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readTraceVData(lean_object* v_a_4004_){
_start:
{
lean_object* v___x_4005_; 
v___x_4005_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_4004_);
if (lean_obj_tag(v___x_4005_) == 0)
{
lean_object* v_a_4006_; lean_object* v_a_4007_; lean_object* v___x_4008_; lean_object* v___x_4009_; 
v_a_4006_ = lean_ctor_get(v___x_4005_, 0);
lean_inc(v_a_4006_);
v_a_4007_ = lean_ctor_get(v___x_4005_, 1);
lean_inc(v_a_4007_);
lean_dec_ref_known(v___x_4005_, 2);
v___x_4008_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readDigest), 1, 0);
v___x_4009_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_4008_, v_a_4007_);
if (lean_obj_tag(v___x_4009_) == 0)
{
lean_object* v_a_4010_; lean_object* v_a_4011_; lean_object* v___x_4013_; uint8_t v_isShared_4014_; uint8_t v_isSharedCheck_4020_; 
v_a_4010_ = lean_ctor_get(v___x_4009_, 0);
v_a_4011_ = lean_ctor_get(v___x_4009_, 1);
v_isSharedCheck_4020_ = !lean_is_exclusive(v___x_4009_);
if (v_isSharedCheck_4020_ == 0)
{
v___x_4013_ = v___x_4009_;
v_isShared_4014_ = v_isSharedCheck_4020_;
goto v_resetjp_4012_;
}
else
{
lean_inc(v_a_4011_);
lean_inc(v_a_4010_);
lean_dec(v___x_4009_);
v___x_4013_ = lean_box(0);
v_isShared_4014_ = v_isSharedCheck_4020_;
goto v_resetjp_4012_;
}
v_resetjp_4012_:
{
lean_object* v___x_4015_; uint32_t v___x_4016_; lean_object* v___x_4018_; 
v___x_4015_ = lean_alloc_ctor(0, 1, 4);
lean_ctor_set(v___x_4015_, 0, v_a_4010_);
v___x_4016_ = lean_unbox_uint32(v_a_4006_);
lean_dec(v_a_4006_);
lean_ctor_set_uint32(v___x_4015_, sizeof(void*)*1, v___x_4016_);
if (v_isShared_4014_ == 0)
{
lean_ctor_set(v___x_4013_, 0, v___x_4015_);
v___x_4018_ = v___x_4013_;
goto v_reusejp_4017_;
}
else
{
lean_object* v_reuseFailAlloc_4019_; 
v_reuseFailAlloc_4019_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4019_, 0, v___x_4015_);
lean_ctor_set(v_reuseFailAlloc_4019_, 1, v_a_4011_);
v___x_4018_ = v_reuseFailAlloc_4019_;
goto v_reusejp_4017_;
}
v_reusejp_4017_:
{
return v___x_4018_;
}
}
}
else
{
lean_object* v_a_4021_; lean_object* v_a_4022_; lean_object* v___x_4024_; uint8_t v_isShared_4025_; uint8_t v_isSharedCheck_4029_; 
lean_dec(v_a_4006_);
v_a_4021_ = lean_ctor_get(v___x_4009_, 0);
v_a_4022_ = lean_ctor_get(v___x_4009_, 1);
v_isSharedCheck_4029_ = !lean_is_exclusive(v___x_4009_);
if (v_isSharedCheck_4029_ == 0)
{
v___x_4024_ = v___x_4009_;
v_isShared_4025_ = v_isSharedCheck_4029_;
goto v_resetjp_4023_;
}
else
{
lean_inc(v_a_4022_);
lean_inc(v_a_4021_);
lean_dec(v___x_4009_);
v___x_4024_ = lean_box(0);
v_isShared_4025_ = v_isSharedCheck_4029_;
goto v_resetjp_4023_;
}
v_resetjp_4023_:
{
lean_object* v___x_4027_; 
if (v_isShared_4025_ == 0)
{
v___x_4027_ = v___x_4024_;
goto v_reusejp_4026_;
}
else
{
lean_object* v_reuseFailAlloc_4028_; 
v_reuseFailAlloc_4028_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4028_, 0, v_a_4021_);
lean_ctor_set(v_reuseFailAlloc_4028_, 1, v_a_4022_);
v___x_4027_ = v_reuseFailAlloc_4028_;
goto v_reusejp_4026_;
}
v_reusejp_4026_:
{
return v___x_4027_;
}
}
}
}
else
{
lean_object* v_a_4030_; lean_object* v_a_4031_; lean_object* v___x_4033_; uint8_t v_isShared_4034_; uint8_t v_isSharedCheck_4038_; 
v_a_4030_ = lean_ctor_get(v___x_4005_, 0);
v_a_4031_ = lean_ctor_get(v___x_4005_, 1);
v_isSharedCheck_4038_ = !lean_is_exclusive(v___x_4005_);
if (v_isSharedCheck_4038_ == 0)
{
v___x_4033_ = v___x_4005_;
v_isShared_4034_ = v_isSharedCheck_4038_;
goto v_resetjp_4032_;
}
else
{
lean_inc(v_a_4031_);
lean_inc(v_a_4030_);
lean_dec(v___x_4005_);
v___x_4033_ = lean_box(0);
v_isShared_4034_ = v_isSharedCheck_4038_;
goto v_resetjp_4032_;
}
v_resetjp_4032_:
{
lean_object* v___x_4036_; 
if (v_isShared_4034_ == 0)
{
v___x_4036_ = v___x_4033_;
goto v_reusejp_4035_;
}
else
{
lean_object* v_reuseFailAlloc_4037_; 
v_reuseFailAlloc_4037_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4037_, 0, v_a_4030_);
lean_ctor_set(v_reuseFailAlloc_4037_, 1, v_a_4031_);
v___x_4036_ = v_reuseFailAlloc_4037_;
goto v_reusejp_4035_;
}
v_reusejp_4035_:
{
return v___x_4036_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readGkrLayerClaims(lean_object* v_a_4039_){
_start:
{
lean_object* v___x_4040_; 
v___x_4040_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readExt4(v_a_4039_);
if (lean_obj_tag(v___x_4040_) == 0)
{
lean_object* v_a_4041_; lean_object* v_a_4042_; lean_object* v___x_4043_; 
v_a_4041_ = lean_ctor_get(v___x_4040_, 0);
lean_inc(v_a_4041_);
v_a_4042_ = lean_ctor_get(v___x_4040_, 1);
lean_inc(v_a_4042_);
lean_dec_ref_known(v___x_4040_, 2);
v___x_4043_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readExt4(v_a_4042_);
if (lean_obj_tag(v___x_4043_) == 0)
{
lean_object* v_a_4044_; lean_object* v_a_4045_; lean_object* v___x_4046_; 
v_a_4044_ = lean_ctor_get(v___x_4043_, 0);
lean_inc(v_a_4044_);
v_a_4045_ = lean_ctor_get(v___x_4043_, 1);
lean_inc(v_a_4045_);
lean_dec_ref_known(v___x_4043_, 2);
v___x_4046_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readExt4(v_a_4045_);
if (lean_obj_tag(v___x_4046_) == 0)
{
lean_object* v_a_4047_; lean_object* v_a_4048_; lean_object* v___x_4049_; 
v_a_4047_ = lean_ctor_get(v___x_4046_, 0);
lean_inc(v_a_4047_);
v_a_4048_ = lean_ctor_get(v___x_4046_, 1);
lean_inc(v_a_4048_);
lean_dec_ref_known(v___x_4046_, 2);
v___x_4049_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readExt4(v_a_4048_);
if (lean_obj_tag(v___x_4049_) == 0)
{
lean_object* v_a_4050_; lean_object* v_a_4051_; lean_object* v___x_4053_; uint8_t v_isShared_4054_; uint8_t v_isSharedCheck_4059_; 
v_a_4050_ = lean_ctor_get(v___x_4049_, 0);
v_a_4051_ = lean_ctor_get(v___x_4049_, 1);
v_isSharedCheck_4059_ = !lean_is_exclusive(v___x_4049_);
if (v_isSharedCheck_4059_ == 0)
{
v___x_4053_ = v___x_4049_;
v_isShared_4054_ = v_isSharedCheck_4059_;
goto v_resetjp_4052_;
}
else
{
lean_inc(v_a_4051_);
lean_inc(v_a_4050_);
lean_dec(v___x_4049_);
v___x_4053_ = lean_box(0);
v_isShared_4054_ = v_isSharedCheck_4059_;
goto v_resetjp_4052_;
}
v_resetjp_4052_:
{
lean_object* v___x_4055_; lean_object* v___x_4057_; 
v___x_4055_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_4055_, 0, v_a_4041_);
lean_ctor_set(v___x_4055_, 1, v_a_4044_);
lean_ctor_set(v___x_4055_, 2, v_a_4047_);
lean_ctor_set(v___x_4055_, 3, v_a_4050_);
if (v_isShared_4054_ == 0)
{
lean_ctor_set(v___x_4053_, 0, v___x_4055_);
v___x_4057_ = v___x_4053_;
goto v_reusejp_4056_;
}
else
{
lean_object* v_reuseFailAlloc_4058_; 
v_reuseFailAlloc_4058_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4058_, 0, v___x_4055_);
lean_ctor_set(v_reuseFailAlloc_4058_, 1, v_a_4051_);
v___x_4057_ = v_reuseFailAlloc_4058_;
goto v_reusejp_4056_;
}
v_reusejp_4056_:
{
return v___x_4057_;
}
}
}
else
{
lean_object* v_a_4060_; lean_object* v_a_4061_; lean_object* v___x_4063_; uint8_t v_isShared_4064_; uint8_t v_isSharedCheck_4068_; 
lean_dec(v_a_4047_);
lean_dec(v_a_4044_);
lean_dec(v_a_4041_);
v_a_4060_ = lean_ctor_get(v___x_4049_, 0);
v_a_4061_ = lean_ctor_get(v___x_4049_, 1);
v_isSharedCheck_4068_ = !lean_is_exclusive(v___x_4049_);
if (v_isSharedCheck_4068_ == 0)
{
v___x_4063_ = v___x_4049_;
v_isShared_4064_ = v_isSharedCheck_4068_;
goto v_resetjp_4062_;
}
else
{
lean_inc(v_a_4061_);
lean_inc(v_a_4060_);
lean_dec(v___x_4049_);
v___x_4063_ = lean_box(0);
v_isShared_4064_ = v_isSharedCheck_4068_;
goto v_resetjp_4062_;
}
v_resetjp_4062_:
{
lean_object* v___x_4066_; 
if (v_isShared_4064_ == 0)
{
v___x_4066_ = v___x_4063_;
goto v_reusejp_4065_;
}
else
{
lean_object* v_reuseFailAlloc_4067_; 
v_reuseFailAlloc_4067_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4067_, 0, v_a_4060_);
lean_ctor_set(v_reuseFailAlloc_4067_, 1, v_a_4061_);
v___x_4066_ = v_reuseFailAlloc_4067_;
goto v_reusejp_4065_;
}
v_reusejp_4065_:
{
return v___x_4066_;
}
}
}
}
else
{
lean_object* v_a_4069_; lean_object* v_a_4070_; lean_object* v___x_4072_; uint8_t v_isShared_4073_; uint8_t v_isSharedCheck_4077_; 
lean_dec(v_a_4044_);
lean_dec(v_a_4041_);
v_a_4069_ = lean_ctor_get(v___x_4046_, 0);
v_a_4070_ = lean_ctor_get(v___x_4046_, 1);
v_isSharedCheck_4077_ = !lean_is_exclusive(v___x_4046_);
if (v_isSharedCheck_4077_ == 0)
{
v___x_4072_ = v___x_4046_;
v_isShared_4073_ = v_isSharedCheck_4077_;
goto v_resetjp_4071_;
}
else
{
lean_inc(v_a_4070_);
lean_inc(v_a_4069_);
lean_dec(v___x_4046_);
v___x_4072_ = lean_box(0);
v_isShared_4073_ = v_isSharedCheck_4077_;
goto v_resetjp_4071_;
}
v_resetjp_4071_:
{
lean_object* v___x_4075_; 
if (v_isShared_4073_ == 0)
{
v___x_4075_ = v___x_4072_;
goto v_reusejp_4074_;
}
else
{
lean_object* v_reuseFailAlloc_4076_; 
v_reuseFailAlloc_4076_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4076_, 0, v_a_4069_);
lean_ctor_set(v_reuseFailAlloc_4076_, 1, v_a_4070_);
v___x_4075_ = v_reuseFailAlloc_4076_;
goto v_reusejp_4074_;
}
v_reusejp_4074_:
{
return v___x_4075_;
}
}
}
}
else
{
lean_object* v_a_4078_; lean_object* v_a_4079_; lean_object* v___x_4081_; uint8_t v_isShared_4082_; uint8_t v_isSharedCheck_4086_; 
lean_dec(v_a_4041_);
v_a_4078_ = lean_ctor_get(v___x_4043_, 0);
v_a_4079_ = lean_ctor_get(v___x_4043_, 1);
v_isSharedCheck_4086_ = !lean_is_exclusive(v___x_4043_);
if (v_isSharedCheck_4086_ == 0)
{
v___x_4081_ = v___x_4043_;
v_isShared_4082_ = v_isSharedCheck_4086_;
goto v_resetjp_4080_;
}
else
{
lean_inc(v_a_4079_);
lean_inc(v_a_4078_);
lean_dec(v___x_4043_);
v___x_4081_ = lean_box(0);
v_isShared_4082_ = v_isSharedCheck_4086_;
goto v_resetjp_4080_;
}
v_resetjp_4080_:
{
lean_object* v___x_4084_; 
if (v_isShared_4082_ == 0)
{
v___x_4084_ = v___x_4081_;
goto v_reusejp_4083_;
}
else
{
lean_object* v_reuseFailAlloc_4085_; 
v_reuseFailAlloc_4085_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4085_, 0, v_a_4078_);
lean_ctor_set(v_reuseFailAlloc_4085_, 1, v_a_4079_);
v___x_4084_ = v_reuseFailAlloc_4085_;
goto v_reusejp_4083_;
}
v_reusejp_4083_:
{
return v___x_4084_;
}
}
}
}
else
{
lean_object* v_a_4087_; lean_object* v_a_4088_; lean_object* v___x_4090_; uint8_t v_isShared_4091_; uint8_t v_isSharedCheck_4095_; 
v_a_4087_ = lean_ctor_get(v___x_4040_, 0);
v_a_4088_ = lean_ctor_get(v___x_4040_, 1);
v_isSharedCheck_4095_ = !lean_is_exclusive(v___x_4040_);
if (v_isSharedCheck_4095_ == 0)
{
v___x_4090_ = v___x_4040_;
v_isShared_4091_ = v_isSharedCheck_4095_;
goto v_resetjp_4089_;
}
else
{
lean_inc(v_a_4088_);
lean_inc(v_a_4087_);
lean_dec(v___x_4040_);
v___x_4090_ = lean_box(0);
v_isShared_4091_ = v_isSharedCheck_4095_;
goto v_resetjp_4089_;
}
v_resetjp_4089_:
{
lean_object* v___x_4093_; 
if (v_isShared_4091_ == 0)
{
v___x_4093_ = v___x_4090_;
goto v_reusejp_4092_;
}
else
{
lean_object* v_reuseFailAlloc_4094_; 
v_reuseFailAlloc_4094_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4094_, 0, v_a_4087_);
lean_ctor_set(v_reuseFailAlloc_4094_, 1, v_a_4088_);
v___x_4093_ = v_reuseFailAlloc_4094_;
goto v_reusejp_4092_;
}
v_reusejp_4092_:
{
return v___x_4093_;
}
}
}
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readGkrProof___closed__0(void){
_start:
{
lean_object* v___x_4096_; lean_object* v___x_4097_; lean_object* v___x_4098_; 
v___x_4096_ = lean_unsigned_to_nat(3u);
v___x_4097_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readExt4), 1, 0);
v___x_4098_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readVectorN___boxed), 4, 3);
lean_closure_set(v___x_4098_, 0, lean_box(0));
lean_closure_set(v___x_4098_, 1, v___x_4097_);
lean_closure_set(v___x_4098_, 2, v___x_4096_);
return v___x_4098_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readGkrProof___closed__1(void){
_start:
{
lean_object* v___x_4099_; lean_object* v___x_4100_; 
v___x_4099_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readGkrProof___closed__0, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readGkrProof___closed__0_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readGkrProof___closed__0);
v___x_4100_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr), 3, 2);
lean_closure_set(v___x_4100_, 0, lean_box(0));
lean_closure_set(v___x_4100_, 1, v___x_4099_);
return v___x_4100_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readGkrProof(lean_object* v_a_4101_){
_start:
{
lean_object* v___x_4102_; 
v___x_4102_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readCanonicalFBB(v_a_4101_);
if (lean_obj_tag(v___x_4102_) == 0)
{
lean_object* v_a_4103_; lean_object* v_a_4104_; lean_object* v___x_4105_; 
v_a_4103_ = lean_ctor_get(v___x_4102_, 0);
lean_inc(v_a_4103_);
v_a_4104_ = lean_ctor_get(v___x_4102_, 1);
lean_inc(v_a_4104_);
lean_dec_ref_known(v___x_4102_, 2);
v___x_4105_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readExt4(v_a_4104_);
if (lean_obj_tag(v___x_4105_) == 0)
{
lean_object* v_a_4106_; lean_object* v_a_4107_; lean_object* v___x_4108_; lean_object* v___x_4109_; 
v_a_4106_ = lean_ctor_get(v___x_4105_, 0);
lean_inc(v_a_4106_);
v_a_4107_ = lean_ctor_get(v___x_4105_, 1);
lean_inc(v_a_4107_);
lean_dec_ref_known(v___x_4105_, 2);
v___x_4108_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readGkrLayerClaims), 1, 0);
v___x_4109_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_4108_, v_a_4107_);
if (lean_obj_tag(v___x_4109_) == 0)
{
lean_object* v_a_4110_; lean_object* v_a_4111_; lean_object* v___x_4112_; lean_object* v___x_4113_; 
v_a_4110_ = lean_ctor_get(v___x_4109_, 0);
lean_inc(v_a_4110_);
v_a_4111_ = lean_ctor_get(v___x_4109_, 1);
lean_inc(v_a_4111_);
lean_dec_ref_known(v___x_4109_, 2);
v___x_4112_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readGkrProof___closed__1, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readGkrProof___closed__1_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readGkrProof___closed__1);
v___x_4113_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_4112_, v_a_4111_);
if (lean_obj_tag(v___x_4113_) == 0)
{
lean_object* v_a_4114_; lean_object* v_a_4115_; lean_object* v___x_4117_; uint8_t v_isShared_4118_; uint8_t v_isSharedCheck_4124_; 
v_a_4114_ = lean_ctor_get(v___x_4113_, 0);
v_a_4115_ = lean_ctor_get(v___x_4113_, 1);
v_isSharedCheck_4124_ = !lean_is_exclusive(v___x_4113_);
if (v_isSharedCheck_4124_ == 0)
{
v___x_4117_ = v___x_4113_;
v_isShared_4118_ = v_isSharedCheck_4124_;
goto v_resetjp_4116_;
}
else
{
lean_inc(v_a_4115_);
lean_inc(v_a_4114_);
lean_dec(v___x_4113_);
v___x_4117_ = lean_box(0);
v_isShared_4118_ = v_isSharedCheck_4124_;
goto v_resetjp_4116_;
}
v_resetjp_4116_:
{
lean_object* v___x_4119_; uint32_t v___x_4120_; lean_object* v___x_4122_; 
v___x_4119_ = lean_alloc_ctor(0, 3, 4);
lean_ctor_set(v___x_4119_, 0, v_a_4106_);
lean_ctor_set(v___x_4119_, 1, v_a_4110_);
lean_ctor_set(v___x_4119_, 2, v_a_4114_);
v___x_4120_ = lean_unbox_uint32(v_a_4103_);
lean_dec(v_a_4103_);
lean_ctor_set_uint32(v___x_4119_, sizeof(void*)*3, v___x_4120_);
if (v_isShared_4118_ == 0)
{
lean_ctor_set(v___x_4117_, 0, v___x_4119_);
v___x_4122_ = v___x_4117_;
goto v_reusejp_4121_;
}
else
{
lean_object* v_reuseFailAlloc_4123_; 
v_reuseFailAlloc_4123_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4123_, 0, v___x_4119_);
lean_ctor_set(v_reuseFailAlloc_4123_, 1, v_a_4115_);
v___x_4122_ = v_reuseFailAlloc_4123_;
goto v_reusejp_4121_;
}
v_reusejp_4121_:
{
return v___x_4122_;
}
}
}
else
{
lean_object* v_a_4125_; lean_object* v_a_4126_; lean_object* v___x_4128_; uint8_t v_isShared_4129_; uint8_t v_isSharedCheck_4133_; 
lean_dec(v_a_4110_);
lean_dec(v_a_4106_);
lean_dec(v_a_4103_);
v_a_4125_ = lean_ctor_get(v___x_4113_, 0);
v_a_4126_ = lean_ctor_get(v___x_4113_, 1);
v_isSharedCheck_4133_ = !lean_is_exclusive(v___x_4113_);
if (v_isSharedCheck_4133_ == 0)
{
v___x_4128_ = v___x_4113_;
v_isShared_4129_ = v_isSharedCheck_4133_;
goto v_resetjp_4127_;
}
else
{
lean_inc(v_a_4126_);
lean_inc(v_a_4125_);
lean_dec(v___x_4113_);
v___x_4128_ = lean_box(0);
v_isShared_4129_ = v_isSharedCheck_4133_;
goto v_resetjp_4127_;
}
v_resetjp_4127_:
{
lean_object* v___x_4131_; 
if (v_isShared_4129_ == 0)
{
v___x_4131_ = v___x_4128_;
goto v_reusejp_4130_;
}
else
{
lean_object* v_reuseFailAlloc_4132_; 
v_reuseFailAlloc_4132_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4132_, 0, v_a_4125_);
lean_ctor_set(v_reuseFailAlloc_4132_, 1, v_a_4126_);
v___x_4131_ = v_reuseFailAlloc_4132_;
goto v_reusejp_4130_;
}
v_reusejp_4130_:
{
return v___x_4131_;
}
}
}
}
else
{
lean_object* v_a_4134_; lean_object* v_a_4135_; lean_object* v___x_4137_; uint8_t v_isShared_4138_; uint8_t v_isSharedCheck_4142_; 
lean_dec(v_a_4106_);
lean_dec(v_a_4103_);
v_a_4134_ = lean_ctor_get(v___x_4109_, 0);
v_a_4135_ = lean_ctor_get(v___x_4109_, 1);
v_isSharedCheck_4142_ = !lean_is_exclusive(v___x_4109_);
if (v_isSharedCheck_4142_ == 0)
{
v___x_4137_ = v___x_4109_;
v_isShared_4138_ = v_isSharedCheck_4142_;
goto v_resetjp_4136_;
}
else
{
lean_inc(v_a_4135_);
lean_inc(v_a_4134_);
lean_dec(v___x_4109_);
v___x_4137_ = lean_box(0);
v_isShared_4138_ = v_isSharedCheck_4142_;
goto v_resetjp_4136_;
}
v_resetjp_4136_:
{
lean_object* v___x_4140_; 
if (v_isShared_4138_ == 0)
{
v___x_4140_ = v___x_4137_;
goto v_reusejp_4139_;
}
else
{
lean_object* v_reuseFailAlloc_4141_; 
v_reuseFailAlloc_4141_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4141_, 0, v_a_4134_);
lean_ctor_set(v_reuseFailAlloc_4141_, 1, v_a_4135_);
v___x_4140_ = v_reuseFailAlloc_4141_;
goto v_reusejp_4139_;
}
v_reusejp_4139_:
{
return v___x_4140_;
}
}
}
}
else
{
lean_object* v_a_4143_; lean_object* v_a_4144_; lean_object* v___x_4146_; uint8_t v_isShared_4147_; uint8_t v_isSharedCheck_4151_; 
lean_dec(v_a_4103_);
v_a_4143_ = lean_ctor_get(v___x_4105_, 0);
v_a_4144_ = lean_ctor_get(v___x_4105_, 1);
v_isSharedCheck_4151_ = !lean_is_exclusive(v___x_4105_);
if (v_isSharedCheck_4151_ == 0)
{
v___x_4146_ = v___x_4105_;
v_isShared_4147_ = v_isSharedCheck_4151_;
goto v_resetjp_4145_;
}
else
{
lean_inc(v_a_4144_);
lean_inc(v_a_4143_);
lean_dec(v___x_4105_);
v___x_4146_ = lean_box(0);
v_isShared_4147_ = v_isSharedCheck_4151_;
goto v_resetjp_4145_;
}
v_resetjp_4145_:
{
lean_object* v___x_4149_; 
if (v_isShared_4147_ == 0)
{
v___x_4149_ = v___x_4146_;
goto v_reusejp_4148_;
}
else
{
lean_object* v_reuseFailAlloc_4150_; 
v_reuseFailAlloc_4150_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4150_, 0, v_a_4143_);
lean_ctor_set(v_reuseFailAlloc_4150_, 1, v_a_4144_);
v___x_4149_ = v_reuseFailAlloc_4150_;
goto v_reusejp_4148_;
}
v_reusejp_4148_:
{
return v___x_4149_;
}
}
}
}
else
{
lean_object* v_a_4152_; lean_object* v_a_4153_; lean_object* v___x_4155_; uint8_t v_isShared_4156_; uint8_t v_isSharedCheck_4160_; 
v_a_4152_ = lean_ctor_get(v___x_4102_, 0);
v_a_4153_ = lean_ctor_get(v___x_4102_, 1);
v_isSharedCheck_4160_ = !lean_is_exclusive(v___x_4102_);
if (v_isSharedCheck_4160_ == 0)
{
v___x_4155_ = v___x_4102_;
v_isShared_4156_ = v_isSharedCheck_4160_;
goto v_resetjp_4154_;
}
else
{
lean_inc(v_a_4153_);
lean_inc(v_a_4152_);
lean_dec(v___x_4102_);
v___x_4155_ = lean_box(0);
v_isShared_4156_ = v_isSharedCheck_4160_;
goto v_resetjp_4154_;
}
v_resetjp_4154_:
{
lean_object* v___x_4158_; 
if (v_isShared_4156_ == 0)
{
v___x_4158_ = v___x_4155_;
goto v_reusejp_4157_;
}
else
{
lean_object* v_reuseFailAlloc_4159_; 
v_reuseFailAlloc_4159_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4159_, 0, v_a_4152_);
lean_ctor_set(v_reuseFailAlloc_4159_, 1, v_a_4153_);
v___x_4158_ = v_reuseFailAlloc_4159_;
goto v_reusejp_4157_;
}
v_reusejp_4157_:
{
return v___x_4158_;
}
}
}
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBatchConstraintProof___closed__0(void){
_start:
{
lean_object* v___x_4161_; lean_object* v___x_4162_; 
v___x_4161_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readExt4), 1, 0);
v___x_4162_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr), 3, 2);
lean_closure_set(v___x_4162_, 0, lean_box(0));
lean_closure_set(v___x_4162_, 1, v___x_4161_);
return v___x_4162_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBatchConstraintProof___closed__1(void){
_start:
{
lean_object* v___x_4163_; lean_object* v___x_4164_; 
v___x_4163_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBatchConstraintProof___closed__0, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBatchConstraintProof___closed__0_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBatchConstraintProof___closed__0);
v___x_4164_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr), 3, 2);
lean_closure_set(v___x_4164_, 0, lean_box(0));
lean_closure_set(v___x_4164_, 1, v___x_4163_);
return v___x_4164_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBatchConstraintProof(lean_object* v_a_4165_){
_start:
{
lean_object* v___x_4166_; lean_object* v___x_4167_; lean_object* v___x_4168_; 
v___x_4166_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readExt4), 1, 0);
v___x_4167_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBatchConstraintProof___closed__0, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBatchConstraintProof___closed__0_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBatchConstraintProof___closed__0);
lean_inc_ref(v___x_4166_);
v___x_4168_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_4166_, v_a_4165_);
if (lean_obj_tag(v___x_4168_) == 0)
{
lean_object* v_a_4169_; lean_object* v_a_4170_; lean_object* v___x_4171_; 
v_a_4169_ = lean_ctor_get(v___x_4168_, 0);
lean_inc(v_a_4169_);
v_a_4170_ = lean_ctor_get(v___x_4168_, 1);
lean_inc(v_a_4170_);
lean_dec_ref_known(v___x_4168_, 2);
lean_inc_ref(v___x_4166_);
v___x_4171_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_4166_, v_a_4170_);
if (lean_obj_tag(v___x_4171_) == 0)
{
lean_object* v_a_4172_; lean_object* v_a_4173_; lean_object* v___x_4174_; 
v_a_4172_ = lean_ctor_get(v___x_4171_, 0);
lean_inc(v_a_4172_);
v_a_4173_ = lean_ctor_get(v___x_4171_, 1);
lean_inc(v_a_4173_);
lean_dec_ref_known(v___x_4171_, 2);
v___x_4174_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_4166_, v_a_4173_);
if (lean_obj_tag(v___x_4174_) == 0)
{
lean_object* v_a_4175_; lean_object* v_a_4176_; lean_object* v___x_4177_; lean_object* v___x_4178_; 
v_a_4175_ = lean_ctor_get(v___x_4174_, 0);
lean_inc(v_a_4175_);
v_a_4176_ = lean_ctor_get(v___x_4174_, 1);
lean_inc(v_a_4176_);
lean_dec_ref_known(v___x_4174_, 2);
v___x_4177_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBatchConstraintProof___closed__1, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBatchConstraintProof___closed__1_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBatchConstraintProof___closed__1);
v___x_4178_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_4167_, v_a_4176_);
if (lean_obj_tag(v___x_4178_) == 0)
{
lean_object* v_a_4179_; lean_object* v_a_4180_; lean_object* v___x_4181_; 
v_a_4179_ = lean_ctor_get(v___x_4178_, 0);
lean_inc(v_a_4179_);
v_a_4180_ = lean_ctor_get(v___x_4178_, 1);
lean_inc(v_a_4180_);
lean_dec_ref_known(v___x_4178_, 2);
v___x_4181_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_4177_, v_a_4180_);
if (lean_obj_tag(v___x_4181_) == 0)
{
lean_object* v_a_4182_; lean_object* v_a_4183_; lean_object* v___x_4185_; uint8_t v_isShared_4186_; uint8_t v_isSharedCheck_4191_; 
v_a_4182_ = lean_ctor_get(v___x_4181_, 0);
v_a_4183_ = lean_ctor_get(v___x_4181_, 1);
v_isSharedCheck_4191_ = !lean_is_exclusive(v___x_4181_);
if (v_isSharedCheck_4191_ == 0)
{
v___x_4185_ = v___x_4181_;
v_isShared_4186_ = v_isSharedCheck_4191_;
goto v_resetjp_4184_;
}
else
{
lean_inc(v_a_4183_);
lean_inc(v_a_4182_);
lean_dec(v___x_4181_);
v___x_4185_ = lean_box(0);
v_isShared_4186_ = v_isSharedCheck_4191_;
goto v_resetjp_4184_;
}
v_resetjp_4184_:
{
lean_object* v___x_4187_; lean_object* v___x_4189_; 
v___x_4187_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_4187_, 0, v_a_4169_);
lean_ctor_set(v___x_4187_, 1, v_a_4172_);
lean_ctor_set(v___x_4187_, 2, v_a_4175_);
lean_ctor_set(v___x_4187_, 3, v_a_4179_);
lean_ctor_set(v___x_4187_, 4, v_a_4182_);
if (v_isShared_4186_ == 0)
{
lean_ctor_set(v___x_4185_, 0, v___x_4187_);
v___x_4189_ = v___x_4185_;
goto v_reusejp_4188_;
}
else
{
lean_object* v_reuseFailAlloc_4190_; 
v_reuseFailAlloc_4190_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4190_, 0, v___x_4187_);
lean_ctor_set(v_reuseFailAlloc_4190_, 1, v_a_4183_);
v___x_4189_ = v_reuseFailAlloc_4190_;
goto v_reusejp_4188_;
}
v_reusejp_4188_:
{
return v___x_4189_;
}
}
}
else
{
lean_object* v_a_4192_; lean_object* v_a_4193_; lean_object* v___x_4195_; uint8_t v_isShared_4196_; uint8_t v_isSharedCheck_4200_; 
lean_dec(v_a_4179_);
lean_dec(v_a_4175_);
lean_dec(v_a_4172_);
lean_dec(v_a_4169_);
v_a_4192_ = lean_ctor_get(v___x_4181_, 0);
v_a_4193_ = lean_ctor_get(v___x_4181_, 1);
v_isSharedCheck_4200_ = !lean_is_exclusive(v___x_4181_);
if (v_isSharedCheck_4200_ == 0)
{
v___x_4195_ = v___x_4181_;
v_isShared_4196_ = v_isSharedCheck_4200_;
goto v_resetjp_4194_;
}
else
{
lean_inc(v_a_4193_);
lean_inc(v_a_4192_);
lean_dec(v___x_4181_);
v___x_4195_ = lean_box(0);
v_isShared_4196_ = v_isSharedCheck_4200_;
goto v_resetjp_4194_;
}
v_resetjp_4194_:
{
lean_object* v___x_4198_; 
if (v_isShared_4196_ == 0)
{
v___x_4198_ = v___x_4195_;
goto v_reusejp_4197_;
}
else
{
lean_object* v_reuseFailAlloc_4199_; 
v_reuseFailAlloc_4199_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4199_, 0, v_a_4192_);
lean_ctor_set(v_reuseFailAlloc_4199_, 1, v_a_4193_);
v___x_4198_ = v_reuseFailAlloc_4199_;
goto v_reusejp_4197_;
}
v_reusejp_4197_:
{
return v___x_4198_;
}
}
}
}
else
{
lean_object* v_a_4201_; lean_object* v_a_4202_; lean_object* v___x_4204_; uint8_t v_isShared_4205_; uint8_t v_isSharedCheck_4209_; 
lean_dec(v_a_4175_);
lean_dec(v_a_4172_);
lean_dec(v_a_4169_);
v_a_4201_ = lean_ctor_get(v___x_4178_, 0);
v_a_4202_ = lean_ctor_get(v___x_4178_, 1);
v_isSharedCheck_4209_ = !lean_is_exclusive(v___x_4178_);
if (v_isSharedCheck_4209_ == 0)
{
v___x_4204_ = v___x_4178_;
v_isShared_4205_ = v_isSharedCheck_4209_;
goto v_resetjp_4203_;
}
else
{
lean_inc(v_a_4202_);
lean_inc(v_a_4201_);
lean_dec(v___x_4178_);
v___x_4204_ = lean_box(0);
v_isShared_4205_ = v_isSharedCheck_4209_;
goto v_resetjp_4203_;
}
v_resetjp_4203_:
{
lean_object* v___x_4207_; 
if (v_isShared_4205_ == 0)
{
v___x_4207_ = v___x_4204_;
goto v_reusejp_4206_;
}
else
{
lean_object* v_reuseFailAlloc_4208_; 
v_reuseFailAlloc_4208_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4208_, 0, v_a_4201_);
lean_ctor_set(v_reuseFailAlloc_4208_, 1, v_a_4202_);
v___x_4207_ = v_reuseFailAlloc_4208_;
goto v_reusejp_4206_;
}
v_reusejp_4206_:
{
return v___x_4207_;
}
}
}
}
else
{
lean_object* v_a_4210_; lean_object* v_a_4211_; lean_object* v___x_4213_; uint8_t v_isShared_4214_; uint8_t v_isSharedCheck_4218_; 
lean_dec(v_a_4172_);
lean_dec(v_a_4169_);
v_a_4210_ = lean_ctor_get(v___x_4174_, 0);
v_a_4211_ = lean_ctor_get(v___x_4174_, 1);
v_isSharedCheck_4218_ = !lean_is_exclusive(v___x_4174_);
if (v_isSharedCheck_4218_ == 0)
{
v___x_4213_ = v___x_4174_;
v_isShared_4214_ = v_isSharedCheck_4218_;
goto v_resetjp_4212_;
}
else
{
lean_inc(v_a_4211_);
lean_inc(v_a_4210_);
lean_dec(v___x_4174_);
v___x_4213_ = lean_box(0);
v_isShared_4214_ = v_isSharedCheck_4218_;
goto v_resetjp_4212_;
}
v_resetjp_4212_:
{
lean_object* v___x_4216_; 
if (v_isShared_4214_ == 0)
{
v___x_4216_ = v___x_4213_;
goto v_reusejp_4215_;
}
else
{
lean_object* v_reuseFailAlloc_4217_; 
v_reuseFailAlloc_4217_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4217_, 0, v_a_4210_);
lean_ctor_set(v_reuseFailAlloc_4217_, 1, v_a_4211_);
v___x_4216_ = v_reuseFailAlloc_4217_;
goto v_reusejp_4215_;
}
v_reusejp_4215_:
{
return v___x_4216_;
}
}
}
}
else
{
lean_object* v_a_4219_; lean_object* v_a_4220_; lean_object* v___x_4222_; uint8_t v_isShared_4223_; uint8_t v_isSharedCheck_4227_; 
lean_dec(v_a_4169_);
lean_dec_ref(v___x_4166_);
v_a_4219_ = lean_ctor_get(v___x_4171_, 0);
v_a_4220_ = lean_ctor_get(v___x_4171_, 1);
v_isSharedCheck_4227_ = !lean_is_exclusive(v___x_4171_);
if (v_isSharedCheck_4227_ == 0)
{
v___x_4222_ = v___x_4171_;
v_isShared_4223_ = v_isSharedCheck_4227_;
goto v_resetjp_4221_;
}
else
{
lean_inc(v_a_4220_);
lean_inc(v_a_4219_);
lean_dec(v___x_4171_);
v___x_4222_ = lean_box(0);
v_isShared_4223_ = v_isSharedCheck_4227_;
goto v_resetjp_4221_;
}
v_resetjp_4221_:
{
lean_object* v___x_4225_; 
if (v_isShared_4223_ == 0)
{
v___x_4225_ = v___x_4222_;
goto v_reusejp_4224_;
}
else
{
lean_object* v_reuseFailAlloc_4226_; 
v_reuseFailAlloc_4226_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4226_, 0, v_a_4219_);
lean_ctor_set(v_reuseFailAlloc_4226_, 1, v_a_4220_);
v___x_4225_ = v_reuseFailAlloc_4226_;
goto v_reusejp_4224_;
}
v_reusejp_4224_:
{
return v___x_4225_;
}
}
}
}
else
{
lean_object* v_a_4228_; lean_object* v_a_4229_; lean_object* v___x_4231_; uint8_t v_isShared_4232_; uint8_t v_isSharedCheck_4236_; 
lean_dec_ref(v___x_4166_);
v_a_4228_ = lean_ctor_get(v___x_4168_, 0);
v_a_4229_ = lean_ctor_get(v___x_4168_, 1);
v_isSharedCheck_4236_ = !lean_is_exclusive(v___x_4168_);
if (v_isSharedCheck_4236_ == 0)
{
v___x_4231_ = v___x_4168_;
v_isShared_4232_ = v_isSharedCheck_4236_;
goto v_resetjp_4230_;
}
else
{
lean_inc(v_a_4229_);
lean_inc(v_a_4228_);
lean_dec(v___x_4168_);
v___x_4231_ = lean_box(0);
v_isShared_4232_ = v_isSharedCheck_4236_;
goto v_resetjp_4230_;
}
v_resetjp_4230_:
{
lean_object* v___x_4234_; 
if (v_isShared_4232_ == 0)
{
v___x_4234_ = v___x_4231_;
goto v_reusejp_4233_;
}
else
{
lean_object* v_reuseFailAlloc_4235_; 
v_reuseFailAlloc_4235_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4235_, 0, v_a_4228_);
lean_ctor_set(v_reuseFailAlloc_4235_, 1, v_a_4229_);
v___x_4234_ = v_reuseFailAlloc_4235_;
goto v_reusejp_4233_;
}
v_reusejp_4233_:
{
return v___x_4234_;
}
}
}
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readStackingProof___closed__0(void){
_start:
{
lean_object* v___x_4237_; lean_object* v___x_4238_; lean_object* v___x_4239_; 
v___x_4237_ = lean_unsigned_to_nat(2u);
v___x_4238_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readExt4), 1, 0);
v___x_4239_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readVectorN___boxed), 4, 3);
lean_closure_set(v___x_4239_, 0, lean_box(0));
lean_closure_set(v___x_4239_, 1, v___x_4238_);
lean_closure_set(v___x_4239_, 2, v___x_4237_);
return v___x_4239_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readStackingProof(lean_object* v_a_4240_){
_start:
{
lean_object* v___x_4241_; 
v___x_4241_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readCanonicalFBB(v_a_4240_);
if (lean_obj_tag(v___x_4241_) == 0)
{
lean_object* v_a_4242_; lean_object* v_a_4243_; lean_object* v___x_4244_; lean_object* v___x_4245_; lean_object* v___x_4246_; 
v_a_4242_ = lean_ctor_get(v___x_4241_, 0);
lean_inc(v_a_4242_);
v_a_4243_ = lean_ctor_get(v___x_4241_, 1);
lean_inc(v_a_4243_);
lean_dec_ref_known(v___x_4241_, 2);
v___x_4244_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readExt4), 1, 0);
v___x_4245_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBatchConstraintProof___closed__0, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBatchConstraintProof___closed__0_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBatchConstraintProof___closed__0);
v___x_4246_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_4244_, v_a_4243_);
if (lean_obj_tag(v___x_4246_) == 0)
{
lean_object* v_a_4247_; lean_object* v_a_4248_; lean_object* v___x_4249_; lean_object* v___x_4250_; 
v_a_4247_ = lean_ctor_get(v___x_4246_, 0);
lean_inc(v_a_4247_);
v_a_4248_ = lean_ctor_get(v___x_4246_, 1);
lean_inc(v_a_4248_);
lean_dec_ref_known(v___x_4246_, 2);
v___x_4249_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readStackingProof___closed__0, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readStackingProof___closed__0_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readStackingProof___closed__0);
v___x_4250_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_4249_, v_a_4248_);
if (lean_obj_tag(v___x_4250_) == 0)
{
lean_object* v_a_4251_; lean_object* v_a_4252_; lean_object* v___x_4253_; 
v_a_4251_ = lean_ctor_get(v___x_4250_, 0);
lean_inc(v_a_4251_);
v_a_4252_ = lean_ctor_get(v___x_4250_, 1);
lean_inc(v_a_4252_);
lean_dec_ref_known(v___x_4250_, 2);
v___x_4253_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_4245_, v_a_4252_);
if (lean_obj_tag(v___x_4253_) == 0)
{
lean_object* v_a_4254_; lean_object* v_a_4255_; lean_object* v___x_4257_; uint8_t v_isShared_4258_; uint8_t v_isSharedCheck_4264_; 
v_a_4254_ = lean_ctor_get(v___x_4253_, 0);
v_a_4255_ = lean_ctor_get(v___x_4253_, 1);
v_isSharedCheck_4264_ = !lean_is_exclusive(v___x_4253_);
if (v_isSharedCheck_4264_ == 0)
{
v___x_4257_ = v___x_4253_;
v_isShared_4258_ = v_isSharedCheck_4264_;
goto v_resetjp_4256_;
}
else
{
lean_inc(v_a_4255_);
lean_inc(v_a_4254_);
lean_dec(v___x_4253_);
v___x_4257_ = lean_box(0);
v_isShared_4258_ = v_isSharedCheck_4264_;
goto v_resetjp_4256_;
}
v_resetjp_4256_:
{
lean_object* v___x_4259_; uint32_t v___x_4260_; lean_object* v___x_4262_; 
v___x_4259_ = lean_alloc_ctor(0, 3, 4);
lean_ctor_set(v___x_4259_, 0, v_a_4247_);
lean_ctor_set(v___x_4259_, 1, v_a_4251_);
lean_ctor_set(v___x_4259_, 2, v_a_4254_);
v___x_4260_ = lean_unbox_uint32(v_a_4242_);
lean_dec(v_a_4242_);
lean_ctor_set_uint32(v___x_4259_, sizeof(void*)*3, v___x_4260_);
if (v_isShared_4258_ == 0)
{
lean_ctor_set(v___x_4257_, 0, v___x_4259_);
v___x_4262_ = v___x_4257_;
goto v_reusejp_4261_;
}
else
{
lean_object* v_reuseFailAlloc_4263_; 
v_reuseFailAlloc_4263_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4263_, 0, v___x_4259_);
lean_ctor_set(v_reuseFailAlloc_4263_, 1, v_a_4255_);
v___x_4262_ = v_reuseFailAlloc_4263_;
goto v_reusejp_4261_;
}
v_reusejp_4261_:
{
return v___x_4262_;
}
}
}
else
{
lean_object* v_a_4265_; lean_object* v_a_4266_; lean_object* v___x_4268_; uint8_t v_isShared_4269_; uint8_t v_isSharedCheck_4273_; 
lean_dec(v_a_4251_);
lean_dec(v_a_4247_);
lean_dec(v_a_4242_);
v_a_4265_ = lean_ctor_get(v___x_4253_, 0);
v_a_4266_ = lean_ctor_get(v___x_4253_, 1);
v_isSharedCheck_4273_ = !lean_is_exclusive(v___x_4253_);
if (v_isSharedCheck_4273_ == 0)
{
v___x_4268_ = v___x_4253_;
v_isShared_4269_ = v_isSharedCheck_4273_;
goto v_resetjp_4267_;
}
else
{
lean_inc(v_a_4266_);
lean_inc(v_a_4265_);
lean_dec(v___x_4253_);
v___x_4268_ = lean_box(0);
v_isShared_4269_ = v_isSharedCheck_4273_;
goto v_resetjp_4267_;
}
v_resetjp_4267_:
{
lean_object* v___x_4271_; 
if (v_isShared_4269_ == 0)
{
v___x_4271_ = v___x_4268_;
goto v_reusejp_4270_;
}
else
{
lean_object* v_reuseFailAlloc_4272_; 
v_reuseFailAlloc_4272_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4272_, 0, v_a_4265_);
lean_ctor_set(v_reuseFailAlloc_4272_, 1, v_a_4266_);
v___x_4271_ = v_reuseFailAlloc_4272_;
goto v_reusejp_4270_;
}
v_reusejp_4270_:
{
return v___x_4271_;
}
}
}
}
else
{
lean_object* v_a_4274_; lean_object* v_a_4275_; lean_object* v___x_4277_; uint8_t v_isShared_4278_; uint8_t v_isSharedCheck_4282_; 
lean_dec(v_a_4247_);
lean_dec(v_a_4242_);
v_a_4274_ = lean_ctor_get(v___x_4250_, 0);
v_a_4275_ = lean_ctor_get(v___x_4250_, 1);
v_isSharedCheck_4282_ = !lean_is_exclusive(v___x_4250_);
if (v_isSharedCheck_4282_ == 0)
{
v___x_4277_ = v___x_4250_;
v_isShared_4278_ = v_isSharedCheck_4282_;
goto v_resetjp_4276_;
}
else
{
lean_inc(v_a_4275_);
lean_inc(v_a_4274_);
lean_dec(v___x_4250_);
v___x_4277_ = lean_box(0);
v_isShared_4278_ = v_isSharedCheck_4282_;
goto v_resetjp_4276_;
}
v_resetjp_4276_:
{
lean_object* v___x_4280_; 
if (v_isShared_4278_ == 0)
{
v___x_4280_ = v___x_4277_;
goto v_reusejp_4279_;
}
else
{
lean_object* v_reuseFailAlloc_4281_; 
v_reuseFailAlloc_4281_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4281_, 0, v_a_4274_);
lean_ctor_set(v_reuseFailAlloc_4281_, 1, v_a_4275_);
v___x_4280_ = v_reuseFailAlloc_4281_;
goto v_reusejp_4279_;
}
v_reusejp_4279_:
{
return v___x_4280_;
}
}
}
}
else
{
lean_object* v_a_4283_; lean_object* v_a_4284_; lean_object* v___x_4286_; uint8_t v_isShared_4287_; uint8_t v_isSharedCheck_4291_; 
lean_dec(v_a_4242_);
v_a_4283_ = lean_ctor_get(v___x_4246_, 0);
v_a_4284_ = lean_ctor_get(v___x_4246_, 1);
v_isSharedCheck_4291_ = !lean_is_exclusive(v___x_4246_);
if (v_isSharedCheck_4291_ == 0)
{
v___x_4286_ = v___x_4246_;
v_isShared_4287_ = v_isSharedCheck_4291_;
goto v_resetjp_4285_;
}
else
{
lean_inc(v_a_4284_);
lean_inc(v_a_4283_);
lean_dec(v___x_4246_);
v___x_4286_ = lean_box(0);
v_isShared_4287_ = v_isSharedCheck_4291_;
goto v_resetjp_4285_;
}
v_resetjp_4285_:
{
lean_object* v___x_4289_; 
if (v_isShared_4287_ == 0)
{
v___x_4289_ = v___x_4286_;
goto v_reusejp_4288_;
}
else
{
lean_object* v_reuseFailAlloc_4290_; 
v_reuseFailAlloc_4290_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4290_, 0, v_a_4283_);
lean_ctor_set(v_reuseFailAlloc_4290_, 1, v_a_4284_);
v___x_4289_ = v_reuseFailAlloc_4290_;
goto v_reusejp_4288_;
}
v_reusejp_4288_:
{
return v___x_4289_;
}
}
}
}
else
{
lean_object* v_a_4292_; lean_object* v_a_4293_; lean_object* v___x_4295_; uint8_t v_isShared_4296_; uint8_t v_isSharedCheck_4300_; 
v_a_4292_ = lean_ctor_get(v___x_4241_, 0);
v_a_4293_ = lean_ctor_get(v___x_4241_, 1);
v_isSharedCheck_4300_ = !lean_is_exclusive(v___x_4241_);
if (v_isSharedCheck_4300_ == 0)
{
v___x_4295_ = v___x_4241_;
v_isShared_4296_ = v_isSharedCheck_4300_;
goto v_resetjp_4294_;
}
else
{
lean_inc(v_a_4293_);
lean_inc(v_a_4292_);
lean_dec(v___x_4241_);
v___x_4295_ = lean_box(0);
v_isShared_4296_ = v_isSharedCheck_4300_;
goto v_resetjp_4294_;
}
v_resetjp_4294_:
{
lean_object* v___x_4298_; 
if (v_isShared_4296_ == 0)
{
v___x_4298_ = v___x_4295_;
goto v_reusejp_4297_;
}
else
{
lean_object* v_reuseFailAlloc_4299_; 
v_reuseFailAlloc_4299_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4299_, 0, v_a_4292_);
lean_ctor_set(v_reuseFailAlloc_4299_, 1, v_a_4293_);
v___x_4298_ = v_reuseFailAlloc_4299_;
goto v_reusejp_4297_;
}
v_reusejp_4297_:
{
return v___x_4298_;
}
}
}
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__0(void){
_start:
{
lean_object* v___x_4301_; lean_object* v___x_4302_; 
v___x_4301_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readDigest), 1, 0);
v___x_4302_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr), 3, 2);
lean_closure_set(v___x_4302_, 0, lean_box(0));
lean_closure_set(v___x_4302_, 1, v___x_4301_);
return v___x_4302_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__1(void){
_start:
{
lean_object* v___x_4303_; lean_object* v___x_4304_; 
v___x_4303_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readCanonicalFBB), 1, 0);
v___x_4304_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr), 3, 2);
lean_closure_set(v___x_4304_, 0, lean_box(0));
lean_closure_set(v___x_4304_, 1, v___x_4303_);
return v___x_4304_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__2(void){
_start:
{
lean_object* v___x_4305_; lean_object* v___x_4306_; 
v___x_4305_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__1, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__1_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__1);
v___x_4306_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr), 3, 2);
lean_closure_set(v___x_4306_, 0, lean_box(0));
lean_closure_set(v___x_4306_, 1, v___x_4305_);
return v___x_4306_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__3(void){
_start:
{
lean_object* v___x_4307_; lean_object* v___x_4308_; 
v___x_4307_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__2, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__2_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__2);
v___x_4308_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr), 3, 2);
lean_closure_set(v___x_4308_, 0, lean_box(0));
lean_closure_set(v___x_4308_, 1, v___x_4307_);
return v___x_4308_;
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__4(void){
_start:
{
lean_object* v___x_4309_; lean_object* v___x_4310_; 
v___x_4309_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__0, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__0_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__0);
v___x_4310_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr), 3, 2);
lean_closure_set(v___x_4310_, 0, lean_box(0));
lean_closure_set(v___x_4310_, 1, v___x_4309_);
return v___x_4310_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof(lean_object* v_a_4311_){
_start:
{
lean_object* v___x_4312_; lean_object* v___x_4313_; lean_object* v___x_4314_; 
v___x_4312_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readExt4), 1, 0);
v___x_4313_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readStackingProof___closed__0, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readStackingProof___closed__0_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readStackingProof___closed__0);
v___x_4314_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_4313_, v_a_4311_);
if (lean_obj_tag(v___x_4314_) == 0)
{
lean_object* v_a_4315_; lean_object* v_a_4316_; lean_object* v___x_4317_; lean_object* v___x_4318_; 
v_a_4315_ = lean_ctor_get(v___x_4314_, 0);
lean_inc(v_a_4315_);
v_a_4316_ = lean_ctor_get(v___x_4314_, 1);
lean_inc(v_a_4316_);
lean_dec_ref_known(v___x_4314_, 2);
v___x_4317_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readDigest), 1, 0);
v___x_4318_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_4317_, v_a_4316_);
if (lean_obj_tag(v___x_4318_) == 0)
{
lean_object* v_a_4319_; lean_object* v_a_4320_; lean_object* v___x_4321_; 
v_a_4319_ = lean_ctor_get(v___x_4318_, 0);
lean_inc(v_a_4319_);
v_a_4320_ = lean_ctor_get(v___x_4318_, 1);
lean_inc(v_a_4320_);
lean_dec_ref_known(v___x_4318_, 2);
lean_inc_ref(v___x_4312_);
v___x_4321_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_4312_, v_a_4320_);
if (lean_obj_tag(v___x_4321_) == 0)
{
lean_object* v_a_4322_; lean_object* v_a_4323_; lean_object* v___x_4324_; lean_object* v___x_4325_; 
v_a_4322_ = lean_ctor_get(v___x_4321_, 0);
lean_inc(v_a_4322_);
v_a_4323_ = lean_ctor_get(v___x_4321_, 1);
lean_inc(v_a_4323_);
lean_dec_ref_known(v___x_4321_, 2);
v___x_4324_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readCanonicalFBB), 1, 0);
lean_inc_ref(v___x_4324_);
v___x_4325_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_4324_, v_a_4323_);
if (lean_obj_tag(v___x_4325_) == 0)
{
lean_object* v_a_4326_; lean_object* v_a_4327_; lean_object* v___x_4328_; 
v_a_4326_ = lean_ctor_get(v___x_4325_, 0);
lean_inc(v_a_4326_);
v_a_4327_ = lean_ctor_get(v___x_4325_, 1);
lean_inc(v_a_4327_);
lean_dec_ref_known(v___x_4325_, 2);
v___x_4328_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_4324_, v_a_4327_);
if (lean_obj_tag(v___x_4328_) == 0)
{
lean_object* v_a_4329_; lean_object* v_a_4330_; lean_object* v___x_4331_; lean_object* v___x_4332_; 
v_a_4329_ = lean_ctor_get(v___x_4328_, 0);
lean_inc(v_a_4329_);
v_a_4330_ = lean_ctor_get(v___x_4328_, 1);
lean_inc(v_a_4330_);
lean_dec_ref_known(v___x_4328_, 2);
v___x_4331_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__3, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__3_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__3);
v___x_4332_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_4331_, v_a_4330_);
if (lean_obj_tag(v___x_4332_) == 0)
{
lean_object* v_a_4333_; lean_object* v_a_4334_; lean_object* v___x_4335_; lean_object* v___x_4336_; 
v_a_4333_ = lean_ctor_get(v___x_4332_, 0);
lean_inc(v_a_4333_);
v_a_4334_ = lean_ctor_get(v___x_4332_, 1);
lean_inc(v_a_4334_);
lean_dec_ref_known(v___x_4332_, 2);
v___x_4335_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__4, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__4_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof___closed__4);
v___x_4336_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_4335_, v_a_4334_);
if (lean_obj_tag(v___x_4336_) == 0)
{
lean_object* v_a_4337_; lean_object* v_a_4338_; lean_object* v___x_4339_; lean_object* v___x_4340_; 
v_a_4337_ = lean_ctor_get(v___x_4336_, 0);
lean_inc(v_a_4337_);
v_a_4338_ = lean_ctor_get(v___x_4336_, 1);
lean_inc(v_a_4338_);
lean_dec_ref_known(v___x_4336_, 2);
v___x_4339_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBatchConstraintProof___closed__1, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBatchConstraintProof___closed__1_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBatchConstraintProof___closed__1);
v___x_4340_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_4339_, v_a_4338_);
if (lean_obj_tag(v___x_4340_) == 0)
{
lean_object* v_a_4341_; lean_object* v_a_4342_; lean_object* v___x_4343_; 
v_a_4341_ = lean_ctor_get(v___x_4340_, 0);
lean_inc(v_a_4341_);
v_a_4342_ = lean_ctor_get(v___x_4340_, 1);
lean_inc(v_a_4342_);
lean_dec_ref_known(v___x_4340_, 2);
v___x_4343_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_4335_, v_a_4342_);
if (lean_obj_tag(v___x_4343_) == 0)
{
lean_object* v_a_4344_; lean_object* v_a_4345_; lean_object* v___x_4346_; 
v_a_4344_ = lean_ctor_get(v___x_4343_, 0);
lean_inc(v_a_4344_);
v_a_4345_ = lean_ctor_get(v___x_4343_, 1);
lean_inc(v_a_4345_);
lean_dec_ref_known(v___x_4343_, 2);
v___x_4346_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_4312_, v_a_4345_);
if (lean_obj_tag(v___x_4346_) == 0)
{
lean_object* v_a_4347_; lean_object* v_a_4348_; lean_object* v___x_4350_; uint8_t v_isShared_4351_; uint8_t v_isSharedCheck_4356_; 
v_a_4347_ = lean_ctor_get(v___x_4346_, 0);
v_a_4348_ = lean_ctor_get(v___x_4346_, 1);
v_isSharedCheck_4356_ = !lean_is_exclusive(v___x_4346_);
if (v_isSharedCheck_4356_ == 0)
{
v___x_4350_ = v___x_4346_;
v_isShared_4351_ = v_isSharedCheck_4356_;
goto v_resetjp_4349_;
}
else
{
lean_inc(v_a_4348_);
lean_inc(v_a_4347_);
lean_dec(v___x_4346_);
v___x_4350_ = lean_box(0);
v_isShared_4351_ = v_isSharedCheck_4356_;
goto v_resetjp_4349_;
}
v_resetjp_4349_:
{
lean_object* v___x_4352_; lean_object* v___x_4354_; 
v___x_4352_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_4352_, 0, v_a_4315_);
lean_ctor_set(v___x_4352_, 1, v_a_4319_);
lean_ctor_set(v___x_4352_, 2, v_a_4322_);
lean_ctor_set(v___x_4352_, 3, v_a_4326_);
lean_ctor_set(v___x_4352_, 4, v_a_4329_);
lean_ctor_set(v___x_4352_, 5, v_a_4333_);
lean_ctor_set(v___x_4352_, 6, v_a_4337_);
lean_ctor_set(v___x_4352_, 7, v_a_4341_);
lean_ctor_set(v___x_4352_, 8, v_a_4344_);
lean_ctor_set(v___x_4352_, 9, v_a_4347_);
if (v_isShared_4351_ == 0)
{
lean_ctor_set(v___x_4350_, 0, v___x_4352_);
v___x_4354_ = v___x_4350_;
goto v_reusejp_4353_;
}
else
{
lean_object* v_reuseFailAlloc_4355_; 
v_reuseFailAlloc_4355_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4355_, 0, v___x_4352_);
lean_ctor_set(v_reuseFailAlloc_4355_, 1, v_a_4348_);
v___x_4354_ = v_reuseFailAlloc_4355_;
goto v_reusejp_4353_;
}
v_reusejp_4353_:
{
return v___x_4354_;
}
}
}
else
{
lean_object* v_a_4357_; lean_object* v_a_4358_; lean_object* v___x_4360_; uint8_t v_isShared_4361_; uint8_t v_isSharedCheck_4365_; 
lean_dec(v_a_4344_);
lean_dec(v_a_4341_);
lean_dec(v_a_4337_);
lean_dec(v_a_4333_);
lean_dec(v_a_4329_);
lean_dec(v_a_4326_);
lean_dec(v_a_4322_);
lean_dec(v_a_4319_);
lean_dec(v_a_4315_);
v_a_4357_ = lean_ctor_get(v___x_4346_, 0);
v_a_4358_ = lean_ctor_get(v___x_4346_, 1);
v_isSharedCheck_4365_ = !lean_is_exclusive(v___x_4346_);
if (v_isSharedCheck_4365_ == 0)
{
v___x_4360_ = v___x_4346_;
v_isShared_4361_ = v_isSharedCheck_4365_;
goto v_resetjp_4359_;
}
else
{
lean_inc(v_a_4358_);
lean_inc(v_a_4357_);
lean_dec(v___x_4346_);
v___x_4360_ = lean_box(0);
v_isShared_4361_ = v_isSharedCheck_4365_;
goto v_resetjp_4359_;
}
v_resetjp_4359_:
{
lean_object* v___x_4363_; 
if (v_isShared_4361_ == 0)
{
v___x_4363_ = v___x_4360_;
goto v_reusejp_4362_;
}
else
{
lean_object* v_reuseFailAlloc_4364_; 
v_reuseFailAlloc_4364_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4364_, 0, v_a_4357_);
lean_ctor_set(v_reuseFailAlloc_4364_, 1, v_a_4358_);
v___x_4363_ = v_reuseFailAlloc_4364_;
goto v_reusejp_4362_;
}
v_reusejp_4362_:
{
return v___x_4363_;
}
}
}
}
else
{
lean_object* v_a_4366_; lean_object* v_a_4367_; lean_object* v___x_4369_; uint8_t v_isShared_4370_; uint8_t v_isSharedCheck_4374_; 
lean_dec(v_a_4341_);
lean_dec(v_a_4337_);
lean_dec(v_a_4333_);
lean_dec(v_a_4329_);
lean_dec(v_a_4326_);
lean_dec(v_a_4322_);
lean_dec(v_a_4319_);
lean_dec(v_a_4315_);
lean_dec_ref(v___x_4312_);
v_a_4366_ = lean_ctor_get(v___x_4343_, 0);
v_a_4367_ = lean_ctor_get(v___x_4343_, 1);
v_isSharedCheck_4374_ = !lean_is_exclusive(v___x_4343_);
if (v_isSharedCheck_4374_ == 0)
{
v___x_4369_ = v___x_4343_;
v_isShared_4370_ = v_isSharedCheck_4374_;
goto v_resetjp_4368_;
}
else
{
lean_inc(v_a_4367_);
lean_inc(v_a_4366_);
lean_dec(v___x_4343_);
v___x_4369_ = lean_box(0);
v_isShared_4370_ = v_isSharedCheck_4374_;
goto v_resetjp_4368_;
}
v_resetjp_4368_:
{
lean_object* v___x_4372_; 
if (v_isShared_4370_ == 0)
{
v___x_4372_ = v___x_4369_;
goto v_reusejp_4371_;
}
else
{
lean_object* v_reuseFailAlloc_4373_; 
v_reuseFailAlloc_4373_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4373_, 0, v_a_4366_);
lean_ctor_set(v_reuseFailAlloc_4373_, 1, v_a_4367_);
v___x_4372_ = v_reuseFailAlloc_4373_;
goto v_reusejp_4371_;
}
v_reusejp_4371_:
{
return v___x_4372_;
}
}
}
}
else
{
lean_object* v_a_4375_; lean_object* v_a_4376_; lean_object* v___x_4378_; uint8_t v_isShared_4379_; uint8_t v_isSharedCheck_4383_; 
lean_dec(v_a_4337_);
lean_dec(v_a_4333_);
lean_dec(v_a_4329_);
lean_dec(v_a_4326_);
lean_dec(v_a_4322_);
lean_dec(v_a_4319_);
lean_dec(v_a_4315_);
lean_dec_ref(v___x_4312_);
v_a_4375_ = lean_ctor_get(v___x_4340_, 0);
v_a_4376_ = lean_ctor_get(v___x_4340_, 1);
v_isSharedCheck_4383_ = !lean_is_exclusive(v___x_4340_);
if (v_isSharedCheck_4383_ == 0)
{
v___x_4378_ = v___x_4340_;
v_isShared_4379_ = v_isSharedCheck_4383_;
goto v_resetjp_4377_;
}
else
{
lean_inc(v_a_4376_);
lean_inc(v_a_4375_);
lean_dec(v___x_4340_);
v___x_4378_ = lean_box(0);
v_isShared_4379_ = v_isSharedCheck_4383_;
goto v_resetjp_4377_;
}
v_resetjp_4377_:
{
lean_object* v___x_4381_; 
if (v_isShared_4379_ == 0)
{
v___x_4381_ = v___x_4378_;
goto v_reusejp_4380_;
}
else
{
lean_object* v_reuseFailAlloc_4382_; 
v_reuseFailAlloc_4382_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4382_, 0, v_a_4375_);
lean_ctor_set(v_reuseFailAlloc_4382_, 1, v_a_4376_);
v___x_4381_ = v_reuseFailAlloc_4382_;
goto v_reusejp_4380_;
}
v_reusejp_4380_:
{
return v___x_4381_;
}
}
}
}
else
{
lean_object* v_a_4384_; lean_object* v_a_4385_; lean_object* v___x_4387_; uint8_t v_isShared_4388_; uint8_t v_isSharedCheck_4392_; 
lean_dec(v_a_4333_);
lean_dec(v_a_4329_);
lean_dec(v_a_4326_);
lean_dec(v_a_4322_);
lean_dec(v_a_4319_);
lean_dec(v_a_4315_);
lean_dec_ref(v___x_4312_);
v_a_4384_ = lean_ctor_get(v___x_4336_, 0);
v_a_4385_ = lean_ctor_get(v___x_4336_, 1);
v_isSharedCheck_4392_ = !lean_is_exclusive(v___x_4336_);
if (v_isSharedCheck_4392_ == 0)
{
v___x_4387_ = v___x_4336_;
v_isShared_4388_ = v_isSharedCheck_4392_;
goto v_resetjp_4386_;
}
else
{
lean_inc(v_a_4385_);
lean_inc(v_a_4384_);
lean_dec(v___x_4336_);
v___x_4387_ = lean_box(0);
v_isShared_4388_ = v_isSharedCheck_4392_;
goto v_resetjp_4386_;
}
v_resetjp_4386_:
{
lean_object* v___x_4390_; 
if (v_isShared_4388_ == 0)
{
v___x_4390_ = v___x_4387_;
goto v_reusejp_4389_;
}
else
{
lean_object* v_reuseFailAlloc_4391_; 
v_reuseFailAlloc_4391_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4391_, 0, v_a_4384_);
lean_ctor_set(v_reuseFailAlloc_4391_, 1, v_a_4385_);
v___x_4390_ = v_reuseFailAlloc_4391_;
goto v_reusejp_4389_;
}
v_reusejp_4389_:
{
return v___x_4390_;
}
}
}
}
else
{
lean_object* v_a_4393_; lean_object* v_a_4394_; lean_object* v___x_4396_; uint8_t v_isShared_4397_; uint8_t v_isSharedCheck_4401_; 
lean_dec(v_a_4329_);
lean_dec(v_a_4326_);
lean_dec(v_a_4322_);
lean_dec(v_a_4319_);
lean_dec(v_a_4315_);
lean_dec_ref(v___x_4312_);
v_a_4393_ = lean_ctor_get(v___x_4332_, 0);
v_a_4394_ = lean_ctor_get(v___x_4332_, 1);
v_isSharedCheck_4401_ = !lean_is_exclusive(v___x_4332_);
if (v_isSharedCheck_4401_ == 0)
{
v___x_4396_ = v___x_4332_;
v_isShared_4397_ = v_isSharedCheck_4401_;
goto v_resetjp_4395_;
}
else
{
lean_inc(v_a_4394_);
lean_inc(v_a_4393_);
lean_dec(v___x_4332_);
v___x_4396_ = lean_box(0);
v_isShared_4397_ = v_isSharedCheck_4401_;
goto v_resetjp_4395_;
}
v_resetjp_4395_:
{
lean_object* v___x_4399_; 
if (v_isShared_4397_ == 0)
{
v___x_4399_ = v___x_4396_;
goto v_reusejp_4398_;
}
else
{
lean_object* v_reuseFailAlloc_4400_; 
v_reuseFailAlloc_4400_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4400_, 0, v_a_4393_);
lean_ctor_set(v_reuseFailAlloc_4400_, 1, v_a_4394_);
v___x_4399_ = v_reuseFailAlloc_4400_;
goto v_reusejp_4398_;
}
v_reusejp_4398_:
{
return v___x_4399_;
}
}
}
}
else
{
lean_object* v_a_4402_; lean_object* v_a_4403_; lean_object* v___x_4405_; uint8_t v_isShared_4406_; uint8_t v_isSharedCheck_4410_; 
lean_dec(v_a_4326_);
lean_dec(v_a_4322_);
lean_dec(v_a_4319_);
lean_dec(v_a_4315_);
lean_dec_ref(v___x_4312_);
v_a_4402_ = lean_ctor_get(v___x_4328_, 0);
v_a_4403_ = lean_ctor_get(v___x_4328_, 1);
v_isSharedCheck_4410_ = !lean_is_exclusive(v___x_4328_);
if (v_isSharedCheck_4410_ == 0)
{
v___x_4405_ = v___x_4328_;
v_isShared_4406_ = v_isSharedCheck_4410_;
goto v_resetjp_4404_;
}
else
{
lean_inc(v_a_4403_);
lean_inc(v_a_4402_);
lean_dec(v___x_4328_);
v___x_4405_ = lean_box(0);
v_isShared_4406_ = v_isSharedCheck_4410_;
goto v_resetjp_4404_;
}
v_resetjp_4404_:
{
lean_object* v___x_4408_; 
if (v_isShared_4406_ == 0)
{
v___x_4408_ = v___x_4405_;
goto v_reusejp_4407_;
}
else
{
lean_object* v_reuseFailAlloc_4409_; 
v_reuseFailAlloc_4409_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4409_, 0, v_a_4402_);
lean_ctor_set(v_reuseFailAlloc_4409_, 1, v_a_4403_);
v___x_4408_ = v_reuseFailAlloc_4409_;
goto v_reusejp_4407_;
}
v_reusejp_4407_:
{
return v___x_4408_;
}
}
}
}
else
{
lean_object* v_a_4411_; lean_object* v_a_4412_; lean_object* v___x_4414_; uint8_t v_isShared_4415_; uint8_t v_isSharedCheck_4419_; 
lean_dec_ref(v___x_4324_);
lean_dec(v_a_4322_);
lean_dec(v_a_4319_);
lean_dec(v_a_4315_);
lean_dec_ref(v___x_4312_);
v_a_4411_ = lean_ctor_get(v___x_4325_, 0);
v_a_4412_ = lean_ctor_get(v___x_4325_, 1);
v_isSharedCheck_4419_ = !lean_is_exclusive(v___x_4325_);
if (v_isSharedCheck_4419_ == 0)
{
v___x_4414_ = v___x_4325_;
v_isShared_4415_ = v_isSharedCheck_4419_;
goto v_resetjp_4413_;
}
else
{
lean_inc(v_a_4412_);
lean_inc(v_a_4411_);
lean_dec(v___x_4325_);
v___x_4414_ = lean_box(0);
v_isShared_4415_ = v_isSharedCheck_4419_;
goto v_resetjp_4413_;
}
v_resetjp_4413_:
{
lean_object* v___x_4417_; 
if (v_isShared_4415_ == 0)
{
v___x_4417_ = v___x_4414_;
goto v_reusejp_4416_;
}
else
{
lean_object* v_reuseFailAlloc_4418_; 
v_reuseFailAlloc_4418_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4418_, 0, v_a_4411_);
lean_ctor_set(v_reuseFailAlloc_4418_, 1, v_a_4412_);
v___x_4417_ = v_reuseFailAlloc_4418_;
goto v_reusejp_4416_;
}
v_reusejp_4416_:
{
return v___x_4417_;
}
}
}
}
else
{
lean_object* v_a_4420_; lean_object* v_a_4421_; lean_object* v___x_4423_; uint8_t v_isShared_4424_; uint8_t v_isSharedCheck_4428_; 
lean_dec(v_a_4319_);
lean_dec(v_a_4315_);
lean_dec_ref(v___x_4312_);
v_a_4420_ = lean_ctor_get(v___x_4321_, 0);
v_a_4421_ = lean_ctor_get(v___x_4321_, 1);
v_isSharedCheck_4428_ = !lean_is_exclusive(v___x_4321_);
if (v_isSharedCheck_4428_ == 0)
{
v___x_4423_ = v___x_4321_;
v_isShared_4424_ = v_isSharedCheck_4428_;
goto v_resetjp_4422_;
}
else
{
lean_inc(v_a_4421_);
lean_inc(v_a_4420_);
lean_dec(v___x_4321_);
v___x_4423_ = lean_box(0);
v_isShared_4424_ = v_isSharedCheck_4428_;
goto v_resetjp_4422_;
}
v_resetjp_4422_:
{
lean_object* v___x_4426_; 
if (v_isShared_4424_ == 0)
{
v___x_4426_ = v___x_4423_;
goto v_reusejp_4425_;
}
else
{
lean_object* v_reuseFailAlloc_4427_; 
v_reuseFailAlloc_4427_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4427_, 0, v_a_4420_);
lean_ctor_set(v_reuseFailAlloc_4427_, 1, v_a_4421_);
v___x_4426_ = v_reuseFailAlloc_4427_;
goto v_reusejp_4425_;
}
v_reusejp_4425_:
{
return v___x_4426_;
}
}
}
}
else
{
lean_object* v_a_4429_; lean_object* v_a_4430_; lean_object* v___x_4432_; uint8_t v_isShared_4433_; uint8_t v_isSharedCheck_4437_; 
lean_dec(v_a_4315_);
lean_dec_ref(v___x_4312_);
v_a_4429_ = lean_ctor_get(v___x_4318_, 0);
v_a_4430_ = lean_ctor_get(v___x_4318_, 1);
v_isSharedCheck_4437_ = !lean_is_exclusive(v___x_4318_);
if (v_isSharedCheck_4437_ == 0)
{
v___x_4432_ = v___x_4318_;
v_isShared_4433_ = v_isSharedCheck_4437_;
goto v_resetjp_4431_;
}
else
{
lean_inc(v_a_4430_);
lean_inc(v_a_4429_);
lean_dec(v___x_4318_);
v___x_4432_ = lean_box(0);
v_isShared_4433_ = v_isSharedCheck_4437_;
goto v_resetjp_4431_;
}
v_resetjp_4431_:
{
lean_object* v___x_4435_; 
if (v_isShared_4433_ == 0)
{
v___x_4435_ = v___x_4432_;
goto v_reusejp_4434_;
}
else
{
lean_object* v_reuseFailAlloc_4436_; 
v_reuseFailAlloc_4436_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4436_, 0, v_a_4429_);
lean_ctor_set(v_reuseFailAlloc_4436_, 1, v_a_4430_);
v___x_4435_ = v_reuseFailAlloc_4436_;
goto v_reusejp_4434_;
}
v_reusejp_4434_:
{
return v___x_4435_;
}
}
}
}
else
{
lean_object* v_a_4438_; lean_object* v_a_4439_; lean_object* v___x_4441_; uint8_t v_isShared_4442_; uint8_t v_isSharedCheck_4446_; 
lean_dec_ref(v___x_4312_);
v_a_4438_ = lean_ctor_get(v___x_4314_, 0);
v_a_4439_ = lean_ctor_get(v___x_4314_, 1);
v_isSharedCheck_4446_ = !lean_is_exclusive(v___x_4314_);
if (v_isSharedCheck_4446_ == 0)
{
v___x_4441_ = v___x_4314_;
v_isShared_4442_ = v_isSharedCheck_4446_;
goto v_resetjp_4440_;
}
else
{
lean_inc(v_a_4439_);
lean_inc(v_a_4438_);
lean_dec(v___x_4314_);
v___x_4441_ = lean_box(0);
v_isShared_4442_ = v_isSharedCheck_4446_;
goto v_resetjp_4440_;
}
v_resetjp_4440_:
{
lean_object* v___x_4444_; 
if (v_isShared_4442_ == 0)
{
v___x_4444_ = v___x_4441_;
goto v_reusejp_4443_;
}
else
{
lean_object* v_reuseFailAlloc_4445_; 
v_reuseFailAlloc_4445_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4445_, 0, v_a_4438_);
lean_ctor_set(v_reuseFailAlloc_4445_, 1, v_a_4439_);
v___x_4444_ = v_reuseFailAlloc_4445_;
goto v_reusejp_4443_;
}
v_reusejp_4443_:
{
return v___x_4444_;
}
}
}
}
}
static lean_object* _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readRawProofM___closed__0(void){
_start:
{
lean_object* v___x_4447_; lean_object* v___x_4448_; 
v___x_4447_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readTraceVData), 1, 0);
v___x_4448_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readOption), 3, 2);
lean_closure_set(v___x_4448_, 0, lean_box(0));
lean_closure_set(v___x_4448_, 1, v___x_4447_);
return v___x_4448_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readRawProofM(lean_object* v_a_4449_){
_start:
{
lean_object* v___x_4450_; lean_object* v___x_4451_; 
v___x_4450_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicProof;
v___x_4451_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readHeader(v___x_4450_, v_a_4449_);
if (lean_obj_tag(v___x_4451_) == 0)
{
lean_object* v_a_4452_; lean_object* v___x_4453_; 
v_a_4452_ = lean_ctor_get(v___x_4451_, 1);
lean_inc(v_a_4452_);
lean_dec_ref_known(v___x_4451_, 2);
v___x_4453_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readDigest(v_a_4452_);
if (lean_obj_tag(v___x_4453_) == 0)
{
lean_object* v_a_4454_; lean_object* v_a_4455_; lean_object* v___x_4456_; lean_object* v___x_4457_; 
v_a_4454_ = lean_ctor_get(v___x_4453_, 0);
lean_inc(v_a_4454_);
v_a_4455_ = lean_ctor_get(v___x_4453_, 1);
lean_inc(v_a_4455_);
lean_dec_ref_known(v___x_4453_, 2);
v___x_4456_ = lean_obj_once(&lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readRawProofM___closed__0, &lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readRawProofM___closed__0_once, _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readRawProofM___closed__0);
v___x_4457_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArr___redArg(v___x_4456_, v_a_4455_);
if (lean_obj_tag(v___x_4457_) == 0)
{
lean_object* v_a_4458_; lean_object* v_a_4459_; lean_object* v___x_4460_; 
v_a_4458_ = lean_ctor_get(v___x_4457_, 0);
lean_inc(v_a_4458_);
v_a_4459_ = lean_ctor_get(v___x_4457_, 1);
lean_inc(v_a_4459_);
lean_dec_ref_known(v___x_4457_, 2);
v___x_4460_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readGkrProof(v_a_4459_);
if (lean_obj_tag(v___x_4460_) == 0)
{
lean_object* v_a_4461_; lean_object* v_a_4462_; lean_object* v___x_4463_; 
v_a_4461_ = lean_ctor_get(v___x_4460_, 0);
lean_inc(v_a_4461_);
v_a_4462_ = lean_ctor_get(v___x_4460_, 1);
lean_inc(v_a_4462_);
lean_dec_ref_known(v___x_4460_, 2);
v___x_4463_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readBatchConstraintProof(v_a_4462_);
if (lean_obj_tag(v___x_4463_) == 0)
{
lean_object* v_a_4464_; lean_object* v_a_4465_; lean_object* v___x_4466_; 
v_a_4464_ = lean_ctor_get(v___x_4463_, 0);
lean_inc(v_a_4464_);
v_a_4465_ = lean_ctor_get(v___x_4463_, 1);
lean_inc(v_a_4465_);
lean_dec_ref_known(v___x_4463_, 2);
v___x_4466_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readStackingProof(v_a_4465_);
if (lean_obj_tag(v___x_4466_) == 0)
{
lean_object* v_a_4467_; lean_object* v_a_4468_; lean_object* v___x_4469_; 
v_a_4467_ = lean_ctor_get(v___x_4466_, 0);
lean_inc(v_a_4467_);
v_a_4468_ = lean_ctor_get(v___x_4466_, 1);
lean_inc(v_a_4468_);
lean_dec_ref_known(v___x_4466_, 2);
v___x_4469_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readWhirProof(v_a_4468_);
if (lean_obj_tag(v___x_4469_) == 0)
{
lean_object* v_a_4470_; lean_object* v_a_4471_; lean_object* v___x_4473_; uint8_t v_isShared_4474_; uint8_t v_isSharedCheck_4479_; 
v_a_4470_ = lean_ctor_get(v___x_4469_, 0);
v_a_4471_ = lean_ctor_get(v___x_4469_, 1);
v_isSharedCheck_4479_ = !lean_is_exclusive(v___x_4469_);
if (v_isSharedCheck_4479_ == 0)
{
v___x_4473_ = v___x_4469_;
v_isShared_4474_ = v_isSharedCheck_4479_;
goto v_resetjp_4472_;
}
else
{
lean_inc(v_a_4471_);
lean_inc(v_a_4470_);
lean_dec(v___x_4469_);
v___x_4473_ = lean_box(0);
v_isShared_4474_ = v_isSharedCheck_4479_;
goto v_resetjp_4472_;
}
v_resetjp_4472_:
{
lean_object* v___x_4475_; lean_object* v___x_4477_; 
v___x_4475_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_4475_, 0, v_a_4454_);
lean_ctor_set(v___x_4475_, 1, v_a_4458_);
lean_ctor_set(v___x_4475_, 2, v_a_4461_);
lean_ctor_set(v___x_4475_, 3, v_a_4464_);
lean_ctor_set(v___x_4475_, 4, v_a_4467_);
lean_ctor_set(v___x_4475_, 5, v_a_4470_);
if (v_isShared_4474_ == 0)
{
lean_ctor_set(v___x_4473_, 0, v___x_4475_);
v___x_4477_ = v___x_4473_;
goto v_reusejp_4476_;
}
else
{
lean_object* v_reuseFailAlloc_4478_; 
v_reuseFailAlloc_4478_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4478_, 0, v___x_4475_);
lean_ctor_set(v_reuseFailAlloc_4478_, 1, v_a_4471_);
v___x_4477_ = v_reuseFailAlloc_4478_;
goto v_reusejp_4476_;
}
v_reusejp_4476_:
{
return v___x_4477_;
}
}
}
else
{
lean_object* v_a_4480_; lean_object* v_a_4481_; lean_object* v___x_4483_; uint8_t v_isShared_4484_; uint8_t v_isSharedCheck_4488_; 
lean_dec(v_a_4467_);
lean_dec(v_a_4464_);
lean_dec(v_a_4461_);
lean_dec(v_a_4458_);
lean_dec(v_a_4454_);
v_a_4480_ = lean_ctor_get(v___x_4469_, 0);
v_a_4481_ = lean_ctor_get(v___x_4469_, 1);
v_isSharedCheck_4488_ = !lean_is_exclusive(v___x_4469_);
if (v_isSharedCheck_4488_ == 0)
{
v___x_4483_ = v___x_4469_;
v_isShared_4484_ = v_isSharedCheck_4488_;
goto v_resetjp_4482_;
}
else
{
lean_inc(v_a_4481_);
lean_inc(v_a_4480_);
lean_dec(v___x_4469_);
v___x_4483_ = lean_box(0);
v_isShared_4484_ = v_isSharedCheck_4488_;
goto v_resetjp_4482_;
}
v_resetjp_4482_:
{
lean_object* v___x_4486_; 
if (v_isShared_4484_ == 0)
{
v___x_4486_ = v___x_4483_;
goto v_reusejp_4485_;
}
else
{
lean_object* v_reuseFailAlloc_4487_; 
v_reuseFailAlloc_4487_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4487_, 0, v_a_4480_);
lean_ctor_set(v_reuseFailAlloc_4487_, 1, v_a_4481_);
v___x_4486_ = v_reuseFailAlloc_4487_;
goto v_reusejp_4485_;
}
v_reusejp_4485_:
{
return v___x_4486_;
}
}
}
}
else
{
lean_object* v_a_4489_; lean_object* v_a_4490_; lean_object* v___x_4492_; uint8_t v_isShared_4493_; uint8_t v_isSharedCheck_4497_; 
lean_dec(v_a_4464_);
lean_dec(v_a_4461_);
lean_dec(v_a_4458_);
lean_dec(v_a_4454_);
v_a_4489_ = lean_ctor_get(v___x_4466_, 0);
v_a_4490_ = lean_ctor_get(v___x_4466_, 1);
v_isSharedCheck_4497_ = !lean_is_exclusive(v___x_4466_);
if (v_isSharedCheck_4497_ == 0)
{
v___x_4492_ = v___x_4466_;
v_isShared_4493_ = v_isSharedCheck_4497_;
goto v_resetjp_4491_;
}
else
{
lean_inc(v_a_4490_);
lean_inc(v_a_4489_);
lean_dec(v___x_4466_);
v___x_4492_ = lean_box(0);
v_isShared_4493_ = v_isSharedCheck_4497_;
goto v_resetjp_4491_;
}
v_resetjp_4491_:
{
lean_object* v___x_4495_; 
if (v_isShared_4493_ == 0)
{
v___x_4495_ = v___x_4492_;
goto v_reusejp_4494_;
}
else
{
lean_object* v_reuseFailAlloc_4496_; 
v_reuseFailAlloc_4496_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4496_, 0, v_a_4489_);
lean_ctor_set(v_reuseFailAlloc_4496_, 1, v_a_4490_);
v___x_4495_ = v_reuseFailAlloc_4496_;
goto v_reusejp_4494_;
}
v_reusejp_4494_:
{
return v___x_4495_;
}
}
}
}
else
{
lean_object* v_a_4498_; lean_object* v_a_4499_; lean_object* v___x_4501_; uint8_t v_isShared_4502_; uint8_t v_isSharedCheck_4506_; 
lean_dec(v_a_4461_);
lean_dec(v_a_4458_);
lean_dec(v_a_4454_);
v_a_4498_ = lean_ctor_get(v___x_4463_, 0);
v_a_4499_ = lean_ctor_get(v___x_4463_, 1);
v_isSharedCheck_4506_ = !lean_is_exclusive(v___x_4463_);
if (v_isSharedCheck_4506_ == 0)
{
v___x_4501_ = v___x_4463_;
v_isShared_4502_ = v_isSharedCheck_4506_;
goto v_resetjp_4500_;
}
else
{
lean_inc(v_a_4499_);
lean_inc(v_a_4498_);
lean_dec(v___x_4463_);
v___x_4501_ = lean_box(0);
v_isShared_4502_ = v_isSharedCheck_4506_;
goto v_resetjp_4500_;
}
v_resetjp_4500_:
{
lean_object* v___x_4504_; 
if (v_isShared_4502_ == 0)
{
v___x_4504_ = v___x_4501_;
goto v_reusejp_4503_;
}
else
{
lean_object* v_reuseFailAlloc_4505_; 
v_reuseFailAlloc_4505_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4505_, 0, v_a_4498_);
lean_ctor_set(v_reuseFailAlloc_4505_, 1, v_a_4499_);
v___x_4504_ = v_reuseFailAlloc_4505_;
goto v_reusejp_4503_;
}
v_reusejp_4503_:
{
return v___x_4504_;
}
}
}
}
else
{
lean_object* v_a_4507_; lean_object* v_a_4508_; lean_object* v___x_4510_; uint8_t v_isShared_4511_; uint8_t v_isSharedCheck_4515_; 
lean_dec(v_a_4458_);
lean_dec(v_a_4454_);
v_a_4507_ = lean_ctor_get(v___x_4460_, 0);
v_a_4508_ = lean_ctor_get(v___x_4460_, 1);
v_isSharedCheck_4515_ = !lean_is_exclusive(v___x_4460_);
if (v_isSharedCheck_4515_ == 0)
{
v___x_4510_ = v___x_4460_;
v_isShared_4511_ = v_isSharedCheck_4515_;
goto v_resetjp_4509_;
}
else
{
lean_inc(v_a_4508_);
lean_inc(v_a_4507_);
lean_dec(v___x_4460_);
v___x_4510_ = lean_box(0);
v_isShared_4511_ = v_isSharedCheck_4515_;
goto v_resetjp_4509_;
}
v_resetjp_4509_:
{
lean_object* v___x_4513_; 
if (v_isShared_4511_ == 0)
{
v___x_4513_ = v___x_4510_;
goto v_reusejp_4512_;
}
else
{
lean_object* v_reuseFailAlloc_4514_; 
v_reuseFailAlloc_4514_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4514_, 0, v_a_4507_);
lean_ctor_set(v_reuseFailAlloc_4514_, 1, v_a_4508_);
v___x_4513_ = v_reuseFailAlloc_4514_;
goto v_reusejp_4512_;
}
v_reusejp_4512_:
{
return v___x_4513_;
}
}
}
}
else
{
lean_object* v_a_4516_; lean_object* v_a_4517_; lean_object* v___x_4519_; uint8_t v_isShared_4520_; uint8_t v_isSharedCheck_4524_; 
lean_dec(v_a_4454_);
v_a_4516_ = lean_ctor_get(v___x_4457_, 0);
v_a_4517_ = lean_ctor_get(v___x_4457_, 1);
v_isSharedCheck_4524_ = !lean_is_exclusive(v___x_4457_);
if (v_isSharedCheck_4524_ == 0)
{
v___x_4519_ = v___x_4457_;
v_isShared_4520_ = v_isSharedCheck_4524_;
goto v_resetjp_4518_;
}
else
{
lean_inc(v_a_4517_);
lean_inc(v_a_4516_);
lean_dec(v___x_4457_);
v___x_4519_ = lean_box(0);
v_isShared_4520_ = v_isSharedCheck_4524_;
goto v_resetjp_4518_;
}
v_resetjp_4518_:
{
lean_object* v___x_4522_; 
if (v_isShared_4520_ == 0)
{
v___x_4522_ = v___x_4519_;
goto v_reusejp_4521_;
}
else
{
lean_object* v_reuseFailAlloc_4523_; 
v_reuseFailAlloc_4523_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4523_, 0, v_a_4516_);
lean_ctor_set(v_reuseFailAlloc_4523_, 1, v_a_4517_);
v___x_4522_ = v_reuseFailAlloc_4523_;
goto v_reusejp_4521_;
}
v_reusejp_4521_:
{
return v___x_4522_;
}
}
}
}
else
{
lean_object* v_a_4525_; lean_object* v_a_4526_; lean_object* v___x_4528_; uint8_t v_isShared_4529_; uint8_t v_isSharedCheck_4533_; 
v_a_4525_ = lean_ctor_get(v___x_4453_, 0);
v_a_4526_ = lean_ctor_get(v___x_4453_, 1);
v_isSharedCheck_4533_ = !lean_is_exclusive(v___x_4453_);
if (v_isSharedCheck_4533_ == 0)
{
v___x_4528_ = v___x_4453_;
v_isShared_4529_ = v_isSharedCheck_4533_;
goto v_resetjp_4527_;
}
else
{
lean_inc(v_a_4526_);
lean_inc(v_a_4525_);
lean_dec(v___x_4453_);
v___x_4528_ = lean_box(0);
v_isShared_4529_ = v_isSharedCheck_4533_;
goto v_resetjp_4527_;
}
v_resetjp_4527_:
{
lean_object* v___x_4531_; 
if (v_isShared_4529_ == 0)
{
v___x_4531_ = v___x_4528_;
goto v_reusejp_4530_;
}
else
{
lean_object* v_reuseFailAlloc_4532_; 
v_reuseFailAlloc_4532_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4532_, 0, v_a_4525_);
lean_ctor_set(v_reuseFailAlloc_4532_, 1, v_a_4526_);
v___x_4531_ = v_reuseFailAlloc_4532_;
goto v_reusejp_4530_;
}
v_reusejp_4530_:
{
return v___x_4531_;
}
}
}
}
else
{
lean_object* v_a_4534_; lean_object* v_a_4535_; lean_object* v___x_4537_; uint8_t v_isShared_4538_; uint8_t v_isSharedCheck_4542_; 
v_a_4534_ = lean_ctor_get(v___x_4451_, 0);
v_a_4535_ = lean_ctor_get(v___x_4451_, 1);
v_isSharedCheck_4542_ = !lean_is_exclusive(v___x_4451_);
if (v_isSharedCheck_4542_ == 0)
{
v___x_4537_ = v___x_4451_;
v_isShared_4538_ = v_isSharedCheck_4542_;
goto v_resetjp_4536_;
}
else
{
lean_inc(v_a_4535_);
lean_inc(v_a_4534_);
lean_dec(v___x_4451_);
v___x_4537_ = lean_box(0);
v_isShared_4538_ = v_isSharedCheck_4542_;
goto v_resetjp_4536_;
}
v_resetjp_4536_:
{
lean_object* v___x_4540_; 
if (v_isShared_4538_ == 0)
{
v___x_4540_ = v___x_4537_;
goto v_reusejp_4539_;
}
else
{
lean_object* v_reuseFailAlloc_4541_; 
v_reuseFailAlloc_4541_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4541_, 0, v_a_4534_);
lean_ctor_set(v_reuseFailAlloc_4541_, 1, v_a_4535_);
v___x_4540_ = v_reuseFailAlloc_4541_;
goto v_reusejp_4539_;
}
v_reusejp_4539_:
{
return v___x_4540_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readRawProof(lean_object* v_data_4543_){
_start:
{
lean_object* v___x_4544_; lean_object* v___x_4545_; 
v___x_4544_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readRawProofM), 1, 0);
v___x_4545_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_runParser___redArg(v___x_4544_, v_data_4543_);
return v___x_4545_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readPvForArities(lean_object* v_x_4547_, lean_object* v_x_4548_, lean_object* v_a_4549_){
_start:
{
if (lean_obj_tag(v_x_4547_) == 0)
{
lean_object* v___x_4550_; 
v___x_4550_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4550_, 0, v_x_4548_);
lean_ctor_set(v___x_4550_, 1, v_a_4549_);
return v___x_4550_;
}
else
{
lean_object* v_head_4551_; lean_object* v_tail_4552_; lean_object* v___x_4553_; 
v_head_4551_ = lean_ctor_get(v_x_4547_, 0);
lean_inc(v_head_4551_);
v_tail_4552_ = lean_ctor_get(v_x_4547_, 1);
lean_inc(v_tail_4552_);
lean_dec_ref_known(v_x_4547_, 2);
lean_inc_ref(v_a_4549_);
v___x_4553_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_4549_);
if (lean_obj_tag(v___x_4553_) == 0)
{
lean_object* v_a_4554_; lean_object* v_a_4555_; lean_object* v___x_4557_; uint8_t v_isShared_4558_; uint8_t v_isSharedCheck_4603_; 
v_a_4554_ = lean_ctor_get(v___x_4553_, 0);
v_a_4555_ = lean_ctor_get(v___x_4553_, 1);
v_isSharedCheck_4603_ = !lean_is_exclusive(v___x_4553_);
if (v_isSharedCheck_4603_ == 0)
{
v___x_4557_ = v___x_4553_;
v_isShared_4558_ = v_isSharedCheck_4603_;
goto v_resetjp_4556_;
}
else
{
lean_inc(v_a_4555_);
lean_inc(v_a_4554_);
lean_dec(v___x_4553_);
v___x_4557_ = lean_box(0);
v_isShared_4558_ = v_isSharedCheck_4603_;
goto v_resetjp_4556_;
}
v_resetjp_4556_:
{
uint32_t v___x_4578_; uint32_t v___x_4579_; uint8_t v___x_4580_; 
v___x_4578_ = 0;
v___x_4579_ = lean_unbox_uint32(v_a_4554_);
v___x_4580_ = lean_uint32_dec_eq(v___x_4579_, v___x_4578_);
if (v___x_4580_ == 0)
{
uint32_t v___x_4581_; lean_object* v___x_4582_; uint8_t v___x_4583_; 
v___x_4581_ = lean_unbox_uint32(v_a_4554_);
v___x_4582_ = lean_uint32_to_nat(v___x_4581_);
v___x_4583_ = lean_nat_dec_eq(v___x_4582_, v_head_4551_);
if (v___x_4583_ == 0)
{
lean_object* v_offset_4584_; lean_object* v___x_4586_; uint8_t v_isShared_4587_; uint8_t v_isSharedCheck_4601_; 
lean_dec(v_a_4554_);
lean_dec(v_tail_4552_);
lean_dec_ref(v_x_4548_);
v_offset_4584_ = lean_ctor_get(v_a_4549_, 1);
v_isSharedCheck_4601_ = !lean_is_exclusive(v_a_4549_);
if (v_isSharedCheck_4601_ == 0)
{
lean_object* v_unused_4602_; 
v_unused_4602_ = lean_ctor_get(v_a_4549_, 0);
lean_dec(v_unused_4602_);
v___x_4586_ = v_a_4549_;
v_isShared_4587_ = v_isSharedCheck_4601_;
goto v_resetjp_4585_;
}
else
{
lean_inc(v_offset_4584_);
lean_dec(v_a_4549_);
v___x_4586_ = lean_box(0);
v_isShared_4587_ = v_isSharedCheck_4601_;
goto v_resetjp_4585_;
}
v_resetjp_4585_:
{
lean_object* v___x_4588_; lean_object* v___x_4589_; lean_object* v___x_4590_; lean_object* v___x_4591_; lean_object* v___x_4592_; lean_object* v___x_4593_; lean_object* v___x_4594_; lean_object* v___x_4596_; 
v___x_4588_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readPvForArities___closed__0));
v___x_4589_ = l_Nat_reprFast(v_head_4551_);
v___x_4590_ = lean_string_append(v___x_4588_, v___x_4589_);
lean_dec_ref(v___x_4589_);
v___x_4591_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__2));
v___x_4592_ = lean_string_append(v___x_4590_, v___x_4591_);
v___x_4593_ = l_Nat_reprFast(v___x_4582_);
v___x_4594_ = lean_string_append(v___x_4592_, v___x_4593_);
lean_dec_ref(v___x_4593_);
if (v_isShared_4587_ == 0)
{
lean_ctor_set_tag(v___x_4586_, 3);
lean_ctor_set(v___x_4586_, 1, v___x_4594_);
lean_ctor_set(v___x_4586_, 0, v_offset_4584_);
v___x_4596_ = v___x_4586_;
goto v_reusejp_4595_;
}
else
{
lean_object* v_reuseFailAlloc_4600_; 
v_reuseFailAlloc_4600_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4600_, 0, v_offset_4584_);
lean_ctor_set(v_reuseFailAlloc_4600_, 1, v___x_4594_);
v___x_4596_ = v_reuseFailAlloc_4600_;
goto v_reusejp_4595_;
}
v_reusejp_4595_:
{
lean_object* v___x_4598_; 
if (v_isShared_4558_ == 0)
{
lean_ctor_set_tag(v___x_4557_, 1);
lean_ctor_set(v___x_4557_, 0, v___x_4596_);
v___x_4598_ = v___x_4557_;
goto v_reusejp_4597_;
}
else
{
lean_object* v_reuseFailAlloc_4599_; 
v_reuseFailAlloc_4599_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4599_, 0, v___x_4596_);
lean_ctor_set(v_reuseFailAlloc_4599_, 1, v_a_4555_);
v___x_4598_ = v_reuseFailAlloc_4599_;
goto v_reusejp_4597_;
}
v_reusejp_4597_:
{
return v___x_4598_;
}
}
}
}
else
{
lean_dec(v___x_4582_);
lean_del_object(v___x_4557_);
lean_dec(v_head_4551_);
lean_dec_ref(v_a_4549_);
goto v___jp_4559_;
}
}
else
{
lean_del_object(v___x_4557_);
lean_dec(v_head_4551_);
lean_dec_ref(v_a_4549_);
goto v___jp_4559_;
}
v___jp_4559_:
{
lean_object* v___x_4560_; uint32_t v___x_4561_; lean_object* v___x_4562_; lean_object* v___x_4563_; lean_object* v___x_4564_; 
v___x_4560_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readCanonicalFBB), 1, 0);
v___x_4561_ = lean_unbox_uint32(v_a_4554_);
lean_dec(v_a_4554_);
v___x_4562_ = lean_uint32_to_nat(v___x_4561_);
v___x_4563_ = lean_mk_empty_array_with_capacity(v___x_4562_);
v___x_4564_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readArrayExact___redArg(v___x_4560_, v___x_4562_, v___x_4563_, v_a_4555_);
if (lean_obj_tag(v___x_4564_) == 0)
{
lean_object* v_a_4565_; lean_object* v_a_4566_; lean_object* v___x_4567_; 
v_a_4565_ = lean_ctor_get(v___x_4564_, 0);
lean_inc(v_a_4565_);
v_a_4566_ = lean_ctor_get(v___x_4564_, 1);
lean_inc(v_a_4566_);
lean_dec_ref_known(v___x_4564_, 2);
v___x_4567_ = lean_array_push(v_x_4548_, v_a_4565_);
v_x_4547_ = v_tail_4552_;
v_x_4548_ = v___x_4567_;
v_a_4549_ = v_a_4566_;
goto _start;
}
else
{
lean_object* v_a_4569_; lean_object* v_a_4570_; lean_object* v___x_4572_; uint8_t v_isShared_4573_; uint8_t v_isSharedCheck_4577_; 
lean_dec(v_tail_4552_);
lean_dec_ref(v_x_4548_);
v_a_4569_ = lean_ctor_get(v___x_4564_, 0);
v_a_4570_ = lean_ctor_get(v___x_4564_, 1);
v_isSharedCheck_4577_ = !lean_is_exclusive(v___x_4564_);
if (v_isSharedCheck_4577_ == 0)
{
v___x_4572_ = v___x_4564_;
v_isShared_4573_ = v_isSharedCheck_4577_;
goto v_resetjp_4571_;
}
else
{
lean_inc(v_a_4570_);
lean_inc(v_a_4569_);
lean_dec(v___x_4564_);
v___x_4572_ = lean_box(0);
v_isShared_4573_ = v_isSharedCheck_4577_;
goto v_resetjp_4571_;
}
v_resetjp_4571_:
{
lean_object* v___x_4575_; 
if (v_isShared_4573_ == 0)
{
v___x_4575_ = v___x_4572_;
goto v_reusejp_4574_;
}
else
{
lean_object* v_reuseFailAlloc_4576_; 
v_reuseFailAlloc_4576_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4576_, 0, v_a_4569_);
lean_ctor_set(v_reuseFailAlloc_4576_, 1, v_a_4570_);
v___x_4575_ = v_reuseFailAlloc_4576_;
goto v_reusejp_4574_;
}
v_reusejp_4574_:
{
return v___x_4575_;
}
}
}
}
}
}
else
{
lean_object* v_a_4604_; lean_object* v_a_4605_; lean_object* v___x_4607_; uint8_t v_isShared_4608_; uint8_t v_isSharedCheck_4612_; 
lean_dec(v_tail_4552_);
lean_dec(v_head_4551_);
lean_dec_ref(v_a_4549_);
lean_dec_ref(v_x_4548_);
v_a_4604_ = lean_ctor_get(v___x_4553_, 0);
v_a_4605_ = lean_ctor_get(v___x_4553_, 1);
v_isSharedCheck_4612_ = !lean_is_exclusive(v___x_4553_);
if (v_isSharedCheck_4612_ == 0)
{
v___x_4607_ = v___x_4553_;
v_isShared_4608_ = v_isSharedCheck_4612_;
goto v_resetjp_4606_;
}
else
{
lean_inc(v_a_4605_);
lean_inc(v_a_4604_);
lean_dec(v___x_4553_);
v___x_4607_ = lean_box(0);
v_isShared_4608_ = v_isSharedCheck_4612_;
goto v_resetjp_4606_;
}
v_resetjp_4606_:
{
lean_object* v___x_4610_; 
if (v_isShared_4608_ == 0)
{
v___x_4610_ = v___x_4607_;
goto v_reusejp_4609_;
}
else
{
lean_object* v_reuseFailAlloc_4611_; 
v_reuseFailAlloc_4611_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4611_, 0, v_a_4604_);
lean_ctor_set(v_reuseFailAlloc_4611_, 1, v_a_4605_);
v___x_4610_ = v_reuseFailAlloc_4611_;
goto v_reusejp_4609_;
}
v_reusejp_4609_:
{
return v___x_4610_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readPublicValueRowsM(lean_object* v_arities_4614_, lean_object* v_a_4615_){
_start:
{
lean_object* v___x_4616_; lean_object* v___x_4617_; 
v___x_4616_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicPv;
v___x_4617_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readHeader(v___x_4616_, v_a_4615_);
if (lean_obj_tag(v___x_4617_) == 0)
{
lean_object* v_a_4618_; lean_object* v___x_4619_; 
v_a_4618_ = lean_ctor_get(v___x_4617_, 1);
lean_inc_n(v_a_4618_, 2);
lean_dec_ref_known(v___x_4617_, 2);
v___x_4619_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readUInt32(v_a_4618_);
if (lean_obj_tag(v___x_4619_) == 0)
{
lean_object* v_a_4620_; lean_object* v_a_4621_; lean_object* v___x_4623_; uint8_t v_isShared_4624_; uint8_t v_isSharedCheck_4669_; 
v_a_4620_ = lean_ctor_get(v___x_4619_, 0);
v_a_4621_ = lean_ctor_get(v___x_4619_, 1);
v_isSharedCheck_4669_ = !lean_is_exclusive(v___x_4619_);
if (v_isSharedCheck_4669_ == 0)
{
v___x_4623_ = v___x_4619_;
v_isShared_4624_ = v_isSharedCheck_4669_;
goto v_resetjp_4622_;
}
else
{
lean_inc(v_a_4621_);
lean_inc(v_a_4620_);
lean_dec(v___x_4619_);
v___x_4623_ = lean_box(0);
v_isShared_4624_ = v_isSharedCheck_4669_;
goto v_resetjp_4622_;
}
v_resetjp_4622_:
{
uint32_t v___x_4625_; lean_object* v___x_4626_; lean_object* v___x_4627_; uint8_t v___x_4628_; 
v___x_4625_ = lean_unbox_uint32(v_a_4620_);
lean_dec(v_a_4620_);
v___x_4626_ = lean_uint32_to_nat(v___x_4625_);
v___x_4627_ = lean_array_get_size(v_arities_4614_);
v___x_4628_ = lean_nat_dec_eq(v___x_4626_, v___x_4627_);
if (v___x_4628_ == 0)
{
lean_object* v_offset_4629_; lean_object* v___x_4631_; uint8_t v_isShared_4632_; uint8_t v_isSharedCheck_4646_; 
lean_dec_ref(v_arities_4614_);
v_offset_4629_ = lean_ctor_get(v_a_4618_, 1);
v_isSharedCheck_4646_ = !lean_is_exclusive(v_a_4618_);
if (v_isSharedCheck_4646_ == 0)
{
lean_object* v_unused_4647_; 
v_unused_4647_ = lean_ctor_get(v_a_4618_, 0);
lean_dec(v_unused_4647_);
v___x_4631_ = v_a_4618_;
v_isShared_4632_ = v_isSharedCheck_4646_;
goto v_resetjp_4630_;
}
else
{
lean_inc(v_offset_4629_);
lean_dec(v_a_4618_);
v___x_4631_ = lean_box(0);
v_isShared_4632_ = v_isSharedCheck_4646_;
goto v_resetjp_4630_;
}
v_resetjp_4630_:
{
lean_object* v___x_4633_; lean_object* v___x_4634_; lean_object* v___x_4635_; lean_object* v___x_4636_; lean_object* v___x_4637_; lean_object* v___x_4638_; lean_object* v___x_4639_; lean_object* v___x_4641_; 
v___x_4633_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readPublicValueRowsM___closed__0));
v___x_4634_ = l_Nat_reprFast(v___x_4627_);
v___x_4635_ = lean_string_append(v___x_4633_, v___x_4634_);
lean_dec_ref(v___x_4634_);
v___x_4636_ = ((lean_object*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString___closed__2));
v___x_4637_ = lean_string_append(v___x_4635_, v___x_4636_);
v___x_4638_ = l_Nat_reprFast(v___x_4626_);
v___x_4639_ = lean_string_append(v___x_4637_, v___x_4638_);
lean_dec_ref(v___x_4638_);
if (v_isShared_4632_ == 0)
{
lean_ctor_set_tag(v___x_4631_, 3);
lean_ctor_set(v___x_4631_, 1, v___x_4639_);
lean_ctor_set(v___x_4631_, 0, v_offset_4629_);
v___x_4641_ = v___x_4631_;
goto v_reusejp_4640_;
}
else
{
lean_object* v_reuseFailAlloc_4645_; 
v_reuseFailAlloc_4645_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4645_, 0, v_offset_4629_);
lean_ctor_set(v_reuseFailAlloc_4645_, 1, v___x_4639_);
v___x_4641_ = v_reuseFailAlloc_4645_;
goto v_reusejp_4640_;
}
v_reusejp_4640_:
{
lean_object* v___x_4643_; 
if (v_isShared_4624_ == 0)
{
lean_ctor_set_tag(v___x_4623_, 1);
lean_ctor_set(v___x_4623_, 0, v___x_4641_);
v___x_4643_ = v___x_4623_;
goto v_reusejp_4642_;
}
else
{
lean_object* v_reuseFailAlloc_4644_; 
v_reuseFailAlloc_4644_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4644_, 0, v___x_4641_);
lean_ctor_set(v_reuseFailAlloc_4644_, 1, v_a_4621_);
v___x_4643_ = v_reuseFailAlloc_4644_;
goto v_reusejp_4642_;
}
v_reusejp_4642_:
{
return v___x_4643_;
}
}
}
}
else
{
lean_object* v___x_4648_; lean_object* v___x_4649_; lean_object* v___x_4650_; 
lean_dec(v___x_4626_);
lean_del_object(v___x_4623_);
lean_dec(v_a_4618_);
v___x_4648_ = lean_array_to_list(v_arities_4614_);
v___x_4649_ = lean_mk_empty_array_with_capacity(v___x_4627_);
v___x_4650_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readPvForArities(v___x_4648_, v___x_4649_, v_a_4621_);
if (lean_obj_tag(v___x_4650_) == 0)
{
lean_object* v_a_4651_; lean_object* v_a_4652_; lean_object* v___x_4654_; uint8_t v_isShared_4655_; uint8_t v_isSharedCheck_4659_; 
v_a_4651_ = lean_ctor_get(v___x_4650_, 0);
v_a_4652_ = lean_ctor_get(v___x_4650_, 1);
v_isSharedCheck_4659_ = !lean_is_exclusive(v___x_4650_);
if (v_isSharedCheck_4659_ == 0)
{
v___x_4654_ = v___x_4650_;
v_isShared_4655_ = v_isSharedCheck_4659_;
goto v_resetjp_4653_;
}
else
{
lean_inc(v_a_4652_);
lean_inc(v_a_4651_);
lean_dec(v___x_4650_);
v___x_4654_ = lean_box(0);
v_isShared_4655_ = v_isSharedCheck_4659_;
goto v_resetjp_4653_;
}
v_resetjp_4653_:
{
lean_object* v___x_4657_; 
if (v_isShared_4655_ == 0)
{
v___x_4657_ = v___x_4654_;
goto v_reusejp_4656_;
}
else
{
lean_object* v_reuseFailAlloc_4658_; 
v_reuseFailAlloc_4658_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4658_, 0, v_a_4651_);
lean_ctor_set(v_reuseFailAlloc_4658_, 1, v_a_4652_);
v___x_4657_ = v_reuseFailAlloc_4658_;
goto v_reusejp_4656_;
}
v_reusejp_4656_:
{
return v___x_4657_;
}
}
}
else
{
lean_object* v_a_4660_; lean_object* v_a_4661_; lean_object* v___x_4663_; uint8_t v_isShared_4664_; uint8_t v_isSharedCheck_4668_; 
v_a_4660_ = lean_ctor_get(v___x_4650_, 0);
v_a_4661_ = lean_ctor_get(v___x_4650_, 1);
v_isSharedCheck_4668_ = !lean_is_exclusive(v___x_4650_);
if (v_isSharedCheck_4668_ == 0)
{
v___x_4663_ = v___x_4650_;
v_isShared_4664_ = v_isSharedCheck_4668_;
goto v_resetjp_4662_;
}
else
{
lean_inc(v_a_4661_);
lean_inc(v_a_4660_);
lean_dec(v___x_4650_);
v___x_4663_ = lean_box(0);
v_isShared_4664_ = v_isSharedCheck_4668_;
goto v_resetjp_4662_;
}
v_resetjp_4662_:
{
lean_object* v___x_4666_; 
if (v_isShared_4664_ == 0)
{
v___x_4666_ = v___x_4663_;
goto v_reusejp_4665_;
}
else
{
lean_object* v_reuseFailAlloc_4667_; 
v_reuseFailAlloc_4667_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4667_, 0, v_a_4660_);
lean_ctor_set(v_reuseFailAlloc_4667_, 1, v_a_4661_);
v___x_4666_ = v_reuseFailAlloc_4667_;
goto v_reusejp_4665_;
}
v_reusejp_4665_:
{
return v___x_4666_;
}
}
}
}
}
}
else
{
lean_object* v_a_4670_; lean_object* v_a_4671_; lean_object* v___x_4673_; uint8_t v_isShared_4674_; uint8_t v_isSharedCheck_4678_; 
lean_dec(v_a_4618_);
lean_dec_ref(v_arities_4614_);
v_a_4670_ = lean_ctor_get(v___x_4619_, 0);
v_a_4671_ = lean_ctor_get(v___x_4619_, 1);
v_isSharedCheck_4678_ = !lean_is_exclusive(v___x_4619_);
if (v_isSharedCheck_4678_ == 0)
{
v___x_4673_ = v___x_4619_;
v_isShared_4674_ = v_isSharedCheck_4678_;
goto v_resetjp_4672_;
}
else
{
lean_inc(v_a_4671_);
lean_inc(v_a_4670_);
lean_dec(v___x_4619_);
v___x_4673_ = lean_box(0);
v_isShared_4674_ = v_isSharedCheck_4678_;
goto v_resetjp_4672_;
}
v_resetjp_4672_:
{
lean_object* v___x_4676_; 
if (v_isShared_4674_ == 0)
{
v___x_4676_ = v___x_4673_;
goto v_reusejp_4675_;
}
else
{
lean_object* v_reuseFailAlloc_4677_; 
v_reuseFailAlloc_4677_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4677_, 0, v_a_4670_);
lean_ctor_set(v_reuseFailAlloc_4677_, 1, v_a_4671_);
v___x_4676_ = v_reuseFailAlloc_4677_;
goto v_reusejp_4675_;
}
v_reusejp_4675_:
{
return v___x_4676_;
}
}
}
}
else
{
lean_object* v_a_4679_; lean_object* v_a_4680_; lean_object* v___x_4682_; uint8_t v_isShared_4683_; uint8_t v_isSharedCheck_4687_; 
lean_dec_ref(v_arities_4614_);
v_a_4679_ = lean_ctor_get(v___x_4617_, 0);
v_a_4680_ = lean_ctor_get(v___x_4617_, 1);
v_isSharedCheck_4687_ = !lean_is_exclusive(v___x_4617_);
if (v_isSharedCheck_4687_ == 0)
{
v___x_4682_ = v___x_4617_;
v_isShared_4683_ = v_isSharedCheck_4687_;
goto v_resetjp_4681_;
}
else
{
lean_inc(v_a_4680_);
lean_inc(v_a_4679_);
lean_dec(v___x_4617_);
v___x_4682_ = lean_box(0);
v_isShared_4683_ = v_isSharedCheck_4687_;
goto v_resetjp_4681_;
}
v_resetjp_4681_:
{
lean_object* v___x_4685_; 
if (v_isShared_4683_ == 0)
{
v___x_4685_ = v___x_4682_;
goto v_reusejp_4684_;
}
else
{
lean_object* v_reuseFailAlloc_4686_; 
v_reuseFailAlloc_4686_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4686_, 0, v_a_4679_);
lean_ctor_set(v_reuseFailAlloc_4686_, 1, v_a_4680_);
v___x_4685_ = v_reuseFailAlloc_4686_;
goto v_reusejp_4684_;
}
v_reusejp_4684_:
{
return v___x_4685_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readPublicValueRows(lean_object* v_arities_4688_, lean_object* v_data_4689_){
_start:
{
lean_object* v___x_4690_; lean_object* v___x_4691_; 
v___x_4690_ = lean_alloc_closure((void*)(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readPublicValueRowsM), 2, 1);
lean_closure_set(v___x_4690_, 0, v_arities_4688_);
v___x_4691_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_runParser___redArg(v___x_4690_, v_data_4689_);
return v___x_4691_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Swirl_Protocol_Noninteractive_Wire_Raw_RawVk_publicValueArities_spec__0(size_t v_sz_4692_, size_t v_i_4693_, lean_object* v_bs_4694_){
_start:
{
uint8_t v___x_4695_; 
v___x_4695_ = lean_usize_dec_lt(v_i_4693_, v_sz_4692_);
if (v___x_4695_ == 0)
{
return v_bs_4694_;
}
else
{
lean_object* v_v_4696_; lean_object* v_params_4697_; uint32_t v_numPublicValues_4698_; lean_object* v___x_4699_; lean_object* v_bs_x27_4700_; lean_object* v___x_4701_; size_t v___x_4702_; size_t v___x_4703_; lean_object* v___x_4704_; 
v_v_4696_ = lean_array_uget_borrowed(v_bs_4694_, v_i_4693_);
v_params_4697_ = lean_ctor_get(v_v_4696_, 1);
v_numPublicValues_4698_ = lean_ctor_get_uint32(v_params_4697_, sizeof(void*)*1);
v___x_4699_ = lean_unsigned_to_nat(0u);
v_bs_x27_4700_ = lean_array_uset(v_bs_4694_, v_i_4693_, v___x_4699_);
v___x_4701_ = lean_uint32_to_nat(v_numPublicValues_4698_);
v___x_4702_ = ((size_t)1ULL);
v___x_4703_ = lean_usize_add(v_i_4693_, v___x_4702_);
v___x_4704_ = lean_array_uset(v_bs_x27_4700_, v_i_4693_, v___x_4701_);
v_i_4693_ = v___x_4703_;
v_bs_4694_ = v___x_4704_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Swirl_Protocol_Noninteractive_Wire_Raw_RawVk_publicValueArities_spec__0___boxed(lean_object* v_sz_4706_, lean_object* v_i_4707_, lean_object* v_bs_4708_){
_start:
{
size_t v_sz_boxed_4709_; size_t v_i_boxed_4710_; lean_object* v_res_4711_; 
v_sz_boxed_4709_ = lean_unbox_usize(v_sz_4706_);
lean_dec(v_sz_4706_);
v_i_boxed_4710_ = lean_unbox_usize(v_i_4707_);
lean_dec(v_i_4707_);
v_res_4711_ = lp_swirl_x2drbr_x2dformal___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Swirl_Protocol_Noninteractive_Wire_Raw_RawVk_publicValueArities_spec__0(v_sz_boxed_4709_, v_i_boxed_4710_, v_bs_4708_);
return v_res_4711_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawVk_publicValueArities(lean_object* v_vk_4712_){
_start:
{
lean_object* v_inner_4713_; lean_object* v_perAir_4714_; size_t v_sz_4715_; size_t v___x_4716_; lean_object* v___x_4717_; 
v_inner_4713_ = lean_ctor_get(v_vk_4712_, 0);
lean_inc_ref(v_inner_4713_);
lean_dec_ref(v_vk_4712_);
v_perAir_4714_ = lean_ctor_get(v_inner_4713_, 1);
lean_inc_ref(v_perAir_4714_);
lean_dec_ref(v_inner_4713_);
v_sz_4715_ = lean_array_size(v_perAir_4714_);
v___x_4716_ = ((size_t)0ULL);
v___x_4717_ = lp_swirl_x2drbr_x2dformal___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Swirl_Protocol_Noninteractive_Wire_Raw_RawVk_publicValueArities_spec__0(v_sz_4715_, v___x_4716_, v_perAir_4714_);
return v___x_4717_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readRawPublicValues(lean_object* v_vk_4718_, lean_object* v_data_4719_){
_start:
{
lean_object* v___x_4720_; lean_object* v___x_4721_; 
v___x_4720_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_RawVk_publicValueArities(v_vk_4718_);
v___x_4721_ = lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_readPublicValueRows(v___x_4720_, v_data_4719_);
return v___x_4721_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
lean_initialize_runtime_module();
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_BB__prime = _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_BB__prime();
lean_mark_persistent(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_BB__prime);
lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedParseError_default = _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedParseError_default();
lean_mark_persistent(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedParseError_default);
lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedParseError = _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedParseError();
lean_mark_persistent(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedParseError);
lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedCursor_default = _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedCursor_default();
lean_mark_persistent(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedCursor_default);
lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedCursor = _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedCursor();
lean_mark_persistent(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedCursor);
lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicProof = _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicProof();
lean_mark_persistent(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicProof);
lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicVk = _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicVk();
lean_mark_persistent(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicVk);
lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicPv = _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicPv();
lean_mark_persistent(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_magicPv);
lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_wireVersion = _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_wireVersion();
lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirRoundConfig_default = _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirRoundConfig_default();
lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirRoundConfig = _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirRoundConfig();
lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProximityStrategy_default = _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProximityStrategy_default();
lean_mark_persistent(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProximityStrategy_default);
lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProximityStrategy = _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProximityStrategy();
lean_mark_persistent(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawWhirProximityStrategy);
lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__0___boxed__const__1 = _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__0___boxed__const__1();
lean_mark_persistent(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default___closed__0___boxed__const__1);
lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default = _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default();
lean_mark_persistent(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData_default);
lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData = _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData();
lean_mark_persistent(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVerifierSinglePreprocessedData);
lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVk_default = _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVk_default();
lean_mark_persistent(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVk_default);
lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVk = _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVk();
lean_mark_persistent(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawVk);
lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims_default = _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims_default();
lean_mark_persistent(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims_default);
lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims = _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims();
lean_mark_persistent(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrLayerClaims);
lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrProof_default = _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrProof_default();
lean_mark_persistent(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrProof_default);
lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrProof = _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrProof();
lean_mark_persistent(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawGkrProof);
lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawProof_default = _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawProof_default();
lean_mark_persistent(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawProof_default);
lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawProof = _init_lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawProof();
lean_mark_persistent(lp_swirl_x2drbr_x2dformal_Swirl_Protocol_Noninteractive_Wire_Raw_instInhabitedRawProof);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
