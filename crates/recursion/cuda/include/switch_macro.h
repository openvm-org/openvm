#pragma once

#define UNWRAP(...) __VA_ARGS__

#define CASE_BLOCK(N, name, body)                                                                  \
    case N: {                                                                                      \
        constexpr size_t name = N;                                                                 \
        UNWRAP body break;                                                                         \
    }

#define APPLY_CASES_1(name, body, c1) CASE_BLOCK(c1, name, body)
#define APPLY_CASES_2(name, body, c1, c2) APPLY_CASES_1(name, body, c1) CASE_BLOCK(c2, name, body)
#define APPLY_CASES_3(name, body, c1, c2, c3)                                                      \
    APPLY_CASES_2(name, body, c1, c2) CASE_BLOCK(c3, name, body)
#define APPLY_CASES_4(name, body, c1, c2, c3, c4)                                                  \
    APPLY_CASES_3(name, body, c1, c2, c3) CASE_BLOCK(c4, name, body)
#define APPLY_CASES_5(name, body, c1, c2, c3, c4, c5)                                              \
    APPLY_CASES_4(name, body, c1, c2, c3, c4) CASE_BLOCK(c5, name, body)
#define APPLY_CASES_6(name, body, c1, c2, c3, c4, c5, c6)                                          \
    APPLY_CASES_5(name, body, c1, c2, c3, c4, c5) CASE_BLOCK(c6, name, body)
#define APPLY_CASES_7(name, body, c1, c2, c3, c4, c5, c6, c7)                                      \
    APPLY_CASES_6(name, body, c1, c2, c3, c4, c5, c6) CASE_BLOCK(c7, name, body)
#define APPLY_CASES_8(name, body, c1, c2, c3, c4, c5, c6, c7, c8)                                  \
    APPLY_CASES_7(name, body, c1, c2, c3, c4, c5, c6, c7) CASE_BLOCK(c8, name, body)

#define GET_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, NAME, ...) NAME

#define SWITCH_BLOCK(value, name, body, ...)                                                                                                            \
    switch (value) {                                                                                                                                    \
        GET_MACRO(__VA_ARGS__, APPLY_CASES_8, APPLY_CASES_7, APPLY_CASES_6, APPLY_CASES_5, APPLY_CASES_4, APPLY_CASES_3, APPLY_CASES_2, APPLY_CASES_1)( \
            name, body, __VA_ARGS__                                                                                                                     \
        ) default : return cudaErrorInvalidConfiguration;                                                                                               \
    }
