#pragma once

template <typename T> struct ExecutionState {
    T pc[2];
    T timestamp;
};
