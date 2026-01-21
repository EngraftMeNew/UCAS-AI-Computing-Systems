// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vmatrix_mult_4x4x4x16.h for the primary calling header

#include "Vmatrix_mult_4x4x4x16__pch.h"
#include "Vmatrix_mult_4x4x4x16__Syms.h"
#include "Vmatrix_mult_4x4x4x16___024root.h"

#ifdef VL_DEBUG
VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root___dump_triggers__stl(Vmatrix_mult_4x4x4x16___024root* vlSelf);
#endif  // VL_DEBUG

VL_ATTR_COLD void Vmatrix_mult_4x4x4x16___024root___eval_triggers__stl(Vmatrix_mult_4x4x4x16___024root* vlSelf) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_mult_4x4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_mult_4x4x4x16___024root___eval_triggers__stl\n"); );
    // Body
    vlSelf->__VstlTriggered.set(0U, (IData)(vlSelf->__VstlFirstIteration));
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vmatrix_mult_4x4x4x16___024root___dump_triggers__stl(vlSelf);
    }
#endif
}
