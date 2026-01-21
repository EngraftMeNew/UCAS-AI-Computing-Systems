// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "Vmatrix_mult_4x4x4x16__Syms.h"


void Vmatrix_mult_4x4x4x16___024root__trace_chg_0_sub_0(Vmatrix_mult_4x4x4x16___024root* vlSelf, VerilatedVcd::Buffer* bufp);

void Vmatrix_mult_4x4x4x16___024root__trace_chg_0(void* voidSelf, VerilatedVcd::Buffer* bufp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_mult_4x4x4x16___024root__trace_chg_0\n"); );
    // Init
    Vmatrix_mult_4x4x4x16___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vmatrix_mult_4x4x4x16___024root*>(voidSelf);
    Vmatrix_mult_4x4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    if (VL_UNLIKELY(!vlSymsp->__Vm_activity)) return;
    // Body
    Vmatrix_mult_4x4x4x16___024root__trace_chg_0_sub_0((&vlSymsp->TOP), bufp);
}

void Vmatrix_mult_4x4x4x16___024root__trace_chg_0_sub_0(Vmatrix_mult_4x4x4x16___024root* vlSelf, VerilatedVcd::Buffer* bufp) {
    if (false && vlSelf) {}  // Prevent unused
    Vmatrix_mult_4x4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_mult_4x4x4x16___024root__trace_chg_0_sub_0\n"); );
    // Init
    uint32_t* const oldp VL_ATTR_UNUSED = bufp->oldp(vlSymsp->__Vm_baseCode + 1);
    // Body
    if (VL_UNLIKELY(vlSelf->__Vm_traceActivity[1U])) {
        bufp->chgSData(oldp+0,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv0__activations[0]),16);
        bufp->chgSData(oldp+1,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv0__activations[1]),16);
        bufp->chgSData(oldp+2,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv0__activations[2]),16);
        bufp->chgSData(oldp+3,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv0__activations[3]),16);
        bufp->chgSData(oldp+4,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu0__weights[0]),16);
        bufp->chgSData(oldp+5,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu0__weights[1]),16);
        bufp->chgSData(oldp+6,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu0__weights[2]),16);
        bufp->chgSData(oldp+7,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu0__weights[3]),16);
        bufp->chgSData(oldp+8,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu1__weights[0]),16);
        bufp->chgSData(oldp+9,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu1__weights[1]),16);
        bufp->chgSData(oldp+10,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu1__weights[2]),16);
        bufp->chgSData(oldp+11,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu1__weights[3]),16);
        bufp->chgSData(oldp+12,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu2__weights[0]),16);
        bufp->chgSData(oldp+13,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu2__weights[1]),16);
        bufp->chgSData(oldp+14,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu2__weights[2]),16);
        bufp->chgSData(oldp+15,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu2__weights[3]),16);
        bufp->chgSData(oldp+16,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu3__weights[0]),16);
        bufp->chgSData(oldp+17,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu3__weights[1]),16);
        bufp->chgSData(oldp+18,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu3__weights[2]),16);
        bufp->chgSData(oldp+19,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellinp__ipu3__weights[3]),16);
        bufp->chgSData(oldp+20,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv1__activations[0]),16);
        bufp->chgSData(oldp+21,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv1__activations[1]),16);
        bufp->chgSData(oldp+22,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv1__activations[2]),16);
        bufp->chgSData(oldp+23,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv1__activations[3]),16);
        bufp->chgSData(oldp+24,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu0__weights[0]),16);
        bufp->chgSData(oldp+25,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu0__weights[1]),16);
        bufp->chgSData(oldp+26,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu0__weights[2]),16);
        bufp->chgSData(oldp+27,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu0__weights[3]),16);
        bufp->chgSData(oldp+28,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu1__weights[0]),16);
        bufp->chgSData(oldp+29,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu1__weights[1]),16);
        bufp->chgSData(oldp+30,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu1__weights[2]),16);
        bufp->chgSData(oldp+31,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu1__weights[3]),16);
        bufp->chgSData(oldp+32,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu2__weights[0]),16);
        bufp->chgSData(oldp+33,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu2__weights[1]),16);
        bufp->chgSData(oldp+34,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu2__weights[2]),16);
        bufp->chgSData(oldp+35,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu2__weights[3]),16);
        bufp->chgSData(oldp+36,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu3__weights[0]),16);
        bufp->chgSData(oldp+37,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu3__weights[1]),16);
        bufp->chgSData(oldp+38,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu3__weights[2]),16);
        bufp->chgSData(oldp+39,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellinp__ipu3__weights[3]),16);
        bufp->chgSData(oldp+40,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv2__activations[0]),16);
        bufp->chgSData(oldp+41,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv2__activations[1]),16);
        bufp->chgSData(oldp+42,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv2__activations[2]),16);
        bufp->chgSData(oldp+43,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv2__activations[3]),16);
        bufp->chgSData(oldp+44,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu0__weights[0]),16);
        bufp->chgSData(oldp+45,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu0__weights[1]),16);
        bufp->chgSData(oldp+46,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu0__weights[2]),16);
        bufp->chgSData(oldp+47,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu0__weights[3]),16);
        bufp->chgSData(oldp+48,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu1__weights[0]),16);
        bufp->chgSData(oldp+49,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu1__weights[1]),16);
        bufp->chgSData(oldp+50,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu1__weights[2]),16);
        bufp->chgSData(oldp+51,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu1__weights[3]),16);
        bufp->chgSData(oldp+52,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu2__weights[0]),16);
        bufp->chgSData(oldp+53,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu2__weights[1]),16);
        bufp->chgSData(oldp+54,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu2__weights[2]),16);
        bufp->chgSData(oldp+55,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu2__weights[3]),16);
        bufp->chgSData(oldp+56,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu3__weights[0]),16);
        bufp->chgSData(oldp+57,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu3__weights[1]),16);
        bufp->chgSData(oldp+58,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu3__weights[2]),16);
        bufp->chgSData(oldp+59,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellinp__ipu3__weights[3]),16);
        bufp->chgSData(oldp+60,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv3__activations[0]),16);
        bufp->chgSData(oldp+61,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv3__activations[1]),16);
        bufp->chgSData(oldp+62,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv3__activations[2]),16);
        bufp->chgSData(oldp+63,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellinp__mxv3__activations[3]),16);
        bufp->chgSData(oldp+64,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu0__weights[0]),16);
        bufp->chgSData(oldp+65,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu0__weights[1]),16);
        bufp->chgSData(oldp+66,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu0__weights[2]),16);
        bufp->chgSData(oldp+67,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu0__weights[3]),16);
        bufp->chgSData(oldp+68,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu1__weights[0]),16);
        bufp->chgSData(oldp+69,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu1__weights[1]),16);
        bufp->chgSData(oldp+70,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu1__weights[2]),16);
        bufp->chgSData(oldp+71,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu1__weights[3]),16);
        bufp->chgSData(oldp+72,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu2__weights[0]),16);
        bufp->chgSData(oldp+73,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu2__weights[1]),16);
        bufp->chgSData(oldp+74,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu2__weights[2]),16);
        bufp->chgSData(oldp+75,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu2__weights[3]),16);
        bufp->chgSData(oldp+76,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu3__weights[0]),16);
        bufp->chgSData(oldp+77,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu3__weights[1]),16);
        bufp->chgSData(oldp+78,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu3__weights[2]),16);
        bufp->chgSData(oldp+79,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellinp__ipu3__weights[3]),16);
    }
    if (VL_UNLIKELY(vlSelf->__Vm_traceActivity[2U])) {
        bufp->chgIData(oldp+80,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv0__results[0]),32);
        bufp->chgIData(oldp+81,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv0__results[1]),32);
        bufp->chgIData(oldp+82,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv0__results[2]),32);
        bufp->chgIData(oldp+83,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv0__results[3]),32);
        bufp->chgIData(oldp+84,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellout__ipu0__result),32);
        bufp->chgSData(oldp+85,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__activations_reg[0]),16);
        bufp->chgSData(oldp+86,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__activations_reg[1]),16);
        bufp->chgSData(oldp+87,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__activations_reg[2]),16);
        bufp->chgSData(oldp+88,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__activations_reg[3]),16);
        bufp->chgSData(oldp+89,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__weights_reg[0]),16);
        bufp->chgSData(oldp+90,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__weights_reg[1]),16);
        bufp->chgSData(oldp+91,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__weights_reg[2]),16);
        bufp->chgSData(oldp+92,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__weights_reg[3]),16);
        bufp->chgIData(oldp+93,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__activations_reg
                                                               [0U]), 
                                             VL_EXTENDS_II(32,16, 
                                                           vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__weights_reg
                                                           [0U])) 
                                 + (VL_MULS_III(32, 
                                                VL_EXTENDS_II(32,16, 
                                                              vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__activations_reg
                                                              [1U]), 
                                                VL_EXTENDS_II(32,16, 
                                                              vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__weights_reg
                                                              [1U])) 
                                    + (VL_MULS_III(32, 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__activations_reg
                                                                 [2U]), 
                                                   VL_EXTENDS_II(32,16, 
                                                                 vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__weights_reg
                                                                 [2U])) 
                                       + VL_MULS_III(32, 
                                                     VL_EXTENDS_II(32,16, 
                                                                   vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__activations_reg
                                                                   [3U]), 
                                                     VL_EXTENDS_II(32,16, 
                                                                   vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__weights_reg
                                                                   [3U])))))),32);
        bufp->chgIData(oldp+94,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu0__DOT__i),32);
        bufp->chgIData(oldp+95,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellout__ipu1__result),32);
        bufp->chgSData(oldp+96,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__activations_reg[0]),16);
        bufp->chgSData(oldp+97,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__activations_reg[1]),16);
        bufp->chgSData(oldp+98,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__activations_reg[2]),16);
        bufp->chgSData(oldp+99,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__activations_reg[3]),16);
        bufp->chgSData(oldp+100,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__weights_reg[0]),16);
        bufp->chgSData(oldp+101,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__weights_reg[1]),16);
        bufp->chgSData(oldp+102,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__weights_reg[2]),16);
        bufp->chgSData(oldp+103,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__weights_reg[3]),16);
        bufp->chgIData(oldp+104,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__activations_reg
                                                                [0U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__weights_reg
                                                            [0U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__activations_reg
                                                               [1U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__weights_reg
                                                               [1U])) 
                                     + (VL_MULS_III(32, 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__activations_reg
                                                                  [2U]), 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__weights_reg
                                                                  [2U])) 
                                        + VL_MULS_III(32, 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__activations_reg
                                                                    [3U]), 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__weights_reg
                                                                    [3U])))))),32);
        bufp->chgIData(oldp+105,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu1__DOT__i),32);
        bufp->chgIData(oldp+106,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellout__ipu2__result),32);
        bufp->chgSData(oldp+107,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__activations_reg[0]),16);
        bufp->chgSData(oldp+108,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__activations_reg[1]),16);
        bufp->chgSData(oldp+109,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__activations_reg[2]),16);
        bufp->chgSData(oldp+110,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__activations_reg[3]),16);
        bufp->chgSData(oldp+111,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__weights_reg[0]),16);
        bufp->chgSData(oldp+112,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__weights_reg[1]),16);
        bufp->chgSData(oldp+113,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__weights_reg[2]),16);
        bufp->chgSData(oldp+114,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__weights_reg[3]),16);
        bufp->chgIData(oldp+115,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__activations_reg
                                                                [0U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__weights_reg
                                                            [0U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__activations_reg
                                                               [1U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__weights_reg
                                                               [1U])) 
                                     + (VL_MULS_III(32, 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__activations_reg
                                                                  [2U]), 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__weights_reg
                                                                  [2U])) 
                                        + VL_MULS_III(32, 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__activations_reg
                                                                    [3U]), 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__weights_reg
                                                                    [3U])))))),32);
        bufp->chgIData(oldp+116,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu2__DOT__i),32);
        bufp->chgIData(oldp+117,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT____Vcellout__ipu3__result),32);
        bufp->chgSData(oldp+118,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__activations_reg[0]),16);
        bufp->chgSData(oldp+119,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__activations_reg[1]),16);
        bufp->chgSData(oldp+120,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__activations_reg[2]),16);
        bufp->chgSData(oldp+121,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__activations_reg[3]),16);
        bufp->chgSData(oldp+122,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__weights_reg[0]),16);
        bufp->chgSData(oldp+123,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__weights_reg[1]),16);
        bufp->chgSData(oldp+124,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__weights_reg[2]),16);
        bufp->chgSData(oldp+125,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__weights_reg[3]),16);
        bufp->chgIData(oldp+126,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__activations_reg
                                                                [0U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__weights_reg
                                                            [0U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__activations_reg
                                                               [1U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__weights_reg
                                                               [1U])) 
                                     + (VL_MULS_III(32, 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__activations_reg
                                                                  [2U]), 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__weights_reg
                                                                  [2U])) 
                                        + VL_MULS_III(32, 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__activations_reg
                                                                    [3U]), 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__weights_reg
                                                                    [3U])))))),32);
        bufp->chgIData(oldp+127,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv0__DOT__ipu3__DOT__i),32);
        bufp->chgIData(oldp+128,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv1__results[0]),32);
        bufp->chgIData(oldp+129,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv1__results[1]),32);
        bufp->chgIData(oldp+130,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv1__results[2]),32);
        bufp->chgIData(oldp+131,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv1__results[3]),32);
        bufp->chgIData(oldp+132,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellout__ipu0__result),32);
        bufp->chgSData(oldp+133,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__activations_reg[0]),16);
        bufp->chgSData(oldp+134,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__activations_reg[1]),16);
        bufp->chgSData(oldp+135,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__activations_reg[2]),16);
        bufp->chgSData(oldp+136,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__activations_reg[3]),16);
        bufp->chgSData(oldp+137,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__weights_reg[0]),16);
        bufp->chgSData(oldp+138,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__weights_reg[1]),16);
        bufp->chgSData(oldp+139,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__weights_reg[2]),16);
        bufp->chgSData(oldp+140,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__weights_reg[3]),16);
        bufp->chgIData(oldp+141,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__activations_reg
                                                                [0U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__weights_reg
                                                            [0U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__activations_reg
                                                               [1U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__weights_reg
                                                               [1U])) 
                                     + (VL_MULS_III(32, 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__activations_reg
                                                                  [2U]), 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__weights_reg
                                                                  [2U])) 
                                        + VL_MULS_III(32, 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__activations_reg
                                                                    [3U]), 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__weights_reg
                                                                    [3U])))))),32);
        bufp->chgIData(oldp+142,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu0__DOT__i),32);
        bufp->chgIData(oldp+143,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellout__ipu1__result),32);
        bufp->chgSData(oldp+144,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__activations_reg[0]),16);
        bufp->chgSData(oldp+145,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__activations_reg[1]),16);
        bufp->chgSData(oldp+146,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__activations_reg[2]),16);
        bufp->chgSData(oldp+147,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__activations_reg[3]),16);
        bufp->chgSData(oldp+148,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__weights_reg[0]),16);
        bufp->chgSData(oldp+149,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__weights_reg[1]),16);
        bufp->chgSData(oldp+150,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__weights_reg[2]),16);
        bufp->chgSData(oldp+151,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__weights_reg[3]),16);
        bufp->chgIData(oldp+152,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__activations_reg
                                                                [0U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__weights_reg
                                                            [0U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__activations_reg
                                                               [1U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__weights_reg
                                                               [1U])) 
                                     + (VL_MULS_III(32, 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__activations_reg
                                                                  [2U]), 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__weights_reg
                                                                  [2U])) 
                                        + VL_MULS_III(32, 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__activations_reg
                                                                    [3U]), 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__weights_reg
                                                                    [3U])))))),32);
        bufp->chgIData(oldp+153,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu1__DOT__i),32);
        bufp->chgIData(oldp+154,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellout__ipu2__result),32);
        bufp->chgSData(oldp+155,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__activations_reg[0]),16);
        bufp->chgSData(oldp+156,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__activations_reg[1]),16);
        bufp->chgSData(oldp+157,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__activations_reg[2]),16);
        bufp->chgSData(oldp+158,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__activations_reg[3]),16);
        bufp->chgSData(oldp+159,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__weights_reg[0]),16);
        bufp->chgSData(oldp+160,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__weights_reg[1]),16);
        bufp->chgSData(oldp+161,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__weights_reg[2]),16);
        bufp->chgSData(oldp+162,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__weights_reg[3]),16);
        bufp->chgIData(oldp+163,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__activations_reg
                                                                [0U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__weights_reg
                                                            [0U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__activations_reg
                                                               [1U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__weights_reg
                                                               [1U])) 
                                     + (VL_MULS_III(32, 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__activations_reg
                                                                  [2U]), 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__weights_reg
                                                                  [2U])) 
                                        + VL_MULS_III(32, 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__activations_reg
                                                                    [3U]), 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__weights_reg
                                                                    [3U])))))),32);
        bufp->chgIData(oldp+164,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu2__DOT__i),32);
        bufp->chgIData(oldp+165,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT____Vcellout__ipu3__result),32);
        bufp->chgSData(oldp+166,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__activations_reg[0]),16);
        bufp->chgSData(oldp+167,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__activations_reg[1]),16);
        bufp->chgSData(oldp+168,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__activations_reg[2]),16);
        bufp->chgSData(oldp+169,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__activations_reg[3]),16);
        bufp->chgSData(oldp+170,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__weights_reg[0]),16);
        bufp->chgSData(oldp+171,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__weights_reg[1]),16);
        bufp->chgSData(oldp+172,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__weights_reg[2]),16);
        bufp->chgSData(oldp+173,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__weights_reg[3]),16);
        bufp->chgIData(oldp+174,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__activations_reg
                                                                [0U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__weights_reg
                                                            [0U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__activations_reg
                                                               [1U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__weights_reg
                                                               [1U])) 
                                     + (VL_MULS_III(32, 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__activations_reg
                                                                  [2U]), 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__weights_reg
                                                                  [2U])) 
                                        + VL_MULS_III(32, 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__activations_reg
                                                                    [3U]), 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__weights_reg
                                                                    [3U])))))),32);
        bufp->chgIData(oldp+175,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv1__DOT__ipu3__DOT__i),32);
        bufp->chgIData(oldp+176,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv2__results[0]),32);
        bufp->chgIData(oldp+177,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv2__results[1]),32);
        bufp->chgIData(oldp+178,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv2__results[2]),32);
        bufp->chgIData(oldp+179,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv2__results[3]),32);
        bufp->chgIData(oldp+180,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellout__ipu0__result),32);
        bufp->chgSData(oldp+181,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__activations_reg[0]),16);
        bufp->chgSData(oldp+182,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__activations_reg[1]),16);
        bufp->chgSData(oldp+183,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__activations_reg[2]),16);
        bufp->chgSData(oldp+184,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__activations_reg[3]),16);
        bufp->chgSData(oldp+185,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__weights_reg[0]),16);
        bufp->chgSData(oldp+186,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__weights_reg[1]),16);
        bufp->chgSData(oldp+187,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__weights_reg[2]),16);
        bufp->chgSData(oldp+188,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__weights_reg[3]),16);
        bufp->chgIData(oldp+189,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__activations_reg
                                                                [0U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__weights_reg
                                                            [0U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__activations_reg
                                                               [1U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__weights_reg
                                                               [1U])) 
                                     + (VL_MULS_III(32, 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__activations_reg
                                                                  [2U]), 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__weights_reg
                                                                  [2U])) 
                                        + VL_MULS_III(32, 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__activations_reg
                                                                    [3U]), 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__weights_reg
                                                                    [3U])))))),32);
        bufp->chgIData(oldp+190,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu0__DOT__i),32);
        bufp->chgIData(oldp+191,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellout__ipu1__result),32);
        bufp->chgSData(oldp+192,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__activations_reg[0]),16);
        bufp->chgSData(oldp+193,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__activations_reg[1]),16);
        bufp->chgSData(oldp+194,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__activations_reg[2]),16);
        bufp->chgSData(oldp+195,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__activations_reg[3]),16);
        bufp->chgSData(oldp+196,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__weights_reg[0]),16);
        bufp->chgSData(oldp+197,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__weights_reg[1]),16);
        bufp->chgSData(oldp+198,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__weights_reg[2]),16);
        bufp->chgSData(oldp+199,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__weights_reg[3]),16);
        bufp->chgIData(oldp+200,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__activations_reg
                                                                [0U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__weights_reg
                                                            [0U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__activations_reg
                                                               [1U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__weights_reg
                                                               [1U])) 
                                     + (VL_MULS_III(32, 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__activations_reg
                                                                  [2U]), 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__weights_reg
                                                                  [2U])) 
                                        + VL_MULS_III(32, 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__activations_reg
                                                                    [3U]), 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__weights_reg
                                                                    [3U])))))),32);
        bufp->chgIData(oldp+201,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu1__DOT__i),32);
        bufp->chgIData(oldp+202,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellout__ipu2__result),32);
        bufp->chgSData(oldp+203,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__activations_reg[0]),16);
        bufp->chgSData(oldp+204,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__activations_reg[1]),16);
        bufp->chgSData(oldp+205,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__activations_reg[2]),16);
        bufp->chgSData(oldp+206,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__activations_reg[3]),16);
        bufp->chgSData(oldp+207,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__weights_reg[0]),16);
        bufp->chgSData(oldp+208,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__weights_reg[1]),16);
        bufp->chgSData(oldp+209,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__weights_reg[2]),16);
        bufp->chgSData(oldp+210,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__weights_reg[3]),16);
        bufp->chgIData(oldp+211,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__activations_reg
                                                                [0U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__weights_reg
                                                            [0U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__activations_reg
                                                               [1U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__weights_reg
                                                               [1U])) 
                                     + (VL_MULS_III(32, 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__activations_reg
                                                                  [2U]), 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__weights_reg
                                                                  [2U])) 
                                        + VL_MULS_III(32, 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__activations_reg
                                                                    [3U]), 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__weights_reg
                                                                    [3U])))))),32);
        bufp->chgIData(oldp+212,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu2__DOT__i),32);
        bufp->chgIData(oldp+213,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT____Vcellout__ipu3__result),32);
        bufp->chgSData(oldp+214,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__activations_reg[0]),16);
        bufp->chgSData(oldp+215,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__activations_reg[1]),16);
        bufp->chgSData(oldp+216,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__activations_reg[2]),16);
        bufp->chgSData(oldp+217,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__activations_reg[3]),16);
        bufp->chgSData(oldp+218,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__weights_reg[0]),16);
        bufp->chgSData(oldp+219,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__weights_reg[1]),16);
        bufp->chgSData(oldp+220,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__weights_reg[2]),16);
        bufp->chgSData(oldp+221,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__weights_reg[3]),16);
        bufp->chgIData(oldp+222,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__activations_reg
                                                                [0U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__weights_reg
                                                            [0U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__activations_reg
                                                               [1U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__weights_reg
                                                               [1U])) 
                                     + (VL_MULS_III(32, 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__activations_reg
                                                                  [2U]), 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__weights_reg
                                                                  [2U])) 
                                        + VL_MULS_III(32, 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__activations_reg
                                                                    [3U]), 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__weights_reg
                                                                    [3U])))))),32);
        bufp->chgIData(oldp+223,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv2__DOT__ipu3__DOT__i),32);
        bufp->chgIData(oldp+224,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv3__results[0]),32);
        bufp->chgIData(oldp+225,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv3__results[1]),32);
        bufp->chgIData(oldp+226,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv3__results[2]),32);
        bufp->chgIData(oldp+227,(vlSelf->matrix_mult_4x4x4x16__DOT____Vcellout__mxv3__results[3]),32);
        bufp->chgIData(oldp+228,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellout__ipu0__result),32);
        bufp->chgSData(oldp+229,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__activations_reg[0]),16);
        bufp->chgSData(oldp+230,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__activations_reg[1]),16);
        bufp->chgSData(oldp+231,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__activations_reg[2]),16);
        bufp->chgSData(oldp+232,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__activations_reg[3]),16);
        bufp->chgSData(oldp+233,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__weights_reg[0]),16);
        bufp->chgSData(oldp+234,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__weights_reg[1]),16);
        bufp->chgSData(oldp+235,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__weights_reg[2]),16);
        bufp->chgSData(oldp+236,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__weights_reg[3]),16);
        bufp->chgIData(oldp+237,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__activations_reg
                                                                [0U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__weights_reg
                                                            [0U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__activations_reg
                                                               [1U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__weights_reg
                                                               [1U])) 
                                     + (VL_MULS_III(32, 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__activations_reg
                                                                  [2U]), 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__weights_reg
                                                                  [2U])) 
                                        + VL_MULS_III(32, 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__activations_reg
                                                                    [3U]), 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__weights_reg
                                                                    [3U])))))),32);
        bufp->chgIData(oldp+238,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu0__DOT__i),32);
        bufp->chgIData(oldp+239,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellout__ipu1__result),32);
        bufp->chgSData(oldp+240,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__activations_reg[0]),16);
        bufp->chgSData(oldp+241,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__activations_reg[1]),16);
        bufp->chgSData(oldp+242,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__activations_reg[2]),16);
        bufp->chgSData(oldp+243,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__activations_reg[3]),16);
        bufp->chgSData(oldp+244,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__weights_reg[0]),16);
        bufp->chgSData(oldp+245,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__weights_reg[1]),16);
        bufp->chgSData(oldp+246,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__weights_reg[2]),16);
        bufp->chgSData(oldp+247,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__weights_reg[3]),16);
        bufp->chgIData(oldp+248,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__activations_reg
                                                                [0U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__weights_reg
                                                            [0U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__activations_reg
                                                               [1U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__weights_reg
                                                               [1U])) 
                                     + (VL_MULS_III(32, 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__activations_reg
                                                                  [2U]), 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__weights_reg
                                                                  [2U])) 
                                        + VL_MULS_III(32, 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__activations_reg
                                                                    [3U]), 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__weights_reg
                                                                    [3U])))))),32);
        bufp->chgIData(oldp+249,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu1__DOT__i),32);
        bufp->chgIData(oldp+250,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellout__ipu2__result),32);
        bufp->chgSData(oldp+251,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__activations_reg[0]),16);
        bufp->chgSData(oldp+252,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__activations_reg[1]),16);
        bufp->chgSData(oldp+253,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__activations_reg[2]),16);
        bufp->chgSData(oldp+254,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__activations_reg[3]),16);
        bufp->chgSData(oldp+255,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__weights_reg[0]),16);
        bufp->chgSData(oldp+256,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__weights_reg[1]),16);
        bufp->chgSData(oldp+257,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__weights_reg[2]),16);
        bufp->chgSData(oldp+258,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__weights_reg[3]),16);
        bufp->chgIData(oldp+259,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__activations_reg
                                                                [0U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__weights_reg
                                                            [0U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__activations_reg
                                                               [1U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__weights_reg
                                                               [1U])) 
                                     + (VL_MULS_III(32, 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__activations_reg
                                                                  [2U]), 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__weights_reg
                                                                  [2U])) 
                                        + VL_MULS_III(32, 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__activations_reg
                                                                    [3U]), 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__weights_reg
                                                                    [3U])))))),32);
        bufp->chgIData(oldp+260,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu2__DOT__i),32);
        bufp->chgIData(oldp+261,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT____Vcellout__ipu3__result),32);
        bufp->chgSData(oldp+262,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__activations_reg[0]),16);
        bufp->chgSData(oldp+263,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__activations_reg[1]),16);
        bufp->chgSData(oldp+264,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__activations_reg[2]),16);
        bufp->chgSData(oldp+265,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__activations_reg[3]),16);
        bufp->chgSData(oldp+266,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__weights_reg[0]),16);
        bufp->chgSData(oldp+267,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__weights_reg[1]),16);
        bufp->chgSData(oldp+268,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__weights_reg[2]),16);
        bufp->chgSData(oldp+269,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__weights_reg[3]),16);
        bufp->chgIData(oldp+270,((VL_MULS_III(32, VL_EXTENDS_II(32,16, 
                                                                vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__activations_reg
                                                                [0U]), 
                                              VL_EXTENDS_II(32,16, 
                                                            vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__weights_reg
                                                            [0U])) 
                                  + (VL_MULS_III(32, 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__activations_reg
                                                               [1U]), 
                                                 VL_EXTENDS_II(32,16, 
                                                               vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__weights_reg
                                                               [1U])) 
                                     + (VL_MULS_III(32, 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__activations_reg
                                                                  [2U]), 
                                                    VL_EXTENDS_II(32,16, 
                                                                  vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__weights_reg
                                                                  [2U])) 
                                        + VL_MULS_III(32, 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__activations_reg
                                                                    [3U]), 
                                                      VL_EXTENDS_II(32,16, 
                                                                    vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__weights_reg
                                                                    [3U])))))),32);
        bufp->chgIData(oldp+271,(vlSelf->matrix_mult_4x4x4x16__DOT__mxv3__DOT__ipu3__DOT__i),32);
    }
    bufp->chgBit(oldp+272,(vlSelf->clk));
    bufp->chgBit(oldp+273,(vlSelf->reset));
    bufp->chgBit(oldp+274,(vlSelf->enable));
    bufp->chgSData(oldp+275,(vlSelf->activations[0U]
                             [0U]),16);
    bufp->chgSData(oldp+276,(vlSelf->activations[0U]
                             [1U]),16);
    bufp->chgSData(oldp+277,(vlSelf->activations[0U]
                             [2U]),16);
    bufp->chgSData(oldp+278,(vlSelf->activations[0U]
                             [3U]),16);
    bufp->chgSData(oldp+279,(vlSelf->activations[1U]
                             [0U]),16);
    bufp->chgSData(oldp+280,(vlSelf->activations[1U]
                             [1U]),16);
    bufp->chgSData(oldp+281,(vlSelf->activations[1U]
                             [2U]),16);
    bufp->chgSData(oldp+282,(vlSelf->activations[1U]
                             [3U]),16);
    bufp->chgSData(oldp+283,(vlSelf->activations[2U]
                             [0U]),16);
    bufp->chgSData(oldp+284,(vlSelf->activations[2U]
                             [1U]),16);
    bufp->chgSData(oldp+285,(vlSelf->activations[2U]
                             [2U]),16);
    bufp->chgSData(oldp+286,(vlSelf->activations[2U]
                             [3U]),16);
    bufp->chgSData(oldp+287,(vlSelf->activations[3U]
                             [0U]),16);
    bufp->chgSData(oldp+288,(vlSelf->activations[3U]
                             [1U]),16);
    bufp->chgSData(oldp+289,(vlSelf->activations[3U]
                             [2U]),16);
    bufp->chgSData(oldp+290,(vlSelf->activations[3U]
                             [3U]),16);
    bufp->chgSData(oldp+291,(vlSelf->weights[0U][0U]),16);
    bufp->chgSData(oldp+292,(vlSelf->weights[0U][1U]),16);
    bufp->chgSData(oldp+293,(vlSelf->weights[0U][2U]),16);
    bufp->chgSData(oldp+294,(vlSelf->weights[0U][3U]),16);
    bufp->chgSData(oldp+295,(vlSelf->weights[1U][0U]),16);
    bufp->chgSData(oldp+296,(vlSelf->weights[1U][1U]),16);
    bufp->chgSData(oldp+297,(vlSelf->weights[1U][2U]),16);
    bufp->chgSData(oldp+298,(vlSelf->weights[1U][3U]),16);
    bufp->chgSData(oldp+299,(vlSelf->weights[2U][0U]),16);
    bufp->chgSData(oldp+300,(vlSelf->weights[2U][1U]),16);
    bufp->chgSData(oldp+301,(vlSelf->weights[2U][2U]),16);
    bufp->chgSData(oldp+302,(vlSelf->weights[2U][3U]),16);
    bufp->chgSData(oldp+303,(vlSelf->weights[3U][0U]),16);
    bufp->chgSData(oldp+304,(vlSelf->weights[3U][1U]),16);
    bufp->chgSData(oldp+305,(vlSelf->weights[3U][2U]),16);
    bufp->chgSData(oldp+306,(vlSelf->weights[3U][3U]),16);
    bufp->chgIData(oldp+307,(vlSelf->results[0U][0U]),32);
    bufp->chgIData(oldp+308,(vlSelf->results[0U][1U]),32);
    bufp->chgIData(oldp+309,(vlSelf->results[0U][2U]),32);
    bufp->chgIData(oldp+310,(vlSelf->results[0U][3U]),32);
    bufp->chgIData(oldp+311,(vlSelf->results[1U][0U]),32);
    bufp->chgIData(oldp+312,(vlSelf->results[1U][1U]),32);
    bufp->chgIData(oldp+313,(vlSelf->results[1U][2U]),32);
    bufp->chgIData(oldp+314,(vlSelf->results[1U][3U]),32);
    bufp->chgIData(oldp+315,(vlSelf->results[2U][0U]),32);
    bufp->chgIData(oldp+316,(vlSelf->results[2U][1U]),32);
    bufp->chgIData(oldp+317,(vlSelf->results[2U][2U]),32);
    bufp->chgIData(oldp+318,(vlSelf->results[2U][3U]),32);
    bufp->chgIData(oldp+319,(vlSelf->results[3U][0U]),32);
    bufp->chgIData(oldp+320,(vlSelf->results[3U][1U]),32);
    bufp->chgIData(oldp+321,(vlSelf->results[3U][2U]),32);
    bufp->chgIData(oldp+322,(vlSelf->results[3U][3U]),32);
}

void Vmatrix_mult_4x4x4x16___024root__trace_cleanup(void* voidSelf, VerilatedVcd* /*unused*/) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vmatrix_mult_4x4x4x16___024root__trace_cleanup\n"); );
    // Init
    Vmatrix_mult_4x4x4x16___024root* const __restrict vlSelf VL_ATTR_UNUSED = static_cast<Vmatrix_mult_4x4x4x16___024root*>(voidSelf);
    Vmatrix_mult_4x4x4x16__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    // Body
    vlSymsp->__Vm_activity = false;
    vlSymsp->TOP.__Vm_traceActivity[0U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[1U] = 0U;
    vlSymsp->TOP.__Vm_traceActivity[2U] = 0U;
}
