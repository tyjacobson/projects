//
//  Author:     Tyler Jacobson
//
//  Email:      tj545714@ohio.edu
//
//  Name:       compile.rs (ir)
//
//  Date:       April 14, 2019
//
//  Purpose:    Will translate a program from GrumpyIR to equivalent GrumpyVM code.
//
//  Notes:      <Exp>  -> <term> <exp'> $
//              <prog> -> <fun_list> % <exp>
//

use crate::types::*;
use crate::types::Binop::*;
use crate::types::BoolBinop::*;
use crate::types::Exp::*;
use crate::types::Instr::*;

pub fn compile(e: &Exp) -> Vec<Instr> {
    //INVARIANT: e's result left on top of stack
    match e {
        EI32(i) => vec![II32(*i)],
        EBinop(b) => {
            let mut is_lhs = compile(&b.lhs);
            let mut is_rhs = compile(&b.rhs);
            let mut is_op =
                match b.op.clone() {
                    BPlus => vec![IPlus],
                    BTimes => vec![ITimes],
                    BMinus => vec![IMinus],
                    BDiv => vec![IDiv],
                    //BEq => vec![IEq],
                    //BLess => vec![ILess],
                };
            let mut is = vec![];
            is.append(&mut is_lhs);
            is.append(&mut is_rhs);
            is.append(&mut is_op);
            is
        }
        EBoolBinop(bb) => {
            let mut is_lhs = compile(&bb.lhs);
            let mut is_rhs = compile(&bb.rhs);
            let mut is_op =
                match bb.op.clone() {
                    BEq => vec![IEq],
                    BLess => vec![ILess],
                };
            let mut is = vec![];
            is.append(&mut is_lhs);
            is.append(&mut is_rhs);
            is.append(&mut is_op);
            is
        }
    }
}