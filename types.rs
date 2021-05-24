//
//  Author:     Tyler Jacobson
//
//  Email:      tj545714@ohio.edu
//
//  Name:       types.rs (ir)
//
//  Date:       April 14, 2019
//
//  Purpose:    Will translate a program from GrumpyIR to equivalent GrumpyVM code.
//
//  Notes:      <Exp>  -> <term> <exp'> $
//              <prog> -> <fun_list> % <exp>
//
//              Types Done:
//                  +, -, *, /, =, <, i32, 

use std::string::{ToString};

/********************************************
 * Expression language
 ********************************************/

/*
trait ToString {
    fn show(&self) -> String;
}
*/
    
// Binary Expression Type (i32)
#[derive(Debug,Clone)]
pub enum Binop {
    BPlus,
    BTimes,
    BMinus,
    BDiv,
}

use crate::types::Binop::*;

// To String for i32 Binary Types
impl ToString for Binop {
    fn to_string(&self) -> String {
        match self {
            BPlus   => "+".to_string(),
            BTimes  => "*".to_string(),
            BMinus  => "-".to_string(),
            BDiv    => "/".to_string(),
        }
    }
}

// Binary Expression Type (bool)
#[derive(Debug,Clone)]
pub enum BoolBinop
{
    BEq,
    BLess,
}

use crate::types::BoolBinop::*;

// To String for boolean Binary Types
impl ToString for BoolBinop {
    fn to_string(&self) -> String {
        match self {
            BEq     => "=".to_string(),
            BLess   => "<".to_string()
        }
    }
}

// i32 Binary Expression struct
#[derive(Debug,Clone)]
pub struct Binexp {
    pub op: Binop,
    pub lhs: Exp,
    pub rhs: Exp
}

// i32 Binop trait
pub trait Interp {
    fn interp(&self) -> i32;
}

// implement i32 Binary Expression
impl Interp for Binexp {
    fn interp(&self) -> i32 {
        match self.op {
            BPlus => self.lhs.interp() + self.rhs.interp(),
            BTimes => self.lhs.interp() * self.rhs.interp(),
            BMinus => self.lhs.interp() - self.rhs.interp(),
            BDiv => self.lhs.interp() / self.rhs.interp(),
        }
    }
}

// To String for i32 Binary Expression
impl ToString for Binexp {
    fn to_string(&self) -> String {
        format!("({} {} {})", self.lhs.to_string(), self.op.to_string(), self.rhs.to_string())
    }
}

// boolean Binary Expression struct
#[derive(Debug,Clone)]
pub struct BoolBinexp {
    pub op: BoolBinop,
    pub lhs: Exp,
    pub rhs: Exp
}

// bool Binary trait
pub trait BoolInterp {
    fn bool_interp(&self) -> bool;
}

// implement boolean Binary Expression
impl BoolInterp for BoolBinexp {
    fn bool_interp(&self) -> bool {
        match self.op {
            BEq => self.lhs.interp() == self.rhs.interp(),
            BLess => self.lhs.interp() < self.rhs.interp(),
        }
    }
}

// To String boolean Binary Expression
impl ToString for BoolBinexp {
    fn to_string(&self) -> String {
        format!("({} {} {})", self.lhs.to_string(), self.op.to_string(), self.rhs.to_string())
    }
}

// Expression Enum
#[derive(Debug,Clone)]
pub enum Exp {
    EI32(i32),
    EBinop(Box<Binexp>),
    EBoolBinop(Box<BoolBinexp>),
}

use crate::types::Exp::*;

// implement the expression
impl Interp for Exp {
    fn interp(&self) -> i32 {
        match self {
            EI32(i)         => *i,
            EBinop(b)       => b.interp(),
            _               => {panic!("Wrong type sent to i32 interpretor");}
        }
    }
}

// implement the expression for boolean operators
impl BoolInterp for Exp {
    fn bool_interp(&self) -> bool {
        match self{
            EBoolBinop(bb)  => bb.bool_interp(),
            _               => {panic!("Wrong type sent to boolean interpretor");}
        }
    }
}

// To String for Expressions
impl ToString for Exp {
    fn to_string(&self) -> String {
        match self {
            EI32(i)         => i.to_string(),
            EBinop(b)       => b.to_string(),
            EBoolBinop(bb)  => bb.to_string()
        }
    }
}

// Instruction Type
#[derive(Debug,Clone)]
pub enum Instr {
    IPlus,
    ITimes,
    IMinus,
    IDiv,
    IEq,
    ILess,
    II32(i32),
}

// Data Types
#[derive(Debug,Clone)]
pub enum Var {
    Vbool(bool),
    Vi32(i32),
    //Vloc(u32),
    Vunit,
    //Vundef,
}

use crate::types::Instr::*;
use crate::types::Var::*;

// trait for unwrapping variables
pub trait Unwrap {
    fn unwrap(&self) -> i32;
}

// unwrap Var into i32
impl Unwrap for Var {
    fn unwrap(&self) -> i32 {
        match self {
            Vi32(i) => *i,
            _       => {panic!("Unwrapping type not Vi32");}
        }
    }
}

// VM Structure
#[derive(Debug)]
pub struct VM {
    pub stack: Vec<Var>,
    pub instrs: Vec<Instr>,
    pub pc: usize
}

// Implement the VM
impl VM {
    // initialize
    pub fn init(instrs: &[Instr]) -> VM {
        VM {
            stack: vec![],
            instrs: instrs.to_vec(),
            pc: 0
        }            
    }

    // execute
    pub fn run(&mut self) -> Option<Var> {
        'mainloop:loop {
            if self.pc >= self.instrs.len() { break 'mainloop };
            match self.instrs[self.pc] {
                // +
                IPlus => {
                    let v2 = self.stack.pop().expect("IPlus: missing arg v2");
                    let v1 = self.stack.pop().expect("IPlus: missing arg v1");
                    self.stack.push(Var::Vi32(v1.unwrap() + v2.unwrap()))
                },
                // *
                ITimes => {
                    let v2 = self.stack.pop().expect("ITimes: missing arg v2");
                    let v1 = self.stack.pop().expect("ITimes: missing arg v1");
                    self.stack.push(Var::Vi32(v1.unwrap() * v2.unwrap()))
                },
                // -
                IMinus => {
                    let v2 = self.stack.pop().expect("IPlus: missing arg v2");
                    let v1 = self.stack.pop().expect("IPlus: missing arg v1");
                    self.stack.push(Var::Vi32(v1.unwrap() - v2.unwrap()))
                }
                // /
                IDiv => {
                    let v2 = self.stack.pop().expect("IPlus: missing arg v2");
                    let v1 = self.stack.pop().expect("IPlus: missing arg v1");
                    self.stack.push(Var::Vi32(v1.unwrap() / v2.unwrap()))
                }
                // =
                IEq => {
                    let v2 = self.stack.pop().expect("IEq: missing arg v2");
                    let v1 = self.stack.pop().expect("IEq: missing arg v1");
                    self.stack.push(Var::Vbool(v1.unwrap() == v2.unwrap()))
                }
                // <
                ILess => {
                    let v2 = self.stack.pop().expect("ILess: missing arg v2");
                    let v1 = self.stack.pop().expect("ILess: missing arg v1");
                    self.stack.push(Var::Vbool(v1.unwrap() < v2.unwrap()))
                }
                // i32
                II32(i) => {
                    self.stack.push(Var::Vi32(i))
                }
            };
            self.pc = self.pc + 1
        }
        let res = self.stack[self.stack.len() - 1].clone();
        Some(res)
    }
}