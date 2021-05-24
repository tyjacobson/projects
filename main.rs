//
//  Author:     Tyler Jacobson
//
//  Email:      tj545714@ohio.edu
//
//  Name:       main.rs (ir)
//
//  Date:       April 14, 2019
//
//  Purpose:    Will translate a program from GrumpyIR to equivalent GrumpyVM code.
//
//  Notes:      <Exp>  -> <term> <exp'> $
//              <prog> -> <fun_list> % <exp>
//

use regex::Regex;                   // allows the use of Regular Expressions
use std::io;                        // allows use of std input
use std::fs;                        // allows files to be accessed
use std::env;                       // allows use of env
use std::collections::HashMap;      // allows the use of HashMaps

#[allow(dead_code)]
mod lexer;
use self::lexer::{LexerState,Tok};

#[allow(dead_code)]
mod parser;
use self::parser::{parse};

#[allow(dead_code)]
mod types;
use self::types::{Interp, VM};

#[allow(dead_code)]
mod compile;
use self::compile::{compile};

fn main() -> io::Result<()> {

    // reads in the file name from the command line
    let input = env::args().nth(1).unwrap();

    // Removes line endings
    let file_name = input.trim();

    // Read file into a vector of bytecode
    let file = fs::read(file_name)?;


    std::process::exit(0);  // returns 0 as an exitcode
}