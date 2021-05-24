//
//  Author:     Tyler Jacobson
//
//  Email:      tj545714@ohio.edu
//
//  Name:       lexer.rs (ir)
//
//  Date:       April 14, 2019
//
//  Purpose:    Will translate a program from GrumpyIR to equivalent GrumpyVM code.
//
//  Notes:      <Exp>  -> <term> <exp'> $
//              <prog> -> <fun_list> % <exp>
//

use regex::Regex;                   // allows the use of Regular Expressions

// Token Enum
#[derive(Debug,Clone,PartialEq)]
pub enum Tok{
    PLUS,
    TIMES,
    MINUS,
    DIV,
    EQ,
    LESS,
    I32(i32),
    DOLLAR,
    NEG,
    LET,
    SEQ,
    ALLOC,
    SET,
    GET,
    COND,
    FUNPTR,
    CALL,
    F,
    LPAREN,
    RPAREN,
    BOOL(bool),
    VAR(String),
    UNIT,
}

// Line Info for an instruction
#[derive(Debug, Clone)]
pub struct LineInfo {
    pub line_no: u64,
    pub col_no: u64
}

// implement functions for the line info
impl LineInfo {
    // increase line counter
    fn incr_line(&mut self, n: u64) {
        self.col_no = 0;
        self.line_no = self.line_no + n
    }

    // increase column counter
    fn incr_col(&mut self, n: u64){
        self.col_no = self.col_no + n
    }
}

// Lexer update macro
macro_rules! lex_upd {
    ($l:expr, $no_chars:expr, $tok:expr ) => {{
        $l.info.incr_col($no_chars);
        $l.rest = $l.rest.split_at($no_chars).1;
        if $l.comment_depth > 0 { lex($l)}
        else { Ok($tok) }
    }}
}

// Lexer State
#[derive(Debug, Clone)]
pub struct LexerState<'a> {
    comment_depth: u64,
    pub rest: &'a str,
    pub info: LineInfo,
}

// implement functions for the lexer state
impl<'a> LexerState<'a> {
    // initialize a new state
    pub fn new(s: &'a str) -> Self{
        LexerState{
            comment_depth: 0,
            rest: s.trim_end(),
            info: LineInfo{line_no: 1, col_no: 0},
        }
    }

    // peek at the current token
    pub fn peek(self: &mut LexerState<'a>) -> Option<Tok> {
        let revert = self.clone();
        match lex(self) {
            Ok(tok)  => {
                            *self = revert;
                            Some(tok)
                        },
            Err(err) => {
                            eprintln!("lexer error: {} at {}:{}",
                                        err, self.info.line_no, self.info.col_no);
                            None
                        }
            
        }
    }

    // get the next token
    pub fn next(self: &mut LexerState<'a>) -> Option<Tok> {
        match lex(self) {
            Ok(tok)  => Some(tok),
            Err(err) => {
                            eprintln!(r"lexer error: {} at {}:{}",
                                        err, self.info.line_no, self.info.col_no);
                            None
            }
        }
    }

    // eat a token
    pub fn eat(self: &mut LexerState<'a>, expected: Tok) -> Option<()> {
        if let Some(t) = self.next() {
            if t == expected { Some(()) }
            else { None }}
        else { None }
    }
}

// main lexer function
fn lex<'a>(l: &mut LexerState<'a>) -> Result<Tok, String> {
    let s = l.rest;

    //Comments
    if s.starts_with("/*") { 
        l.comment_depth = l.comment_depth + 1;
        l.rest = s.split_at(2).1;
        lex(l)
    }                
    else if s.starts_with("*/") {
        l.comment_depth = l.comment_depth - 1;
        l.rest = s.split_at(2).1;
        lex(l)
    }
    
    //Whitespace characters
    else if s.starts_with(" ") {
        l.info.incr_col(1);
        l.rest = s.split_at(1).1;
        lex(l)
    }

    else if s.starts_with("\t") {
        l.info.incr_col(1);
        l.rest = s.split_at(1).1;
        lex(l)
    }        

    //Newline character sequences
    else if s.starts_with("\r\n") {
        l.info.incr_line(1);
        l.rest = s.split_at(2).1;
        lex(l)
    }    
    else if s.starts_with("\r") {
        l.info.incr_line(1);
        l.rest = s.split_at(1).1;
        lex(l)
    }
    else if s.starts_with("\n") {
        l.info.incr_line(1);
        l.rest = s.split_at(1).1;
        lex(l)
    }

    //The rest
    else if s.starts_with("+") { lex_upd!(l, 1, Tok::PLUS) }    // Plus
    else if s.starts_with("*") { lex_upd!(l, 1, Tok::TIMES) }   // Times
    else if s.starts_with("-") { lex_upd!(l, 1, Tok::MINUS) }   // Minus
    else if s.starts_with("/") { lex_upd!(l, 1, Tok::DIV) }     // Divide
    else if s.starts_with("=") { lex_upd!(l, 1, Tok::EQ) }      // Equals
    else if s.starts_with("<") { lex_upd!(l, 1, Tok::LESS) }    // Less Than
    else if s.starts_with("$") { lex_upd!(l, 1, Tok::DOLLAR) }  // Dollar Sign
    else if s.starts_with("neg") {lex_upd!(l, 3, Tok::NEG) }    // Neg
    else if s.starts_with("let") {lex_upd!(l, 3, Tok::LET) }    // Let
    else if s.starts_with("seq") {lex_upd!(l, 3, Tok::SEQ) }    // Seq
    else if s.starts_with("alloc") {lex_upd!(l, 5, Tok::ALLOC) }// Alloc
    else if s.starts_with("set") {lex_upd!(l, 3, Tok::SET) }    // Set
    else if s.starts_with("get") {lex_upd!(l, 3, Tok::GET) }    // Get
    else if s.starts_with("cond") {lex_upd!(l, 4, Tok::COND) }  // Cond
    else if s.starts_with("funptr") {lex_upd!(l, 6, Tok::FUNPTR) }// Funptr
    else if s.starts_with("call") {lex_upd!(l, 4, Tok::CALL) }  // Call
    //else if s.starts_with("f") {lex_upd!(l, 1, Tok::F) }        // F (function pointer)
    else if s.starts_with("(") {lex_upd!(l, 1, Tok::LPAREN) }   // (
    else if s.starts_with(")") {lex_upd!(l, 1, Tok::RPAREN) }   // )
    else if s.starts_with("true") {lex_upd!(l, 4, Tok::BOOL(true)) }// True
    else if s.starts_with("false") {lex_upd!(l, 5, Tok::BOOL(false)) }// False
    else if s.starts_with("unit") {lex_upd!(l, 4, Tok::UNIT)}   // Unit
    else if s.starts_with("L") || s.starts_with("_L") {
        // Function (f e1 e2 eN)
        match Regex::new(r"^L[a-zA-Z0-9]+ | ^_L[a-zA-Z0-9]+").unwrap().find(s) {
            Some(mat) => {
                assert_eq!(mat.start(), 0);
                let (n, rest) = s.split_at(mat.end());
                l.info.incr_col(mat.end() as u64);
                l.rest = rest;
                if l.comment_depth > 0 {lex(l)}
                else { Ok(Tok::VAR(n.parse::<String>().unwrap()))}
            },
            None => {
                //Fall-through cases
                if s.len() > 0 {
                    if l.comment_depth > 0 {
                        //1. Currently lexing a comment
                        l.info.incr_col(1);
                        l.rest = l.rest.split_at(1).1;
                        lex(l)
                    } else {
                        //2. Otherwise, saw an unexpected token
                        Err(format!(r"unexpected token '{}'", s.split_at(1).0))
                    }
                } else {
                    //3. A token was requested but none exists
                    Err(format!("unexpected end of program"))
                }
            }
        }
    }
    else {
        match Regex::new(r"^\A[[:digit:]]+").unwrap().find(s) {
            Some(mat) => {
                assert_eq!(mat.start(), 0);
                let (n, rest) = s.split_at(mat.end());
                l.info.incr_col(mat.end() as u64);
                l.rest = rest;
                if l.comment_depth > 0 { lex(l) }
                else { Ok(Tok::I32(n.parse::<i32>().unwrap())) }
            },
            None => {
                //Fall-through cases
                if s.len() > 0 {
                    if l.comment_depth > 0 {
                        //1. Currently lexing a comment
                        l.info.incr_col(1);
                        l.rest = l.rest.split_at(1).1;
                        lex(l)
                    } else {
                        //2. Otherwise, saw an unexpected token
                        Err(format!(r"unexpected token '{}'", s.split_at(1).0))
                    }
                } else {
                    //3. A token was requested but none exists
                    Err(format!("unexpected end of program"))
                }
            }
        }
    }
}










