use std::convert::TryFrom;
use std::collections::hash_map::HashMap;

mod parse;
mod ast;
mod smallvec;

use ast::*;
use smallvec::*;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

macro_rules! error {
    ($ctx: expr, $pos: expr, $($fmt: expr),*) => {{
        $ctx.error($pos, format!($($fmt),*))
    }}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Intern(u32);

struct InternBuf {
    buf: String,
    #[allow(dead_code)]
    next: Option<Box<InternBuf>>
}

struct Interns {
    bufs: *mut InternBuf,
    ids: HashMap<&'static str, Intern>,
    interns: Vec<&'static str>,
}

impl Interns {
    #[cfg(test)]
    const BUF_CAPACITY: usize = 12;
    #[cfg(not(test))]
    const BUF_CAPACITY: usize = 16*1024*1024;

    pub fn new() -> Interns {
        Interns { bufs: std::ptr::null_mut(), ids: HashMap::new(), interns: Vec::new() }
    }

    pub fn empty_string() -> Intern {
        Intern(0)
    }

    // The whole point of this is to pack interned strings tightly in memory
    // with very few allocations.
    //
    // Rust will give us a static lifetime to dynamically allocated strs through
    // `bufs`, which we can store in `ids`. Consequently, we need to ensure
    // `bufs.buf` never reallocates its internal storage.
    //
    // You can do this without unsafe by inserting `hash(str)` as keys, and
    // managing collisions yourself, but, eh
    fn push(&mut self, str: &str) -> &'static str {
        let (fits, next) = if self.bufs.is_null() {
            (false, None)
        } else {
            let buf = unsafe { &(*self.bufs).buf };
            let fits =  buf.len() + str.len() < buf.capacity();
            (fits, Some(self.bufs))
        };
        if !fits {
            // We assume str.len() <<<< BUF_CAPACITY, and that any leftover
            // space in the current buffer is too small to care about
            let cap = str.len().max(Self::BUF_CAPACITY);
            let link = Box::new(InternBuf {
                buf: String::with_capacity(cap),
                next: next.and_then(|bufs| unsafe { Some(Box::from_raw(bufs)) })
            });
            self.bufs = Box::leak(link);
        }
        let buf = unsafe { &mut (*self.bufs).buf };
        let old_len = buf.len();
        let old_cap = buf.capacity();
        // It would be nice to do something that cannot realloc even in
        // principle but miri doesn't like set_len-ing and copying into the
        // internal Vec<u8> ?_?
        buf.push_str(str);
        assert_eq!(old_cap, buf.capacity());
        &buf[old_len..]
    }

    pub fn put(&mut self, str: &str) -> Intern {
        match self.ids.get(str) {
            None => {
                assert!(self.interns.len() < u32::MAX as usize);
                let id = Intern(self.ids.len() as u32);
                let owned = self.push(str);
                self.interns.push(owned);
                self.ids.insert(owned, id);
                id
            },
            Some(id) => *id
        }
    }

    pub fn to_str(&self, id: Intern) -> Option<&str> {
        self.interns.get(id.0 as usize).copied()
    }
}

impl Drop for Interns {
    fn drop(&mut self) {
        unsafe { if self.bufs.is_null() == false { Box::from_raw(self.bufs); } }
    }
}

#[test]
fn interns() {
    let mut interns = Interns::new();

    let empty = interns.put("");
    let big = "d".repeat(Interns::BUF_CAPACITY * 3);

    let a = interns.put("a");
    let b = interns.put("b");
    let c = interns.put("c");
    let a2 = interns.put("a");
    let d = interns.put(&big);
    let e = interns.put("e");

    assert_eq!(a, Intern(1));
    assert_eq!(b, Intern(2));
    assert_eq!(c, Intern(3));
    assert_eq!(d, Intern(4));
    assert_eq!(e, Intern(5));
    assert_eq!(a, a2);

    assert_eq!(interns.to_str(empty), Some(""));
    assert_eq!(interns.to_str(a), Some("a"));
    assert_eq!(interns.to_str(b), Some("b"));
    assert_eq!(interns.to_str(c), Some("c"));
    assert_eq!(interns.to_str(d), Some(big.as_str()));
    assert_eq!(interns.to_str(e), Some("e"));

    assert_eq!(interns.ids.len(), interns.interns.len());
}

fn new_interns_with_keywords() -> Interns {
    let mut result = Interns::new();
    result.put("");
    for k in (1..).scan((), |(), v| parse::Keyword::from_u32(v)) {
        result.put(k.to_str());
    }
    result
}

#[test]
fn keywords() {
    let mut interns = new_interns_with_keywords();

    let mut i = Intern(1);
    for k in (1..).scan((), |(), v| parse::Keyword::from_u32(v)) {
        assert_eq!(interns.put(k.to_str()), i);
        i.0 += 1;
    }
}

#[derive(Copy, Clone, Default, PartialEq, Eq, Debug, Hash)]
pub struct Type(u32);

#[allow(non_upper_case_globals)]
impl Type {
    const None: Type = Type(TypeKind::None as u32);
    const Int:  Type = Type(TypeKind::Int as u32);
    const I8:   Type = Type(TypeKind::I8 as u32);
    const I16:  Type = Type(TypeKind::I16 as u32);
    const I32:  Type = Type(TypeKind::I32 as u32);
    const I64:  Type = Type(TypeKind::I64 as u32);
    const U8:   Type = Type(TypeKind::U8 as u32);
    const U16:  Type = Type(TypeKind::U16 as u32);
    const U32:  Type = Type(TypeKind::U32 as u32);
    const U64:  Type = Type(TypeKind::U64 as u32);
    const F32:  Type = Type(TypeKind::F32 as u32);
    const F64:  Type = Type(TypeKind::F64 as u32);
    const Bool: Type = Type(TypeKind::Bool as u32);

    fn is_integer(self) -> bool {
        matches!(self, Type::I8|Type::I16|Type::I32|Type::I64|Type::U8|Type::U16|Type::U32|Type::U64|Type::Int)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum TypeKind {
    None,
    Int,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Bool,
    Func,
}

#[derive(Clone, Debug)]
pub struct TypeInfo {
    kind: TypeKind,
    name: Intern,
    items: Vec<Type>
}

fn eval_type(ctx: &Compiler, expr: TypeExpr) -> std::result::Result<Type, IncompleteError> {
    match expr {
        TypeExpr::Infer => todo!("error"),
        TypeExpr::Name(intern) => ctx.lookup_type(intern).ok_or_else(|| {
            IncompleteError { source_position: 0, msg: format!("unknown type '{}'", ctx.str(intern)) }
        })
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Op {
    Halt = 255,
    Noop = 0,
    Immediate,
    IntNeg,
    IntAdd,
    IntSub,
    IntMul,
    IntDiv,
    IntMod,
    IntLt,
    IntGt,
    IntEq,
    IntNEq,
    IntLtEq,
    IntGtEq,
    BitNeg,
    BitAnd,
    BitOr,
    BitXor,
    Not,
    CmpZero,
    LShift,
    RShift,
    LogicOr,
    LogicAnd,
    F32Neg,
    F32Add,
    F32Sub,
    F32Mul,
    F32Div,
    F32Lt,
    F32Gt,
    F32Eq,
    F32NEq,
    F32LtEq,
    F32GtEq,
    F64Neg,
    F64Add,
    F64Sub,
    F64Mul,
    F64Div,
    F64Lt,
    F64Gt,
    F64Eq,
    F64NEq,
    F64LtEq,
    F64GtEq,
    Move,
    MoveLower8,
    MoveLower16,
    MoveLower32,
    MoveAndSignExtendLower8,
    MoveAndSignExtendLower16,
    MoveAndSignExtendLower32,
    IntToFloat32,
    IntToFloat64,
    Float32ToInt,
    Float32To64,
    Float64ToInt,
    Float64To32,
    Jump,
    JumpIfZero,
    JumpIfNotZero,
    Return,
    Call,
    CallIndirect
}

#[derive(Debug)]
struct Instr {
    op: Op,
    dest: u8,
    left: u8,
    right: u8,
}

#[derive(Copy, Clone, Debug)]
struct FatInstr {
    op: Op,
    dest: u32,
    left: u32,
    right: u32,
}

impl FatInstr {
    const HALT: FatInstr = FatInstr { op: Op::Halt, dest: 0, left: 0, right: 0 };

    fn is_jump(&self) -> bool {
        use Op::*;
        matches!(self.op, Jump|JumpIfZero|JumpIfNotZero)
    }
}

#[derive(Clone, Copy)]
union RegValue {
    int: usize,
    sint: isize,
    wint: std::num::Wrapping<usize>,
    int64: u64,
    int32: (u32, u32),
    int16: (u16, u16, u16, u16),
    int8: (u8, u8, u8, u8, u8, u8, u8, u8),
    sint32: (i32, i32),
    sint16: (i16, i16, i16, i16),
    sint8: (i8, i8, i8, i8, i8, i8, i8, i8),
    float64: f64,
    float32: (f32, f32),
    b8: (bool, [bool; 7]),
}

impl RegValue {
    fn unpack(&self) -> FatInstr {
        unsafe { FatInstr { op: Op::Noop, dest: 0, left: self.int32.0, right: self.int32.1 } }
    }

    fn is_true(&self) -> bool {
        unsafe {
            debug_assert!(self.b8.0 == (self.int != 0));
            self.b8.0
        }
    }
}

impl std::fmt::Debug for RegValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            write!(f, "RegValue {{ {:#016x} }}", self.int)
        }
    }
}

impl From<usize> for RegValue {
    fn from(value: usize) -> Self {
        RegValue { int: value }
    }
}

impl From<std::num::Wrapping<usize>> for RegValue {
    fn from(value: std::num::Wrapping<usize>) -> Self {
        RegValue { wint: value }
    }
}

impl From<isize> for RegValue {
    fn from(value: isize) -> Self {
        RegValue { sint: value }
    }
}

impl From<u32> for RegValue {
    fn from(value: u32) -> Self {
        RegValue { int: value as usize }
    }
}

impl From<i32> for RegValue {
    fn from(value: i32) -> Self {
        RegValue { int: value as usize }
    }
}

impl From<(u32, u32)> for RegValue {
    fn from(value: (u32, u32)) -> Self {
        RegValue { int32: value }
    }
}

impl From<f64> for RegValue {
    fn from(value: f64) -> Self {
        RegValue { float64: value }
    }
}

impl From<f32> for RegValue {
    fn from(value: f32) -> Self {
        RegValue { float32: (value, 0.0f32) }
    }
}

impl From<bool> for RegValue {
    fn from(value: bool) -> Self {
        RegValue { b8: (value, [false;7]) }
    }
}

impl From<Intern> for RegValue {
    fn from(value: Intern) -> Self {
        RegValue { int32: (value.0, 0) }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Value {
    Int(isize),
    UInt(usize),
    F32(f32),
    F64(f64),
    Bool(bool),
    Ident(Intern)
}

impl Value {
    fn from(reg: RegValue, ty: Type) -> Value {
        match ty {
            Type::U8|Type::U16|Type::U32|Type::U64 => unsafe { Value::UInt(reg.int) }
            Type::F32  => unsafe { Value::F32(reg.float32.0) }
            Type::F64  => unsafe { Value::F64(reg.float64) }
            Type::Bool => unsafe { Value::Bool(reg.b8.0) }
            _ => unsafe { Value::Int(reg.sint) }
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Int(v) => v.fmt(f),
            Value::UInt(v) => v.fmt(f),
            Value::F32(v) => v.fmt(f),
            Value::F64(v) => v.fmt(f),
            Value::Bool(v) => v.fmt(f),
            Value::Ident(v) => v.0.fmt(f),
        }
    }
}

impl From<Value> for RegValue {
    fn from(value: Value) -> RegValue {
        match value {
            Value::Int(v) => v.into(),
            Value::UInt(v) => v.into(),
            Value::F32(v) => v.into(),
            Value::F64(v) => v.into(),
            Value::Bool(v) => v.into(),
            Value::Ident(v) => v.into(),
        }
    }
}

fn integer_promote(ty: Type) -> Type {
    match ty {
        Type::I8|Type::I16|Type::I32|Type::I64|Type::U8|Type::U16|Type::U32|Type::U64 => Type::Int,
        ty => ty
    }
}

fn unary_op(op: parse::TokenKind, ty: Type) -> Option<(Op, Type)> {
    use parse::TokenKind;
    match (op, integer_promote(ty)) {
        (TokenKind::Sub,    Type::Int) => Some((Op::IntNeg, Type::Int)),
        (TokenKind::BitNeg, Type::Int) => Some((Op::BitNeg, Type::Int)),
        (TokenKind::Not,    _        ) => Some((Op::Not,    Type::Bool)),
        (TokenKind::Sub,    Type::F32) => Some((Op::F32Neg, Type::F32)),
        (TokenKind::Sub,    Type::F64) => Some((Op::F64Neg, Type::F64)),
        _ => None
    }
}

fn binary_op(op: parse::TokenKind, left: Type, right: Type) -> Option<(Op, Type)> {
    use parse::TokenKind;
    let left = integer_promote(left);
    let right = integer_promote(right);
    if left != right {
        return None;
    }
    match (op, left) {
        (TokenKind::Add,      Type::Int) => Some((Op::IntAdd,  Type::Int)),
        (TokenKind::Sub,      Type::Int) => Some((Op::IntSub,  Type::Int)),
        (TokenKind::Mul,      Type::Int) => Some((Op::IntMul,  Type::Int)),
        (TokenKind::Div,      Type::Int) => Some((Op::IntDiv,  Type::Int)),
        (TokenKind::Mod,      Type::Int) => Some((Op::IntMod,  Type::Int)),
        (TokenKind::BitNeg,   Type::Int) => Some((Op::BitNeg,  Type::Int)),
        (TokenKind::BitAnd,   Type::Int) => Some((Op::BitAnd,  Type::Int)),
        (TokenKind::BitOr,    Type::Int) => Some((Op::BitOr,   Type::Int)),
        (TokenKind::BitXor,   Type::Int) => Some((Op::BitXor,  Type::Int)),
        (TokenKind::LShift,   Type::Int) => Some((Op::LShift,  Type::Int)),
        (TokenKind::RShift,   Type::Int) => Some((Op::RShift,  Type::Int)),
        (TokenKind::GtEq,     Type::Int) => Some((Op::IntGtEq, Type::Bool)),
        (TokenKind::Lt,       Type::Int) => Some((Op::IntLt,   Type::Bool)),
        (TokenKind::Gt,       Type::Int) => Some((Op::IntGt,   Type::Bool)),
        (TokenKind::Eq,       Type::Int) => Some((Op::IntEq,   Type::Bool)),
        (TokenKind::NEq,      Type::Int) => Some((Op::IntNEq,  Type::Bool)),
        (TokenKind::LtEq,     Type::Int) => Some((Op::IntLtEq, Type::Bool)),

        (TokenKind::LogicAnd, Type::Bool) => Some((Op::LogicAnd, Type::Bool)),
        (TokenKind::LogicOr,  Type::Bool) => Some((Op::LogicOr,  Type::Bool)),

        (TokenKind::Add,      Type::F32) => Some((Op::F32Add,  Type::F32)),
        (TokenKind::Sub,      Type::F32) => Some((Op::F32Sub,  Type::F32)),
        (TokenKind::Mul,      Type::F32) => Some((Op::F32Mul,  Type::F32)),
        (TokenKind::Div,      Type::F32) => Some((Op::F32Div,  Type::F32)),
        (TokenKind::Lt,       Type::F32) => Some((Op::F32Lt,   Type::Bool)),
        (TokenKind::Gt,       Type::F32) => Some((Op::F32Gt,   Type::Bool)),
        (TokenKind::Eq,       Type::F32) => Some((Op::F32Eq,   Type::Bool)),
        (TokenKind::NEq,      Type::F32) => Some((Op::F32NEq,  Type::Bool)),
        (TokenKind::LtEq,     Type::F32) => Some((Op::F32LtEq, Type::Bool)),
        (TokenKind::GtEq,     Type::F32) => Some((Op::F32GtEq, Type::Bool)),

        (TokenKind::Add,      Type::F64) => Some((Op::F64Add,  Type::F64)),
        (TokenKind::Sub,      Type::F64) => Some((Op::F64Sub,  Type::F64)),
        (TokenKind::Mul,      Type::F64) => Some((Op::F64Mul,  Type::F64)),
        (TokenKind::Div,      Type::F64) => Some((Op::F64Div,  Type::F64)),
        (TokenKind::Lt,       Type::F64) => Some((Op::F64Lt,   Type::Bool)),
        (TokenKind::Gt,       Type::F64) => Some((Op::F64Gt,   Type::Bool)),
        (TokenKind::Eq,       Type::F64) => Some((Op::F64Eq,   Type::Bool)),
        (TokenKind::NEq,      Type::F64) => Some((Op::F64NEq,  Type::Bool)),
        (TokenKind::LtEq,     Type::F64) => Some((Op::F64LtEq, Type::Bool)),
        (TokenKind::GtEq,     Type::F64) => Some((Op::F64GtEq, Type::Bool)),

        _ => None
    }
}

fn convert_op(from: Type, to: Type) -> Option<Op> {
    let from = integer_promote(from);
    let op = if from == to {
        Op::Noop
    } else if from == Type::Int {
        match to {
            Type::Bool => Op::CmpZero,
            Type::U8   => Op::MoveLower8,
            Type::U16  => Op::MoveLower16,
            Type::U32  => Op::MoveLower32,
            Type::U64  => Op::Move,
            Type::I8   => Op::MoveAndSignExtendLower8,
            Type::I16  => Op::MoveAndSignExtendLower16,
            Type::I32  => Op::MoveAndSignExtendLower32,
            Type::I64  => Op::Move,
            Type::F32  => Op::IntToFloat32,
            Type::F64  => Op::IntToFloat64,
            _ => return None
        }
    } else if to.is_integer() {
        match from {
            Type::Bool => Op::Noop,
            Type::F32  => Op::Float32ToInt,
            Type::F64  => Op::Float64ToInt,
            _ => return None
        }
    } else {
        match (from, to) {
            (Type::F32, Type::F64) => Op::Float32To64,
            (Type::F64, Type::F32) => Op::Float64To32,
            _ => return None
        }
    };
    Some(op)
}

fn apply_unary_op(op: Op, value: RegValue) -> RegValue {
    unsafe {
        match op {
            Op::Noop                     => value,
            Op::IntNeg                   => RegValue::from(-value.sint),
            Op::BitNeg                   => RegValue::from(!value.int),
            Op::Not                      => RegValue::from((value.int == 0) as usize),
            Op::CmpZero                  => RegValue::from((value.int != 0) as usize),
            Op::MoveLower8               => RegValue::from(value.int8.0     as usize),
            Op::MoveLower16              => RegValue::from(value.int16.0    as usize),
            Op::MoveLower32              => RegValue::from(value.int32.0    as usize),
            Op::MoveAndSignExtendLower8  => RegValue::from(value.sint8.0    as usize),
            Op::MoveAndSignExtendLower16 => RegValue::from(value.sint16.0   as usize),
            Op::MoveAndSignExtendLower32 => RegValue::from(value.sint32.0   as usize),
            Op::IntToFloat32             => RegValue::from(value.sint       as f32),
            Op::IntToFloat64             => RegValue::from(value.sint       as f64),
            Op::Float32ToInt             => RegValue::from(value.float32.0  as usize),
            Op::Float32To64              => RegValue::from(value.float32.0  as f64),
            Op::Float64ToInt             => RegValue::from(value.float64    as usize),
            Op::Float64To32              => RegValue::from(value.float64    as f32),
            Op::F32Neg                   => RegValue::from(-value.float32.0),
            Op::F64Neg                   => RegValue::from(-value.float64),
            _ => unreachable!()
        }
    }
}

fn apply_binary_op(op: Op, left: RegValue, right: RegValue) -> RegValue {
    unsafe {
        match op {
            Op::IntAdd   => RegValue::from(left.wint + right.wint),
            Op::IntSub   => RegValue::from(left.wint - right.wint),
            Op::IntMul   => RegValue::from(left.wint * right.wint),
            Op::IntDiv   => RegValue::from(left.wint / right.wint),
            Op::IntMod   => RegValue::from(left.wint % right.wint),
            Op::IntLt    => RegValue::from(left.int < right.int),
            Op::IntGt    => RegValue::from(left.int > right.int),
            Op::IntEq    => RegValue::from(left.int == right.int),
            Op::IntNEq   => RegValue::from(left.int != right.int),
            Op::IntLtEq  => RegValue::from(left.int <= right.int),
            Op::IntGtEq  => RegValue::from(left.int >= right.int),
            Op::BitAnd   => RegValue::from(left.int & right.int),
            Op::BitOr    => RegValue::from(left.int | right.int),
            Op::BitXor   => RegValue::from(left.int ^ right.int),

            Op::LShift   => RegValue::from(left.int << right.int),
            Op::RShift   => RegValue::from(left.int >> right.int),

            Op::LogicOr  => RegValue::from((left.b8.0 || right.b8.0) as usize),
            Op::LogicAnd => RegValue::from((left.b8.0 && right.b8.0) as usize),

            Op::F32Add   => RegValue::from(left.float32.0 + right.float32.0),
            Op::F32Sub   => RegValue::from(left.float32.0 - right.float32.0),
            Op::F32Mul   => RegValue::from(left.float32.0 * right.float32.0),
            Op::F32Div   => RegValue::from(left.float32.0 / right.float32.0),
            Op::F32Lt    => RegValue::from(left.float32.0 < right.float32.0),
            Op::F32Gt    => RegValue::from(left.float32.0 > right.float32.0),
            Op::F32Eq    => RegValue::from(left.float32.0 == right.float32.0),
            Op::F32NEq   => RegValue::from(left.float32.0 != right.float32.0),
            Op::F32LtEq  => RegValue::from(left.float32.0 <= right.float32.0),
            Op::F32GtEq  => RegValue::from(left.float32.0 >= right.float32.0),

            Op::F64Add   => RegValue::from(left.float64 + right.float64),
            Op::F64Sub   => RegValue::from(left.float64 - right.float64),
            Op::F64Mul   => RegValue::from(left.float64 * right.float64),
            Op::F64Div   => RegValue::from(left.float64 / right.float64),
            Op::F64Lt    => RegValue::from(left.float64 < right.float64),
            Op::F64Gt    => RegValue::from(left.float64 > right.float64),
            Op::F64Eq    => RegValue::from(left.float64 == right.float64),
            Op::F64NEq   => RegValue::from(left.float64 != right.float64),
            Op::F64LtEq  => RegValue::from(left.float64 <= right.float64),
            Op::F64GtEq  => RegValue::from(left.float64 >= right.float64),
            _ => unreachable!()
        }
    }
}

#[derive(Copy, Clone)]
struct Local {
    reg: u32,
    ty: Type,
}

impl Local {
    fn new(reg: u32, ty: Type) -> Local {
        Local { reg: reg, ty: ty }
    }
}

#[derive(Clone, Copy)]
struct Global {
    value: Value,
    ty: Type
}

#[derive(Default)]
struct Locals {
    local_values: Vec<Local>,
    local_keys: Vec<Intern>,
    top_scope_index: usize,
}

impl Locals {
    fn assert_invariants(&self) {
        debug_assert!(self.local_keys.len() == self.local_values.len());
        debug_assert!(self.top_scope_index <= self.local_values.len());
    }

    fn insert(&mut self, ident: Intern, local: Local) {
        self.assert_invariants();
        self.local_keys.push(ident);
        self.local_values.push(local);
    }

    fn get(&self, ident: Intern) -> Option<&Local> {
        self.assert_invariants();
        self.local_keys.iter()
            .rposition(|v| *v == ident)
            .map(|i| unsafe { self.local_values.get_unchecked(i) })
    }

    fn push_scope(&mut self) -> usize {
        self.assert_invariants();
        let result = self.top_scope_index;
        self.top_scope_index = self.local_values.len();
        result
    }

    fn restore_scope(&mut self, mark: usize) {
        self.assert_invariants();
        debug_assert!(mark <= self.top_scope_index);
        self.local_values.truncate(self.top_scope_index);
        self.local_keys.truncate(self.top_scope_index);
        self.top_scope_index = mark;
    }
}

fn value_fits(value: Option<RegValue>, ty: Type) -> bool {
    unsafe {
        match (value, ty) {
            (Some(value), Type::I8)  =>  i8::MIN as isize <= value.sint && value.sint <=  i8::MAX as isize,
            (Some(value), Type::I16) => i16::MIN as isize <= value.sint && value.sint <= i16::MAX as isize,
            (Some(value), Type::I32) => i32::MIN as isize <= value.sint && value.sint <= i32::MAX as isize,
            (Some(value), Type::U8)  =>  u8::MIN as usize <= value.int  && value.int  <=  u8::MAX as usize,
            (Some(value), Type::U16) => u16::MIN as usize <= value.int  && value.int  <= u16::MAX as usize,
            (Some(value), Type::U32) => u32::MIN as usize <= value.int  && value.int  <= u32::MAX as usize,
            _ => true
        }
    }
}

struct ExprGen {
    reg: u32,
    ty: Type,
    value: Option<RegValue>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Label(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Location(usize);

struct FatGen<'a> {
    ast: &'a Ast,
    code: Vec<FatInstr>,
    locals: Locals,
    reg_counter: i32,
    break_stack: Vec<Label>,
    continue_stack: Vec<Label>,
    labels: Vec<Location>,
    patches: Vec<(Label, Location)>,
    return_type: Type,
    error: Option<IncompleteError>,
}

impl FatGen<'_> {
    const BAD_REGISTER: u32 = 0x70000000;

    fn new(ast: &Ast) -> FatGen {
        FatGen {
            ast,
            code: Vec::new(),
            locals: Default::default(),
            reg_counter: 0,
            break_stack: Vec::new(),
            continue_stack: Vec::new(),
            labels: Vec::new(),
            patches: Vec::new(),
            return_type: Type(0),
            error: None,
        }
    }

    fn eval_type(&mut self, ctx: &Compiler, expr: TypeExpr) -> Type {
        eval_type(ctx, expr).unwrap_or_else(|err| {
            self.error(err.source_position, err.msg);
            Type(0)
        })
    }

    fn error(&mut self, source_position: usize, msg: String) {
        if self.error.is_none() {
            self.error = Some(IncompleteError { source_position, msg });
        }
    }

    fn inc_reg(&mut self) -> u32 {
        let result = self.reg_counter;
        self.reg_counter += 1;
        result as u32
    }

    fn put_inc(&mut self, op: Op, data: RegValue) -> u32 {
        let dest = self.inc_reg();
        self.code.push(FatInstr { op, dest, .. data.unpack() });
        dest
    }

    fn put(&mut self, op: Op, dest: u32, data: RegValue) {
        assert_ne!(dest, Self::BAD_REGISTER);
        self.code.push(FatInstr { op, dest, .. data.unpack() });
    }

    fn put_immediate(&mut self, expr: &ExprGen) -> u32 {
        match expr.value {
            Some(v) => self.put_inc(Op::Immediate, v),
            None => expr.reg
        }
    }

    fn put_jump(&mut self, label: Label) {
        self.patches.push((label, Location(self.code.len())));
        self.put(Op::Jump, 0, label.0.into());
    }

    fn put_jump_zero(&mut self, cond_register: u32, label: Label) {
        self.patches.push((label, Location(self.code.len())));
        self.put(Op::JumpIfZero, cond_register, label.0.into());
    }

    fn put_jump_nonzero(&mut self, cond_register: u32, label: Label) {
        self.patches.push((label, Location(self.code.len())));
        self.put(Op::JumpIfNotZero, cond_register, label.0.into());
    }

    fn label(&mut self) -> Label {
        let result = Label(self.labels.len());
        self.labels.push(Location(!0));
        result
    }

    fn label_here(&mut self) -> Label {
        let result = self.label();
        self.patch(result);
        result
    }

    fn push_break_label(&mut self) -> Label {
        let result = self.label();
        self.break_stack.push(result);
        result
    }

    fn pop_break_label(&mut self, label: Label) {
        let popped = self.break_stack.pop();
        assert_eq!(Some(label), popped);
    }

    fn break_label(&self) -> Option<Label> {
        self.break_stack.last().copied()
    }

    fn push_continue_label(&mut self) -> Label {
        let result = self.label();
        self.continue_stack.push(result);
        result
    }

    fn pop_continue_label(&mut self, label: Label) {
        let popped = self.continue_stack.pop();
        assert_eq!(Some(label), popped);
    }

    fn continue_label(&self) -> Option<Label> {
        self.continue_stack.last().copied()
    }

    fn patch(&mut self, label: Label) {
        assert!(self.labels[label.0] == Location(!0));
        self.labels[label.0] = Location(self.code.len());
    }

    fn apply_patches(&mut self) {
        for &(Label(label), Location(from)) in self.patches.iter() {
            let Location(to) = self.labels[label];
            assert!(to != !0);
            assert!(self.code[from].is_jump() && self.code[from].left == label as u32);
            let offset = (to as isize) - (from as isize) - 1;
            let offset = i32::try_from(offset).expect("Function too large") as u32;
            self.code[from].left = offset;
        }
        self.patches.clear();
    }

    fn expr_in_register(&mut self, ctx: &Compiler, expr: Expr) -> ExprGen {
        let mut result = self.expr(ctx, expr);
        result.reg = self.put_immediate(&result);
        result
    }

    fn expr(&mut self, ctx: &Compiler, expr: Expr) -> ExprGen {
        let mut result = ExprGen { reg: Self::BAD_REGISTER, ty: Type::None, value: None };
        match self.ast.expr_data(expr) {
            ExprData::Int(value) => {
                result.value = Some(value.into());
                result.ty = Type::Int;
            }
            ExprData::Float32(value) => {
                result.value = Some(value.into());
                result.ty = Type::F32;
            }
            ExprData::Float64(value) => {
                result.value = Some(value.into());
                result.ty = Type::F64;
            }
            ExprData::Name(name) => {
                if let Some(local) = self.locals.get(name) {
                    result.reg = local.reg;
                    result.ty = local.ty;
                } else if let Some(&global) = ctx.globals.get(&name) {
                    result.value = Some(global.value.into());
                    result.ty = global.ty;
                } else {
                    error!(self, self.ast.expr_source_position(expr), "Unknown identifier '{}'", ctx.str(name));
                }
            }
            ExprData::Unary(op_token, right_expr) => {
                let right = self.expr(ctx, right_expr);
                if let Some((op, result_ty)) = unary_op(op_token, right.ty) {
                    match right.value {
                        Some(rv) => result.value = Some(apply_unary_op(op, rv)),
                        None => {
                            let right_register = self.put_immediate(&right);
                            result.reg = self.put_inc(op, right_register.into());
                        }
                    }
                    result.ty = result_ty;
                } else {
                    error!(self, self.ast.expr_source_position(right_expr), "incompatible type {}{}", op_token, ctx.type_str(right.ty));
                }
            }
            ExprData::Binary(op_token, left_expr, right_expr) => {
                if op_token == parse::TokenKind::LogicAnd || op_token == parse::TokenKind::LogicOr {
                    // No constant folding here because it makes this code complicated for nothing in return
                    let short_circuit = self.label();
                    let done = self.label();
                    let left = self.expr_in_register(ctx, left_expr);
                    if op_token == parse::TokenKind::LogicAnd {
                        self.put_jump_zero(left.reg, short_circuit);
                    } else {
                        self.put_jump_nonzero(left.reg, short_circuit);
                    }
                    let right = self.expr_in_register(ctx, right_expr);
                    if let Some((_, result_ty)) = binary_op(op_token, left.ty, right.ty) {
                        result.ty = result_ty;
                        result.reg = self.inc_reg();
                        self.put(Op::Move, result.reg, right.reg.into());
                        self.put_jump(done);
                        self.patch(short_circuit);
                        self.put(Op::Move, result.reg, left.reg.into());
                        self.patch(done);
                    } else {
                        error!(self, self.ast.expr_source_position(left_expr), "incompatible types ({} {} {}), logical operators use booleans", ctx.type_str(left.ty), op_token, ctx.type_str(right.ty));
                    }
                } else {
                    let left = self.expr(ctx, left_expr);
                    let right = self.expr(ctx, right_expr);
                    if let Some((op, result_ty)) = binary_op(op_token, left.ty, right.ty) {
                        match (left.value, right.value) {
                            (Some(lv), Some(rv)) => result.value = Some(apply_binary_op(op, lv, rv)),
                            _ => {
                                let left_register = self.put_immediate(&left);
                                let right_register = self.put_immediate(&right);
                                result.reg = self.put_inc(op, (left_register, right_register).into());
                            }
                        }
                        result.ty = result_ty;
                    } else {
                        error!(self, self.ast.expr_source_position(left_expr), "incompatible types ({} {} {})", ctx.type_str(left.ty), op_token, ctx.type_str(right.ty));
                    }
                }
            }
            ExprData::Ternary(cond_expr, left_expr, right_expr) => {
                let cond = self.expr(ctx, cond_expr);
                if cond.ty == Type::Bool {
                    if let Some(cv) = cond.value {
                        // Descend into both branches to type check, then discard the generated code.
                        {
                            let top = self.code.len();
                            self.expr(ctx, left_expr);
                            self.expr(ctx, right_expr);
                            self.code.truncate(top);
                        }
                        if cv.is_true() {
                            result = self.expr(ctx, left_expr);
                        } else {
                            result = self.expr(ctx, right_expr);
                        }
                    } else {
                        let right_branch = self.label();
                        let done = self.label();
                        result.reg = self.inc_reg();
                        self.put_jump_zero(cond.reg, right_branch);
                        let left = self.expr_in_register(ctx, left_expr);
                        self.put(Op::Move, result.reg, left.reg.into());
                        self.put_jump(done);
                        self.patch(right_branch);
                        let right = self.expr_in_register(ctx, right_expr);
                        self.put(Op::Move, result.reg, right.reg.into());
                        self.patch(done);
                        if left.ty == right.ty {
                            result.ty = right.ty;
                        } else {
                            error!(self, self.ast.expr_source_position(left_expr), "incompatible types (... ? {} :: {})", ctx.type_str(left.ty), ctx.type_str(right.ty));
                        }
                    }
                } else {
                    error!(self, self.ast.expr_source_position(cond_expr), "ternary expression requires a boolean condition");
                }
            }
            ExprData::Call(callable, args) => {
                let addr = self.expr(ctx, callable);
                let info = &ctx.type_info(addr.ty).unwrap();
                if info.kind == TypeKind::Func {
                    let mut arg_gens = SmallVec::new();
                    for (i, expr) in args.enumerate() {
                        let gen = self.expr(ctx, expr);
                        if gen.ty != info.items[i] {
                            error!(self, self.ast.expr_source_position(expr), "argument {} is of type {}, found {}", i, ctx.type_str(info.items[i]), ctx.type_str(gen.ty));
                            break;
                        }
                        arg_gens.push((gen.value, gen.reg));
                    }
                    for &gen in arg_gens.iter() {
                        match gen.0 {
                            Some(gv) => self.put_inc(Op::Immediate, gv),
                            _ => self.put_inc(Op::Move, gen.1.into())
                        };
                    }
                    result.reg = match addr.value {
                        Some(func) => self.put_inc(Op::Call, func.into()),
                        _ => self.put_inc(Op::CallIndirect, addr.reg.into())
                    };
                    result.ty = *info.items.last().expect("tried to generate code for func without return type");
                } else {
                    // TODO: get a string representation of the whole `callable` expr for nicer error message
                    error!(self, self.ast.expr_source_position(callable), "cannot call a {:?}", info.kind);
                }
            }
            ExprData::Cast(expr, type_expr) => {
                let left = self.expr(ctx, expr);
                let to_ty = self.eval_type(ctx, type_expr);
                if let Some(op) = convert_op(left.ty, to_ty) {
                    match left.value {
                        Some(lv) => result.value = Some(apply_unary_op(op, lv)),
                        None => result.reg = if op != Op::Noop { self.put_inc(op, left.reg.into()) } else { left.reg }
                    }
                    result.ty = to_ty;
                } else {
                    error!(self, self.ast.expr_source_position(expr), "cannot cast from {} to {}", ctx.type_str(left.ty), ctx.type_str(to_ty));
                }
            }
        };
        result
    }

    fn stmt(&mut self, ctx: &Compiler, stmt: Stmt) -> Option<Type> {
        let mut return_type = None;
        match self.ast.stmt_data(stmt) {
            StmtData::Block(body) => {
                return_type = self.stmts(ctx, body)
            }
            StmtData::Return(Some(expr)) => {
                let ret_expr = self.expr_in_register(ctx, expr);
                if integer_promote(ret_expr.ty) == self.return_type {
                    self.put(Op::Return, 0, ret_expr.reg.into());
                    return_type = Some(ret_expr.ty);
                } else {
                    error!(self, self.ast.expr_source_position(expr), "type mismatch: expected a return value of type {} (found {})", ctx.type_str(self.return_type), ctx.type_str(ret_expr.ty));
                }
            }
            StmtData::Return(None) => {
                if self.return_type == Type(0) {
                    self.put(Op::Return, 0, 0.into());
                } else {
                    // TODO: stmt source position
                    error!(self, 0, "empty return, expected a return value of type {}", ctx.type_str(self.return_type));
                }
            }
            StmtData::Break => {
                if let Some(label) = self.break_label() {
                    self.put_jump(label);
                } else {
                    // TODO: stmt source position
                    error!(self, 0, "break outside of loop/switch");
                }
            }
            StmtData::Continue => {
                if let Some(label) = self.continue_label() {
                    self.put_jump(label);
                } else {
                    // TODO: stmt source position
                    error!(self, 0, "continue outside of loop");
                }
            }
            StmtData::If(cond_expr, then_body, else_body) => {
                let cond = self.expr_in_register(ctx, cond_expr);
                if cond.ty == Type::Bool {
                    let else_branch = self.label();
                    self.put_jump_zero(cond.reg, else_branch);
                    let then_ret = self.stmts(ctx, then_body);
                    if else_body.is_nonempty() {
                        let skip_else = self.label();
                        self.put_jump(skip_else);
                        self.patch(else_branch);
                        let else_ret = self.stmts(ctx, else_body);
                        self.patch(skip_else);
                        return_type = then_ret.and(else_ret);
                    } else {
                        self.patch(else_branch);
                    }
                } else {
                    error!(self, self.ast.expr_source_position(cond_expr), "if statement requires a boolean condition");
                }
            }
            StmtData::For(pre_stmt, cond_expr, post_stmt, body) => {
                let break_label = self.push_break_label();
                let continue_label = self.push_continue_label();
                let mark = self.locals.push_scope();
                if let Some(stmt) = pre_stmt {
                    self.stmt(ctx, stmt);
                }
                let pre_cond = self.expr_in_register(ctx, cond_expr);
                if pre_cond.ty == Type::Bool {
                    self.put_jump_zero(pre_cond.reg, break_label);
                    let loop_entry = self.label_here();
                    return_type = self.stmts(ctx, body);
                    self.patch(continue_label);
                    if let Some(stmt) = post_stmt {
                        self.stmt(ctx, stmt);
                    }
                    let post_cond = self.expr_in_register(ctx, cond_expr);
                    self.put_jump_nonzero(post_cond.reg, loop_entry);
                } else {
                    error!(self, self.ast.expr_source_position(cond_expr), "for statement requires a boolean condition");
                }
                self.locals.restore_scope(mark);
                self.patch(break_label);
                self.pop_break_label(break_label);
                self.pop_continue_label(continue_label);
            }
            StmtData::While(cond_expr, body) => {
                let break_label = self.push_break_label();
                let continue_label = self.push_continue_label();
                let pre_cond = self.expr_in_register(ctx, cond_expr);
                if pre_cond.ty == Type::Bool {
                    self.put_jump_zero(pre_cond.reg, break_label);
                    let loop_entry = self.label_here();
                    return_type = self.stmts(ctx, body);
                    self.patch(continue_label);
                    let post_cond = self.expr_in_register(ctx, cond_expr);
                    self.put_jump_nonzero(post_cond.reg, loop_entry);
                } else {
                    error!(self, self.ast.expr_source_position(cond_expr), "while statement requires a boolean condition");
                }
                self.patch(break_label);
                self.pop_break_label(break_label);
                self.pop_continue_label(continue_label);
            }
            StmtData::Switch(control_expr, cases) => {
                // Totally naive switch implementation. Break by default,
                // fall-through unimplemented.
                let end = self.push_break_label();
                let mut labels = SmallVec::new();
                let control = self.expr_in_register(ctx, control_expr);
                for case in cases {
                    let block_label = self.label();
                    labels.push(block_label);
                    if let SwitchCaseData::Cases(exprs, _) = self.ast.switch_case_data(case) {
                        for case_expr in exprs {
                            let expr = self.expr_in_register(ctx, case_expr);
                            if let Some(_ev) = expr.value {
                                if let Some((op, ty)) = binary_op(parse::TokenKind::Eq, control.ty, expr.ty) {
                                    debug_assert_eq!(ty, Type::Bool);
                                    let matched = self.put_inc(op, (control.reg, expr.reg).into());
                                    self.put_jump_nonzero(matched, block_label);
                                } else {
                                    error!(self, self.ast.expr_source_position(case_expr), "type mismatch between switch control ({}) and case ({})", ctx.type_str(control.ty), ctx.type_str(expr.ty))
                                }
                            } else {
                                error!(self, self.ast.expr_source_position(case_expr), "non-constant switch case");
                            }
                        }
                    }
                }
                let mut first_block = true;
                for (i, case) in cases.enumerate() {
                    let block = match self.ast.switch_case_data(case) {
                        SwitchCaseData::Cases(_, block) => block,
                        SwitchCaseData::Else(block) => block
                    };
                    let block_label = labels[i];
                    self.patch(block_label);
                    let ret = self.stmts(ctx, block);
                    if first_block {
                        return_type = ret;
                        first_block = false;
                    } else {
                        return_type = return_type.and(ret);
                    }
                    self.put_jump(end);
                }
                self.patch(end);
                self.pop_break_label(end);
            }
            StmtData::Do(cond_expr, body) => {
                let break_label = self.push_break_label();
                let continue_label = self.push_continue_label();
                let loop_entry = self.label_here();
                return_type = self.stmts(ctx, body);
                self.patch(continue_label);
                let post_cond = self.expr_in_register(ctx, cond_expr);
                if post_cond.ty != Type::Bool {
                    error!(self, self.ast.expr_source_position(cond_expr), "do statement requires a boolean condition");
                }
                self.put_jump_nonzero(post_cond.reg, loop_entry);
                self.patch(break_label);
                self.pop_break_label(break_label);
                self.pop_continue_label(continue_label);
            }
            StmtData::Expr(expr) => {
                if let ExprData::Cast(inner, _ty) = self.ast.expr_data(expr) {
                    if let ExprData::Name(_name) = self.ast.expr_data(inner) {
                        // This looks like an attempt to declare an uninitialised value,
                        //      name: ty;
                        // which will fail with a confusing error message if `name` is not declared,
                        // and be a noop otherwise. We disallow the latter case because it's confusing
                        // and provide a better error message for the former.
                        error!(self, self.ast.expr_source_position(expr), "cannot declare a value without initialising it");
                    }
                }
                self.expr(ctx, expr);
            }
            StmtData::VarDecl(ty_expr, left, right) => {
                if let ExprData::Name(var) = self.ast.expr_data(left) {
                    let expr = self.expr_in_register(ctx, right);
                    let ty = match ty_expr {
                        TypeExpr::Infer => expr.ty,
                        _ => self.eval_type(ctx, ty_expr)
                    };
                    if integer_promote(ty) == expr.ty {
                        if value_fits(expr.value, ty) {
                            self.locals.insert(var, Local::new(expr.reg, ty));
                        } else {
                            error!(self, self.ast.expr_source_position(left), "constant expression does not fit in {}", ctx.type_str(ty));
                        }
                    } else {
                        error!(self, self.ast.expr_source_position(left), "type mismatch between declaration ({}) and value ({})", ctx.type_str(ty), ctx.type_str(expr.ty))
                    }
                } else {
                    error!(self, self.ast.expr_source_position(left), "cannot declare '{}' as a variable", self.ast.expr_data(left));
                }
            }
            StmtData::Assign(left, right) => {
                if let ExprData::Name(var) = self.ast.expr_data(left) {
                    let expr = self.expr_in_register(ctx, right);
                    if let Some(local) = self.locals.get(var) {
                        let reg = local.reg;
                        self.put(Op::Move, reg, expr.reg.into());
                    } else {
                        error!(self, self.ast.expr_source_position(left), "variable '{}' not found", ctx.str(var));
                    }
                } else {
                    todo!("resolve to memory address");
                }
            }
        }
        return_type
    }

    fn stmts(&mut self, ctx: &Compiler, stmts: StmtList) -> Option<Type> {
        let mark = self.locals.push_scope();
        let mut return_type = None;
        for stmt in stmts {
            if ctx.have_error() {
                break;
            }
            return_type = self.stmt(ctx, stmt).or(return_type);
        }
        self.locals.restore_scope(mark);
        return_type
    }

    fn func(&mut self, ctx: &Compiler, signature: Type, decl: &CallableDecl) -> Func {
        let sig = ctx.type_info(signature).expect("tried to generate code for function without a signature");
        assert!(self.code.len() == 0);
        assert!(sig.kind == TypeKind::Func);
        assert!(decl.params.len() == sig.items.len() - 1, "the last element of a func's type signature's items must be the return type. it is not optional");
        self.return_type = *sig.items.last().unwrap();
        self.reg_counter = 1 - i32::try_from(sig.items.len()).expect("tried to generate code for a function with an excessive number of arguments");
        let param_names = self.ast.items(decl.params).iter().map(|item| item.name);
        let param_types = sig.items.iter().copied();
        for (name, ty) in Iterator::zip(param_names, param_types) {
            let reg = self.inc_reg();
            self.locals.insert(name, Local::new(reg, ty));
        }
        let return_register = self.inc_reg();
        assert_eq!(return_register, 0);
        let returned = self.stmts(ctx, decl.body);
        if matches!(returned, None) {
            error!(self, decl.pos, "func {}: not all control paths return a value", ctx.str(decl.name))
        }
        if self.error.is_none() {
            self.apply_patches();
        } else {
            self.patches.clear();
        }
        assert!(self.patches.len() == 0);
        assert!(self.break_stack.len() == 0);
        assert!(self.continue_stack.len() == 0);
        Func { _signature: signature, code: std::mem::take(&mut self.code) }
    }
}

fn position_to_line_column(str: &str, pos: usize) -> (usize, usize) {
    let start_of_line = str[..pos].bytes().rposition(|c| c == b'\n').unwrap_or(0);
    let line = str[..start_of_line].lines().count() + 1;
    let col = str[start_of_line..pos].chars().count() + 1;
    (line, col)
}

pub struct IncompleteError {
    source_position: usize,
    msg: String,
}

#[derive(Debug)]
pub struct Error {
    source_position: usize,
    msg: String
}

impl Error {
    fn new(err: IncompleteError, source: &str) -> Error {
        let (line, col) = position_to_line_column(source, err.source_position);
        Error {
            source_position: err.source_position,
            msg: format!("{}:{}: error: {}", line, col, err.msg),
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.msg.fmt(f)
    }
}

impl std::error::Error for Error {}

pub struct Compiler {
    interns: Interns,
    types: Vec<TypeInfo>,
    type_by_name: HashMap<Intern, Type>,
    type_by_hash: HashMap<u64, SmallVecN<Type, 5>>,
    globals: HashMap<Intern, Global>,
    funcs: HashMap<Intern, Func>,
    error: Option<IncompleteError>,
    entry_stub: [FatInstr; 2]
}

impl Compiler {
    fn new() -> Compiler {
        let mut compiler = Compiler {
            interns: new_interns_with_keywords(),
            types: Vec::new(),
            type_by_name: HashMap::new(),
            type_by_hash: HashMap::new(),
            globals: HashMap::new(),
            funcs: HashMap::new(),
            error: None,
            entry_stub: [FatInstr::HALT; 2]
        };

        let builtins = [
            ("",     TypeKind::None),
            ("int",  TypeKind::Int),
            ("i8",   TypeKind::I8),
            ("i16",  TypeKind::I16),
            ("i32",  TypeKind::I32),
            ("i64",  TypeKind::I64),
            ("u8",   TypeKind::U8),
            ("u16",  TypeKind::U16),
            ("u32",  TypeKind::U32),
            ("u64",  TypeKind::U64),
            ("f32",  TypeKind::F32),
            ("f64",  TypeKind::F64),
            ("bool", TypeKind::Bool),
        ];

        for (i, &(str, kind)) in builtins.iter().enumerate() {
            assert!(i == kind as usize);
            let name = compiler.interns.put(str);
            let items = Vec::new();
            compiler.types.push(TypeInfo { kind, name, items });
            compiler.type_by_name.insert(name, Type(i as u32));
        }

        compiler
    }

    fn intern(&mut self, str: &str) -> Intern {
        self.interns.put(str)
    }

    fn str(&self, intern: Intern) -> &str {
        self.interns.to_str(intern).expect("Missing intern")
    }

    fn type_str(&self, ty: Type) -> &str {
        self.str(self.types[ty.0 as usize].name)
    }

    fn lookup_type(&self, intern: Intern) -> Option<Type> {
        self.type_by_name.get(&intern).copied()
    }

    fn type_info(&self, ty: Type) -> Option<&TypeInfo> {
        self.types.get(ty.0 as usize)
    }

    fn push_anonymous_type(&mut self, kind: TypeKind, items: &[Type]) -> Type {
        let idx = u32::try_from(self.types.len()).expect("Program too big!!");
        self.types.push(TypeInfo { kind: kind, name: Intern(0), items: items.into() });
        Type(idx)
    }

    fn error(&mut self, source_position: usize, msg: String) {
        if self.error.is_none() {
            self.error = Some(IncompleteError { source_position, msg });
        }
    }

    fn have_error(&self) -> bool {
        self.error.is_some()
    }

    fn check_and_clear_error(&mut self, source: &str) -> Result<()> {
        match self.error.take() {
            None => Ok(()),
            Some(err) => Err(Error::new(err, source).into())
        }
    }

    fn eval_type(&mut self, expr: TypeExpr) -> Type {
        eval_type(self, expr).unwrap_or_else(|err| {
            self.error(err.source_position, err.msg);
            Type(0)
        })
    }

    fn run(&self) -> Value {
        let mut code = &self.entry_stub[..];
        let mut sp: usize = 0;
        let mut ip: usize = 0;
        let mut stack = [RegValue { int64: 0 }; 1024];
        let mut call_stack = vec![(code, ip, sp, 0)];
        loop {
            let instr = &code[ip];
            let dest = sp.wrapping_add(sign_extend(instr.dest));
            let left = sp.wrapping_add(sign_extend(instr.left));
            let right = sp.wrapping_add(sign_extend(instr.right));

            unsafe {
                match instr.op {
                    Op::Halt       => { break; }
                    Op::Noop       => {}
                    Op::Immediate  => { stack[dest] = RegValue::from((instr.left, instr.right)); }
                    Op::IntNeg     => { stack[dest].sint = -stack[left].sint; }
                    Op::IntAdd     => { stack[dest].wint = stack[left].wint + stack[right].wint; }
                    Op::IntSub     => { stack[dest].wint = stack[left].wint - stack[right].wint; }
                    Op::IntMul     => { stack[dest].wint = stack[left].wint * stack[right].wint; }
                    Op::IntDiv     => { stack[dest].wint = stack[left].wint / stack[right].wint; }
                    Op::IntMod     => { stack[dest].wint = stack[left].wint % stack[right].wint; }
                    Op::IntLt      => { stack[dest].int = (stack[left].int < stack[right].int) as usize; }
                    Op::IntGt      => { stack[dest].int = (stack[left].int > stack[right].int) as usize; }
                    Op::IntEq      => { stack[dest].int = (stack[left].int == stack[right].int) as usize; }
                    Op::IntNEq     => { stack[dest].int = (stack[left].int != stack[right].int) as usize; }
                    Op::IntLtEq    => { stack[dest].int = (stack[left].int <= stack[right].int) as usize; }
                    Op::IntGtEq    => { stack[dest].int = (stack[left].int >= stack[right].int) as usize; }
                    Op::BitNeg     => { stack[dest].int = !stack[left].int }
                    Op::BitAnd     => { stack[dest].int = stack[left].int & stack[right].int; }
                    Op::BitOr      => { stack[dest].int = stack[left].int | stack[right].int; }
                    Op::BitXor     => { stack[dest].int = stack[left].int ^ stack[right].int; }

                    Op::LShift     => { stack[dest].int = stack[left].int << stack[right].int; }
                    Op::RShift     => { stack[dest].int = stack[left].int >> stack[right].int; }

                    Op::Not        => { stack[dest].int = (stack[left].int == 0) as usize; }
                    Op::CmpZero    => { stack[dest].int = (stack[left].int != 0) as usize; }
                    Op::LogicOr    => { stack[dest].int = (stack[left].b8.0 || stack[right].b8.0) as usize; }
                    Op::LogicAnd   => { stack[dest].int = (stack[left].b8.0 && stack[right].b8.0) as usize; }

                    Op::F32Neg     => { stack[dest].float32.0 = -stack[left].float32.0; }
                    Op::F32Add     => { stack[dest].float32.0 = stack[left].float32.0 + stack[right].float32.0; }
                    Op::F32Sub     => { stack[dest].float32.0 = stack[left].float32.0 - stack[right].float32.0; }
                    Op::F32Mul     => { stack[dest].float32.0 = stack[left].float32.0 * stack[right].float32.0; }
                    Op::F32Div     => { stack[dest].float32.0 = stack[left].float32.0 / stack[right].float32.0; }
                    Op::F32Lt      => { stack[dest].int = (stack[left].float32.0 < stack[right].float32.0) as usize; }
                    Op::F32Gt      => { stack[dest].int = (stack[left].float32.0 > stack[right].float32.0) as usize; }
                    Op::F32Eq      => { stack[dest].int = (stack[left].float32.0 == stack[right].float32.0) as usize; }
                    Op::F32NEq     => { stack[dest].int = (stack[left].float32.0 != stack[right].float32.0) as usize; }
                    Op::F32LtEq    => { stack[dest].int = (stack[left].float32.0 <= stack[right].float32.0) as usize; }
                    Op::F32GtEq    => { stack[dest].int = (stack[left].float32.0 >= stack[right].float32.0) as usize; }

                    Op::F64Neg     => { stack[dest].float64 = -stack[left].float64; }
                    Op::F64Add     => { stack[dest].float64 = stack[left].float64 + stack[right].float64; }
                    Op::F64Sub     => { stack[dest].float64 = stack[left].float64 - stack[right].float64; }
                    Op::F64Mul     => { stack[dest].float64 = stack[left].float64 * stack[right].float64; }
                    Op::F64Div     => { stack[dest].float64 = stack[left].float64 / stack[right].float64; }
                    Op::F64Lt      => { stack[dest].int = (stack[left].float64 < stack[right].float64) as usize; }
                    Op::F64Gt      => { stack[dest].int = (stack[left].float64 > stack[right].float64) as usize; }
                    Op::F64Eq      => { stack[dest].int = (stack[left].float64 == stack[right].float64) as usize; }
                    Op::F64NEq     => { stack[dest].int = (stack[left].float64 != stack[right].float64) as usize; }
                    Op::F64LtEq    => { stack[dest].int = (stack[left].float64 <= stack[right].float64) as usize; }
                    Op::F64GtEq    => { stack[dest].int = (stack[left].float64 >= stack[right].float64) as usize; }

                    Op::MoveLower8               => { stack[dest].int       = stack[left].int8.0    as usize; }
                    Op::MoveLower16              => { stack[dest].int       = stack[left].int16.0   as usize; }
                    Op::MoveLower32              => { stack[dest].int       = stack[left].int32.0   as usize; }
                    Op::MoveAndSignExtendLower8  => { stack[dest].int       = stack[left].sint8.0   as usize; }
                    Op::MoveAndSignExtendLower16 => { stack[dest].int       = stack[left].sint16.0  as usize; }
                    Op::MoveAndSignExtendLower32 => { stack[dest].int       = stack[left].sint32.0  as usize; }
                    Op::IntToFloat32             => { stack[dest].float32.0 = stack[left].sint      as f32; }
                    Op::IntToFloat64             => { stack[dest].float64   = stack[left].sint      as f64; }
                    Op::Float32ToInt             => { stack[dest].int       = stack[left].float32.0 as usize; }
                    Op::Float32To64              => { stack[dest].float64   = stack[left].float32.0 as f64; }
                    Op::Float64ToInt             => { stack[dest].int       = stack[left].float64   as usize; }
                    Op::Float64To32              => { stack[dest].float32.0 = stack[left].float64   as f32; }

                    Op::Move          => { stack[dest] = stack[left]; }
                    Op::Jump          => { ip = ip.wrapping_add(sign_extend(instr.left)); }
                    Op::JumpIfZero    => { if stack[dest].int == 0 { ip = ip.wrapping_add(sign_extend(instr.left)); } }
                    Op::JumpIfNotZero => { if stack[dest].int != 0 { ip = ip.wrapping_add(sign_extend(instr.left)); } }
                    Op::Return        => {
                        let ret = call_stack.pop().expect("Bad bytecode");
                        code = ret.0;
                        ip = ret.1;
                        sp = ret.2;
                        stack[ret.3] = stack[left];
                    },
                    Op::Call          => {
                        call_stack.push((code, ip, sp, dest));
                        let f = self.funcs.get(&Intern(instr.left as u32)).expect("Bad bytecode");
                        code = &f.code[..];
                        ip = !0;
                        sp = dest;
                    },
                    Op::CallIndirect  => { todo!() },
                }
            }
            ip = ip.wrapping_add(1);
        }
        Value::from(stack[0 as usize], Type::Int)
    }
}

#[derive(Default)]
struct Func {
    _signature: Type,
    code: Vec<FatInstr>
}

fn resolve_callable_type(ctx: &mut Compiler, ast: &Ast, decl: &CallableDecl) -> Type {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut decl_types = SmallVec::new();
    for item in ast.items(decl.params) {
        let ty = ctx.eval_type(item.expr);
        decl_types.push(ty);
    }
    if let Some(returns) = decl.returns {
        let return_type = ctx.eval_type(returns);
        decl_types.push(return_type);
    }
    let hash = { let mut h = DefaultHasher::new(); Hash::hash_slice(&decl_types, &mut h); h.finish() };
    if let Some(types) = ctx.type_by_hash.get(&hash) {
        for &ty in types.iter() {
            let info = ctx.type_info(ty).unwrap();
            if info.items.iter().eq(decl_types.iter()) {
                return ty;
            }
        }
    }
    let ty = ctx.push_anonymous_type(TypeKind::Func, &decl_types);
    ctx.type_by_hash.entry(hash).or_insert_with(|| SmallVecN::new()).push(ty);
    ty
}

fn compile(str: &str) -> Result<Compiler> {
    let mut c = Compiler::new();
    let ast = parse::parse(&mut c, str);
    let main = c.intern("main");
    for decl in ast.decls() {
        use std::collections::hash_map::Entry;
        let sig = resolve_callable_type(&mut c, &ast, decl);
        match c.globals.entry(decl.name) {
            // TODO: location of other definition
            Entry::Occupied(_) => { error!(c, decl.pos, "{} has already been defined", c.str(decl.name)); }
            Entry::Vacant(entry) => { entry.insert(Global { value: Value::Ident(decl.name), ty: sig }); }
        }
        if decl.name == main {
            match c.type_info(sig) {
                Some(info) => if !matches!(info.items[..], [Type::Int]) {
                    error!(c, decl.pos, "main must take zero arguments and return int");
                }
                _ => ()
            }
        }
    }

    c.check_and_clear_error(str)?;

    let mut gen = FatGen::new(&ast);
    for decl in ast.decls() {
        let sig = c.globals.get(&decl.name).unwrap().ty;
        c.funcs.insert(decl.name, gen.func(&c, sig, &decl));
    }

    // :P
    c.error = gen.error.take();
    c.check_and_clear_error(str)?;

    if let Some(_func) = c.funcs.get(&main) {
        c.entry_stub[0] = FatInstr { op: Op::Call, dest: 0, left: main.0, right: 0 };
    }
    Ok(c)
}

fn compile_and_run(str: &str) -> Result<Value> {
    let c = compile(str)?;
    Ok(c.run())
}

fn sign_extend(val: u32) -> usize {
    val as i32 as usize
}

fn eval_expr(str: &str) -> Result<Value> {
    compile_and_run(&format!("func main(): int {{ return {}; }}", str))
}

fn repl() {
    use std::io;
    let mut line = String::new();
    loop {
        match io::stdin().read_line(&mut line) {
            Ok(_) => match eval_expr(&line) {
               Ok(value) => println!("{}", value),
               Err(err) => println!("{}", err)
            },
            Err(_) => break
        }

        line.clear();
    }
}

fn main() {
    repl();
}

#[test]
fn expr() {
    macro_rules! eval {
        ($x: expr) => {
            let result = eval_expr(stringify!($x)).unwrap();
            assert_eq!(result, Value::Int($x), stringify!($x))
        };
        ($x: literal, $val: expr) => {
            let result = eval_expr($x).unwrap();
            assert_eq!(result, Value::Int($val), stringify!($x))
        };
    }
    eval!(1);
    eval!(1+1);
    eval!(1-1);
    eval!(-1-1);
    eval!(-1+1);
    eval!(-1+-1);
    eval!(-1);
    eval!(--1);
    eval!(---1);
    eval!(1 << 2);
    eval!(16 >> 3);
    eval!(24 / 4);
    eval!(3 * 3);
    eval!(2*3+2);
    eval!(2+3*2);
    eval!(1 * -9);
    eval!(2 | 4);
    eval!(9 & 7);
    eval!(9 & 7 | 3);
    eval!(84 ^ 23);
    eval!(42 % 23);
    eval!(1+2+3*4/4+(2*-4+2)*3-1);
    // << and >> have same associativity as * and /
    eval!("1 + 2 << 3 + 4",      1 + (2 << 3) + 4);
    eval!("1 * 2 << 3 * 4",      (1 * 2 << 3) * 4);
    eval!("(3 + 4) >> (1 + 2)",  3 + 4 >> 1 + 2);
    eval!("(3 * 4) >> (1 * 2)",  3 * 4 >> 1 * 2);
    eval!("!165: int", 0);
    eval!("!0: int", 1);
    eval!("~23", !23);
    eval!("(!!(2*3) || !!(3-3)) : int", 1);
    eval!("(!!(2*3) && !!(3-3)) : int", 0);
    eval!("256: i8", 0);
    eval!("-1: u8: int", 255);
    eval!("-1: i8: int", -1);
    eval!("1 == 1 ? 2 :: 3", 2);
    eval!("0 == 1 ? 2 :: 3", 3);
    eval!("(43 * 43) > 0 ? (1+1+1) :: (2 << 3)", 3);
    eval!("!!0 ? (!!1 ? 2 :: 3) :: (!!4 ? 5 :: 6)", 5);
    eval!("!!0 ?  !!1 ? 2 :: 3  ::  !!4 ? 5 :: 6",  5);
    eval!("!!1 ? (!!2 ? 3 :: 4) :: (!!5 ? 6 :: 7)", 3);
    eval!("!!1 ?  !!2 ? 3 :: 4  ::  !!5 ? 6 :: 7",  3);
}

#[test]
fn stmt() {
    fn compile(c: &mut Compiler, str: &str) -> Result<()> {
        let (ast, stmt) = parse::parse_stmt(c, str);
        c.check_and_clear_error(str)?;
        let mut fg = FatGen::new(&ast);
        fg.stmts(c, StmtList::from(stmt));
        c.error = fg.error;
        c.check_and_clear_error(str)
    }
    let c = &mut Compiler::new();
    let ok = [
        "{ a := 1; do { if (a > 2) { a = (a & 1) == 1 ? a * 2 :: a + 1; } else { a = a + 1; } } while (a < 10); }",
        "{ a := 1.0; a = a + 1.0d:f32; }",
        "{ a := !165: int; }",
        "{ (!!(2*3)) || !!(3-3); }",
        "{ a := (0: bool); }",
        "{ a := 0; a := 1; }",
        "{ if (1:bool) {} }",
        "{ while (1:bool) {} }",
        "{ do {} while (1:bool); }",
    ];
    let err = [
        "{ a := -(0: bool); }",
        "{ (); }"
    ];
    for str in ok.iter() {
        compile(c, str).map_err(|e| { println!("input: \"{}\"", str); e }).unwrap();
    }
    for str in err.iter() {
        // Not checking it's the right error yet--maybe later
        compile(c, str).map(|v| { println!("input: \"{}\"", str); v }).unwrap_err();
    }
}

#[test]
fn decls() {
    let ok = [
        "func add(a: int, b: int): int { return a + b; }",
        "func add'(a: int, b: int): int { return a + b; }",
    ];
    let err = [
        "func func(a: int, b: int): int { return a + b; }",
        "func add(a: int, b: int): int { return a + b; } func add(a: int, b: int): int { return a + b; }",
    ];
    for str in ok.iter() {
        compile_and_run(str).map_err(|e| { println!("input: \"{}\"", str); e }).unwrap();
    }
    for str in err.iter() {
        // Not checking it's the right error yet--maybe later
        compile_and_run(str).map(|v| { println!("input: \"{}\"", str); v }).unwrap_err();
    }
}

#[test]
fn control_paths() {
    let ok = [
        "func control_paths(a: int, b: int, c: bool): int { if (c) { return a + b; } else { return a - b; } }",
        "func control_paths(a: int, b: int, c: bool): int { while (c) { return a; } }",
        "func control_paths(a: int, b: int, c: bool): int { return a; while (c) { if (c) { return a; } } }",
    ];
    let err = [
        "func control_paths(a: int) {}",
        "func control_paths(a: int, b: int, c: bool): int { if (c) { return a + b; } }",
        "func control_paths(a: int, b: int, c: bool): int { if (c) { return a + b; } else { } }",
        "func control_paths(a: int, b: int, c: bool): int { if (c) {} else { return a - b; } }",
        "func control_paths(a: int, b: int, c: bool): int { while (c) { if (c) { return a; } } }",
    ];
    for str in ok.iter() {
        compile_and_run(str).map_err(|e| { println!("input: \"{}\"", str); e }).unwrap();
    }
    for str in err.iter() {
        // Not checking it's the right error yet--maybe later
        compile_and_run(str).map(|v| { println!("input: \"{}\"", str); v }).unwrap_err();
    }
}

#[test]
fn int_promotion() {
    let ok = [
        "func pro(): int { a: u8 = 1; b: u8 = 2; return a + b; }",
        "func pro(): int { a: u8 = 1; b: u16 = 2; return a + b; }",
        "func pro(): int { a: u8 = 1; b: i16 = 2; return a + b; }",
    ];
    let err = [
        "func pro(): u8 { a: u8 = 1; b: u8 = 2; return a + b; }",
    ];
    for str in ok.iter() {
        compile_and_run(str).map_err(|e| { println!("input: \"{}\"", str); e }).unwrap();
    }
    for str in err.iter() {
        // Not checking it's the right error yet--maybe later
        compile_and_run(str).map(|v| { println!("input: \"{}\"", str); v }).unwrap_err();
    }
}

#[test]
fn fits() {
    let err = [
        "func fits(): int { a: u8 = -1; return a; }",
        "func fits(): int { a: i8 = 128; return a; }",
        "func fits(): int { a: u8 = 256; return a; }",
        "func fits(): int { a: i16 = 32768; return a; }",
        "func fits(): int { a: u16 = 65536; return a; }",
        "func fits(): int { a: i32 = 2147483648; return a; }",
        "func fits(): int { a: u32 = 4294967296; return a; }",
        "func fits(): int { a := 18446744073709551616; return a; }",
    ];
    for str in err.iter() {
        // Not checking it's the right error yet--maybe later
        compile_and_run(str).map(|v| { println!("input: \"{}\"", str); v }).unwrap_err();
    }
}

#[test]
fn func() {
    let ok = [
(r#"
func madd(a: int, b: int, c: int): int {
    return a * b + c;
}

func main(): int {
    return madd(1, 2, 3);
}
"#, Value::Int(5)),
(r#"
func fib(n: int): int {
    if (n <= 1) {
        return n;
    }

    return fib(n - 2) + fib(n - 1);
}

func main(): int {
    return fib(6);
}
"#, Value::Int(8)),
(r#"
func even(n: int): bool {
    return n == 0 ? 1:bool :: odd(n - 1);
}

func odd(n: int): bool {
    return n == 0 ? 0:bool :: even(n - 1);
}

func main(): int {
    return even(10):int;
}
"#, Value::Int(1))
    ];
    for test in ok.iter() {
        let str = test.0;
        let expect = test.1;
        let ret = compile_and_run(str).map_err(|e| { println!("input: \"{}\"", str); e }).unwrap();
        assert_eq!(expect, ret);
    }
}

#[test]
fn control_structures() {
    let ok = [
(r#"
func main(): int {
    a := 0;
    for (i := 0; i < 10; i = i + 1) {
        a = a + 1;
    }
    return a;
}
"#, Value::Int(10)),
(r#"
func main(): int {
    a := 0;
    for (; a < 10; a = a + 2) {}
    return a;
}
"#, Value::Int(10)),
(r#"
func main(): int {
    a := 0;
    while (1:bool) {
        a = a + 1;
        if (a == 10) {
            break;
        }
        continue;
    }
    return a;
}
"#, Value::Int(10)),
(r#"
func main(): int {
    a := 0;
    b := 0;
    for (; a < 10; a = a + 1) {
        if (a > 5) {
            continue;
        }
        b = a;
    }
    return a + b;
}
"#, Value::Int(15)),
(r#"
func main(): int {
    a := 10;
    switch (a) {
        1 {}
        2, 3 {
            return 1;
        }
        10 {
            return 100;
        }
    }
    return 0;
}
"#, Value::Int(100)),
(r#"
func main(): int {
    a := 3;
    b := 0;
    switch (a) {
        1 {
            b = 0;
        }
        2, 3 {
            b = 1;
        }
        10 {
            b = 100;
        }
    }
    return b;
}
"#, Value::Int(1)),
(r#"func main(): int {
    a := 1;
    do {
        if (a > 2) {
            a = (a & 1) == 1 ? a * 2 :: a + 1;
        } else {
            a = a + 1;
        }
    } while (a < 10);
    return a;
}"#, Value::Int(14))
    ];
    for test in ok.iter() {
        let str = test.0;
        let expect = test.1;
        let ret = compile_and_run(str).map_err(|e| { println!("input: \"{}\"", str); e }).unwrap();
        assert_eq!(expect, ret);
    }
}
