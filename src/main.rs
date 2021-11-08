#![cfg_attr(feature = "bench", feature(test))]
#[cfg(not(target_pointer_width = "64"))]
compile_error!("64 bit is assumed");

use std::collections::hash_map::{HashMap, DefaultHasher};
use std::hash::{Hash, Hasher};
use std::fmt::Display;

mod parse;
mod ast;
mod sym;
mod types;
mod gen;
mod smallvec;

use ast::*;
use sym::*;
use types::*;
use gen::*;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn hash<T: Hash>(value: &T) -> u64 {
    let mut h = DefaultHasher::new();
    value.hash(&mut h);
    h.finish()
}

pub fn align_up(value: usize, alignment: usize) -> usize {
    assert!(alignment.is_power_of_two());
    let result = value.wrapping_add(alignment - 1) & !(alignment - 1);
    debug_assert_eq!((result) & (alignment - 1), 0);
    result
}

#[macro_export]
macro_rules! error {
    ($ctx: expr, $pos: expr, $($fmt: expr),*) => {{
        $ctx.error($pos, format!($($fmt),*))
    }}
}

#[macro_export]
macro_rules! assert_implies {
    ($p:expr, $q:expr) => { assert!(!($p) || ($q)) }
}

#[macro_export]
macro_rules! debug_assert_implies {
    ($p:expr, $q:expr) => { debug_assert!(!($p) || ($q)) }
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
    #[cfg(all(test, not(feature = "bench")))]
    const BUF_CAPACITY: usize = 12;
    #[cfg(any(not(test), feature = "bench"))]
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
        // It would be nice to do something that as realloc even in
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

// Clean up in test code so its not a leak false-positive
#[cfg(test)]
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
    let mut i = 1;
    for k in (1..).scan((), |(), v| parse::Keyword::from_u32(v)) {
        result.put(k.to_str());
        i += 1;
    }
    for k in (i..).scan((), |(), v| parse::Keytype::from_u32(v)) {
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
    for k in (i.0..).scan((), |(), v| parse::Keytype::from_u32(v)) {
        assert_eq!(interns.put(k.to_str()), i);
        i.0 += 1;
    }
}

fn position_to_line_column(str: &str, pos: usize) -> (usize, usize) {
    let start_of_line = str[..pos].bytes().rposition(|c| c == b'\n').unwrap_or(0);
    let line = str[..start_of_line].lines().count() + 1;
    let col = str[start_of_line..pos].chars().count() + 1;
    (line, col)
}

#[derive(Debug)]
pub struct Error {
    source_position: usize,
    msg: String,
}

#[derive(Debug, Default)]
pub struct Code {
    #[allow(dead_code)]
    signature: Type,
    addr: usize
}

struct TypeStrFormatter<'a> {
    ctx: &'a Compiler,
    ty: Type,
    callable_name: Intern,
}

impl Display for TypeStrFormatter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        let base = self.ctx.types.base_type(self.ty);
        let info = self.ctx.types.info(self.ty);
        let have_name = info.name != Intern(0) || self.callable_name != Intern(0);
        let base_type = self.ctx.types.base_type(self.ty);
        let needs_parens = self.ctx.types.base_type(base_type) != base_type;
        match info.kind {
            // Todo: string of func signature, anonymous structs
            TypeKind::Callable if have_name   => write!(f, "{} {}", if info.mutable { "proc" } else { "func" }, self.ctx.str(self.callable_name)),
            TypeKind::Callable                => write!(f, "{}", if info.mutable { "proc" } else { "func" }),
            TypeKind::Struct if have_name     => write!(f, "{}", self.ctx.str(info.name)),
            TypeKind::Struct                  => write!(f, "anonymous struct"),
            TypeKind::Array if needs_parens   => write!(f, "arr ({}) [{}]", self.ctx.type_str(base), info.num_array_elements),
            TypeKind::Array                   => write!(f, "arr {} [{}]", self.ctx.type_str(base), info.num_array_elements),
            TypeKind::Pointer if needs_parens => write!(f, "ptr ({})", self.ctx.type_str(base)),
            TypeKind::Pointer                 => write!(f, "ptr {}", self.ctx.type_str(base)),
            _ => self.ctx.str(info.name).fmt(f)
        }
    }
}

pub struct Compiler {
    interns: Interns,
    types: Types,
    symbols: Symbols,
    funcs: HashMap<Intern, Code>,
    errors: Vec<Error>,
    ast: Ast,
    code: Vec<FatInstr>,
    data: Vec<u8>,
    fgs: Vec<FatGen>,
    ip_start: usize,
}

impl Compiler {
    fn new() -> Compiler {
        let mut result = Compiler {
            interns: new_interns_with_keywords(),
            types: Types::new(),
            symbols: Default::default(),
            funcs: HashMap::new(),
            errors: Vec::new(),
            ast: Ast::new(),
            code: Vec::new(),
            data: Vec::new(),
            fgs: Vec::new(),
            ip_start: 0,
        };
        types::builtins(&mut result);
        result
    }

    fn intern(&mut self, str: &str) -> Intern {
        self.interns.put(str)
    }

    fn str(&self, intern: Intern) -> &str {
        self.interns.to_str(intern).expect("Missing intern")
    }

    fn callable_str(&self, name: Intern) -> TypeStrFormatter {
        TypeStrFormatter { ctx: self, ty: sym::lookup_value(self, name).map(|sym| sym.ty).unwrap_or(Type::None), callable_name: name }
    }

    fn type_str(&self, ty: Type) -> TypeStrFormatter {
        TypeStrFormatter { ctx: self, ty, callable_name: Intern(0) }
    }

    fn error(&mut self, source_position: usize, msg: String) {
        self.errors.push(Error { source_position, msg });
    }

    fn have_error(&self) -> bool {
        !self.errors.is_empty()
    }

    fn report_errors(&mut self, source: &str) -> Result<()> {
        if self.errors.is_empty() {
            return Ok(());
        }

        for err in &self.errors {
            let (line, col) = position_to_line_column(source, err.source_position);
            println!("{}:{}: error: {}", line, col, err.msg);
        }

        Err(format!("{} errors reported.", self.errors.len()).into())
    }

    fn run(&mut self, ip: usize, sp: usize, stack: &mut [u8]) -> Result<Value> {
        let code = &mut self.code[..];
        let mut sp = sp;
        let mut ip = ip;

        let stack = unsafe { stack.align_to_mut::<RegValue>().1 };
        let stack: *mut [u8] = unsafe { std::slice::from_raw_parts_mut(stack.as_mut_ptr() as *mut u8, stack.len() * std::mem::size_of::<RegValue>()) };

        // The VM allows taking (real) pointers to addresses on the VM stack. These are
        // immediately invalidated by taking a & reference to the stack. Solution: never
        // take a reference to the stack (even one immediately turned into a pointer).

        // This just ensures the compiler knows the correct invariants involved, it
        // doesn't make the vm sound

        // Access the stack by byte
        macro_rules! stack {
            ($addr:expr) => { (*stack)[$addr] }
        }

        // Access the stack by register. Register accesses must be aligned correctly (asserted below)
        macro_rules! reg {
            ($addr:expr) => { *(std::ptr::addr_of_mut!(stack![$addr]) as *mut RegValue) }
        }

        unsafe {
            reg![Location::RETURN_ADDRESS_REGISTER as usize].int = ip;
            reg![Location::FP_REGISTER as usize].int = sp;
            reg![Location::SP_REGISTER as usize].int = std::ptr::addr_of_mut!(stack![0]) as usize;
        }

        loop {
            let instr = code[ip];
            let dest = sp.wrapping_add(instr.dest as usize);
            let left = sp.wrapping_add(instr.left as usize);
            let right = sp.wrapping_add(instr.right as usize);

            assert_implies!(requires_register_destination(instr.op), (dest & (FatGen::REGISTER_SIZE - 1)) == 0);

            unsafe {
                reg![sp + Location::IP_REGISTER as usize].int = std::ptr::addr_of_mut!(code[ip]) as usize;

                 match instr.op {
                    Op::Halt       => { break; }
                    Op::Panic      => {
                        // terrible
                        if reg![dest].int == PanicReason::GlobalNotInitialized as usize {
                            // but which one
                            return Err("global value used before initialization".into())
                        } else if reg![dest].int == PanicReason::AssertionFailed as usize {
                            // but where
                            return Err("assertion failed".into())
                        } else if reg![dest].int == PanicReason::IndexOutOfBounds as usize {
                            // :(((((((((((
                            return Err("index is out of bounds".into())
                        } else {
                            unreachable!();
                        }
                    }
                    Op::Noop       => {}
                    Op::Immediate  => { reg![dest] = RegValue::from((instr.left, instr.right)); }
                    Op::IntNeg     => { reg![dest].sint = -reg![left].sint; }
                    Op::IntAdd     => { reg![dest].wint = reg![left].wint + reg![right].wint; }
                    Op::IntSub     => { reg![dest].wint = reg![left].wint - reg![right].wint; }
                    Op::IntMul     => { reg![dest].wint = reg![left].wint * reg![right].wint; }
                    Op::IntDiv     => { reg![dest].wint = reg![left].wint / reg![right].wint; }
                    Op::IntMod     => { reg![dest].wint = reg![left].wint % reg![right].wint; }
                    Op::IntLt      => { reg![dest].int = (reg![left].int < reg![right].int) as usize; }
                    Op::IntGt      => { reg![dest].int = (reg![left].int > reg![right].int) as usize; }
                    Op::IntEq      => { reg![dest].int = (reg![left].int == reg![right].int) as usize; }
                    Op::IntNEq     => { reg![dest].int = (reg![left].int != reg![right].int) as usize; }
                    Op::IntLtEq    => { reg![dest].int = (reg![left].int <= reg![right].int) as usize; }
                    Op::IntGtEq    => { reg![dest].int = (reg![left].int >= reg![right].int) as usize; }
                    Op::BitNeg     => { reg![dest].int = !reg![left].int }
                    Op::BitAnd     => { reg![dest].int = reg![left].int & reg![right].int; }
                    Op::BitOr      => { reg![dest].int = reg![left].int | reg![right].int; }
                    Op::BitXor     => { reg![dest].int = reg![left].int ^ reg![right].int; }

                    Op::LShift     => { reg![dest].int = reg![left].int << reg![right].int; }
                    Op::RShift     => { reg![dest].int = reg![left].int >> reg![right].int; }

                    Op::Not        => { reg![dest].int = (reg![left].int == 0) as usize; }
                    Op::CmpZero    => { reg![dest].int = (reg![left].int != 0) as usize; }
                    Op::LogicOr    => { reg![dest].int = (reg![left].b8.0 || reg![right].b8.0) as usize; }
                    Op::LogicAnd   => { reg![dest].int = (reg![left].b8.0 && reg![right].b8.0) as usize; }

                    Op::F32Neg     => { reg![dest].float32.0 = -reg![left].float32.0;
                                        reg![dest].float32.1 = 0.0; }
                    Op::F32Add     => { reg![dest].float32.0 = reg![left].float32.0 + reg![right].float32.0;
                                        reg![dest].float32.1 = 0.0; }
                    Op::F32Sub     => { reg![dest].float32.0 = reg![left].float32.0 - reg![right].float32.0;
                                        reg![dest].float32.1 = 0.0; }
                    Op::F32Mul     => { reg![dest].float32.0 = reg![left].float32.0 * reg![right].float32.0;
                                        reg![dest].float32.1 = 0.0; }
                    Op::F32Div     => { reg![dest].float32.0 = reg![left].float32.0 / reg![right].float32.0;
                                        reg![dest].float32.1 = 0.0; }
                    Op::F32Lt      => { reg![dest].int = (reg![left].float32.0 < reg![right].float32.0) as usize; }
                    Op::F32Gt      => { reg![dest].int = (reg![left].float32.0 > reg![right].float32.0) as usize; }
                    Op::F32Eq      => { reg![dest].int = (reg![left].float32.0 == reg![right].float32.0) as usize; }
                    Op::F32NEq     => { reg![dest].int = (reg![left].float32.0 != reg![right].float32.0) as usize; }
                    Op::F32LtEq    => { reg![dest].int = (reg![left].float32.0 <= reg![right].float32.0) as usize; }
                    Op::F32GtEq    => { reg![dest].int = (reg![left].float32.0 >= reg![right].float32.0) as usize; }

                    Op::F64Neg     => { reg![dest].float64 = -reg![left].float64; }
                    Op::F64Add     => { reg![dest].float64 = reg![left].float64 + reg![right].float64; }
                    Op::F64Sub     => { reg![dest].float64 = reg![left].float64 - reg![right].float64; }
                    Op::F64Mul     => { reg![dest].float64 = reg![left].float64 * reg![right].float64; }
                    Op::F64Div     => { reg![dest].float64 = reg![left].float64 / reg![right].float64; }
                    Op::F64Lt      => { reg![dest].int = (reg![left].float64 < reg![right].float64) as usize; }
                    Op::F64Gt      => { reg![dest].int = (reg![left].float64 > reg![right].float64) as usize; }
                    Op::F64Eq      => { reg![dest].int = (reg![left].float64 == reg![right].float64) as usize; }
                    Op::F64NEq     => { reg![dest].int = (reg![left].float64 != reg![right].float64) as usize; }
                    Op::F64LtEq    => { reg![dest].int = (reg![left].float64 <= reg![right].float64) as usize; }
                    Op::F64GtEq    => { reg![dest].int = (reg![left].float64 >= reg![right].float64) as usize; }

                    Op::MoveLower8               => { reg![dest].int       = reg![left].int8.0    as usize; }
                    Op::MoveLower16              => { reg![dest].int       = reg![left].int16.0   as usize; }
                    Op::MoveLower32              => { reg![dest].int       = reg![left].int32.0   as usize; }
                    Op::MoveAndSignExtendLower8  => { reg![dest].int       = reg![left].sint8.0   as usize; }
                    Op::MoveAndSignExtendLower16 => { reg![dest].int       = reg![left].sint16.0  as usize; }
                    Op::MoveAndSignExtendLower32 => { reg![dest].int       = reg![left].sint32.0  as usize; }
                    Op::IntToFloat32             => { reg![dest].float32.0 = reg![left].sint      as f32;
                                                      reg![dest].float32.1 = 0.0; }
                    Op::IntToFloat64             => { reg![dest].float64   = reg![left].sint      as f64; }
                    Op::Float32ToInt             => { reg![dest].int       = reg![left].float32.0 as usize; }
                    Op::Float32To64              => { reg![dest].float64   = reg![left].float32.0 as f64; }
                    Op::Float64ToInt             => { reg![dest].int       = reg![left].float64   as usize; }
                    Op::Float64To32              => { reg![dest].float32.0 = reg![left].float64   as f32;
                                                      reg![dest].float32.1 = 0.0; }

                    Op::Move          => { reg![dest] = reg![left]; }
                    Op::Jump          => { ip = ip.wrapping_add(instr.left as usize); }
                    Op::JumpIfZero    => { if reg![dest].int == 0 { ip = ip.wrapping_add(instr.left as usize); } }
                    Op::JumpIfNotZero => { if reg![dest].int != 0 { ip = ip.wrapping_add(instr.left as usize); } }
                    Op::Return        => {
                        ip = reg![sp + Location::RETURN_ADDRESS_REGISTER as usize].int;
                        sp = reg![sp + Location::FP_REGISTER as usize].int;
                    },
                    Op::Call => {
                        reg![dest + Location::RETURN_ADDRESS_REGISTER as usize].int = ip;
                        reg![dest + Location::FP_REGISTER as usize].int = sp;
                        reg![dest + Location::SP_REGISTER as usize].int = std::ptr::addr_of_mut!(stack![dest]) as usize;
                        ip = instr.left as usize;
                        sp = dest;
                    },
                    Op::CallIndirect  => { todo!() },

                    Op::Store8  => { *(reg![dest].int.wrapping_add(instr.right as usize) as *mut u8)  =   stack![left]; },
                    Op::Store16 => { *(reg![dest].int.wrapping_add(instr.right as usize) as *mut u16) = *(stack![left..].as_ptr() as *const u16); },
                    Op::Store32 => { *(reg![dest].int.wrapping_add(instr.right as usize) as *mut u32) = *(stack![left..].as_ptr() as *const u32); },
                    Op::Store64 => { *(reg![dest].int.wrapping_add(instr.right as usize) as *mut u64) = *(stack![left..].as_ptr() as *const u64); },
                    Op::Load8   => { reg![dest].int = *(reg![left].int.wrapping_add(instr.right as usize) as *const u8)  as usize; },
                    Op::Load16  => { reg![dest].int = *(reg![left].int.wrapping_add(instr.right as usize) as *const u16) as usize; },
                    Op::Load32  => { reg![dest].int = *(reg![left].int.wrapping_add(instr.right as usize) as *const u32) as usize; },
                    Op::Load64  => { reg![dest].int = *(reg![left].int.wrapping_add(instr.right as usize) as *const u64) as usize; },
                    Op::LoadAndSignExtend8  => { reg![dest].sint = *(reg![left].int.wrapping_add(instr.right as usize) as *const i8)  as isize; },
                    Op::LoadAndSignExtend16 => { reg![dest].sint = *(reg![left].int.wrapping_add(instr.right as usize) as *const i16) as isize; },
                    Op::LoadAndSignExtend32 => { reg![dest].sint = *(reg![left].int.wrapping_add(instr.right as usize) as *const i32) as isize; },
                    Op::LoadBool => { reg![dest].b8.0 = *(reg![left].int.wrapping_add(instr.right as usize) as *const u8) != 0; },
                    Op::Copy => { std::ptr::copy(reg![left].int as *const u8, reg![dest].int as *mut u8, reg![right].int); },
                    Op::Zero => { for b in std::slice::from_raw_parts_mut(reg![dest].int as *mut u8, reg![left].int) { *b = 0 }},
                }
            }
            ip = ip.wrapping_add(1);
        }
        let result = unsafe { Value::from(reg![0 as usize], Type::Int) };

        Ok(result)
    }
}

fn compile(str: &str) -> Result<Compiler> {
    let mut c = Compiler::new();

    // We can't parse successive files like this in general beacuse the AST assumes all nodes
    // come from the same file. This works for now because nothing ever asks where these come from
    parse::parse(&mut c, r#"
panic: func () int {
    // built-in
}

assert: func (condition: bool) int {
    if (!condition) {
        panic();
    }
    return 0;
}
"#);

    parse::parse(&mut c, str);

    c.report_errors(str)?;

    sym::resolve_decls(&mut c);

    c.report_errors(str)?;

    let mut data = std::mem::take(&mut c.data);

    data.reserve(8*1024*1024);

    let aligned = align_up(data.len(), std::mem::align_of::<FatInstr>());
    data.resize_with(aligned, Default::default);

    // Reinterpret the u8 data vec as a vec of FatInstrs. This is UB if we have
    // to grow the array while generating code because the layout (element size
    // + alignment) of the underlying allocation differs, which doesn't seem to
    // actually matter but whatever. This is just the easiest way to get a
    // continous allocation for data and code. To not be UB, FatGen needs to
    // dump code into u8 buffers but it does everything with FatInstr-strided
    // indices and I am going to defer switching it over for now, because any
    // subsequent pass throws a wrench in this anyway. This makes MIRI scream to
    // high heaven which is unfortunate
    let data = unsafe {
        let mut data = std::mem::ManuallyDrop::new(data);
        Vec::from_raw_parts(data.as_mut_ptr() as *mut FatInstr, data.len() / std::mem::size_of::<FatInstr>(), data.capacity() / std::mem::size_of::<FatInstr>())
    };

    c.ip_start = data.len();

    let mut gen = FatGen::new(data);
    for decl in c.ast.decl_list() {
        if let DeclData::Var(VarDecl { value, .. }) = *c.ast.decl(decl) {
            let name = c.ast.decl(decl).name();
            let sym = sym::compiling(&mut c, name);
            if sym.state == sym::State::Compiling {
                let (ty, location) = (sym.ty, sym.location);
                gen.global_expr(&mut c, ty, value, location, 0);
                sym::compiled(&mut c, name);
            } else {
                assert!(sym.state == sym::State::Compiled);
            }
        }
    }

    let main = c.intern("main");
    if let Some(&sym) = sym::lookup_value(&c, main) {
        let info = c.types.info(sym.ty);
        if info.mutable
        && matches!(info.arguments(), Some(&[]))
        && matches!(info.bare_return_type(), Some(BareType::Int)) {
            gen.code.extend_from_slice(&[FatInstr::call(main), FatInstr::HALT]);
        } else {
            let pos = c.ast.decl(sym.decl).pos();
            error!(c, pos, "main must be a procedure that takes zero arguments and returns int");
        }
    } else {
        gen.code.push(FatInstr::HALT);
    }

    for decl in c.ast.decl_list().skip(1) {
        if c.ast.is_callable(decl) {
            let name = c.ast.decl(decl).name();
            let sym = sym::compiling(&mut c, name);
            assert_eq!(sym.state, sym::State::Compiling, "sym.state == sym::State::Compiling");
            let (ty, decl) = (sym.ty, sym.decl);
            let code = gen.callable(&mut c, ty, decl);
            c.funcs.insert(name, code);
            sym::compiled(&mut c, name);
        }
    }

    let panic = c.intern("panic");
    if let Some(&panic_def) = sym::lookup_value(&c, panic) {
        let info = c.types.info(panic_def.ty);
        assert!(matches!(info.arguments(), Some(&[])));
        assert!(matches!(info.bare_return_type(), Some(BareType::Int)));
        let addr = gen.code.len();
        gen.code.push(FatInstr { op: Op::Panic, dest: 0, left: PanicReason::AssertionFailed as i32, right: 0});
        c.funcs.insert(panic, Code { signature: panic_def.ty, addr });
    }

    c.code = std::mem::take(&mut gen.code);
    c.report_errors(str)?;

    for i in c.ip_start..c.code.len() {
        if let FatInstr { op: Op::Call, left: name, ..} = c.code[i] {
            let addr = c.funcs.get(&Intern(name as u32)).expect("Bad bytecode").addr;
            c.code[i].left = i32::try_from(addr - 1).expect("Todo: Data+code is larger than 2gb");
        }
    }

    Ok(c)
}

fn compile_and_run(str: &str) -> Result<Value> {
    let mut c = compile(str)?;
    let mut stack = vec![0; 8192 * 8];
    c.run(c.ip_start, 0, &mut stack)
}

fn eval_expr(str: &str) -> Result<Value> {
    compile_and_run(&format!("main: proc () int {{ return {}; }}", str))
}

fn repl() {
    use std::io;
    let mut line = String::new();
    println!("lang's bad repl");
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

fn main() {{
    let ok = [
(r#"
Buf: (
    buf: ptr u8 [len],
    buf1: ptr u8 [buf[0]],
    buf2: ptr u8 [*buf],
    buf3: ptr u8 [buf[len]],
    len: int,
    cap: int
);
fn: func (buf: ptr u8 [len], len: int) -> ptr u8  {
    assert((buf:int) == 0);
    return buf:ptr u8;
}
main: proc () -> int {
    b: Buf = {len=1};
    p := fn(b.buf, b.len):ptr u8 [8];
    q := &b.buf[0];
    r := &p[0];
    assert(p == r);
    assert(q == p);
    assert(b.buf == &b.buf[0]);
    assert(b.buf == (0:ptr u8));
    assert(b.buf == p);
    assert(p == (0:ptr u8));
    assert(q == r);
    return 1;
"#, Value::Int(1)),
];
    for test in ok.iter() {
        let str = test.0;
        let expect = test.1;
        compile_and_run(str)
            .and_then(|ret| if ret == expect { Ok(ret) } else { Err(format!("expected {:?}, got {:?}", expect, ret).into())})
            .map_err(|e| { println!("input: \"{}\"", str); e })
            .unwrap();
    }
}
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
    eval!("1 == 1 ? 2, 3", 2);
    eval!("0 == 1 ? 2, 3", 3);
    eval!("(43 * 43) > 0 ? (1+1+1), (2 << 3)", 3);
    eval!("!!0 ? (!!1 ? 2, 3), (!!4 ? 5, 6)", 5);
    eval!("!!0 ?  !!1 ? 2, 3 ,  !!4 ? 5, 6",  5);
    eval!("!!1 ? (!!2 ? 3, 4), (!!5 ? 6, 7)", 3);
    eval!("!!1 ?  !!2 ? 3, 4 ,  !!5 ? 6, 7",  3);
}

#[test]
fn stmt() {
    let ok = [
        "a := 1; do { if (a > 2) { a = (a & 1) == 1 ? a * 2, a + 1; } else { a = a + 1; } } while (a < 10);",
        "a := 1.0; a = a + (1.0:f32);",
        "a := !165: int;",
        "(!!(2*3)) || !!(3-3);",
        "a := (0: bool);",
        "a := 0; a := 1;",
        "if (1:bool) {}",
        "while (0:bool) {}",
        "do {} while (0:bool);",
    ];
    let err = [
        "a := -(0: bool);",
        "();"
    ];
    for str in ok.iter() {
        compile_and_run(&format!("main: proc () int {{ {{ {} }} return 0; }}", str)).map_err(|e| { println!("input: \"{}\"", str); e }).unwrap();
    }
    for str in err.iter() {
        // Not checking it's the right error yet--maybe later
        compile_and_run(&format!("main: proc () int {{ {{ {} }} return 0; }}", str)).map(|e| { println!("input: \"{}\"", str); e }).unwrap_err();
    }
}

#[test]
fn decls() {
    let ok = [
        "add: func (a: int, b: int) int { return a + b; }",
        "add': func (a: int, b: int) int { return a + b; }",
        "add: func (a: int, b: int) -> int { return a + b; }",
        "add: (a: int, b: int) -> int { return a + b; }",
        "add: func (a: struct (a: int), b: (b: int)) int { return a.a + b.b; }",
        "V2: struct (x: f32, y: f32);",
        "V2: struct (x, y: f32);",
        "Node: struct (next: ptr Node, value: int);",
        "Node: struct (children: arr (ptr Node) [2], value: int);",
        "Node: struct (children: ptr (arr Node [2]), value: int);",
        "func: func (func, proc: int) int { return func + proc; }",
        "V1: struct (x: f32); V2: struct (x: V1, y: V1);",
        "struct: struct (struct: struct (struct: int));",
        "struct: (struct: (struct: int));",
        "int: (int: int) int { return int; }"
    ];
    let err = [
        "add: func (a: (a: int) a, b: int) int { return a.a + b; }",
        "add: func (a: struct (a: int) a, b: int) int { return a.a + b; }",
        "dup_func: func (a: int, b: int) int { return a + b; } dup_func: func (a: int, b: int) int { return a + b; }",
        "dup_param: func (a: int, a: int) int { return a + a; }",
        "empty: struct ();",
        "empty: ();",
        "V2: struct (x: f32, y: f32) a;",
        "V2: struct (x: f32, y z: f32);",
        "Node: struct (next: ptr Node2, value: int);",
        "dup: struct (x: f32); dup: struct (y: f32);",
        "dup_: struct{: struct (: f32) dup_: struct{: struct (: f32);",
        "struct: struct (struct: struct (struct: int)) (struct (struct: struct));",
        "dup_field: struct (x: f32, x: f32);",
        "circular: struct (x: circular);",
        "circular1: struct (x: circular2); circular2: struct (x: circular1);",
        "struct: struct (struct: struct);",
        "struct: (struct: (struct: struct));",
        "struct: (struct: struct) struct {}"
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
        "control_paths: func (a: int, b: int, c: bool) int { if (c) { return a + b; } else { return a - b; } }",
        "control_paths: func (a: int, b: int, c: bool) int { while (c) { return a; } }",
        "control_paths: func (a: int, b: int, c: bool) int { return a; while (c) { if (c) { return a; } } }",
    ];
    let err = [
        "control_paths: func (a: int) {}",
        "control_paths: func (a: int, b: int, c: bool) int { if (c) { return a + b; } }",
        "control_paths: func (a: int, b: int, c: bool) int { if (c) { return a + b; } else { } }",
        "control_paths: func (a: int, b: int, c: bool) int { if (c) {} else { return a - b; } }",
        "control_paths: func (a: int, b: int, c: bool) int { while (c) { if (c) { return a; } } }",
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
        "pro: func () int { a: u8 = 1; b: u8 = 2; return a + b; }",
        "pro: func () int { a: u8 = 1; b: u16 = 2; return a + b; }",
        "pro: func () int { a: u8 = 1; b: i16 = 2; return a + b; }",
        "pro: func () int { a: u8 = 255:i16:u8; return a; }",
        "pro: func () int { a: u8 = 256:i16:u8; return a; }",
    ];
    let err = [
        "pro: func () u8 { a: u8 = 1; b: u8 = 2; return a + b; }",
        "pro: func () int { a: u16 = 256:i16:u8; return a; }",
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
        "fits: func () int { a: u8 = -1; return a; }",
        "fits: func () int { a: i8 = 128; return a; }",
        "fits: func () int { a: u8 = 256; return a; }",
        "fits: func () int { a: i16 = 32768; return a; }",
        "fits: func () int { a: u16 = 65536; return a; }",
        "fits: func () int { a: i32 = 2147483648; return a; }",
        "fits: func () int { a: u32 = 4294967296; return a; }",
        "fits: func () int { a := 18446744073709551616; return a; }",
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
madd: func ((a: int, b: int, c: int)) int {
    return a * b + c;
}

main: proc () int {
    return madd(1, 2, 3);
}
"#, Value::Int(5)),
(r#"
fib: func (n: int) int {
    if (n <= 1) {
        return n;
    }

    return fib(n - 2) + fib(n - 1);
}

main: proc () int {
    return fib(6);
}
"#, Value::Int(8)),
(r#"
even: func (n: int) bool {
    return n == 0 ? 1:bool, odd(n - 1);
}

odd: func (n: int) bool {
    return n == 0 ? 0:bool, even(n - 1);
}

main: proc () int {
    return even(10):int;
}
"#, Value::Int(1)),
(r#"
V2: struct (x: f32, y: f32);

dot: func (a: V2, b: V2) f32 {
    return a.x*b.x + a.y*b.y;
}

main: proc () int {
    return dot({ 1.0, 2.0 }, { 3.0, 4.0 }):i32;
}
"#, Value::Int(3+8)),
(r#"
V2: struct (x, y: i32);

add: func (a: V2, b: V2) V2 {
    return { (a.x + b.x):i32, (a.y + b.y):i32 };
}

main: proc () int {
    v := add({1, 2}, {3, 4});
    return v.y;
}
"#, Value::Int(2+4)),
(r#"
// identical codegen
sum: func (a: arr i32 [3]) int {
    acc := 0;
    for (i := 0; i < 3; i = i + 1) {
        acc = acc + a[i];
    }
    return acc;
}
sum2: func (a: ptr (arr i32 [3])) int {
    acc := 0;
    for (i := 0; i < 3; i = i + 1) {
        acc = acc + (*a)[i];
    }
    return acc;
}
sum3: func (a: ptr i32 [3]) int {
    acc := 0;
    for (i := 0; i < 3; i = i + 1) {
        acc = acc + a[i];
    }
    return acc;
}

main: proc () int {
    return sum({2,3,1}) + sum2(&{2,3,1}) + sum3({2,3,1}:arr i32 [3]);
}
"#, Value::Int(18)),
(r#"
does_something_without_return_value: proc (a: ptr i32) {
    if (*a == 0) {
        *a = 1;
        return;
    }

    *a = 2;
}

main: proc () int {
    a: i32 = 1;
    does_something_without_return_value(&a);
    return a;
}
"#, Value::Int(2)),
    ];
    let err = [r#"
even: proc (n: int) bool {
    return n == 0 ? 1:bool, odd(n - 1);
}

odd: func (n: int) bool {
    return n == 0 ? 0:bool, even(n - 1);
}

main: proc () int {
    return even(10):int;
}
"#, r#"
madd: func ((a: int, b: int, c: int)) int {
    return a * b + c;
}

main: proc () int {
    return madd(1, 2);
}
"#,r#"
madd: func ((a: int, b: int, c: int)) int {
    return a * b + c;
}

main: proc () int {
    return madd(1, 2, 3, 4);
}
"#];
    for test in ok.iter() {
        let str = test.0;
        let expect = test.1;
        compile_and_run(str)
            .and_then(|ret| if ret == expect { Ok(ret) } else { Err(format!("expected {:?}, got {:?}", expect, ret).into())})
            .map_err(|e| { println!("input: \"{}\"", str); e })
            .unwrap();
    }
    for str in err.iter() {
        compile_and_run(str).map(|e| { println!("input: \"{}\"", str); e }).unwrap_err();
    }
}

#[test]
fn control_structures() {
    let ok = [
(r#"
main: proc () int {
    a := 0;
    for (i := 0; i < 10; i = i + 1) {
        a = a + 2;
        a = a - 1;
    }
    return a;
}
"#, Value::Int(10)),
(r#"
main: proc () int {
    a := 0;
    for (; a < 10; a = a + 2) {}
    return a;
}
"#, Value::Int(10)),
(r#"
main: proc () int {
    a := 1:bool;
    b := 1:bool;
    c := 1:bool;
    d := 0:bool;
    while (!(a && b && c && d)) {
        d = 1:bool;
    }
    return d:int;
}
"#, Value::Int(1)),
(r#"
main: proc () int {
    a := 0;
    for (;;) {
        a = a + 1;
        if (a == 10) {
            break;
        }
        // continue;
    }
    return a;
}
"#, Value::Int(10)),
(r#"
//
main: proc () int {
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
main: proc () int {
    a := 8;
    switch (a) {
        1 {}
        8 { a = a + 2; }
        3 {}
    }
    switch (a) {
        1 {}
        10 {}
        3 { a = a + 2; }
    }
    switch (a) {
        1 { a = a + 2; }
        10 {}
        3 {}
    }
    switch (a) {
        1 {}
        2 {}
        3 { a = a + 2; }
    }
    // Emits jumps to the next instruction, which we debug assert against not emitting
    // But this case doesn't matter, so...
    // switch (a) {
    //     1 {}
    //     2 {}
    //     3 {}
    // }
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
main: proc () int {
    a := 1;
    do {
        if (a > 2) {
            a = (a & 1) == 1 ? a * 2, a + 1;
        } else {
            a = a + 1;
        }
    } while (a < 10);
    return a;
}"#, Value::Int(14)),
(r#"
main: proc () int {
    a := 1;
    b := 2;
    c := 3;
    (a == 1 ? b, c) = 4;
    return b;
}"#, Value::Int(4)),
(r#"
fn: func (x: int) -> int {
    if (x == 0) {
        return 1;
    } else if (x == 1) {
        return 2;
    } else if (x == 2) {
        return 3;
    }
    return 0;
}
main: proc () int {
    assert(fn(0) == 1);
    assert(fn(1) == 2);
    assert(fn(2) == 3);
    assert(fn(3) == 0);
    return 4;
}"#, Value::Int(4)),
(r#"
inc: proc (p: ptr int) bool {
    *p = *p + 2;
    return 0;
}
main: proc () int {
    t := !!1;
    f := !!0;
    v := 1;
    if ((t || f) && (f || f)) {
        return 9;
    } else if ((t && f) || (t && t) || (f && inc(&v))) {
        return v;
    }
    return 0;
}"#, Value::Int(1)),
    ];
    for test in ok.iter() {
        let str = test.0;
        let expect = test.1;
        compile_and_run(str)
            .and_then(|ret| if ret == expect { Ok(ret) } else { Err(format!("expected {:?}, got {:?}", expect, ret).into())})
            .map_err(|e| { println!("input: \"{}\"", str); e })
            .unwrap();
    }
}

#[test]
fn structs() {
    let ok = [
(r#"
i32x4: struct (
    a: i32,
    b: i32,
    c: i32,
    d: i32,
);

main: proc () int {
    a := {-1,-1,-1,-1}:i32x4;
    return a.d:int;
}
"#, Value::Int(-1)),
(r#"
RGBA: struct (
    r: i8,
    g: i8,
    b: i8,
    a: i8,
);

main: proc () int {
    c := { r = 1, g = 2, b = 3, a = 4 }:RGBA;
    d := { g = 5, r = 6, a = 7, b = 8, }:RGBA;
    e: RGBA = { 9, 10, 11, 12 };
    f: RGBA = { 13, 14, 15 };
    return (c.r << 24) + (d.g << 16) + (e.b << 8) + f.a;
}
"#, Value::Int(0x01050B00)),
(r#"
Outer: struct (
    inner: Inner
);

Inner: struct (
    ignored: i32,
    value: i32
);

main: proc () int {
    a: Outer = { inner = { value = -1234 }};
    b: Outer = { inner = { value = -1234 }:Inner };
    c := { inner = { value = a.inner.value }:Inner }:Outer;
    d: Outer = { inner = { value = b.inner.value }:Inner }:Outer;
    e := { inner = d.inner }:Outer;
    f := { inner = d.inner:Inner }:Outer;
    g := {}:Outer;
    g.inner = f.inner;
    h: Outer = {};
    h.inner.value = f.inner.value;
    i: Outer = {}:Outer;
    i.inner = { value = d.inner.value };
    z: Outer = {};
    if ((a.inner.value == b.inner.value)
     && (b.inner.value == c.inner.value)
     && (c.inner.value == d.inner.value)
     && (d.inner.value == e.inner.value)
     && (e.inner.value == f.inner.value)
     && (f.inner.value == g.inner.value)
     && (g.inner.value == h.inner.value)
     && (h.inner.value != z.inner.value)) {
        return a.inner.value;
    }
    return 0;
}
"#, Value::Int(-1234)),
(r#"
V2: struct (
    x: i32,
    y: i32
);

fn: func () i32 {
    a := { x = 2 }:V2;
    b := { x = 3 }:V2;
    a = b;
    return b.x;
}

main: proc () int {
    a := 0;
    b := 0;
    return fn():int;
}
"#, Value::Int(3)),
(r#"
V4: (
    x: i32,
    y: i32,
    z: i32,
    w: i32,
);

main: proc () int {
    a := ({ x = -1 }:V4).x;
    b := (a == -1) ? { w = 3 }:V4, { w = 4 }:V4;
    return b.w:int;
}
"#, Value::Int(3)),
(r#"
add: func (a: struct (x, y: int), b: (x, y: int)) (x, y: int) {
    return { a.x+b.x, a.y+b.y };
}

main: proc () int {
    a: ptr (x, y: int) = &{};
    *a = add({1,2},{3,4});
    return a.x + a.y;
}
"#, Value::Int(10)),
];
    let err = [r#"
main: proc () int {
    v := {0}:int;
    return v;
}
"#, r#"
V2: struct (x: int, y: int);
main: proc () int {
    v := {}:V2;
    v.a = 0;
    return v.x:int;
}
"#, r#"
V2: struct (x: int, y: int);
main: proc () int {
    v := { a = 0 }:V2;
    return v.a;
},
V2: struct (x: int, y: int);
main: proc () int {
    v := { x = 0, 1 }:V2;
    return v.a;
}
"#];
    for test in ok.iter() {
        let str = test.0;
        let expect = test.1;
        compile_and_run(str)
            .and_then(|ret| if ret == expect { Ok(ret) } else { Err(format!("expected {:?}, got {:?}", expect, ret).into())})
            .map_err(|e| { println!("input: \"{}\"", str); e })
            .unwrap();
    }
    for str in err.iter() {
        compile_and_run(str).map(|e| { println!("input: \"{}\"", str); e }).unwrap_err();
    }
}


#[test]
fn arr() {
    let ok = [
(r#"
asdf: struct (
    arr: arr int [4],
);
main: proc () int {
    asdf := {}:asdf;
    arr: arr int [4:i8] = {};
    asdf.arr[0] = 1;
    asdf.arr[1] = 2;
    asdf.arr[2] = 3;
    asdf.arr[3] = 4;
    arr[0] = 10;
    arr[1] = 20;
    arr[2] = 30;
    arr[3] = 40;
    more := { 100, 200, 300, 400, 500 }:arr int [5];
    return asdf.arr[256:i8] + arr[1] + more[2];
}
"#, Value::Int(321)),
(r#"
main: proc () int {
    arr := {}:arr u8 [8];
    for (i := 0; i != 8; i = i + 1) {
        arr[i] = 1;
    }
    acc := 0;
    for (i := 0; i < 8; i = i + 1) {
        acc = acc + arr[i];
    }
    return acc;
}
"#, Value::Int(8)),
(r#"
V4: struct (padding: i16, c: arr i32 [4]);
M4: struct (padding: i16, r: arr V4 [4]);
main: proc () int {
    a: M4 = {
        r = {
            { c = { 1, 2, 3, 4}},
            { c = { 5, 6, 7, 8}},
            { c = { 9,10,11,12}},
            { c = {13,14,15,16}}
        }
    };
    i := 3;
    j := 2;
    return a.r[i].c[j];
}
"#, Value::Int(15)),
];
    let err = [r#"
main: proc () int {
    a := 2;
    arr: arr int [a] = {};
    return arr[0];
}
#",r#"
main: proc () int {
    arr: arr int [1] = { 0 = 1 };
    return arr[0];
}
"#, r#"
main: proc () int {
    arr: arr int [4.0] = {};
    return arr[0];
}
"#];
    for test in ok.iter() {
        let str = test.0;
        let expect = test.1;
        compile_and_run(str)
            .and_then(|ret| if ret == expect { Ok(ret) } else { Err(format!("expected {:?}, got {:?}", expect, ret).into())})
            .map_err(|e| { println!("input: \"{}\"", str); e })
            .unwrap();
    }
    for str in err.iter() {
        compile_and_run(str).map(|e| { println!("input: \"{}\"", str); e }).unwrap_err();
    }
}

#[test]
fn ptr() {
    let ok = [
(r#"
main: proc () int {
    a := 1;
    b := &a;
    *b = 2;
    return *b;
}
"#, Value::Int(2)),
(r#"
main: proc () int {
    a := 1;
    *&a = 2;
    b := &a:int;
    return a;
}
"#, Value::Int(2)),
(r#"
main: proc () int {
    a := {}:arr int [2];
    a0 := &a[0];
    a0' := &a[0];
    a1 := &a[1];
    assert(a0 == a0');
    assert(a0 + 1 == a1);
    return *a0;
}
"#, Value::Int(0)),
(r#"
V2: struct (x: i32, y: i32);
main: proc () int {
    aa := {1, 2}:V2;
    bb := aa;
    a := &aa;
    b := &bb;
    c := &{1, 2}:V2;
    *a = {3, 4}:V2;
    *b = *a;
    c.x = b.x;
    c.y = b.y;
    d := *c;
    if (a.x == aa.x && b.x == bb.x && a.x == b.x && b.x == c.x
     && a.y == aa.y && b.y == bb.y && a.y == b.y && b.y == c.y
     && c.x == d.x && c.x == d.x && c.x == d.x && c.x == d.x
     && c.y == d.y && c.y == d.y && c.y == d.y && c.y == d.y) {
        return aa.x;
    }
    return 0;
}
"#, Value::Int(3)),
(r#"
main: proc () int {
    arr := {}:arr u8 [8];
    for (p := &arr[0]; p != &arr[0]+8; p = p + 1) {
        *p = 1;
    }
    arrp := &arr[0];
    acc := 0;
    for (i := 0; i < 8; i = i + 1) {
        acc = acc + *(arrp + i);
    }
    return acc;
}
"#, Value::Int(8)),
 (r#"
main: proc () int {
    arr := {}:arr u16 [8];
    for (p := &arr[0]; p != &arr[0]+8; p = p + 1) {
        *p = p - &arr[0] : u16;
    }
    arrp := &arr[0];
    acc := 0;
    for (i := 0; i < 8; i = i + 1) {
        acc = acc + *(arrp + (&arr[i] - arrp));
    }
    return acc;
}
"#, Value::Int(0+1+2+3+4+5+6+7)),
(r#"
Buf: struct (
    buf: ptr u8 [len],
    len: int
);
main: proc () int {
    arr := {}:arr u8 [8];
    arr[0] = 255;
    arr[1] = 1;
    buf: Buf = { arr, 8 };
    a := &arr;
    b := buf.buf;
    bb := &buf;
    c: ptr u8 = &arr[0];
    i := 1;
    assert(bb.buf[0] == (*a)[0]);
    assert(&arr[0] == buf.buf);
    assert(arr[0] == buf.buf[0]);
    assert(arr[1] == buf.buf[i]);
    assert(arr[1] == *(b+i));
    assert(&arr[1] == &*(b+i));
    assert(arr[0] == *c);
    assert((arr:ptr u8) == c);
    assert(a == c);
    return buf.buf[1];
}
"#, Value::Int(1)),
(r#"
A: struct (
    padding: arr i16 [9],
    b: ptr B
);
B: struct (
    padding: arr i16 [3],
    c: ptr C
);
C: struct (
    padding: i16,
    d: ptr i32
);
main: proc () int {
    d := -1234;
    c: C = {d=&d:ptr(i32)};
    b: B = {c=&c};
    a: A = {b=&b};
    return *a.b.c.d;
}
"#, Value::Int(-1234)),
(r#"
V2: struct (x: i32, y: i32);
rot:(proc)(v:ptr(V2))(int) {
    *v = { -v.y:i32, v.x };
    return 0;
}
main: proc () int {
    v: V2 = {0,1};
    rot(&v);
    return v.x;
}
"#, Value::Int(-1)),
(r#"
V2:(struct (x: i32, y: i32));
main: (proc () int) {
    return (&(0:ptr V2).y:ptr) - (0:ptr);
}
"#, Value::Int(4)),
(r#"
inc: proc (v: ptr (arr int [1])) {
    (*v)[0] = (*v)[0] + 1;
}
implicit_pointer_arg_ref: proc (v: arr int [1]) int {
    inc(&v);
    return v[0];
}
main: proc () int {
    return implicit_pointer_arg_ref({3});
}
"#, Value::Int(4)),
(r#"
V2: struct (x: i32, y: i32);
rot22: proc(vv: ptr -> ptr V2) {
    **vv = { -(*vv).y:i32, (*vv).x };
}
rot12: proc (v: ptr V2) {
    vv := &v;
    rot22(vv);
}
rot21: proc(vv: ptr -> ptr V2) {
    (*vv).x = -1;
}
rot11: proc(v: ptr V2) {
    vv := &v;
    rot21(vv);
}
main: proc () int {
    v1: V2 = {0,1};
    v2: V2 = {0,2};
    rot11(&v1);
    rot12(&v2);
    assert(v1.x == -1);
    assert(v2.x == -2);
    return v1.x;
}
"#, Value::Int(-1)),
];
    let err = [r#"
main: proc () int {
    a := 1;
    return *a;
}
"#,r#"
main: proc () -> int {
    a := &-1:u8;
    return *a;
}
"#,r#"
main: proc () int {
    a := -1;
    b := &a:f32;
    return *b;
}
"#,r#"
main: proc () int {
    pp := 0:ptr u8;
    pq := 0:ptr u16;
    a := pq - pp;
    return 0;
}
"#, r#"
main: proc () int {
    a := &3;
    return 0;
}
"#,r#"
inc: proc (v: ptr int) {
    *v = *v + 1;
}
main: proc () int {
    a: proc (v: ptr (arr int [1])) = inc;
    return 0;
}"#,r#"
inc: proc (v: ptr int) {
    *v = *v + 1;
}
main: proc () int {
    a := &inc;
    return 0;
}"#,r#"
main: proc () int {
    arr := {}:arr u8 [8];
    a := &arr;
    assert(a[0] == arr);
    return 0;
}
"#];
    for test in ok.iter() {
        let str = test.0;
        let expect = test.1;
        compile_and_run(str)
            .and_then(|ret| if ret == expect { Ok(ret) } else { Err(format!("expected {:?}, got {:?}", expect, ret).into())})
            .map_err(|e| { println!("input: \"{}\"", str); e })
            .unwrap();
    }
    for str in err.iter() {
        compile_and_run(str).map(|e| { println!("input: \"{}\"", str); e }).unwrap_err();
    }
}

#[test]
fn globals() {
    let ok = [
        (r#"
a: (v: int) = {1};
b: (v: int) = {1};
main: proc () -> int {
    assert(b.v == 1);
    return b.v;
}"#, Value::Int(1)),
(r#"
V2: struct (x, y: int);
a: int = 1 + 32;
b: V2 = { 2, 3 }:V2;
c: V2 = { 4, 5 };
d: arr u8 [3] = { 0, 1, 2 };
main: proc () -> int {
    assert(a == 33);
    assert(b.x == 2);
    assert(b.y == 3);
    assert(c.x == 4);
    assert(c.y == 5);
    assert(d[0] == 0);
    assert(d[1] == 1);
    assert(d[2] == 2);
    d[0] = d[0] + d[1] : u8;
    assert(d[0] == d[1]);
    return d[1];
}"#, Value::Int(1)),
(r#"
add: func (a, b: int) int {
    return a + b;
}
madd: func (a, b, c: int) int {
    return add(a, b * c);
}
A: int = madd(2,3,3);
main: proc () -> int {
    assert(A == 11);
    return A;
}"#, Value::Int(11)),
(r#"
inc: proc (a: ptr int) int {
    old := *a;
    *a = *a + 1;
    return old;
}
a: int = 1;
b: int = inc(&a);
main: proc () -> int {
    assert(b == 1);
    return a;
}"#, Value::Int(2)),
(r#"
inc: proc (a: ptr int) int {
    old := *a;
    *a = *a + 1;
    return old;
}
d: int = a;
b: int = inc(&c);
c: int = a;
a: int = 1;
main: proc () -> int {
    assert(d == 1);
    assert(c == 2);
    assert(b == 1);
    assert(a == 1);
    return a;
}"#, Value::Int(1)),
];
    let err = [r#"
b: int = a;
a: int = b;
}"#,r#"
g: () int {
    return a + 1;
}
b: int = g();
a: int = 0;
main: proc () -> int {
    return b;
}"#];
    for test in ok.iter() {
        let str = test.0;
        let expect = test.1;
        compile_and_run(str)
            .and_then(|ret| if ret == expect { Ok(ret) } else { Err(format!("expected {:?}, got {:?}", expect, ret).into())})
            .map_err(|e| { println!("input: \"{}\"", str); e })
            .unwrap();
    }
    for str in err.iter() {
        compile_and_run(str).map(|e| { println!("input: \"{}\"", str); e }).unwrap_err();
    }
}

#[test]
fn bounds() {
    let ok = [
        (r#"
Buf: (
    buf3: ptr u8 [buf[len]],
    buf2: ptr u8 [*buf],
    buf1: ptr u8 [buf[0]],
    buf: ptr u8 [len],
    len: int,
    cap: int
);
fn: func (buf: ptr u8 [len], len: int) -> ptr u8 [len] {
    assert((buf:int) == 0);
    return buf;
}
main: proc () -> int {
    b: Buf = {len=1};
    p := fn(b.buf, b.len);
    q := &b.buf[0];
    r := &p[0];
    assert(p == r);
    assert(q == p);
    assert(b.buf == &b.buf[0]);
    assert(b.buf == (0:ptr u8));
    assert(b.buf == p);
    assert(p == (0:ptr u8));
    assert(q == r);
    return 1;
}"#, Value::Int(1)), (r#"
aaa: (a: int) -> ptr int [8] {
    return a:ptr int [8];
}
main: proc () -> int {
    assert(aaa(3) == (3:ptr u8 [8]));
    return 0;
}"#, Value::Int(0)),
];
    let err = [r#"
Buf: (buf: ptr u8 [len], len: int);
fn: func (buf: ptr u8 [len], len: int) -> ptr u8 [len] { return buf; }
main: proc () -> int {
    b: Buf = {len=1};
    p := fn(b.buf, b.len);
    q := p[1];
    return 0;
}"#];
    for test in ok.iter() {
        let str = test.0;
        let expect = test.1;
        compile_and_run(str)
            .and_then(|ret| if ret == expect { Ok(ret) } else { Err(format!("expected {:?}, got {:?}", expect, ret).into())})
            .map_err(|e| { println!("input: \"{}\"", str); e })
            .unwrap();
    }
    for str in err.iter() {
        compile_and_run(str).map(|e| { println!("input: \"{}\"", str); e }).unwrap_err();
    }
}

// cargo +nightly bench --features bench
#[cfg(feature = "bench")]
#[cfg(test)]
mod bench {
    extern crate test;
    use test::Bencher;
    use super::*;

    #[bench]
    fn compile(b: &mut test::Bencher) {
        let fragment = r#"
// _00 = prev
// _01 = current
Struct_01: struct (
    field0: Struct_00,
    field1: int,
    ptr: ptr Struct_00,
    arr: arr int [16]
);

g_01: int = 0;

func_01: func (arg0: Struct_01, arg1: ptr Struct_00) -> int {
    if (arg0.field1 > 0) {
        for (i := arg0.field1; i != 0; i = i - 1) {
            switch (arg1.field1) {
                0 { return arg0.field1; }
                1 { return arg0.field1; }
                2 { return arg0.field1; }
                523 { return arg0.field1; }
            }
        }
    }
    return arg0.field1 + arg1.field1 + g_01;
}

proc_01: proc (arg: ptr Struct_00) -> int {
    proc_00(&arg.field0);
    return func_01({ *arg:Struct_00, 1, arg }, arg) + func_00(*arg, &{});
}

"#;
        let mut code = String::from(r#"
Struct0: (field0, field1: int);

Struct1: struct (
    field0: Struct0,
    field1: int,
    ptr: ptr Struct0,
    arr: arr int [16]
);

g0: int = 0;

func1: proc (arg0: Struct1, arg1: ptr Struct0) -> int {
    return arg0.field1 + arg1.field1 + g0;
}

proc1: proc (arg: ptr Struct0) -> int {
    return arg.field1;
}

"#);
        for i in 2..2*1024 {
            code.push_str(&fragment.replace("_00", &(i-1).to_string()).replace("_01", &i.to_string()));
        }
        b.bytes = code.len() as u64;
        b.iter(|| {
            let c = crate::compile(&code).expect("ok");
            test::black_box(&c);
        });
    }
}
