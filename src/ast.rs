use std::convert::TryFrom;
use std::fmt::Display;

use crate::Intern;
use crate::parse::TokenKind;
use crate::parse::Keytype;

macro_rules! define_node_list {
    ($Node: ident, $List: ident, $ListIter: ident) => {
        #[derive(Copy, Clone, Debug)]
        pub struct $List {
            begin: u32,
            end: u32
        }

        #[allow(dead_code)]
        impl $List {
            pub fn empty() -> $List {
                $List { begin: 0, end: 0 }
            }

            pub fn from(value: $Node) -> $List {
                $List { begin: value.0, end: value.0 + 1 }
            }

            pub fn is_nonempty(self) -> bool {
                self.begin != self.end
            }

            pub fn is_empty(self) -> bool {
                self.begin == self.end
            }

            pub fn len(self) -> usize {
                (self.end - self.begin) as usize
            }

            pub fn as_range(self) -> std::ops::Range<usize> {
                self.begin as usize .. self.end as usize
            }
        }

        impl Iterator for $List {
            type Item = $Node;
            fn next(&mut self) -> Option<Self::Item> {
                if self.begin < self.end {
                    let result = Some($Node(self.begin));
                    self.begin += 1;
                    result
                } else {
                    None
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompoundField(u32);
define_node_list!(CompoundField, CompoundFieldList, CompoundFieldListIter);

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub enum CompoundPath {
    Implicit,
    Path(Expr),
    Index(Expr)
}

#[derive(Clone, Copy, Debug)]
pub struct CompoundFieldData {
    pub path: CompoundPath,
    pub value: Expr
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Expr(u32);

define_node_list!(Expr, ExprList, ExprListIter);

#[derive(Clone, Copy, Debug)]
pub enum ExprData {
    Int(usize),
    Float32(f32),
    Float64(f64),
    Name(Intern),
    Compound(CompoundFieldList),
    Field(Expr, Intern),
    Index(Expr, Expr),
    Unary(TokenKind, Expr),
    Binary(TokenKind, Expr, Expr),
    Ternary(Expr, Expr, Expr),
    Call(Expr, ExprList),
    Cast(Expr, TypeExpr),
}

impl Display for ExprData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            ExprData::Int(a) => write!(f, "integer ({})", a),
            ExprData::Float32(a) => write!(f, "f32 ({})", a),
            ExprData::Float64(a) => write!(f, "f64 ({})", a),
            ExprData::Name(_) => write!(f, "identifier"),
            _ => write!(f, "expression"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ExprAux {
    pos: usize
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TypeExpr(pub u32);
define_node_list!(TypeExpr, TypeExprList, TypeExprListIter);

#[allow(non_upper_case_globals)]
impl TypeExpr {
    pub const Infer: TypeExpr = TypeExpr(0);
}

#[derive(Clone, Copy, Debug)]
pub enum TypeExprData {
    Infer,
    Name(Intern),
    Expr(Expr),
    Items(ItemList),
    List(TypeExprList),
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SwitchCase(u32);
define_node_list!(SwitchCase, SwitchCaseList, SwitchCaseListIter);

#[derive(Clone, Copy, Debug)]
pub enum SwitchCaseData {
    Cases(StmtList, ExprList),
    Else(StmtList),
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Stmt(u32);
define_node_list!(Stmt, StmtList, StmtListIter);

// Zero perf difference, half the size
#[derive(Clone, Copy, Debug)]
pub enum PackedStmtData {
    Block(StmtList),
    Return(Option<Expr>),
    Break,
    Continue,
    If(Expr, Stmt, Stmt, Stmt),
    IfElseIf(Expr, Stmt, Stmt, Stmt),
    While(Expr, StmtList),
    For000(StmtList),
    For001(Stmt, StmtList),
    For010(Expr, StmtList),
    For011(Expr, Stmt, StmtList),
    For100(Stmt, StmtList),
    For101(Stmt, StmtList),
    For110(Expr, Stmt, StmtList),
    For111(Expr, Stmt, StmtList),
    Switch(Expr, SwitchCaseList),
    Do(Expr, StmtList),
    Expr(Expr),
    Assign(Expr, Expr),
    VarDecl(TypeExpr, Expr, Expr)
}

impl PackedStmtData {
    fn unpack(&self) -> StmtData {
        use StmtData::*;
        match *self {
            PackedStmtData::Block(stmt_list)                                                    =>  Block(stmt_list),
            PackedStmtData::Return(stmt)                                                        =>  Return(stmt),
            PackedStmtData::Break                                                               =>  Break,
            PackedStmtData::Continue                                                            =>  Continue,
            PackedStmtData::IfElseIf(expr, Stmt(then_begin), Stmt(then_end), Stmt(elseif_end))  =>  If(expr, StmtList { begin: then_begin, end: then_end }, StmtList { begin: then_end + 1, end: elseif_end }),
            PackedStmtData::If(expr, Stmt(then_begin), Stmt(else_begin), Stmt(else_end))        =>  If(expr, StmtList { begin: then_begin, end: else_begin }, StmtList { begin: else_begin, end: else_end }),
            PackedStmtData::While(expr, stmt_list)                                              =>  While(expr, stmt_list),
            PackedStmtData::For000(            stmt_list)                                       =>  For(None,                    None,       None,        stmt_list),
            PackedStmtData::For001(      stmt, stmt_list)                                       =>  For(None,                    None,       Some(stmt),  stmt_list),
            PackedStmtData::For010(expr,       stmt_list)                                       =>  For(None,                    Some(expr), None,        stmt_list),
            PackedStmtData::For011(expr, stmt, stmt_list)                                       =>  For(None,                    Some(expr), Some(stmt),  stmt_list),
            PackedStmtData::For100(      stmt, stmt_list)                                       =>  For(Some(stmt),   None,       None,                   stmt_list),
            PackedStmtData::For101(      stmt, stmt_list)                                       =>  For(Some(stmt),   None,       Some(Stmt(stmt.0 + 1)), stmt_list),
            PackedStmtData::For110(expr, stmt, stmt_list)                                       =>  For(Some(stmt),   Some(expr), None,                   stmt_list),
            PackedStmtData::For111(expr, stmt, stmt_list)                                       =>  For(Some(stmt),   Some(expr), Some(Stmt(stmt.0 + 1)), stmt_list),
            PackedStmtData::Switch(expr, switch_case_list)                                      =>  Switch(expr, switch_case_list),
            PackedStmtData::Do(expr, stmt_list)                                                 =>  Do(expr, stmt_list),
            PackedStmtData::Expr(expr)                                                          =>  Expr(expr),
            PackedStmtData::Assign(left, right)                                                 =>  Assign(left, right),
            PackedStmtData::VarDecl(type_expr, left, right)                                     =>  VarDecl(type_expr, left, right),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum StmtData {
    Block(StmtList),
    Return(Option<Expr>),
    Break,
    Continue,
    If(Expr, StmtList, StmtList),
    While(Expr, StmtList),
    For(Option<Stmt>, Option<Expr>, Option<Stmt>, StmtList),
    Switch(Expr, SwitchCaseList),
    Do(Expr, StmtList),
    Expr(Expr),
    Assign(Expr, Expr),
    VarDecl(TypeExpr, Expr, Expr)
}

impl StmtData {
    fn pack(&self) -> PackedStmtData {
        use StmtData::*;
        match *self {
            Block(stmt_list)                                       => PackedStmtData::Block(stmt_list),
            Return(stmt)                                           => PackedStmtData::Return(stmt),
            Break                                                  => PackedStmtData::Break,
            Continue                                               => PackedStmtData::Continue,
            If(expr, then, elseif) if then.end + 1 == elseif.begin => PackedStmtData::IfElseIf(expr, Stmt(then.begin), Stmt(then.end), Stmt(elseif.end)),
            If(expr, then, ellse)                                  => PackedStmtData::If(expr, Stmt(then.begin), Stmt(ellse.begin), Stmt(ellse.end)),
            While(expr, stmt_list)                                 => PackedStmtData::While(expr, stmt_list),
            For(None,        None,         None,       stmt_list)  => PackedStmtData::For000(            stmt_list),
            For(None,        None,         Some(post), stmt_list)  => PackedStmtData::For001(      post, stmt_list),
            For(None,        Some(expr),   None,       stmt_list)  => PackedStmtData::For010(expr,       stmt_list),
            For(None,        Some(expr),   Some(post), stmt_list)  => PackedStmtData::For011(expr, post, stmt_list),
            For(Some(pre),   None,         None,       stmt_list)  => PackedStmtData::For100(      pre,  stmt_list),
            For(Some(pre),   None,         Some(_),    stmt_list)  => PackedStmtData::For101(      pre,  stmt_list),
            For(Some(pre),   Some(expr),   None,       stmt_list)  => PackedStmtData::For110(expr, pre,  stmt_list),
            For(Some(pre),   Some(expr),   Some(_),    stmt_list)  => PackedStmtData::For111(expr, pre,  stmt_list),
            Switch(expr, switch_case_list)                         => PackedStmtData::Switch(expr, switch_case_list),
            Do(expr, stmt_list)                                    => PackedStmtData::Do(expr, stmt_list),
            Expr(expr)                                             => PackedStmtData::Expr(expr),
            Assign(left, right)                                    => PackedStmtData::Assign(left, right),
            VarDecl(type_expr, left, right)                        => PackedStmtData::VarDecl(type_expr, left, right),
        }
    }
}

/// A named declaration of a type, as in function parameters and
/// struct fields.
#[derive(Clone, Copy, Debug, Default)]
pub struct Item(u32);
define_node_list!(Item, ItemList, ItemListIter);

#[derive(Clone, Copy, Debug)]
pub struct ItemData {
    pub name: Intern,
    pub expr: TypeExpr,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Decl(u32);
define_node_list!(Decl, DeclList, DeclListIter);

impl Decl {
    pub const BUILTIN: Decl = Decl(!0);
}

#[derive(Clone, Copy, Debug)]
pub struct CallableDecl {
    pub pos: usize,
    pub name: Intern,
    pub expr: TypeExpr,
    pub body: StmtList
}

#[derive(Clone, Copy, Debug)]
pub struct StructDecl {
    pub pos: usize,
    pub name: Intern,
    pub expr: TypeExpr,
}

#[derive(Clone, Copy, Debug)]
pub struct VarDecl {
    pub pos: usize,
    pub name: Intern,
    pub expr: TypeExpr,
    pub value: Expr
}

#[derive(Clone, Copy, Debug)]
pub enum DeclData {
    Callable(CallableDecl),
    Struct(StructDecl),
    Var(VarDecl)
}

impl DeclData {
    pub fn name(&self) -> Intern {
        match &self {
            DeclData::Callable(decl) => decl.name,
            DeclData::Struct(decl) => decl.name,
            DeclData::Var(decl) => decl.name,
        }
    }

    pub fn pos(&self) -> usize {
        match &self {
            DeclData::Callable(decl) => decl.pos,
            DeclData::Struct(decl) => decl.pos,
            DeclData::Var(decl) => decl.pos,
        }
    }

    pub fn expr(&self) -> TypeExpr {
        match &self {
            DeclData::Callable(decl) => decl.expr,
            DeclData::Struct(decl) => decl.expr,
            DeclData::Var(decl) => decl.expr,
        }
    }
}

impl Display for DeclData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            // todo: func/proc
            DeclData::Callable(_) => write!(f, "callable"),
            DeclData::Struct(_) => write!(f, "struct"),
            DeclData::Var(_) => write!(f, "var"),
        }
    }
}

fn push_data<T>(vec: &mut Vec<T>, data: T) -> u32 {
    let result = u32::try_from(vec.len()).expect("Program too big!");
    vec.push(data);
    result
}

fn push_slice<T: Clone>(vec: &mut Vec<T>, data: &[T]) -> (u32, u32) {
    let begin = u32::try_from(vec.len()).expect("Program too big!");
    vec.extend_from_slice(data);
    let end = u32::try_from(vec.len()).expect("Program too big!");
    (begin, end)
}

#[derive(Default)]
pub struct Ast {
    exprs: Vec<ExprData>,
    exprs_aux: Vec<ExprAux>,
    fields: Vec<CompoundFieldData>,
    stmts: Vec<PackedStmtData>,
    cases: Vec<SwitchCaseData>,
    items: Vec<ItemData>,
    decls: Vec<DeclData>,
    type_exprs: Vec<TypeExprData>,
}

impl Ast {
    pub fn new() -> Ast {
        let mut result: Ast = Default::default();
        result.type_exprs.push(TypeExprData::Infer);
        result
    }

    #[allow(dead_code)]
    pub fn size_in_bytes(&self) -> usize {
          self.exprs.len()      * std::mem::size_of::<ExprData>()
        + self.exprs_aux.len()  * std::mem::size_of::<ExprAux>()
        + self.fields.len()     * std::mem::size_of::<CompoundFieldData>()
        + self.stmts.len()      * std::mem::size_of::<PackedStmtData>()
        + self.cases.len()      * std::mem::size_of::<SwitchCaseData>()
        + self.items.len()      * std::mem::size_of::<ItemData>()
        + self.decls.len()      * std::mem::size_of::<DeclData>()
        + self.type_exprs.len() * std::mem::size_of::<TypeExprData>()
    }

    pub fn expr(&self, expr: Expr) -> ExprData {
        self.exprs[expr.0 as usize]
    }

    pub fn expr_source_position(&self, expr: Expr) -> usize {
        self.exprs_aux[expr.0 as usize].pos
    }

    pub fn compound_field(&self, field: CompoundField) -> CompoundFieldData {
        self.fields[field.0 as usize]
    }

    fn push_expr(&mut self, pos: usize, data: ExprData) -> Expr {
        let result = push_data(&mut self.exprs, data);
        self.exprs_aux.push(ExprAux { pos });
        Expr(result)
    }

    pub fn push_exprs(&mut self, data: &[(ExprData, ExprAux)]) -> ExprList {
        let begin = u32::try_from(self.exprs.len()).expect("Program too big!");
        self.exprs.extend(data.iter().map(|e| e.0));
        self.exprs_aux.extend(data.iter().map(|e| e.1));
        let end = u32::try_from(self.exprs.len()).expect("Program too big!");
        ExprList { begin, end }
    }

    pub fn push_expr_int(&mut self, pos: usize, value: usize) -> Expr {
        self.push_expr(pos, ExprData::Int(value))
    }

    pub fn push_expr_float32(&mut self, pos: usize, value: f32) -> Expr {
        self.push_expr(pos, ExprData::Float32(value))
    }

    pub fn push_expr_float64(&mut self, pos: usize, value: f64) -> Expr {
        self.push_expr(pos, ExprData::Float64(value))
    }

    pub fn push_expr_name(&mut self, pos: usize, value: Intern) -> Expr {
        self.push_expr(pos, ExprData::Name(value))
    }

    pub fn push_expr_compound(&mut self, pos: usize, fields: &[CompoundFieldData]) -> Expr {
        let (begin, end) = push_slice(&mut self.fields, fields);
        self.push_expr(pos, ExprData::Compound(CompoundFieldList { begin, end }))
    }

    pub fn push_expr_field(&mut self, pos: usize, expr: Expr, field: Intern) -> Expr {
        self.push_expr(pos, ExprData::Field(expr, field))
    }

    pub fn push_expr_index(&mut self, pos: usize, expr: Expr, index: Expr) -> Expr {
        self.push_expr(pos, ExprData::Index(expr, index))
    }

    pub fn push_expr_unary(&mut self, pos: usize, op: TokenKind, right: Expr) -> Expr {
        self.push_expr(pos, ExprData::Unary(op, right))
    }

    pub fn push_expr_binary(&mut self, pos: usize, op: TokenKind, left: Expr, right: Expr) -> Expr {
        self.push_expr(pos, ExprData::Binary(op, left, right))
    }

    pub fn push_expr_ternary(&mut self, pos: usize, cond: Expr, left: Expr, right: Expr) -> Expr {
        self.push_expr(pos, ExprData::Ternary(cond, left, right))
    }

    pub fn push_expr_call(&mut self, pos: usize, callable: Expr, args: ExprList) -> Expr {
        self.push_expr(pos, ExprData::Call(callable, args))
    }

    pub fn push_expr_cast(&mut self, pos: usize, expr: Expr, ty: TypeExpr) -> Expr {
        self.push_expr(pos, ExprData::Cast(expr, ty))
    }

    pub fn pop_expr(&mut self, expr: Expr) -> (ExprData, ExprAux) {
        debug_assert!(expr.0 as usize == self.exprs.len() - 1);
        (self.exprs.remove(expr.0 as usize), self.exprs_aux.remove(expr.0 as usize))
    }


    pub fn stmt(&self, stmt: Stmt) -> StmtData {
        self.stmts[stmt.0 as usize].unpack()
    }

    fn push_stmt(&mut self, data: StmtData) -> Stmt {
        let result = push_data(&mut self.stmts, data.pack());
        Stmt(result)
    }

    pub fn push_stmts(&mut self, data: &[StmtData]) -> StmtList {
        let begin = u32::try_from(self.stmts.len()).expect("Program too big!");
        self.stmts.extend(data.iter().map(|d| d.pack()));
        let end = u32::try_from(self.stmts.len()).expect("Program too big!");
        StmtList { begin, end }
    }

    pub fn push_stmt_block(&mut self, list: StmtList) -> Stmt {
        self.push_stmt(StmtData::Block(list))
    }

    pub fn push_stmt_return(&mut self, expr: Option<Expr>) -> Stmt {
        self.push_stmt(StmtData::Return(expr))
    }

    pub fn push_stmt_break(&mut self) -> Stmt {
        self.push_stmt(StmtData::Break)
    }

    pub fn push_stmt_continue(&mut self) -> Stmt {
        self.push_stmt(StmtData::Continue)
    }

    pub fn push_stmt_if(&mut self, cond: Expr, mut then_stmt: StmtList, mut else_stmt: StmtList) -> Stmt {
        // Simplifies stmt packing
        if then_stmt.is_empty() {
            then_stmt = StmtList { begin: else_stmt.begin, end: else_stmt.begin };
        }
        if else_stmt.is_empty() {
            else_stmt = StmtList { begin: then_stmt.end, end: then_stmt.end };
        }
        self.push_stmt(StmtData::If(cond, then_stmt, else_stmt))
    }

    pub fn push_stmt_for(&mut self, pre: Option<Stmt>, cond: Option<Expr>, post: Option<Stmt>, body: StmtList) -> Stmt {
        if let (Some(pre), Some(post)) = (pre, post) {
            // Condition for stmt packing
            debug_assert!(pre.0 + 1 == post.0);
        }
        self.push_stmt(StmtData::For(pre, cond, post, body))
    }

    pub fn push_stmt_while(&mut self, cond: Expr, body: StmtList) -> Stmt {
        self.push_stmt(StmtData::While(cond, body))
    }

    pub fn push_stmt_do(&mut self, cond: Expr, body: StmtList) -> Stmt {
        self.push_stmt(StmtData::Do(cond, body))
    }

    pub fn push_stmt_switch(&mut self, control: Expr, cases: &[SwitchCaseData]) -> Stmt {
        let (begin, end) = push_slice(&mut self.cases, cases);
        self.push_stmt(StmtData::Switch(control, SwitchCaseList { begin, end }))
    }

    pub fn push_stmt_expr(&mut self, expr: Expr) -> Stmt {
        self.push_stmt(StmtData::Expr(expr))
    }

    pub fn push_stmt_assign(&mut self, left: Expr, right: Expr) -> Stmt {
        self.push_stmt(StmtData::Assign(left, right))
    }

    pub fn push_stmt_vardecl(&mut self, ty: TypeExpr, left: Expr, right: Expr) -> Stmt {
        self.push_stmt(StmtData::VarDecl(ty, left, right))
    }

    pub fn pop_stmt(&mut self, stmt: Stmt) -> StmtData {
        debug_assert!(stmt.0 as usize == self.stmts.len() - 1);
        self.stmts.remove(stmt.0 as usize).unpack()
    }

    pub fn switch_case(&self, case: SwitchCase) -> SwitchCaseData {
        self.cases[case.0 as usize]
    }


    pub fn push_items(&mut self, data: &[ItemData]) -> ItemList {
        let (begin, end) = push_slice(&mut self.items, data);
        ItemList { begin, end }
    }

    pub fn item(&self, item: Item) -> ItemData {
        self.items[item.0 as usize]
    }

    pub fn items(&self, items: ItemList) -> &[ItemData] {
        &self.items[items.as_range()]
    }

    fn push_decl(&mut self, data: DeclData) -> Decl {
        let result = push_data(&mut self.decls, data);
        Decl(result)
    }

    pub fn push_decl_callable(&mut self, decl: CallableDecl) -> Decl {
        self.push_decl(DeclData::Callable(decl))
    }

    pub fn push_decl_struct(&mut self, decl: StructDecl) -> Decl {
        self.push_decl(DeclData::Struct(decl))
    }

    pub fn push_decl_var(&mut self, decl: VarDecl) -> Decl {
        self.push_decl(DeclData::Var(decl))
    }

    pub fn decl(&self, decl: Decl) -> &DeclData {
        &self.decls[decl.0 as usize]
    }

    pub fn decl_list(&self) -> DeclList {
        DeclList { begin: 0, end: self.decls.len() as u32 }
    }

    pub fn callable(&self, decl: Decl) -> &CallableDecl {
        match &self.decls[decl.0 as usize] {
            DeclData::Callable(callable) => callable,
            _ => panic!("decl is not a callable")
        }
    }

    pub fn is_callable(&self, decl: Decl) -> bool {
        match &self.decls[decl.0 as usize] {
            DeclData::Callable(_) => true,
            _ => false
        }
    }

    pub fn push_type_expr(&mut self, data: TypeExprData) -> TypeExpr {
        let result = push_data(&mut self.type_exprs, data);
        TypeExpr(result)
    }

    pub fn push_type_expr_list(&mut self, list: &[TypeExprData]) -> TypeExpr {
        let slice = push_slice(&mut self.type_exprs, list);
        let result = push_data(&mut self.type_exprs, TypeExprData::List(TypeExprList { begin: slice.0, end: slice.1 }));
        TypeExpr(result)
    }

    pub fn type_expr(&self, expr: TypeExpr) -> &TypeExprData {
        &self.type_exprs[expr.0 as usize]
    }

    // I wonder what the codegen looks like for these

    fn type_expr_index(&self, expr: TypeExpr, index: usize) -> Option<TypeExpr> {
        match self.type_expr(expr) {
            TypeExprData::List(list) => { let mut list = *list; list.nth(index) },
            _ => None
        }
    }

    fn type_expr_has_explicit_keytype(&self, expr: TypeExpr) -> bool {
        match self.type_expr(expr) {
            TypeExprData::Name(key) => Keytype::from_intern(*key).is_some(),
            TypeExprData::List(_) => self.type_expr_index(expr, 0).map(|t| match self.type_expr(t) {
                TypeExprData::Name(key) => Keytype::from_intern(*key).is_some(),
                _ => false
            }).unwrap_or(false),
            _ => false
        }
    }

    // All type exprs

    pub fn type_expr_keytype(&self, expr: TypeExpr) -> Option<Keytype> {
        match self.type_expr(expr) {
            TypeExprData::Name(key) => Keytype::from_intern(*key),
            TypeExprData::Items(_) => Some(Keytype::Struct),
            TypeExprData::List(list) => self.type_expr_index(expr, 0).and_then(|t| match self.type_expr(t) {
                TypeExprData::Name(key) => Keytype::from_intern(*key),
                TypeExprData::Items(_) => if list.len() == 2 { Some(Keytype::Func) } else { None },
                _ => None
            }),
            _ => None
        }
    }

    pub fn type_expr_len(&self, expr: TypeExpr) -> usize {
        let b = if self.type_expr_has_explicit_keytype(expr) { 0 } else { 1 };
        match self.type_expr(expr) {
            TypeExprData::List(list) => list.len() + b,
            _ => 1
        }
    }

    // struct, func type exprs
    // returns nonsense if used on others

    pub fn type_expr_items(&self, expr: TypeExpr) -> Option<ItemList> {
        match self.type_expr(expr) {
            TypeExprData::Items(items) => Some(*items),
            TypeExprData::List(_) => {
                let index = if self.type_expr_has_explicit_keytype(expr) { 1 } else { 0 };
                self.type_expr_index(expr, index).and_then(|t| match self.type_expr(t) {
                    TypeExprData::Items(items) => Some(*items),
                    _ => None
                })
            }
            _ => None,
        }
    }

    pub fn type_expr_returns(&self, expr: TypeExpr) -> Option<TypeExpr> {
        let index = if self.type_expr_has_explicit_keytype(expr) { 2 } else { 1 };
        match self.type_expr(expr) {
            TypeExprData::List(_) => self.type_expr_index(expr, index),
            _ => None,
        }
    }

    // arr, ptr type exprs
    // returns nonsense if used on others

    pub fn type_expr_base_type(&self, expr: TypeExpr) -> Option<TypeExpr> {
        let index = if self.type_expr_has_explicit_keytype(expr) { 1 } else { 0 };
        match self.type_expr(expr) {
            TypeExprData::List(_) => self.type_expr_index(expr, index),
            _ => None,
        }
    }

    pub fn type_expr_bound(&self, expr: TypeExpr) -> Option<TypeExpr> {
        let index = if self.type_expr_has_explicit_keytype(expr) { 2 } else { 1 };
        match self.type_expr(expr) {
            TypeExprData::List(_) => self.type_expr_index(expr, index),
            _ => None,
        }
    }

    pub fn pop_type_expr(&mut self, expr: TypeExpr) -> TypeExprData {
        debug_assert!(expr.0 as usize == self.type_exprs.len() - 1);
        self.type_exprs.remove(expr.0 as usize)
    }
}
