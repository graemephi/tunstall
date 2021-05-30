use std::convert::TryFrom;

use crate::Intern;
use crate::parse::TokenKind;

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
pub struct Expr(u32);

define_node_list!(Expr, ExprList, ExprListIter);

#[derive(Clone, Copy, Debug)]
pub enum ExprData {
    Int(usize),
    Float32(f32),
    Float64(f64),
    Name(Intern),
    Unary(TokenKind, Expr),
    Binary(TokenKind, Expr, Expr),
    Ternary(Expr, Expr, Expr),
    Call(Expr, ExprList),
    Cast(Expr, TypeExpr)
}

impl std::fmt::Display for ExprData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            ExprData::Int(a) => write!(f, "integer ({})", a),
            ExprData::Name(_) => write!(f, "identifier"),
            _ => write!(f, "expression"),
        }
    }
}

struct ExprAux {
    pos: usize
}

#[derive(Clone, Copy, Debug)]
pub enum TypeExpr {
    Infer,
    Name(Intern)
}

#[derive(Clone, Copy, Debug)]
pub struct Stmt(u32);
define_node_list!(Stmt, StmtList, StmtListIter);

#[derive(Clone, Copy, Debug)]
pub struct SwitchCase(u32);
define_node_list!(SwitchCase, SwitchCaseList, SwitchCaseListIter);

#[derive(Clone, Copy, Debug)]
pub enum SwitchCaseData {
    Cases(ExprList, StmtList),
    Else(StmtList),
}

#[derive(Clone, Copy, Debug)]
pub enum StmtData {
    Block(StmtList),
    Return(Option<Expr>),
    Break,
    Continue,
    If(Expr, StmtList, StmtList),
    While(Expr, StmtList),
    For(Option<Stmt>, Expr, Option<Stmt>, StmtList),
    Switch(Expr, SwitchCaseList),
    Do(Expr, StmtList),
    Expr(Expr),
    Assign(Expr, Expr),
    VarDecl(TypeExpr, Expr, Expr)
}

pub enum CallableKind {
    Function,
    // Procedure
}

// An Item is a named declaration of a type, as in function parameters and
// struct fields.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct Item(u32);

#[derive(Clone, Copy, Debug)]
pub struct ItemData {
    pub name: Intern,
    pub expr: TypeExpr,
}

define_node_list!(Item, ItemList, ItemListIter);

pub struct CallableDecl {
    pub kind: CallableKind,
    pub pos: usize,
    pub name: Intern,
    pub params: ItemList,
    pub returns: Option<TypeExpr>,
    pub body: StmtList
}

pub struct Ast {
    exprs: Vec<ExprData>,
    exprs_aux: Vec<ExprAux>,
    stmts: Vec<StmtData>,
    cases: Vec<SwitchCaseData>,
    items: Vec<ItemData>,
    decls: Vec<CallableDecl>,
}

impl Ast {
    pub fn new() -> Ast {
        Ast {
            exprs: vec![ExprData::Int(0)],
            exprs_aux: vec![ExprAux { pos: 0}],
            stmts: vec![StmtData::Return(None)],
            cases: Vec::new(),
            items: Vec::new(),
            decls: Vec::new(),
        }
    }

    pub fn expr_data(&self, expr: Expr) -> ExprData {
        self.exprs[expr.0 as usize]
    }

    pub fn expr_source_position(&self, expr: Expr) -> usize {
        self.exprs_aux[expr.0 as usize].pos
    }

    fn push_expr(&mut self, pos: usize, data: ExprData) -> Expr {
        let result = u32::try_from(self.exprs.len()).expect("Program too big!");
        self.exprs.push(data);
        self.exprs_aux.push(ExprAux { pos });
        Expr(result)
    }

    pub fn push_exprs(&mut self, data: &[ExprData]) -> ExprList {
        let begin = u32::try_from(self.exprs.len()).expect("Program too big!");
        self.exprs.extend_from_slice(data);
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

    pub fn pop_expr(&mut self, expr: Expr) -> ExprData {
        debug_assert!(expr.0 as usize == self.exprs.len() - 1);
        self.exprs.remove(expr.0 as usize)
    }


    pub fn stmt_data(&self, stmt: Stmt) -> StmtData {
        self.stmts[stmt.0 as usize]
    }

    fn push_stmt(&mut self, data: StmtData) -> Stmt {
        let index = u32::try_from(self.stmts.len()).expect("Program too big!");
        self.stmts.push(data);
        Stmt(index)
    }

    pub fn push_stmts(&mut self, data: &[StmtData]) -> StmtList {
        let begin = u32::try_from(self.stmts.len()).expect("Program too big!");
        self.stmts.extend_from_slice(data);
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

    pub fn push_stmt_if(&mut self, cond: Expr, then_stmt: StmtList, else_stmt: StmtList) -> Stmt {
        self.push_stmt(StmtData::If(cond, then_stmt, else_stmt))
    }

    pub fn push_stmt_for(&mut self, pre: Option<Stmt>, cond: Expr, post: Option<Stmt>, body: StmtList) -> Stmt {
        self.push_stmt(StmtData::For(pre, cond, post, body))
    }

    pub fn push_stmt_while(&mut self, cond: Expr, body: StmtList) -> Stmt {
        self.push_stmt(StmtData::While(cond, body))
    }

    pub fn push_stmt_do(&mut self, cond: Expr, body: StmtList) -> Stmt {
        self.push_stmt(StmtData::Do(cond, body))
    }

    pub fn push_stmt_switch(&mut self, control: Expr, cases: &[SwitchCaseData]) -> Stmt {
        let begin = u32::try_from(self.cases.len()).expect("Program too big!");
        self.cases.extend_from_slice(cases);
        let end = u32::try_from(self.cases.len()).expect("Program too big!");
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
        self.stmts.remove(stmt.0 as usize)
    }

    pub fn switch_case_data(&self, case: SwitchCase) -> SwitchCaseData {
        self.cases[case.0 as usize]
    }


    pub fn push_items(&mut self, data: &[ItemData]) -> ItemList {
        let begin = u32::try_from(self.items.len()).expect("Program too big!");
        self.items.extend_from_slice(data);
        let end = u32::try_from(self.items.len()).expect("Program too big!");
        ItemList { begin, end }
    }

    pub fn items(&self, items: ItemList) -> &[ItemData] {
        &self.items[items.as_range()]
    }


    pub fn push_callable_decl(&mut self, decl: CallableDecl) {
        self.decls.push(decl);
    }

    pub fn decls(&self) -> &[CallableDecl] {
        &self.decls
    }
}
