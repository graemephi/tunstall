use std::collections::hash_map::HashMap;

use crate::Compiler;
use crate::Type;
use crate::TypeKind;
use crate::Intern;

use crate::eval_type;

use crate::error;

use crate::ast::*;
use crate::smallvec::*;
use crate::parse::Keytype;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Kind {
    Type,
    Value,
    // Variable
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum State {
    Resolving,
    Resolved,
}

#[derive(Clone, Copy, Debug)]
pub struct Symbol {
    pub kind: Kind,
    pub name: Intern,
    pub state: State,
    pub expr: TypeExpr,
    pub ty: Type,
}

#[derive(Debug, Default)]
pub struct Symbols {
    table: HashMap<(Kind, Intern), Symbol>,
}

impl Symbols {
    fn insert_new(&mut self, sym: Symbol) {
        let existing_symbol = self.table.insert((sym.kind, sym.name), sym);
        assert!(matches!(existing_symbol, None));
    }

    fn update(&mut self, sym: Symbol) -> &Symbol {
        match self.table.entry((sym.kind, sym.name)) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                e.insert(sym);
                e.into_mut()
            },
            _ => unreachable!()
        }
    }

    fn get(&self, kind: Kind, name: Intern) -> Option<&Symbol> {
        self.table.get(&(kind, name))
    }
}

fn find_first_duplicate(ast: &Ast, items: ItemList) -> Option<Intern> {
    let items = ast.items(items);
    for i in 0..items.len() {
        for j in i+1..items.len() {
            if items[i].name == items[j].name {
                return Some(items[i].name);
            }
        }
    }
    None
}

fn resolve_callable(ctx: &mut Compiler, name: Intern, expr: TypeExpr) -> &Symbol {
    ctx.symbols.insert_new(Symbol { kind: Kind::Value, state: State::Resolving, expr, ty: Type::None, name });
    let mut decl_types = SmallVec::new();
    let kind = ctx.ast.type_expr_keytype(expr);
    let params = ctx.ast.type_expr_items(expr);
    let returns = ctx.ast.type_expr_returns(expr);
    if let Some(Keytype::Func)|Some(Keytype::Proc) = kind {
        if let Some(params) = params {
            for param in params {
                let item = ctx.ast.item(param);
                let ty = eval_type(ctx, item.expr);
                decl_types.push(ty);
            }

            if let Some(returns) = returns {
                let return_type = eval_type(ctx, returns);
                decl_types.push(return_type);
            } else {
                decl_types.push(Type::None);

                if let Some(Keytype::Func) = kind {
                    // todo: type expr pos
                    error!(ctx, 0, "func without return type");
                }
            }
            if let Some(dup) = find_first_duplicate(&ctx.ast, params) {
                error!(ctx, 0, "duplicate parameter name {} in func {}", ctx.str(dup), ctx.str(name));
            }
        } else {
            error!(ctx, 0, "{} without parameter list", kind.unwrap())
        }
    } else {
        panic!("non-callable type in callable decl");
    }
    let ty = ctx.types.signature(TypeKind::Callable, &decl_types, kind == Some(Keytype::Proc));
    ctx.symbols.update(Symbol { kind: Kind::Value, state: State::Resolved, expr, ty, name })
}

fn resolve_struct(ctx: &mut Compiler, name: Intern, expr: TypeExpr) -> &Symbol {
    let ty = ctx.types.strukt(name);
    let resolving = Symbol { kind: Kind::Type, state: State::Resolving, expr, ty, name };
    ctx.symbols.insert_new(resolving);
    if let Some(fields) = ctx.ast.type_expr_items(expr) {
        for field in fields {
            let item = ctx.ast.item(field);
            let item_type = eval_type(ctx, item.expr);
            ctx.types.add_item_to_type(ty, item.name, item_type);
        }
        if fields.len() == 0 {
            // todo: type expr pos
            error!(ctx, 0, "struct {} has no fields", ctx.str(name));
        }
        if let Some(dup) = find_first_duplicate(&ctx.ast, fields) {
            error!(ctx, 0, "{} has already been defined on struct {}", ctx.str(dup), ctx.str(name));
        }
    } else {
        error!(ctx, 0, "struct {} has no fields", ctx.str(name));
    }
    ctx.types.complete_type(ty);
    ctx.symbols.update(Symbol { state: State::Resolved, ..resolving })
}

pub fn resolve_anonymous_struct(ctx: &mut Compiler, fields: ItemList) -> Type {
    // Anonymous structs do not have a symbol, but we still have to resolve their fields.
    let mut decl_types = SmallVec::new();
    if fields.is_nonempty() {
        for field in fields {
            let item = ctx.ast.item(field);
            let item_type = eval_type(ctx, item.expr);
            decl_types.push((item.name, item_type));
        }
        if fields.len() == 0 {
            // todo: type expr pos
            error!(ctx, 0, "anonymous struct has no fields");
        }
        if let Some(dup) = find_first_duplicate(&ctx.ast, fields) {
            error!(ctx, 0, "{} has already been defined on anonymous struct", ctx.str(dup));
        }
    } else {
        error!(ctx, 0, "anonymous struct has no fields");
    }
    ctx.types.tuple(TypeKind::Struct, &decl_types)
}

fn resolve_decl(ctx: &mut Compiler, decl: Decl) -> &Symbol {
    let result = match *ctx.ast.decl(decl) {
        DeclData::Callable(func) => resolve_callable(ctx, func.name, func.expr),
        DeclData::Struct(strukt) => resolve_struct(ctx, strukt.name, strukt.expr),
        DeclData::Var(_) => unreachable!(),
    };
    result
}

fn resolve(ctx: &mut Compiler, kind: Kind, name: Intern) -> Option<&Symbol> {
    if let Some(sym) = ctx.symbols.get(kind, name) {
        if sym.state == State::Resolved {
            return ctx.symbols.get(kind, name);
        }

        error!(ctx, 0, "{} is circular", ctx.str(name));
        None
    } else if let Some(decl) = ctx.ast.lookup_decl(name) {
        match *ctx.ast.decl(decl) {
            DeclData::Callable(func) if kind == Kind::Value => Some(resolve_callable(ctx, func.name, func.expr)),
            DeclData::Struct(strukt) if kind == Kind::Type => Some(resolve_struct(ctx, strukt.name, strukt.expr)),
            DeclData::Var(_) => todo!(),
            _ => None
        }
    } else {
        // todo: source location of where we are resolving from. push/restore
        error!(ctx, 0, "cannot find {}", ctx.str(name));
        None
    }
}

pub fn resolve_all(ctx: &mut Compiler) {
    for decl in ctx.ast.decl_list() {
        let kind = if let DeclData::Struct(_) = ctx.ast.decl(decl) {
            Kind::Type
        } else {
            Kind::Value
        };
        let name = ctx.ast.decl(decl).name();
        if let Some(sym) = ctx.symbols.get(kind, name) {
            if sym.state != State::Resolved {
                let pos = ctx.ast.decl(decl).pos();
                error!(ctx, pos, "{} is circular", ctx.str(name));
                break;
            }
        } else {
            resolve_decl(ctx, decl);
        }
    }
}

pub fn resolve_type(ctx: &mut Compiler, name: Intern) -> Type {
    if let Some(&Symbol{ kind: Kind::Type, ty, .. }) = resolve(ctx, Kind::Type, name) {
        return ty;
    }
    // todo: source location of where we are resolving from. push/restore
    error!(ctx, 0, "cannot find type {}", ctx.str(name));
    Type::None
}

pub fn builtin(ctx: &mut Compiler, name: Intern, ty: Type) {
    ctx.symbols.insert_new(Symbol { kind: Kind::Type, name, state: State::Resolved, expr: TypeExpr::Infer, ty });
}

pub fn lookup_type(ctx: &Compiler, name: Intern) -> Option<&Symbol> {
    ctx.symbols.get(Kind::Type, name)
}

pub fn lookup_value(ctx: &Compiler, name: Intern) -> Option<&Symbol> {
    ctx.symbols.get(Kind::Value, name)
}
