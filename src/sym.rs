// use std::collections::hash_map::HashMap;
use std::collections::hash_map::Entry;

use crate::Compiler;
use crate::Global;
use crate::Value;
use crate::Type;
use crate::TypeKind;
use crate::Intern;

use crate::eval_type;

use crate::error;

use crate::ast::*;
use crate::smallvec::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    Type,
    Constant,
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
    pub state: State,
    pub decl: Decl,
    pub ty: Type,
    pub name: Intern,
}

fn check_duplicate_items(ast: &Ast, items: ItemList) -> Option<Intern> {
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

fn resolve_callable(ctx: &mut Compiler, decl: Decl, data: &CallableDecl) -> Symbol {
    let mut decl_types = SmallVec::new();
    for param in data.params {
        let item = ctx.ast.item(param);
        let ty = eval_type(ctx, item.expr);
        decl_types.push(ty);
    }
    if let Some(returns) = data.returns {
        let return_type = eval_type(ctx, returns);
        decl_types.push(return_type);
    }
    let ty = ctx.types.make_anonymous(TypeKind::Func, &decl_types);
    let name = data.name;
    if let Some(dup) = check_duplicate_items(&ctx.ast, data.params) {
        error!(ctx, data.pos, "duplicate parameter name {} in func {}", ctx.str(dup), ctx.str(name));
    }
    Symbol { kind: Kind::Constant, state: State::Resolved, decl, ty, name }
}

fn resolve_struct(ctx: &mut Compiler, decl: Decl, data: &StructDecl) -> Symbol {
    let name = data.name;
    let ty = ctx.types.make(TypeKind::Struct, name);
    let resolving = Symbol { kind: Kind::Type, state: State::Resolving, decl, ty, name };
    let existing_symbol = ctx.symbols.insert(name, resolving);
    assert!(matches!(existing_symbol, None));
    for field in data.fields {
        let item = ctx.ast.item(field);
        let item_type = eval_type(ctx, item.expr);
        ctx.types.add_item_to_type(ty, item.name, item_type);
    }
    ctx.types.complete_type(ty);
    if data.fields.len() == 0 {
        error!(ctx, data.pos, "struct {} has no fields", ctx.str(data.name));
    }
    if let Some(dup) = check_duplicate_items(&ctx.ast, data.fields) {
        error!(ctx, data.pos, "{} has already been defined on struct {}", ctx.str(dup), ctx.str(data.name));
    }
    Symbol { state: State::Resolved, ..resolving }
}

fn resolve_decl(ctx: &mut Compiler, decl: Decl) -> Symbol {
    let result = match *ctx.ast.decl(decl) {
        DeclData::Callable(func) => {
            let sym = resolve_callable(ctx, decl, &func);
            // todo: move out of here or remove entirely
            match ctx.globals.entry(func.name) {
                Entry::Occupied(_) => { error!(ctx, func.pos, "func {} has already been defined", ctx.str(func.name)); }
                Entry::Vacant(entry) => { entry.insert(Global { decl: decl, value: Value::Ident(func.name), ty: sym.ty }); }
            }
            sym
        }
        DeclData::Struct(data) => {
            resolve_struct(ctx, decl, &data)
        }
    };
    let name = ctx.ast.decl(decl).name();
    ctx.symbols.insert(name, result);
    result
}

fn resolve(ctx: &mut Compiler, name: Intern) -> Symbol {
    if let Some(&sym) = ctx.symbols.get(&name) {
        if sym.state != State::Resolved {
            error!(ctx, 0, "{} is circular", ctx.str(name));
        }
        sym
    } else if let Some(decl) = ctx.ast.lookup_decl(name) {
        resolve_decl(ctx, decl)
    } else {
        // todo: source location of where we are resolving from. push/restore
        error!(ctx, 0, "cannot find {}", ctx.str(name));
        ctx.symbols.get(&Intern(0)).copied().unwrap()
    }
}

pub fn resolve_all(ctx: &mut Compiler) {
    for decl in ctx.ast.decl_list() {
        let name = ctx.ast.decl(decl).name();
        if let Some(sym) = ctx.symbols.get(&name) {
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

pub fn resolve_type(ctx: &mut Compiler, name: Intern) -> Symbol {
    let sym = resolve(ctx, name);
    if sym.kind != Kind::Type {
        // todo: source location of where we are resolving from. push/restore
        error!(ctx, 0, "{} is not a type", ctx.str(name));
    }
    sym
}

pub fn builtin(ctx: &mut Compiler, name: Intern, ty: Type) {
    ctx.symbols.insert(name, Symbol { kind: Kind::Type, state: State::Resolved, decl: Decl::default(), ty, name });
}
