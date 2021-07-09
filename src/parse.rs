use crate::Compiler;
use crate::Expr;
use crate::Stmt;
use crate::StmtList;
use crate::Interns;
use crate::Intern;

use crate::ast::*;
use crate::smallvec::*;

macro_rules! parse_error {
    ($parser: expr, $($fmt: expr),*) => {{
        $parser.ctx.error($parser.pos(), format!($($fmt),*));
        $parser.fail();
        std::default::Default::default()
    }}
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TokenKind {
    Eof,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    BitNeg,
    Not,
    BitAnd,
    BitOr,
    BitXor,
    LogicAnd,
    LogicOr,
    Lt,
    Gt,
    Eq,
    NEq,
    LtEq,
    GtEq,
    LShift,
    RShift,
    Assign,
    Dot,
    Comma,
    ColonAssign,
    Colon,
    ColonColon,
    Semicolon,
    Question,
    Keyword,
    Name,
    Int,
    Float,
    Char,
    String,
    Comment,
}

impl std::fmt::Display for TokenKind {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", match *self {
            TokenKind::Eof => "end of file",
            TokenKind::LogicAnd => "&&",
            TokenKind::LogicOr => "||",
            TokenKind::Eq => "==",
            TokenKind::NEq => "!=",
            TokenKind::LtEq => "<=",
            TokenKind::GtEq => ">=",
            TokenKind::LShift => "<<",
            TokenKind::RShift => ">>",
            TokenKind::ColonAssign => ":=",
            TokenKind::Comment => "//",
            TokenKind::LParen => "(",
            TokenKind::RParen => ")",
            TokenKind::LBracket => "[",
            TokenKind::RBracket => "]",
            TokenKind::LBrace => "{",
            TokenKind::RBrace => "}",
            TokenKind::Add => "+",
            TokenKind::Sub => "-",
            TokenKind::Mul => "*",
            TokenKind::Div => "/",
            TokenKind::Mod => "%",
            TokenKind::BitNeg => "~",
            TokenKind::Not => "!",
            TokenKind::BitAnd => "&",
            TokenKind::BitOr => "|",
            TokenKind::BitXor => "^",
            TokenKind::Lt => "<",
            TokenKind::Gt => ">",
            TokenKind::Assign => "=",
            TokenKind::Dot => ".",
            TokenKind::Comma => ",",
            TokenKind::Colon => ":",
            TokenKind::ColonColon => "::",
            TokenKind::Semicolon => ";",
            TokenKind::Question => "?",
            TokenKind::Char => "char",
            TokenKind::String => "string",
            TokenKind::Int => "integer",
            TokenKind::Keyword => "keyword",
            TokenKind::Float => "float",
            TokenKind::Name => "name"
        })
    }
}

#[derive(Copy, Clone, Debug)]
struct Token<'a> {
    str: &'a str,
    source_index: usize,
    kind: TokenKind,
    intern: Intern,
    keyword: Option<Keyword>,
    suffix: Option<u8>
}

impl Token<'_> {
    fn eof() -> Token<'static> {
        Token {
            str: "",
            source_index: 0,
            kind: TokenKind::Eof,
            intern: Interns::empty_string(),
            keyword: None,
            suffix: None
        }
    }
}

impl std::default::Default for Token<'_> {
    fn default() -> Token<'static> {
        Token::eof()
    }
}

impl std::fmt::Display for Token<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let result = self.kind.fmt(f);
        use TokenKind::*;
        match self.kind {
            String|Int|Float|Keyword|Name => result.and_then(|_| write!(f, " {}", self.str)),
            _ => result,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Keyword {
    Return,
    Break,
    Continue,
    If,
    Else,
    While,
    For,
    Do,
    Switch,
    Func,
    Struct
}

impl Keyword {
    pub const fn from_u32(v: u32) -> Option<Keyword> {
        match v {
            1  => Some(Keyword::Return),
            2  => Some(Keyword::Break),
            3  => Some(Keyword::Continue),
            4  => Some(Keyword::If),
            5  => Some(Keyword::Else),
            6  => Some(Keyword::While),
            7  => Some(Keyword::For),
            8  => Some(Keyword::Do),
            9  => Some(Keyword::Switch),
            10 => Some(Keyword::Func),
            11 => Some(Keyword::Struct),
            _  => None,
        }
    }

    pub const fn to_str(self) -> &'static str {
        match self {
            Keyword::Return => "return",
            Keyword::Break => "break",
            Keyword::Continue => "continue",
            Keyword::If => "if",
            Keyword::Else => "else",
            Keyword::While => "while",
            Keyword::For => "for",
            Keyword::Do => "do",
            Keyword::Switch => "switch",
            Keyword::Func => "func",
            Keyword::Struct => "struct"
        }
    }
}

impl std::fmt::Display for Keyword {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.to_str().fmt(f)
    }
}

/// The index of `substr` within `str`. `str` and `substr` must point to the same allocation.
unsafe fn offset_from(str: &str, substr: &str) -> usize {
    let a = str.as_ptr();
    let b = substr.as_ptr();
    assert!(b <= a.offset(str.len() as isize));
    b.offset_from(a) as usize
}

struct Parser<'c, 'a> {
    ast: Ast,
    ctx: &'c mut Compiler,
    str: &'a str,
    p: &'a str,
    token: Token<'a>,
}

fn one_after_escaped_position(str: &str, delim: u8) -> usize {
    str.as_bytes()
       .windows(2)
       .position(|w| matches!(w, &[a, b] if a != b'\\' && b == delim))
       .map(|v| v + 1)
       .unwrap_or(!0)
}

fn is_valid_identifier_character(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '\''
}

impl<'c, 'a> Parser<'_, '_> {
    fn new(ctx: &'c mut Compiler, str: &'a str) -> Parser<'c, 'a> {
        let mut result = Parser { ast: Ast::new(), ctx: ctx, str: str, p: str, token: Token::eof() };
        result.next_token();
        result
    }

    fn pos(&self) -> usize {
        unsafe { offset_from(self.str, self.p) }
    }

    fn fail(&mut self) {
        self.p = &self.str[..0];
        self.token = Token::eof();
    }

    fn next_token(&mut self) -> Token {
        let mut result = Token { source_index: self.pos(), ..Token::eof() };

        let p = self.p.trim_start();
        let len = match *p.as_bytes() {
            [] => 0,
            [b'&', b'&', ..] => { result.kind = TokenKind::LogicAnd; 2 }
            [b'|', b'|', ..] => { result.kind = TokenKind::LogicOr; 2 }
            [b'=', b'=', ..] => { result.kind = TokenKind::Eq; 2 }
            [b'!', b'=', ..] => { result.kind = TokenKind::NEq; 2 }
            [b'<', b'=', ..] => { result.kind = TokenKind::LtEq; 2 }
            [b'>', b'=', ..] => { result.kind = TokenKind::GtEq; 2 }
            [b'<', b'<', ..] => { result.kind = TokenKind::LShift; 2 }
            [b'>', b'>', ..] => { result.kind = TokenKind::RShift; 2 }
            [b':', b':', ..] => { result.kind = TokenKind::ColonColon; 2 }
            [b':', b'=', ..] => { result.kind = TokenKind::ColonAssign; 2 }
            [b'/', b'/', ..] => { result.kind = TokenKind::Comment; one_after_escaped_position(p, b'\n') }
            [b'(', ..] => { result.kind = TokenKind::LParen; 1 }
            [b')', ..] => { result.kind = TokenKind::RParen; 1 }
            [b'[', ..] => { result.kind = TokenKind::LBracket; 1 }
            [b']', ..] => { result.kind = TokenKind::RBracket; 1 }
            [b'{', ..] => { result.kind = TokenKind::LBrace; 1 }
            [b'}', ..] => { result.kind = TokenKind::RBrace; 1 }
            [b'+', ..] => { result.kind = TokenKind::Add; 1 }
            [b'-', ..] => { result.kind = TokenKind::Sub; 1 }
            [b'*', ..] => { result.kind = TokenKind::Mul; 1 }
            [b'/', ..] => { result.kind = TokenKind::Div; 1 }
            [b'%', ..] => { result.kind = TokenKind::Mod; 1 }
            [b'~', ..] => { result.kind = TokenKind::BitNeg; 1 }
            [b'!', ..] => { result.kind = TokenKind::Not; 1 }
            [b'&', ..] => { result.kind = TokenKind::BitAnd; 1 }
            [b'|', ..] => { result.kind = TokenKind::BitOr; 1 }
            [b'^', ..] => { result.kind = TokenKind::BitXor; 1 }
            [b'<', ..] => { result.kind = TokenKind::Lt; 1 }
            [b'>', ..] => { result.kind = TokenKind::Gt; 1 }
            [b'=', ..] => { result.kind = TokenKind::Assign; 1 }
            [b'.', ..] => { result.kind = TokenKind::Dot; 1 }
            [b',', ..] => { result.kind = TokenKind::Comma; 1 }
            [b':', ..] => { result.kind = TokenKind::Colon; 1 }
            [b';', ..] => { result.kind = TokenKind::Semicolon; 1 }
            [b'?', ..] => { result.kind = TokenKind::Question; 1 }
            [b'\'', ..] => { result.kind = TokenKind::Char; one_after_escaped_position(p, b'\'') }
            [b'\"', ..] => { result.kind = TokenKind::String; one_after_escaped_position(p, b'\"') }
            // todo: 0x, 0b
            [b'0'..= b'9', ..] => {
                fn is_digit_or_underscore(&b: &u8) -> bool {
                    b == b'_' || char::from(b).is_digit(10)
                }
                let mut digits = p.bytes().enumerate();
                let mut scan = || digits.by_ref().find(|(_,c)| !is_digit_or_underscore(c))
                                                 .unwrap_or((p.len(), b'\0'));
                match scan() {
                    (_, b'.') => {
                        match scan() {
                            (n, b'd') => {
                                result.kind = TokenKind::Float;
                                result.suffix = Some(b'd');
                                n
                            }
                            (n, _) => { result.kind = TokenKind::Float; n }
                        }
                    },
                    (n, _) => { result.kind = TokenKind::Int; n }
                }
            },
            [..] => {
                let c = p.chars().nth(0).unwrap();
                if is_valid_identifier_character(c) {
                    let q = p.split(|c: char| !is_valid_identifier_character(c)).next().unwrap();
                    result.intern = self.ctx.intern(q);
                    result.keyword = Keyword::from_u32(result.intern.0);
                    result.kind = match result.keyword { Some(_) => TokenKind::Keyword, None => TokenKind::Name };
                    q.len()
                } else {
                    return parse_error!(self, "unexpected {}", c);
                }
            }
        };

        if len == !0 && result.kind != TokenKind::Comment {
            return parse_error!(self, "{}", match result.kind {
                TokenKind::Char => "unclosed \'",
                TokenKind::String => "unclosed \"",
                _ => unreachable!()
            });
        }

        let (chomped, rest) = p.split_at(len);
        debug_assert!(self.p != rest || result.kind == TokenKind::Eof);

        result.str = chomped;

        self.p = if result.suffix.is_none() { rest } else { &rest[1..] };
        self.token = result;

        result
    }

    fn is_eof(&self) -> bool {
        self.token.kind == TokenKind::Eof
    }

    fn not(&self, kind: TokenKind) -> bool {
        !self.is_eof() && self.token.kind != kind
    }

    fn comment(&mut self) {
        while self.token.kind == TokenKind::Comment {
            self.next_token();
        }
    }

    fn try_token(&mut self, kind: TokenKind) -> Option<Token> {
        self.comment();
        if self.token.kind == kind {
            let result = Some(self.token);
            self.next_token();
            return result;
        }

        None
    }

    fn token(&mut self, kind: TokenKind) -> Token {
        self.comment();
        if self.token.kind == kind {
            let result = self.token;
            self.next_token();
            return result;
        }

        parse_error!(self, "expected {}, found {}", kind, self.token)
    }

    fn try_keyword(&mut self, keyword: Keyword) -> Option<Token> {
        self.comment();
        if self.token.kind == TokenKind::Keyword && self.token.keyword == Some(keyword) {
            let result = Some(self.token);
            self.next_token();
            return result;
        }

        None
    }

    fn keyword(&mut self, keyword: Keyword) -> Token {
        self.comment();
        if self.token.kind == TokenKind::Keyword && self.token.keyword == Some(keyword) {
            let result = self.token;
            self.next_token();
            return result;
        }

        parse_error!(self, "expected {}, found {}", keyword, self.token)
    }

    fn name(&mut self) -> Intern {
        let result = self.token.intern;
        self.token(TokenKind::Name);
        result
    }

    fn type_expr(&mut self) -> TypeExpr {
        let name = self.name();

        use TokenKind::*;
        if matches!(self.token.kind, Name|LBracket|LParen) {
            let mut list = SmallVecN::<_, 8>::new();
            loop {
                match self.token.kind {
                    Name => {
                        let name = self.name();
                        list.push(TypeExprData::Name(name))
                    }
                    LBracket => {
                        self.token(LBracket);
                        let expr = self.expr();
                        self.token(RBracket);
                        list.push(TypeExprData::Expr(expr))
                    }
                    LParen => {
                        self.token(LParen);
                        let expr = self.type_expr();
                        self.token(RParen);
                        list.push(self.ast.pop_type_expr(expr))
                    }
                    _ => break
                }
            }
            self.ast.push_type_expr_list(name, &list)
        } else {
            self.ast.push_type_expr_name(name)
        }
    }

    fn paren_expr(&mut self) -> Expr {
        self.token(TokenKind::LParen);
        let result = self.expr();
        self.token(TokenKind::RParen);
        result
    }

    fn expr_list(&mut self) -> ExprList {
        let mut list = SmallVec::new();
        let expr = self.expr();
        list.push(self.ast.pop_expr(expr));
        while self.try_token(TokenKind::Comma).is_some() {
            let expr = self.expr();
            list.push(self.ast.pop_expr(expr));
        }
        self.ast.push_exprs(&list)
    }

    fn base_expr(&mut self) -> Expr {
        use TokenKind::*;
        let mut result = match self.token.kind {
            LParen => {
                self.paren_expr()
            },
            Int => {
                let token = self.token;
                self.token(Int);
                let value: usize = token.str.parse().unwrap_or_else(|err| { parse_error!(self, "{}", err) });
                self.ast.push_expr_int(token.source_index, value)
            },
            Float => {
                let token = self.token;
                self.token(Float);
                if token.suffix == Some(b'd') {
                    let value: f64 = token.str.parse().unwrap_or_else(|err| { parse_error!(self, "{}", err) });
                    self.ast.push_expr_float64(token.source_index, value)
                } else {
                    let value: f32 = token.str.parse().unwrap_or_else(|err| { parse_error!(self, "{}", err) });
                    self.ast.push_expr_float32(token.source_index, value)
                }
            },
            Name => {
                let index = self.token.source_index;
                let name = self.name();
                self.ast.push_expr_name(index, name)
            },
            LBrace => {
                let pos = self.pos();
                self.token(LBrace);
                let mut fields = SmallVec::new();

                if self.not(RBrace) {
                    let path_or_value = self.expr();
                    if self.not(Assign) {
                        fields.push(CompoundFieldData { path: CompoundPath::Implicit, value: path_or_value });
                        if self.try_token(Comma).is_some() {
                            while self.not(RBrace) {
                                let value = self.expr();
                                fields.push(CompoundFieldData { path: CompoundPath::Implicit, value: value });
                                if self.try_token(Comma).is_none() {
                                    break;
                                }
                            }
                        }
                    } else {
                        self.token(Assign);
                        let value = self.expr();
                        fields.push(CompoundFieldData { path: CompoundPath::Path(path_or_value), value });
                        if self.try_token(Comma).is_some() {
                            while self.not(RBrace) {
                                let path = CompoundPath::Path(self.expr());
                                self.token(Assign);
                                let value = self.expr();
                                fields.push(CompoundFieldData { path, value });
                                if self.try_token(Comma).is_none() {
                                    break;
                                }
                            }
                        }
                    }
                }
                self.token(RBrace);
                self.ast.push_expr_compound(pos, &fields)
            }
            _ => {
                parse_error!(self, "Unexpected {}", self.token)
            }
        };

        loop {
            match self.token.kind {
                LParen => {
                    let pos = self.token(LParen).source_index;
                    let args = if self.token.kind != RParen {
                        self.expr_list()
                    } else {
                        ExprList::empty()
                    };
                    self.token(RParen);
                    result = self.ast.push_expr_call(pos, result, args);
                }
                LBracket => {
                    let pos = self.token(LBracket).source_index;
                    let index = self.expr();
                    self.token(RBracket);
                    result = self.ast.push_expr_index(pos, result, index);
                }
                Dot => {
                    let pos = self.token(Dot).source_index;
                    let field = self.name();
                    result = self.ast.push_expr_field(pos, result, field);
                }
                _ => break
            }
        }

        result
    }

    fn unary_expr(&mut self) -> Expr {
        use TokenKind::*;
        match self.token.kind {
            Sub => {
                let token = self.token;
                self.token(Sub);
                let operand = self.unary_expr();
                self.ast.push_expr_unary(token.source_index, token.kind, operand)
            },
            BitNeg => {
                let token = self.token;
                self.token(BitNeg);
                let operand = self.unary_expr();
                self.ast.push_expr_unary(token.source_index, token.kind, operand)
            },
            Not => {
                let token = self.token;
                self.token(Not);
                let operand = self.unary_expr();
                self.ast.push_expr_unary(token.source_index, token.kind, operand)
            },
            BitAnd => {
                let token = self.token;
                self.token(BitAnd);
                let operand = self.unary_expr();
                self.ast.push_expr_unary(token.source_index, token.kind, operand)
            },
            Mul => {
                let token = self.token;
                self.token(Mul);
                let operand = self.unary_expr();
                self.ast.push_expr_unary(token.source_index, token.kind, operand)
            },
            _ => {
                self.base_expr()
            }
        }
    }

    fn cast_expr(&mut self) -> Expr {
        let mut result = self.unary_expr();

        while self.try_token(TokenKind::Colon).is_some() {
            let token = self.token;
            let ty = self.type_expr();
            result = self.ast.push_expr_cast(token.source_index, result, ty);
        }

        result
    }

    fn mul_expr(&mut self) -> Expr {
        use TokenKind::*;
        let mut result = self.cast_expr();
        while matches!(self.token.kind, Mul|Div|Mod|BitAnd|LShift|RShift) {
            let token = self.token;
            self.next_token();
            let right = self.cast_expr();
            result = self.ast.push_expr_binary(token.source_index, token.kind, result, right);
        }
        result
    }

    fn add_expr(&mut self) -> Expr {
        use TokenKind::*;
        let mut result = self.mul_expr();
        while matches!(self.token.kind, Add|Sub|BitOr|BitXor) {
            let token = self.token;
            self.next_token();
            let right = self.mul_expr();
            result = self.ast.push_expr_binary(token.source_index, token.kind, result, right);
        }
        result
    }

    fn cmp_expr(&mut self) -> Expr {
        use TokenKind::*;
        let mut result = self.add_expr();
        while matches!(self.token.kind, Lt|Gt|Eq|NEq|LtEq|GtEq) {
            let token = self.token;
            self.next_token();
            let right = self.add_expr();
            result = self.ast.push_expr_binary(token.source_index, token.kind, result, right);
        }
        result
    }

    fn and_expr(&mut self) -> Expr {
        let mut result = self.cmp_expr();
        while matches!(self.token.kind, TokenKind::LogicAnd) {
            let token = self.token;
            self.next_token();
            let right = self.cmp_expr();
            result = self.ast.push_expr_binary(token.source_index, token.kind, result, right);
        }
        result
    }

    fn or_expr(&mut self) -> Expr {
        let mut result = self.and_expr();
        while matches!(self.token.kind, TokenKind::LogicOr) {
            let token = self.token;
            self.next_token();
            let right = self.and_expr();
            result = self.ast.push_expr_binary(token.source_index, token.kind, result, right);
        }
        result
    }

    fn ternary_expr(&mut self) -> Expr {
        let mut result = self.or_expr();
        if matches!(self.token.kind, TokenKind::Question) {
            let token = self.token;
            self.token(TokenKind::Question);
            let left = self.ternary_expr();
            self.token(TokenKind::ColonColon);
            let right = self.ternary_expr();
            result = self.ast.push_expr_ternary(token.source_index, result, left, right);
        }
        result
    }

    fn expr(&mut self) -> Expr {
        self.ternary_expr()
    }

    fn stmt_block(&mut self) -> StmtList {
        self.token(TokenKind::LBrace);
        let mut list = SmallVec::new();
        while !self.is_eof() && self.try_token(TokenKind::RBrace).is_none() {
            let stmt = self.stmt();
            list.push(self.ast.pop_stmt(stmt));
        }
        self.ast.push_stmts(&list)
    }

    fn simple_stmt(&mut self) -> Stmt {
        use TokenKind::*;
        let left = self.expr();
        let result = match self.token.kind {
            ColonAssign => {
                self.next_token();
                let right = self.expr();
                self.ast.push_stmt_vardecl(TypeExpr::Infer, left, right)
            }
            Assign => {
                self.next_token();
                // We do a bit of AST rewriting here to make : work as a universal type operator
                // thing. Take `a: int = 0;`. Conceptually, we expect a `:` after the expr `a`,
                // but `a: int` parses as Expr::Cast(a, int). We can immediately see this,
                // and so rewrite it as a variable declaration.
                match self.ast.expr(left) {
                    ExprData::Cast(expr, ty) => {
                        self.ast.pop_expr(left);
                        let right = self.expr();
                        self.ast.push_stmt_vardecl(ty, expr, right)
                    }
                    _ => {
                        let right = self.expr();
                        self.ast.push_stmt_assign(left, right)
                    }
                }
            }
            _ => {
                self.ast.push_stmt_expr(left)
            }
        };
        result
    }

    fn stmt(&mut self) -> Stmt {
        use self::Keyword::*;
        use TokenKind::*;
        match (self.token.kind, self.token.keyword) {
            (Keyword, Some(Return)) => {
                self.token(Keyword);
                let expr = if self.try_token(Semicolon).is_some() {
                    None
                } else {
                    let expr = self.expr();
                    self.token(Semicolon);
                    Some(expr)
                };
                self.ast.push_stmt_return(expr)
            },
            (Keyword, Some(Break)) => {
                self.token(Keyword);
                self.token(Semicolon);
                self.ast.push_stmt_break()
            },
            (Keyword, Some(Continue)) => {
                self.token(Keyword);
                self.token(Semicolon);
                self.ast.push_stmt_continue()
            },
            (LBrace, _) => {
                let list = self.stmt_block();
                self.ast.push_stmt_block(list)
            },
            (Keyword, Some(If)) => {
                self.token(Keyword);
                let expr = self.paren_expr();
                let then_stmt = self.stmt_block();
                let else_stmt = if self.try_keyword(Else).is_some() {
                    if matches!(self.token.keyword, Some(If)) {
                        let stmt = self.stmt();
                        StmtList::from(stmt)
                    } else {
                        self.stmt_block()
                    }
                } else {
                    StmtList::empty()
                };
                self.ast.push_stmt_if(expr, then_stmt, else_stmt)
            },
            (Keyword, Some(While)) => {
                self.token(Keyword);
                let cond = self.paren_expr();
                let body = self.stmt_block();
                self.ast.push_stmt_while(cond, body)
            },
            (Keyword, Some(For)) => {
                self.token(Keyword);
                self.token(LParen);
                let pre = if self.not(Semicolon) {
                    Some(self.simple_stmt())
                } else {
                    None
                };
                self.token(Semicolon);
                let cond = if self.not(Semicolon) {
                    Some(self.expr())
                } else {
                    None
                };
                self.token(Semicolon);
                let post = if self.not(RParen) {
                    Some(self.simple_stmt())
                } else {
                    None
                };
                self.token(RParen);
                let block = self.stmt_block();
                self.ast.push_stmt_for(pre, cond, post, block)
            },
            (Keyword, Some(Do)) => {
                self.token(Keyword);
                let body = self.stmt_block();
                self.keyword(While);
                let cond = self.paren_expr();
                self.token(Semicolon);
                self.ast.push_stmt_do(cond, body)
            },
            (Keyword, Some(Switch)) => {
                self.token(Keyword);
                self.token(LParen);
                let control = self.expr();
                self.token(RParen);
                self.token(LBrace);
                let mut cases = SmallVec::new();
                let mut seen_else = false;
                while self.not(RBrace) {
                    if matches!(self.token.keyword, Some(Else)) {
                        if seen_else {
                            parse_error!(self, "multiple else cases in switch")
                        } else {
                            seen_else = true;
                            self.token(Keyword);
                            let block = self.stmt_block();
                            cases.push(SwitchCaseData::Else(block));
                        }
                    } else {
                        let exprs = self.expr_list();
                        let block = self.stmt_block();
                        cases.push(SwitchCaseData::Cases(exprs, block));
                    }
                }
                self.token(RBrace);
                self.ast.push_stmt_switch(control, &cases)
            },
            _ => {
                let stmt = self.simple_stmt();
                self.token(Semicolon);
                stmt
            }
        }
    }

    fn items(&mut self) -> ItemList {
        let mut items = SmallVec::new();
        while self.token.kind == TokenKind::Name {
            let name = self.name();
            self.token(TokenKind::Colon);
            let expr = self.type_expr();
            items.push(ItemData { name, expr });
            self.try_token(TokenKind::Comma);
        }
        self.ast.push_items(&items)
    }

    fn func_decl(&mut self) {
        let kind = CallableKind::Function;
        let pos = self.pos();
        self.keyword(Keyword::Func);
        let name = self.name();
        self.token(TokenKind::LParen);
        let params;
        if let None = self.try_token(TokenKind::RParen) {
            params = self.items();
            self.token(TokenKind::RParen);
        } else {
            params = ItemList::empty();
        }
        self.token(TokenKind::Colon);
        let returns = Some(self.type_expr());
        let body = self.stmt_block();
        self.ast.push_callable_decl(CallableDecl { kind, pos, name, params, returns, body })
            .unwrap_or_else(|| parse_error!(self, "{} has already been defined", self.ctx.str(name)));
    }

    fn struct_decl(&mut self) {
        let pos = self.pos();
        self.keyword(Keyword::Struct);
        let name = self.name();
        self.token(TokenKind::LBrace);
        let fields = self.items();
        self.token(TokenKind::RBrace);
        self.ast.push_struct_decl(StructDecl { pos, name, fields })
            .unwrap_or_else(|| parse_error!(self, "{} has already been defined", self.ctx.str(name)));
    }
}

pub fn parse(ctx: &mut Compiler, str: &str) -> Ast {
    let mut parser = Parser::new(ctx, str);
    while !parser.is_eof() {
        parser.comment();
        match parser.token.keyword {
            Some(Keyword::Func) => parser.func_decl(),
            Some(Keyword::Struct) => parser.struct_decl(),
            Some(keyword) => parse_error!(parser, "Unexpected keyword '{}'", keyword),
            None => parse_error!(parser, "Unexpected {}", parser.token)
        }
    }
    parser.ast
}

#[allow(dead_code)]
pub fn parse_stmt(ctx: &mut Compiler, str: &str) -> (Ast, Stmt) {
    let mut parser = Parser::new(ctx, str);
    let stmt = parser.stmt();
    (parser.ast, stmt)
}

#[allow(dead_code)]
pub fn parse_expr(ctx: &mut Compiler, str: &str) -> (Ast, Expr) {
    let mut parser = Parser::new(ctx, str);
    let expr = parser.expr();
    (parser.ast, expr)
}
