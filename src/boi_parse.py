
"""
GRAMMAR in BNF form:

<id> ::= [a-zA-Z][a-zA-Z0-9_]*
<float> ::= [-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?
<value> ::= <float>
<expr> ::=  <function_call_expr>
        |   <lambda_expr>
        |   <let_expr>
        |   <if_expr>
        |   <value>
        |   <var> 
        |   "(" <additive_expr> ")"
<multiplicative_expr> ::= <expr> ( ("*" | "/") <rhs: expr> )*
<pow_expr> ::= <multiplicative_expr> ( ("*" | "/") <multiplicative_expr> )*
<additive_expr> ::= <pow_expr> ( ("+" | "-") <pow_expr> )*

<base_expr> ::= <additive_expr>

<bool_expr> ::= <comparison_expr>
            |   <condition_expr>
<comparison_expr> ::= <base_expr> (">" | ">=" | "<" | "<=" | "=" | "!=") <base_expr>
<condition_expr> ::= <base_expr>

<function_call_expr> ::= <id> ~ ( "(" ")" | <base_expr>+ )
<let_expr> ::= "let" <id> "=" <base_expr> "in" <base_expr>
<lambda_expr> ::= "let" <id> ( "(" ")" | <id>+ ) "=" <base_expr>
<if_expr> ::= "if" <bool_expr> "then" <base_expr> "else" <base_expr>
<function> ::= <lambda_expr>

<program> ::= ( <function> | base_expr )*

"""