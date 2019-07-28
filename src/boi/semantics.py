from boi.ast import *

STR_TO_MUL_OP = {"/": MultiplicativeExpr.DIVISION, "*": MultiplicativeExpr.MULTIPLICATION}
STR_TO_ADD_OP = {"+": AdditiveExpr.ADDITION, "-": AdditiveExpr.SUBTRACTION}
STR_TO_CMP_OP = {">": ComparisonExpr.GT, ">=": ComparisonExpr.GTE, "<": ComparisonExpr.LT,
                 "<=": ComparisonExpr.LTE, "=": ComparisonExpr.EQ, "<>": ComparisonExpr.NEQ,}


class BoiSemantics(object):
    def id(self, ast):
        assert type(ast) == str
        return Id(ast, Span.EMPTY)

    def var(self, ast):
        assert type(ast) == Id
        return ast

    def float(self, ast):
        assert type(ast) == str
        value = float(ast)
        return Value(value, Span.EMPTY)

    def value(self, ast):
        assert type(ast) == Value
        return ast

    def expr(self, ast):
        # assert issubclass(ast, Expr)
        return ast

    def multiplicative_expr(self, ast):
        assert len(ast) >= 1
        
        if len(ast) == 1:
            return ast[0]

        rhs = None
        op = None

        tail = ast[1:]
        tail.reverse()

        for chunk in tail:
            if rhs is None:
                rhs = chunk[1]
                op = STR_TO_MUL_OP[chunk[0]]
                continue

            rhs = MultiplicativeExpr(chunk[1], op, rhs, Span.EMPTY)

        return MultiplicativeExpr(ast[0], op, rhs, Span.EMPTY)

    def pow_expr(self, ast):
        assert len(ast) >= 1
        
        if len(ast) == 1:
            return ast[0]

        rhs = None

        tail = ast[1:]
        tail.reverse()

        for chunk in tail:
            if rhs is None:
                rhs = chunk
                continue

            rhs = PowExpr(chunk, rhs, Span.EMPTY)

        return PowExpr(ast[0], rhs, Span.EMPTY)

    def additive_expr(self, ast):
        assert len(ast) >= 1
        
        if len(ast) == 1:
            return ast[0]

        rhs = None
        op = None

        tail = ast[1:]
        tail.reverse()
        
        for chunk in tail:
            if rhs is None:
                rhs = chunk[1]
                op = STR_TO_ADD_OP[chunk[0]]
                continue

            rhs = AdditiveExpr(chunk[1], op, rhs, Span.EMPTY)

        return AdditiveExpr(ast[0], op, rhs, Span.EMPTY)

    def base_expr(self, ast):
        # assert issubclass(ast, Expr)
        return ast

    def bool_expr(self, ast):
        assert type(ast) in {ComparisonExpr, ConditionExpr}
        return ast

    def comparison_expr(self, ast):
        assert len(ast) == 3

        return ComparisonExpr(ast[0], STR_TO_CMP_OP[ast[1]], ast[2], Span.EMPTY)

    def condition_expr(self, ast):
        assert issubclass(ast, Expr)
        return ConditionExpr(ast, Span.EMPTY)

    def function_call_expr(self, ast):
        assert len(ast) >= 1
        name = ast[0]
        argv = ast[1]
        return FunctionCall(name, argv, Span.EMPTY)

    def let_expr(self, ast):
        assert len(ast) == 3
        return LetExpr(ast[0], ast[1], ast[2], Span.EMPTY)

    def lambda_expr(self, ast):
        assert len(ast) == 4
        name = ast[0]

        args = ast[1]
        # args[0] will be "(" if there are no arguments (must call with unit)
        if type(args[0]) == str:
            args = []
    
        value = ast[2]
        body = ast[3]

        return LambdaExpr(name, args, value, body, Span.EMPTY)

    def if_expr(self, ast):
        assert len(ast) == 3
        cond = ast[0]
        true_expr = ast[1]
        false_expr = ast[2]

        return IfExpr(cond, true_expr, false_expr, Span.EMPTY)

    def function(self, ast):
        assert len(ast) == 3
        name = ast[0]

        args = ast[1]
        # args[0] will be "(" if there are no arguments (must call with unit)
        if type(args[0]) == str:
            args = []

        body = ast[2]

        return Function(name, args, body, Span.EMPTY)

    def program(self, ast):
        return Program(ast)

    def start(self, ast):
        return ast