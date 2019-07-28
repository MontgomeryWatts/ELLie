from boi.test.utils import success
from boi.ast import *

def test_if_expr():
    test_true_expr = IfExpr(ConditionExpr(Value(4.0, Span.EMPTY), Span.EMPTY), Value(1.0, Span.EMPTY), Value(0.0, Span.EMPTY), Span.EMPTY)
    test_false_expr = IfExpr(ConditionExpr(Value(0.0, Span.EMPTY), Span.EMPTY), Value(1.0, Span.EMPTY), Value(0.0, Span.EMPTY), Span.EMPTY)

    assert test_true_expr.eval(Context()).value == 1.0
    assert test_false_expr.eval(Context()).value == 0.0

    success("if_expr")

def test_let_expr():
    let_expr = LetExpr(Id("a", Span.EMPTY), Value(5.0, Span.EMPTY), Id("a", Span.EMPTY), Span.EMPTY)

    assert let_expr.eval(Context()).value == 5.0

    success("let_expr")

def test_simple_lambda_expr():
    lambda_expr = LambdaExpr(Id("lambda", Span.EMPTY), [Id("a", Span.EMPTY), Id("b", Span.EMPTY)], AdditiveExpr(Id("a", Span.EMPTY), AdditiveExpr.ADDITION, Id("b", Span.EMPTY), Span.EMPTY),
                                FunctionCall(Id("lambda", Span.EMPTY), [Value(4.0, Span.EMPTY), Value(1.0, Span.EMPTY)], Span.EMPTY), Span.EMPTY)

    assert lambda_expr.eval(Context()).value == 5.0

    success("simple_lambda_expr")

def test_simple_function_call():
    function_ast = Function(Id("sum", Span.EMPTY), [Id("a", Span.EMPTY), Id("b", Span.EMPTY)], AdditiveExpr(Id("a", Span.EMPTY), AdditiveExpr.ADDITION, Id("b", Span.EMPTY), Span.EMPTY), Span.EMPTY)
    
    program = Program([function_ast])
    program.run()

    function_call_ast = FunctionCall(Id("sum", Span.EMPTY), [Value(4.0, Span.EMPTY), Value(5.0, Span.EMPTY)], Span.EMPTY)

    assert function_call_ast.eval(program.context).value == 9

    success("simple_function_call")

def test_additive_expr():
    context = Context()
    ast = AdditiveExpr(Value(4.0, Span.EMPTY), AdditiveExpr.ADDITION, Value(4.0, Span.EMPTY), Span.EMPTY)

    assert ast.eval(context).value == 8.0

    success("additive_expr")

def test_condition_expr():
    context = Context()
    test_false_ast = ConditionExpr(Value(0.0, Span.EMPTY), Span.EMPTY)
    test_true_ast = ConditionExpr(Value(1.4, Span.EMPTY), Span.EMPTY)

    assert test_false_ast.eval(context) == False
    assert test_true_ast.eval(context) == True
    
    success("condition_expr")

def test_comparison_expr():
    context = Context()

    test_gt_true = ComparisonExpr(Value(0.0, Span.EMPTY), ComparisonExpr.GT, Value(-1.0, Span.EMPTY), Span.EMPTY)
    test_gt_false = ComparisonExpr(Value(0.0, Span.EMPTY), ComparisonExpr.GT, Value(1.0, Span.EMPTY), Span.EMPTY)

    test_gte_true = ComparisonExpr(Value(-1.0, Span.EMPTY), ComparisonExpr.GTE, Value(-1.0, Span.EMPTY), Span.EMPTY)
    test_gte_false = ComparisonExpr(Value(0.0, Span.EMPTY), ComparisonExpr.GTE, Value(1.0, Span.EMPTY), Span.EMPTY)

    test_lt_true = ComparisonExpr(Value(0.0, Span.EMPTY), ComparisonExpr.LT, Value(1.0, Span.EMPTY), Span.EMPTY)
    test_lt_false = ComparisonExpr(Value(0.0, Span.EMPTY), ComparisonExpr.LT, Value(-1.0, Span.EMPTY), Span.EMPTY)

    test_lte_true = ComparisonExpr(Value(1.0, Span.EMPTY), ComparisonExpr.LTE, Value(1.0, Span.EMPTY), Span.EMPTY)
    test_lte_false = ComparisonExpr(Value(0.0, Span.EMPTY), ComparisonExpr.LTE, Value(-1.0, Span.EMPTY), Span.EMPTY)

    test_eq_true = ComparisonExpr(Value(0.0, Span.EMPTY), ComparisonExpr.EQ, Value(0.0, Span.EMPTY), Span.EMPTY)
    test_eq_false = ComparisonExpr(Value(-1.0, Span.EMPTY), ComparisonExpr.EQ, Value(1.0, Span.EMPTY), Span.EMPTY)
    
    test_neq_true = ComparisonExpr(Value(0.0, Span.EMPTY), ComparisonExpr.NEQ, Value(-1.0, Span.EMPTY), Span.EMPTY)
    test_neq_false = ComparisonExpr(Value(0.0, Span.EMPTY), ComparisonExpr.NEQ, Value(0.0, Span.EMPTY), Span.EMPTY)

    assert test_gt_true.eval(context) == True
    assert test_gt_false.eval(context) == False

    assert test_gte_true.eval(context) == True
    assert test_gte_false.eval(context) == False

    assert test_lt_true.eval(context) == True
    assert test_lt_false.eval(context) == False

    assert test_lte_true.eval(context) == True
    assert test_lte_false.eval(context) == False

    assert test_eq_true.eval(context) == True
    assert test_eq_false.eval(context) == False

    assert test_neq_true.eval(context) == True
    assert test_neq_false.eval(context) == False

    success("comparison_expr")