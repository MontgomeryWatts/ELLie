from typing import Any, List, Tuple, Optional, Union, Callable
import sys

from boi.utils import list_to_str

class Span:

    """
    a slice of the input string
    """

    EMPTY = None

    def __init__(self, start: int, end: int, input: str):
        self.start = start
        self.end = end
        self.slice = input[start:end]
        self.input = input
    
    def __repr__(self):
        return f"Span({self.start}, {self.end}, '{self.input}')"

    def __str__(self):
        return f"{self.start} - {self.end}: '{self.slice}'"

Span.EMPTY = Span(0, 0, "")

class InterpreterException(Exception):

    def __init__(self, reason: str, span: 'Span' = None):
        super().__init__()

        self.reason = reason
        self.span = Span.EMPTY if span is None else span

    def __str__(self):
        return f"Interpreter Error: {self.reason}.\n" + \
                "note: encountered error here\n" + \
                self.span.slice

class Context:

    # Must be initialized after the definition of the IntrinsicFunction class
    INTRINSICS = None

    def __init__(self):
        self.stack_frames = [{}]
        for intrinsic in Context.INTRINSICS:
            self.push_function(intrinsic)
        
        self.stdout = sys.stdout

    def push_stack_frame(self):
        new_dict = dict()
        self.stack_frames.append(new_dict)

    def pop_stack_frame(self):
        return self.stack_frames.pop()

    def current_stack_frame(self):
        assert len(self.stack_frames) > 0

        return self.stack_frames[-1]

    def push_args(self, args: List['Id'], values: List['Value']):
        assert len(args) == len(values)

        for arg, value in zip(args, values):
            self.push_var(arg, value)

    def pop_args(self, args: List['Id']):
        return list(map(lambda arg: self.pop_var(arg), args))

    def push_var(self, var: 'Id', value: 'Value'):
        sf = self.current_stack_frame()

        if var in sf:
            raise InterpreterException("name shadowing is not supported")
        else:
            sf[var] = value
    
    def pop_var(self, var: 'Id') -> 'Value':
        sf = self.current_stack_frame()

        if var in sf:
            return sf.pop(var)
        else:
            raise InterpreterException("tried to pop var which hasn't been added to the current stack frame", var.span)

    def push_function(self, fn: 'Function'):
        sf = self.current_stack_frame()

        argn = len(fn.args)

        if (fn.name, argn) in sf:
            raise InterpreterException(f"function '{fn.name}' with {argn} arguments is already defined, and name shadowing is not supported", fn.span)
        else:
            sf[(fn.name, argn)] = fn

    def pop_function(self, fn: 'Function'):
        sf = self.current_stack_frame()
        argn = len(fn.args)

        if (fn.name, argn) in sf:
            return sf.pop((fn.name, argn))
        else:
            raise InterpreterException(f"tried top pop function '{fn.name}' with {argn} arguments which hasn't been added to the current stack frame", fn.span)

    def get_stack_frame(self, offset):
        assert offset <= 0

        if len(self.stack_frames) < abs(offset):
            return None

        return self.stack_frames[offset]


    def get_function(self, var: 'Id', argn: int):
        assert argn >= 0

        sf = self.current_stack_frame()
        offset = -1
        
        while sf is not None:
            if (var, argn) in sf:
                return sf[(var, argn)]

            sf = self.get_stack_frame(offset)
            offset -= 1
        
        raise InterpreterException(f"no such function '{var}' with {argn} arguments", var.span)

    def get_var(self, var: 'Id'):

        sf = self.current_stack_frame()
        offset = -1
        
        while sf is not None:
            if var in sf:
                return sf[var]

            sf = self.get_stack_frame(offset)
            offset -= 1
        
        raise InterpreterException(f"no variable with name '{var}'", var.span)

class AST:

    @staticmethod
    def from_parse_tree(tokens: List[Any]):
        raise Exception("Unimplemented")

    def __init__(self, span: Span):
        self.span = span
    
    def eval(self, context: Context) -> Any:
        raise Exception("Unimplemented")

    def __str__(self):
        return self.span.slice
    
    def __repr__(self):
        raise Exception("Unimplemented")

class Expr(AST):

    def __init__(self, span: Span):
        super().__init__(span)

class Id(Expr):

    def __init__(self, id: str, span: Span):
        super().__init__(span)

        self.id = id


    def eval(self, context: Context) -> 'Value':
        return context.get_var(self)

    def __repr__(self):
        return f"Id('{self.id}', {repr(self.span)})"

    def __str__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id
    
    def __ne__(self, other):
        return self.id != other.id

    def __hash__(self):
        return hash(self.id)


class Value(Expr):
    
    def __init__(self, value: float, span: Span):
        super().__init__(span)
        self.value = value

    def __repr__(self):
        return f"Value({self.value}, {repr(self.span)})"

    def __str__(self):
        return str(self.value)

    def eval(self, _context: Context) -> 'Value':
        return self

    def __add__(self, other):
        return self.value + other.value

    def __sub__(self, other):
        return self.value - other.value
    
    def __mul__(self, other):
        return self.value * other.value
    
    def __div__(self, other):
        return self.value / other.value

    def __mod__(self, other):
        return self.value % other.value

    def __pow__(self, other):
        return self.value % other.value

    def __lt__(self, other):
        return self.value < other.value
    
    def __le__(self, other):
        return self.value <= other.value
    
    def __gt__(self, other):
        return self.value > other.value
    
    def __ge__(self, other):
        return self.value >= other.value
    
    def __eq__(self, other):
        return self.value == other.value
    
    def __ne__(self, other):
        return self.value != other.value


class MultiplicativeExpr(Expr):

    MULTIPLICATION  = 0
    DIVISION        = 1

    def __init__(self, lhs: Expr, operator: int, rhs: Expr, span: Span):
        super().__init__(span)

        assert 0 <= operator <= 1 

        self.lhs = lhs
        self.operator = operator
        self.rhs = rhs

    def __str__(self):
        if self.operator == 0:
            return f"({self.lhs} * {self.rhs})"
        else:
            return f"({self.lhs} / {self.rhs})"
    
    def __repr__(self):
        return f"MultiplicativeExpr({repr(self.lhs)}, {self.operator}, {repr(self.rhs)}, {repr(self.span)})"

    def eval(self, context: Context) -> Value:
        lhs_value = self.lhs.eval(context)
        rhs_value = self.rhs.eval(context)

        if self.operator == MultiplicativeExpr.MULTIPLICATION:
            new_value = lhs_value.value * rhs_value.value
        else:
            new_value = lhs_value.value / rhs_value.value

        assert type(new_value) == float

        return Value(new_value, self.span)


class PowExpr(Expr):

    def __init__(self, lhs: MultiplicativeExpr, rhs: MultiplicativeExpr, span: Span):
        super().__init__(span)

        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f"({self.lhs} ^ {self.rhs}) "

    def __repr__(self):
        return f"PowExpr({repr(self.lhs)}, {repr(self.rhs)}, {repr(self.span)})"

    def eval(self, context: Context) -> Value:
        lhs_value = self.lhs.eval(context)
        rhs_value = self.rhs.eval(context)

        new_value = lhs_value ** rhs_value
    
        assert type(new_value) == float

        return Value(new_value, self.span)


class AdditiveExpr(Expr):

    ADDITION    = 0
    SUBTRACTION = 1

    def __init__(self, lhs: PowExpr, operator: int, rhs: PowExpr, span: Span):
        super().__init__(span)

        assert 0 <= operator <= 1 

        self.lhs = lhs
        self.operator = operator
        self.rhs = rhs

    def __str__(self):
        if self.operator == 0:
            return f"({self.lhs} + {self.rhs})"
        else:
            return f"({self.lhs} - {self.rhs})"
    def __repr__(self):
        return f"AdditiveExpr({repr(self.lhs)}, {self.operator}, {repr(self.rhs)}, {repr(self.span)})"

    def eval(self, context: Context) -> Value:
        lhs_value = self.lhs.eval(context)
        rhs_value = self.rhs.eval(context)

        if self.operator == AdditiveExpr.ADDITION:
            new_value = lhs_value + rhs_value
        else:
            new_value = lhs_value - rhs_value

        assert type(new_value) == float

        return Value(new_value, self.span)


class ConditionExpr(Expr):

    FALSE_VALUES = [0.0, "", []]

    def __init__(self, expr: Expr, span: Span):
        super().__init__(span)

        self.expr = expr

    def __str__(self):
        return str(self.expr)
    
    def __repr__(self):
        return f"ConditionExpr({repr(self.expr)}, {repr(self.span)})"

    def eval(self, context: Context) -> bool:
        return self.expr.eval(context).value not in ConditionExpr.FALSE_VALUES


class ComparisonExpr(Expr):

    GT = 0
    GTE = 1
    LT = 2
    LTE = 3
    EQ = 4
    NEQ = 5

    COMPARISON_FNS = {
        GT:     lambda l, r: l > r,
        GTE:    lambda l, r: l >= r,
        LT:     lambda l, r: l < r,
        LTE:    lambda l, r: l <= r,
        EQ:     lambda l, r: l == r,
        NEQ:    lambda l, r: l != r
    }

    COMPARISON_OP_TO_STR = {
        GT:     ">",
        GTE:    ">=",
        LT:     "<",
        LTE:    "<=",
        EQ:     "=",
        NEQ:    "<>"
    }

    def __init__(self, lhs: Expr, operator: int, rhs: Expr, span: Span):
        super().__init__(span)

        assert 0 <= operator <= 5

        self.lhs = lhs
        self.rhs = rhs
        self.operator = operator

    def __str__(self):
        return f"{self.lhs} + {ComparisonExpr.COMPARISON_OP_TO_STR[self.operator]} + {self.rhs}"

    def __repr__(self):
        return f"ComparisonExpr({repr(self.lhs)}, {self.operator}, {repr(self.rhs)}, {repr(self.span)})"

    def eval(self, context: Context):
        return ComparisonExpr.COMPARISON_FNS[self.operator](self.lhs.eval(context), self.rhs.eval(context))


class LambdaExpr(Expr):

    def __init__(self, name: Id, args: List[Id], value: Expr, body: Expr, span: Span):
        super().__init__(span)

        self.name = name
        self.args = args
        self.value = value
        self.body = body

        self.as_fn = Function(name, args, value, span, is_lambda=True)
    
    def __str__(self):
        return f"let {self.name} {list_to_str(self.args)} = {self.value} in \n    {self.body}"

    def __repr__(self):
        return f"LambdaExpr({repr(self.name)}, {repr(self.args)}, {repr(self.value)}, {repr(self.body)}, {repr(self.span)})"

    def eval(self, context: Context) -> Value:
        context.push_function(self.as_fn)

        result = self.body.eval(context)

        _fn = context.pop_function(self.as_fn)

        return result

class LetExpr(Expr):

    def __init__(self, name: Id, value: Expr, body: Expr, span: Span):
        super().__init__(span)

        self.name = name
        self.value = value
        self.body = body
        self.value = value

    def __str__(self):
        return f"let {self.name} = {self.value} in {self.body}"

    def __repr__(self):
        return f"LetExpr({repr(self.name)}, {repr(self.value)}, {repr(self.body)}, {repr(self.span)})"
    
    def eval(self, context: Context) -> Value:
        val = self.value.eval(context) 
        
        context.push_var(self.name, val)

        body_value = self.body.eval(context)

        context.pop_var(self.name)

        return body_value


class FunctionCall(Expr):

    def __init__(self, name: Id, argv: List[Expr], span: Span):
        super().__init__(span)

        self.name = name
        self.argv = argv

    def __str__(self):
        return f"({self.name} {list_to_str(self.argv)})"

    def __repr__(self):
        return f"FunctionCall({repr(self.name)}, {repr(self.argv)}, {repr(self.span)})"

    def eval(self, context: Context):
        fn = context.get_function(self.name, len(self.argv))
        argv = list(map(lambda exp: exp.eval(context), self.argv))
        fn.set_args(argv)

        return fn.eval(context)


class IfExpr(Expr):

    def __init__(self, condition: ConditionExpr, true_expr: Expr, false_expr: Expr, span: Span):
        super().__init__(span)

        self.condition = condition
        self.true_expr = true_expr
        self.false_expr = false_expr

    def __repr__(self):
        return f"IfExpr({repr(self.condition)}, {repr(self.true_expr)}, {repr(self.false_expr)}, {repr(self.span)})"
    
    def __str__(self):
        return f"if {self.condition} then {self.true_expr} else {self.false_expr}"

    def eval(self, context: Context) -> Value:
        if self.condition.eval(context):
            return self.true_expr.eval(context)
        else:
            return self.false_expr.eval(context)


class Function(AST):

    def __init__(self, name: Id, args: List[Id], value: Expr, span: Span, is_lambda: bool = False):
        super().__init__(span)

        self.name = name
        self.args = args
        self.value = value
        self.is_lambda = is_lambda

        self.argv = None
    
    def set_args(self, argv: List[Value]):
        if len(argv) != len(self.args):
            raise Exception("Interpreter Error: Calling function with wrong number of arguments.")
        
        self.argv = argv

    def __str__(self):
        return f"let {self.name} {list_to_str(self.args)} = {self.value}"

    def __repr__(self):
        return f"Function({repr(self.name)}, {repr(self.args)}, {repr(self.value)}, {repr(self.span)}, is_lambda = {self.is_lambda})"

    def eval(self, context: Context) -> Value:
        """
        set_args must be called before this function is called
        """

        assert self.argv != None 

        if not self.is_lambda:
            context.push_stack_frame()

        context.push_args(self.args, self.argv)
        
        r = self.value.eval(context)
        
        self.argv = None

        _argv = context.pop_args(self.args)

        if not self.is_lambda:
            context.pop_stack_frame()

        return r

class IntrinsicFunction(AST):

    INTRINSIC_FUNCTIONS = [
        (Id("print", Span.EMPTY), [Id("x", Span.EMPTY)], lambda context, x: print(x.value, file=context.stdout))
    ]

    def __init__(self, name: Id, args: List[Id], fn: Callable):
        super().__init__(Span.EMPTY)

        self.name = name
        self.args = args
        self.fn = fn

        self.argv = None

    def __str__(self):
        return f"<intrinsic> {self.name} {list_to_str(self.args)} = ..."

    def __repr__(self):
        return object.__repr__(self)

    def set_args(self, argv: List[Value]):
        if len(argv) != len(self.args):
            raise InterpreterException("attempted to call a function with wrong number of arguments")
        
        self.argv = argv

    def eval(self, context: Context):
        assert self.argv != None
        
        r = self.fn(context, *self.argv)

        if r is None:
            r = 0.0

        return r

Context.INTRINSICS = []

for fn in IntrinsicFunction.INTRINSIC_FUNCTIONS:
    f = IntrinsicFunction(fn[0], fn[1], fn[2])
    Context.INTRINSICS.append(f)

class Program:

    def __init__(self, statements: List[Union[Function, Expr]]):
        self.statements = statements
        self.context = Context()

    def run(self, stdout=sys.stdout):
        self.context.stdout = stdout
        for statement in self.statements:
            ty = type(statement)

            if ty == Function:
                self.context.push_function(statement)
            else:
                statement.eval(self.context)
    
    def __repr__(self):
        return f"Program({repr(self.statements)})"

    def __str__(self):
        s = ""

        for statement in self.statements:
            s += str(statement)
            s += "\n"
        
        return s

def success(test_name):
    print(f"\n ╠═ {test_name + ' test':34} :: Okay", end='')

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