import logging

from pyverilog.dataflow.dataflow_analyzer import VerilogDataflowAnalyzer
from pyverilog.vparser.ast import *
from pyverilog.vparser.parser import parse

from ast_signal_features import extract_signals

#this maps verilog operators to human readable names
#will be used lator to count the occurences of each operator in the AST
OPERATOR_MAP = {
    # Arithmetic Operators
    Plus: "add",
    Minus: "subtract",
    Times: "multiply",
    Divide: "divide",
    Mod: "modulus",
    # Bitwise Operators
    And: "bitwise_and",
    Or: "bitwise_or",
    Xor: "bitwise_xor",
    Xnor: "bitwise_xnor",
    Unot: "bitwise_not",  # Bitwise NOT (~)
    # Logical Operators
    Land: "logical_and",
    Lor: "logical_or",
    Ulnot: "logical_not",  # Logical NOT (!)
    # Shift Operators
    Sll: "shift_left_logical",
    Srl: "shift_right_logical",
    Sra: "shift_right_arithmetic",
    # Comparison Operators
    GreaterThan: "greater_than",
    LessThan: "less_than",
    GreaterEq: "greater_equal",
    LessEq: "less_equal",
    Eq: "equal",
    NotEq: "not_equal",
    # Ternary (Conditional) Operator
    Cond: "conditional",
}
# Setting up logging to display status and debugging messages
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def detect_topmodule(ast, topmodule=None):
    """Detect top module from AST if not provided"""
    if topmodule is None:
        for definition in ast.description.definitions:
            if definition.__class__.__name__ == "ModuleDef":
                topmodule = definition.name
                logger.info(f"✓ Detected top module: {topmodule}")
                break

    if not topmodule:
        logger.error("Unable to detect top module")
        raise ValueError("Top module not found or specified")

    return topmodule


def load_analyzer(rtl_file, topmodule=None):
    """Load a VerilogDataflowAnalyzer object from an RTL file"""
    #converts verilog file into AST
    ast, _ = parse([rtl_file])
    topmodule = detect_topmodule(ast, topmodule)

    analyzer = VerilogDataflowAnalyzer(rtl_file, topmodule)
    #construncts dataflow info, mapping signals to their defining expressions
    analyzer.generate()

    return ast, analyzer


def extract_ast_features(rtl_file, topmodule=None):
    ast, analyzer = load_analyzer(rtl_file, topmodule)

    features = {}
    features = extract_signals(ast, analyzer, features)

    features["line_count"] = ast.lineno
    operators = extract_operators(ast)
    features.update(operators)

    features["total_operators"] = sum(operators.values())
    features["paren_count"] = count_nodes(ast, "Paren")
    features["conditional_count"] = count_nodes(ast, "Cond")
    features["statement_depth"] = estimate_statement_depth(ast)
    features["if_count"] = count_nodes(ast, "IfStatement")
    return features


def count_nodes(ast, node_type):
    """Counts occurrences of a given node type in the AST"""
    count = 0
    for node in ast.children():
        if node.__class__.__name__ == node_type:
            count += 1
        count += count_nodes(node, node_type)
    return count


def estimate_statement_depth(ast, depth=0):
    """Estimates the depth of statements in the AST"""
    if not hasattr(ast, "children") or not ast.children():
        return depth
    return max(estimate_statement_depth(child, depth + 1) for child in ast.children())


def extract_operators(node, operators=None):
    """Recursively extract all operators from AST."""
    if operators is None:
        operators = {op: 0 for op in OPERATOR_MAP.values()}

    if isinstance(node, tuple(OPERATOR_MAP.keys())):
        operators[OPERATOR_MAP[type(node)]] += 1

    if hasattr(node, "children"):
        for child in node.children():
            extract_operators(child, operators)

    return operators
