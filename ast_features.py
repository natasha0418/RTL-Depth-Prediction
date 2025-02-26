import logging

import pyverilog
from pyverilog.dataflow.dataflow_analyzer import VerilogDataflowAnalyzer
from pyverilog.vparser.ast import *
from pyverilog.vparser.parser import parse

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
    ast, _ = parse([rtl_file])
    topmodule = detect_topmodule(ast, topmodule)

    analyzer = VerilogDataflowAnalyzer(rtl_file, topmodule)
    analyzer.generate()

    return ast, analyzer


def extract_signals(ast, analyzer, features):
    """Extract features for all signals in the RTL design, counting signal types and extracting AST features."""

    signal_features = {}

    signal_names, signals = analyzer.getSignals().keys(), analyzer.getSignals().values()

    for signal in signals:
        signal_name = signal[0].name  # Name of the signal
        signal_class_name = signal[0].__class__.__name__  # Signal class name (e.g., Input, Output, etc.)
        features[signal_class_name] = features.get(signal_class_name, 0) + 1

        signal_features[signal_name] = {
            "fanin_depth": calculate_fanin_depth(signal, analyzer.binddict, ast),
            "module_nesting": get_module_nesting(signal, ast),
            "signal_width": get_signal_width(signal, ast),
        }

    features["signals"] = signal_features
    return features


def extract_features(rtl_file, topmodule=None):
    ast, analyzer = load_analyzer(rtl_file, topmodule)

    features = {}
    features = extract_signals(ast, analyzer, features)

    features["line_count"] = ast.lineno
    operators = extract_operators(ast)
    features.update(operators)

    # Combined operator count
    features["total_operators"] = sum(operators.values())
    features["paren_count"] = count_nodes(ast, "Paren")
    features["conditional_count"] = count_nodes(ast, "Cond")
    features["statement_depth"] = estimate_statement_depth(ast)
    features["if_count"] = count_nodes(ast, "IfStatement")
    return features


def calculate_fanin_depth(signal_name, binddict, ast, visited=None, depth=0):
    """Calculates the recursive fanin depth of a signal"""
    if visited is None:
        visited = set()

    if signal_name in visited:  # Prevent infinite loops
        return depth

    visited.add(signal_name)

    max_depth = depth
    fanin_signals = find_fanin_signals(signal_name, binddict, ast)

    for fanin in fanin_signals:
        max_depth = max(max_depth, calculate_fanin_depth(fanin, binddict, ast, visited, depth + 1))

    return max_depth


def find_fanin_signals(signal_name, binddict, ast):
    """Finds all fanin signals for a given signal using AST"""
    fanin_signals = set()

    for node in ast.children():
        if isinstance(node, pyverilog.vparser.ast.NonblockingSubstitution):  # Check signal assignments
            if node.left.var == signal_name:
                fanin_signals.update(get_identifiers(node.right))  # Extract input signals
        fanin_signals.update(find_fanin_signals(signal_name, binddict, node))  # Recursive check

    return fanin_signals


def get_identifiers(node):
    """Extracts all signal identifiers from an expression node"""
    if isinstance(node, pyverilog.vparser.ast.Identifier):
        return {node.name}
    identifiers = set()
    for child in node.children():
        identifiers.update(get_identifiers(child))
    return identifiers


def get_module_nesting(signal_name, ast):
    """Determines the module nesting level of a signal"""

    def find_module_depth(node, depth=1):
        if isinstance(node, pyverilog.vparser.ast.ModuleDef):
            return depth
        for child in node.children():
            module_depth = find_module_depth(child, depth + 1)
            if module_depth:
                return module_depth
        return None

    for module in ast.children():
        if isinstance(module, pyverilog.vparser.ast.ModuleDef):
            for decl in module.children():
                if isinstance(decl, pyverilog.vparser.ast.Decl):
                    for var in decl.children():
                        if hasattr(var, "name") and var.name == signal_name:
                            return find_module_depth(module)

    return 1  # Default nesting level if signal is not deeply nested


def get_signal_width(signal_name, ast):
    """Extracts the bit width of a signal using AST"""
    for module in ast.children():
        if isinstance(module, pyverilog.vparser.ast.ModuleDef):
            for decl in module.children():
                if isinstance(decl, pyverilog.vparser.ast.Decl):
                    for var in decl.children():
                        if hasattr(var, "name") and var.name == signal_name:
                            if hasattr(var, "width") and var.width is not None:
                                msb, lsb = var.width.msb.value, var.width.lsb.value
                                return abs(int(msb) - int(lsb)) + 1  # Compute signal width
    return 1  # Default width (single bit) if not explicitly defined


def count_nodes(ast, node_type):
    """Counts occurrences of a given node type in the AST"""
    count = 0
    for node in ast.children():
        if node.__class__.__name__ == node_type:
            count += 1
        count += count_nodes(node, node_type)  # Recursive traversal
    return count


def estimate_statement_depth(ast, depth=0):
    """Estimates the depth of statements in the AST"""
    max_depth = depth
    for node in ast.children():
        max_depth = max(max_depth, estimate_statement_depth(node, depth + 1))
    return max_depth


def extract_operators(node, operators=None):
    """Recursively extract all operators from AST."""
    if operators is None:
        operators = {op: 0 for op in OPERATOR_MAP.values()}

    if type(node) in OPERATOR_MAP:
        operators[OPERATOR_MAP[type(node)]] += 1

    for child in node.children():
        extract_operators(child, operators)

    return operators
