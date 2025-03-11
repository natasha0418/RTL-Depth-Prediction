from pyverilog.vparser.ast import *


def extract_signals(ast, analyzer, features):
    """Extract features for all signals in the RTL design, counting signal types and extracting AST features."""

    signal_features = {}

    signals = analyzer.getSignals().values()

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


def calculate_fanin_depth(signal_name, binddict, ast, visited=None, depth=0):
    """Calculates the recursive fanin depth of a signal."""
    if visited is None:
        visited = set()
    if signal_name in visited:
        return depth

    visited.add(signal_name)
    max_depth = depth
    fanin_signals = find_fanin_signals(signal_name, binddict)

    for fanin in fanin_signals:
        max_depth = max(max_depth, calculate_fanin_depth(fanin, binddict, ast, visited, depth + 1))

    return max_depth


def find_fanin_signals(signal_name, binddict):
    """Finds all fanin signals using binding dictionary from PyVerilog."""
    fanin_signals = set()

    if signal_name in binddict:
        for bind in binddict[signal_name]:
            fanin_signals.update(get_identifiers(bind.tree))

    return fanin_signals


def get_identifiers(node):
    """Extracts all signal identifiers from an expression node."""
    if isinstance(node, Identifier):
        return {node.name}
    identifiers = set()
    for child in node.children():
        identifiers.update(get_identifiers(child))
    return identifiers


def get_module_nesting(signal_name, ast):
    """Determines the module nesting level of a signal."""

    def find_module_depth(node, depth=1):
        if isinstance(node, ModuleDef):
            return depth
        for child in node.children():
            module_depth = find_module_depth(child, depth + 1)
            if module_depth:
                return module_depth
        return None

    for module in ast.children():
        if isinstance(module, ModuleDef):
            for decl in module.children():
                if isinstance(decl, Decl):
                    for var in decl.children():
                        if hasattr(var, "name") and var.name == signal_name:
                            return find_module_depth(module)
    return 1


def get_signal_width(signal_name, ast):
    """Extracts the bit width of a signal using AST."""
    for module in ast.children():
        if isinstance(module, ModuleDef):
            for decl in module.children():
                if isinstance(decl, Decl):
                    for var in decl.children():
                        if hasattr(var, "name") and var.name == signal_name:
                            if hasattr(var, "width") and var.width is not None:
                                msb, lsb = int(var.width.msb.value), int(var.width.lsb.value)
                                return abs(msb - lsb) + 1
    return 1
