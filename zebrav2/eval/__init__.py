"""vzebra v2 evaluation platform."""
from zebrav2.eval.base import EvalPlugin, EvalResult, EvalContext
from zebrav2.eval.registry import EvalRegistry, register
from zebrav2.eval.pipeline import Pipeline
from zebrav2.eval.report import ReportBuilder

__all__ = ['EvalPlugin', 'EvalResult', 'EvalContext',
           'EvalRegistry', 'register', 'Pipeline', 'ReportBuilder']
