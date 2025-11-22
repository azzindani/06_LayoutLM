"""
Model module - lazy imports.
"""

__all__ = ['load_model', 'run_inference', 'process_predictions', 'format_output']


def __getattr__(name):
    if name in ('load_model', 'get_label_mapping', 'clear_model_cache'):
        from core.model.loader import load_model, get_label_mapping, clear_model_cache
        mapping = {'load_model': load_model, 'get_label_mapping': get_label_mapping,
                   'clear_model_cache': clear_model_cache}
        return mapping.get(name)
    if name in ('run_inference', 'batch_inference'):
        from core.model.inference import run_inference, batch_inference
        return run_inference if name == 'run_inference' else batch_inference
    if name in ('process_predictions', 'format_output', 'Entity'):
        from core.model.postprocess import process_predictions, format_output, Entity
        mapping = {'process_predictions': process_predictions, 'format_output': format_output,
                   'Entity': Entity}
        return mapping.get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
