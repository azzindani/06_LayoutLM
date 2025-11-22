"""
Pipeline module - lazy imports.
"""

__all__ = ['DocumentProcessor', 'get_processor', 'preprocess_image', 'export_result']


def __getattr__(name):
    if name in ('DocumentProcessor', 'get_processor'):
        from core.pipeline.document_processor import DocumentProcessor, get_processor
        return DocumentProcessor if name == 'DocumentProcessor' else get_processor
    if name in ('preprocess_image', 'load_image', 'validate_image'):
        from core.pipeline.image_utils import preprocess_image, load_image, validate_image
        mapping = {'preprocess_image': preprocess_image, 'load_image': load_image,
                   'validate_image': validate_image}
        return mapping.get(name)
    if name in ('export_result', 'export_json', 'export_csv', 'export_xml'):
        from core.pipeline.export import export_result, export_json, export_csv, export_xml
        mapping = {'export_result': export_result, 'export_json': export_json,
                   'export_csv': export_csv, 'export_xml': export_xml}
        return mapping.get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
