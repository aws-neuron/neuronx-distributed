from dataclasses import dataclass
import importlib
from typing import Any, Optional, Tuple

from torch_neuronx.xla_impl.ops import nki_jit
from neuronxcc.nki import jit


@dataclass
class NKIImport:
    """Configuration for NKI imports."""
    name: str
    module_name: Optional[str] = None
    nki_jit_type: Optional[str] = None

def import_nki(import_config: NKIImport) -> Tuple[Optional[Any], Optional[str]]:
    """
    Import NKI module with default and fallback paths.
    Args:
        import_config: NKIImport configuration dataclass

    Returns:
        tuple: (imported_module, error_message)
            - imported_module: The imported module or None if import failed
            - error_message: Error description if import failed, None otherwise
    """
    if not import_config.module_name:
        import_paths = [
            "neuronxcc.nki._pre_prod_kernels",
            "neuronxcc.nki._private_kernels"
        ]
    else:
        import_paths = [
            f"neuronxcc.nki._pre_prod_kernels.{import_config.module_name}",
            f"neuronxcc.nki._private_kernels.{import_config.module_name}"
        ]
    last_error = None
    for path in import_paths:
        try:
            module = importlib.import_module(path)
            attr = getattr(module, import_config.name)
            if import_config.nki_jit_type == "use_jit_decorator":
                return jit()(attr), None
            elif import_config.nki_jit_type == "use_nki_jit_decorator":
                return nki_jit()(attr), None
            return attr, None
        except ImportError as e:
            last_error = str(e)
            continue
        except AttributeError as e:
            last_error = f"Attribute {import_config.name} not found in {path}: {str(e)}"
            continue

    return None, f"Failed to import {import_config.name}: {last_error}"
