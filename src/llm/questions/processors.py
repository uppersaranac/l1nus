"""Re-export existing QuestionSetProcessor implementations for the
question-generation stage.

For now we rely on the chemistry-specific processors already defined in
`llm_apis.py` so that we do not duplicate logic.  This wrapper keeps the
*question-generation* sub-package self-contained: callers can simply do::

    from llm.questions.processors import PROCESSOR_CLASSES

and obtain the same mapping (`{"iupac_naming": IUPACNamingProcessor, ...}`)
that was previously only available in `llm_apis`.

In the future we could move these class definitions here entirely or load them
via entry-points for true domain-agnostic plug-ins.
"""
from __future__ import annotations

from typing import Dict, Type

# Import implementations from existing module
from llm.llm_apis import (
    QuestionSetProcessor,
    IUPACNamingProcessor,
    MolecularPropertiesProcessor,
    AllPropertiesProcessor,
)

# Public mapping identical to original
PROCESSOR_CLASSES: Dict[str, Type[QuestionSetProcessor]] = {
    "iupac_naming": IUPACNamingProcessor,
    "molecular_properties": MolecularPropertiesProcessor,
    "all_properties": AllPropertiesProcessor,
}
"""
Dictionary mapping question set names to their processor classes.

:rtype: Dict[str, Type[QuestionSetProcessor]]
"""

__all__ = [
    "QuestionSetProcessor",
    "IUPACNamingProcessor",
    "MolecularPropertiesProcessor",
    "AllPropertiesProcessor",
    "PROCESSOR_CLASSES",
]
