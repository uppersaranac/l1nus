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

from llm.llm_apis import (
    AllPropertiesProcessor,
    IUPACNamingProcessor,
    MolecularPropertiesProcessor,
    QuestionSetProcessor,
)
from llm.structure_processor import StructureProcessor
from typing import Dict, Type

# Public mapping identical to original
PROCESSOR_CLASSES: Dict[str, Type[QuestionSetProcessor]] = {
    "all_properties": AllPropertiesProcessor,
    "iupac_naming": IUPACNamingProcessor,
    "molecular_properties": MolecularPropertiesProcessor,
    "structure": StructureProcessor,
}
"""
Dictionary mapping question set names to their processor classes.

:rtype: Dict[str, Type[QuestionSetProcessor]]
"""

__all__ = [
    "AllPropertiesProcessor",
    "IUPACNamingProcessor",
    "MolecularPropertiesProcessor",
    "StructureProcessor",
    "PROCESSOR_CLASSES",
    "StructureProcessor",
    "QuestionSetProcessor",
]
