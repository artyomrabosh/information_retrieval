from collections import defaultdict
from typing import DefaultDict, Dict, List, Set

from core.document import Document
from core.tokenizer import Tokenizer
from core.position_storage import SimplePositionStorage


class DirectIndex:
    """Прямой индекс: doc_id -> term -> сжатые позиции (bytes).

    Позиции внутри документа считаются глобально по всем полям (как в InvertedIndex),
    чтобы расстояния между словами совпадали с тем, что видит инвертированный индекс.
    """

    def __init__(self):
        # doc_id -> term -> compressed positions bytes
        self.index: DefaultDict[str, Dict[str, bytes]] = defaultdict(dict)
        self.tokenizer = Tokenizer()
        self.pos_storage = SimplePositionStorage()
        self.fields: Set[str] = set()

    def add_document(self, doc: Document) -> None:
        """Добавить документ в прямой индекс с дельта-кодированием и побитовым сжатием позиций."""
        all_tokens: List[tuple[str, int]] = []
        current_position = 0

        for field_name, field_text in doc.fields.items():
            self.fields.add(field_name)
            # tokenizer.tokenize_field возвращает (token, local_pos),
            # но для расстояний нам нужен общий счетчик позиций
            tokens_with_positions = self.tokenizer.tokenize_field(field_name, field_text)

            for token, _ in tokens_with_positions:
                all_tokens.append((token, current_position))
                current_position += 1

        # группируем позиции по термам
        term_positions: Dict[str, List[int]] = defaultdict(list)
        for token, pos in all_tokens:
            term_positions[token].append(pos)

        # сжимаем позиции побитовым varbyte + delta-кодированием (как в SimplePositionStorage)
        compressed_terms: Dict[str, bytes] = {}
        for term, positions in term_positions.items():
            compressed_terms[term] = self.pos_storage.compress_positions(positions)

        self.index[doc.id] = compressed_terms

    def get_terms(self, doc_id: str) -> Dict[str, bytes]:
        """Вернуть словарь term -> compressed_positions для документа."""
        return self.index.get(doc_id, {})

    def get_positions(self, doc_id: str, term: str) -> List[int]:
        """Вернуть список (декодированных) позиций терма в документе."""
        term_data = self.index.get(doc_id)
        if not term_data:
            return []
        data = term_data.get(term)
        if not data:
            return []
        return self.pos_storage.decompress_positions(data)

    def get_document_length(self, doc_id: str) -> int:
        """Приблизительная длина документа в термах (по прямому индексу)."""
        term_data = self.index.get(doc_id)
        if not term_data:
            return 0
        length = 0
        for data in term_data.values():
            length += len(self.pos_storage.decompress_positions(data))
        return length