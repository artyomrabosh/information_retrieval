from typing import List


class SimplePositionStorage:
    """
    Простое хранилище позиций. Позиции хранятся без сложного сжатия,
    так как они обычно короткие и PForDelta для них неэффективен.
    """

    def compress_positions(self, positions: List[int]) -> bytes:
        """Простое кодирование позиций."""
        if not positions:
            return b''

        positions.sort()

        result = bytearray()

        if len(positions) < 128:
            result.append(len(positions))
        else:
            result.append(0x80 | (len(positions) >> 8))
            result.append(len(positions) & 0xFF)

        last_pos = 0
        for pos in positions:
            delta = pos - last_pos
            last_pos = pos

            while delta >= 128:
                result.append((delta & 0x7F) | 0x80)
                delta >>= 7
            result.append(delta)

        return bytes(result)

    def decompress_positions(self, data: bytes) -> List[int]:
        """Декомпрессия позиций."""
        if not data:
            return []

        result = []
        offset = 0

        if data[0] & 0x80:
            count = ((data[0] & 0x7F) << 8) | data[1]
            offset = 2
        else:
            count = data[0]
            offset = 1

        last_pos = 0
        for _ in range(count):
            if offset >= len(data):
                break

            delta = 0
            shift = 0
            while offset < len(data):
                byte_val = data[offset]
                offset += 1
                delta |= (byte_val & 0x7F) << shift
                if not (byte_val & 0x80):
                    break
                shift += 7

            last_pos += delta
            result.append(last_pos)

        return result
