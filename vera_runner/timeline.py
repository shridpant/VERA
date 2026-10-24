"""Append-only record of what a run is doing, one JSON event per line, for live display.

Events carry only values the run actually produced (sampled frame indices, the prompt, text
as it is generated, parsed labels, boxes). Image fields are paths relative to the folder that
holds this timeline file.
"""
import json
import threading
import time


class Timeline:
    def __init__(self, path):
        # Kept open and line-buffered: each event is visible at once, without a reopen per event.
        self.handle = open(path, 'a', encoding='utf-8', buffering=1)
        self.lock = threading.Lock()

    def emit(self, event, **fields):
        line = json.dumps({'t': round(time.time(), 3), 'event': event, **fields}, ensure_ascii=False)
        with self.lock:
            self.handle.write(line + '\n')

    def close(self):
        self.handle.close()

    def streamer(self, **fields):
        """on_text callback for predict(): reports generated text at most every quarter second."""
        return lambda text, tokens, done: None if done else self.emit('generating', text=text, tokens=tokens, **fields)


class Silent:
    """Timeline stand-in when nothing should be recorded."""
    def emit(self, event, **fields):
        pass

    def streamer(self, **fields):
        return None

    def close(self):
        pass
