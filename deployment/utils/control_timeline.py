"""Shared nanosecond timeline rules for recorded data and direct array recordings."""

from bisect import bisect_right


class ExcludedTimeline:
    """Half-open intervals; every topic uses the same cumulative time removal."""

    def __init__(self, intervals):
        self.intervals = []
        for start, end in sorted(intervals):
            if end <= start:
                continue
            if self.intervals and start <= self.intervals[-1][1]:
                self.intervals[-1][1] = max(end, self.intervals[-1][1])
            else:
                self.intervals.append([start, end])
        self.starts = [start for start, _ in self.intervals]
        self.removed = [0]
        for start, end in self.intervals:
            self.removed.append(self.removed[-1] + end - start)

    def contains(self, timestamp):
        index = bisect_right(self.starts, timestamp) - 1
        return index >= 0 and timestamp < self.intervals[index][1]

    def shift(self, timestamp):
        index = bisect_right(self.starts, timestamp) - 1
        if index < 0:
            return timestamp
        start, end = self.intervals[index]
        return timestamp - self.removed[index] - min(timestamp - start, end - start)
