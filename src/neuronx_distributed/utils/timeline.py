import time
from abc import ABC, abstractmethod, abstractproperty
from collections import OrderedDict


class Event:
    def __init__(self, label: str, rank: int, start: int = -1, end: int = -1):
        self.label = label
        self.rank = rank
        self.start = start
        self.end = end


class Timeline(ABC):
    """
    Creating a timeline file
    Usage:
    Mark the start of an event with
        Timeline.mark_event_start(label)
    Mark the end of an event with the same label
        Timeline.mark_event_end(label)
    Mark the end of an step, timeline will gather events from all ranks and dump into the trace file
        Timeline.mark_step_end()
    """

    def __init__(self, trace_file_path, rank):
        if trace_file_path is not None:
            self.trace_file_path = trace_file_path
            self.step = 0
            self.rank = rank
            self.enabled = True
            self._clean_states()
            if self.should_record and self.rank == 0:
                with open(self.trace_file_path, "a") as f:
                    f.write("[")
        else:
            self.enabled = False

    @abstractproperty
    def should_record(self):
        pass

    def mark_event_start(self, label):
        if not self.should_record:
            return
        timestamp = self._get_timestamp()
        event = Event(label=label, rank=self.rank, start=timestamp)
        assert label not in self.current_rank_events
        self.current_rank_events[label] = event

    def mark_event_end(self, label):
        if not self.should_record:
            return
        timestamp = self._get_timestamp()
        assert label in self.current_rank_events
        event = self.current_rank_events[label]
        event.end = timestamp

    def mark_step_end(self):
        if not self.should_record:
            return
        self._collect_events_for_all_ranks()
        if self.rank == 0:
            self._dump_events()
        self._clean_states()
        self.step += 1

    def _dump_events(self):
        with open(self.trace_file_path, "a") as f:
            for events in self.all_rank_events:
                for _, event in events.items():
                    trace_events = self._creat_sync_event(event)
                    for trace in trace_events:
                        f.write(trace)

    def _get_timestamp(self):
        return time.time() * 1000000

    def _clean_states(self):
        if not self.should_record:
            return
        self.current_rank_events = OrderedDict()
        self.all_rank_events = None

    @abstractmethod
    def _collect_events_for_all_ranks(self):
        pass

    def _creat_sync_event(self, pp_event):
        events = []
        ph = "B"
        assert pp_event.start != -1
        start_event = (
            '{"cat": "comp", "ph": "'
            + ph
            + '", "name": "'
            + pp_event.label
            + '", "ts": '
            + str(pp_event.start)
            + ', "tid": '
            + str(0)
            + ', "pid": '
            + str(pp_event.rank)
            + "},\n"
        )
        events.append(start_event)
        ph = "E"
        assert pp_event.end != -1
        end_event = (
            '{"cat": "comp", "ph": "'
            + ph
            + '", "name": "'
            + pp_event.label
            + '", "ts": '
            + str(pp_event.end)
            + ', "tid": '
            + str(0)
            + ', "pid": '
            + str(pp_event.rank)
            + "},\n"
        )
        events.append(end_event)
        return events

    def _create_instant_event(self, label, timestamp):
        event = (
            '{"cat": "comp", "ph": "i", "name": "'
            + label
            + '", "ts": '
            + str(timestamp)
            + ', "tid": '
            + str(self.step)
            + ', "pid": '
            + str(self.rank)
            + ', "s": "p"},\n'
        )
        return event
