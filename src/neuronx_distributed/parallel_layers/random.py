import torch
import contextlib
from .parallel_state import get_tensor_model_parallel_rank

    
_MODEL_PARALLEL_RNG_TRACKER_NAME = 'model-parallel-rng'


def _set_xla_rng_state(new_state, device=-1):
    """Sets the random number generator state.

    Args:
        new_state (torch.ByteTensor): The desired state
    
    """
    torch.set_rng_state(new_state)


class XLARNGStatesTracker:
    """Tracker for the xla RNG states.

    Using the `add` method, a xla rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    xla state.
    """

    def __init__(self):
        # Map from a string name to the xla rng state.
        self.states_ = {}
        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def reset(self):
        """Set to the initial state (no tracker)."""
        self.states_ = {}
        self.seeds_ = set()

    def get_states(self):
        """Get rng states. Copy the dictionary so we have direct
        pointers to the states, not just a pointer to the dictionary."""
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states):
        """Set the rng states. For efficiency purposes, we do not check
        the size of seed for compatibility."""
        self.states_ = states

    def add(self, name, seed):
        """Track the rng state."""
        # Check seed is not already used.
        if seed in self.seeds_:
            raise Exception("seed {} already exists".format(seed))
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise Exception("xla rng state {} already exists".format(name))
        # Get the current rng state.
        orig_xla_rng_state = torch.get_rng_state()
        # Set the new state and store it.
        torch.manual_seed(seed)
        self.states_[name] = torch.get_rng_state()
        # Reset rng state to what it was.
        _set_xla_rng_state(orig_xla_rng_state)

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork the rng state, perform operations, and exit with
        the original state."""
        # Check if we have added the state
        if name not in self.states_:
            raise Exception("rng state {} is not added".format(name))
        # Store current rng state.
        orig_xla_rng_state = torch.get_rng_state()
        # Set rng state to the desired one
        _set_xla_rng_state(self.states_[name])
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # Update the current rng state for later use.
            self.states_[name] = torch.get_rng_state()
            # And set the state to the original state we started with.
            _set_xla_rng_state(orig_xla_rng_state)


# RNG tracker object.
_XLA_RNG_STATE_TRACKER = XLARNGStatesTracker()


def get_xla_rng_tracker():
    """Get xla rng tracker."""
    return _XLA_RNG_STATE_TRACKER


def model_parallel_xla_manual_seed(seed):
    """Initialize model parallel xla seed.

    This function should be called after the model parallel is
    initialized. Also, no torch.manual_seed should be called
    after this function. Basically, this is replacement for that
    function.
    Two set of RNG states are tracked:
        default state: This is for data parallelism and is the same among a
                       set of model parallel Neuron devices but different across
                       different model parallel groups. This is used for
                       example for dropout in the non-tensor-model-parallel regions.
        tensor-model-parallel state: This state is different among a set of model
                              parallel Neuron devices, but the same across data parallel
                              groups. This is used for example for dropout in
                              model parallel regions.
    """
    # 2718 is just for fun and any POSITIVE value will work.
    offset = seed + 2718
    tensor_model_parallel_seed = offset + get_tensor_model_parallel_rank()
    # Data parallel gets the original seed.
    data_parallel_seed = seed

    _XLA_RNG_STATE_TRACKER.reset()
    # Set the default state.
    torch.manual_seed(data_parallel_seed)
    # and model parallel state.
    _XLA_RNG_STATE_TRACKER.add(
        _MODEL_PARALLEL_RNG_TRACKER_NAME, tensor_model_parallel_seed
    )