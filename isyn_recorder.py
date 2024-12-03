import logging
import numpy as np

from itertools import chain

from pygenn import VarAccessDim
from typing import Optional
from ml_genn.callbacks import Callback
from ml_genn.utils.filter import ExampleFilter, ExampleFilterType, NeuronFilterType
from ml_genn.utils.network import ConnectionType

from pygenn import get_var_access_dim
from ml_genn.utils.filter import get_neuron_filter_mask
from ml_genn.utils.network import get_underlying_conn
from ml_genn.utils.value import get_genn_var_name


logger = logging.getLogger(__name__)

class IsynRecorder(Callback):
    """Callback used for recording Isyn during simulation. 
    the variable e.g. ``v`` for the membrane voltage of a 
    :class:`ml_genn.neurons.LeakyIntegrateFire` neuron or by the internal name
    of a GeNN state variable e.g. ``LambdaV`` which is a state variable
    added to track gradients by :class:`ml_genn.compilers.EventPropCompiler`.
    
    Args:
        conn:           Connection to record from
        key:            Key to assign recording data produced by this 
                        callback in dictionary  returned by 
                        evaluation/training methods of compiled network
        example_filter: Filter used to select which examples to record from
                        (see :ref:`section-callbacks-recording` 
                        for more information).
        neuron_filter:  Filter used to select which neurons to record from
                        (see :ref:`section-callbacks-recording` 
                        for more information).
    """
    def __init__(self, Conn: ConnectionType, 
                 key=None, example_filter: ExampleFilterType = None,
                 neuron_filter: NeuronFilterType = None):
        # Get underlying connection
        self._conn = get_underlying_conn(Conn)
                
        # Stash key
        self.key = key

        # Create example filter
        self._example_filter = ExampleFilter(example_filter)

        # Create neuron filter mask
        self._neuron_mask = get_neuron_filter_mask(neuron_filter,
                                                   self._conn.target().shape)

    def set_params(self, data, compiled_network, **kwargs):
        self._batch_size = compiled_network.genn_model.batch_size
        self._compiled_network = compiled_network

        # Create default batch mask in case on_batch_begin not called
        self._batch_mask = np.ones(self._batch_size, dtype=bool)

        # Create empty list to hold recorded data
        data[self.key] = []
        self._data = data[self.key]

    def on_timestep_end(self, timestep: int):
        # If anything should be recorded this batch
        if self._batch_count > 0:
            # Copy variable from device
            sg = self._compiled_network.connection_populations[self._conn]
            sg.out_post.pull_from_device()

            # If simulation and variable is batched
            isyn_view = sg.out_post.view
            print(isyn_view)
            if self._batch_size > 1 and self.batched:
                # Apply neuron mask
                data_view = isyn_view[self._batch_mask][:, self._neuron_mask]
            # Otherwise
            else:
                # Apply neuron mask
                data_view = isyn_view[:,self._neuron_mask]

                # If SIMULATION is batched but variable isn't,
                # broadcast batch count copies of variable
                if self._batch_size > 1:
                    data_view = np.broadcast_to(
                        data_view, (self._batch_count,) + data_view.shape)

            # If there isn't already list to hold data, add one
            if len(self._data) == 0:
                self._data.append([])

            # Add copy to newest list
            self._data[-1].append(np.copy(data_view))

    def on_batch_begin(self, batch: int):
        # Get mask for examples in this batch and count
        self._batch_mask = self._example_filter.get_batch_mask(
            batch, self._batch_size)
        self._batch_count = np.sum(self._batch_mask)

        # If there's anything to record in this
        # batch, add list to hold it to data
        if self._batch_count > 0:
            self._data.append([])

    def get_data(self):
        # Stack 1D or 2D numpy arrays containing value
        # for each timestep in each batch to get
        # (time, batch, neuron) or (time, neuron) arrays
        data = [np.stack(d) for d in self._data]

        # If model batched
        if self._batch_size > 1:
            # Split each stacked array along the batch axis and
            # chain together resulting in a list, containing a
            # (time, neuron) matrix for each example
            data = list(chain.from_iterable(np.split(d, d.shape[1], axis=1)
                                            for d in data))

            # Finally, remove the batch axis from each matrix
            # (which will now always be size 1) and return
            data = [np.squeeze(d, axis=1) for d in data]

        return self.key, data
