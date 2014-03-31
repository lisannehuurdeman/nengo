import numpy as np

import nengo
from nengo.utils.distributions import Uniform, UniformHypersphere


class Dict(nengo.Network):
    """Store (key, value) associations, and lookup value by key.

    The main assumption is that the set of keys is orthonormal. It is currently
    unknown how well this will perform when that assumption is broken.

    This network implements a data structure that resembles a Python dict.
    (key, value) pairs can be stored, and values later retrieved by key.

    Voja's rule is used to adapt to the keys. And PES is used to associate
    those keys with their corresponding values.

    Parameters
    ----------
    n_memory : int
        Number of neurons to use in the memory layer.
    n_per_item : int
        Specifies how many neurons you would like to fire for a given key.
        This is an approximate target used to optimize the initial tuning
        curves of the memory layer. The effective storage capacity of this
        dict will be near (n_memory / n_per_item). However, higher values of
        n_per_item will mean more accurate recall.
    d_key : int
        Dimensionality of the keys.
    d_value : int
        Dimensionality of the values.
    n_error : int, optional
        Number of neurons to use in the error ensemble. Defaults to 200.
    voja_learning_rate : float, optional
        Learning rate for Voja's rule. Defaults to 1e-3.
    voja_filter : float, optional
        Post-synaptic filter on the memory layer's activity as input to
        Voja's rule. Defaults to 0.001. Determines how quickly the network
        will adapt to new keys.
    pes_learning_rate : float, optional
        Learning rate for the PES rule. Defaults to 1e-4. Determines how
        quickly the netork will associate a value with the given key.
    intercept_spread : float, optional
        The radius of uniform randomness to add to the calculated intercepts
        which are aiming to satisfy n_per_item. Defaults to 0.05.
    n_dopamine : int, optional
        Number of neurons to use in dopamine ensembles. Defaluts to 50.
        Lower values prevent the learning signal from being consistently
        transmitted.
    dopamine_strength : float, optional
        Strength of inhibitory signals in dopamine ensembles. Defaults to 20.
    dopamine_filter : float, optional
        Post-synaptic filter for connections from dopamine ensembles. Defaults
        to 0.001. Lower values make changes to the learning input propagate
        faster to the learning rules.
    dopamine_intercepts : Distribution, optional
        Distribution of intercepts for dopamine ensembles. Defaults to
        Uniform(0.1, 1), in order to ensure that the neurons do not fire
        when x = 0.
    always_learn : bool, optional
        Specifies whether the memory should always be learning from the given
        keys and values. Defaults to False. Connect to the learning Node to
        make this dynamic (refer to documentation on Network attributes).
    voja_disable : bool, optional
        Set to True to disable Voja's rule. This will not break the dict, but
        it will impact recall accuracy, since some keys might share encoders.
        This is intended for debugging/validation purposes.

    Attributes
    ----------
    key : Node
        Input node for the current key. Can be 0.
    value : Node
        Input node for the current value. Can be anything if key is 0 or
        learning is off. Otherwise, it is the value that will become associated
        with the current key.
    output : Node
        Output node for the value which corresponds to the current key in
        memory. If the key doesn't exist, then the value will be arbitrary.
    learning : Node
        If always_learn is False (the default), then this is an input node
        which scales the learning_rate (for voja) and inhibits the error signal
        (for PES). Set this signal to 0 when you want learning to be turned
        off (so that lookups can be done regardless of the current value),
        and to 1 when you want learning to be on (to store associations).
    memory : Ensemble
        Layer which stores all of the associations. Its encoders remember the
        given keys, and its decoders produce the corresponding values.
    bias : Ensemble
        Bias node that always outputs 1, as input to the nopamine.
    dopamine : Ensemble
        Ensemble holding the learning node's value. Used by voja's rule to
        scale the learning_rate.
    nopamine : Ensemble
        "No dopamine"; 0 when dopamine is sufficiently high, otherwise 1.
    error : Ensemble
        Represents the current error between the given value and the stored
        value. If nopamine is 1 (learning is off), then the error is inhibited
        to 0, so that PES will not adjust any values.
    voja : Voja
        Instance of the Voja learning rule.
    pes : PES
        Instance of the PES learning rule.
    """

    def make(self, n_memory, n_per_item, d_key, d_value, n_error=200,
             voja_learning_rate=1e-3, voja_filter=0.001,
             pes_learning_rate=1e-4, intercept_spread=0.05,
             n_dopamine=50, dopamine_strength=20, dopamine_filter=0.001,
             dopamine_intercepts=Uniform(0.1, 1), always_learn=False,
             voja_disable=False):
        # Create input and output relays
        self.key = nengo.Node(size_in=d_key, label="key")
        self.value = nengo.Node(size_in=d_value, label="value")
        self.output = nengo.Node(size_in=d_value, label="output")
        self.learning = nengo.Node(
            size_in=1, output=lambda t, x: (x, 1)[always_learn],
            label="learning")

        # Create node/ensembles for scaling the learning rate
        self.bias = nengo.Node(output=[1], label="bias")
        self.dopamine = nengo.Ensemble(
            nengo.LIF(n_dopamine), 1, intercepts=dopamine_intercepts,
            encoders=[[1]]*n_dopamine, label="dopamine")
        self.nopamine = nengo.Ensemble(
            nengo.LIF(n_dopamine), 1, intercepts=dopamine_intercepts,
            encoders=[[1]]*n_dopamine, label="nopamine")

        # Create ensemble which acts as the dictionary. The encoders will
        # shift towards the keys with Voja's rule, and the decoders will
        # shift towards the values with the PES learning rule.
        memory_intercept = self._calculate_intercept(
            d_key, n_per_item / float(n_memory))
        memory_intercepts = Uniform(
            memory_intercept - intercept_spread,
            min(memory_intercept + intercept_spread, 1.0))
        self.memory = nengo.Ensemble(
            nengo.LIF(n_memory), d_key, intercepts=memory_intercepts,
            label="memory")

        # Create the ensemble for calculating error * learning
        self.error = nengo.Ensemble(nengo.LIF(n_error), d_value, label="error")

        # Connect the learning signal to the error population
        nengo.Connection(self.learning, self.dopamine)
        nengo.Connection(self.bias, self.nopamine)
        self._inhibit(self.dopamine, self.nopamine, amount=dopamine_strength,
                      filter=dopamine_filter)
        self._inhibit(self.nopamine, self.error, amount=dopamine_strength,
                      filter=dopamine_filter)

        # Connect the key Node to the memory Ensemble with voja's rule
        self.voja = (nengo.Voja(learning_rate=voja_learning_rate,
                                learning=self.dopamine, filter=voja_filter,
                                label="learn_key")
                     if not voja_disable else None)
        nengo.Connection(self.key, self.memory, learning_rule=self.voja)

        # Compute the error
        nengo.Connection(self.value, self.error)
        nengo.Connection(self.output, self.error, transform=-1)

        # Connect the memory Ensemble to the output Node with PES(error)
        # _connect_spheres is just a rough way to initialize the decoders
        # so that they are doing something reasonable when d_key != d_value
        self.pes = nengo.PES(
            self.error, learning_rate=pes_learning_rate, label="learn_value")
        self._connect_spheres(
            self.memory, self.output, d_key, d_value, learning_rule=self.pes)

    @classmethod
    def _calculate_intercept(cls, d, p, dx=0.001):
        """Returns c such that np.dot(u, v) >= c with probability p.

        Here, u and v are two randomly generated vectors of dimension d.
        This works by the following formula, (1 - x**2)**((d - 3)/2.0), which
        gives the probability that a coordinate of a random point on a
        hypersphere is equal to x.

        The probability distribution of the dot product of two randomly chosen
        vectors is equivalent to the above, since we can always rotate the
        sphere such that one of the vectors is a unit vector, and then the
        dot product just becomes the component corresponding to that unit
        vector.

        This can be used to find the intercept such that a randomly generated
        encoder will fire in response to a random input x with probability p.
        """
        x = np.arange(-1+dx, 1, dx)
        y = (1 - x**2)**((d - 3)/2.0)
        py = np.cumsum(y)
        py = py / sum(py) / dx
        return x[py >= 1 - p][0]

    @classmethod
    def _inhibit(cls, pre, post, amount=10, **kwargs):
        """Creates a connection which inhibits post whenever pre fires."""
        return nengo.Connection(
            pre.neurons, post.neurons,
            transform=-amount*np.ones((post.n_neurons, pre.n_neurons)),
            **kwargs)

    @classmethod
    def _connect_spheres(cls, pre, post, d_pre, d_post, **kwargs):
        """Creates a connection between the surface of two spheres.

        Works by using evaluation points sampled from a sphere of dimension
        d_pre, and then mapping d_post of them to the canonical basis of post.

        This should be used with caution for two reasons:
            (1) If two eval points are close together, then the required
                function is highly discontinuous.
            (2) For points in between the two eval points, the trajectory
                may not be anything sensible, and is not guaranteed to stay
                on the surface of the post sphere.
        """
        if d_pre == d_post:
            return nengo.Connection(pre, post, **kwargs)
        eval_points = UniformHypersphere(d_pre, surface=True).sample(d_post)
        eval_points_lookup = dict(
            ((tuple(v), i) for i, v in enumerate(eval_points)))
        function = lambda v: np.eye(d_post)[eval_points_lookup[tuple(v)], :]
        return nengo.Connection(
            pre, post, eval_points=eval_points, function=function, **kwargs)
