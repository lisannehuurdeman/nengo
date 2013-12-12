import logging

import numpy as np
import pytest

import nengo
from nengo.tests.helpers import Plotter

logger = logging.getLogger(__name__)


def test_neuron_parititoning():
    nengo.Model()
    ea_even = nengo.networks.EnsembleArray(nengo.LIF(10), 5)
    for ens in ea_even.ensembles:
        assert ens.n_neurons == 2

    ea_odd = nengo.networks.EnsembleArray(nengo.LIF(19), 4)

    # Order of the sizes shouldn't matter
    sizes = [5, 5, 5, 4]
    for ens in ea_odd.ensembles:
        sizes.remove(ens.n_neurons)
    assert len(sizes) == 0


def test_multidim(Simulator, nl):
    """Test an ensemble array with multiple dimensions per ensemble"""
    dims = 3
    n_neurons = 3 * 60
    radius = 1.5

    rng = np.random.RandomState(523887)
    a = rng.uniform(low=-0.7, high=0.7, size=dims)
    b = rng.uniform(low=-0.7, high=0.7, size=dims)

    ta = np.zeros((6, 3))
    ta[[0, 2, 4], [0, 1, 2]] = 1
    tb = np.zeros((6, 3))
    tb[[1, 3, 5], [0, 1, 2]] = 1
    c = np.dot(ta, a) + np.dot(tb, b)

    model = nengo.Model('Multidim', seed=123)
    inputA = nengo.Node(output=a)
    inputB = nengo.Node(output=b)
    A = nengo.networks.EnsembleArray(nl(n_neurons), dims, radius=radius)
    B = nengo.networks.EnsembleArray(nl(n_neurons), dims, radius=radius)
    C = nengo.networks.EnsembleArray(nl(n_neurons * 2), dims,
                                     dimensions_per_ensemble=2,
                                     radius=radius, label="C")
    nengo.Connection(inputA, A.input)
    nengo.Connection(inputB, B.input)
    nengo.Connection(A.output, C.input, transform=ta)
    nengo.Connection(B.output, C.input, transform=tb)

    A_p = nengo.Probe(A.output, 'output', filter=0.03)
    B_p = nengo.Probe(B.output, 'output', filter=0.03)
    C_p = nengo.Probe(C.output, 'output', filter=0.03)

    sim = Simulator(model)
    sim.run(1.0)

    t = sim.trange()
    with Plotter(Simulator, nl) as plt:
        def plot(sim, a, p, title=""):
            a_ref = np.tile(a, (len(t), 1))
            a_sim = sim.data(p)
            colors = ['b', 'g', 'r', 'c', 'm', 'y']
            for i in range(a_sim.shape[1]):
                plt.plot(t, a_ref[:, i], '--', color=colors[i])
                plt.plot(t, a_sim[:, i], '-', color=colors[i])
            plt.title(title)

        plt.subplot(131)
        plot(sim, a, A_p, title="A")
        plt.subplot(132)
        plot(sim, b, B_p, title="B")
        plt.subplot(133)
        plot(sim, c, C_p, title="C")
        plt.savefig('test_ensemble_array.test_multidim.pdf')
        plt.close()

    a_sim = sim.data(A_p)[t > 0.5].mean(axis=0)
    b_sim = sim.data(B_p)[t > 0.5].mean(axis=0)
    c_sim = sim.data(C_p)[t > 0.5].mean(axis=0)

    rtol, atol = 0.1, 0.05
    assert np.allclose(a, a_sim, atol=atol, rtol=rtol)
    assert np.allclose(b, b_sim, atol=atol, rtol=rtol)
    assert np.allclose(c, c_sim, atol=atol, rtol=rtol)


def test_matrix_mul(Simulator, nl):
    N = 100

    Amat = np.asarray([[.5, -.5]])
    Bmat = np.asarray([[0, -1.], [.7, 0]])
    radius = 1

    model = nengo.Model('Matrix Multiplication', seed=123)
    A = nengo.networks.EnsembleArray(
        nl(N * Amat.size), Amat.size, radius=radius, label="A")
    B = nengo.networks.EnsembleArray(
        nl(N * Bmat.size), Bmat.size, radius=radius, label="B")

    inputA = nengo.Node(output=Amat.ravel())
    inputB = nengo.Node(output=Bmat.ravel())
    nengo.Connection(inputA, A.input)
    nengo.Connection(inputB, B.input)
    A_p = nengo.Probe(A.output, 'output', sample_every=0.01, filter=0.01)
    B_p = nengo.Probe(B.output, 'output', sample_every=0.01, filter=0.01)

    C = nengo.networks.EnsembleArray(nl(N * Amat.size * Bmat.shape[1] * 2),
                                     Amat.size * Bmat.shape[1],
                                     dimensions_per_ensemble=2,
                                     radius=1.5 * radius,
                                     label="C")

    if nl != nengo.Direct:
        for ens in C.ensembles:
            ens.encoders = np.tile([[1, 1], [-1, 1], [1, -1], [-1, -1]],
                                   (ens.n_neurons // 4, 1))

    transformA = np.zeros((C.dimensions, Amat.size))
    transformB = np.zeros((C.dimensions, Bmat.size))

    for i in range(Amat.shape[0]):
        for j in range(Amat.shape[1]):
            for k in range(Bmat.shape[1]):
                tmp = (j + k * Amat.shape[1] + i * Bmat.size)
                transformA[tmp * 2][j + i * Amat.shape[1]] = 1
                transformB[tmp * 2 + 1][k + j * Bmat.shape[1]] = 1

    nengo.Connection(A.output, C.input, transform=transformA)
    nengo.Connection(B.output, C.input, transform=transformB)
    C_p = nengo.Probe(C.output, 'output', sample_every=0.01, filter=0.01)

    D = nengo.networks.EnsembleArray(nl(N * Amat.shape[0] * Bmat.shape[1]),
                                     Amat.shape[0] * Bmat.shape[1],
                                     radius=radius, label="D")

    def product(x):
        return x[0]*x[1]

    transformC = np.zeros((D.dimensions, Bmat.size))
    for i in range(Bmat.size):
        transformC[i // Bmat.shape[0]][i] = 1

    prod = C.add_output("product", product)

    nengo.Connection(prod, D.input, transform=transformC)
    D_p = nengo.Probe(D.output, 'output', sample_every=0.01, filter=0.01)

    sim = Simulator(model)
    sim.run(1)

    with Plotter(Simulator, nl) as plt:
        t = sim.trange(dt=0.01)
        plt.plot(t, sim.data(D_p))
        for d in np.dot(Amat, Bmat).flatten():
            plt.axhline(d, color='k')
        plt.savefig('test_ensemble_array.test_matrix_mul.pdf')
        plt.close()

    atol, rtol = .1, .01
    assert np.allclose(sim.data(A_p)[50:, 0], 0.5, atol=atol, rtol=rtol)
    assert np.allclose(sim.data(A_p)[50:, 1], -0.5, atol=atol, rtol=rtol)

    assert np.allclose(sim.data(B_p)[50:, 0], 0, atol=atol, rtol=rtol)
    assert np.allclose(sim.data(B_p)[50:, 1], -1, atol=atol, rtol=rtol)
    assert np.allclose(sim.data(B_p)[50:, 2], .7, atol=atol, rtol=rtol)
    assert np.allclose(sim.data(B_p)[50:, 3], 0, atol=atol, rtol=rtol)

    Dmat = np.dot(Amat, Bmat)
    for i in range(Amat.shape[0]):
        for k in range(Bmat.shape[1]):
            assert np.allclose(
                sim.data(D_p)[-10:, i * Bmat.shape[1] + k],
                Dmat[i, k],
                atol=atol, rtol=rtol)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
