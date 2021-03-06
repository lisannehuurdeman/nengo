import logging

import numpy as np
import pytest

import nengo
import nengo.simulator
from nengo.builder import (Model, ProdUpdate, Copy, Reset, DotInc, Signal,
                           SimPyFunc)

logger = logging.getLogger(__name__)


def pytest_funcarg__RefSimulator(request):
    return nengo.Simulator


def mock_builder(built_model):
    def builder(model, dt):
        return built_model
    return builder


def build_pyfunc(fn, n_in, n_out):
    sig_in = Signal(np.zeros(n_in), name="pyfunc.input")
    sig_out = Signal(np.zeros(n_out), name="pyfunc.output")
    ops = [Reset(sig_in),
           SimPyFunc(output=sig_out, fn=fn, t_in=True, x=sig_in)]
    return sig_in, sig_out, ops


def test_signal_init_values(RefSimulator):
    """Tests that initial values are not overwritten."""
    zero = Signal([0])
    one = Signal([1])
    five = Signal([5.0])
    zeroarray = Signal([[0], [0], [0]])
    array = Signal([1, 2, 3])

    b = Model(dt=0)
    b.operators += [ProdUpdate(zero, zero, one, five),
                    ProdUpdate(zeroarray, one, one, array)]

    sim = RefSimulator(None, builder=mock_builder(b))
    assert sim.signals[zero][0] == 0
    assert sim.signals[one][0] == 1
    assert sim.signals[five][0] == 5.0
    assert np.all(np.array([1, 2, 3]) == sim.signals[array])
    sim.step()
    assert sim.signals[zero][0] == 0
    assert sim.signals[one][0] == 1
    assert sim.signals[five][0] == 5.0
    assert np.all(np.array([1, 2, 3]) == sim.signals[array])


def test_steps(RefSimulator):
    m = nengo.Network(label="test_steps")
    sim = RefSimulator(m)
    assert sim.n_steps == 0
    sim.step()
    assert sim.n_steps == 1
    sim.step()
    assert sim.n_steps == 2


def test_time_steps(RefSimulator):
    m = nengo.Network(label="test_time_steps")
    sim = RefSimulator(m)
    assert np.allclose(sim.signals["__time__"], 0.00)
    sim.step()
    assert np.allclose(sim.signals["__time__"], 0.001)
    sim.step()
    assert np.allclose(sim.signals["__time__"], 0.002)


def test_time_absolute(Simulator):
    m = nengo.Network(label="test_time_absolute", seed=123)
    sim = Simulator(m)
    sim.run(0.003)
    assert np.allclose(sim.trange(), [0.00, .001, .002])


def test_signal_indexing_1(RefSimulator):
    one = Signal(np.zeros(1), name="a")
    two = Signal(np.zeros(2), name="b")
    three = Signal(np.zeros(3), name="c")
    tmp = Signal(np.zeros(3), name="tmp")

    b = Model(dt=0.001)
    b.operators += [
        ProdUpdate(
            Signal(1, name="A1"), three[:1], Signal(0, name="Z0"), one),
        ProdUpdate(
            Signal(2.0, name="A2"), three[1:], Signal(0, name="Z1"), two),
        Reset(tmp),
        DotInc(
            Signal([[0, 0, 1], [0, 1, 0], [1, 0, 0]], name="A3"), three, tmp),
        Copy(src=tmp, dst=three, as_update=True),
    ]

    sim = RefSimulator(None, builder=mock_builder(b))
    sim.signals[three] = np.asarray([1, 2, 3])
    sim.step()
    assert np.all(sim.signals[one] == 1)
    assert np.all(sim.signals[two] == [4, 6])
    assert np.all(sim.signals[three] == [3, 2, 1])
    sim.step()
    assert np.all(sim.signals[one] == 3)
    assert np.all(sim.signals[two] == [4, 2])
    assert np.all(sim.signals[three] == [1, 2, 3])


def test_simple_pyfunc(RefSimulator):
    dt = 0.001
    time = Signal(np.zeros(1), name="time")
    sig = Signal(np.zeros(1), name="sig")
    sig_in, sig_out, ops = build_pyfunc(lambda t, x: np.sin(x), 1, 1)

    b = Model(dt=dt)
    b.operators += ops
    b.operators += [
        ProdUpdate(Signal(dt), Signal(1), Signal(1), time),
        DotInc(Signal([[1.0]]), time, sig_in),
        ProdUpdate(Signal([[1.0]]), sig_out, Signal(0), sig),
    ]

    sim = RefSimulator(None, builder=mock_builder(b))
    sim.step()
    for i in range(5):
        sim.step()
        t = (i + 2) * dt
        assert np.allclose(sim.signals[time], t)
        assert np.allclose(sim.signals[sig], np.sin(t - dt*2))


def test_encoder_decoder_pathway(RefSimulator):
    """Verifies (like by hand) that the simulator does the right
    things in the right order."""
    foo = Signal([1.0], name="foo")
    decoders = np.asarray([.2, .1])
    decs = Signal(decoders * 0.5)
    sig_in, sig_out, ops = build_pyfunc(lambda t, x: x + 1, 2, 2)

    b = Model(dt=0.001)
    b.operators += ops
    b.operators += [
        DotInc(Signal([[1.0], [2.0]]), foo, sig_in),
        ProdUpdate(decs, sig_out, Signal(0.2), foo)
    ]

    def check(sig, target):
        assert np.allclose(sim.signals[sig], target)

    sim = RefSimulator(None, builder=mock_builder(b))

    check(foo, 1.0)
    check(sig_in, 0)
    check(sig_out, 0)

    sim.step()
    # DotInc to pop.input_signal (input=[1.0,2.0])
    # produpdate updates foo (foo=[0.2])
    # pop updates pop.output_signal (output=[2,3])

    check(sig_in, [1, 2])
    check(sig_out, [2, 3])
    check(foo, .2)
    check(decs, [.1, .05])

    sim.step()
    # DotInc to pop.input_signal (input=[0.2,0.4])
    #  (note that pop resets its own input signal each timestep)
    # produpdate updates foo (foo=[0.39]) 0.2*0.5*2+0.1*0.5*3 + 0.2*0.2
    # pop updates pop.output_signal (output=[1.2,1.4])

    check(decs, [.1, .05])
    check(sig_in, [0.2, 0.4])
    check(sig_out, [1.2, 1.4])
    # -- foo is computed as a prodUpdate of the *previous* output signal
    #    foo <- .2 * foo + dot(decoders * .5, output_signal)
    #           .2 * .2  + dot([.2, .1] * .5, [2, 3])
    #           .04      + (.2 + .15)
    #        <- .39
    check(foo, .39)


def test_encoder_decoder_with_views(RefSimulator):
    foo = Signal([1.0], name="foo")
    decoders = np.asarray([.2, .1])
    sig_in, sig_out, ops = build_pyfunc(lambda t, x: x + 1, 2, 2)

    b = Model(dt=0.001)
    b.operators += ops
    b.operators += [
        DotInc(Signal([[1.0], [2.0]]), foo[:], sig_in),
        ProdUpdate(Signal(decoders * 0.5), sig_out, Signal(0.2), foo[:])
    ]

    def check(sig, target):
        assert np.allclose(sim.signals[sig], target)

    sim = RefSimulator(None, builder=mock_builder(b))

    sim.step()
    # DotInc to pop.input_signal (input=[1.0,2.0])
    # produpdate updates foo (foo=[0.2])
    # pop updates pop.output_signal (output=[2,3])

    check(foo, .2)
    check(sig_in, [1, 2])
    check(sig_out, [2, 3])

    sim.step()
    # DotInc to pop.input_signal (input=[0.2,0.4])
    #  (note that pop resets its own input signal each timestep)
    # produpdate updates foo (foo=[0.39]) 0.2*0.5*2+0.1*0.5*3 + 0.2*0.2
    # pop updates pop.output_signal (output=[1.2,1.4])

    check(foo, .39)
    check(sig_in, [0.2, 0.4])
    check(sig_out, [1.2, 1.4])


def test_probedict():
    raw = {"scalar": 5,
           "list": [2, 4, 6]}
    probedict = nengo.simulator.ProbeDict(raw)
    assert np.all(probedict["scalar"] == np.asarray(raw["scalar"]))
    assert np.all(probedict.get("list") == np.asarray(raw.get("list")))


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, "-v"])
