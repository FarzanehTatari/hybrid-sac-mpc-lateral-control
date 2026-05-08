from experiments.metrics import rmse


def test_rmse_nonnegative():
    assert rmse([0.0, 1.0, -1.0]) >= 0.0
