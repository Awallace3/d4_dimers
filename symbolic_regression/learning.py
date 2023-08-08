import numpy as np
from pysr import PySRRegressor

def t1():
    X = 2 * np.random.randn(100, 5)
    y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5
    model = PySRRegressor(
        procs=4,
        model_selection="best",  # Result is mix of simplicity+accuracy
        niterations=40,
        binary_operators=["+", "*", "-"],
        unary_operators=[
            "exp",
            "sin",
            "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        loss="loss(x, y) = (x - y)^2",
        # ^ Custom loss function (julia syntax)
    )
    model.fit(X,y)
    print(model)

def t2():
    X = 2 * np.random.randn(100, 5)
    print(X)
    y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5

def main():
    # t2()
    t1()
    return


if __name__ == "__main__":
    main()
