import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import src
import os


def time_2body():
    params = src.paramsTable.get_params("HF")
    params = np.array(params, dtype=np.float64)
    df = pd.read_pickle("./data/schr_dft2.pkl")
    r = df.iloc[154]
    start = time.time()
    src.locald4.compute_bj_dimer_f90(
        params,
        r,
    )
    serial_time = time.time() - start
    print(f"Time for 2-body (python): {time.time() - start:.6f} s")
    disp_times = []
    for i in range(1, 11):
        os.environ["OMP_NUM_THREADS"] = f"{i}"
        start = time.time()
        src.locald4.compute_disp_2B(params, r)
        t = time.time() - start
        print(f"Time for 2-body (OMP={i}): {t:.6f} s")
        print(f"Speedup: {serial_time/t:.2f}")
        disp_times.append(t)
    plt.plot(range(1, 11), disp_times, "o-")
    return


def time_2body_timeit(
    iter=1000,
    proc_test=[1, 2, 4, 6, 8, 10],
    py_func=src.locald4.compute_bj_dimer_f90,
    disp_func=src.locald4.compute_disp_2B,
):
    params = src.paramsTable.get_params("HF")
    params = np.array(params, dtype=np.float64)
    df = pd.read_pickle("./data/schr_dft2.pkl")
    r = df.iloc[154]
    serial_times = []
    for i in range(iter):
        start = time.time()
        py_func(
            params,
            r,
        )
        end = time.time()
        serial_times.append(end - start)
    serial_time_py = np.mean(serial_times)
    print(f"Time for 2-body (python): {serial_time_py:.6f} s")

    serial_times_cpp = []
    os.environ["OMP_NUM_THREADS"] = "1"
    for i in range(iter):
        start = time.time()
        d_2B = disp_func(params, r)
        end = time.time()
        serial_times_cpp.append(end - start)
    serial_time_cpp = np.mean(serial_times_cpp)

    print(f"Time for 2-body (cpp   ): {serial_time_cpp:.6f} s")
    disp_times = [[] for i in range(len(proc_test))]
    disp_time_final = []
    for n, i in enumerate(proc_test):
        os.environ["OMP_NUM_THREADS"] = f"{n}"
        for j in range(iter):
            start = time.time()
            d_2B = disp_func(params, r)
            end = time.time()
            disp_times[n].append(end - start)
        disp_time_final.append(np.mean(disp_times[n]))
        speedup = serial_time_py / disp_time_final[n]
        efficiency = speedup / i
        print(f"Time for 2-body (OMP={i}): {disp_time_final[n]:.6f} s")
        print(f"     Speedup (py ): {speedup:.4f}, efficiency (py ): {efficiency:.4f}")
        speedup_cpp = serial_time_cpp / disp_time_final[n]
        efficiency_cpp = speedup_cpp / i
        print(
            f"     Speedup (cpp ): {speedup_cpp:.4f}, efficiency (cpp ): {efficiency_cpp:.4f}"
        )
        print()
    return


def main():
    time_2body_timeit(disp_func=src.locald4.compute_disp_2B)
    time_2body_timeit(disp_func=src.locald4.compute_disp_2B_dimer)
    return


if __name__ == "__main__":
    main()
